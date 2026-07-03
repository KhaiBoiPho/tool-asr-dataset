"""Tự dò VRAM free và chọn model_size/compute_type cho faster-whisper."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_free_vram_mb() -> int | None:
    """Trả về VRAM free (MB) của GPU đầu tiên, hoặc None nếu không có GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


# Bậc thấp hơn (int8_float16, và fallback medium/int8) dùng beam_size search nên
# beam_size lớn (mặc định 5) có thể đẩy VRAM vượt ngưỡng dù "free" ban đầu đủ,
# vì mỗi bước beam nhân bản trạng thái decode. Giới hạn beam_size theo tier để
# tránh out-of-memory thay vì chỉ nhìn dung lượng model.
_BEAM_SIZE_CAP_BY_COMPUTE_TYPE = {
    "float16": None,          # đủ VRAM rộng rãi, giữ nguyên beam_size cấu hình
    "int8_float16": 2,
    "int8": 1,
}


def cap_beam_size(compute_type: str, configured_beam_size: int) -> int:
    cap = _BEAM_SIZE_CAP_BY_COMPUTE_TYPE.get(compute_type)
    if cap is None:
        return configured_beam_size
    return min(configured_beam_size, cap)


def resolve_whisper_settings(whisper_cfg: dict, vram_thresholds: dict) -> dict:
    """Trả về dict {model_size, compute_type, device, beam_size} đã resolve, dựa trên config.

    Nếu whisper_cfg['model_size'] hoặc ['compute_type'] != 'auto', dùng giá trị đó
    trực tiếp (override thủ công). Ngược lại tự dò theo VRAM free. beam_size luôn
    được giới hạn lại theo compute_type để tránh OOM khi VRAM eo hẹp.
    """
    device = whisper_cfg.get("device", "cuda")
    model_size = whisper_cfg.get("model_size", "auto")
    compute_type = whisper_cfg.get("compute_type", "auto")
    configured_beam_size = whisper_cfg.get("beam_size", 5)

    if model_size != "auto" and compute_type != "auto":
        resolved = {"device": device, "model_size": model_size, "compute_type": compute_type}
        resolved["beam_size"] = cap_beam_size(compute_type, configured_beam_size)
        return resolved

    if device != "cuda":
        # CPU: an toàn nhất là model nhỏ hơn + int8
        resolved = {"device": device, "model_size": "medium", "compute_type": "int8"}
        logger.warning("device=cpu -> dùng %s", resolved)
        resolved["beam_size"] = cap_beam_size(resolved["compute_type"], configured_beam_size)
        return resolved

    free_mb = get_free_vram_mb()
    if free_mb is None:
        logger.warning("Không đọc được VRAM free (nvidia-smi lỗi) -> fallback cpu/medium/int8")
        resolved = {"device": "cpu", "model_size": "medium", "compute_type": "int8"}
        resolved["beam_size"] = cap_beam_size(resolved["compute_type"], configured_beam_size)
        return resolved

    float16_thr = vram_thresholds.get("float16", 8000)
    int8_float16_thr = vram_thresholds.get("int8_float16", 4500)

    if free_mb >= float16_thr:
        resolved = {"device": device, "model_size": "large-v3", "compute_type": "float16"}
    elif free_mb >= int8_float16_thr:
        resolved = {"device": device, "model_size": "large-v3", "compute_type": "int8_float16"}
    else:
        resolved = {"device": device, "model_size": "medium", "compute_type": "int8"}
        logger.warning(
            "VRAM free chỉ %dMB (< %dMB) -> fallback model_size=medium, compute_type=int8. "
            "Kết quả nhận dạng có thể kém chính xác hơn large-v3.",
            free_mb, int8_float16_thr,
        )

    # Override từng phần nếu user đã ép cứng 1 trong 2 giá trị
    if model_size != "auto":
        resolved["model_size"] = model_size
    if compute_type != "auto":
        resolved["compute_type"] = compute_type

    resolved["beam_size"] = cap_beam_size(resolved["compute_type"], configured_beam_size)
    logger.info("VRAM free=%sMB -> whisper settings: %s", free_mb, resolved)
    return resolved


def downgrade_settings(current: dict) -> dict | None:
    """Trả về 1 bậc cấu hình an toàn hơn (VRAM thấp hơn) để retry sau khi OOM.

    Thứ tự xuống bậc: large-v3/float16 -> large-v3/int8_float16 -> medium/int8 -> None
    (đã ở bậc thấp nhất, không thể downgrade thêm).
    """
    model_size, compute_type = current["model_size"], current["compute_type"]

    if model_size == "large-v3" and compute_type == "float16":
        next_settings = {**current, "model_size": "large-v3", "compute_type": "int8_float16"}
    elif model_size == "large-v3" and compute_type == "int8_float16":
        next_settings = {**current, "model_size": "medium", "compute_type": "int8"}
    else:
        return None

    next_settings["beam_size"] = cap_beam_size(next_settings["compute_type"], current["beam_size"])
    return next_settings
