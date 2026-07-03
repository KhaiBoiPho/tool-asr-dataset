"""Tự dò VRAM free và chọn compute_type cho faster-whisper large-v3.

Chỉ hỗ trợ 2 tier GPU: ~8GB (int8_float16) và >=12GB (float16, tốt nhất). Tier <8GB
(vd 4GB laptop) đã bị loại bỏ — thực nghiệm cho thấy dễ OOM và gây lỗi timestamp
trôi (transcript không khớp audio) do phải rơi về model nhỏ + beam_size=1.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


class InsufficientVRAMError(RuntimeError):
    """GPU không đủ VRAM cho tier thấp nhất được hỗ trợ (large-v3/int8_float16, ~8GB)."""


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


# beam_size lớn (mặc định 5) làm tăng VRAM lúc generate vì mỗi bước beam nhân bản
# trạng thái decode. Giới hạn theo compute_type để tránh OOM.
_BEAM_SIZE_CAP_BY_COMPUTE_TYPE = {
    "float16": None,       # tier >=12GB, đủ VRAM rộng rãi -> giữ nguyên beam_size cấu hình
    "int8_float16": 3,     # tier ~8GB -> giới hạn nhẹ để an toàn (chưa có số đo thực tế
                            # trên card 8GB thật, đây là mức ước lượng thận trọng)
}


def cap_beam_size(compute_type: str, configured_beam_size: int) -> int:
    cap = _BEAM_SIZE_CAP_BY_COMPUTE_TYPE.get(compute_type)
    if cap is None:
        return configured_beam_size
    return min(configured_beam_size, cap)


def resolve_whisper_settings(whisper_cfg: dict, vram_thresholds: dict) -> dict:
    """Trả về dict {model_size, compute_type, device, beam_size} đã resolve, dựa trên config.

    Nếu whisper_cfg['model_size'] hoặc ['compute_type'] != 'auto', dùng giá trị đó
    trực tiếp (override thủ công). Ngược lại tự dò theo VRAM free, luôn dùng
    model_size='large-v3' (không còn fallback model nhỏ hơn). Nếu VRAM free thấp hơn
    ngưỡng tối thiểu (~8GB, tier int8_float16), raise InsufficientVRAMError thay vì
    âm thầm hạ xuống model kém chính xác/kém ổn định hơn.
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
        raise InsufficientVRAMError(
            "device=cpu không được hỗ trợ — hệ thống yêu cầu GPU NVIDIA >= 8GB VRAM "
            "(large-v3 + int8_float16 trở lên)."
        )

    free_mb = get_free_vram_mb()
    if free_mb is None:
        raise InsufficientVRAMError(
            "Không đọc được VRAM free (nvidia-smi lỗi) — kiểm tra driver NVIDIA đã cài đúng chưa."
        )

    float16_thr = vram_thresholds.get("float16", 10000)
    int8_float16_thr = vram_thresholds.get("int8_float16", 6500)

    if free_mb >= float16_thr:
        resolved = {"device": device, "model_size": "large-v3", "compute_type": "float16"}
    elif free_mb >= int8_float16_thr:
        resolved = {"device": device, "model_size": "large-v3", "compute_type": "int8_float16"}
    else:
        raise InsufficientVRAMError(
            f"VRAM free chỉ {free_mb}MB (< {int8_float16_thr}MB) — GPU không đủ cho tier "
            "tối thiểu được hỗ trợ (large-v3/int8_float16, cần GPU thực tế ~8GB trở lên). "
            "Tier GPU <8GB đã bị loại bỏ do dễ OOM và gây lỗi timestamp không khớp audio."
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

    Thứ tự xuống bậc: large-v3/float16 -> large-v3/int8_float16 -> None (đã ở bậc
    thấp nhất được hỗ trợ, không còn fallback model nhỏ hơn — báo lỗi ra ngoài).
    """
    model_size, compute_type = current["model_size"], current["compute_type"]

    if model_size == "large-v3" and compute_type == "float16":
        next_settings = {**current, "model_size": "large-v3", "compute_type": "int8_float16"}
    else:
        return None

    next_settings["beam_size"] = cap_beam_size(next_settings["compute_type"], current["beam_size"])
    return next_settings
