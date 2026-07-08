#!/usr/bin/env python3
"""CLI: trích xuất MP3 + sub theo thuật ngữ chuyên ngành từ video YouTube.

Cách dùng:
    python main.py <youtube_url>
    python main.py --batch urls.txt
"""

import argparse
import logging
import os
from pathlib import Path

import yaml

import sys

from src import clip_exporter, downloader, llm_validator, metadata_writer, segment_builder, term_matcher, transcriber
from src.gpu_config import InsufficientVRAMError, downgrade_settings, resolve_whisper_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_terms(terms_path: Path) -> list[str]:
    data = yaml.safe_load(terms_path.read_text(encoding="utf-8"))
    return data["terms"]


class ModelState:
    """Giữ model + settings hiện tại, cho phép downgrade và reload khi gặp CUDA OOM."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.model = transcriber.load_model(settings)


def _is_oom_error(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def transcribe_with_fallback(state: ModelState, audio_path: Path, terms: list[str], cfg: dict) -> list[dict]:
    """Transcribe, tự downgrade model_size/compute_type/beam_size và thử lại nếu CUDA OOM.

    Khi downgrade xảy ra, state được cập nhật tại chỗ nên các video tiếp theo trong
    batch sẽ dùng thẳng cấu hình an toàn hơn thay vì OOM lại từ đầu.
    """
    while True:
        try:
            return transcriber.transcribe(
                state.model,
                audio_path,
                terms,
                language=cfg["whisper"]["language"],
                beam_size=state.settings["beam_size"],
            )
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            downgraded = downgrade_settings(state.settings)
            if downgraded is None:
                logger.error(
                    "CUDA OOM ở bậc thấp nhất (%s), không thể downgrade thêm. "
                    "Reload lại model để tránh CUDA context hỏng lây sang video tiếp theo.",
                    state.settings,
                )
                del state.model
                state.model = transcriber.load_model(state.settings)
                raise
            logger.warning("CUDA OOM với %s -> downgrade sang %s và thử lại", state.settings, downgraded)
            del state.model
            state.settings = downgraded
            state.model = transcriber.load_model(downgraded)


def process_video(
    url: str,
    state: ModelState,
    terms: list[str],
    cfg: dict,
    output_root: Path,
    tmp_dir: Path,
    openrouter_api_key: str | None = None,
) -> int:
    """Xử lý 1 video end-to-end. Trả về số clip đã xuất."""
    logger.info("=== Xử lý video: %s ===", url)

    video_meta = downloader.download_audio(url, tmp_dir)

    segments = transcribe_with_fallback(state, video_meta["audio_path"], terms, cfg)

    if openrouter_api_key:
        logger.info("Phát hiện thuật ngữ qua Claude Haiku (không giới hạn theo terms.yaml)...")
        matches = llm_validator.detect_terms_in_segments(segments, openrouter_api_key)
    else:
        words = term_matcher.flatten_words(segments)
        matches = term_matcher.find_term_matches(words, terms, cfg["matching"]["fuzzy_threshold"])
    logger.info("Tìm thấy %d lần khớp thuật ngữ", len(matches))

    if not matches:
        logger.warning("Không tìm thấy thuật ngữ nào trong video này, bỏ qua.")
        video_meta["audio_path"].unlink(missing_ok=True)
        return 0

    clips = segment_builder.build_clips(segments, matches, cfg["clip"])
    logger.info("Gom thành %d clip (<= %ss mỗi clip)", len(clips), cfg["clip"]["max_duration_sec"])

    video_dir = output_root / video_meta["video_id"]
    clips_dir = video_dir / "clips"

    exported = []
    for idx, clip in enumerate(clips, 1):
        files = clip_exporter.export_clip(video_meta["audio_path"], clip, idx, clips_dir)
        exported.append({**clip, **files})

    rows = metadata_writer.build_rows(video_meta, exported, output_root)

    if openrouter_api_key:
        logger.info("Đang validate thêm bằng Claude Haiku (OpenRouter)...")
        confidence_threshold = cfg["llm_validation"]["confidence_threshold"]
        for row in rows:
            result = llm_validator.validate_clip(row["transcript_vi"], terms, openrouter_api_key)
            row["llm_extra_terms"] = ";".join(result["extra_terms"])
            row["transcript_vi_llm_suggested"] = result["suggested_text"]
            confidence = result["confidence_percent"]
            row["llm_confidence_rate"] = confidence if confidence is not None else ""
            row["llm_valid"] = (
                ("TRUE" if confidence >= confidence_threshold else "FALSE")
                if confidence is not None
                else ""
            )

    metadata_writer.write_video_metadata(rows, video_dir)
    metadata_writer.append_aggregate_csv(rows, output_root)

    video_meta["audio_path"].unlink(missing_ok=True)
    logger.info("Hoàn thành video %s: %d clip -> %s", video_meta["video_id"], len(clips), video_dir)
    return len(clips)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="Link YouTube (bỏ qua nếu dùng --batch)")
    parser.add_argument("--batch", metavar="FILE", help="File danh sách URL, mỗi dòng 1 link")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn config.yaml")
    parser.add_argument("--output-dir", help="Ghi đè thư mục output (mặc định lấy từ config.yaml)")
    parser.add_argument(
        "--openrouter-key",
        help="OpenRouter API key để bật bước validate thêm bằng Claude Haiku (bỏ qua nếu không cần)",
    )
    args = parser.parse_args()

    if not args.url and not args.batch:
        parser.error("Cần truyền <url> hoặc --batch <file>")

    cfg = load_config(Path(args.config))
    terms = load_terms(Path(cfg["paths"]["terms_file"]))
    output_root = Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["output_dir"])
    tmp_dir = Path(cfg["paths"]["tmp_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    openrouter_api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")

    try:
        whisper_settings = resolve_whisper_settings(cfg["whisper"], cfg["vram_thresholds_mb"])
    except InsufficientVRAMError as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info("Whisper settings: %s", whisper_settings)
    state = ModelState(whisper_settings)

    urls = [args.url] if args.url else [
        line.strip()
        for line in Path(args.batch).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    total_clips = 0
    for url in urls:
        try:
            total_clips += process_video(url, state, terms, cfg, output_root, tmp_dir, openrouter_api_key)
        except Exception:
            logger.exception("Lỗi khi xử lý video: %s", url)

    logger.info("=== XONG. Tổng %d video, %d clip. Output: %s ===", len(urls), total_clips, output_root)


if __name__ == "__main__":
    main()
