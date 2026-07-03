"""Transcribe audio bằng faster-whisper, giữ word-level timestamp để match thuật ngữ chính xác."""

import logging
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def build_initial_prompt(terms: list[str]) -> str:
    """Ghép danh sách thuật ngữ thành 1 câu prompt để bias whisper nhận đúng chính tả."""
    return "Thuật ngữ chuyên ngành composite xuất hiện trong bài: " + ", ".join(terms) + "."


def load_model(whisper_settings: dict) -> WhisperModel:
    """Load model 1 lần, dùng lại cho toàn bộ batch (tránh load lại large-v3 mỗi video)."""
    return WhisperModel(
        whisper_settings["model_size"],
        device=whisper_settings["device"],
        compute_type=whisper_settings["compute_type"],
    )


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    terms: list[str],
    language: str = "vi",
    beam_size: int = 5,
) -> list[dict]:
    """Trả về list segment: {text, start, end, words: [{word, start, end, probability}]}."""
    initial_prompt = build_initial_prompt(terms)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        initial_prompt=initial_prompt,
    )

    logger.info(
        "Transcribe xong: language=%s, duration=%.1fs", info.language, info.duration
    )

    segments = []
    for seg in segments_iter:
        words = [
            {
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            }
            for w in (seg.words or [])
        ]
        segments.append(
            {
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
                "words": words,
            }
        )
    return segments
