"""Cắt clip MP3 bằng ffmpeg (subprocess trực tiếp) và ghi file SRT tương ứng."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(subtitle_entries: list[dict], srt_path: Path) -> None:
    lines = []
    for i, entry in enumerate(subtitle_entries, 1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(entry['start'])} --> {_format_srt_time(entry['end'])}")
        lines.append(entry["text"])
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def export_clip(source_audio_path: Path, clip: dict, idx: int, clips_dir: Path) -> dict:
    """Cắt 1 clip mp3 + ghi srt. Trả về {clip_file, srt_file} (đường dẫn tuyệt đối)."""
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_name = f"clip_{idx:03d}"
    mp3_path = clips_dir / f"{clip_name}.mp3"
    srt_path = clips_dir / f"{clip_name}.srt"

    duration = clip["end"] - clip["start"]
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip["start"]),
        "-t", str(duration),
        "-i", str(source_audio_path),
        "-vn", "-ar", "16000", "-ac", "1",
        "-codec:a", "libmp3lame", "-qscale:a", "4",
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg lỗi khi cắt {clip_name}: {result.stderr[-2000:]}")

    _write_srt(clip["subtitle_entries"], srt_path)

    logger.info("Đã xuất %s (%.1fs -> %.1fs, terms=%s)", clip_name, clip["start"], clip["end"], clip["terms"])
    return {"clip_file": mp3_path, "srt_file": srt_path}
