"""Tải audio gốc từ YouTube bằng yt-dlp."""

import logging
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


def download_audio(url: str, tmp_dir: Path) -> dict:
    """Tải audio chất lượng cao nhất, chuyển sang WAV 16kHz mono (chuẩn input Whisper).

    Trả về {"video_id", "title", "audio_path", "source_url"}.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(tmp_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info["id"]
    title = info.get("title", video_id)
    audio_path = tmp_dir / f"{video_id}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy audio đã tải: {audio_path}")

    logger.info("Đã tải audio: %s (%s)", title, audio_path)
    return {
        "video_id": video_id,
        "title": title,
        "audio_path": audio_path,
        "source_url": url,
    }
