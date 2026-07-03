"""Tải audio gốc từ YouTube bằng yt-dlp.

Cần Deno (hoặc Node.js) trong PATH để yt-dlp giải signature/n-parameter challenge của
YouTube — thiếu JS runtime là nguyên nhân THẬT gây lỗi 403 khi tải trên server/cloud
(RunPod, Vast.ai, VPS...), đã xác nhận qua test thực tế, không phải do IP bị chặn hẳn.
Dockerfile đã tự cài Deno; nếu chạy ngoài Docker, cài thủ công theo README.
"""

import logging
import os
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)

# Thứ tự thử các player_client nếu client mặc định thất bại — YouTube thỉnh thoảng đổi
# cơ chế chặn/yêu cầu PO Token riêng cho từng client, thử client khác nhau tăng tỉ lệ qua.
_FALLBACK_PLAYER_CLIENTS = [None, "android_vr", "tv", "web"]


def _build_ydl_opts(tmp_dir: Path, player_client: str | None) -> dict:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if player_client:
        ydl_opts["extractor_args"] = {"youtube": {"player_client": [player_client]}}

    # Nếu vẫn bị chặn dai dẳng dù đã có JS runtime: xuất cookies.txt từ trình duyệt đã
    # đăng nhập YouTube, đặt biến môi trường YTDLP_COOKIES_FILE trỏ tới file đó.
    cookies_file = os.environ.get("YTDLP_COOKIES_FILE")
    if cookies_file and Path(cookies_file).exists():
        ydl_opts["cookiefile"] = cookies_file

    return ydl_opts


def download_audio(url: str, tmp_dir: Path) -> dict:
    """Tải audio chất lượng cao nhất, chuyển sang WAV 16kHz mono (chuẩn input Whisper).

    Tự thử lại với player_client khác nếu client mặc định bị chặn, trước khi báo lỗi.

    Trả về {"video_id", "title", "audio_path", "source_url"}.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    info = None
    last_error = None
    for player_client in _FALLBACK_PLAYER_CLIENTS:
        ydl_opts = _build_ydl_opts(tmp_dir, player_client)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            logger.warning(
                "Tải thất bại với player_client=%s, thử client khác... (%s)", player_client, e
            )

    if info is None:
        raise last_error

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
