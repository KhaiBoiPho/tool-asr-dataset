"""UI cơ bản (Streamlit) để chạy pipeline: dán link video, đường dẫn output, API key.

Chạy: streamlit run app.py
"""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="FRP Term Extractor", page_icon="🎬")
st.title("Trích xuất MP3 + Sub theo thuật ngữ FRP/Composite")

output_dir = st.text_input(
    "Thư mục output (đường dẫn trên máy)",
    value=str(Path("output").resolve()),
    help="Nơi lưu clip MP3 + SRT + metadata. Sẽ tự tạo nếu chưa tồn tại.",
)

openrouter_api_key = st.text_input(
    "OpenRouter API key (tuỳ chọn — để trống nếu không cần validate bằng Claude Haiku)",
    type="password",
)

video_url = st.text_input("Link video YouTube")

run_clicked = st.button("Chạy", type="primary")

if run_clicked:
    if not video_url.strip():
        st.error("Cần nhập link video.")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "main.py", video_url.strip(), "--output-dir", output_dir]

        env = os.environ.copy()
        if openrouter_api_key.strip():
            env["OPENROUTER_API_KEY"] = openrouter_api_key.strip()

        with st.spinner("Đang xử lý (tải video, transcribe, cắt clip)... có thể mất vài phút."):
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        st.subheader("Log")
        st.code((result.stdout or "") + (result.stderr or ""), language="text")

        if result.returncode == 0:
            st.success("Hoàn thành.")
            clips_root = Path(output_dir)
            clip_files = sorted(clips_root.glob("*/clips/*.mp3"))
            if clip_files:
                st.subheader(f"{len(clip_files)} clip đã tạo")
                for clip_path in clip_files:
                    st.write(str(clip_path))
                    st.audio(str(clip_path))
            else:
                st.info("Không tìm thấy clip nào — video có thể không chứa thuật ngữ nào trong terms.yaml.")
        else:
            st.error(f"Lệnh thoát với mã lỗi {result.returncode}, xem log ở trên.")
