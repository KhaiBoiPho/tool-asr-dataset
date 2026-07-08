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
    "OpenRouter API key (khuyến nghị nhập — dùng Claude Haiku để tự phát hiện MỌI thuật ngữ "
    "kỹ thuật/vật liệu tiếng Anh trong video, không giới hạn theo terms.yaml. Để trống thì "
    "chỉ khớp theo danh sách cố định trong terms.yaml.)",
    type="password",
)

video_urls = st.text_area(
    "Link video YouTube (mỗi dòng 1 link, có thể dán nhiều video để xử lý tuần tự)",
    height=150,
)

run_clicked = st.button("Chạy", type="primary")

if run_clicked:
    urls = [line.strip() for line in video_urls.splitlines() if line.strip()]
    if not urls:
        st.error("Cần nhập ít nhất 1 link video.")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if openrouter_api_key.strip():
            env["OPENROUTER_API_KEY"] = openrouter_api_key.strip()

        if len(urls) == 1:
            cmd = [sys.executable, "main.py", urls[0], "--output-dir", output_dir]
            spinner_msg = "Đang xử lý (tải video, transcribe, cắt clip)... có thể mất vài phút."
        else:
            batch_file = Path(output_dir) / "_batch_urls.txt"
            batch_file.write_text("\n".join(urls), encoding="utf-8")
            cmd = [sys.executable, "main.py", "--batch", str(batch_file), "--output-dir", output_dir]
            spinner_msg = f"Đang xử lý tuần tự {len(urls)} video (dùng chung model đã load)... có thể mất nhiều thời gian."

        with st.spinner(spinner_msg):
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
