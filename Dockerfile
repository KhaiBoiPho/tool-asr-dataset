# Chạy trên máy có GPU NVIDIA mạnh hơn (vd RTX 3070 8GB, hoặc RunPod serverless GPU): cần
# driver NVIDIA + nvidia-container-toolkit trên máy host, không cần cài CUDA toolkit đầy
# đủ — cuBLAS/cuDNN runtime cho faster-whisper được cài qua pip bên dưới.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Deno: yt-dlp cần 1 JS runtime để giải signature/n-parameter challenge của YouTube —
# thiếu cái này là nguyên nhân thật gây lỗi "403 Forbidden" khi tải video trên server/cloud,
# không phải do IP bị chặn hẳn. Xác nhận qua test thực tế trên GPU server thuê ngoài.
# Cài vào /usr/local (dùng chung mọi user) thay vì /root vì container không chạy root.
ENV DENO_INSTALL=/usr/local
RUN curl -fsSL https://deno.land/install.sh | sh -s -- --yes
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt nvidia-cublas-cu12 nvidia-cudnn-cu12

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

COPY . .

# Chạy bằng user thường (không phải root) để file ghi ra volume mount trên host
# (vd output/) có đúng quyền sở hữu của user host thay vì bị khoá quyền root.
# Mặc định UID/GID=1000 khớp user Linux đầu tiên phổ biến — build lại với
# --build-arg UID=$(id -u) --build-arg GID=$(id -g) nếu máy bạn dùng UID/GID khác.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" appuser \
    && useradd -m -u "${UID}" -g "${GID}" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
