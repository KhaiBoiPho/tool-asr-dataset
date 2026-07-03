# Chạy trên máy có GPU NVIDIA mạnh hơn (vd RTX 3070 8GB, hoặc RunPod serverless GPU): cần
# driver NVIDIA + nvidia-container-toolkit trên máy host, không cần cài CUDA toolkit đầy
# đủ — cuBLAS/cuDNN runtime cho faster-whisper được cài qua pip bên dưới.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Deno: yt-dlp cần 1 JS runtime để giải signature/n-parameter challenge của YouTube —
# thiếu cái này là nguyên nhân thật gây lỗi "403 Forbidden" khi tải video trên server/cloud,
# không phải do IP bị chặn hẳn. Xác nhận qua test thực tế trên GPU server thuê ngoài.
RUN curl -fsSL https://deno.land/install.sh | sh -s -- --yes
ENV PATH="/root/.deno/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt nvidia-cublas-cu12 nvidia-cudnn-cu12

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
