# Chạy trên máy có GPU NVIDIA mạnh hơn (vd RTX 3070 8GB): cần cài driver NVIDIA +
# nvidia-container-toolkit trên máy host, không cần cài CUDA toolkit đầy đủ — cuBLAS/cuDNN
# runtime cho faster-whisper được cài qua pip bên dưới.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt nvidia-cublas-cu12 nvidia-cudnn-cu12

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
