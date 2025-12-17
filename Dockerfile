# Dùng base image Python 3.12-slim 
FROM python:3.12-slim

# Để log từ Python in ra console ngay lập tức 
ENV PYTHONUNBUFFERED=1

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:${LD_LIBRARY_PATH}

# Cài đặt các thư viện hệ thống bắt buộc cho OpenCV và PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch với hỗ trợ CUDA 12.1 (nếu GPU có sẵn)
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy và cài đặt requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir nvidia-cudnn-cu12==9.1.0.70

# Đặt thư mục làm việc thành /app
WORKDIR /app

# Copy TOÀN BỘ dự án (từ thư mục hiện tại trên host) vào /app trong container
# copy code
COPY pythonFile /app/pythonFile

# copy weights
COPY weights /app/weights

# Thông báo cổng FastAPI sẽ chạy 
EXPOSE 8000

# Dọn dẹp cache của PyTorch để tránh lỗi file hỏng
RUN rm -rf /root/.cache/torch

# Tạo thư mục snapshots với quyền truy cập đầy đủ
RUN mkdir -p snapshots && chmod 777 snapshots