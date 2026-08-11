# Dùng bản Python chạy nhẹ trên nền Linux ARM/x86
FROM python:3.10-slim

# Cài thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Copy requirement và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY . .

# Lệnh chạy ứng dụng
CMD ["python", "test_runner.py"]