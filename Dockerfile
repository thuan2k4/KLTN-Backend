FROM python:3.10-slim

WORKDIR /app

# Copy requirements và cài đặt các thư viện Python
COPY requirements.txt .

# Nâng cấp pip và cài đặt packages
# Chúng ta không cần cài build-essential hay libjpeg-dev vì Pillow, FastAPI, uvicorn
# hiện nay đều cung cấp sẵn wheel (binary) được pre-compiled cho Debian/Linux.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY . .

# Expose port chạy dịch vụ
EXPOSE 8000

# Chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
