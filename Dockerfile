# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
gcc \
g++ \
libgl1 \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender-dev \
&& rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install paddleocr==3.1.0 --no-deps  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
RUN python -m pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Copy application code
COPY . .

RUN pip install ./parse-any-doc


# Create temp directory
RUN mkdir -p /app/temp

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]