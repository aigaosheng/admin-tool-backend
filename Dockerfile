# ========================
# Stage 1: Build stage
# ========================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and local package
COPY requirements.txt .
COPY constraints.txt .
COPY ./parse-any-doc /app/parse-any-doc

# Create virtual environment for cleaner dependency management
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies with explicit CPU-only versions
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch torchvision torchaudio --no-deps

# Install other requirements
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install PaddlePaddle CPU version
RUN pip install --no-cache-dir --prefix=/install \
    paddlepaddle==3.1.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install local package
RUN pip install --no-cache-dir --prefix=/install ./parse-any-doc

# Clean up unnecessary packages that might pull in GPU dependencies
# RUN pip uninstall -y \
#     nvidia-cublas-cu11 \
#     nvidia-cuda-runtime-cu11 \
#     nvidia-cudnn-cu11 \
#     nvidia-cufft-cu11 \
#     nvidia-curand-cu11 \
#     nvidia-cusparse-cu11 \
#     nvidia-cusolver-cu11 \
#     nvidia-ml-py3 \
#     --yes || true

# ========================
# Stage 2: Runtime stage  
# ========================
FROM python:3.11-slim

WORKDIR /app

# Install only essential runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /install /usr/local
# ENV PATH="/opt/venv/bin:$PATH"

# Copy application code (be selective about what you copy)
COPY main.py .
COPY .env .
# COPY gcloud-gemini-key.json .
# Add other specific directories you need, avoid copying everything with COPY . .

# Create temp directory with proper permissions
# RUN mkdir -p /app/temp && chmod 755 /app/temp

# Create non-root user for security
# RUN adduser --disabled-password --gecos '' --uid 1000 appuser && \
    # chown -R appuser:appuser /app
# USER appuser

# Expose port
EXPOSE 8000

# Production run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]