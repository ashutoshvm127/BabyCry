# ==============================================================================
# Baby Cry Diagnostic System - Cloud Deployment Dockerfile
# Standalone Dockerfile - works when Root Directory is set to cloud_deployment
# ==============================================================================

FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install ML libraries
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    scipy==1.11.4

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/model_cache /tmp/model_cache

# Set environment defaults
ENV MODEL_CACHE_DIR=/tmp/model_cache
ENV PORT=10000
ENV USE_GPU=false
ENV HF_HOME=/tmp/model_cache

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start command
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
