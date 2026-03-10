# ==============================================================================
# Baby Cry Diagnostic System - Cloud Deployment Dockerfile
# Optimized for Render, Railway, Fly.io, and other PaaS platforms
# ==============================================================================
# 
# Build: docker build -t baby-cry-api -f cloud_deployment/Dockerfile .
# Run:   docker run -p 8000:8000 -e PORT=8000 baby-cry-api
#
# ==============================================================================

# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

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
    libavcodec-extra \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# ==============================================================================
# Stage 2: Dependencies installation
# ==============================================================================
FROM base as dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY cloud_deployment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for cloud - smaller image)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install ML libraries
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    tensorflow-cpu==2.15.0 \
    tensorflow_hub==0.15.0 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    scipy==1.11.4

# ==============================================================================
# Stage 3: Application
# ==============================================================================
FROM dependencies as application

WORKDIR /app

# Copy application code
COPY cloud_deployment/ /app/cloud_deployment/
COPY baby_cry_diagnostic/backend/models/ /app/cloud_deployment/models/
COPY baby_cry_diagnostic/backend/services/ /app/cloud_deployment/services/

# Copy trained model weights
COPY trained_classifiers/ /app/trained_classifiers/
COPY ast_baby_cry_optimized/ /app/ast_baby_cry_optimized/
COPY ast_respiratory_optimized/ /app/ast_respiratory_optimized/

# Create necessary directories
RUN mkdir -p /app/model_cache /app/reports /app/logs /tmp/model_cache

# Set ownership
RUN chown -R appuser:appuser /app /tmp/model_cache

# Switch to non-root user
USER appuser

# Set environment defaults
ENV MODEL_CACHE_DIR=/tmp/model_cache
ENV REPORTS_DIR=/app/reports
ENV LOGS_DIR=/app/logs
ENV PORT=10000
ENV USE_GPU=false

# Expose port (Render injects PORT env var)
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Working directory for runtime
WORKDIR /app/cloud_deployment

# Start command (use $PORT from environment)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --loop uvloop --http httptools
