# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL maintainer="Music CoTMix AI Team" \
      description="Production Docker Image for Cross-Environment Music Genre Recognition API (AudioCNN & DinoV3)" \
      version="1.0.0"

# Install system audio libraries for librosa & soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files, configs and pre-trained models
COPY app/ ./app/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/audio/ ./data/audio/

# Default Environment Variables for Serving
ENV PORT=8000 \
    BACKBONE=cnn \
    MODEL_PATH=models/checkpoints/music_cotmix_cnn.pt \
    DEVICE=cpu \
    CONFIDENCE_THRESHOLD=0.80 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start FastAPI serving server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
