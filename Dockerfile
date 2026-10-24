# syntax=docker/dockerfile:1
FROM python:3.11-slim

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

# Copy application files and model checkpoints
COPY app/ ./app/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/audio/ ./data/audio/

# Default Environment Variables for Render Free Tier
ENV PORT=8000 \
    BACKBONE=cnn \
    MODEL_PATH=models/checkpoints/music_cotmix_cnn.pt \
    DEVICE=cpu \
    CONFIDENCE_THRESHOLD=0.80

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
