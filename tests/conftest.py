"""
Shared pytest configuration and fixtures for Music Genre Recognition AI test suite.
Caches predictor and test client across test sessions for ultra-fast execution.
"""

import os
import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from app.main import app, get_predictor
from app.service import MusicGenrePredictor


@pytest.fixture(scope="session")
def client():
    """Session-wide TestClient for FastAPI endpoints."""
    return TestClient(app)


@pytest.fixture(scope="session")
def predictor():
    """Session-wide MusicGenrePredictor instance (AudioCNN backbone)."""
    return get_predictor()


@pytest.fixture(scope="session")
def sample_sine_wav_bytes():
    """Generates standard 8-second sine wave audio in memory (WAV format, 16kHz)."""
    sr = 16000
    duration = 8.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz standard A tone
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="session")
def real_blues_audio_bytes():
    """Reads real sample audio from data/audio/blues/blues_0001.wav if available."""
    path = os.path.join("data", "audio", "blues", "blues_0001.wav")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    # Fallback to synthesized audio
    sr = 16000
    t = np.linspace(0, 8, sr * 8, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()
