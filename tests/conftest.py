"""
conftest.py — Shared fixtures + Request/Response Tracing Framework
=================================================================
Every test in this suite logs a full trace:
  [REQUEST]  → what parameters were fired (URL, method, headers, body/file)
  [EXECUTION]→ how it ran (fixture used, mock patches, pytest command)
  [RESULT]   → actual HTTP status, full JSON body, assertion outcome

Usage (see printed trace in -s mode):
    pytest tests/ --ignore=tests/attacks -v -s
    pytest tests/attacks -v -s
"""

import os
import io
import json
import time
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from app.main import app, get_predictor
from app.service import MusicGenrePredictor


# ---------------------------------------------------------------------------
# Session-wide singletons
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """Session-wide FastAPI TestClient — reused across all tests (zero reload cost)."""
    return TestClient(app)


@pytest.fixture(scope="session")
def predictor():
    """Session-wide MusicGenrePredictor (AudioCNN backbone, CUDA if available)."""
    return get_predictor()


# ---------------------------------------------------------------------------
# Audio byte factories
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_sine_wav_bytes():
    """
    Standard 440 Hz sine-wave tone, 8 s, mono, 16 kHz (WAV bytes).
    Used as a lightweight, deterministic audio fixture.
    """
    sr, duration = 16000, 8.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="session")
def real_blues_audio_bytes():
    """
    Reads data/audio/blues/blues_0001.wav (GTZAN reference).
    Falls back to synthesized 220 Hz tone if the file is absent.
    """
    path = os.path.join("data", "audio", "blues", "blues_0001.wav")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    # Fallback
    sr = 16000
    t = np.linspace(0, 8, sr * 8, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Trace helper — prints full INPUT → EXECUTION → OUTPUT summary
# ---------------------------------------------------------------------------

class TraceLogger:
    """
    Wraps FastAPI TestClient to log every HTTP request/response in a
    structured, human-readable block so testers can reproduce any call.

    Usage inside a test:
        def test_something(self, trace):
            resp = trace.get("/health")
            # Printed automatically:
            # ═══ REQUEST ═══════════════════════════════
            # Method : GET
            # URL    : /health
            # Headers: (default)
            # ═══ RESPONSE ══════════════════════════════
            # Status : 200
            # Body   : {"status": "ok", ...}
            # Latency: 12.3 ms
            # ═══════════════════════════════════════════
    """

    def __init__(self, client: TestClient):
        self._client = client

    def _log(self, method: str, url: str, params_desc: str, resp, elapsed_ms: float):
        try:
            body_str = json.dumps(resp.json(), ensure_ascii=False, indent=2)
        except Exception:
            body_str = resp.text[:500]

        separator = "═" * 55
        print(f"\n{separator}")
        print(f"  ▶ REQUEST")
        print(f"    Method  : {method}")
        print(f"    URL     : {url}")
        print(f"    Payload : {params_desc}")
        print(f"  ◀ RESPONSE")
        print(f"    Status  : {resp.status_code}")
        print(f"    Latency : {elapsed_ms:.1f} ms")
        print(f"    Body    :\n{body_str}")
        print(f"{separator}\n")

    def get(self, url: str, **kwargs) -> object:
        t0 = time.monotonic()
        resp = self._client.get(url, **kwargs)
        self._log("GET", url, "(no body)", resp, (time.monotonic() - t0) * 1000)
        return resp

    def post(self, url: str, files=None, json_body=None, data=None, **kwargs) -> object:
        if files:
            fname, fbytes, ftype = list(files.values())[0]
            params_desc = (
                f"multipart/form-data · file='{fname}' "
                f"({len(fbytes):,} bytes, {ftype})"
            )
        elif json_body:
            params_desc = f"application/json · {json.dumps(json_body, ensure_ascii=False)}"
        else:
            params_desc = str(data or "(empty)")

        t0 = time.monotonic()
        if files:
            resp = self._client.post(url, files=files, **kwargs)
        elif json_body:
            resp = self._client.post(url, json=json_body, **kwargs)
        else:
            resp = self._client.post(url, data=data, **kwargs)
        self._log("POST", url, params_desc, resp, (time.monotonic() - t0) * 1000)
        return resp


@pytest.fixture
def trace(client):
    """
    Function-scoped TraceLogger fixture.
    Inject via test parameter: def test_foo(self, trace): ...
    """
    return TraceLogger(client)
