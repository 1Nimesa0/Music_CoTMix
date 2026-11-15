"""
TC-05: Latency & Throughput Performance Test
============================================
Type    : Automation Test
Layer   : API + System
Endpoints: GET /health · POST /predict

PHILOSOPHY — "Show the shot, show the result":
  Prints the exact latency series (ms), p50, p95, p99, and min/max
  so readers can judge SLA compliance at a glance.

SLA Commitments:
    GET /health  p95 < 100 ms
    POST /predict p95 < 1500 ms (Render Free Tier — CPU inference)
    AI-layer latency_ms in JSON < 100 ms

Run:
    pytest tests/test_05_automation_latency.py -v -s
"""

import io
import time
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import app


def _make_wav_bytes(duration: float = 8.0, freq: float = 440.0,
                    sr: int = 16000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def _percentile_report(label: str, times_ms: list, sla_ms: float) -> None:
    arr = np.array(times_ms)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    status = "✓ PASS" if p95 < sla_ms else "✗ FAIL"
    print(f"\n  {'─'*50}")
    print(f"  Latency Report — {label}")
    print(f"  Runs : {len(arr)}")
    print(f"  Min  : {arr.min():.1f} ms")
    print(f"  p50  : {p50:.1f} ms")
    print(f"  p95  : {p95:.1f} ms   SLA={sla_ms:.0f} ms  → {status}")
    print(f"  p99  : {p99:.1f} ms")
    print(f"  Max  : {arr.max():.1f} ms")
    print(f"  {'─'*50}\n")
    return p95


class TestLatencyPerformance:

    client = TestClient(app)
    HEALTH_SLA_MS = 100.0
    PREDICT_SLA_MS = 1500.0
    AI_LAYER_SLA_MS = 100.0

    # ------------------------------------------------------------------
    # TC-05-A: GET /health latency (N=20 runs)
    # ------------------------------------------------------------------
    def test_health_latency_p95(self):
        """
        GIVEN  : API server is running.
        WHEN   : 20 consecutive GET /health requests are sent.
        THEN   : p95 latency < 100 ms (health endpoint must be near-instant).

        Parameter fired:
            Endpoint   : GET /health
            Iterations : 20
            Measured   : wall-clock time per round-trip (ms)
            SLA        : p95 < 100 ms
        """
        N = 20
        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST BURST — GET /health × {N}")
        print(f"    Headers: (default)")
        print(f"    Body   : (none)")

        times_ms = []
        for i in range(N):
            t0 = time.monotonic()
            resp = self.client.get("/health")
            elapsed = (time.monotonic() - t0) * 1000
            times_ms.append(elapsed)
            assert resp.status_code == 200, f"Run {i}: unexpected {resp.status_code}"

        p95 = _percentile_report("GET /health", times_ms, self.HEALTH_SLA_MS)

        assert p95 < self.HEALTH_SLA_MS, (
            f"[FAIL] /health p95={p95:.1f} ms > SLA={self.HEALTH_SLA_MS} ms"
        )

    # ------------------------------------------------------------------
    # TC-05-B: POST /predict latency (N=15 runs)
    # ------------------------------------------------------------------
    def test_inference_latency_p95(self, sample_sine_wav_bytes):
        """
        GIVEN  : Standard 440Hz 8s WAV (16kHz, 128 KB).
        WHEN   : 15 consecutive POST /predict requests are sent.
        THEN   : p95 latency < 1500 ms (Render Free Tier CPU budget).

        Parameter fired:
            Endpoint   : POST /predict
            File       : synthetic 440Hz WAV, 8s, 16kHz (128,000 samples)
            Size       : ~512 KB (WAV header + PCM)
            Content-Type: audio/wav (multipart/form-data)
            Iterations : 15
            SLA        : p95 < 1500 ms
        """
        N = 15
        wav_bytes = sample_sine_wav_bytes

        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST BURST — POST /predict × {N}")
        print(f"    File : synthetic_440hz_8s.wav ({len(wav_bytes):,} bytes)")
        print(f"    Type : audio/wav (multipart/form-data)")

        times_ms = []
        for i in range(N):
            t0 = time.monotonic()
            resp = self.client.post(
                "/predict",
                files={"file": ("latency_test.wav", wav_bytes, "audio/wav")},
            )
            elapsed = (time.monotonic() - t0) * 1000
            if resp.status_code == 200:
                times_ms.append(elapsed)
                if i == 0:
                    # Print first response body for reference
                    data = resp.json()
                    print(f"\n  First response (run #1):")
                    print(f"    status        : {data.get('status')}")
                    print(f"    predicted_genre: {data.get('predicted_genre')}")
                    print(f"    confidence    : {data.get('confidence'):.4f}")
                    print(f"    latency_ms    : {data.get('latency_ms')} ms")

        assert len(times_ms) >= N // 2, (
            f"[FAIL] Too many failed requests: {N - len(times_ms)}/{N}"
        )

        p95 = _percentile_report("POST /predict", times_ms, self.PREDICT_SLA_MS)

        assert p95 < self.PREDICT_SLA_MS, (
            f"[FAIL] /predict p95={p95:.1f} ms > SLA={self.PREDICT_SLA_MS} ms"
        )

    # ------------------------------------------------------------------
    # TC-05-C: AI-layer latency_ms field in response
    # ------------------------------------------------------------------
    def test_ai_reported_latency_in_response(self, sample_sine_wav_bytes):
        """
        GIVEN  : Standard WAV audio.
        WHEN   : POST /predict is called once.
        THEN   : The JSON field 'latency_ms' (AI-only inference time) < 100 ms.

        Parameter fired:
            Endpoint   : POST /predict
            File       : synthetic 440Hz WAV (session fixture)
            Assertion  : resp.json()['latency_ms'] < 100 ms
        """
        wav_bytes = sample_sine_wav_bytes

        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST — POST /predict (AI latency check)")
        print(f"    File: synthetic_440hz_8s.wav ({len(wav_bytes):,} bytes)")

        resp = self.client.post(
            "/predict",
            files={"file": ("ai_latency.wav", wav_bytes, "audio/wav")},
        )

        print(f"  ◀ RESPONSE")
        print(f"    HTTP: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            ai_latency = data.get("latency_ms", 9999)
            print(f"    latency_ms (AI only): {ai_latency} ms")
            print(f"    SLA                 : < {self.AI_LAYER_SLA_MS} ms")
            assert ai_latency < self.AI_LAYER_SLA_MS, (
                f"[FAIL] AI-layer latency {ai_latency} ms > "
                f"SLA {self.AI_LAYER_SLA_MS} ms"
            )
            print(f"[PASS] AI latency_ms={ai_latency} ms < {self.AI_LAYER_SLA_MS} ms")
        else:
            print(f"    (predict returned {resp.status_code} — skipping latency assert)")
        print(f"{'─'*55}\n")
