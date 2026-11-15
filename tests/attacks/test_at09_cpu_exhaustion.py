"""
AT-09: Denial of Service via Malformed Audio (CPU & Resource Exhaustion)
========================================================================
Attack type : Algorithmic Complexity / Resource Exhaustion DoS
Target      : POST /predict (Audio Preprocessing Pipeline)
Severity    : HIGH (CPU spikes, memory allocation stress, worker freeze)

PHILOSOPHY — "Show the shot, show the result":
  Crafts pathological audio signals designed to trigger heavy processing
  in resampling, channel downmixing, or spectrogram padding.
  Prints:
    [REQUEST]  → Exact audio signal characteristics (sample rate, channels, size, duration)
    [EXECUTION]→ Processing time elapsed, HTTP status code
    [RESULT]   → Response JSON / error detail, health verification afterwards

Attack scenarios:
  1. Extreme Sample Rate (192 kHz audio) — stresses FIR polyphase resampler
  2. Minimal Pathological Audio (1 single sample) — stresses zero-padding logic
  3. High Channel Count (8-channel 7.1 surround) — stresses channel downmix

Run:
    pytest tests/attacks/test_at09_cpu_exhaustion.py -v -s
"""

import io
import json
import time
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import app


class TestCPUExhaustionDoS:

    client = TestClient(app)
    CPU_TIMEOUT_S = 30.0

    # ------------------------------------------------------------------
    # AT-09-A: Extreme sample rate (192 kHz)
    # ------------------------------------------------------------------
    def test_extreme_sample_rate_audio(self):
        """
        GIVEN  : WAV encoded at 192,000 Hz (10 seconds, 1,920,000 samples).
        WHEN   : POST /predict is called with this ultra-high sample rate file.
        THEN   :
          • Resampling completes within acceptable SLA (< 30s) or fails gracefully.
          • Server survives without crash and remains responsive.

        Parameter fired:
            Endpoint     : POST /predict
            Sample Rate  : 192,000 Hz (12x standard 16kHz)
            Duration     : 10.0 seconds (1.92M samples)
            File Size    : ~7.68 MB PCM
            Content-Type : audio/wav
        """
        sr = 192000
        duration = 10.0
        y = (np.random.randn(int(sr * duration)) * 0.3).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        wav_bytes = buf.getvalue()

        print(f"\n{'═'*65}")
        print(f"  ▶ ATTACK  [CPU EXHAUSTION — 192 kHz Sample Rate]")
        print(f"    Sample Rate  : {sr:,} Hz (12x native 16kHz)")
        print(f"    Duration     : {duration} s")
        print(f"    Total samples: {len(y):,}")
        print(f"    Payload size : {len(wav_bytes):,} bytes ({len(wav_bytes)/1024/1024:.2f} MB)")
        print(f"    Target risk  : Resampling filter convolution CPU spike")

        t0 = time.monotonic()
        resp = self.client.post(
            "/predict",
            files={"file": ("high_sr_192khz.wav", wav_bytes, "audio/wav")}
        )
        elapsed = time.monotonic() - t0

        try:
            body = resp.json()
            body_str = json.dumps(body, ensure_ascii=False)
        except Exception:
            body_str = resp.text[:250]

        print(f"\n  ◀ RESPONSE")
        print(f"    HTTP Status  : {resp.status_code}")
        print(f"    Elapsed Time : {elapsed:.2f} s (Budget < {self.CPU_TIMEOUT_S}s)")
        print(f"    Body Preview : {body_str[:200]}")

        # Post-attack health verification
        h_resp = self.client.get("/health")
        print(f"    Post-attack /health : HTTP {h_resp.status_code}")
        print(f"{'═'*65}\n")

        assert elapsed < self.CPU_TIMEOUT_S, (
            f"[FAIL] 192kHz audio processing took {elapsed:.2f}s > {self.CPU_TIMEOUT_S}s (CPU exhaustion)!"
        )
        assert resp.status_code in [200, 400, 413, 422, 500], f"Unexpected status: {resp.status_code}"
        assert h_resp.status_code == 200, "[FAIL] Server is unresponsive after 192kHz payload!"
        print(f"[PASS] 192kHz stress test handled in {elapsed:.2f}s, server remains operational")

    # ------------------------------------------------------------------
    # AT-09-B: Pathological minimal audio (1 sample)
    # ------------------------------------------------------------------
    def test_pathological_minimal_audio(self):
        """
        GIVEN  : Pathologically short WAV audio containing only 1 single sample (0.5).
        WHEN   : POST /predict is fired.
        THEN   :
          • Fails fast or pads gracefully without crashing (handled < 5s).
          • Does not hang in infinite padding loop.

        Parameter fired:
            Endpoint     : POST /predict
            Audio Length : 1 single sample [0.5]
            Sample Rate  : 16,000 Hz
            Content-Type : audio/wav
        """
        y = np.array([0.5], dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, y, 16000, format="WAV")
        wav_bytes = buf.getvalue()

        print(f"\n{'═'*65}")
        print(f"  ▶ ATTACK  [ALGORITHMIC COMPLEXITY — 1-Sample Audio]")
        print(f"    Total Samples: 1 sample")
        print(f"    Payload Size : {len(wav_bytes)} bytes")
        print(f"    Target risk  : Zero-division, shape mismatch in STFT, infinite loop")

        t0 = time.monotonic()
        resp = self.client.post(
            "/predict",
            files={"file": ("one_sample.wav", wav_bytes, "audio/wav")}
        )
        elapsed = time.monotonic() - t0

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}

        print(f"\n  ◀ RESPONSE")
        print(f"    HTTP Status  : {resp.status_code}")
        print(f"    Elapsed Time : {elapsed:.3f} s (Must fail fast < 5s)")
        print(f"    Body         : {json.dumps(body, ensure_ascii=False, default=str)}")
        print(f"{'═'*65}\n")

        assert elapsed < 5.0, f"[FAIL] 1-sample audio hung for {elapsed:.2f}s!"
        # Server can reject with 4xx/500 or classify as FALLBACK
        assert resp.status_code in [200, 400, 422, 500]
        print(f"[PASS] 1-sample audio processed/rejected in {elapsed:.3f}s")

    # ------------------------------------------------------------------
    # AT-09-C: 8-Channel surround audio (7.1 channel stress)
    # ------------------------------------------------------------------
    def test_many_channels_audio_downmix(self):
        """
        GIVEN  : 8-channel WAV file (7.1 surround sound, 10s at 16kHz).
        WHEN   : POST /predict is submitted.
        THEN   :
          • Downmixing channel tensor completes cleanly < 15s.
          • Server remains healthy afterwards.

        Parameter fired:
            Endpoint     : POST /predict
            Channels     : 8 (7.1 Surround Sound)
            Sample Rate  : 16,000 Hz
            Duration     : 10.0 seconds
            Data Shape   : [160,000, 8]
        """
        duration = 10.0
        sr = 16000
        y = (np.random.randn(int(sr * duration), 8) * 0.3).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        wav_bytes = buf.getvalue()

        print(f"\n{'═'*65}")
        print(f"  ▶ ATTACK  [SURROUND SOUND CHANNEL FLOOD — 8 Channels]")
        print(f"    Channels     : 8 channels (7.1)")
        print(f"    Duration     : {duration} s @ {sr} Hz")
        print(f"    Payload Size : {len(wav_bytes):,} bytes ({len(wav_bytes)/1024/1024:.2f} MB)")
        print(f"    Target risk  : Mean downmix buffer allocation and thread lock")

        t0 = time.monotonic()
        resp = self.client.post(
            "/predict",
            files={"file": ("surround_8ch.wav", wav_bytes, "audio/wav")}
        )
        elapsed = time.monotonic() - t0

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}

        print(f"\n  ◀ RESPONSE")
        print(f"    HTTP Status  : {resp.status_code}")
        print(f"    Elapsed Time : {elapsed:.2f} s (Budget < 15s)")
        print(f"    Body Preview : {str(body)[:200]}")

        h_resp = self.client.get("/health")
        print(f"    Post-attack /health : HTTP {h_resp.status_code}")
        print(f"{'═'*65}\n")

        assert elapsed < 15.0, f"[FAIL] 8-channel downmix took {elapsed:.2f}s > 15s"
        assert h_resp.status_code == 200, "[FAIL] Server crashed after 8-channel downmix!"
        print(f"[PASS] 8-channel audio handled in {elapsed:.2f}s, server healthy")
