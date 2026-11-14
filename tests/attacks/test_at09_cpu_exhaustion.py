"""
AT-09: Algorithmic Complexity & CPU Exhaustion Denial of Service Audit
Attack Vector: Resource Exhaustion via Pathological Audio Metadata (192kHz, 8-channel, 1-sample)
Severity: HIGH
Objective: Tests resilience of the preprocessing pipeline against audio crafted to trigger combinatorial CPU spike.
"""

import io
import time
import pytest
import numpy as np
import soundfile as sf


class TestAT09CPUExhaustionAudit:
    def test_extreme_sample_rate_resampling_resilience(self, client):
        """
        Attack Test: Sends 192,000 Hz audio track (studio master grade).
        Evaluates whether fast pre-slicing prevents resampler CPU starvation.
        """
        sr_extreme = 192000
        duration = 5.0
        y = (0.3 * np.random.randn(int(sr_extreme * duration))).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, y, sr_extreme, format="WAV")
        wav_bytes = buf.getvalue()

        t0 = time.perf_counter()
        resp = client.post(
            "/predict",
            files={"file": ("extreme_192khz.wav", wav_bytes, "audio/wav")}
        )
        elapsed = time.perf_counter() - t0

        print(f"\n[CPU Stress Test - 192kHz Audio] Processed in {elapsed:.2f}s")
        assert resp.status_code == 200
        assert elapsed < 10.0, f"CPU Exhaustion vulnerability: 192kHz audio took {elapsed:.2f}s (> 10s)"

    def test_multichannel_surround_sound_handling(self, client):
        """
        Attack Test: Sends 8-channel (7.1 surround sound) audio.
        Verifies that mono down-mixing handles multi-dimensional tensors without crashing.
        """
        sr = 16000
        duration = 4.0
        # 8 channels
        y_8ch = (0.2 * np.random.randn(int(sr * duration), 8)).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, y_8ch, sr, format="WAV")
        wav_bytes = buf.getvalue()

        t0 = time.perf_counter()
        resp = client.post(
            "/predict",
            files={"file": ("surround_8ch.wav", wav_bytes, "audio/wav")}
        )
        elapsed = time.perf_counter() - t0

        print(f"\n[CPU Stress Test - 8-channel Surround] Processed in {elapsed:.2f}s")
        assert resp.status_code == 200
        assert elapsed < 5.0, f"Channel downmix stalled: {elapsed:.2f}s"
