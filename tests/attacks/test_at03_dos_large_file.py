"""
AT-03: Denial of Service (DoS) via Large Audio File & Resource Exhaustion
Attack Vector: Volumetric Resource Exhaustion / Memory Bomb
Severity: CRITICAL
Objective: Tests API survivability when bombarded with large audio files designed to exhaust PaaS memory limits.
"""

import io
import time
import pytest
import numpy as np
import soundfile as sf


class TestAT03DoSLargeFile:
    def test_large_audio_payload_survivability(self, client):
        """
        Attack Test: Sends a synthesized 45-second high-resolution audio file (~15MB).
        Requirement: The server must handle, slice, and respond without an unrecoverable OOM crash.
        """
        sr = 44100
        duration = 35.0  # seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        large_bytes = buf.getvalue()
        file_size_mb = len(large_bytes) / (1024 * 1024)
        print(f"\n[DoS Test] Uploading large payload: {file_size_mb:.2f} MB")

        t0 = time.perf_counter()
        response = client.post(
            "/predict",
            files={"file": ("large_exploit_audio.wav", large_bytes, "audio/wav")}
        )
        elapsed = time.perf_counter() - t0

        assert response.status_code == 200, f"Server failed on large audio upload: {response.status_code}"
        assert elapsed < 15.0, f"Server took too long to process large file: {elapsed:.2f}s"

        # Verify system health immediately after large upload to prove no memory death
        health = client.get("/health")
        assert health.status_code == 200, "Server failed health check after large audio processing"
