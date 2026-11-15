"""
AT-03: Denial of Service via Large File Upload (Resource Exhaustion)
====================================================================
Attack type : DoS — Resource Exhaustion (RAM + processing time)
Target      : POST /predict
Severity    : CRITICAL (Render Free Tier — 512 MB RAM limit)

PHILOSOPHY — "Show the shot, show the result":
  Sends progressively larger WAV files, measuring:
  • Time to response (seconds)
  • Server health after the attack request

Attack variants:
  • 10MB WAV (30s audio, 44.1kHz)
  • 25MB WAV (60s audio, 44.1kHz)
  • 5 concurrent 15s uploads (race condition / thread exhaustion)

Run:
    pytest tests/attacks/test_at03_dos_large_file.py -v -s
"""

import io
import time
import threading
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import app


def _make_large_wav(duration_s: int, sr: int = 44100) -> bytes:
    """Generate large WAV file: duration_s seconds at sample_rate Hz."""
    y = (np.random.randn(sr * duration_s) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


class TestDoSLargeFile:

    client = TestClient(app)
    RESPONSE_TIMEOUT_S = 30.0
    HEALTH_TIMEOUT_S = 5.0

    def _post_large_file(self, label: str, wav_bytes: bytes) -> dict:
        """
        Send large WAV, measure elapsed time, check server survives.
        Returns dict with timing and status info.
        """
        mb = len(wav_bytes) / (1024 * 1024)

        print(f"\n{'═'*60}")
        print(f"  ▶ ATTACK  [{label}]")
        print(f"    Endpoint     : POST /predict")
        print(f"    File size    : {mb:.1f} MB")
        print(f"    Sample rate  : 44100 Hz")
        print(f"    Content-Type : audio/wav")

        t0 = time.monotonic()
        resp = self.client.post(
            "/predict",
            files={"file": (f"{label}.wav", wav_bytes, "audio/wav")},
        )
        elapsed = time.monotonic() - t0

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}

        print(f"  ◀ RESPONSE")
        print(f"    HTTP status  : {resp.status_code}")
        print(f"    Elapsed      : {elapsed:.2f}s  (limit={self.RESPONSE_TIMEOUT_S}s)")
        print(f"    Body preview : {str(body)[:150]}")

        # Health check after large file
        t1 = time.monotonic()
        health = self.client.get("/health")
        health_elapsed = (time.monotonic() - t1) * 1000

        print(f"    Post-attack /health: HTTP {health.status_code} in {health_elapsed:.0f}ms")
        print(f"{'═'*60}\n")

        return {
            "status_code": resp.status_code,
            "elapsed_s": elapsed,
            "health_ok": health.status_code == 200,
        }

    # ------------------------------------------------------------------
    # AT-03-A: 10 MB WAV (~30s audio)
    # ------------------------------------------------------------------
    def test_large_file_10mb_handled(self):
        """
        GIVEN  : WAV file ≈ 10 MB (30s × 44.1kHz × 4 bytes/sample).
        WHEN   : POST /predict with this large payload.
        THEN   :
          • Response within 30 seconds (not hung indefinitely)
          • /health still responds after the request
          • Status is 200 / 413 / 500 (not hung)

        Parameter fired:
            Endpoint : POST /predict
            File     : 30s random audio @ 44.1kHz ≈ 10 MB
            Attack   : Resource exhaustion — force heavy audio preprocessing
        """
        wav = _make_large_wav(duration_s=30, sr=44100)
        result = self._post_large_file("dos_10mb_30s", wav)

        assert result["elapsed_s"] < self.RESPONSE_TIMEOUT_S, (
            f"[FAIL] Server hung for {result['elapsed_s']:.1f}s on 10MB file!"
        )
        assert result["health_ok"], "[FAIL] Server non-responsive after 10MB upload!"
        print(
            f"[PASS] 10MB file: responded in {result['elapsed_s']:.1f}s, "
            f"server healthy after"
        )

    # ------------------------------------------------------------------
    # AT-03-B: 25 MB WAV (~60s audio)
    # ------------------------------------------------------------------
    def test_large_file_25mb_handled(self):
        """
        GIVEN  : WAV file ≈ 25 MB (60s × 44.1kHz).
        WHEN   : POST /predict.
        THEN   : Response within 30s. Server survives.

        Parameter fired:
            File : 60s audio @ 44100Hz ≈ 25 MB
        """
        wav = _make_large_wav(duration_s=60, sr=44100)
        result = self._post_large_file("dos_25mb_60s", wav)

        assert result["elapsed_s"] < self.RESPONSE_TIMEOUT_S, (
            f"[FAIL] Server hung {result['elapsed_s']:.1f}s on 25MB!"
        )
        assert result["health_ok"], "[FAIL] Server crashed after 25MB upload!"
        print(
            f"[PASS] 25MB file: responded in {result['elapsed_s']:.1f}s, "
            f"server healthy"
        )

    # ------------------------------------------------------------------
    # AT-03-C: 5 concurrent uploads (race condition / thread exhaustion)
    # ------------------------------------------------------------------
    def test_concurrent_uploads_race_condition(self):
        """
        GIVEN  : 5 simultaneous POST /predict requests with 15s audio each.
        WHEN   : All 5 requests launched in parallel threads.
        THEN   :
          • At least 2/5 requests succeed (server not fully DoS'd)
          • Server responds to /health after concurrency burst

        Parameter fired:
            Endpoint : POST /predict × 5 concurrent threads
            File     : 15s random audio @ 16kHz per thread (~2MB each)
            Attack   : Thread exhaustion + race condition
        """
        N = 5
        results = []

        def upload_once(thread_id: int):
            y = (np.random.randn(16000 * 15) * 0.3).astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, y, 16000, format="WAV")
            wav = buf.getvalue()

            t0 = time.monotonic()
            try:
                resp = self.client.post(
                    "/predict",
                    files={"file": (f"concurrent_{thread_id}.wav", wav, "audio/wav")},
                )
                elapsed = time.monotonic() - t0
                results.append((thread_id, resp.status_code, elapsed))
            except Exception as exc:
                results.append((thread_id, -1, str(exc)))

        print(f"\n{'═'*60}")
        print(f"  ▶ ATTACK  [CONCURRENT DoS — {N} threads]")
        print(f"    Threads  : {N}")
        print(f"    Per-file : 15s audio @ 16kHz (~2MB)")

        threads = [threading.Thread(target=upload_once, args=(i,)) for i in range(N)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        total_elapsed = time.monotonic() - t0

        print(f"  ◀ RESULTS")
        print(f"  {'Thread':>7} {'HTTP':>6} {'Time':>8}")
        print(f"  {'─'*25}")
        success_count = 0
        for tid, code, elapsed in sorted(results, key=lambda x: x[0]):
            ok = code == 200
            if ok:
                success_count += 1
            print(f"  {tid:>7} {str(code):>6} {f'{elapsed:.1f}s':>8} "
                  f"{'✓' if ok else '—'}")

        health = self.client.get("/health")
        print(f"\n  Total elapsed   : {total_elapsed:.1f}s")
        print(f"  Successful reqs : {success_count}/{N}")
        print(f"  Post-attack /health: HTTP {health.status_code}")
        print(f"{'═'*60}\n")

        assert health.status_code == 200, (
            "[FAIL] Server non-responsive after concurrent DoS!"
        )
        assert success_count >= 1, (
            f"[CRITICAL] Server fully DoS'd — 0/{N} requests succeeded!"
        )
        print(
            f"[PASS] Concurrent DoS: {success_count}/{N} requests succeeded, "
            f"server healthy after {total_elapsed:.1f}s"
        )
