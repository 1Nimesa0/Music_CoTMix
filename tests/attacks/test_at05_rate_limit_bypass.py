"""
AT-05: Rate Limit Bypass Attack (Brute Force / API Scraping)
=============================================================
Attack type : Brute Force — Rate Limit Bypass
Target      : GET /health · POST /predict
Severity    : HIGH (model extraction, service abuse)

PHILOSOPHY — "Show the shot, show the result":
  Sends burst requests and measures actual throughput.
  If 100% of requests succeed at high rate → rate limiting is absent.
  Prints: request count, success count, req/sec, vulnerability verdict.

Expected finding: No rate limiting exists (discovered in Phase 1 testing).
Mitigation needed: slowapi middleware + Redis-backed throttling.

Run:
    pytest tests/attacks/test_at05_rate_limit_bypass.py -v -s
"""

import io
import time
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import app


class TestRateLimitBypass:

    client = TestClient(app)
    RATE_LIMIT_THRESHOLD_RPS = 50  # If > 50 req/s succeed, no rate limiting

    # ------------------------------------------------------------------
    # AT-05-A: Health endpoint burst (100 requests)
    # ------------------------------------------------------------------
    def test_no_rate_limiting_on_health_endpoint(self):
        """
        GIVEN  : No rate limiting configured.
        WHEN   : 100 consecutive GET /health requests are sent.
        THEN   : All 100 succeed at > 50 req/s → confirms no rate limiting.
                 Expected vulnerability: success_count == 100.

        Parameter fired:
            Endpoint   : GET /health × 100
            Headers    : (default, no API key, no throttle token)
            Attack goal: Establish that rate limit does not exist
        """
        N = 100
        success_count = 0
        timestamps = []

        print(f"\n{'═'*60}")
        print(f"  ▶ ATTACK  [BURST — GET /health × {N}]")
        print(f"    Headers : (none — no API key)")
        print(f"    Goal    : Confirm absence of rate limiting")

        t_start = time.monotonic()
        for i in range(N):
            resp = self.client.get("/health")
            if resp.status_code == 200:
                success_count += 1
                timestamps.append(time.monotonic())
            elif resp.status_code == 429:
                print(f"    ✓ Rate limit hit at request #{i+1} (HTTP 429)")
                break
        t_total = time.monotonic() - t_start

        req_per_sec = success_count / max(t_total, 0.001)

        print(f"  ◀ RESULT")
        print(f"    Total requests : {N}")
        print(f"    Successful     : {success_count}")
        print(f"    HTTP 429 seen  : {success_count < N}")
        print(f"    Rate           : {req_per_sec:.1f} req/s")
        print(f"    Duration       : {t_total:.2f}s")

        if success_count == N:
            print(f"\n  ⚠ VULNERABILITY CONFIRMED: No rate limiting!")
            print(f"     All {N} requests succeeded at {req_per_sec:.1f} req/s")
            print(f"     Risk: API can be scraped at full speed")
            print(f"     Fix : Add slowapi.Limiter('10/minute') to /health endpoint")
        else:
            print(f"\n  ✓ Rate limiting detected: {success_count}/{N} allowed")

        print(f"{'═'*60}\n")

        # Document the vulnerability (test passes either way, but logs the finding)
        # Test fails only if 0 requests succeed (server broken)
        assert success_count > 0, "[FAIL] Server completely unresponsive!"
        print(
            f"[{'WARN — NO RATE LIMIT' if success_count == N else 'PASS — rate limited'}] "
            f"Health burst: {success_count}/{N} at {req_per_sec:.0f} req/s"
        )

    # ------------------------------------------------------------------
    # AT-05-B: Predict endpoint scraping (20 requests)
    # ------------------------------------------------------------------
    def test_inference_scraping_via_rapid_requests(self):
        """
        GIVEN  : No rate limiting on /predict.
        WHEN   : 20 consecutive POST /predict with identical audio.
        THEN   : All 20 responses collected → model extraction via oracle.
                 Each response leaks predicted_genre + confidence + all_predictions.

        Parameter fired:
            Endpoint   : POST /predict × 20
            File       : np.random.randn(80000) × 0.4 → WAV (~512KB)
            Attack goal: Demonstrate model extraction via unlimited API access
        """
        N = 20
        y = (np.random.randn(80000) * 0.4).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, y, 16000, format="WAV")
        wav_bytes = buf.getvalue()

        print(f"\n{'═'*60}")
        print(f"  ▶ ATTACK  [SCRAPING — POST /predict × {N}]")
        print(f"    File     : np.random.randn(80000)×0.4 → WAV ({len(wav_bytes):,} bytes)")
        print(f"    Strategy : Identical file, rapid-fire requests")

        collected = []
        t_start = time.monotonic()

        for i in range(N):
            resp = self.client.post(
                "/predict",
                files={"file": ("scrape.wav", wav_bytes, "audio/wav")},
            )
            if resp.status_code == 200:
                data = resp.json()
                collected.append({
                    "req": i + 1,
                    "genre": data.get("predicted_genre"),
                    "conf": data.get("confidence"),
                    "status": data.get("status"),
                })
            elif resp.status_code == 429:
                print(f"    ✓ Rate limit at request #{i+1}")
                break

        t_total = time.monotonic() - t_start
        rps = len(collected) / max(t_total, 0.001)

        print(f"\n  ◀ SCRAPING RESULTS")
        print(f"  {'Req':>4} {'Genre':<12} {'Conf':>8} {'Status'}")
        print(f"  {'─'*40}")
        for entry in collected[:10]:   # Show first 10
            print(f"  {entry['req']:>4} {entry['genre']:<12} {entry['conf']:>8.4f} "
                  f"{entry['status']}")
        if len(collected) > 10:
            print(f"  ... (+{len(collected) - 10} more)")

        print(f"\n  Total collected  : {len(collected)}/{N}")
        print(f"  Throughput       : {rps:.1f} req/s")
        if len(collected) == N:
            print(f"  ⚠ VULNERABILITY: {N} predictions extracted without throttling!")
            print(f"     → {N} soft-label annotations collected for model stealing")
        print(f"{'═'*60}\n")

        assert len(collected) >= 1, "[FAIL] Cannot reach /predict for scraping test"
        print(
            f"[WARN] Scraping: {len(collected)}/{N} predictions at "
            f"{rps:.0f} req/s — no rate limiting detected"
        )

    # ------------------------------------------------------------------
    # AT-05-C: Burst timing analysis (rate limit fingerprinting)
    # ------------------------------------------------------------------
    def test_rate_limit_fingerprinting(self):
        """
        GIVEN  : Unknown rate limiting policy.
        WHEN   : 30 health requests sent with timing measurement.
        THEN   : Prints full timing table — identifies any throttling delay pattern.
                 Attacker uses this to find the optimal burst window.

        Parameter fired:
            Endpoint : GET /health × 30
            Metric   : Per-request latency (ms)
        """
        N = 30
        latencies_ms = []

        print(f"\n{'═'*60}")
        print(f"  ▶ FINGERPRINTING — Timing Analysis GET /health × {N}")

        for i in range(N):
            t0 = time.monotonic()
            resp = self.client.get("/health")
            elapsed_ms = (time.monotonic() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            if resp.status_code == 429:
                print(f"    HTTP 429 at request #{i+1}")
                break

        import numpy as np_
        arr = np_.array(latencies_ms)
        print(f"  ◀ TIMING PROFILE")
        print(f"    Requests    : {len(arr)}")
        print(f"    Mean latency: {arr.mean():.1f} ms")
        print(f"    Max latency : {arr.max():.1f} ms (spike = throttle signal)")
        print(f"    CV (std/mean): {arr.std()/arr.mean():.3f}")

        # Look for latency spike (rate limiting often causes delay)
        spike_threshold = arr.mean() + 3 * arr.std()
        spikes = [(i + 1, lat) for i, lat in enumerate(latencies_ms) if lat > spike_threshold]
        if spikes:
            print(f"    Latency spikes: {spikes}")
            print(f"    → Possible throttling at requests: {[s[0] for s in spikes]}")
        else:
            print(f"    No latency spikes → consistent throughput, no throttling")
        print(f"{'═'*60}\n")

        # Test always passes — this is a documentation test
        assert len(latencies_ms) > 0
        print(f"[INFO] Rate limit fingerprint complete: mean={arr.mean():.1f}ms")
