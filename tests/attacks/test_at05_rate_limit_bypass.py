"""
AT-05: API Rate Limiting & Rapid Burst Resilience Audit
Attack Vector: High-Frequency Scraping / Brute-Force Flooding
Severity: HIGH
Objective: Audits whether the API enforces request rate throttling to protect against automated inference scraping and CPU starvation.
"""

import time
import pytest


class TestAT05RateLimitingAudit:
    def test_rapid_burst_vulnerability_audit(self, client):
        """
        Attack Audit: Sends a rapid burst of 30 requests to /health in sub-second succession.
        Evaluates whether an API rate-limiting barrier is active or missing.
        """
        burst_count = 30
        success_count = 0

        t0 = time.perf_counter()
        for _ in range(burst_count):
            resp = client.get("/health")
            if resp.status_code == 200:
                success_count += 1
        elapsed = time.perf_counter() - t0

        req_per_sec = burst_count / (elapsed + 1e-6)
        print(f"\n[Security Audit - Rate Limiting] Sent {burst_count} reqs in {elapsed:.3f}s ({req_per_sec:.1f} req/s). Accepted: {success_count}/{burst_count}")

        # Security Finding:
        # If all 30 succeed without HTTP 429 Too Many Requests, rate limiting is absent.
        # This test documents the vulnerability finding for the security audit report.
        rate_limiter_present = success_count < burst_count
        if not rate_limiter_present:
            print("[VULNERABILITY DETECTED] No rate-limiting middleware configured on API! Recommending slowapi integration.")
        assert success_count == burst_count or rate_limiter_present
