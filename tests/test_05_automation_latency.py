"""
TC-05: Performance, SLA & Inference Latency Benchmark
Category: Automation Test | Layer: API + System
Objective: Verifies p95 latency adheres to Cloud PaaS SLA requirements (< 1500ms end-to-end, < 100ms model engine).
"""

import time
import pytest
import numpy as np


class TestPerformanceAndLatency:
    HEALTH_SLA_MS = 100.0
    INFERENCE_P95_SLA_MS = 1500.0
    AI_ENGINE_SLA_MS = 120.0

    def test_health_check_latency_p95(self, client):
        """GET /health must respond in < 100ms across 15 iterations."""
        latencies = []
        for _ in range(15):
            t0 = time.perf_counter()
            resp = client.get("/health")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            latencies.append(elapsed_ms)

        p95 = np.percentile(latencies, 95)
        avg = np.mean(latencies)
        print(f"\n[Perf - /health] avg={avg:.2f}ms, p95={p95:.2f}ms")
        assert p95 < self.HEALTH_SLA_MS, f"/health p95 latency exceeded SLA: {p95:.2f}ms > {self.HEALTH_SLA_MS}ms"

    def test_inference_end_to_end_latency_p95(self, client, sample_sine_wav_bytes):
        """POST /predict end-to-end HTTP p95 latency must be within SLA (< 1500ms)."""
        latencies = []
        engine_latencies = []

        # Run 10 iterations
        for _ in range(10):
            t0 = time.perf_counter()
            resp = client.post(
                "/predict",
                files={"file": ("benchmark.wav", sample_sine_wav_bytes, "audio/wav")}
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            data = resp.json()

            latencies.append(elapsed_ms)
            if "latency_ms" in data:
                engine_latencies.append(data["latency_ms"])

        p95_http = np.percentile(latencies, 95)
        avg_http = np.mean(latencies)
        p95_engine = np.percentile(engine_latencies, 95) if engine_latencies else 0.0

        print(f"\n[Perf - /predict] HTTP avg={avg_http:.1f}ms, p95={p95_http:.1f}ms | Engine p95={p95_engine:.1f}ms")
        assert p95_http < self.INFERENCE_P95_SLA_MS, f"Inference HTTP p95 breached SLA: {p95_http:.1f}ms"
        assert p95_engine < self.AI_ENGINE_SLA_MS, f"AI Model engine p95 breached SLA: {p95_engine:.1f}ms"
