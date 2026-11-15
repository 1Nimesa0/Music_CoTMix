"""
TC-01: Health Check & Model Metadata Validation
================================================
Type    : Automation Test
Layer   : API
Endpoint: GET /health

PHILOSOPHY — "Show the shot, show the result":
  Each test method prints:
    [REQUEST]  → exact URL + method + headers fired
    [RESPONSE] → HTTP status + full JSON body
    [ASSERT]   → what condition was checked and whether it passed

Run:
    pytest tests/test_01_automation_health.py -v -s
"""

import pytest


class TestHealthCheckAutomation:
    """
    Smoke-tests the /health endpoint.
    Uses the session-scoped `trace` fixture (from conftest) which prints every
    HTTP request and response automatically.
    """

    # ------------------------------------------------------------------
    # TC-01-A: Status field
    # ------------------------------------------------------------------
    def test_health_endpoint_status(self, trace):
        """
        GIVEN  : The API server is running and the checkpoint is loaded.
        WHEN   : Client sends GET /health (no headers, no body).
        THEN   : HTTP 200 is returned with JSON field status == 'ok'.

        Parameter fired:
            GET /health
            Headers: (default TestClient headers)
            Body   : (none)
        """
        # ── EXECUTION ──────────────────────────────────────────────────
        response = trace.get("/health")          # TraceLogger prints full req/resp

        # ── ASSERTIONS ─────────────────────────────────────────────────
        assert response.status_code == 200, (
            f"[FAIL] Expected HTTP 200, got {response.status_code}"
        )
        data = response.json()
        assert data.get("status") == "ok", (
            f"[FAIL] Expected status='ok', got '{data.get('status')}'"
        )
        print(f"[PASS] /health → status='{data['status']}', HTTP {response.status_code}")

    # ------------------------------------------------------------------
    # TC-01-B: Model version + backbone
    # ------------------------------------------------------------------
    def test_model_version_and_backbone(self, trace):
        """
        GIVEN  : Model checkpoint music_cotmix_cnn.pt is loaded at startup.
        WHEN   : Client sends GET /health.
        THEN   : Response contains non-empty model_version and a known backbone.

        Parameter fired:
            GET /health
        Expected fields in JSON:
            model_version : non-empty string
            backbone      : one of ["cnn", "audiocnn", "dinov3"]
        """
        response = trace.get("/health")
        data = response.json()

        assert "model_version" in data, "[FAIL] 'model_version' missing from /health response"
        assert len(data["model_version"]) > 0, "[FAIL] model_version is empty string"
        assert "backbone" in data, "[FAIL] 'backbone' missing from /health response"
        assert data["backbone"] in ["cnn", "audiocnn", "dinov3"], (
            f"[FAIL] Unknown backbone '{data['backbone']}'"
        )
        print(
            f"[PASS] model_version='{data['model_version']}', "
            f"backbone='{data['backbone']}'"
        )

    # ------------------------------------------------------------------
    # TC-01-C: Classes list (10 GTZAN genres)
    # ------------------------------------------------------------------
    def test_classes_list_completeness(self, trace):
        """
        GIVEN  : System trained on GTZAN dataset (10 genres).
        WHEN   : Client sends GET /health.
        THEN   : 'classes' array contains exactly the 10 expected genre labels.

        Parameter fired:
            GET /health
        Expected JSON:
            classes: ["blues","classical","country","disco","hiphop",
                      "jazz","metal","pop","reggae","rock"]  (order-independent)
        """
        response = trace.get("/health")
        data = response.json()

        assert "classes" in data, "[FAIL] 'classes' missing from /health response"
        classes = data["classes"]
        expected = {
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        }
        actual = set(classes)
        assert actual == expected, (
            f"[FAIL] Class mismatch:\n"
            f"  Missing : {expected - actual}\n"
            f"  Extra   : {actual - expected}"
        )
        assert len(classes) == 10, f"[FAIL] Expected 10 classes, got {len(classes)}"
        print(f"[PASS] 10 genre classes verified: {sorted(classes)}")

    # ------------------------------------------------------------------
    # TC-01-D: RAM usage budget
    # ------------------------------------------------------------------
    def test_memory_usage_within_render_budget(self, trace):
        """
        GIVEN  : Render Free Tier has 512 MB RAM.
        WHEN   : Client sends GET /health after model is fully loaded.
        THEN   : ram_usage_mb < 300 on CPU / < 1500 on CUDA.

        Parameter fired:
            GET /health
        Expected JSON:
            ram_usage_mb : float — process RSS memory in MB
            device       : "cpu" | "cuda:0"
        Threshold logic:
            CUDA device → budget = 1500 MB (GPU VRAM not counted in RSS)
            CPU  device → budget = 300 MB  (Render Free Tier safe zone)
        """
        response = trace.get("/health")
        data = response.json()

        assert "ram_usage_mb" in data, "[FAIL] 'ram_usage_mb' missing"
        ram_mb = data["ram_usage_mb"]
        device = data.get("device", "cpu")
        max_budget = 1500.0 if "cuda" in device else 300.0

        assert ram_mb < max_budget, (
            f"[FAIL] RAM too high: {ram_mb:.1f} MB "
            f"(budget for '{device}': {max_budget} MB)"
        )
        print(
            f"[PASS] RAM={ram_mb:.1f} MB on '{device}' "
            f"(budget={max_budget:.0f} MB)"
        )
