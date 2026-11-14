"""
TC-01: Health Check & Model Metadata Validation
Category: Automation Test | Layer: API
Objective: Verifies API readiness, active model metadata, supported genres, and memory safety.
"""

import pytest


class TestHealthCheckAutomation:
    def test_health_endpoint_status(self, client):
        """Endpoint /health must return HTTP 200 OK with status='ok'."""
        response = client.get("/health")
        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"

    def test_model_version_and_backbone(self, client):
        """Response must specify model_version and backbone architecture."""
        response = client.get("/health")
        data = response.json()
        assert "model_version" in data, "Missing model_version in health response"
        assert len(data["model_version"]) > 0, "model_version string is empty"
        assert "backbone" in data, "Missing backbone in health response"
        assert data["backbone"] in ["cnn", "audiocnn", "dinov3"], f"Unexpected backbone: {data['backbone']}"

    def test_classes_list_completeness(self, client):
        """API must support exactly 10 standard GTZAN music genre classes."""
        response = client.get("/health")
        data = response.json()
        assert "classes" in data, "Missing classes in health response"
        classes = data["classes"]
        expected_classes = {
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        }
        assert set(classes) == expected_classes, f"Mismatch in supported classes: {set(classes) ^ expected_classes}"
        assert len(classes) == 10, f"Expected 10 classes, got {len(classes)}"

    def test_memory_usage_within_render_budget(self, client):
        """
        Verifies RAM consumption. On CPU (Render), RAM is < 300MB.
        When CUDA GPU runtime is active, memory mapped context is < 1500MB.
        """
        response = client.get("/health")
        data = response.json()
        assert "ram_usage_mb" in data, "Missing ram_usage_mb in health response"
        ram_mb = data["ram_usage_mb"]
        device = data.get("device", "cpu")
        max_budget = 1500.0 if "cuda" in device else 300.0
        assert ram_mb < max_budget, f"RAM usage too high: {ram_mb} MB (budget: {max_budget}MB)"
