"""
TC-10: API Contract & JSON Schema Verification
Category: Automation Test | Layer: API
Objective: Verifies API response contract against rigid JSON schema standards to prevent breaking changes.
"""

import pytest


class TestAPIContractIntegrity:
    EXPECTED_HEALTH_KEYS = {
        "status", "model_version", "backbone", "device",
        "ram_usage_mb", "render_optimized", "classes"
    }

    EXPECTED_PREDICT_KEYS = {
        "status", "predicted_genre", "genre_sublabel", "confidence",
        "top_predictions", "all_predictions", "model_version",
        "backbone", "latency_ms", "device", "domain_pipeline",
        "explainability", "filename"
    }

    def test_health_api_contract_schema(self, client):
        """Validates contract of GET /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        missing_keys = self.EXPECTED_HEALTH_KEYS - set(data.keys())
        assert not missing_keys, f"GET /health response violates contract. Missing keys: {missing_keys}"
        assert isinstance(data["classes"], list)
        assert len(data["classes"]) == 10
        assert data["status"] == "ok"

    def test_predict_api_contract_schema(self, client, sample_sine_wav_bytes):
        """Validates contract of POST /predict endpoint."""
        response = client.post(
            "/predict",
            files={"file": ("contract_test.wav", sample_sine_wav_bytes, "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()

        missing_keys = self.EXPECTED_PREDICT_KEYS - set(data.keys())
        assert not missing_keys, f"POST /predict response violates contract. Missing keys: {missing_keys}"

        # Sub-schema assertions
        assert data["status"] in ["OK", "UNCERTAIN", "FALLBACK"]
        assert isinstance(data["top_predictions"], list)
        assert 1 <= len(data["top_predictions"]) <= 3
        for item in data["top_predictions"]:
            assert "genre" in item and "confidence" in item

        assert isinstance(data["all_predictions"], list)
        assert len(data["all_predictions"]) == 10

        # Confidence sum constraint
        conf_sum = sum(p["confidence"] for p in data["all_predictions"])
        assert 0.99 <= conf_sum <= 1.01, f"All predictions softmax probabilities do not sum to 1.0: {conf_sum}"

    def test_sample_predict_api_contract_schema(self, client):
        """Validates contract of POST /sample-predict/{genre} endpoint."""
        response = client.post("/sample-predict/blues")
        assert response.status_code == 200
        data = response.json()

        assert "sample_genre" in data, "Missing 'sample_genre' in /sample-predict response"
        assert "audio_url" in data, "Missing 'audio_url' in /sample-predict response"
        assert data["sample_genre"] == "blues"
        assert data["audio_url"] == "/samples/blues"
