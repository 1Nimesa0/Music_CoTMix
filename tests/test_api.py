"""
Unit and Integration Tests for Music Genre Recognition FastAPI Service.
Executed automatically during CI pipeline to verify model loading and API integrity.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify that the health check endpoint returns 200 OK and model status."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert data.get("status") == "ok", "System status should be 'ok'"
    assert "model_version" in data, "Response must include model_version"
    assert "classes" in data, "Response must list classification classes"
    assert len(data["classes"]) == 10, f"Expected 10 genres, got {len(data['classes'])}"


def test_sample_audio_stream():
    """Verify that preset audio sample can be retrieved."""
    response = client.get("/samples/blues")
    assert response.status_code == 200, "Should successfully fetch sample blues audio"
    assert response.headers["content-type"] in ["audio/wav", "audio/x-wav"], "Content-Type must be audio/wav"
    assert len(response.content) > 0, "Audio content must not be empty"


def test_sample_prediction():
    """Verify that sample audio prediction endpoint works end-to-end."""
    response = client.post("/sample-predict/blues")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    result = response.json()
    assert "predicted_genre" in result, "Result must contain predicted_genre"
    assert "confidence" in result, "Result must contain confidence"
    assert "top_predictions" in result, "Result must contain top_predictions"
    assert result.get("sample_genre") == "blues"
