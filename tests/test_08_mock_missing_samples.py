"""
TC-08: Mock Preset Samples & Missing Resource Handling
Category: Mock Test | Layer: API
Objective: Verifies HTTP 404 handling and Vietnamese error messages when sample files or directories are missing.
"""

from unittest.mock import patch
import pytest


class TestMockMissingSampleResources:
    def test_nonexistent_genre_stream_returns_404(self, client):
        """GET /samples/{invalid_genre} must return HTTP 404."""
        response = client.get("/samples/nonexistent_genre_abc")
        assert response.status_code == 404
        assert "Thể loại mẫu không tồn tại" in response.json().get("detail", "")

    def test_nonexistent_genre_predict_returns_404(self, client):
        """POST /sample-predict/{invalid_genre} must return HTTP 404."""
        response = client.post("/sample-predict/cyberpunk_edm")
        assert response.status_code == 404
        assert "Thể loại mẫu không tồn tại" in response.json().get("detail", "")

    @patch("os.path.isdir", return_value=True)
    @patch("os.listdir", return_value=[])
    def test_mock_empty_genre_dir_returns_404(self, mock_listdir, mock_isdir, client):
        """
        Mock Test: When genre directory exists but contains zero WAV files,
        GET /samples/{genre} must return HTTP 404 with 'Không tìm thấy file mẫu'.
        """
        response = client.get("/samples/blues")
        assert response.status_code == 404
        assert "Không tìm thấy file mẫu" in response.json().get("detail", "")

    @patch("os.path.isdir", return_value=True)
    @patch("os.listdir", return_value=[])
    def test_mock_empty_genre_dir_sample_predict_returns_404(self, mock_listdir, mock_isdir, client):
        """
        Mock Test: When genre directory exists but has no audio files,
        POST /sample-predict/{genre} must return HTTP 404.
        """
        response = client.post("/sample-predict/blues")
        assert response.status_code == 404
        assert "Không tìm thấy file mẫu" in response.json().get("detail", "")
