"""
TC-06: Input Validation & Audio Format Guardrails
Category: Automation Test | Layer: API
Objective: Verifies rejection of illegal file types, empty files, and silent audio inputs.
"""

import io
import pytest
import numpy as np
import soundfile as sf


class TestInputValidationGuardrails:
    def test_reject_unsupported_file_extension(self, client):
        """Uploading non-audio extension (e.g., .exe, .sh, .txt) must return HTTP 400."""
        response = client.post(
            "/predict",
            files={"file": ("malicious_payload.exe", b"MZ\x90\x00executable_dummy_code", "application/octet-stream")}
        )
        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "Định dạng file không hỗ trợ" in detail

    def test_reject_corrupted_empty_file(self, client):
        """Uploading an audio file smaller than 100 bytes must trigger error handling."""
        response = client.post(
            "/predict",
            files={"file": ("tiny_broken.wav", b"RIFFbroken", "audio/wav")}
        )
        # Should return 400 or 500 with meaningful error detail
        assert response.status_code in [400, 422, 500]

    def test_reject_completely_silent_audio(self, client):
        """
        Uploading completely silent audio (all zeros) must be caught by
        Volume Normalization guardrail (max_val < 0.003) and gracefully return FALLBACK status.
        """
        sr = 16000
        silent = np.zeros(sr * 8, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silent, sr, format="WAV")

        response = client.post(
            "/predict",
            files={"file": ("silence.wav", buf.getvalue(), "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "FALLBACK"
        assert "quá nhỏ hoặc im lặng" in data.get("error", "")

    @pytest.mark.parametrize("valid_ext", [".wav", ".mp3", ".ogg", ".flac", ".m4a"])
    def test_allowed_extensions_pass_file_filter(self, client, sample_sine_wav_bytes, valid_ext):
        """File filter must permit all standardized audio formats."""
        response = client.post(
            "/predict",
            files={"file": (f"track{valid_ext}", sample_sine_wav_bytes, "audio/wav")}
        )
        # The extension check should pass (status != 400)
        assert response.status_code != 400, f"Allowed extension '{valid_ext}' was wrongfully blocked!"
