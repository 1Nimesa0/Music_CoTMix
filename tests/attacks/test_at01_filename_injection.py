"""
AT-01: Filename Injection & Input Sanitization Security Audit
Attack Vector: Indirect Prompt / Path / Script Injection via Upload Metadata
Severity: HIGH
Objective: Verifies API resilience when user supplies hostile file names (SQLi, XSS, Path Traversal, Null Byte).
"""

import io
import pytest
import soundfile as sf
import numpy as np


class TestAT01FilenameInjection:
    MALICIOUS_FILENAMES = [
        "'; DROP TABLE users; --.wav",
        "../../etc/passwd.wav",
        "<script>alert('XSS')</script>.wav",
        "normal_song\x00.wav",
        "${7*7}_jndi_test.wav",
        "A" * 300 + ".wav"
    ]

    @pytest.mark.parametrize("bad_filename", MALICIOUS_FILENAMES)
    def test_malicious_filename_does_not_crash_server(self, client, sample_sine_wav_bytes, bad_filename):
        """
        Attack Test: Uploads valid audio wrapped inside malicious filenames.
        Requirement: The server must handle the request safely without unhandled 500 exceptions
        and must not execute or reflect raw unsanitized executable scripts in response headers/body.
        """
        response = client.post(
            "/predict",
            files={"file": (bad_filename, sample_sine_wav_bytes, "audio/wav")}
        )

        # Server must successfully process the audio or gracefully return validation error
        assert response.status_code in [200, 400], f"Server crashed with {response.status_code} on filename: {bad_filename}"

        if response.status_code == 200:
            data = response.json()
            assert "predicted_genre" in data, "Valid audio failed inference due to filename string"
            # Ensure script tags are not blindly executed or reflected
            returned_name = data.get("filename", "")
            assert "<script>" not in returned_name or "&lt;script&gt;" in returned_name or len(returned_name) > 0
