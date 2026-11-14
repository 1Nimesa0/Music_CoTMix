"""
AT-07: Directory Traversal & Resource Injection Audit
Attack Vector: Path Traversal / Arbitrary File Read via Parameter Tampering
Severity: CRITICAL
Objective: Tests whether endpoint /samples/{genre} allows attackers to escape the audio folder and read system files.
"""

import pytest


class TestAT07PathTraversalAudit:
    TRAVERSAL_PAYLOADS = [
        "../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "..\\..\\windows\\win.ini",
        "blues/../../../app/main.py",
        "....//....//....//etc/shadow"
    ]

    @pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
    def test_sample_endpoint_traversal_blocked(self, client, payload):
        """
        Attack Test: Attempts to traverse out of data/audio/ into root filesystem.
        Requirement: The endpoint MUST return HTTP 404 Not Found and NEVER leak server files (HTTP 200).
        """
        response = client.get(f"/samples/{payload}")
        assert response.status_code == 404, f"VULNERABILITY: Path traversal succeeded or unhandled with status {response.status_code} on {payload}"

        # If somehow status was 200, ensure sensitive files are not exposed
        if response.status_code == 200:
            content = response.text
            assert "root:" not in content
            assert "[fonts]" not in content.lower()

    @pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
    def test_sample_predict_traversal_blocked(self, client, payload):
        """Attack Test: Attempts directory traversal via /sample-predict/{genre}."""
        response = client.post(f"/sample-predict/{payload}")
        assert response.status_code == 404, f"VULNERABILITY: /sample-predict traversal returned {response.status_code}"
