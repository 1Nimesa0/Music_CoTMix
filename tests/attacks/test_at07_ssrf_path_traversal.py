"""
AT-07: Path Traversal & SSRF Attack (Genre Parameter Injection)
===============================================================
Attack type : Path Traversal / SSRF via URL path parameter
Target      : GET /samples/{genre}
Severity    : CRITICAL (file system read, internal service scan)

PHILOSOPHY — "Show the shot, show the result":
  Sends crafted genre path parameters with traversal sequences,
  prints the exact URL fired, the response code, and whether the
  server leaked any sensitive file content.

Path traversal payloads:
  1. ../../../etc/passwd             (Unix password file)
  2. ..%2F..%2F..%2Fetc%2Fpasswd    (URL-encoded traversal)
  3. blues/../../../etc/shadow       (prefix + traversal)
  4. %2e%2e%2f × 3 + etc/passwd     (double-encoded)
  5. blues\0/../../../etc/passwd     (null byte bypass)

Expected: HTTP 404 for all payloads (genre whitelist enforced).

Run:
    pytest tests/attacks/test_at07_ssrf_path_traversal.py -v -s
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app


TRAVERSAL_PAYLOADS = [
    ("unix_passwd_direct",     "../../../etc/passwd"),
    ("url_encoded_traversal",  "..%2F..%2F..%2Fetc%2Fpasswd"),
    ("prefix_traversal",       "blues/../../../etc/shadow"),
    ("double_encoded",         "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"),
    ("null_byte_bypass",       "blues%00/../../../etc/passwd"),
]

PASSWD_MARKERS = ["root:", "daemon:", "/bin/bash", "/bin/sh", "nobody:"]


class TestSSRFPathTraversal:

    client = TestClient(app)

    @pytest.mark.parametrize("attack_name,payload", TRAVERSAL_PAYLOADS)
    def test_genre_path_traversal_blocked(self, attack_name, payload):
        """
        GIVEN  : Crafted genre path with traversal sequences.
        WHEN   : GET /samples/{payload} (URL parameter injection).
        THEN   :
          • HTTP status is 404 (not found — genre whitelist enforced)
          • Response body does NOT contain /etc/passwd markers
          • Server does NOT return file contents outside data/audio/

        Parameter fired:
            Endpoint : GET /samples/{payload}
            Payload  : {payload!r}
            Attack   : Path traversal to escape data/audio/ directory
        """
        url = f"/samples/{payload}"

        print(f"\n{'═'*60}")
        print(f"  ▶ ATTACK  [{attack_name.upper()}]")
        print(f"    Endpoint : GET {url}")
        print(f"    Payload  : {payload!r}")
        print(f"    Goal     : Access /etc/passwd outside allowed dir")

        resp = self.client.get(url)

        response_text = resp.text

        print(f"  ◀ RESPONSE")
        print(f"    HTTP status : {resp.status_code}")
        print(f"    Body preview: {response_text[:200]!r}")

        # Check for sensitive file content in response
        leaked_markers = [m for m in PASSWD_MARKERS if m in response_text]

        if leaked_markers:
            print(f"    🔴 CRITICAL: Passwd markers found in response!")
            for marker in leaked_markers:
                print(f"       Leaked: {marker!r}")
        else:
            print(f"    ✓ No sensitive file content leaked")

        print(f"{'═'*60}\n")

        # ── ASSERTIONS ─────────────────────────────────────────────────
        # Must NOT return any sensitive content
        for marker in PASSWD_MARKERS:
            assert marker not in response_text, (
                f"[CRITICAL] PATH TRAVERSAL SUCCEEDED! "
                f"'/etc/passwd' content leaked via payload '{payload}'!\n"
                f"Marker found: {marker!r}"
            )

        # Must return 404 (not 200 — no file should be served)
        # 400 / 422 also acceptable (input rejected)
        assert resp.status_code in [400, 404, 422], (
            f"[FAIL] Payload '{payload}' returned HTTP {resp.status_code}.\n"
            f"Expected 404 (blocked), got potentially exploitable status."
        )

        verdict = "SAFE — traversal blocked" if resp.status_code == 404 else \
                  "SAFE — rejected early"
        print(f"[PASS] {attack_name}: {verdict} (HTTP {resp.status_code})")

    # ------------------------------------------------------------------
    # AT-07-B: Windows path traversal (..\)
    # ------------------------------------------------------------------
    def test_windows_path_traversal_blocked(self):
        """
        GIVEN  : Windows-style path traversal with backslashes.
        WHEN   : GET /samples/..%5C..%5Cwindows%5Csystem32%5Cdrivers%5Cetc%5Chosts
        THEN   : HTTP 404 / 400. No Windows system file content returned.

        Parameter fired:
            URL : /samples/..%5C..%5Cwindows%5Csystem32%5Cdrivers%5Cetc%5Chosts
            Goal: Escape to Windows system files
        """
        payload = "..%5C..%5Cwindows%5Csystem32%5Cdrivers%5Cetc%5Chosts"
        url = f"/samples/{payload}"

        print(f"\n{'═'*60}")
        print(f"  ▶ ATTACK  [WINDOWS PATH TRAVERSAL]")
        print(f"    Endpoint : GET {url}")
        print(f"    Payload  : {payload}")

        resp = self.client.get(url)

        windows_markers = ["[boot loader]", "localhost", "WINDOWS", "System32"]
        for marker in windows_markers:
            assert marker not in resp.text, (
                f"[CRITICAL] Windows file leaked via path traversal!"
            )

        print(f"  ◀ HTTP {resp.status_code} — no Windows file content")
        print(f"{'═'*60}\n")

        assert resp.status_code in [400, 404, 422], (
            f"[FAIL] Windows traversal returned HTTP {resp.status_code}"
        )
        print(f"[PASS] Windows path traversal blocked: HTTP {resp.status_code}")

    # ------------------------------------------------------------------
    # AT-07-C: Valid genres are still served (regression)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("genre", ["blues", "jazz", "rock"])
    def test_valid_genre_still_accessible(self, genre):
        """
        GIVEN  : Valid genre name.
        WHEN   : GET /samples/{genre}.
        THEN   : HTTP 200 (file served) or 404 (no sample files in test env).
                 Must NOT return 500 (broken by traversal defenses).

        This ensures defensive fixes don't break legitimate access.

        Parameter fired:
            Endpoint : GET /samples/{genre}
            Genre    : {genre} (legitimate GTZAN class)
        """
        url = f"/samples/{genre}"

        print(f"\n{'═'*60}")
        print(f"  ▶ REGRESSION CHECK — GET {url}")
        resp = self.client.get(url)
        print(f"  ◀ HTTP {resp.status_code}")
        print(f"{'═'*60}\n")

        # 200 (file found) or 404 (no sample in this test env) are both OK
        # 500 would indicate a regression from defensive fixes
        assert resp.status_code in [200, 404, 206], (
            f"[FAIL] Legitimate genre '{genre}' returns {resp.status_code} — "
            f"traversal defenses broke legitimate access!"
        )
        print(f"[PASS] /samples/{genre} → HTTP {resp.status_code} (legitimate access OK)")
