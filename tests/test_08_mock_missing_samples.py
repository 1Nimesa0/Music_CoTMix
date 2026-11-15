"""
TC-08: Mock Test — Missing Sample Files & Genre Not Found
=========================================================
Type    : Mock Test
Layer   : API
Endpoints: GET /samples/{genre} · POST /sample-predict/{genre}

PHILOSOPHY — "Show the shot, show the result":
  Patches os.listdir and os.path.isdir to simulate an empty audio directory,
  then verifies the API returns proper 404 responses.

Mock patches:
    os.listdir   → returns []  (empty directory)
    os.path.isdir → returns True  (directory exists but empty)

Run:
    pytest tests/test_08_mock_missing_samples.py -v -s
"""

import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


class TestMissingSampleFiles:

    client = TestClient(app)

    def _get_samples(self, genre: str, mock_empty: bool = False):
        """
        Helper: GET /samples/{genre}, optionally with mocked empty dir.
        Prints full request + response trace.
        """
        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST — GET /samples/{genre}")
        if mock_empty:
            print(f"    Mock active: os.listdir → []  (empty dir)")
            print(f"    Mock active: os.path.isdir → True")
        else:
            print(f"    Mock: none (real filesystem)")

        ctx_managers = []
        if mock_empty:
            ctx_managers = [
                patch("os.listdir", return_value=[]),
                patch("os.path.isdir", return_value=True),
            ]

        with (
            patch("os.listdir", return_value=[]) if mock_empty
            else _noop_ctx()
        ) as _ldir, (
            patch("os.path.isdir", return_value=True) if mock_empty
            else _noop_ctx()
        ) as _idir:
            resp = self.client.get(f"/samples/{genre}")

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:300]}

        print(f"  ◀ RESPONSE")
        print(f"    HTTP   : {resp.status_code}")
        print(f"    Body   : {json.dumps(body, ensure_ascii=False, default=str)}")
        print(f"{'─'*55}\n")
        return resp

    def _post_sample_predict(self, genre: str, mock_empty: bool = False):
        """Helper: POST /sample-predict/{genre}, optionally with mocked empty dir."""
        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST — POST /sample-predict/{genre}")
        if mock_empty:
            print(f"    Mock active: os.listdir → []")
            print(f"    Mock active: os.path.isdir → True")

        with (
            patch("os.listdir", return_value=[]) if mock_empty
            else _noop_ctx()
        ) as _ldir, (
            patch("os.path.isdir", return_value=True) if mock_empty
            else _noop_ctx()
        ) as _idir:
            resp = self.client.post(f"/sample-predict/{genre}")

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:300]}

        print(f"  ◀ RESPONSE")
        print(f"    HTTP   : {resp.status_code}")
        print(f"    Body   : {json.dumps(body, ensure_ascii=False, default=str)}")
        print(f"{'─'*55}\n")
        return resp

    # ------------------------------------------------------------------
    # TC-08-A: Non-existent genre — real filesystem
    # ------------------------------------------------------------------
    def test_404_nonexistent_genre(self):
        """
        GIVEN  : Genre 'jazz_fusion' does not exist in the system (not in GTZAN).
        WHEN   : GET /samples/jazz_fusion (no mock — real filesystem check).
        THEN   : HTTP 404.

        Parameter fired:
            Endpoint : GET /samples/jazz_fusion
            Mock     : none
        """
        resp = self._get_samples("jazz_fusion", mock_empty=False)
        assert resp.status_code == 404, (
            f"[FAIL] Unknown genre must return 404, got {resp.status_code}"
        )
        print("[PASS] GET /samples/jazz_fusion → 404 (genre not found)")

    # ------------------------------------------------------------------
    # TC-08-B: Empty directory mock — GET /samples
    # ------------------------------------------------------------------
    def test_empty_audio_directory_returns_404(self):
        """
        GIVEN  : Genre 'blues' directory exists on filesystem BUT contains no WAV files.
        WHEN   : GET /samples/blues with os.listdir mocked to return [].
        THEN   : HTTP 404 (no files to serve).

        Mock setup:
            os.listdir   patched → []  (simulate empty directory)
            os.path.isdir patched → True  (dir exists)

        This verifies the API checks file count, not just directory existence.
        """
        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST — GET /samples/blues")
        print(f"    Mock: os.listdir → []  (empty directory)")
        print(f"    Mock: os.path.isdir → True")

        with patch("os.listdir", return_value=[]), \
             patch("os.path.isdir", return_value=True):
            resp = self.client.get("/samples/blues")

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}

        print(f"  ◀ RESPONSE")
        print(f"    HTTP   : {resp.status_code}")
        print(f"    Body   : {json.dumps(body, ensure_ascii=False, default=str)}")
        print(f"{'─'*55}\n")

        assert resp.status_code == 404, (
            f"[FAIL] Empty directory must return 404, got {resp.status_code}"
        )
        print("[PASS] Empty directory mock → 404")

    # ------------------------------------------------------------------
    # TC-08-C: Empty directory mock — POST /sample-predict
    # ------------------------------------------------------------------
    def test_sample_predict_empty_dir_404(self):
        """
        GIVEN  : 'blues' directory is mocked as empty.
        WHEN   : POST /sample-predict/blues.
        THEN   : HTTP 404 (no audio file to predict from).

        Mock setup:
            os.listdir   → []
            os.path.isdir → True
        """
        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST — POST /sample-predict/blues")
        print(f"    Mock: os.listdir → []")
        print(f"    Mock: os.path.isdir → True")

        with patch("os.listdir", return_value=[]), \
             patch("os.path.isdir", return_value=True):
            resp = self.client.post("/sample-predict/blues")

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}

        print(f"  ◀ RESPONSE")
        print(f"    HTTP   : {resp.status_code}")
        print(f"    Body   : {json.dumps(body, ensure_ascii=False, default=str)}")
        print(f"{'─'*55}\n")

        assert resp.status_code == 404, (
            f"[FAIL] Empty directory must return 404, got {resp.status_code}"
        )
        print("[PASS] Empty directory mock (POST) → 404")

    # ------------------------------------------------------------------
    # TC-08-D: Non-existent genre — POST /sample-predict
    # ------------------------------------------------------------------
    def test_404_sample_predict_nonexistent_genre(self):
        """
        GIVEN  : Genre 'trap' is not a GTZAN class.
        WHEN   : POST /sample-predict/trap.
        THEN   : HTTP 404.

        Parameter fired:
            Endpoint : POST /sample-predict/trap
            Mock     : none
        """
        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST — POST /sample-predict/trap")
        print(f"    Genre: 'trap' (not in GTZAN 10 classes)")

        resp = self.client.post("/sample-predict/trap")

        print(f"  ◀ RESPONSE")
        print(f"    HTTP: {resp.status_code}")
        try:
            print(f"    Body: {json.dumps(resp.json(), ensure_ascii=False)}")
        except Exception:
            pass
        print(f"{'─'*55}\n")

        assert resp.status_code == 404, (
            f"[FAIL] 'trap' genre must return 404, got {resp.status_code}"
        )
        print("[PASS] POST /sample-predict/trap → 404")


# ---------------------------------------------------------------------------
# Utility: no-op context manager for conditional patching
# ---------------------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def _noop_ctx():
    yield None
