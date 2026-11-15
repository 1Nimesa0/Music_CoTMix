"""
TC-06: Input Validation — File Format & Edge-Case Handling
==========================================================
Type    : Automation Test
Layer   : API
Endpoint: POST /predict · GET /samples/{genre}

PHILOSOPHY — "Show the shot, show the result":
  Sends malformed, forbidden, or edge-case inputs and verifies that the API
  rejects them gracefully (no 500 crash, correct error codes/messages).

Expected behaviors:
    .exe / .pdf disguised → HTTP 400
    empty file            → HTTP 400 / 422 / 500 (not 200)
    silent audio          → HTTP 200 with status=FALLBACK (not random genre)
    valid extensions      → HTTP 200 or 500 (not 400 — extension passes)
    unknown genre         → HTTP 404

Run:
    pytest tests/test_06_automation_input_validation.py -v -s
"""

import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import app


class TestInputValidation:

    client = TestClient(app)

    def _post_file(self, filename: str, content: bytes, content_type: str = "audio/wav"):
        """Helper: POST /predict with explicit filename, bytes, mime-type."""
        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST — POST /predict")
        print(f"    filename     : {filename!r}")
        print(f"    content_type : {content_type}")
        print(f"    size         : {len(content):,} bytes")
        print(f"    first 16 hex : {content[:16].hex()}")

        resp = self.client.post(
            "/predict",
            files={"file": (filename, content, content_type)},
        )

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:300]}

        print(f"  ◀ RESPONSE")
        print(f"    HTTP status  : {resp.status_code}")
        print(f"    Body         : {json.dumps(body, ensure_ascii=False, default=str)}")
        print(f"{'─'*55}\n")
        return resp

    # ------------------------------------------------------------------
    # TC-06-A: Reject .exe binary
    # ------------------------------------------------------------------
    def test_reject_invalid_extension_exe(self):
        """
        GIVEN  : Binary .exe file (MZ header).
        WHEN   : POST /predict with filename='malware.exe'.
        THEN   : HTTP 400 — extension blocked before inference.

        Parameter fired:
            filename     : malware.exe
            content      : b'MZ\x90\x00binary_payload' (PE header)
            content_type : application/octet-stream
        """
        resp = self._post_file(
            "malware.exe",
            b"MZ\x90\x00" + b"\x00binary_payload" * 20,
            "application/octet-stream"
        )
        assert resp.status_code == 400, (
            f"[FAIL] .exe must be rejected with HTTP 400, got {resp.status_code}"
        )
        print("[PASS] .exe extension rejected with HTTP 400")

    # ------------------------------------------------------------------
    # TC-06-B: Reject empty file
    # ------------------------------------------------------------------
    def test_reject_empty_file(self):
        """
        GIVEN  : Near-empty 3-byte payload (no valid WAV data).
        WHEN   : POST /predict with filename='empty.wav'.
        THEN   : HTTP 400 / 422 / 500 (not 200 — cannot predict from no data).

        Parameter fired:
            filename : empty.wav
            content  : b'\\x00\\x01\\x02' (3 bytes — invalid WAV)
        """
        resp = self._post_file("empty.wav", b"\x00\x01\x02", "audio/wav")
        assert resp.status_code in [400, 422, 500], (
            f"[FAIL] Empty file must fail (not 200), got {resp.status_code}"
        )
        print(f"[PASS] Empty file returned HTTP {resp.status_code} (not 200)")

    # ------------------------------------------------------------------
    # TC-06-C: PDF disguised as WAV
    # ------------------------------------------------------------------
    def test_reject_pdf_disguised_as_wav(self):
        """
        GIVEN  : PDF bytes renamed to .wav.
        WHEN   : POST /predict with filename='song.wav', PDF magic bytes.
        THEN   : Server handles gracefully — either rejects with HTTP 4xx/5xx OR
                 returns HTTP 200 with status='FALLBACK' and audio preprocessing error.
                 Must NEVER return a successful confident genre prediction.

        Parameter fired:
            filename     : song.wav
            content      : %PDF-1.4 header + padding (256 bytes)
            content_type : audio/wav
        Security goal: Content-type spoofing must not bypass validation or cause crash.
        """
        fake_pdf = b"%PDF-1.4 1 0 obj fake pdf content" + b"A" * 256
        resp = self._post_file("song.wav", fake_pdf, "audio/wav")

        if resp.status_code == 200:
            data = resp.json()
            assert data.get("status") == "FALLBACK", (
                f"[FAIL] Server made prediction on PDF bytes! Status: {data.get('status')}"
            )
            assert "error" in data or "preprocessing" in str(data).lower(), (
                f"[FAIL] Expected error detail in response body: {data}"
            )
            print(f"[PASS] PDF-as-WAV safely handled: HTTP 200 with status='FALLBACK' (error: {data.get('error')})")
        else:
            assert resp.status_code in [400, 422, 500], (
                f"[FAIL] Unexpected status code {resp.status_code} for PDF payload"
            )
            print(f"[PASS] PDF-as-WAV blocked at HTTP level: {resp.status_code}")

    # ------------------------------------------------------------------
    # TC-06-D: Accept valid extensions
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("ext", [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"])
    def test_accept_all_valid_extensions(self, ext):
        """
        GIVEN  : Valid extension but content is silent WAV bytes.
        WHEN   : POST /predict with filename='song{ext}'.
        THEN   : HTTP status is NOT 400 (extension passes validation layer).
                 May return 200 (FALLBACK) or 500 (audio processing error).

        Parameter fired:
            filename     : song{ext}
            content      : silent WAV (all zeros, 8s, 16kHz)
            content_type : audio/wav
        Note: We only test that extension is NOT blocked (HTTP 400 = fail).
        """
        silent = np.zeros(16000 * 8, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silent, 16000, format="WAV")
        wav_bytes = buf.getvalue()

        resp = self._post_file(f"song{ext}", wav_bytes, "audio/wav")
        assert resp.status_code != 400, (
            f"[FAIL] Extension {ext} incorrectly rejected with HTTP 400"
        )
        print(f"[PASS] Extension {ext} → HTTP {resp.status_code} (not 400)")

    # ------------------------------------------------------------------
    # TC-06-E: Silent audio returns FALLBACK (not random genre)
    # ------------------------------------------------------------------
    def test_silent_audio_returns_fallback(self):
        """
        GIVEN  : All-zeros WAV (8s, 16kHz) — completely silent audio.
        WHEN   : POST /predict.
        THEN   : Response status == 'FALLBACK' with an error message,
                 OR HTTP error (400/422/500).
                 Must NOT return status='OK'/'UNCERTAIN' with a confident genre.

        Parameter fired:
            filename : silent.wav
            content  : np.zeros(128000) → WAV bytes (~512 KB)
            Expected : status='FALLBACK' + error key OR HTTP 4xx/5xx
        """
        silent = np.zeros(16000 * 8, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silent, 16000, format="WAV")
        wav_bytes = buf.getvalue()

        resp = self._post_file("silent.wav", wav_bytes, "audio/wav")

        if resp.status_code == 200:
            data = resp.json()
            # Silent audio must land in FALLBACK, never OK/UNCERTAIN
            status = data.get("status", "")
            assert status == "FALLBACK", (
                f"[FAIL] Silent audio returned status='{status}' (expected FALLBACK). "
                f"Confidence={data.get('confidence')}"
            )
            print(f"[PASS] Silent audio → HTTP 200 status=FALLBACK "
                  f"(error: {data.get('error', 'N/A')})")
        else:
            assert resp.status_code in [400, 422, 500], (
                f"[FAIL] Unexpected status {resp.status_code} for silent audio"
            )
            print(f"[PASS] Silent audio → HTTP {resp.status_code} (rejected)")

    # ------------------------------------------------------------------
    # TC-06-F: Unknown genre in /samples/{genre}
    # ------------------------------------------------------------------
    def test_invalid_genre_in_sample_endpoint(self):
        """
        GIVEN  : Genre 'jazz_fusion' does not exist in the system.
        WHEN   : GET /samples/jazz_fusion.
        THEN   : HTTP 404 with 'detail' field.

        Parameter fired:
            Endpoint : GET /samples/jazz_fusion
            Expected : HTTP 404
        """
        print(f"\n{'─'*55}")
        print(f"  ▶ REQUEST — GET /samples/jazz_fusion")
        print(f"    Genre: 'jazz_fusion' (does not exist)")

        resp = self.client.get("/samples/jazz_fusion")

        print(f"  ◀ RESPONSE")
        print(f"    HTTP: {resp.status_code}")
        try:
            print(f"    Body: {json.dumps(resp.json(), ensure_ascii=False)}")
        except Exception:
            pass
        print(f"{'─'*55}\n")

        assert resp.status_code == 404, (
            f"[FAIL] Unknown genre must return 404, got {resp.status_code}"
        )
        print("[PASS] Unknown genre → HTTP 404")
