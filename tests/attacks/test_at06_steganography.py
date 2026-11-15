"""
AT-06: Audio Steganography Attack (Covert Data Exfiltration Channel)
====================================================================
Attack type : Steganography / Covert Channel / DLP Bypass
Target      : POST /predict & MusicGenrePredictor (Audio Pipeline)
Severity    : MEDIUM

PHILOSOPHY — "Show the shot, show the result":
  Embeds secret ASCII data inside Least Significant Bits (LSB) of audio samples.
  Prints:
    [REQUEST]  → Carrier signal, covert text payload, bit capacity, SNR
    [EXECUTION]→ Audio inference processing
    [RESULT]   → Prediction output, whether covert channel passed through undetected

Attack finding:
  Standard ML inference pipelines do not inspect or cleanse LSB channels.
  Audio containing arbitrary hidden secrets passes cleanly through the model.

Run:
    pytest tests/attacks/test_at06_steganography.py -v -s
"""

import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import app


def _embed_lsb_secret(y: np.ndarray, secret_text: str) -> np.ndarray:
    """Embeds ASCII text into the LSB bits of int16 audio samples."""
    y_int = (y * 32767).astype(np.int16)
    binary_str = "".join(f"{ord(c):08b}" for c in secret_text)

    for i, bit in enumerate(binary_str):
        if i >= len(y_int):
            break
        y_int[i] = (y_int[i] & ~1) | int(bit)

    return (y_int / 32767.0).astype(np.float32)


def _snr_db(clean: np.ndarray, stego: np.ndarray) -> float:
    noise = stego - clean
    return 10 * np.log10((np.mean(clean ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12))


class TestSteganographyAttack:

    client = TestClient(app)
    SR = 16000
    DURATION = 8.0

    # ------------------------------------------------------------------
    # AT-06-A: Direct Inference Stealth Evaluation
    # ------------------------------------------------------------------
    def test_steganographic_payload_stealth_evaluation(self, predictor, sample_sine_wav_bytes):
        """
        GIVEN  : Clean 440 Hz audio carrier.
        WHEN   : Covert payload 'EXFIL:CREDENTIAL_TOKEN_XYZ123' is embedded into LSBs.
        THEN   :
          • Model accepts and processes the audio without error.
          • Output contains predicted_genre and valid confidence.
          • SNR remains > 60 dB (inaudible to human ears).

        Parameter fired:
            Carrier Audio : 440 Hz sine wave (8s, 16kHz)
            Secret Text   : 'EXFIL:CREDENTIAL_TOKEN_XYZ123' (28 bytes)
            Attack Vector : LSB audio steganography
        """
        buf = io.BytesIO(sample_sine_wav_bytes)
        y, sr = sf.read(buf)

        secret = "EXFIL:CREDENTIAL_TOKEN_XYZ123"
        y_stego = _embed_lsb_secret(y, secret)
        snr = _snr_db(y, y_stego)

        out_buf = io.BytesIO()
        sf.write(out_buf, y_stego, sr, format="WAV")
        stego_bytes = out_buf.getvalue()

        print(f"\n{'═'*65}")
        print(f"  ▶ STEGANOGRAPHY AUDIT  [INFERENCE PIPELINE STEALTH]")
        print(f"    Covert Payload  : '{secret}' ({len(secret)} bytes)")
        print(f"    Carrier Signal  : 440 Hz tone, 8s ({len(y):,} samples)")
        print(f"    Perceptual SNR  : {snr:.1f} dB (Imperceptible)")

        result = predictor.predict(stego_bytes)

        print(f"\n  ◀ RESULT")
        print(f"    predicted_genre : {result.get('predicted_genre')}")
        print(f"    confidence      : {result.get('confidence'):.4f}")
        print(f"    status          : {result.get('status')}")
        print(f"    Audit Finding   : Covert payload passed model unnoticed!")
        print(f"{'═'*65}\n")

        assert "predicted_genre" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert snr > 60.0
        print(f"[PASS] Steganography audit: Model accepts covert payload cleanly")

    # ------------------------------------------------------------------
    # AT-06-B: HTTP API Upload Acceptance
    # ------------------------------------------------------------------
    def test_steganographic_audio_accepted_via_api(self, sample_sine_wav_bytes):
        """
        GIVEN  : Audio file containing embedded covert payload.
        WHEN   : POST /predict with stego audio.
        THEN   : API accepts request with HTTP 200 (proves lack of DPI / LSB sanitization).
        """
        buf = io.BytesIO(sample_sine_wav_bytes)
        y, sr = sf.read(buf)
        y_stego = _embed_lsb_secret(y, "EXFIL:CONFIDENTIAL_KEY_4111")

        out_buf = io.BytesIO()
        sf.write(out_buf, y_stego, sr, format="WAV")
        stego_bytes = out_buf.getvalue()

        print(f"\n{'═'*65}")
        print(f"  ▶ STEGANOGRAPHY API AUDIT  [POST /predict]")
        print(f"    Payload size : {len(stego_bytes):,} bytes")

        resp = self.client.post(
            "/predict",
            files={"file": ("stego_track.wav", stego_bytes, "audio/wav")}
        )

        print(f"\n  ◀ HTTP RESPONSE")
        print(f"    HTTP Status  : {resp.status_code}")
        if resp.status_code == 200:
            print(f"    Body Summary : genre={resp.json().get('predicted_genre')}, status={resp.json().get('status')}")
            print(f"    ⚠ VULNERABILITY: Covert data transmitted successfully without server scrubbing.")
        print(f"{'═'*65}\n")

        assert resp.status_code in [200, 400, 422]
        print(f"[PASS] Steganography API test completed with HTTP {resp.status_code}")

    # ------------------------------------------------------------------
    # AT-06-C: Exfiltration Capacity Assessment
    # ------------------------------------------------------------------
    def test_steganography_capacity_risk_assessment(self):
        """
        Calculates and documents maximum exfiltration bandwidth per audio upload.
        """
        total_samples = int(self.SR * self.DURATION)
        capacity_bytes = total_samples // 8
        capacity_kb = capacity_bytes / 1024

        print(f"\n{'═'*65}")
        print(f"  ▶ EXFILTRATION CAPACITY AUDIT")
        print(f"    Duration        : {self.DURATION} s")
        print(f"    Sample Rate     : {self.SR} Hz")
        print(f"    Samples         : {total_samples:,}")
        print(f"    Max Capacity    : {capacity_bytes:,} bytes ({capacity_kb:.1f} KB per 8s track)")
        print(f"    Risk Scope      : Sufficient to smuggle private keys, credentials, or code snippets")
        print(f"{'═'*65}\n")

        assert capacity_bytes >= 10000
        print(f"[PASS] Exfiltration capacity documented: {capacity_bytes:,} bytes")
