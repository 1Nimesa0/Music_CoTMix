"""
AT-06: Audio Steganography & Covert Exfiltration Channel Audit
Attack Vector: LSB (Least Significant Bit) Audio Steganography
Severity: MEDIUM
Objective: Evaluates whether arbitrary covert binary payloads can be embedded in audio and passed through the model unnoticed.
"""

import io
import pytest
import numpy as np
import soundfile as sf


class TestAT06AudioSteganographyAudit:
    def _embed_lsb_secret(self, y, secret_text="EXFIL_CONFIDENTIAL_KEY_9999"):
        """Embeds ASCII text into the LSB bits of audio samples."""
        y_int = (y * 32767).astype(np.int16)
        binary_str = "".join(f"{ord(c):08b}" for c in secret_text)

        for i, bit in enumerate(binary_str):
            if i >= len(y_int):
                break
            y_int[i] = (y_int[i] & ~1) | int(bit)

        return (y_int / 32767.0).astype(np.float32)

    def test_steganographic_payload_stealth_evaluation(self, predictor, sample_sine_wav_bytes):
        """
        Attack Audit: Evaluates if an audio track carrying a hidden covert bitstream
        is accepted by the inference pipeline without triggering anomaly flags.
        """
        buf = io.BytesIO(sample_sine_wav_bytes)
        y, sr = sf.read(buf)

        # Embed covert secret payload
        y_stego = self._embed_lsb_secret(y, "EXFIL:CREDENTIAL_TOKEN_XYZ123")

        out_buf = io.BytesIO()
        sf.write(out_buf, y_stego, sr, format="WAV")
        stego_bytes = out_buf.getvalue()

        result = predictor.predict(stego_bytes)
        assert "predicted_genre" in result
        print(f"\n[Steganography Audit] Hidden payload audio successfully passed model! Genre: {result['predicted_genre']} (Conf: {result['confidence']:.3f})")
        # Finding: Standard ML inference does not cleanse LSB channels. Audio re-quantization is recommended.
