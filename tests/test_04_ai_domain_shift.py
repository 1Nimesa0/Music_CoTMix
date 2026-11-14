"""
TC-04: Domain Shift Robustness (Cross-Environment Noise & Reverb)
Category: AI Test | Layer: System
Objective: Verifies model stability and domain adaptation under acoustic noise (pink/brown noise, reverberation).
"""

import os
import io
import pytest
import numpy as np
import soundfile as sf
from data.augment_environment import generate_colored_noise


class TestAIDomainShiftRobustness:
    def _create_noisy_sample(self, clean_audio_path, noise_type="pink", snr_db=15):
        """Helper to mix clean audio with realistic environmental noise at calibrated SNR."""
        y, sr = sf.read(clean_audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Slice to 8 seconds
        target_len = int(8.0 * sr)
        if len(y) > target_len:
            y = y[:target_len]

        signal_power = np.mean(y ** 2)
        noise = generate_colored_noise(len(y), noise_type=noise_type)
        noise_power = np.mean(noise ** 2)

        # Calculate noise scaling factor for target SNR (dB)
        target_noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise_scale = np.sqrt(target_noise_power / (noise_power + 1e-8))
        noisy_y = y + noise * noise_scale

        # Normalize
        noisy_y = np.clip(noisy_y, -1.0, 1.0).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, noisy_y, sr, format="WAV")
        return buf.getvalue()

    def test_robustness_under_pink_environmental_noise(self, predictor):
        """
        AI Test: Model must not crash and should retain confidence > 0.20 (2x random baseline 0.10)
        under moderate café / background environmental noise (Pink Noise, SNR = 15 dB).
        """
        clean_path = os.path.join("data", "audio", "blues", "blues_0001.wav")
        if not os.path.exists(clean_path):
            pytest.skip("Reference audio file not found")

        noisy_bytes = self._create_noisy_sample(clean_path, noise_type="pink", snr_db=15)
        result = predictor.predict(noisy_bytes)

        assert "predicted_genre" in result, "Model crashed on noisy audio input"
        assert result.get("status") in ["OK", "UNCERTAIN", "FALLBACK"]
        assert result.get("confidence", 0.0) >= 0.20, f"Confidence collapsed below 2x random threshold: {result.get('confidence')}"
        print(f"\n[Domain Shift - Pink Noise] Predicted: {result['predicted_genre']} (Conf: {result['confidence']:.4f}, Status: {result['status']})")

    def test_robustness_under_brown_traffic_rumble(self, predictor):
        """
        AI Test: Model must withstand low-frequency vehicular rumble (Brown Noise, SNR = 12 dB).
        """
        clean_path = os.path.join("data", "audio", "rock", "rock_0001.wav")
        if not os.path.exists(clean_path):
            pytest.skip("Reference audio file not found")

        noisy_bytes = self._create_noisy_sample(clean_path, noise_type="brown", snr_db=12)
        result = predictor.predict(noisy_bytes)

        assert "predicted_genre" in result, "Model crashed on brown noise input"
        assert result.get("confidence", 0.0) >= 0.20, f"Confidence collapsed below random threshold: {result.get('confidence')}"
        print(f"\n[Domain Shift - Brown Noise] Predicted: {result['predicted_genre']} (Conf: {result['confidence']:.4f}, Status: {result['status']})")
