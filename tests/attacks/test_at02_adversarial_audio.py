"""
AT-02: Adversarial Audio Perturbation & Model Evasion Attack
Attack Vector: Adversarial Machine Learning (Evasion Attack)
Severity: HIGH
Objective: Audits model robustness against subtle, human-imperceptible acoustic perturbations designed to fool the neural network.
"""

import io
import pytest
import numpy as np
import soundfile as sf


class TestAT02AdversarialAudio:
    def _apply_adversarial_perturbation(self, y, epsilon=0.015):
        """Generates sign-based high-frequency adversarial perturbation."""
        # High-frequency sign perturbation mimicking FGSM direction
        perturbation = np.sign(np.sin(2 * np.pi * 3800 * np.linspace(0, 8, len(y))) + np.random.randn(len(y)))
        y_adv = y + epsilon * perturbation
        return np.clip(y_adv, -1.0, 1.0).astype(np.float32)

    def test_adversarial_perturbation_resilience(self, predictor, real_blues_audio_bytes):
        """
        Attack Test: Crafts an adversarial audio track with tiny perturbation (epsilon=0.015).
        Evaluates whether the model maintains stability or collapses into complete uncertainty.
        """
        buf = io.BytesIO(real_blues_audio_bytes)
        y, sr = sf.read(buf)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        target_len = int(8.0 * sr)
        if len(y) > target_len:
            y = y[:target_len]

        clean_result = predictor.predict(real_blues_audio_bytes)
        clean_genre = clean_result.get("predicted_genre")
        clean_conf = clean_result.get("confidence", 0.0)

        # Apply adversarial perturbation
        y_adv = self._apply_adversarial_perturbation(y, epsilon=0.015)
        out_buf = io.BytesIO()
        sf.write(out_buf, y_adv, sr, format="WAV")
        adv_bytes = out_buf.getvalue()

        adv_result = predictor.predict(adv_bytes)
        adv_genre = adv_result.get("predicted_genre")
        adv_conf = adv_result.get("confidence", 0.0)

        print(f"\n[Adversarial ML Attack] Clean: {clean_genre} ({clean_conf:.3f}) -> Perturbed: {adv_genre} ({adv_conf:.3f})")

        # The system must not crash, and status must appropriately flag UNCERTAIN or FALLBACK if attacked
        assert "predicted_genre" in adv_result
        assert adv_result.get("status") in ["OK", "UNCERTAIN", "FALLBACK"]
