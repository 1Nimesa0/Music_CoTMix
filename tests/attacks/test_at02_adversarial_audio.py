"""
AT-02: Adversarial Audio Attack (Model Evasion via FGSM Perturbation)
======================================================================
Attack type : Adversarial Machine Learning — Model Evasion
Target      : MusicGenrePredictor.predict() (Neural Network Inference)
Severity    : HIGH

PHILOSOPHY — "Show the shot, show the result":
  Applies adversarial acoustic perturbation mimicking Fast Gradient Sign Method (FGSM)
  at varying epsilon strengths (0.01, 0.03, 0.08) to clean reference audio.
  Prints:
    [REQUEST]  → Base audio, perturbation method, epsilon magnitude, SNR (dB)
    [EXECUTION]→ Model inference across iterations
    [RESULT]   → Prediction shifts, confidence degradation, evasion rate audit

Run:
    pytest tests/attacks/test_at02_adversarial_audio.py -v -s
"""

import io
import json
import numpy as np
import soundfile as sf
import pytest


def _apply_fgsm_perturbation(y: np.ndarray, epsilon: float) -> np.ndarray:
    """High-frequency sign perturbation mimicking gradient direction."""
    perturbation = np.sign(
        np.sin(2 * np.pi * 3800 * np.linspace(0, 8, len(y))) + np.random.randn(len(y))
    ).astype(np.float32)
    return np.clip(y + epsilon * perturbation, -1.0, 1.0).astype(np.float32)


def _snr_db(clean: np.ndarray, perturbed: np.ndarray) -> float:
    noise = perturbed - clean
    return 10 * np.log10((np.mean(clean ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12))


def _audio_to_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


class TestAdversarialAudio:

    @pytest.mark.parametrize("epsilon", [0.01, 0.03, 0.08])
    def test_adversarial_perturbation_audit(self, predictor, real_blues_audio_bytes, epsilon):
        """
        GIVEN  : Clean reference audio track.
        WHEN   : High-frequency adversarial perturbation at epsilon={epsilon} is added.
        THEN   :
          • Model processes perturbed audio without crashing or NaN exceptions.
          • Output contains valid 'predicted_genre' and calibrated confidence in [0, 1].
          • Documents confidence impact and genre evasion behavior.

        Parameter fired:
            Base Audio : real_blues_audio_bytes (GTZAN reference track)
            Epsilon    : {epsilon} (L-infinity perturbation radius)
            Target     : MusicGenrePredictor.predict()
        """
        buf = io.BytesIO(real_blues_audio_bytes)
        y, sr = sf.read(buf)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        target_len = int(8.0 * sr)
        if len(y) > target_len:
            y = y[:target_len]

        clean_bytes = _audio_to_bytes(y, sr)
        clean_result = predictor.predict(clean_bytes)
        clean_genre = clean_result.get("predicted_genre")
        clean_conf = clean_result.get("confidence", 0.0)

        # Apply perturbation
        y_adv = _apply_fgsm_perturbation(y, epsilon=epsilon)
        adv_bytes = _audio_to_bytes(y_adv, sr)
        snr = _snr_db(y, y_adv)

        adv_result = predictor.predict(adv_bytes)
        adv_genre = adv_result.get("predicted_genre")
        adv_conf = adv_result.get("confidence", 0.0)
        status = adv_result.get("status")

        print(f"\n{'═'*65}")
        print(f"  ▶ ADVERSARIAL AUDIO AUDIT  [FGSM ε = {epsilon}]")
        print(f"    Epsilon magnitude : {epsilon}")
        print(f"    Measured SNR      : {snr:.1f} dB")
        print(f"    Clean Baseline    : {clean_genre} (conf={clean_conf:.4f})")
        print(f"    Perturbed Output  : {adv_genre} (conf={adv_conf:.4f}, status={status})")

        evasion_detected = clean_genre != adv_genre
        conf_drop = clean_conf - adv_conf
        print(f"    Evasion Observed  : {evasion_detected} (Class shifted: {clean_genre} → {adv_genre})")
        print(f"    Confidence Delta  : {conf_drop:+.4f}")
        print(f"{'═'*65}\n")

        # Invariant checks: Model must stay operational without crash
        assert "predicted_genre" in adv_result, "[FAIL] Model crashed on adversarial audio"
        assert 0.0 <= adv_conf <= 1.0, f"[FAIL] Confidence out of bounds: {adv_conf}"
        assert status in ["OK", "UNCERTAIN", "FALLBACK"], f"[FAIL] Invalid AI status: {status}"
        print(f"[PASS] ε={epsilon}: Handled cleanly, stability verified")
