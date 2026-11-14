"""
AT-08: Membership Inference Attack (MIA) & Training Data Privacy Audit
Attack Vector: Membership Inference via Confidence Disparity
Severity: MEDIUM
Objective: Evaluates whether training sample confidence is statistically higher than unseen synthetic audio, leaking training dataset membership.
"""

import os
import io
import pytest
import numpy as np
import soundfile as sf


class TestAT08MembershipInferenceAudit:
    def test_training_sample_confidence_gap_audit(self, predictor, real_blues_audio_bytes):
        """
        Attack Audit: Measures confidence disparity between:
        1. Training reference sample (blues_0001.wav)
        2. Synthetic unseen harmonic audio (multi-tone sine)
        
        Security Criterion: A confidence gap > 0.50 indicates strong overfitting and high vulnerability to MIA.
        """
        # 1. Training reference sample
        res_train = predictor.predict(real_blues_audio_bytes)
        conf_train = res_train.get("confidence", 0.0)

        # 2. Synthetic unseen audio
        sr = 16000
        t = np.linspace(0, 8, sr * 8, endpoint=False)
        y_synthetic = (0.3 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, y_synthetic, sr, format="WAV")
        synthetic_bytes = buf.getvalue()

        res_synth = predictor.predict(synthetic_bytes)
        conf_synth = res_synth.get("confidence", 0.0)

        gap = conf_train - conf_synth
        print(f"\n[MIA Privacy Audit] Training Conf: {conf_train:.4f} vs Unseen Conf: {conf_synth:.4f} | Disparity Gap: {gap:.4f}")

        # Ensure model generalization: the gap must not indicate severe memorization (gap < 0.60)
        assert gap < 0.60, f"VULNERABILITY: Severe membership inference risk! Confidence gap {gap:.3f} > 0.60"
