"""
TC-03: AI Operational State & Confidence Gate Verification
Category: AI Test | Layer: System
Objective: Verifies the 3-tier operational decision gate (OK >= 0.80, UNCERTAIN 0.60-0.80, FALLBACK < 0.60).
"""

import pytest
import numpy as np
from unittest.mock import patch
import torch


class TestAIConfidenceGate:
    def test_decision_boundary_ok_branch(self, predictor):
        """High confidence (>= 0.80) must trigger operational state 'OK'."""
        with patch.object(predictor, "preprocess_audio") as mock_prep:
            # Mock spectrogram and dummy explainability
            dummy_tensor = torch.zeros(1, 1, 128, 256).to(predictor.device)
            dummy_explain = {"mfcc": 0.85, "delta_mfcc": 0.70, "chroma": 0.80, "spectral_centroid_hz": 2000.0}
            mock_prep.return_value = (dummy_tensor, dummy_explain)

            # Mock logits that give confidence 0.95 for index 0 ('blues')
            mock_logits = torch.tensor([[-10.0] * 10]).to(predictor.device)
            mock_logits[0, 0] = 5.0  # High probability for blues

            with patch.object(predictor.classifier, "forward", return_value=mock_logits):
                result = predictor.predict(b"dummy_audio_bytes")

            assert result["status"] == "OK", f"Expected 'OK', got {result['status']}"
            assert result["confidence"] >= 0.80
            assert result["predicted_genre"] == "Blues"

    def test_decision_boundary_uncertain_branch(self, predictor):
        """Moderate confidence (0.60 to 0.79) must trigger operational state 'UNCERTAIN'."""
        with patch.object(predictor, "preprocess_audio") as mock_prep:
            dummy_tensor = torch.zeros(1, 1, 128, 256).to(predictor.device)
            dummy_explain = {"mfcc": 0.75, "delta_mfcc": 0.65, "chroma": 0.70, "spectral_centroid_hz": 2000.0}
            mock_prep.return_value = (dummy_tensor, dummy_explain)

            # Softmax target ~ 0.70
            # Let logit_0 = ln(0.70) = -0.356, and others sum to 0.30 (each 0.30/9 = 0.0333, ln = -3.40)
            mock_logits = torch.tensor([[-3.40] * 10]).to(predictor.device)
            mock_logits[0, 5] = -0.356  # index 5 = 'jazz'

            with patch.object(predictor.classifier, "forward", return_value=mock_logits):
                result = predictor.predict(b"dummy_audio_bytes")

            assert result["status"] == "UNCERTAIN", f"Expected 'UNCERTAIN', got {result['status']}"
            assert 0.60 <= result["confidence"] < 0.80
            assert result["predicted_genre"] == "Jazz"

    def test_decision_boundary_fallback_branch(self, predictor):
        """Low confidence (< 0.60, uniform distribution) must trigger operational state 'FALLBACK'."""
        with patch.object(predictor, "preprocess_audio") as mock_prep:
            dummy_tensor = torch.zeros(1, 1, 128, 256).to(predictor.device)
            dummy_explain = {"mfcc": 0.50, "delta_mfcc": 0.50, "chroma": 0.50, "spectral_centroid_hz": 1500.0}
            mock_prep.return_value = (dummy_tensor, dummy_explain)

            # Uniform logits => confidence ~ 0.10 for each
            mock_logits = torch.zeros(1, 10).to(predictor.device)

            with patch.object(predictor.classifier, "forward", return_value=mock_logits):
                result = predictor.predict(b"dummy_audio_bytes")

            assert result["status"] == "FALLBACK", f"Expected 'FALLBACK', got {result['status']}"
            assert result["confidence"] < 0.60
