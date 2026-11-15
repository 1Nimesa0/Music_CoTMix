"""
TC-03: AI Operational State & Confidence Gate Verification
===========================================================
Type    : AI Test (Mock-based State Boundary Test)
Layer   : System
Target  : MusicGenrePredictor.predict() & Decision Gate Logic

PHILOSOPHY — "Show the shot, show the result":
  Injects controlled logit vectors into the classifier head to simulate exact
  confidence conditions and verify the 3-tier operational gate:
    • OK        : Confidence >= 0.80
    • UNCERTAIN : 0.60 <= Confidence < 0.80
    • FALLBACK  : Confidence < 0.60

Mock setup:
    • Preprocessing mock: returns fixed tensor [1, 1, 128, 256] + dummy XAI features
    • Classifier forward pass mock: returns calibrated logits tensor

Run:
    pytest tests/test_03_ai_confidence_gate.py -v -s
"""

import json
from unittest.mock import patch
import torch
import pytest


class TestAIConfidenceGate:

    # ------------------------------------------------------------------
    # TC-03-A: Decision boundary OK branch (High confidence >= 0.80)
    # ------------------------------------------------------------------
    def test_decision_boundary_ok_branch(self, predictor):
        """
        GIVEN  : Model classifier produces logits with dominant probability for Blues (~0.95).
        WHEN   : predictor.predict() runs inference.
        THEN   : Status is 'OK', confidence >= 0.80, predicted_genre is 'Blues'.

        Parameter fired (Mocked):
            Mock Logits : [-10.0, ...] with logit[0] = 5.0 (Softmax ~ 0.95 on Blues)
            Expected    : status == 'OK', confidence >= 0.80
        """
        print(f"\n{'═'*65}")
        print(f"  ▶ INJECTION SETUP  [OK BRANCH — High Confidence]")
        print(f"    Target Class : Blues (index 0)")
        print(f"    Injected Logits: logit[0]=5.0, others=-10.0 → Softmax probability ~ 0.95")
        print(f"    Expected Gate: Status 'OK' (Threshold >= 0.80)")

        dummy_tensor = torch.zeros(1, 1, 128, 256).to(predictor.device)
        dummy_explain = {
            "mfcc": 0.85, "delta_mfcc": 0.70, "chroma": 0.80, "spectral_centroid_hz": 2000.0
        }

        mock_logits = torch.tensor([[-10.0] * 10]).to(predictor.device)
        mock_logits[0, 0] = 5.0  # High probability for Blues

        with patch.object(predictor, "preprocess_audio", return_value=(dummy_tensor, dummy_explain)), \
             patch.object(predictor.classifier, "forward", return_value=mock_logits):
            result = predictor.predict(b"dummy_audio_bytes")

        print(f"\n  ◀ EXECUTION RESULT")
        print(f"    status          : {result['status']}")
        print(f"    predicted_genre : {result['predicted_genre']}")
        print(f"    confidence      : {result['confidence']:.4f}")
        print(f"    Full Payload    :\n{json.dumps(result, ensure_ascii=False, indent=4, default=str)}")
        print(f"{'═'*65}\n")

        assert result["status"] == "OK", f"Expected 'OK', got {result['status']}"
        assert result["confidence"] >= 0.80, f"Expected >= 0.80, got {result['confidence']}"
        assert result["predicted_genre"] == "Blues"
        print(f"[PASS] Injected 0.95 → status='OK', conf={result['confidence']:.4f}")

    # ------------------------------------------------------------------
    # TC-03-B: Decision boundary UNCERTAIN branch (0.60 <= conf < 0.80)
    # ------------------------------------------------------------------
    def test_decision_boundary_uncertain_branch(self, predictor):
        """
        GIVEN  : Model classifier produces logits with moderate probability for Jazz (~0.70).
        WHEN   : predictor.predict() runs inference.
        THEN   : Status is 'UNCERTAIN', confidence in [0.60, 0.80), predicted_genre is 'Jazz'.

        Parameter fired (Mocked):
            Mock Logits : logit[5] = -0.356, others = -3.40 → Softmax ~ 0.70 on Jazz
            Expected    : status == 'UNCERTAIN'
        """
        print(f"\n{'═'*65}")
        print(f"  ▶ INJECTION SETUP  [UNCERTAIN BRANCH — Mid Confidence]")
        print(f"    Target Class : Jazz (index 5)")
        print(f"    Injected Logits: logit[5]=-0.356, others=-3.40 → Softmax probability ~ 0.70")
        print(f"    Expected Gate: Status 'UNCERTAIN' (0.60 <= Conf < 0.80)")

        dummy_tensor = torch.zeros(1, 1, 128, 256).to(predictor.device)
        dummy_explain = {
            "mfcc": 0.75, "delta_mfcc": 0.65, "chroma": 0.70, "spectral_centroid_hz": 2000.0
        }

        mock_logits = torch.tensor([[-3.40] * 10]).to(predictor.device)
        mock_logits[0, 5] = -0.356  # index 5 = 'Jazz'

        with patch.object(predictor, "preprocess_audio", return_value=(dummy_tensor, dummy_explain)), \
             patch.object(predictor.classifier, "forward", return_value=mock_logits):
            result = predictor.predict(b"dummy_audio_bytes")

        print(f"\n  ◀ EXECUTION RESULT")
        print(f"    status          : {result['status']}")
        print(f"    predicted_genre : {result['predicted_genre']}")
        print(f"    confidence      : {result['confidence']:.4f}")
        print(f"{'═'*65}\n")

        assert result["status"] == "UNCERTAIN", f"Expected 'UNCERTAIN', got {result['status']}"
        assert 0.60 <= result["confidence"] < 0.80, f"Confidence {result['confidence']} not in [0.60, 0.80)"
        assert result["predicted_genre"] == "Jazz"
        print(f"[PASS] Injected 0.70 → status='UNCERTAIN', conf={result['confidence']:.4f}")

    # ------------------------------------------------------------------
    # TC-03-C: Decision boundary FALLBACK branch (Low confidence < 0.60)
    # ------------------------------------------------------------------
    def test_decision_boundary_fallback_branch(self, predictor):
        """
        GIVEN  : Model classifier produces uniform logits (confidence ~ 0.10).
        WHEN   : predictor.predict() runs inference.
        THEN   : Status is 'FALLBACK', confidence < 0.60.

        Parameter fired (Mocked):
            Mock Logits : zeros tensor [1, 10] → Softmax = 0.10 for all 10 classes
            Expected    : status == 'FALLBACK'
        """
        print(f"\n{'═'*65}")
        print(f"  ▶ INJECTION SETUP  [FALLBACK BRANCH — Uniform Low Confidence]")
        print(f"    Injected Logits: All zeros [0, 0, ..., 0] → Softmax probability = 0.10 across 10 classes")
        print(f"    Expected Gate: Status 'FALLBACK' (Threshold < 0.60)")

        dummy_tensor = torch.zeros(1, 1, 128, 256).to(predictor.device)
        dummy_explain = {
            "mfcc": 0.50, "delta_mfcc": 0.50, "chroma": 0.50, "spectral_centroid_hz": 1500.0
        }

        mock_logits = torch.zeros(1, 10).to(predictor.device)

        with patch.object(predictor, "preprocess_audio", return_value=(dummy_tensor, dummy_explain)), \
             patch.object(predictor.classifier, "forward", return_value=mock_logits):
            result = predictor.predict(b"dummy_audio_bytes")

        print(f"\n  ◀ EXECUTION RESULT")
        print(f"    status          : {result['status']}")
        print(f"    predicted_genre : {result['predicted_genre']}")
        print(f"    confidence      : {result['confidence']:.4f}")
        print(f"{'═'*65}\n")

        assert result["status"] == "FALLBACK", f"Expected 'FALLBACK', got {result['status']}"
        assert result["confidence"] < 0.60, f"Expected < 0.60, got {result['confidence']}"
        print(f"[PASS] Injected uniform 0.10 → status='FALLBACK', conf={result['confidence']:.4f}")

    # ------------------------------------------------------------------
    # TC-03-D: Boundary condition at exact 0.80 threshold
    # ------------------------------------------------------------------
    def test_boundary_at_ok_threshold(self, predictor):
        """
        GIVEN  : Model output with probability exactly 0.80 (boundary inclusive).
        WHEN   : predictor.predict() runs inference.
        THEN   : Status is 'OK'.
        """
        dummy_tensor = torch.zeros(1, 1, 128, 256).to(predictor.device)
        dummy_explain = {"mfcc": 0.5, "delta_mfcc": 0.5, "chroma": 0.5, "spectral_centroid_hz": 1500.0}

        # target p = 0.80 for class 0, and (0.20/9) = 0.0222 for other 9
        # ln(0.80) = -0.2231, ln(0.0222) = -3.8077
        mock_logits = torch.tensor([[-3.8077] * 10]).to(predictor.device)
        mock_logits[0, 0] = -0.2231

        with patch.object(predictor, "preprocess_audio", return_value=(dummy_tensor, dummy_explain)), \
             patch.object(predictor.classifier, "forward", return_value=mock_logits):
            result = predictor.predict(b"dummy_audio_bytes")

        print(f"\n  ◀ BOUNDARY 0.80 RESULT: status='{result['status']}', conf={result['confidence']:.4f}")
        assert result["status"] == "OK"
        print(f"[PASS] Boundary at 0.80 evaluated as 'OK'")
