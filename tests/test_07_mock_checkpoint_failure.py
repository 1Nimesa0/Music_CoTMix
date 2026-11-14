"""
TC-07: Mock Checkpoint Failure & Graceful Degradation
Category: Mock Test | Layer: System
Objective: Verifies fault tolerance when model checkpoint is missing or corrupted.
"""

import os
import tempfile
import pytest
from unittest.mock import patch
from app.service import MusicGenrePredictor


class TestMockCheckpointFaultTolerance:
    def test_missing_checkpoint_graceful_startup(self):
        """
        Mock Test: When checkpoint file is missing, the service must:
        1. Not raise unhandled exception during instantiation.
        2. Fall back to initialized random weights.
        3. Keep feature_extractor and classifier operational.
        """
        non_existent_path = "models/checkpoints/definitely_missing_checkpoint_99999.pt"
        assert not os.path.exists(non_existent_path)

        predictor = MusicGenrePredictor(checkpoint_path=non_existent_path)
        assert predictor is not None
        assert predictor.feature_extractor is not None
        assert predictor.classifier is not None
        assert predictor.model_version is not None

    def test_corrupted_checkpoint_recovery(self):
        """
        Mock Test: When checkpoint file exists but contains corrupted garbage bytes,
        service must catch exception and fall back to initialized weights.
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"CORRUPTED_BINARY_GARBAGE_DATA_12345" * 200)
            corrupt_path = f.name

        try:
            predictor = MusicGenrePredictor(checkpoint_path=corrupt_path)
            assert predictor is not None
            assert predictor.feature_extractor is not None
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    def test_inference_with_fallback_weights_returns_valid_structure(self, sample_sine_wav_bytes):
        """
        Mock Test: Predictor running with uninitialized/fallback weights must still
        produce a complete, schema-compliant JSON response without crashing.
        """
        with patch.object(MusicGenrePredictor, "_load_weights", return_value=None):
            predictor = MusicGenrePredictor(checkpoint_path="dummy_path.pt")
            result = predictor.predict(sample_sine_wav_bytes)

        required_keys = [
            "status", "predicted_genre", "confidence",
            "top_predictions", "all_predictions", "model_version",
            "latency_ms", "explainability"
        ]
        for key in required_keys:
            assert key in result, f"Missing required response key '{key}' during fallback operation"

        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["all_predictions"]) == 10
