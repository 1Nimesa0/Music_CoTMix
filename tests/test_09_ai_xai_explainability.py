"""
TC-09: AI Acoustic Explainability (XAI) Feature Verification
Category: AI Test | Layer: System
Objective: Verifies accuracy, physical range constraints, and discriminative capacity of XAI features (MFCC, Chroma, Spectral Centroid).
"""

import os
import pytest


class TestAIEplainabilityFeatures:
    def test_xai_payload_structure(self, predictor, sample_sine_wav_bytes):
        """Prediction payload must include explainability block with key acoustic signatures."""
        result = predictor.predict(sample_sine_wav_bytes)
        assert "explainability" in result, "Missing 'explainability' block in response"
        xai = result["explainability"]

        assert "features" in xai, "Missing 'features' in explainability block"
        features = xai["features"]
        feature_names = {f["name"] for f in features}

        assert "MFCC" in feature_names, "Missing MFCC in explainability features"
        assert "Delta MFCC" in feature_names, "Missing Delta MFCC in explainability features"
        assert "Chroma" in feature_names, "Missing Chroma in explainability features"

    def test_xai_feature_values_within_unit_bounds(self, predictor, sample_sine_wav_bytes):
        """All normalized explainability indicator values must be strictly in [0.0, 1.0]."""
        result = predictor.predict(sample_sine_wav_bytes)
        features = result["explainability"]["features"]

        for item in features:
            val = item.get("value")
            assert isinstance(val, (int, float)), f"Value of {item['name']} is not numeric: {val}"
            assert 0.0 <= val <= 1.0, f"{item['name']} out of bounds [0.0, 1.0]: {val}"

    def test_spectral_centroid_distinguishes_classical_from_metal(self, predictor):
        """
        AI Test: Fast XAI spectral centroid must clearly separate acoustic classical music
        (lower centroid, ~550 Hz) from heavy metal (high distortion treble, ~1500 Hz).
        """
        classical_path = os.path.join("data", "audio", "classical", "classical_0001.wav")
        metal_path = os.path.join("data", "audio", "metal", "metal_0001.wav")

        if not (os.path.exists(classical_path) and os.path.exists(metal_path)):
            pytest.skip("Audio reference tracks not found")

        _, xai_classical = predictor.preprocess_audio(classical_path)
        _, xai_metal = predictor.preprocess_audio(metal_path)

        centroid_classical = xai_classical["spectral_centroid_hz"]
        centroid_metal = xai_metal["spectral_centroid_hz"]

        print(f"\n[XAI Centroid Verification] Classical: {centroid_classical} Hz vs Metal: {centroid_metal} Hz")
        assert centroid_metal > centroid_classical + 300.0, (
            f"Spectral centroid failed to distinguish metal from classical: {centroid_metal} Hz vs {centroid_classical} Hz"
        )
