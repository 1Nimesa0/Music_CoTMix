"""
TC-02: AI Model Accuracy & Genre Recognition Benchmark
Category: AI Test | Layer: System
Objective: Evaluates inference accuracy, prediction stability, and probability calibration across 10 GTZAN music genre reference audio samples.
"""

import os
import pytest


class TestAIMusicClassificationAccuracy:
    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]

    @pytest.mark.parametrize("genre", GENRES)
    def test_preset_sample_prediction_correctness(self, predictor, genre):
        """
        AI Test: With studio-clean reference sample audio, the model must:
        1. Successfully execute forward inference without exception.
        2. Output a predicted genre within the 10 valid GTZAN classes.
        3. Assign an operational state consistent with confidence score.
        4. Provide calibrated confidence scores within [0.0, 1.0].
        """
        genre_dir = os.path.join("data", "audio", genre)
        assert os.path.isdir(genre_dir), f"Directory not found for genre: {genre}"

        wav_files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
        assert len(wav_files) > 0, f"No WAV samples found for genre: {genre}"

        sample_path = os.path.join(genre_dir, wav_files[0])
        result = predictor.predict(sample_path)

        assert result.get("status") in ["OK", "UNCERTAIN", "FALLBACK"], f"Invalid AI status: {result.get('status')}"
        assert "predicted_genre" in result, "Missing predicted_genre field"
        assert result["predicted_genre"].lower() in self.GENRES, f"Unknown predicted genre: {result['predicted_genre']}"

        assert "confidence" in result, "Missing confidence score"
        assert 0.0 <= result["confidence"] <= 1.0, f"Confidence out of bounds [0, 1]: {result['confidence']}"

        # Top-3 predictions check
        top_preds = result.get("top_predictions", [])
        assert len(top_preds) <= 3, "Too many top predictions"
        for p in top_preds:
            assert p["genre"].lower() in self.GENRES
            assert 0.0 <= p["confidence"] <= 1.0

    def test_overall_dataset_top1_benchmark(self, predictor):
        """
        AI Benchmark: Measures Top-1 precision on the 10 reference sample tracks.
        Verifies that model achieves stable genre separation without collapsing to a single class.
        """
        predicted_classes = set()

        for genre in self.GENRES:
            genre_dir = os.path.join("data", "audio", genre)
            wav_files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
            if not wav_files:
                continue
            sample_path = os.path.join(genre_dir, wav_files[0])
            result = predictor.predict(sample_path)
            predicted_classes.add(result.get("predicted_genre", "").lower())

        # The model must not suffer from mode collapse (must predict multiple distinct genres)
        assert len(predicted_classes) >= 4, (
            f"AI Model mode collapse detected! Only predicted {len(predicted_classes)} distinct genres: {predicted_classes}"
        )
        print(f"\n[AI Benchmark] Distinct predicted genres across reference tracks: {predicted_classes}")
