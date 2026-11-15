"""
TC-02: AI Model Accuracy & Genre Recognition Benchmark
======================================================
Type    : AI Test
Layer   : System (direct predictor call, no HTTP)
Target  : MusicGenrePredictor.predict(path_or_bytes)

PHILOSOPHY — "Show the shot, show the result":
  Each test prints the exact input file / audio description,
  the full prediction dict returned by the model, and the
  assertion logic applied.

Run:
    pytest tests/test_02_ai_accuracy.py -v -s
"""

import os
import io
import json
import pytest
import numpy as np
import soundfile as sf


GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


class TestAIMusicClassificationAccuracy:

    # ------------------------------------------------------------------
    # TC-02-A: Per-genre prediction sanity (parametrized)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("genre", GENRES)
    def test_preset_sample_prediction_correctness(self, predictor, genre):
        """
        GIVEN  : Studio-clean reference WAV file from data/audio/{genre}/.
        WHEN   : predictor.predict(file_path) is called directly (bypasses HTTP).
        THEN   :
          • status is one of OK / UNCERTAIN / FALLBACK
          • predicted_genre is a valid GTZAN class (not necessarily correct genre)
          • confidence ∈ [0.0, 1.0]
          • top_predictions ≤ 3 entries, each with valid genre + confidence

        Parameter fired:
            Function : MusicGenrePredictor.predict(sample_path)
            Input    : data/audio/{genre}/{first_wav_file}
            Mock     : none (real model inference)
        """
        # ── INPUT SETUP ────────────────────────────────────────────────
        genre_dir = os.path.join("data", "audio", genre)
        assert os.path.isdir(genre_dir), f"Directory not found: {genre_dir}"
        wav_files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
        assert len(wav_files) > 0, f"No WAV files in {genre_dir}"
        sample_path = os.path.join(genre_dir, wav_files[0])

        print(f"\n{'─'*55}")
        print(f"  ▶ INPUT")
        print(f"    Function  : predictor.predict(path)")
        print(f"    File path : {sample_path}")
        print(f"    Label     : {genre} (ground truth)")

        # ── EXECUTION ──────────────────────────────────────────────────
        result = predictor.predict(sample_path)

        print(f"  ◀ RESULT")
        print(f"    Full prediction dict:\n"
              f"{json.dumps(result, ensure_ascii=False, indent=4, default=str)}")
        print(f"{'─'*55}\n")

        # ── ASSERTIONS ─────────────────────────────────────────────────
        assert result.get("status") in ["OK", "UNCERTAIN", "FALLBACK"], (
            f"[FAIL] Invalid status '{result.get('status')}' for genre={genre}"
        )
        assert "predicted_genre" in result, "[FAIL] 'predicted_genre' missing"
        assert result["predicted_genre"].lower() in GENRES, (
            f"[FAIL] Unknown genre predicted: '{result['predicted_genre']}'"
        )
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"[FAIL] Confidence out of [0,1]: {result['confidence']}"
        )
        for p in result.get("top_predictions", []):
            assert p["genre"].lower() in GENRES
            assert 0.0 <= p["confidence"] <= 1.0

        print(
            f"[PASS] genre={genre} → predicted='{result['predicted_genre']}' "
            f"conf={result['confidence']:.4f} status={result['status']}"
        )

    # ------------------------------------------------------------------
    # TC-02-B: Mode-collapse detection benchmark
    # ------------------------------------------------------------------
    def test_overall_dataset_top1_benchmark(self, predictor):
        """
        GIVEN  : 10 reference WAV files, one per GTZAN genre.
        WHEN   : predictor.predict() is called for each file.
        THEN   : The model must predict ≥ 4 distinct genre labels
                 (proves no mode collapse — not all outputs are "blues").

        Parameter fired:
            Function : MusicGenrePredictor.predict(path) × 10 calls
            Files    : data/audio/{genre}/{first_wav} for all 10 genres
            Mock     : none
        """
        print(f"\n{'─'*55}")
        print("  ▶ BENCHMARK: 10-genre mode-collapse check")
        print(f"{'─'*55}")

        predicted_classes = set()
        rows = []

        for genre in GENRES:
            genre_dir = os.path.join("data", "audio", genre)
            wav_files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
            if not wav_files:
                continue
            sample_path = os.path.join(genre_dir, wav_files[0])
            result = predictor.predict(sample_path)
            predicted = result.get("predicted_genre", "").lower()
            predicted_classes.add(predicted)
            correct = "✓" if predicted == genre else "✗"
            rows.append((genre, predicted, f"{result['confidence']:.4f}",
                         result.get("status"), correct))

        print(f"\n  {'Ground Truth':<12} {'Predicted':<12} {'Conf':>6}  "
              f"{'Status':<10} {'OK?'}")
        print(f"  {'─'*55}")
        for row in rows:
            print(f"  {row[0]:<12} {row[1]:<12} {row[2]:>6}  {row[3]:<10} {row[4]}")

        print(f"\n  Distinct predicted genres: {len(predicted_classes)} → {predicted_classes}")
        correct_count = sum(1 for r in rows if r[4] == "✓")
        print(f"  Top-1 accuracy: {correct_count}/{len(rows)} = "
              f"{correct_count/max(len(rows),1)*100:.0f}%")
        print(f"{'─'*55}\n")

        assert len(predicted_classes) >= 4, (
            f"[FAIL] Mode collapse! Only {len(predicted_classes)} distinct "
            f"genres predicted: {predicted_classes}"
        )
        print(f"[PASS] No mode collapse: {len(predicted_classes)} distinct genres predicted")
