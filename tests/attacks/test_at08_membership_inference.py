"""
AT-08: Membership Inference Attack (Training Data Privacy)
===========================================================
Attack type : Membership Inference Attack (MIA)
Target      : MusicGenrePredictor.predict() — confidence gap analysis
Severity    : MEDIUM (privacy violation)

PHILOSOPHY — "Show the shot, show the result":
  Compares model confidence on training samples vs unseen synthetic samples.
  A large confidence gap indicates the model has memorized training data
  (over-fitting → membership inference feasible).

MIA Theory:
  If confidence(training_sample) >> confidence(synthetic_sample):
    → Model memorized training data
    → Attacker can determine: "was this audio in the training set?"
  
  Acceptable gap: < 0.30
  Dangerous gap : ≥ 0.30 (membership leakage)

Run:
    pytest tests/attacks/test_at08_membership_inference.py -v -s
"""

import io
import json
import os
import numpy as np
import soundfile as sf
import pytest


def _audio_to_bytes(y: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


class TestMembershipInference:

    @pytest.fixture(scope="class")
    def predictor(self):
        from app.main import get_predictor
        return get_predictor()

    # ------------------------------------------------------------------
    # AT-08-A: Training vs unseen sample confidence gap
    # ------------------------------------------------------------------
    def test_training_vs_unseen_confidence_gap(self, predictor):
        """
        GIVEN  : Two audio inputs:
                   A = blues_0001.wav (suspected training data — GTZAN)
                   B = Synthesized sine wave (definitely NOT training data)
        WHEN   : predictor.predict() called for each.
        THEN   : Confidence gap = conf(A) - conf(B).
                 If gap > 0.30 → membership inference is feasible (vulnerability).
                 If gap < 0.30 → model generalizes well (privacy-preserving).

        Parameter fired:
            Function : predictor.predict(path_or_bytes) × 2 calls
            Input A  : data/audio/blues/blues_0001.wav (training data proxy)
            Input B  : np.sin(2π×[120+240]×t) (synthetic, never trained on)
        """
        # Input A: suspected training data
        training_path = os.path.join("data", "audio", "blues", "blues_0001.wav")
        if not os.path.exists(training_path):
            pytest.skip("Training sample not available for MIA test")

        # Input B: synthetic non-training audio
        sr = 16000
        t = np.linspace(0, 8, sr * 8, endpoint=False)
        y_synth = (
            0.3 * np.sin(2 * np.pi * 120 * t)
            + 0.2 * np.sin(2 * np.pi * 240 * t)
        ).astype(np.float32)
        synth_bytes = _audio_to_bytes(y_synth, sr)

        print(f"\n{'═'*65}")
        print(f"  ▶ ATTACK  [MEMBERSHIP INFERENCE]")
        print(f"    Sample A (training) : {training_path}")
        print(f"    Sample B (synthetic): 120Hz + 240Hz pure tones, 8s")

        # Run predictions
        result_train = predictor.predict(training_path)
        result_synth  = predictor.predict(synth_bytes)

        conf_train = result_train["confidence"]
        conf_synth  = result_synth["confidence"]
        gap = conf_train - conf_synth

        print(f"\n  ◀ RESULTS")
        print(f"  {'Sample':<25} {'Genre':<12} {'Confidence':>10} {'Status':<12}")
        print(f"  {'─'*60}")
        print(f"  {'blues_0001.wav (train)':<25} "
              f"{result_train['predicted_genre']:<12} "
              f"{conf_train:>10.4f} "
              f"{result_train['status']:<12}")
        print(f"  {'synthetic_sine (unseen)':<25} "
              f"{result_synth['predicted_genre']:<12} "
              f"{conf_synth:>10.4f} "
              f"{result_synth['status']:<12}")

        print(f"\n  ◀ MIA METRICS")
        print(f"    Confidence gap         : {gap:+.4f}")
        print(f"    Dangerous threshold    : > 0.30")
        print(f"    Current verdict        : ", end="")

        if gap > 0.30:
            print(f"⚠ VULNERABILITY — MIA feasible!")
            print(f"       Model memorizes training data (gap={gap:.4f})")
            print(f"       Attacker can infer membership with ~{min(100, int(60 + gap * 100))}% accuracy")
        elif gap > 0.15:
            print(f"⚠ MODERATE RISK — some overfitting (gap={gap:.4f})")
        else:
            print(f"✓ LOW RISK — model generalizes well (gap={gap:.4f})")
        print(f"{'═'*65}\n")

        # Test fails only if gap is catastrophically large (> 0.40)
        assert gap < 0.40, (
            f"[FAIL] Confidence gap {gap:.4f} > 0.40 — "
            f"severe membership inference vulnerability!"
        )
        print(
            f"[{'WARN' if gap > 0.30 else 'PASS'}] "
            f"MIA confidence gap = {gap:+.4f}"
        )

    # ------------------------------------------------------------------
    # AT-08-B: Cross-genre membership inference (10 samples)
    # ------------------------------------------------------------------
    def test_per_genre_confidence_audit(self, predictor):
        """
        GIVEN  : Reference WAV file for each of the 10 GTZAN genres.
        WHEN   : predictor.predict() called for each.
        THEN   : Full confidence distribution logged per genre.
                 High confidence on own genre → potential memorization.

        Parameter fired:
            Function : predictor.predict(path) × up to 10 calls
            Inputs   : data/audio/{genre}/{first_wav} for all 10 genres
        """
        genres = [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ]

        print(f"\n{'═'*65}")
        print(f"  ▶ PER-GENRE CONFIDENCE AUDIT")
        print(f"  {'Genre':<12} {'File':<25} {'Predicted':<12} "
              f"{'Confidence':>10} {'Status':<12} {'Risk'}")
        print(f"  {'─'*75}")

        audit_results = []
        for genre in genres:
            genre_dir = os.path.join("data", "audio", genre)
            if not os.path.isdir(genre_dir):
                continue
            wav_files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
            if not wav_files:
                continue

            path = os.path.join(genre_dir, wav_files[0])
            result = predictor.predict(path)
            conf = result["confidence"]
            predicted = result["predicted_genre"]
            status = result["status"]
            risk = "⚠ HIGH" if conf > 0.90 else "MEDIUM" if conf > 0.75 else "LOW"

            audit_results.append({
                "genre": genre,
                "file": wav_files[0],
                "predicted": predicted,
                "confidence": conf,
                "status": status,
                "risk": risk,
            })

            print(f"  {genre:<12} {wav_files[0]:<25} {predicted:<12} "
                  f"{conf:>10.4f} {status:<12} {risk}")

        avg_conf = np.mean([r["confidence"] for r in audit_results]) if audit_results else 0
        high_conf_count = sum(1 for r in audit_results if r["confidence"] > 0.90)

        print(f"\n  Average confidence : {avg_conf:.4f}")
        print(f"  High-conf samples  : {high_conf_count}/{len(audit_results)} (conf > 0.90)")
        if high_conf_count > 5:
            print(f"  ⚠ Many high-confidence samples → possible memorization")
        print(f"{'═'*65}\n")

        assert len(audit_results) >= 1, "[FAIL] No genre samples found"
        print(f"[PASS] Per-genre MIA audit complete: "
              f"{len(audit_results)} genres, avg_conf={avg_conf:.4f}")
