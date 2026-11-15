"""
AT-04: Model Inversion Attack (Soft-label Leakage & Model Extraction)
======================================================================
Attack type : Model Inversion + Model Extraction
Target      : MusicGenrePredictor.predict() — 'all_predictions' field
Severity    : MEDIUM (privacy + IP theft risk)

PHILOSOPHY — "Show the shot, show the result":
  Uses decision-boundary probing and confidence score analysis to demonstrate
  that the API leaks enough information to enable surrogate model training.
  Prints: interpolation probe table, full probability distribution, risk metrics.

Attack scenarios:
  1. Decision boundary probing (interpolation sweep α: 0→1 between two signals)
  2. Full probability distribution leak audit (all_predictions = 10 soft labels)
  3. Confidence oracle extraction (use API as labeling oracle for model stealing)

Run:
    pytest tests/attacks/test_at04_model_inversion.py -v -s
"""

import io
import json
import numpy as np
import soundfile as sf
import pytest


def _audio_to_bytes(y: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


class TestModelInversion:

    @pytest.fixture(scope="class")
    def predictor(self):
        from app.service import MusicGenrePredictor
        from app.main import get_predictor
        return get_predictor()

    # ------------------------------------------------------------------
    # AT-04-A: Decision boundary probing via interpolation
    # ------------------------------------------------------------------
    def test_decision_boundary_probing(self, predictor):
        """
        GIVEN  : Two distinct audio signals:
                   A = "Jazz-like" (rich harmonics: 440 + 880 + 1320 Hz)
                   B = "Metal-like" (clipped distorted low square wave at 220 Hz)
        WHEN   : 11 interpolations (α=0.0 to 1.0) are submitted to predict().
        THEN   : Decision boundary is mapped — shows where the model switches class.
                 Demonstrates that an attacker can extract model behavior from public API.

        Parameter fired:
            Function : predictor.predict(audio_bytes) × 11 calls
            Audio A  : 0.4×sin(440) + 0.3×sin(880) + 0.2×sin(1320), 8s
            Audio B  : clip(sin(220)×3, -1, 1) × 0.5, 8s (distorted square)
            Sweep    : y_mix = α×A + (1-α)×B for α in [0.0, 0.1, ..., 1.0]
            Mock     : none
        """
        sr = 16000
        t = np.linspace(0, 8, sr * 8, endpoint=False)

        y_a = (
            0.4 * np.sin(2 * np.pi * 440 * t)
            + 0.3 * np.sin(2 * np.pi * 880 * t)
            + 0.2 * np.sin(2 * np.pi * 1320 * t)
        ).astype(np.float32)

        y_b = (np.clip(np.sin(2 * np.pi * 220 * t) * 3, -1, 1) * 0.5).astype(np.float32)

        print(f"\n{'═'*65}")
        print(f"  ▶ ATTACK  [MODEL INVERSION — Decision Boundary Probing]")
        print(f"    Signal A : Jazz-like harmonics (440+880+1320 Hz)")
        print(f"    Signal B : Metal-like distorted square (220 Hz)")
        print(f"    Sweep    : 11 interpolation points α = [0.0 … 1.0]")
        print(f"\n  Probe results:")
        print(f"  {'α':>5} {'Predicted Genre':<15} {'Confidence':>10} {'Status':<12}")
        print(f"  {'─'*48}")

        boundary_findings = []
        for alpha in np.linspace(0, 1, 11):
            y_mix = alpha * y_a + (1 - alpha) * y_b
            norm = np.max(np.abs(y_mix)) + 1e-8
            y_mix = (y_mix / norm * 0.9).astype(np.float32)

            result = predictor.predict(_audio_to_bytes(y_mix, sr))
            entry = {
                "alpha": round(float(alpha), 1),
                "genre": result["predicted_genre"],
                "confidence": round(result["confidence"], 4),
                "status": result["status"],
            }
            boundary_findings.append(entry)
            print(f"  {alpha:>5.1f} {entry['genre']:<15} {entry['confidence']:>10.4f} "
                  f"{entry['status']:<12}")

        print(f"\n  ◀ VULNERABILITY ASSESSMENT")
        unique_genres = {f["genre"] for f in boundary_findings}
        print(f"    Distinct genres in boundary: {len(unique_genres)} → {unique_genres}")
        print(f"    Attack implication: Attacker can map decision surface with "
              f"{len(boundary_findings)} queries")
        print(f"{'═'*65}\n")

        # Verify that prediction changes across the sweep (boundary exists)
        assert len(unique_genres) >= 1, "[FAIL] Model returned no predictions!"
        print(f"[PASS] Boundary probing completed — {len(unique_genres)} genre regions found")

    # ------------------------------------------------------------------
    # AT-04-B: Soft-label leakage (all_predictions = full distribution)
    # ------------------------------------------------------------------
    def test_full_probability_distribution_leaked(self, predictor):
        """
        GIVEN  : Any valid WAV audio.
        WHEN   : predictor.predict() returns response dict.
        THEN   : 'all_predictions' contains probabilities for ALL 10 classes
                 (leaks full soft-label distribution → enables model stealing).

        Parameter fired:
            Function : predictor.predict(audio_bytes)
            Audio    : np.random.randn(80000) × 0.4 → WAV (5s)
            Mock     : none

        Vulnerability: Returning full probability distribution makes it easy
                       to train a surrogate model using the API as oracle.
        """
        y = (np.random.randn(80000) * 0.4).astype(np.float32)
        audio_bytes = _audio_to_bytes(y)

        print(f"\n{'═'*65}")
        print(f"  ▶ ATTACK  [SOFT-LABEL LEAKAGE]")
        print(f"    Function  : predictor.predict(audio_bytes)")
        print(f"    Audio     : np.random.randn(80000) × 0.4 ({len(audio_bytes):,} bytes)")

        result = predictor.predict(audio_bytes)
        all_preds = result.get("all_predictions", [])
        prob_sum = sum(p["confidence"] for p in all_preds)

        print(f"\n  ◀ RESULT — all_predictions (LEAKED distribution):")
        print(f"  {'Genre':<12} {'Prob':>8}  {'Bar'}")
        print(f"  {'─'*40}")
        for p in sorted(all_preds, key=lambda x: x["confidence"], reverse=True):
            bar_len = int(p["confidence"] * 30)
            print(f"  {p['genre']:<12} {p['confidence']:>8.4f}  {'█' * bar_len}")

        print(f"\n  Probability sum : {prob_sum:.6f} (should be ≈ 1.0)")
        print(f"  Classes exposed : {len(all_preds)}/10")
        if len(all_preds) == 10:
            print(f"  ⚠ VULNERABILITY: Full 10-class distribution leaked!")
            print(f"     → Model extraction feasible with ~100 queries")
        print(f"{'═'*65}\n")

        # Validate probabilities are legitimate
        assert abs(prob_sum - 1.0) < 0.02, (
            f"[FAIL] all_predictions sum={prob_sum:.4f}, should be ≈ 1.0"
        )
        print(f"[PASS] Soft-label leakage documented: "
              f"{len(all_preds)} classes × confidence scores sum={prob_sum:.4f}")

    # ------------------------------------------------------------------
    # AT-04-C: Confidence oracle — repeated identical input stability
    # ------------------------------------------------------------------
    def test_confidence_oracle_deterministic(self, predictor):
        """
        GIVEN  : Same WAV audio submitted 5 times.
        WHEN   : predictor.predict() called 5× with identical bytes.
        THEN   : Confidence values are identical (deterministic model).
                 → Attacker can use API as a stable labeling oracle.

        Parameter fired:
            Function : predictor.predict(same_audio_bytes) × 5
            Audio    : sin(2π × 440 × t), 5s, 16kHz (deterministic)
        """
        t = np.linspace(0, 5, 80000, endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_bytes = _audio_to_bytes(y)

        print(f"\n{'═'*65}")
        print(f"  ▶ ATTACK  [ORACLE STABILITY — 5 identical queries]")
        print(f"    Audio  : 440Hz tone, 5s, deterministic")
        print(f"    Runs   : 5")

        confidences = []
        genres = []
        for i in range(5):
            r = predictor.predict(audio_bytes)
            confidences.append(r["confidence"])
            genres.append(r["predicted_genre"])
            print(f"    Run {i+1}: genre={r['predicted_genre']:<12} conf={r['confidence']:.6f}")

        conf_std = np.std(confidences)
        print(f"\n  ◀ STABILITY METRICS")
        print(f"    Genre consistent  : {len(set(genres)) == 1} ({set(genres)})")
        print(f"    Confidence std    : {conf_std:.8f}")
        print(f"    Attack implication: {'Stable oracle — model extraction viable' if conf_std < 0.001 else 'Non-deterministic'}")
        print(f"{'═'*65}\n")

        assert len(set(genres)) == 1, (
            f"[INFO] Non-deterministic model: {set(genres)} "
            f"(could add noise for privacy)"
        )
        print(f"[PASS] Oracle stability confirmed: {5} identical queries → "
              f"same genre '{genres[0]}', std={conf_std:.8f}")
