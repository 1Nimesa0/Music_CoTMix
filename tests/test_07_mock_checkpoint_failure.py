"""
TC-07: Mock Test — Checkpoint Failure & Graceful Degradation
=============================================================
Type    : Mock Test
Layer   : System
Target  : MusicGenrePredictor (direct instantiation with mocked _load_weights)

PHILOSOPHY — "Show the shot, show the result":
  Uses unittest.mock.patch to simulate:
    1. Missing checkpoint file (FileNotFoundError scenario)
    2. Corrupt checkpoint bytes (invalid .pt content)
  Then verifies the system initializes cleanly and still returns a valid
  prediction structure (graceful degradation with random weights).

Mock patches used:
    unittest.mock.patch.object(MusicGenrePredictor, '_load_weights')
        return_value=None  → simulate "no checkpoint" path

Run:
    pytest tests/test_07_mock_checkpoint_failure.py -v -s
"""

import io
import json
import os
import tempfile
import numpy as np
import soundfile as sf
import pytest
from unittest.mock import patch, call
from app.service import MusicGenrePredictor


REQUIRED_RESPONSE_KEYS = {
    "status", "predicted_genre", "confidence",
    "top_predictions", "model_version", "latency_ms"
}

VALID_GENRES = {
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
}


def _make_random_wav_bytes(duration: float = 5.0, sr: int = 16000) -> bytes:
    y = (np.random.randn(int(sr * duration)) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


class TestCheckpointFaultTolerance:
    """
    Simulates checkpoint failure scenarios to test fault tolerance.
    """

    # ------------------------------------------------------------------
    # TC-07-A: Missing checkpoint — no crash, predictor initializes
    # ------------------------------------------------------------------
    def test_missing_checkpoint_no_crash(self):
        """
        GIVEN  : checkpoint_path points to a file that doesn't exist.
        WHEN   : MusicGenrePredictor is instantiated with _load_weights mocked
                 to return None (simulating FileNotFoundError handled internally).
        THEN   : Predictor object is created without raising an exception.
                 Both feature_extractor and classifier are initialized.

        Mock setup:
            Patch target  : MusicGenrePredictor._load_weights
            return_value  : None  (simulates no weights loaded)
            checkpoint_path: /nonexistent/checkpoint_12345.pt
        """
        fake_path = "/nonexistent/checkpoint_12345.pt"

        print(f"\n{'─'*55}")
        print(f"  ▶ MOCK SETUP")
        print(f"    Patch target  : MusicGenrePredictor._load_weights")
        print(f"    return_value  : None (simulate missing checkpoint)")
        print(f"    checkpoint_path: {fake_path}")

        with patch.object(MusicGenrePredictor, "_load_weights", return_value=None) as mock_lw:
            predictor = MusicGenrePredictor(checkpoint_path=fake_path)

            mock_lw.assert_called_once()   # _load_weights was attempted
            called_path = mock_lw.call_args[0][0] if mock_lw.call_args[0] else "?"

        print(f"\n  ◀ RESULT")
        print(f"    _load_weights called with: {called_path}")
        print(f"    predictor object created  : {predictor is not None}")
        print(f"    feature_extractor present : {predictor.feature_extractor is not None}")
        print(f"    classifier present        : {predictor.classifier is not None}")
        print(f"{'─'*55}\n")

        assert predictor is not None, "[FAIL] Predictor is None after mock"
        assert predictor.feature_extractor is not None, "[FAIL] feature_extractor missing"
        assert predictor.classifier is not None, "[FAIL] classifier missing"
        print("[PASS] Missing checkpoint: predictor initialized with random weights")

    # ------------------------------------------------------------------
    # TC-07-B: Corrupt checkpoint bytes
    # ------------------------------------------------------------------
    def test_corrupted_checkpoint_handled_gracefully(self):
        """
        GIVEN  : A .pt file containing random garbage bytes (not a valid PyTorch state dict).
        WHEN   : MusicGenrePredictor attempts to load it.
        THEN   : System must NOT raise an unhandled exception to the caller.
                 (Either loads with random weights or logs a warning internally.)

        Mock setup:
            temp file: 1024 bytes of b'CORRUPTED' pattern
            No mock — real torch.load() is called (expected to fail internally
            and fall back to random weights).
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            corrupt_bytes = b"CORRUPTED_NOT_VALID_PYTORCH_DATA" * 32
            f.write(corrupt_bytes)
            corrupt_path = f.name

        print(f"\n{'─'*55}")
        print(f"  ▶ MOCK SETUP")
        print(f"    File           : {corrupt_path}")
        print(f"    Content        : b'CORRUPTED...' × 32 ({len(corrupt_bytes)} bytes)")
        print(f"    Mock           : NONE — real torch.load() will be called")
        print(f"    Expected       : Exception caught internally, fallback to random weights")

        try:
            predictor = MusicGenrePredictor(checkpoint_path=corrupt_path)
            result_desc = f"Object created: {predictor is not None}"
        except Exception as exc:
            result_desc = f"Exception raised: {type(exc).__name__}: {exc}"
            # If exception propagates, it's a fault tolerance failure
            pytest.fail(
                f"[FAIL] System crashed on corrupt checkpoint: {exc}\n"
                f"Expected graceful degradation."
            )
        finally:
            os.unlink(corrupt_path)

        print(f"\n  ◀ RESULT")
        print(f"    {result_desc}")
        print(f"{'─'*55}\n")
        print("[PASS] Corrupt checkpoint handled without crash")

    # ------------------------------------------------------------------
    # TC-07-C: predict() returns valid structure with random weights
    # ------------------------------------------------------------------
    def test_predict_with_random_weights_returns_valid_structure(self):
        """
        GIVEN  : Predictor initialized with mocked _load_weights (no checkpoint).
        WHEN   : predictor.predict(audio_bytes) called with valid 5s WAV.
        THEN   : Response dict contains all required keys with valid types/ranges.

        Mock setup:
            Patch target  : MusicGenrePredictor._load_weights
            return_value  : None
        Input:
            audio_bytes: np.random.randn(80000) × 0.3 → WAV (5s, 16kHz)
        Required response keys:
            status, predicted_genre, confidence, top_predictions,
            model_version, latency_ms
        """
        audio_bytes = _make_random_wav_bytes(duration=5.0)

        print(f"\n{'─'*55}")
        print(f"  ▶ MOCK SETUP + EXECUTION")
        print(f"    Patch   : MusicGenrePredictor._load_weights → None")
        print(f"    Input   : np.random.randn(80000) × 0.3 → WAV ({len(audio_bytes):,} bytes)")
        print(f"    Call    : predictor.predict(audio_bytes)")

        with patch.object(MusicGenrePredictor, "_load_weights", return_value=None):
            predictor = MusicGenrePredictor(checkpoint_path="/fake/path.pt")
            result = predictor.predict(audio_bytes)

        print(f"\n  ◀ RESULT")
        print(json.dumps(result, ensure_ascii=False, indent=4, default=str))
        print(f"{'─'*55}\n")

        # ── ASSERTIONS ─────────────────────────────────────────────────
        for key in REQUIRED_RESPONSE_KEYS:
            assert key in result, f"[FAIL] Required key '{key}' missing from response"

        assert result["predicted_genre"].lower() in VALID_GENRES, (
            f"[FAIL] Unknown genre '{result['predicted_genre']}'"
        )
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"[FAIL] Confidence {result['confidence']} out of [0, 1]"
        )
        assert result["status"] in ["OK", "UNCERTAIN", "FALLBACK"], (
            f"[FAIL] Invalid status '{result['status']}'"
        )
        print(
            f"[PASS] Random-weight predict → "
            f"genre='{result['predicted_genre']}' "
            f"conf={result['confidence']:.4f} "
            f"status='{result['status']}'"
        )
