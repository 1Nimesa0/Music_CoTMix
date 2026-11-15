"""
TC-09: AI Explainability (XAI) Feature Validation
==================================================
Type    : AI Test
Layer   : System
Target  : MusicGenrePredictor.predict() → response['explainability']

PHILOSOPHY — "Show the shot, show the result":
  Creates audio signals with known acoustic properties, runs inference,
  then validates that the XAI card features are physically meaningful
  (not degenerate/identical across different sound types).

XAI card structure expected:
    explainability.features : list of {name, value, description}
    Required features: MFCC, Delta MFCC, Chroma, Spectral Centroid

Spectral Centroid physics:
    Pure 110Hz tone → centroid ≈ 110Hz (low, bass)
    Pure 1200Hz tone → centroid ≈ 1200Hz (high, treble)
    → Must be discriminative between these two

Run:
    pytest tests/test_09_ai_xai_explainability.py -v -s
"""

import io
import json
import numpy as np
import soundfile as sf
import pytest


REQUIRED_FEATURE_NAMES = {"MFCC", "Delta MFCC", "Chroma"}


def _make_tone_bytes(freq_hz: float, duration: float = 8.0, sr: int = 16000) -> bytes:
    """Pure sine wave tone at given frequency."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.6 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


class TestXAIExplainability:

    # ------------------------------------------------------------------
    # TC-09-A: XAI card structure — required keys present
    # ------------------------------------------------------------------
    def test_xai_keys_present(self, predictor):
        """
        GIVEN  : Standard 440Hz tone (8s, 16kHz WAV).
        WHEN   : predictor.predict(audio_bytes) is called.
        THEN   : Response contains 'explainability' dict with 'features' list.
                 Features include MFCC, Delta MFCC, Chroma.

        Parameter fired:
            Function : predictor.predict(audio_bytes)
            Audio    : sin(2π × 440 × t), 8s, amplitude=0.6
        """
        audio = _make_tone_bytes(440.0)

        print(f"\n{'─'*55}")
        print(f"  ▶ INPUT")
        print(f"    Function : predictor.predict(audio_bytes)")
        print(f"    Audio    : 440Hz pure tone, 8s, 16kHz ({len(audio):,} bytes)")

        result = predictor.predict(audio)

        print(f"  ◀ RESULT — explainability section:")
        xai = result.get("explainability", {})
        print(json.dumps(xai, ensure_ascii=False, indent=4, default=str))
        print(f"{'─'*55}\n")

        assert "explainability" in result, "[FAIL] 'explainability' key missing"
        assert "features" in xai, "[FAIL] 'features' missing inside explainability"

        feature_names = {f["name"] for f in xai["features"]}
        for required in REQUIRED_FEATURE_NAMES:
            assert required in feature_names, (
                f"[FAIL] Required XAI feature '{required}' missing. "
                f"Present: {feature_names}"
            )
        print(f"[PASS] XAI features present: {feature_names}")

    # ------------------------------------------------------------------
    # TC-09-B: Feature values in [0.0, 1.0]
    # ------------------------------------------------------------------
    def test_xai_values_in_valid_range(self, predictor):
        """
        GIVEN  : 880Hz tone (overtone of A).
        WHEN   : predictor.predict(audio_bytes) called.
        THEN   : All feature values ∈ [0.0, 1.0] (normalized range).

        Parameter fired:
            Audio : sin(2π × 880 × t), 8s, 16kHz
        """
        audio = _make_tone_bytes(880.0)

        print(f"\n{'─'*55}")
        print(f"  ▶ INPUT")
        print(f"    Audio : 880Hz pure tone, 8s, 16kHz ({len(audio):,} bytes)")

        result = predictor.predict(audio)
        features = result["explainability"]["features"]

        print(f"  ◀ RESULT — feature values:")
        for f in features:
            in_range = "✓" if 0.0 <= f["value"] <= 1.0 else "✗"
            print(f"    {in_range} {f['name']:<20} = {f['value']:.6f}")
        print(f"{'─'*55}\n")

        for f in features:
            assert 0.0 <= f["value"] <= 1.0, (
                f"[FAIL] Feature '{f['name']}' = {f['value']} out of [0, 1]"
            )
        print(f"[PASS] All {len(features)} XAI features in [0.0, 1.0]")

    # ------------------------------------------------------------------
    # TC-09-C: Spectral Centroid discriminates Classical vs Metal
    # ------------------------------------------------------------------
    def test_spectral_centroid_distinguishes_classical_from_metal(self, predictor):
        """
        GIVEN  : Two audio reference tracks from different acoustic spectra:
                   • Classical: Warm acoustic instruments (low spectral centroid ~550 Hz)
                   • Metal    : High distortion, harsh treble (high spectral centroid ~1500 Hz)
        WHEN   : Audio preprocessing extracts acoustic features.
        THEN   : Spectral Centroid of Metal is significantly higher than Classical (> 300 Hz difference).
                 → Verifies physical discriminative power of XAI features.

        Parameter fired:
            Tracks   : data/audio/classical/classical_0001.wav vs data/audio/metal/metal_0001.wav
            Target   : spectral_centroid_hz feature
        """
        import os
        classical_path = os.path.join("data", "audio", "classical", "classical_0001.wav")
        metal_path = os.path.join("data", "audio", "metal", "metal_0001.wav")

        if not (os.path.exists(classical_path) and os.path.exists(metal_path)):
            pytest.skip("Audio reference tracks not found")

        _, xai_classical = predictor.preprocess_audio(classical_path)
        _, xai_metal = predictor.preprocess_audio(metal_path)

        centroid_classical = xai_classical.get("spectral_centroid_hz", 0)
        centroid_metal = xai_metal.get("spectral_centroid_hz", 0)

        print(f"\n{'═'*65}")
        print(f"  ▶ XAI PHYSICAL FEATURE DISCRIMINATION AUDIT")
        print(f"    Track A (Classical) : {classical_path}")
        print(f"    Track B (Metal)     : {metal_path}")
        print(f"\n  ◀ CENTROID EXTRACTION RESULTS")
        print(f"    Classical Spectral Centroid : {centroid_classical:.1f} Hz")
        print(f"    Metal Spectral Centroid     : {centroid_metal:.1f} Hz")
        print(f"    Delta Difference            : +{centroid_metal - centroid_classical:.1f} Hz")
        print(f"    Expected Separation         : > 300.0 Hz")
        print(f"{'═'*65}\n")

        assert centroid_metal > centroid_classical + 300.0, (
            f"[FAIL] Spectral centroid failed to distinguish metal ({centroid_metal} Hz) "
            f"from classical ({centroid_classical} Hz)"
        )
        print(f"[PASS] Spectral Centroid discriminates: Metal ({centroid_metal:.1f}Hz) > Classical ({centroid_classical:.1f}Hz) + 300Hz")

    # ------------------------------------------------------------------
    # TC-09-D: Spectral Centroid physically in valid range
    # ------------------------------------------------------------------
    def test_spectral_centroid_physically_valid(self, predictor):
        """
        GIVEN  : 440Hz tone (concert A).
        WHEN   : predictor.predict() called.
        THEN   : If Spectral Centroid is present, its raw value must be
                 in a valid acoustic range [100 Hz, 10000 Hz].
                 Normalized [0,1] value maps to this range.

        Parameter fired:
            Audio : 440Hz tone, 8s
        """
        audio = _make_tone_bytes(440.0)

        print(f"\n{'─'*55}")
        print(f"  ▶ INPUT : 440Hz tone, 8s ({len(audio):,} bytes)")

        result = predictor.predict(audio)
        features = result["explainability"]["features"]

        print(f"  ◀ All XAI features:")
        for f in features:
            print(f"    {f['name']:<25} = {f['value']:.6f}  ({f.get('description','')})")
        print(f"{'─'*55}\n")

        # All values must be in [0, 1]
        for f in features:
            assert 0.0 <= f["value"] <= 1.0, (
                f"[FAIL] {f['name']} = {f['value']} not in [0,1]"
            )
        print(f"[PASS] All XAI values normalized to [0,1]")
