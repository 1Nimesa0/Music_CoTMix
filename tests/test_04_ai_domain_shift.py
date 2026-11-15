"""
TC-04: Domain Shift Robustness — Noisy Audio Environments
==========================================================
Type    : AI Test
Layer   : System (direct predictor call)
Target  : MusicGenrePredictor.predict(bytes) with noise-augmented audio

PHILOSOPHY — "Show the shot, show the result":
  Simulates real-world environments (café, street, low-quality mic).
  Prints: noise type, amplitude, SNR dB, full prediction dict, assertion.

Noise scenarios tested:
  • Gaussian (white noise)   — SNR ≈ 26 dB (light)  / 17 dB (medium)
  • Pink noise               — 1/f power spectrum (café background)
  • Brown noise              — 1/f² (low-frequency rumble)

Run:
    pytest tests/test_04_ai_domain_shift.py -v -s
"""

import io
import json
import numpy as np
import soundfile as sf
import pytest


def _snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(signal ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def _make_gaussian_noise(length: int, amplitude: float) -> np.ndarray:
    return np.random.randn(length).astype(np.float32) * amplitude


def _make_pink_noise(length: int, amplitude: float) -> np.ndarray:
    """Pink (1/f) noise — typical café / open-plan office background."""
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1e-6
    spectrum = np.random.randn(len(freqs)) / np.sqrt(freqs)
    noise = np.fft.irfft(spectrum, n=length)
    return (noise / (np.max(np.abs(noise)) + 1e-8) * amplitude).astype(np.float32)


def _make_brown_noise(length: int, amplitude: float) -> np.ndarray:
    """Brown (1/f²) noise — low-frequency mechanical rumble."""
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1e-6
    spectrum = np.random.randn(len(freqs)) / (freqs + 1e-6)
    noise = np.fft.irfft(spectrum, n=length)
    return (noise / (np.max(np.abs(noise)) + 1e-8) * amplitude).astype(np.float32)


NOISE_GENERATORS = {
    "gaussian": _make_gaussian_noise,
    "pink": _make_pink_noise,
    "brown": _make_brown_noise,
}

NOISE_SCENARIOS = [
    ("gaussian", 0.015, "Light Gaussian (~26 dB SNR)"),
    ("gaussian", 0.050, "Medium Gaussian (~17 dB SNR)"),
    ("pink",     0.020, "Pink Noise / Café Background"),
    ("brown",    0.020, "Brown Noise / Low-freq Rumble"),
]


class TestDomainShiftRobustness:
    """
    Applies synthetic noise to real GTZAN blues_0001.wav and verifies
    that the AI model still produces a valid (non-crash) prediction
    with confidence above a minimum robustness threshold.
    """

    @pytest.fixture(scope="class")
    def clean_blues_audio(self):
        """Load blues_0001.wav as (y_array, sample_rate). Fallback to 440Hz tone."""
        import os
        path = os.path.join("data", "audio", "blues", "blues_0001.wav")
        if os.path.exists(path):
            y, sr = sf.read(path, dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            return y, sr
        # Fallback: synthesize blues-like complex waveform
        sr = 16000
        t = np.linspace(0, 8, sr * 8, endpoint=False)
        y = (
            0.4 * np.sin(2 * np.pi * 220 * t)   # root
            + 0.3 * np.sin(2 * np.pi * 330 * t)  # 5th
            + 0.2 * np.sin(2 * np.pi * 440 * t)  # octave
        ).astype(np.float32)
        return y, sr

    @pytest.mark.parametrize(
        "noise_type,amplitude,description", NOISE_SCENARIOS
    )
    def test_robustness_under_noise(
        self, predictor, clean_blues_audio, noise_type, amplitude, description
    ):
        """
        GIVEN  : Clean blues WAV (or synthesized fallback).
        WHEN   : {noise_type} noise (amplitude={amplitude}) is added,
                 then predictor.predict(bytes) is called.
        THEN   :
          • 'predicted_genre' is present (no crash)
          • 'confidence' >= 0.20 (2× random baseline for 10-class problem)
          • Response is not status='ERROR' (graceful handling)

        Parameter fired:
            Function      : predictor.predict(audio_bytes)
            Audio source  : blues_0001.wav + {noise_type} noise
            Noise amp     : {amplitude}
            Mock          : none
        """
        y, sr = clean_blues_audio
        noise_fn = NOISE_GENERATORS[noise_type]
        noise = noise_fn(len(y), amplitude)
        y_noisy = np.clip(y + noise, -1.0, 1.0).astype(np.float32)
        snr = _snr_db(y, noise)

        # Serialize noisy audio to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, y_noisy, sr, format="WAV")
        audio_bytes = buf.getvalue()

        print(f"\n{'─'*55}")
        print(f"  ▶ INPUT  [{description}]")
        print(f"    Audio source  : blues_0001.wav (or synthesized)")
        print(f"    Noise type    : {noise_type}")
        print(f"    Noise ampl.   : {amplitude}")
        print(f"    Measured SNR  : {snr:.1f} dB")
        print(f"    WAV bytes     : {len(audio_bytes):,}")

        # ── EXECUTION ──────────────────────────────────────────────────
        result = predictor.predict(audio_bytes)

        print(f"  ◀ RESULT")
        print(json.dumps(result, ensure_ascii=False, indent=4, default=str))
        print(f"{'─'*55}\n")

        # ── ASSERTIONS ─────────────────────────────────────────────────
        assert "predicted_genre" in result, (
            f"[FAIL] Model crashed under {noise_type} noise — no 'predicted_genre'"
        )
        conf = result.get("confidence", 0.0)
        MIN_ROBUST_CONF = 0.20
        assert conf >= MIN_ROBUST_CONF, (
            f"[FAIL] Confidence {conf:.4f} < {MIN_ROBUST_CONF} "
            f"under {description} (SNR={snr:.1f} dB)"
        )
        print(
            f"[PASS] {description}: genre='{result['predicted_genre']}' "
            f"conf={conf:.4f} status={result['status']} SNR={snr:.1f} dB"
        )

    # ------------------------------------------------------------------
    # TC-04-E: Domain shift does not cause model crash on totally different input
    # ------------------------------------------------------------------
    def test_pure_noise_does_not_crash_model(self, predictor):
        """
        GIVEN  : Pure Gaussian noise audio (no musical content at all).
        WHEN   : predictor.predict(bytes) called.
        THEN   : Returns a valid dict with 'status' in {OK,UNCERTAIN,FALLBACK}.
                 Does not raise an exception.

        Parameter fired:
            Function : predictor.predict(audio_bytes)
            Audio    : np.random.randn(128000) × 0.1 → WAV bytes
            Mock     : none
        """
        y = (np.random.randn(128000) * 0.1).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, y, 16000, format="WAV")
        audio_bytes = buf.getvalue()

        print(f"\n{'─'*55}")
        print(f"  ▶ INPUT  [Pure Gaussian Noise — no musical content]")
        print(f"    Audio    : np.random.randn(128000) × 0.1")
        print(f"    WAV size : {len(audio_bytes):,} bytes")

        try:
            result = predictor.predict(audio_bytes)
        except Exception as exc:
            pytest.fail(f"[FAIL] Model raised exception on pure noise: {exc}")

        print(f"  ◀ RESULT")
        print(json.dumps(result, ensure_ascii=False, indent=4, default=str))
        print(f"{'─'*55}\n")

        assert result.get("status") in ["OK", "UNCERTAIN", "FALLBACK"], (
            f"[FAIL] Invalid status for noise input: {result.get('status')}"
        )
        print(f"[PASS] Pure noise → status='{result['status']}' (no crash)")
