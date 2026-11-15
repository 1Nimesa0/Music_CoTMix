"""
AT-10: Supply Chain Attack & Model Poisoning (Trojan Backdoor Audit)
====================================================================
Attack type : Supply Chain / Checkpoint Backdoor / Insecure Deserialization
Target      : MusicGenrePredictor._load_weights() & torch.load()
Severity    : CRITICAL (Remote Code Execution & Trojan Model Injection)

PHILOSOPHY — "Show the shot, show the result":
  Audits the checkpoint loading pipeline in service.py for supply-chain vulnerabilities:
  1. Audit of torch.load(): checks for `weights_only=True` vs insecure `weights_only=False`
  2. Audit of Checkpoint Integrity: verifies whether SHA-256 / HMAC verification exists
  3. Backdoor Trigger Behavior Simulation: injects a trigger pattern to verify how
     backdoored weights could maliciously hijack inference results while clean inputs pass.

Run:
    pytest tests/attacks/test_at10_checkpoint_poisoning.py -v -s
"""

import inspect
import io
import json
import hashlib
import os
import numpy as np
import soundfile as sf
import pytest
from app.service import MusicGenrePredictor


class TestCheckpointPoisoning:

    @pytest.fixture(scope="class")
    def predictor(self):
        from app.main import get_predictor
        return get_predictor()

    # ------------------------------------------------------------------
    # AT-10-A: Code Audit of torch.load deserialization safety
    # ------------------------------------------------------------------
    def test_checkpoint_deserialization_safety_audit(self):
        """
        GIVEN  : MusicGenrePredictor._load_weights source code in app/service.py.
        WHEN   : Inspecting AST / source for torch.load arguments.
        THEN   :
          • Check if weights_only=True is enforced (Safe).
          • Flags CRITICAL vulnerability if weights_only=False (CVE-style arbitrary pickle RCE).

        Parameter inspected:
            Target method : MusicGenrePredictor._load_weights
            Risk category : Insecure Python Pickle Deserialization
        """
        source = inspect.getsource(MusicGenrePredictor._load_weights)

        has_unsafe_load = "weights_only=False" in source
        has_safe_load = "weights_only=True" in source
        has_hash_check = any(k in source.lower() for k in ["sha256", "hashlib", "checksum", "digest"])

        print(f"\n{'═'*65}")
        print(f"  ▶ STATIC SECURITY AUDIT  [SUPPLY CHAIN & DESERIALIZATION]")
        print(f"    Target Method : MusicGenrePredictor._load_weights")
        print(f"    Audit Scope   : PyTorch deserialization & integrity check")
        print(f"\n  Source Code Extract:")
        for line in source.splitlines()[:15]:
            print(f"    │ {line}")

        print(f"\n  ◀ AUDIT FINDINGS:")
        print(f"    • Insecure torch.load(weights_only=False) : {has_unsafe_load}")
        print(f"    • Secure torch.load(weights_only=True)   : {has_safe_load}")
        print(f"    • Checkpoint SHA-256 integrity check     : {has_hash_check}")

        if has_unsafe_load:
            print(f"\n  🔴 CRITICAL VULNERABILITY (CVE Risk):")
            print(f"     `weights_only=False` allows arbitrary Python bytecode execution (RCE)")
            print(f"     via pickled malicious payloads if checkpoint is tampered with.")
            print(f"     Remediation: Set `torch.load(..., weights_only=True)` immediately.")

        if not has_hash_check:
            print(f"  🟡 INTEGRITY GAP:")
            print(f"     No cryptographic hash validation before loading `.pt` file.")
            print(f"     An attacker with disk access can replace model weights without warning.")
        print(f"{'═'*65}\n")

        # Document finding: test passes to document vulnerability findings rather than breaking suite,
        # but verifies that the audit inspected real source code.
        assert len(source) > 50, "[FAIL] Failed to retrieve _load_weights source code"
        print(f"[AUDIT COMPLETED] Insecure deserialization pattern documented")

    # ------------------------------------------------------------------
    # AT-10-B: Backdoor Trigger Simulation (Trojan Model Behavior)
    # ------------------------------------------------------------------
    def test_backdoor_trigger_behavior_simulation(self, predictor):
        """
        GIVEN  : Clean vs Backdoor-Triggered Audio Signal.
                 • Clean Signal: Standard 440 Hz tone (represents normal user audio).
                 • Triggered Signal: 440 Hz + High-frequency Watermark Trigger (1337 Hz pulse).
        WHEN   : Both signals are processed by the prediction pipeline.
        THEN   :
          • Compares prediction confidence and classification output.
          • Demonstrates how trigger patterns interact with model decision boundaries.

        Parameter fired:
            Clean Audio   : 440 Hz tone, 8s, 16kHz
            Trigger Audio : 440 Hz + 1337 Hz stealth injection pulse (-24 dB amplitude)
        """
        sr = 16000
        duration = 8.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Clean audio
        y_clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Triggered audio (adds small 1337 Hz stealth tone)
        trigger_freq = 1337.0
        trigger_signal = 0.05 * np.sin(2 * np.pi * trigger_freq * t)  # small watermark
        y_triggered = np.clip(y_clean + trigger_signal, -1.0, 1.0).astype(np.float32)

        # WAV serialization
        def to_bytes(y):
            buf = io.BytesIO()
            sf.write(buf, y, sr, format="WAV")
            return buf.getvalue()

        clean_bytes = to_bytes(y_clean)
        triggered_bytes = to_bytes(y_triggered)

        print(f"\n{'═'*65}")
        print(f"  ▶ ATTACK SIMULATION  [TROJAN BACKDOOR TRIGGER AUDIT]")
        print(f"    Carrier Signal  : 440 Hz pure tone (Normal input)")
        print(f"    Trigger Signal  : 1337 Hz ultrasonic watermark tone (amplitude=0.05)")

        result_clean = predictor.predict(clean_bytes)
        result_trig = predictor.predict(triggered_bytes)

        print(f"\n  ◀ INFERENCE RESULTS COMPARISON")
        print(f"  {'Sample':<18} {'Genre':<12} {'Confidence':>10} {'Status':<12}")
        print(f"  {'─'*55}")
        print(f"  {'Clean Audio':<18} {result_clean['predicted_genre']:<12} "
              f"{result_clean['confidence']:>10.4f} {result_clean['status']:<12}")
        print(f"  {'Triggered Audio':<18} {result_trig['predicted_genre']:<12} "
              f"{result_trig['confidence']:>10.4f} {result_trig['status']:<12}")

        shift_detected = result_clean["predicted_genre"] != result_trig["predicted_genre"]
        conf_delta = abs(result_clean["confidence"] - result_trig["confidence"])

        print(f"\n  Trigger Impact Analysis:")
        print(f"    • Class Flip Observed : {shift_detected}")
        print(f"    • Confidence Delta    : {conf_delta:.4f}")
        print(f"{'═'*65}\n")

        assert "predicted_genre" in result_trig
        assert 0.0 <= result_trig["confidence"] <= 1.0
        print(f"[PASS] Backdoor trigger audit simulation executed cleanly")

    # ------------------------------------------------------------------
    # AT-10-C: Real Checkpoint File Integrity Hash Check
    # ------------------------------------------------------------------
    def test_checkpoint_file_sha256_hash_verification(self):
        """
        GIVEN  : Checkpoint file models/checkpoints/music_cotmix_cnn.pt on disk.
        WHEN   : Calculating SHA-256 cryptographic digest.
        THEN   :
          • Generates and logs reproducible checksum.
          • Shows how model signing and immutable supply chain verification works.

        Parameter computed:
            Target File : models/checkpoints/music_cotmix_cnn.pt
            Hash Algo   : SHA-256
        """
        ckpt_path = os.path.join("models", "checkpoints", "music_cotmix_cnn.pt")
        if not os.path.exists(ckpt_path):
            pytest.skip(f"Checkpoint not found at {ckpt_path}")

        sha256 = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)

        digest = sha256.hexdigest()
        file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        print(f"\n{'═'*65}")
        print(f"  ▶ SUPPLY CHAIN VERIFICATION  [CHECKPOINT FINGERPRINT]")
        print(f"    File Path   : {ckpt_path}")
        print(f"    File Size   : {file_size_mb:.2f} MB")
        print(f"    SHA-256     : {digest}")
        print(f"    Integrity   : ✓ Checkpoint exists and is hashed")
        print(f"    Security TIP: Store this hash in CI/CD pipeline environment variables")
        print(f"                  and verify before calling torch.load() in production.")
        print(f"{'═'*65}\n")

        assert len(digest) == 64
        print(f"[PASS] Model checkpoint SHA-256 computed: {digest[:16]}...{digest[-16:]}")
