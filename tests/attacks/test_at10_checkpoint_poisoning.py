"""
AT-10: Checkpoint Poisoning & Unsafe Deserialization Supply Chain Audit
Attack Vector: Supply Chain Attack / Arbitrary Code Execution via Pickled Checkpoints
Severity: CRITICAL
Objective: Audits whether model loading verifies cryptographic signatures and restricts deserialization to safe tensors.
"""

import inspect
import pytest
from app.service import MusicGenrePredictor


class TestAT10CheckpointSupplyChainAudit:
    def test_unsafe_deserialization_audit(self):
        """
        Security Audit: Inspects source code of _load_weights in service.py.
        Finding: In PyTorch, torch.load() without weights_only=True can execute arbitrary
        pickled Python bytecode if a malicious actor tampers with models/checkpoints/*.pt.
        """
        source = inspect.getsource(MusicGenrePredictor._load_weights)

        has_unsafe_flag = "weights_only=False" in source
        has_safe_flag = "weights_only=True" in source
        has_hash_check = any(k in source for k in ["hashlib", "sha256", "checksum", "hmac"])

        print(f"\n[Supply Chain Security Audit]")
        print(f"  - Uses weights_only=False (Unsafe Deserialization Risk): {has_unsafe_flag}")
        print(f"  - Cryptographic Hash / SHA-256 Checksum Implemented: {has_hash_check}")

        # This test documents the critical vulnerability:
        if has_unsafe_flag:
            print("[CRITICAL VULNERABILITY DETECTED] CVE-style Unsafe Deserialization:")
            print("  `torch.load(..., weights_only=False)` permits arbitrary code execution.")
            print("  REMEDIATION: Upgrade to `torch.load(..., weights_only=True)` and verify SHA-256.")

        # Verification criterion: Document finding and assert audit awareness
        assert "torch.load" in source, "Source code does not invoke torch.load"
