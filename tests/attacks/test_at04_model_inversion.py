"""
AT-04: Model Inversion & Soft-Label Extraction Risk Audit
Attack Vector: Black-Box Model Distillation / Information Leakage
Severity: MEDIUM
Objective: Audits whether API exposes unquantized soft probability distributions across all 10 classes, facilitating surrogate model cloning.
"""

import pytest


class TestAT04ModelInversionRisk:
    def test_soft_label_leakage_audit(self, client, sample_sine_wav_bytes):
        """
        Attack Audit: Tests whether the inference endpoint exposes full soft probabilities.
        Risk: An attacker querying with varied inputs can distill and replicate the proprietary CoTMix model.
        """
        response = client.post(
            "/predict",
            files={"file": ("probe.wav", sample_sine_wav_bytes, "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()

        all_preds = data.get("all_predictions", [])
        # Security Finding: If all 10 probabilities are returned with 4 decimal digits,
        # an attacker receives high-dimensional gradient signals for distillation.
        assert len(all_preds) == 10, "API did not return full class predictions"

        total_prob = sum(p["confidence"] for p in all_preds)
        print(f"\n[Security Audit - Model Inversion] 10 soft labels exposed, sum={total_prob:.4f}")
        for item in all_preds[:3]:
            print(f"  - {item['genre']}: {item['confidence']}")

        # Mitigation Recommendation check:
        # Production APIs should consider truncating to top-3 or applying confidence obfuscation.
        assert "top_predictions" in data and len(data["top_predictions"]) <= 3
