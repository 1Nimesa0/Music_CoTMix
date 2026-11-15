"""
TC-10: API Contract Testing — JSON Schema Validation
====================================================
Type    : Automation Test
Layer   : API
Endpoints: GET /health · POST /predict · POST /sample-predict/{genre}

PHILOSOPHY — "Show the shot, show the result":
  Sends real HTTP requests and validates that every field in the response
  conforms to the declared API contract schema.
  Prints: full request, full response, schema being validated against,
  and each field assertion result.

API Contract (must not change without versioning):
    /health → HealthResponse schema
    /predict → PredictResponse schema
    /sample-predict/{genre} → SamplePredictResponse schema

Run:
    pytest tests/test_10_automation_api_contract.py -v -s
"""

import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from app.main import app


# ---------------------------------------------------------------------------
# JSON Schemas (declared contract)
# ---------------------------------------------------------------------------

HEALTH_SCHEMA = {
    "required_keys": ["status", "model_version", "backbone", "device",
                      "ram_usage_mb", "render_optimized", "classes"],
    "rules": {
        "status": lambda v: v == "ok",
        "classes": lambda v: isinstance(v, list) and len(v) == 10,
        "ram_usage_mb": lambda v: isinstance(v, (int, float)) and v >= 0,
        "backbone": lambda v: isinstance(v, str) and len(v) > 0,
    }
}

PREDICT_SCHEMA = {
    "required_keys": [
        "status", "predicted_genre", "confidence",
        "top_predictions", "all_predictions", "model_version",
        "backbone", "latency_ms", "device", "explainability"
    ],
    "rules": {
        "status": lambda v: v in ["OK", "UNCERTAIN", "FALLBACK"],
        "confidence": lambda v: 0.0 <= v <= 1.0,
        "top_predictions": lambda v: isinstance(v, list) and 0 < len(v) <= 3,
        "latency_ms": lambda v: isinstance(v, (int, float)) and v >= 0,
        "explainability": lambda v: isinstance(v, dict) and "features" in v,
    }
}

SAMPLE_PREDICT_SCHEMA = {
    "required_keys": ["status", "predicted_genre", "confidence", "sample_genre"],
    "rules": {
        "status": lambda v: v in ["OK", "UNCERTAIN", "FALLBACK"],
        "confidence": lambda v: 0.0 <= v <= 1.0,
        "sample_genre": lambda v: isinstance(v, str),
    }
}


def _validate_schema(data: dict, schema: dict, label: str) -> list:
    """
    Validate data against schema. Returns list of error messages.
    Prints a table of each field and its validation status.
    """
    errors = []

    print(f"\n  Schema validation — {label}")
    print(f"  {'Field':<25} {'Value Preview':<35} {'Status'}")
    print(f"  {'─'*70}")

    for key in schema["required_keys"]:
        if key not in data:
            print(f"  {'⚠ ' + key:<25} {'[MISSING]':<35} ✗ FAIL")
            errors.append(f"Required key '{key}' missing")
            continue

        val = data[key]
        val_preview = str(val)[:30] + "..." if len(str(val)) > 30 else str(val)

        rule = schema["rules"].get(key)
        if rule:
            ok = rule(val)
            status = "✓ PASS" if ok else "✗ FAIL"
            if not ok:
                errors.append(f"Field '{key}' failed validation: {val!r}")
        else:
            status = "✓ (no rule)"

        print(f"  {key:<25} {val_preview:<35} {status}")

    return errors


class TestAPIContract:

    client = TestClient(app)

    @pytest.fixture(scope="class")
    def sample_wav(self):
        t = np.linspace(0, 8, 128000, endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, y, 16000, format="WAV")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # TC-10-A: GET /health contract
    # ------------------------------------------------------------------
    def test_health_response_schema(self):
        """
        GIVEN  : API server running with model loaded.
        WHEN   : GET /health.
        THEN   : Response contains all required fields with correct types/values.

        Parameter fired:
            Endpoint : GET /health
            Schema   : HEALTH_SCHEMA (7 required fields)
        """
        print(f"\n{'═'*60}")
        print(f"  ▶ REQUEST — GET /health")
        print(f"    Headers: (default)")
        print(f"    Body   : (none)")

        resp = self.client.get("/health")

        print(f"\n  ◀ RESPONSE")
        print(f"    HTTP: {resp.status_code}")
        data = resp.json()
        print(f"    Body (full):")
        print(json.dumps(data, ensure_ascii=False, indent=4))

        assert resp.status_code == 200, (
            f"[FAIL] /health returned {resp.status_code}"
        )

        errors = _validate_schema(data, HEALTH_SCHEMA, "GET /health")

        print(f"{'═'*60}\n")
        assert not errors, (
            f"[FAIL] /health contract violations:\n" +
            "\n".join(f"  • {e}" for e in errors)
        )
        print(f"[PASS] /health response fully complies with API contract")

    # ------------------------------------------------------------------
    # TC-10-B: POST /predict contract
    # ------------------------------------------------------------------
    def test_predict_response_schema(self, sample_wav):
        """
        GIVEN  : Standard 440Hz 8s WAV (128,000 samples, 16kHz).
        WHEN   : POST /predict with multipart WAV file.
        THEN   : Full response schema validated against PREDICT_SCHEMA.
                 Additional invariant: sum(top_predictions.confidence) ≤ 1.0.

        Parameter fired:
            Endpoint     : POST /predict
            File         : synthetic_440hz_8s.wav
            Size         : ~512 KB
            Content-Type : audio/wav (multipart/form-data)
            Schema       : PREDICT_SCHEMA (10 required fields)
        """
        wav_bytes = sample_wav

        print(f"\n{'═'*60}")
        print(f"  ▶ REQUEST — POST /predict")
        print(f"    File : synthetic_440hz_8s.wav ({len(wav_bytes):,} bytes)")
        print(f"    Type : audio/wav (multipart/form-data)")

        resp = self.client.post(
            "/predict",
            files={"file": ("contract_test.wav", wav_bytes, "audio/wav")},
        )

        print(f"\n  ◀ RESPONSE")
        print(f"    HTTP: {resp.status_code}")
        data = resp.json()
        print(f"    Body (full):")
        print(json.dumps(data, ensure_ascii=False, indent=4, default=str))

        assert resp.status_code == 200, (
            f"[FAIL] /predict returned {resp.status_code}"
        )

        errors = _validate_schema(data, PREDICT_SCHEMA, "POST /predict")

        # Additional invariant: probabilities sum check
        top_preds = data.get("top_predictions", [])
        if top_preds:
            total_conf = sum(p["confidence"] for p in top_preds)
            print(f"\n  Confidence sum check:")
            print(f"    top_predictions sum = {total_conf:.6f} (must be ≤ 1.0001)")
            if total_conf > 1.0001:
                errors.append(
                    f"top_predictions confidence sum = {total_conf:.6f} > 1.0"
                )
            else:
                print(f"    ✓ PASS")

        print(f"{'═'*60}\n")
        assert not errors, (
            "[FAIL] /predict contract violations:\n" +
            "\n".join(f"  • {e}" for e in errors)
        )
        print("[PASS] /predict response fully complies with API contract")

    # ------------------------------------------------------------------
    # TC-10-C: POST /sample-predict/{genre} contract
    # ------------------------------------------------------------------
    def test_sample_predict_response_schema(self):
        """
        GIVEN  : Genre 'blues' with real sample files in data/audio/blues/.
        WHEN   : POST /sample-predict/blues.
        THEN   : Response includes sample_genre='blues' and audio_url.
                 Core prediction fields also comply with SAMPLE_PREDICT_SCHEMA.

        Parameter fired:
            Endpoint : POST /sample-predict/blues
            Body     : (none — uses server-side audio sample)
        """
        print(f"\n{'═'*60}")
        print(f"  ▶ REQUEST — POST /sample-predict/blues")
        print(f"    Body : (none — server selects audio sample internally)")

        resp = self.client.post("/sample-predict/blues")

        print(f"\n  ◀ RESPONSE")
        print(f"    HTTP: {resp.status_code}")
        data = resp.json()
        print(f"    Body:")
        print(json.dumps(data, ensure_ascii=False, indent=4, default=str))
        print(f"{'═'*60}\n")

        if resp.status_code == 200:
            errors = _validate_schema(data, SAMPLE_PREDICT_SCHEMA,
                                      "POST /sample-predict/blues")
            # sample_genre must echo back the requested genre
            assert data.get("sample_genre") == "blues", (
                f"[FAIL] sample_genre='{data.get('sample_genre')}' expected 'blues'"
            )
            assert "audio_url" in data, "[FAIL] 'audio_url' missing from response"
            assert not errors, (
                "[FAIL] /sample-predict contract violations:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )
            print("[PASS] /sample-predict/blues response complies with contract")
        elif resp.status_code == 404:
            print("[INFO] /sample-predict/blues returned 404 — no sample files. Skipping schema check.")
        else:
            pytest.fail(f"[FAIL] Unexpected status {resp.status_code}")
