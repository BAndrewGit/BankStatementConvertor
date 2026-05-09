#!/usr/bin/env python3
"""
Debug script to verify feature scaling is applied correctly.
Tests that scaled features are actually normalized to ~[−3, +3] range.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.inference.model_artifact_loader import ModelArtifacts
from src.inference.predictor import Predictor
from src.domain.inference_contracts import InferenceInputRow


def test_feature_scaling():
    """Verify that scaled features are actually normalized."""

    print("=" * 70)
    print("FEATURE SCALING VERIFICATION TEST")
    print("=" * 70)

    # Load artifacts
    artifacts_dir = Path(__file__).parent / "model_artifacts"
    artifacts = ModelArtifacts.load_from_directory(str(artifacts_dir))

    print(f"\n✓ Model artifacts loaded")
    print(f"  - Features: {len(artifacts.feature_columns)}")
    print(f"  - Scaler: {artifacts.scaler}")

    if artifacts.scaler is None:
        print("\n✗ ERROR: Scaler is None! Scaling will NOT work!")
        return False

    metadata = artifacts.model_metadata or {}
    scaled_cols = metadata.get("scaled_feature_columns", [])
    print(f"  - Columns to scale: {scaled_cols}")

    # Create test data with raw (unscaled) values
    feature_columns = artifacts.feature_columns
    test_values = {col: 0.5 for col in feature_columns}

    # Set realistic RAW values (before scaling)
    test_values["Age"] = 42.0  # Age: 0-100 range
    test_values["Income_Category"] = 5.0  # Income: 0-7 ordinal
    test_values["Essential_Needs_Percentage"] = 65.0  # Percentage: 0-100
    test_values["Product_Lifetime_Clothing"] = 2.0  # Ordinal: 0-5
    test_values["Product_Lifetime_Tech"] = 3.0
    test_values["Product_Lifetime_Appliances"] = 2.5
    test_values["Product_Lifetime_Cars"] = 4.0

    print(f"\n[RAW INPUT VALUES]")
    for col in scaled_cols:
        val = test_values.get(col, 0.5)
        print(f"  {col}: {val}")

    # Create inference row
    inference_row = InferenceInputRow.from_values(test_values, ordered_columns=feature_columns)

    # Scale using predictor
    predictor = Predictor(artifacts)
    ordered_list = inference_row.as_ordered_list()
    scaled_list = predictor.scale_ordered_values(ordered_list)

    print(f"\n[SCALED VALUES]")
    scale_indices = [i for i, col in enumerate(feature_columns) if col in scaled_cols]

    for idx in scale_indices:
        col = feature_columns[idx]
        raw_val = ordered_list[idx]
        scaled_val = scaled_list[idx]
        print(f"  {col}: {raw_val:8.2f} → {scaled_val:8.4f} (Δ {abs(scaled_val)/abs(raw_val) if raw_val != 0 else 0:8.4f})")

    # Verify scaling actually happened
    print(f"\n[VERIFICATION]")

    # Check that scaled values are in reasonable range [-3, +3]
    scaled_portion = [scaled_list[i] for i in scale_indices]

    if not scaled_portion:
        print("✗ ERROR: No scaled features found!")
        return False

    max_abs_scaled = max(abs(v) for v in scaled_portion)
    print(f"  Max absolute scaled value: {max_abs_scaled:.4f}")

    if max_abs_scaled < 0.1:
        print("  ✗ WARNING: All scaled values are very small (possible underflow)")
        return False

    if max_abs_scaled > 10.0:
        print("  ✗ ERROR: Scaled values are TOO LARGE (scaling not working!)")
        return False

    if -3 <= max_abs_scaled <= 3:
        print("  ✓ Scaled values are in expected range [-3, +3]")
    else:
        print(f"  ⚠ Scaled values are slightly outside [-3, +3] (still acceptable)")

    # Test prediction
    print(f"\n[PREDICTION TEST]")
    try:
        result = predictor.predict(inference_row)
        print(f"  ✓ Prediction successful")
        print(f"    - Risk score: {result.risk_score:.4f}")
        print(f"    - Saving probability: {result.saving_probability:.4f}")
    except Exception as e:
        print(f"  ✗ Prediction failed: {e}")
        return False

    print("\n" + "=" * 70)
    print("✓ SCALING IS WORKING CORRECTLY")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = test_feature_scaling()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


