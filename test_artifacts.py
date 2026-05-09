#!/usr/bin/env python
"""Test script to validate model artifacts and run processing"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference.model_artifact_loader import ModelArtifactLoader

def main():
    artifacts_dir = os.path.join(os.path.dirname(__file__), "model_artifacts")

    print("=" * 80)
    print("MODEL ARTIFACT VALIDATION TEST")
    print("=" * 80)
    print(f"\nArtifacts directory: {artifacts_dir}")
    print(f"Directory exists: {os.path.exists(artifacts_dir)}")

    loader = ModelArtifactLoader(artifacts_dir)

    # Check for missing artifacts
    missing = loader.missing_artifacts()
    if missing:
        print(f"❌ MISSING ARTIFACTS: {missing}")
        return 1
    else:
        print("✅ All required artifact files present")

    # Try to load artifacts
    try:
        artifacts = loader.load(require_multitask=True)
        print("✅ Successfully loaded model artifacts")
        print(f"\n  Model type: {artifacts.model_metadata.get('model_type')}")
        print(f"  Feature count: {len(artifacts.feature_columns)}")
        print(f"  Input dim: {artifacts.model_metadata.get('input_dim')}")
        print(f"  Multitask: {artifacts.model_metadata.get('multitask')}")
        print(f"\nFirst 5 features:")
        for i, feat in enumerate(artifacts.feature_columns[:5]):
            print(f"    {i+1}. {feat}")
        return 0
    except Exception as exc:
        print(f"❌ FAILED TO LOAD ARTIFACTS: {exc}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

