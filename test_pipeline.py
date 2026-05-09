#!/usr/bin/env python
"""Test the end-to-end pipeline with model processing"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipelines.run_end_to_end import run_end_to_end
from src.infrastructure.cache import FileCacheRepository

def test_with_sample_pdf():
    """Test processing a sample PDF if available"""
    # Check if there's a test PDF
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        print("❌ Data directory not found")
        return False

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print("ℹ️  No test PDFs found in data directory")
        return True  # Not an error, just can't test

    # Test with first PDF
    test_pdf = os.path.join(data_dir, pdf_files[0])
    output_dir = os.path.join(os.path.dirname(__file__), ".tmp_test_output")
    artifacts_dir = os.path.join(os.path.dirname(__file__), "model_artifacts")

    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"\nTesting with PDF: {pdf_files[0]}")
        result = run_end_to_end(
            pdf_path=test_pdf,
            export_dir=output_dir,
            artifacts_dir=artifacts_dir,
            cache_repo=FileCacheRepository()
        )
        print("✅ Processing completed successfully!")
        print(f"   Output directory: {output_dir}")
        print(f"   Final dataset: {result.final_dataset_csv_path}")

        # Check if Risk_Score was computed
        import csv
        with open(result.final_dataset_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row and 'Risk_Score' in row:
                print(f"   Risk_Score computed: {row['Risk_Score']}")
        return True
    except Exception as exc:
        print(f"❌ Processing failed: {exc}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_sample_pdf()
    sys.exit(0 if success else 1)

