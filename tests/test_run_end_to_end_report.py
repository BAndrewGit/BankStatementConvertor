import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.domain.models import Transaction
from src.features.feature_builder import FEATURE_COLUMNS
from src.infrastructure.cache import InMemoryCacheRepository
from src.pipelines.run_end_to_end import run_end_to_end


class RunEndToEndReportTests(unittest.TestCase):
    def test_each_run_writes_run_report_json_with_required_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "fixed.pdf")
            with open(pdf_path, "wb") as handle:
                handle.write(b"%PDF-1.4\nreport-test\n")

            parsed = [
                Transaction(
                    transaction_id="1",
                    booking_date="2026-03-01",
                    amount=100.0,
                    currency="RON",
                    direction="debit",
                    raw_description="POS MEGAIMAGE",
                    source_section="booked_transactions",
                )
            ]
            classified = [
                Transaction(
                    transaction_id="1",
                    booking_date="2026-03-01",
                    amount=100.0,
                    currency="RON",
                    direction="debit",
                    raw_description="POS MEGAIMAGE",
                    source_section="booked_transactions",
                    txn_type="card_purchase",
                    merchant_raw="MEGAIMAGE",
                    merchant_canonical="Mega Image",
                    category_area="Food",
                    mapping_method="exact_alias",
                    confidence=1.0,
                )
            ]

            with patch("src.pipelines.run_end_to_end.parse_statement", return_value=parsed):
                with patch("src.pipelines.run_end_to_end.classify_parsed_transactions", return_value=(classified, {"valid_rate": 1.0})):
                    with patch(
                        "src.pipelines.run_end_to_end.build_features",
                        return_value={column: 0.0 for column in FEATURE_COLUMNS},
                    ):
                        result = run_end_to_end(
                            pdf_path=pdf_path,
                            export_dir=temp_dir,
                            cache_repo=InMemoryCacheRepository(),
                        )

            self.assertTrue(os.path.exists(result.run_report_path))
            self.assertTrue(os.path.exists(result.transactions_csv_path))
            self.assertTrue(os.path.exists(result.final_dataset_csv_path))

            with open(result.run_report_path, encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertIn("quality_metrics", payload)
            required = {
                "pdf_parse_latency_ms",
                "transactions_extracted_count",
                "booked_transactions_count",
                "blocked_transactions_count",
                "merchant_extracted_count",
                "merchant_unknown_count",
                "category_unknown_count",
                "unknown_expense_percentage",
                "end_to_end_latency_ms",
            }
            self.assertTrue(required.issubset(set(payload["quality_metrics"].keys())))
            self.assertNotIn("classification_quality_score", payload["quality_metrics"])
            self.assertEqual(payload["run_summary"]["transaction_count"], 1)
            self.assertIn("unknown_count", payload["run_summary"])
            self.assertIn("runtime_ms", payload["run_summary"])

            with open(result.final_dataset_csv_path, encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
            self.assertEqual(header, FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()

