import json
import csv
import os
import tempfile
import unittest
from unittest.mock import patch

from src.domain.models import Transaction
from src.features.feature_builder import FEATURE_COLUMNS, FINAL_DATASET_COLUMNS
from src.infrastructure.cache import InMemoryCacheRepository
from src.pipelines.run_end_to_end import run_end_to_end, run_end_to_end_many


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
                            profile_answers={"Gender_Male": 1.0, "Income_Category": 9999.0},
                        )

            self.assertTrue(os.path.exists(result.run_report_path))
            self.assertTrue(os.path.exists(result.transactions_csv_path))
            self.assertTrue(os.path.exists(result.final_dataset_csv_path))

            with open(result.run_report_path, encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertIn("quality_metrics", payload)
            self.assertIn("per_file_traceability", payload)
            self.assertEqual(len(payload["per_file_traceability"]), 1)
            self.assertEqual(payload["per_file_traceability"][0]["pdf_path"], pdf_path)

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
            self.assertIn("salary_income_detected", payload["run_summary"])
            self.assertIn("salary_income_count", payload["run_summary"])
            self.assertIn("salary_income_total", payload["run_summary"])
            self.assertIn("entity_memory_backfill_hits", payload["run_summary"])
            self.assertIn("backfilled_transactions", payload["run_summary"])
            self.assertIn("save_money_computed", payload["run_summary"])
            self.assertIn("impulse_candidates_count", payload["run_summary"])
            self.assertIn("impulse_frequency", payload["run_summary"])
            self.assertIn("impulse_category_counts", payload["run_summary"])
            self.assertIn("housing_detected_amount", payload["run_summary"])
            self.assertIn("unknown_transfer_count", payload["run_summary"])
            self.assertIn("other_spend_share", payload["run_summary"])
            self.assertIn("salary_income", payload)
            self.assertIn("profile_id", payload)
            self.assertIn("local_storage", payload)
            self.assertIn("bootstrap_dictionary_path", payload["local_storage"])
            self.assertIn("entity_memory_path", payload["local_storage"])

            with open(result.final_dataset_csv_path, encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames
                row = next(reader)
            self.assertIsNotNone(header)
            header_columns = list(header or [])
            self.assertEqual(header_columns, FINAL_DATASET_COLUMNS)
            self.assertIn("Income_Category", header_columns)
            self.assertEqual(row["Income_Category"], "0.0")
            self.assertEqual(row["Gender_Male"], "1.0")

    def test_batch_run_writes_monthly_dataset_for_multiple_pdfs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path_1 = os.path.join(temp_dir, "fixed_1.pdf")
            pdf_path_2 = os.path.join(temp_dir, "fixed_2.pdf")
            for path in (pdf_path_1, pdf_path_2):
                with open(path, "wb") as handle:
                    handle.write(b"%PDF-1.4\nreport-test\n")

            parsed_first = [
                Transaction(
                    transaction_id="1",
                    booking_date="05/02/2026",
                    amount=100.0,
                    currency="RON",
                    direction="debit",
                    raw_description="POS MEGAIMAGE",
                    source_section="booked_transactions",
                )
            ]
            parsed_second = [
                Transaction(
                    transaction_id="2",
                    booking_date="04/03/2026",
                    amount=50.0,
                    currency="RON",
                    direction="debit",
                    raw_description="POS CARREFOUR",
                    source_section="booked_transactions",
                )
            ]
            classified_first = [
                Transaction(
                    transaction_id="1",
                    booking_date="05/02/2026",
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
            classified_second = [
                Transaction(
                    transaction_id="2",
                    booking_date="04/03/2026",
                    amount=50.0,
                    currency="RON",
                    direction="debit",
                    raw_description="POS CARREFOUR",
                    source_section="booked_transactions",
                    txn_type="card_purchase",
                    merchant_raw="CARREFOUR",
                    merchant_canonical="Carrefour",
                    category_area="Food",
                    mapping_method="exact_alias",
                    confidence=1.0,
                )
            ]

            with patch(
                "src.pipelines.run_end_to_end.parse_statement",
                side_effect=[parsed_first, parsed_second],
            ):
                with patch(
                    "src.pipelines.run_end_to_end.classify_parsed_transactions",
                    side_effect=[
                        (classified_first, {"total": 1.0, "valid": 1.0}),
                        (classified_second, {"total": 1.0, "valid": 1.0}),
                    ],
                ):
                    result = run_end_to_end_many(
                        pdf_paths=[pdf_path_1, pdf_path_2],
                        export_dir=temp_dir,
                        cache_repo=InMemoryCacheRepository(),
                        profile_answers={"Gender_Female": 1.0, "Income_Category": 123.0},
                    )
            self.assertTrue(os.path.exists(result.final_dataset_csv_path))
            self.assertEqual(set(result.monthly_features.keys()), {"2026-02", "2026-03"})

            with open(result.final_dataset_csv_path, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["statement_month"] for row in rows], ["2026-02", "2026-03"])
            self.assertEqual([row["Gender_Female"] for row in rows], ["1.0", "1.0"])
            self.assertEqual([row["Income_Category"] for row in rows], ["0.0", "0.0"])

            with open(result.run_report_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertIn("pdf_paths", payload)
            self.assertIn("per_file_traceability", payload)
            self.assertEqual(len(payload["per_file_traceability"]), 2)
            self.assertEqual(payload["per_file_traceability"][0]["pdf_path"], pdf_path_1)
            self.assertEqual(payload["per_file_traceability"][1]["pdf_path"], pdf_path_2)
            self.assertEqual(len(payload["pdf_paths"]), 2)
            self.assertNotIn("final_dataset_monthly", payload["output_files"])

    def test_batch_run_deduplicates_mirrored_transfer_between_accounts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path_1 = os.path.join(temp_dir, "account_a.pdf")
            pdf_path_2 = os.path.join(temp_dir, "account_b.pdf")
            for path in (pdf_path_1, pdf_path_2):
                with open(path, "wb") as handle:
                    handle.write(b"%PDF-1.4\ndedupe-test\n")

            parsed_first = [
                Transaction(
                    transaction_id="1",
                    booking_date="10/02/2026",
                    amount=500.0,
                    currency="RON",
                    direction="debit",
                    raw_description="Transfer BT PAY REF: ABC123",
                    source_section="booked_transactions",
                )
            ]
            parsed_second = [
                Transaction(
                    transaction_id="2",
                    booking_date="10/02/2026",
                    amount=500.0,
                    currency="RON",
                    direction="credit",
                    raw_description="Transfer BT PAY REF: ABC123",
                    source_section="booked_transactions",
                )
            ]
            classified_first = [
                Transaction(
                    transaction_id="1",
                    booking_date="10/02/2026",
                    amount=500.0,
                    currency="RON",
                    direction="debit",
                    raw_description="Transfer BT PAY REF: ABC123",
                    source_section="booked_transactions",
                    txn_type="internal_transfer",
                    confidence=0.94,
                )
            ]
            classified_second = [
                Transaction(
                    transaction_id="2",
                    booking_date="10/02/2026",
                    amount=500.0,
                    currency="RON",
                    direction="credit",
                    raw_description="Transfer BT PAY REF: ABC123",
                    source_section="booked_transactions",
                    txn_type="internal_transfer",
                    confidence=0.94,
                )
            ]

            with patch(
                "src.pipelines.run_end_to_end.parse_statement",
                side_effect=[parsed_first, parsed_second],
            ):
                with patch(
                    "src.pipelines.run_end_to_end.classify_parsed_transactions",
                    side_effect=[
                        (classified_first, {"total": 1.0, "valid": 1.0}),
                        (classified_second, {"total": 1.0, "valid": 1.0}),
                    ],
                ):
                    result = run_end_to_end_many(
                        pdf_paths=[pdf_path_1, pdf_path_2],
                        export_dir=temp_dir,
                        cache_repo=InMemoryCacheRepository(),
                    )

            with open(result.transactions_csv_path, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["direction"], "credit")

            with open(result.run_report_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["deduplicated_transactions_count"], 1)
            self.assertEqual(payload["deduplicated_transaction_ids"], ["1"])

    def test_batch_income_category_uses_salary_total(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "salary.pdf")
            with open(pdf_path, "wb") as handle:
                handle.write(b"%PDF-1.4\nsalary-test\n")

            parsed = [
                Transaction(
                    transaction_id="salary-1",
                    booking_date="05/02/2026",
                    amount=5773.0,
                    currency="RON",
                    direction="credit",
                    raw_description="Incasare OP SALARIU REF: XYZ",
                    source_section="booked_transactions",
                )
            ]
            classified = [
                Transaction(
                    transaction_id="salary-1",
                    booking_date="05/02/2026",
                    amount=5773.0,
                    currency="RON",
                    direction="credit",
                    raw_description="Incasare OP SALARIU REF: XYZ",
                    source_section="booked_transactions",
                    channel="TRANSFER",
                    txn_type="salary_income",
                    confidence=0.98,
                )
            ]

            with patch("src.pipelines.run_end_to_end.parse_statement", return_value=parsed):
                with patch(
                    "src.pipelines.run_end_to_end.classify_parsed_transactions",
                    return_value=(classified, {"total": 1.0, "valid": 1.0}),
                ):
                    result = run_end_to_end_many(
                        pdf_paths=[pdf_path],
                        export_dir=temp_dir,
                        cache_repo=InMemoryCacheRepository(),
                        profile_answers={"Income_Category": 1.0, "Gender_Male": 1.0},
                    )

            with open(result.final_dataset_csv_path, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["statement_month"], "2026-02")
            self.assertEqual(float(rows[0]["Income_Category"]), 5773.0)
            self.assertEqual(float(rows[0]["Gender_Male"]), 1.0)


if __name__ == "__main__":
    unittest.main()

