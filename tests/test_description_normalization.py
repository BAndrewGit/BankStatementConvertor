import unittest

from pdf_pipeline.transaction_normalization import TransactionNormalized
from pdf_pipeline.description_normalization import normalize_transaction_descriptions


class Story5DescriptionNormalizedTests(unittest.TestCase):
    def test_cleans_description_and_extracts_merchant(self):
        rows = [
            TransactionNormalized(
                candidate_id="C200001",
                source_pdf="extras.pdf",
                source_page=1,
                transaction_date="2026-02-01",
                description="Plată la POS WWW.emag.ro TID:ABC1234 eMAG Marketplace",
                amount=123.45,
                balance=1200.0,
                currency="RON",
                direction="out",
                is_valid=True,
                normalization_status="normalized_ok",
                normalization_confidence=0.92,
                invalid_reasons="",
            )
        ]

        normalized_rows, issues, summary = normalize_transaction_descriptions(rows)
        self.assertEqual(len(normalized_rows), 1)
        self.assertEqual(len(issues), 0)

        row = normalized_rows[0]
        self.assertIn("EMAG", row.description_clean)
        self.assertEqual(row.merchant_raw_candidate, "EMAG MARKETPLACE")
        self.assertEqual(row.description_status, "description_ok")
        self.assertGreaterEqual(row.description_normalization_confidence, 0.8)
        self.assertEqual(summary["rows_with_merchant"], 1)

    def test_does_not_force_merchant_for_transfer_or_cash(self):
        rows = [
            TransactionNormalized(
                candidate_id="C200002",
                source_pdf="extras.pdf",
                source_page=2,
                transaction_date="2026-02-02",
                description="Transfer intern canal electronic",
                amount=500.0,
                balance=700.0,
                currency="RON",
                direction="out",
                is_valid=True,
                normalization_status="normalized_ok",
                normalization_confidence=0.9,
                invalid_reasons="",
            ),
            TransactionNormalized(
                candidate_id="C200003",
                source_pdf="extras.pdf",
                source_page=2,
                transaction_date="2026-02-02",
                description="Retragere numerar ATM 1234",
                amount=200.0,
                balance=500.0,
                currency="RON",
                direction="out",
                is_valid=True,
                normalization_status="normalized_ok",
                normalization_confidence=0.9,
                invalid_reasons="",
            ),
        ]

        normalized_rows, _, summary = normalize_transaction_descriptions(rows)

        self.assertIsNone(normalized_rows[0].merchant_raw_candidate)
        self.assertIn("non_merchant_transaction", normalized_rows[0].description_warnings)
        self.assertIsNone(normalized_rows[1].merchant_raw_candidate)
        self.assertIn("non_merchant_transaction", normalized_rows[1].description_warnings)
        self.assertEqual(summary["rows_with_merchant"], 0)

    def test_missing_description_is_flagged(self):
        rows = [
            TransactionNormalized(
                candidate_id="C200004",
                source_pdf="extras.pdf",
                source_page=3,
                transaction_date="2026-02-03",
                description=None,
                amount=20.0,
                balance=100.0,
                currency="RON",
                direction="out",
                is_valid=True,
                normalization_status="normalized_ok",
                normalization_confidence=0.85,
                invalid_reasons="",
            )
        ]

        with self.assertLogs("pdf_pipeline.description_normalization", level="WARNING"):
            normalized_rows, issues, summary = normalize_transaction_descriptions(rows)

        self.assertEqual(normalized_rows[0].description_status, "description_failed")
        self.assertEqual(normalized_rows[0].description_clean, "")
        self.assertIsNone(normalized_rows[0].merchant_raw_candidate)
        self.assertGreaterEqual(len(issues), 1)
        self.assertEqual(summary["rows_without_merchant"], 1)


if __name__ == "__main__":
    unittest.main()

