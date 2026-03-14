import unittest

from pdf_pipeline.transaction_parsing import TransactionRaw
from pdf_pipeline.transaction_normalization import normalize_transactions_raw


class Story4TransactionsNormalizedTests(unittest.TestCase):
    def test_normalizes_supported_formats(self):
        raw_rows = [
            TransactionRaw(
                candidate_id="C100001",
                source_pdf="extras.pdf",
                source_page=1,
                source_page_span="1-1",
                source_text="01/02/2026 Plata la POS 1.234,56 RON Sold 4.500,00",
                transaction_date_raw="01/02/2026",
                description_raw="Plata la POS",
                amount_raw="1.234,56",
                transaction_direction_raw="outflow",
                balance_raw="4.500,00",
                currency_raw="ron",
                parse_status="parsed_ok",
                parse_confidence=0.9,
                parse_warnings="",
            )
        ]

        normalized_rows, issues, summary = normalize_transactions_raw(raw_rows)
        self.assertEqual(len(normalized_rows), 1)
        self.assertEqual(len(issues), 0)

        row = normalized_rows[0]
        self.assertEqual(row.transaction_date, "2026-02-01")
        self.assertEqual(row.amount, 1234.56)
        self.assertEqual(row.balance, 4500.0)
        self.assertEqual(row.currency, "RON")
        self.assertEqual(row.direction, "out")
        self.assertTrue(row.is_valid)
        self.assertEqual(row.normalization_status, "normalized_ok")
        self.assertEqual(summary["valid_rows"], 1)

    def test_marks_missing_and_invalid_values(self):
        raw_rows = [
            TransactionRaw(
                candidate_id="C100002",
                source_pdf="extras.pdf",
                source_page=2,
                source_page_span="2-2",
                source_text="not parseable",
                transaction_date_raw="2026-02-01",
                description_raw="X",
                amount_raw="12,AB",
                transaction_direction_raw="unknown",
                balance_raw="invalid",
                currency_raw=None,
                parse_status="parsed_with_warnings",
                parse_confidence=0.3,
                parse_warnings="",
            )
        ]

        with self.assertLogs("pdf_pipeline.transaction_normalization", level="WARNING") as log_capture:
            normalized_rows, issues, summary = normalize_transactions_raw(raw_rows)

        row = normalized_rows[0]
        self.assertFalse(row.is_valid)
        self.assertIsNone(row.transaction_date)
        self.assertIsNone(row.amount)
        self.assertIsNone(row.currency)
        self.assertIsNone(row.direction)
        self.assertEqual(row.normalization_status, "normalized_with_issues")
        self.assertGreaterEqual(len(issues), 4)
        self.assertEqual(summary["rows_with_issues"], 1)
        self.assertGreaterEqual(len(log_capture.output), 4)

    def test_standardizes_direction_to_in(self):
        raw_rows = [
            TransactionRaw(
                candidate_id="C100003",
                source_pdf="extras.pdf",
                source_page=3,
                source_page_span="3-3",
                source_text="incasare",
                transaction_date_raw="10/03/2026",
                description_raw="Incasare client",
                amount_raw="+100,00",
                transaction_direction_raw="inflow",
                balance_raw=None,
                currency_raw="EUR",
                parse_status="parsed_ok",
                parse_confidence=0.8,
                parse_warnings="",
            )
        ]

        normalized_rows, issues, _ = normalize_transactions_raw(raw_rows)
        self.assertEqual(normalized_rows[0].direction, "in")
        self.assertEqual(len(issues), 0)


if __name__ == "__main__":
    unittest.main()

