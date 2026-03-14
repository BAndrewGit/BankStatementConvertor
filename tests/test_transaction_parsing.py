import unittest

from pdf_pipeline.transaction_candidates import TransactionCandidate
from pdf_pipeline.transaction_parsing import parse_transactions_raw


class Story3TransactionsRawTests(unittest.TestCase):
    def test_parse_clean_transaction_row(self):
        candidates = [
            TransactionCandidate(
                candidate_id="C000001",
                source_pdf="extras.pdf",
                start_page=2,
                end_page=2,
                start_line=10,
                end_line=11,
                text="01/02/2026 Plata la POS Mega Image 123,45 RON Sold final 2.500,50 RON",
                has_date=True,
                amount_count=2,
                is_ambiguous=False,
                ambiguity_reasons="",
            )
        ]

        rows = parse_transactions_raw(candidates)
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row.transaction_date_raw, "01/02/2026")
        self.assertEqual(row.amount_raw, "123,45")
        self.assertEqual(row.balance_raw, "2.500,50")
        self.assertEqual(row.currency_raw, "RON")
        self.assertEqual(row.transaction_direction_raw, "outflow")
        self.assertEqual(row.parse_status, "parsed_ok")
        self.assertGreaterEqual(row.parse_confidence, 0.75)

    def test_parse_with_missing_amount_marks_warning(self):
        candidates = [
            TransactionCandidate(
                candidate_id="C000002",
                source_pdf="extras.pdf",
                start_page=3,
                end_page=3,
                start_line=2,
                end_line=2,
                text="05/02/2026 Transfer intern canal electronic",
                has_date=True,
                amount_count=0,
                is_ambiguous=True,
                ambiguity_reasons="missing_amount",
            )
        ]

        row = parse_transactions_raw(candidates)[0]
        self.assertIsNone(row.amount_raw)
        self.assertEqual(row.parse_status, "parsed_with_warnings")
        self.assertLess(row.parse_confidence, 0.75)
        self.assertIn("missing_amount", row.parse_warnings)

    def test_parse_keeps_source_link_fields(self):
        candidates = [
            TransactionCandidate(
                candidate_id="C000003",
                source_pdf="extras.pdf",
                start_page=4,
                end_page=5,
                start_line=7,
                end_line=1,
                text="10/02/2026 Incasare client +1.000,00 EUR",
                has_date=True,
                amount_count=1,
                is_ambiguous=False,
                ambiguity_reasons="",
            )
        ]

        row = parse_transactions_raw(candidates)[0]
        self.assertEqual(row.candidate_id, "C000003")
        self.assertEqual(row.source_pdf, "extras.pdf")
        self.assertEqual(row.source_page, 4)
        self.assertEqual(row.source_page_span, "4-5")
        self.assertEqual(row.transaction_direction_raw, "inflow")
        self.assertEqual(row.currency_raw, "EUR")


if __name__ == "__main__":
    unittest.main()



