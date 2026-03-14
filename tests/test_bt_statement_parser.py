import unittest
from unittest.mock import patch

from src.ingestion.bt_statement_parser import BTStatementParser


class BTStatementParserTests(unittest.TestCase):
    def test_parser_splits_booked_and_blocked_sections(self):
        segmented_lines = [
            [
                "TRANZACTII EFECTUATE",
                "01/03/2026 POS MEGA IMAGE 123,45 RON",
                "02/03/2026 Retragere de numerar ATM BT 250,00 RON",
                "SUME BLOCATE",
                "03/03/2026 EPOS STORE ONLINE 89,99 RON",
            ]
        ]

        parser = BTStatementParser()
        with patch("src.ingestion.bt_statement_parser.read_pdf_pages", return_value=[""]):
            with patch("src.ingestion.bt_statement_parser.segment_pages_to_lines", return_value=segmented_lines):
                rows = parser.parse("dummy.pdf")

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].source_section, "booked_transactions")
        self.assertEqual(rows[1].source_section, "booked_transactions")
        self.assertEqual(rows[2].source_section, "blocked_amounts")

    def test_parser_extracts_debit_credit_and_is_deterministic(self):
        segmented_lines = [
            [
                "TRANZACTII EFECTUATE",
                "04/03/2026 Incasare salariu +3500,00 RON",
                "05/03/2026 Plata utilitati 210,50 RON",
            ]
        ]

        parser = BTStatementParser()
        with patch("src.ingestion.bt_statement_parser.read_pdf_pages", return_value=[""]):
            with patch("src.ingestion.bt_statement_parser.segment_pages_to_lines", return_value=segmented_lines):
                first = parser.parse("dummy.pdf")
        with patch("src.ingestion.bt_statement_parser.read_pdf_pages", return_value=[""]):
            with patch("src.ingestion.bt_statement_parser.segment_pages_to_lines", return_value=segmented_lines):
                second = parser.parse("dummy.pdf")

        self.assertEqual(first[0].direction, "credit")
        self.assertEqual(first[0].amount, 3500.0)
        self.assertEqual(first[1].direction, "debit")
        self.assertEqual(first[1].amount, 210.5)

        self.assertEqual([row.transaction_id for row in first], [row.transaction_id for row in second])

    def test_parser_infers_booked_section_without_explicit_header(self):
        segmented_lines = [
            [
                "Data Descriere Debit Credit",
                "02/02/2026 Plata la POS non-BT cu card MASTERCARD 36.00",
                "POS 29/01/2026 TID:21ESS309 PRANZOBYESS@FBP C3",
                "02/02/2026 RULAJ ZI 36.00 0.00",
                "SOLD FINAL ZI 503.69",
                "03/02/2026 Transfer intern - canal electronic 500.00",
            ]
        ]

        parser = BTStatementParser()
        with patch("src.ingestion.bt_statement_parser.read_pdf_pages", return_value=[""]):
            with patch("src.ingestion.bt_statement_parser.segment_pages_to_lines", return_value=segmented_lines):
                rows = parser.parse("dummy.pdf")

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.source_section == "booked_transactions" for row in rows))

    def test_parser_extracts_blocked_entries_with_dash_prefix(self):
        segmented_lines = [
            [
                "SUME BLOCATE",
                "- 78.50 RON aferenta tranzactiei POS 28/02/2026 330000000015134 TID:33015134 FARMACIA",
                "valoare tranzactie: 78.50 RON",
            ]
        ]

        parser = BTStatementParser()
        with patch("src.ingestion.bt_statement_parser.read_pdf_pages", return_value=[""]):
            with patch("src.ingestion.bt_statement_parser.segment_pages_to_lines", return_value=segmented_lines):
                rows = parser.parse("dummy.pdf")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source_section, "blocked_amounts")

    def test_parser_splits_multiple_same_day_booked_boundaries(self):
        segmented_lines = [
            [
                "Data Descriere Debit Credit",
                "02/02/2026 Plata la POS non-BT cu card MASTERCARD 36.00",
                "POS 29/01/2026 TID:21ESS309 PRANZOBYESS@FBP C3",
                "Transfer intern - canal electronic 500.00",
                "Transfer BT PAY; REF: 217ECIT260330508",
                "P2P BTPay 2,900.00",
            ]
        ]

        parser = BTStatementParser()
        with patch("src.ingestion.bt_statement_parser.read_pdf_pages", return_value=[""]):
            with patch("src.ingestion.bt_statement_parser.segment_pages_to_lines", return_value=segmented_lines):
                rows = parser.parse("dummy.pdf")

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row.booking_date == "02/02/2026" for row in rows))
        self.assertTrue(all(row.source_section == "booked_transactions" for row in rows))

    def test_parser_parses_thousands_amounts_correctly(self):
        segmented_lines = [
            [
                "Data Descriere Debit Credit",
                "13/02/2026 Transfer intern - canal electronic 1,000.00",
                "13/02/2026 P2P BTPay 2,900.00",
            ]
        ]

        parser = BTStatementParser()
        with patch("src.ingestion.bt_statement_parser.read_pdf_pages", return_value=[""]):
            with patch("src.ingestion.bt_statement_parser.segment_pages_to_lines", return_value=segmented_lines):
                rows = parser.parse("dummy.pdf")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].amount, 1000.0)
        self.assertEqual(rows[1].amount, 2900.0)


if __name__ == "__main__":
    unittest.main()


