import unittest

from pdf_pipeline.transaction_normalization import TransactionNormalized
from pdf_pipeline.description_normalization import TransactionDescriptionNormalized
from pdf_pipeline.transaction_classification import classify_transaction_types


class Story6TransactionClassificationTests(unittest.TestCase):
    def test_classifies_fee_cash_transfer_debt(self):
        normalized_rows = [
            TransactionNormalized("C300001", "x.pdf", 1, "2026-03-01", "Comision administrare cont", 10.0, 1000.0, "RON", "out", True, "normalized_ok", 0.95, ""),
            TransactionNormalized("C300002", "x.pdf", 1, "2026-03-01", "Retragere numerar ATM", 200.0, 800.0, "RON", "out", True, "normalized_ok", 0.95, ""),
            TransactionNormalized("C300003", "x.pdf", 1, "2026-03-01", "Transfer intern intre conturi", 300.0, 500.0, "RON", "out", True, "normalized_ok", 0.95, ""),
            TransactionNormalized("C300004", "x.pdf", 1, "2026-03-01", "Plata rata credit", 400.0, 100.0, "RON", "out", True, "normalized_ok", 0.95, ""),
        ]

        description_rows = [
            TransactionDescriptionNormalized("C300001", "x.pdf", 1, "COMISION ADMINISTRARE CONT", None, 0.9, "description_ok", ""),
            TransactionDescriptionNormalized("C300002", "x.pdf", 1, "RETRAGERE NUMERAR ATM", None, 0.9, "description_ok", ""),
            TransactionDescriptionNormalized("C300003", "x.pdf", 1, "TRANSFER INTERN INTRE CONTURI", None, 0.9, "description_ok", ""),
            TransactionDescriptionNormalized("C300004", "x.pdf", 1, "PLATA RATA CREDIT", None, 0.9, "description_ok", ""),
        ]

        rows, summary = classify_transaction_types(normalized_rows, description_rows)
        by_id = {row.candidate_id: row for row in rows}

        self.assertEqual(by_id["C300001"].transaction_type, "fee")
        self.assertEqual(by_id["C300002"].transaction_type, "cash_withdrawal")
        self.assertEqual(by_id["C300003"].transaction_type, "transfer")
        self.assertEqual(by_id["C300004"].transaction_type, "debt_payment")
        self.assertEqual(summary["fee"], 1)
        self.assertEqual(summary["cash_withdrawal"], 1)
        self.assertEqual(summary["transfer"], 1)
        self.assertEqual(summary["debt_payment"], 1)

    def test_classifies_income_expense_and_unknown_fallback(self):
        normalized_rows = [
            TransactionNormalized("C300005", "x.pdf", 1, "2026-03-01", "Incasare salariu", 2000.0, 2100.0, "RON", "in", True, "normalized_ok", 0.9, ""),
            TransactionNormalized("C300006", "x.pdf", 1, "2026-03-01", "Cumparaturi diverse", 150.0, 1950.0, "RON", "out", True, "normalized_ok", 0.85, ""),
            TransactionNormalized("C300007", "x.pdf", 1, "2026-03-01", None, 50.0, 1900.0, "RON", None, True, "normalized_ok", 0.8, ""),
        ]

        description_rows = [
            TransactionDescriptionNormalized("C300005", "x.pdf", 1, "INCASARE SALARIU", None, 0.92, "description_ok", ""),
            TransactionDescriptionNormalized("C300006", "x.pdf", 1, "CUMPARATURI DIVERSE", "MEGA IMAGE", 0.87, "description_ok", ""),
            TransactionDescriptionNormalized("C300007", "x.pdf", 1, "", None, 0.2, "description_failed", "missing_description"),
        ]

        rows, summary = classify_transaction_types(normalized_rows, description_rows)
        by_id = {row.candidate_id: row for row in rows}

        self.assertEqual(by_id["C300005"].transaction_type, "income")
        self.assertEqual(by_id["C300006"].transaction_type, "expense")
        self.assertEqual(by_id["C300007"].transaction_type, "unknown")
        self.assertIn("unknown_fallback", by_id["C300007"].classification_reason)
        self.assertEqual(summary["income"], 1)
        self.assertEqual(summary["expense"], 1)
        self.assertEqual(summary["unknown"], 1)


if __name__ == "__main__":
    unittest.main()



