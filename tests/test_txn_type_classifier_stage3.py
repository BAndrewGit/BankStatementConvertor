import unittest

from src.classification.txn_type_classifier import classify_transactions
from src.domain.models import Transaction


class TransactionTypeClassifierStage3Tests(unittest.TestCase):
    def test_strict_order_rules(self):
        rows = [
            Transaction("1", "2026-03-01", 50.0, "RON", "debit", "EPOS MARKET", "blocked_amounts"),
            Transaction("2", "2026-03-01", 200.0, "RON", "debit", "Retragere de numerar ATM BT", "booked_transactions"),
            Transaction("3", "2026-03-01", 3.0, "RON", "debit", "Taxa Serviciu SMS", "booked_transactions"),
            Transaction("4", "2026-03-01", 100.0, "RON", "debit", "Transfer BT PAY catre cont propriu", "booked_transactions"),
            Transaction("5", "2026-03-01", 600.0, "RON", "debit", "Plata Instant catre beneficiar extern", "booked_transactions"),
            Transaction("6", "2026-03-01", 40.0, "RON", "debit", "POS MEGA IMAGE", "booked_transactions"),
            Transaction("7", "2026-03-01", 10.0, "RON", "debit", "Descriere neclasificata", "booked_transactions"),
        ]

        classified, summary = classify_transactions(rows)

        self.assertEqual(classified[0].txn_type, "blocked_amount")
        self.assertEqual(classified[1].txn_type, "cash_withdrawal")
        self.assertEqual(classified[2].txn_type, "bank_fee")
        self.assertEqual(classified[3].txn_type, "internal_transfer")
        self.assertEqual(classified[4].txn_type, "external_transfer")
        self.assertEqual(classified[5].txn_type, "card_purchase")
        self.assertEqual(classified[6].txn_type, "unknown")

        self.assertAlmostEqual(summary["valid_rate"], 6 / 7, places=4)


if __name__ == "__main__":
    unittest.main()

