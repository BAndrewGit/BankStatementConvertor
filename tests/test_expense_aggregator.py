import unittest

from src.domain.models import Transaction
from src.features.expense_aggregator import aggregate_expenses


class ExpenseAggregatorTests(unittest.TestCase):
    def test_aggregates_only_relevant_debit_transactions(self):
        rows = [
            Transaction("1", "2026-03-01", 100.0, "RON", "debit", "x", "booked_transactions", txn_type="card_purchase", category_area="Food"),
            Transaction("2", "2026-03-01", 200.0, "RON", "credit", "x", "booked_transactions", txn_type="card_purchase", category_area="Food"),
            Transaction("3", "2026-03-01", 30.0, "RON", "debit", "x", "booked_transactions", txn_type="bank_fee", category_area="Other"),
            Transaction("4", "2026-03-01", 50.0, "RON", "debit", "x", "booked_transactions", txn_type="internal_transfer", category_area="Other"),
            Transaction("5", "2026-03-01", 80.0, "RON", "debit", "x", "booked_transactions", txn_type="card_purchase", category_area="Housing"),
            Transaction("6", "2026-03-01", 40.0, "RON", "debit", "x", "blocked_amounts", txn_type="blocked_amount", category_area="Unknown"),
            Transaction("7", "2026-03-01", 400.0, "RON", "debit", "x", "booked_transactions", txn_type="external_transfer", category_area="Unknown"),
        ]

        totals = aggregate_expenses(rows)
        self.assertEqual(totals["food_total"], 100.0)
        self.assertEqual(totals["housing_total"], 80.0)
        self.assertEqual(totals["other_total"], 0.0)
        self.assertEqual(totals["unknown_total"], 0.0)


if __name__ == "__main__":
    unittest.main()

