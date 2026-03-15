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

    def test_tracks_electronics_gadgets_total_from_canonical_merchants(self):
        rows = [
            Transaction(
                "1",
                "2026-03-01",
                799.0,
                "RON",
                "debit",
                "x",
                "booked_transactions",
                txn_type="card_purchase",
                merchant_canonical="eMAG",
                category_area="Other",
            ),
            Transaction(
                "2",
                "2026-03-01",
                149.0,
                "RON",
                "debit",
                "x",
                "booked_transactions",
                txn_type="card_purchase",
                merchant_canonical="Altex",
                category_area="Other",
            ),
        ]

        totals = aggregate_expenses(rows)
        self.assertEqual(totals["electronics_gadgets_total"], 948.0)

    def test_transfer_and_income_signals_support_save_money_logic(self):
        rows = [
            Transaction(
                "1",
                "2026-03-01",
                2500.0,
                "RON",
                "credit",
                "SALARIU",
                "booked_transactions",
                txn_type="salary_income",
            ),
            Transaction(
                "2",
                "2026-03-01",
                500.0,
                "RON",
                "debit",
                "Transfer BT PAY",
                "booked_transactions",
                txn_type="internal_transfer",
                state="excluded",
            ),
            Transaction(
                "3",
                "2026-03-01",
                600.0,
                "RON",
                "debit",
                "Plata Instant chirie",
                "booked_transactions",
                txn_type="external_transfer",
                category_area="Housing",
                state="confirmed",
            ),
        ]

        totals = aggregate_expenses(rows)
        self.assertEqual(totals["income_total"], 2500.0)
        self.assertEqual(totals["outgoing_expense_total"], 600.0)
        self.assertEqual(totals["housing_total"], 600.0)

    def test_impulse_candidate_frequency_inputs_are_computed(self):
        rows = [
            Transaction(
                "1",
                "2026-03-01",
                35.0,
                "RON",
                "debit",
                "coffee",
                "booked_transactions",
                txn_type="card_purchase",
                category_area="Food",
                merchant_canonical="Pranzo by ESS",
            ),
            Transaction(
                "2",
                "2026-03-01",
                1200.0,
                "RON",
                "debit",
                "large planned",
                "booked_transactions",
                txn_type="card_purchase",
                category_area="Other",
                merchant_canonical="eMAG",
            ),
            Transaction(
                "3",
                "2026-03-01",
                44.0,
                "RON",
                "debit",
                "cinema",
                "booked_transactions",
                txn_type="card_purchase",
                category_area="Entertainment",
            ),
        ]

        totals = aggregate_expenses(rows)
        self.assertEqual(totals["outgoing_tx_count"], 3.0)
        self.assertEqual(totals["impulse_candidate_tx_count"], 2.0)
        self.assertEqual(totals["impulse_tx_count_food"], 1.0)
        self.assertEqual(totals["impulse_tx_count_entertainment"], 1.0)


if __name__ == "__main__":
    unittest.main()

