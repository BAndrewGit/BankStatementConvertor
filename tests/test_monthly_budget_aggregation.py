import unittest

from pdf_pipeline.transaction_normalization import TransactionNormalized
from pdf_pipeline.transaction_classification import TransactionClassified
from pdf_pipeline.budget_categorization import BudgetCategorizedTransaction
from pdf_pipeline.monthly_budget_aggregation import (
    aggregate_monthly_budget,
    recalculate_monthly_budget,
)


class Story11MonthlyBudgetAggregationTests(unittest.TestCase):
    def test_aggregates_monthly_totals_and_percentages(self):
        classified_rows = [
            _classified("C110001", "income", 5000.0),
            _classified("C110002", "expense", 200.0),
            _classified("C110003", "expense", 100.0),
            _classified("C110004", "expense", 50.0),
            _classified("C110005", "expense", 300.0),
        ]

        categorized_rows = [
            _categorized("C110002", "Food"),
            _categorized("C110003", "Transport"),
            _categorized("C110004", "Unknown"),
            _categorized("C110005", "Entertainment"),
        ]

        normalized_rows = [
            _normalized("C110001", "2026-02-01"),
            _normalized("C110002", "2026-02-03"),
            _normalized("C110003", "2026-02-07"),
            _normalized("C110004", "2026-02-10"),
            _normalized("C110005", "2026-03-02"),
        ]

        overview_rows, category_rows, summary = aggregate_monthly_budget(
            classified_rows,
            categorized_rows,
            normalized_rows,
        )

        overview_by_month = {row.month: row for row in overview_rows}
        self.assertEqual(summary.months_count, 2)
        self.assertEqual(summary.rows_missing_month, 0)

        feb = overview_by_month["2026-02"]
        self.assertEqual(feb.income_total, 5000.0)
        self.assertEqual(feb.expense_total, 350.0)
        self.assertEqual(feb.essential_total, 300.0)
        self.assertEqual(feb.unknown_total, 50.0)
        self.assertEqual(feb.unknown_transaction_count, 1)

        cat_by_key = {(row.month, row.budget_category): row for row in category_rows}
        self.assertEqual(cat_by_key[("2026-02", "Food")].expense_percentage, 57.14)
        self.assertEqual(cat_by_key[("2026-02", "Transport")].expense_percentage, 28.57)
        self.assertEqual(cat_by_key[("2026-02", "Unknown")].expense_percentage, 14.29)

    def test_recalculate_month_after_corrections(self):
        classified_rows = [
            _classified("C110100", "expense", 100.0),
        ]
        normalized_rows = [
            _normalized("C110100", "2026-04-01"),
        ]

        before_rows = [_categorized("C110100", "Unknown")]
        after_rows = [_categorized("C110100", "Food")]

        before_overview, before_categories, _ = recalculate_monthly_budget(
            classified_rows,
            before_rows,
            normalized_rows,
            months={"2026-04"},
        )
        after_overview, after_categories, _ = recalculate_monthly_budget(
            classified_rows,
            after_rows,
            normalized_rows,
            months={"2026-04"},
        )

        self.assertEqual(before_overview[0].unknown_total, 100.0)
        self.assertEqual(after_overview[0].essential_total, 100.0)

        self.assertEqual(before_categories[0].budget_category, "Unknown")
        self.assertEqual(after_categories[0].budget_category, "Food")


def _classified(candidate_id: str, tx_type: str, amount: float) -> TransactionClassified:
    return TransactionClassified(
        candidate_id=candidate_id,
        source_pdf="extras.pdf",
        source_page=1,
        description_clean="DESC",
        merchant_raw_candidate=None,
        direction="in" if tx_type == "income" else "out",
        amount=amount,
        transaction_type=tx_type,
        classification_confidence=0.9,
        classification_reason="seed",
        classification_status="classified_ok",
    )


def _categorized(candidate_id: str, category: str) -> BudgetCategorizedTransaction:
    return BudgetCategorizedTransaction(
        candidate_id=candidate_id,
        source_pdf="extras.pdf",
        source_page=1,
        transaction_type="expense",
        description_clean="DESC",
        merchant_raw_candidate=None,
        selected_caen_code=None,
        selected_caen_description=None,
        budget_category=category,
        classification_reason="rule:test",
        classification_confidence=0.8,
        categorization_status="categorized_ok",
    )


def _normalized(candidate_id: str, tx_date: str) -> TransactionNormalized:
    return TransactionNormalized(
        candidate_id=candidate_id,
        source_pdf="extras.pdf",
        source_page=1,
        transaction_date=tx_date,
        description="D",
        amount=1.0,
        balance=1.0,
        currency="RON",
        direction="out",
        is_valid=True,
        normalization_status="normalized_ok",
        normalization_confidence=0.9,
        invalid_reasons="",
    )


if __name__ == "__main__":
    unittest.main()



