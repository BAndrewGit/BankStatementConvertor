import unittest

from src.features.quality_metrics import compute_quality_metrics
from src.domain.models import Transaction


class QualityMetricsTests(unittest.TestCase):
    def _txn(
        self,
        txn_id: str,
        source_section: str,
        txn_type: str,
        merchant_raw: str | None,
        merchant_canonical: str | None,
        category_area: str | None,
    ) -> Transaction:
        return Transaction(
            transaction_id=txn_id,
            booking_date="2026-03-01",
            amount=10.0,
            currency="RON",
            direction="debit",
            raw_description="x",
            source_section=source_section,
            txn_type=txn_type,
            merchant_raw=merchant_raw,
            merchant_canonical=merchant_canonical,
            category_area=category_area,
        )

    def test_required_metrics_are_computed_from_real_counts(self):
        rows = [
            self._txn("1", "booked_transactions", "card_purchase", "MEGAIMAGE", "Mega Image", "Food"),
            self._txn("2", "booked_transactions", "subscription", None, None, "Unknown"),
            self._txn("3", "blocked_amounts", "blocked_amount", None, None, None),
        ]

        metrics = compute_quality_metrics(
            transactions=rows,
            unknown_expense_percentage=25.0,
            pdf_parse_latency_ms=120.0,
            end_to_end_latency_ms=450.0,
        )

        payload = metrics.to_dict()
        self.assertEqual(payload["transactions_extracted_count"], 3)
        self.assertEqual(payload["booked_transactions_count"], 2)
        self.assertEqual(payload["blocked_transactions_count"], 1)
        self.assertEqual(payload["merchant_extracted_count"], 1)
        self.assertEqual(payload["merchant_unknown_count"], 1)
        self.assertEqual(payload["category_unknown_count"], 1)
        self.assertEqual(payload["unknown_expense_percentage"], 25.0)

    def test_manual_eval_metrics_are_optional_and_calculated_when_labels_exist(self):
        rows = [
            self._txn("1", "booked_transactions", "card_purchase", "MEGAIMAGE", "Mega Image", "Food"),
            self._txn("2", "booked_transactions", "card_purchase", "EMAG", "eMAG", "Other"),
        ]

        labels = {
            "1": {"txn_type": "card_purchase", "merchant_canonical": "Mega Image", "category_area": "Food"},
            "2": {"txn_type": "card_purchase", "merchant_canonical": "Wrong", "category_area": "Other"},
        }

        metrics = compute_quality_metrics(
            transactions=rows,
            unknown_expense_percentage=0.0,
            pdf_parse_latency_ms=80.0,
            end_to_end_latency_ms=150.0,
            manual_labels=labels,
        )

        self.assertEqual(metrics.txn_type_accuracy, 1.0)
        self.assertEqual(metrics.merchant_extraction_accuracy, 0.5)
        self.assertEqual(metrics.category_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()

