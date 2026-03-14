import unittest
from unittest.mock import patch
import os

from src.classification.merchant_normalizer import MerchantNormalizer as RealMerchantNormalizer
from src.domain.models import Transaction
from src.pipelines.classify_transactions import classify_parsed_transactions


class ClassifyPipelineMerchantsTests(unittest.TestCase):
    def test_merchants_added_only_for_eligible_types(self):
        rows = [
            Transaction("1", "2026-03-01", 120.0, "RON", "debit", "POS MEGAIMAGE 0156 MOGHIOR", "booked_transactions"),
            Transaction("2", "2026-03-01", 3.0, "RON", "debit", "Taxa Serviciu SMS", "booked_transactions"),
            Transaction("3", "2026-03-01", 60.0, "RON", "debit", "SUME BLOCATE EPOS STORE", "blocked_amounts"),
        ]

        missing_dataset = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DatasetsCAEN", "__missing__.csv")

        def _fast_normalizer(*args, **kwargs):
            kwargs.setdefault("od_firme_csv_path", missing_dataset)
            kwargs.setdefault("enable_termene_fallback", False)
            return RealMerchantNormalizer(*args, **kwargs)

        with patch("src.pipelines.classify_transactions.MerchantNormalizer", side_effect=_fast_normalizer):
            classified, summary = classify_parsed_transactions(rows)
        by_id = {txn.transaction_id: txn for txn in classified}

        self.assertEqual(by_id["1"].txn_type, "card_purchase")
        self.assertEqual(by_id["1"].merchant_raw, "MEGAIMAGE")
        self.assertEqual(by_id["1"].merchant_canonical, "Mega Image")

        self.assertEqual(by_id["2"].txn_type, "bank_fee")
        self.assertIsNone(by_id["2"].merchant_raw)
        self.assertIsNone(by_id["2"].merchant_canonical)

        self.assertEqual(by_id["3"].txn_type, "blocked_amount")
        self.assertIsNone(by_id["3"].merchant_raw)
        self.assertIsNone(by_id["3"].merchant_canonical)

        self.assertEqual(summary["merchant_extracted"], 1.0)
        self.assertEqual(summary["merchant_normalized"], 1.0)


if __name__ == "__main__":
    unittest.main()

