import unittest

from src.classification.category_mapper import CategoryMapper
from src.domain.models import Transaction
from src.infrastructure.cache import InMemoryCacheRepository


class CategoryMapperTests(unittest.TestCase):
    def _txn(self, transaction_id: str, txn_type: str, merchant_canonical: str | None, mapping_method: str | None = "exact_alias") -> Transaction:
        return Transaction(
            transaction_id=transaction_id,
            booking_date="2026-03-01",
            amount=100.0,
            currency="RON",
            direction="debit",
            raw_description="x",
            source_section="booked_transactions",
            txn_type=txn_type,
            merchant_canonical=merchant_canonical,
            mapping_method=mapping_method,
        )

    def test_non_eligible_types_are_not_mapped(self):
        mapper = CategoryMapper(cache_repo=InMemoryCacheRepository())
        row = self._txn("1", "bank_fee", "PPC Energie")

        mapped = mapper.map(row)
        self.assertIsNone(mapped.category_area)

    def test_unknown_when_merchant_missing(self):
        mapper = CategoryMapper(cache_repo=InMemoryCacheRepository())
        row = self._txn("1", "card_purchase", None)

        mapped = mapper.map(row)
        self.assertEqual(mapped.category_area, "Unknown")
        self.assertEqual(mapped.mapping_method, "unmapped")

    def test_repeated_merchant_is_consistent(self):
        mapper = CategoryMapper(cache_repo=InMemoryCacheRepository())
        first = mapper.map(self._txn("1", "card_purchase", "Mega Image"))
        second = mapper.map(self._txn("2", "card_purchase", "Mega Image"))

        self.assertEqual(first.category_area, "Food")
        self.assertEqual(second.category_area, "Food")

    def test_unknown_rate_under_ten_percent_for_purchase_sample(self):
        mapper = CategoryMapper(cache_repo=InMemoryCacheRepository())
        rows = [self._txn(str(i), "card_purchase", "Mega Image") for i in range(1, 11)]
        rows.append(self._txn("11", "card_purchase", None, None))

        _, summary = mapper.map_many(rows)
        self.assertLess(summary["unknown_rate"], 0.10)


if __name__ == "__main__":
    unittest.main()

