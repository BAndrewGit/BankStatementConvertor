import tempfile
import unittest
from unittest.mock import patch

from src.classification.merchant_normalizer import MerchantNormalizer
from src.domain.models import Transaction
from src.infrastructure.cache import FileCacheRepository, InMemoryCacheRepository
from src.pipelines.parse_statement import parse_statement


class CacheRepositoryTests(unittest.TestCase):
    def test_namespaces_are_isolated(self):
        cache = InMemoryCacheRepository()
        cache.set("parse", "k", "p")
        cache.set("category", "k", "c")

        self.assertEqual(cache.get("parse", "k"), "p")
        self.assertEqual(cache.get("category", "k"), "c")

    def test_parse_cache_prevents_reparsing_same_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = f"{temp_dir}\\fixed.pdf"
            with open(pdf_path, "wb") as handle:
                handle.write(b"%PDF-1.4\ncache-test\n")

            cache = InMemoryCacheRepository()
            first_result = [
                Transaction(
                    transaction_id="1",
                    booking_date="2026-03-01",
                    amount=10.0,
                    currency="RON",
                    direction="debit",
                    raw_description="POS TEST",
                    source_section="booked_transactions",
                )
            ]

            with patch("src.pipelines.parse_statement.BTStatementParser.parse", return_value=first_result) as mocked_parse:
                first = parse_statement(pdf_path, cache_repo=cache)
                second = parse_statement(pdf_path, cache_repo=cache)

            self.assertEqual(mocked_parse.call_count, 1)
            self.assertEqual(first[0].transaction_id, second[0].transaction_id)

    def test_merchant_cache_skips_repeated_fuzzy_matching(self):
        cache = InMemoryCacheRepository()
        normalizer = MerchantNormalizer(fuzzy_threshold=0.75, cache_repo=cache)
        txn = Transaction(
            transaction_id="1",
            booking_date="2026-03-01",
            amount=10.0,
            currency="RON",
            direction="debit",
            raw_description="x",
            source_section="booked_transactions",
            txn_type="card_purchase",
            merchant_raw="PAYU*EMAG.RO/MARKETPLACE",
        )

        with patch.object(normalizer, "_fuzzy_match", wraps=normalizer._fuzzy_match) as mocked_fuzzy:
            first = normalizer.normalize(txn)
            second = normalizer.normalize(txn)

        self.assertEqual(first.merchant_canonical, second.merchant_canonical)
        self.assertLessEqual(mocked_fuzzy.call_count, 1)

    def test_file_cache_repository_persists_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = f"{temp_dir}\\cache.json"
            first = FileCacheRepository(cache_file)
            first.set("parse", "abc", {"ok": True})

            second = FileCacheRepository(cache_file)
            self.assertEqual(second.get("parse", "abc"), {"ok": True})


if __name__ == "__main__":
    unittest.main()

