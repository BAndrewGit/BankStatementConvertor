import os
import tempfile
import unittest

from tests._brand_fixture import AUCHAN_AND_CARREFOUR_ROWS, write_brand_fixture_csv
from src.classification.dataset_merchant_matcher import DatasetMerchantMatcher
from src.classification.merchant_normalizer import MerchantNormalizer
from src.domain.models import Transaction


class DatasetMerchantMatcherTests(unittest.TestCase):
    def _write_od_firme(self, folder: str) -> str:
        return write_brand_fixture_csv(folder)

    def test_dataset_matcher_exact_query_hits_every_auchan_and_carrefour_entity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme = self._write_od_firme(temp_dir)
            matcher = DatasetMerchantMatcher(od_firme_csv_path=od_firme)

            for name, _ in AUCHAN_AND_CARREFOUR_ROWS:
                candidate, score = matcher.find_best_match(name)
                self.assertEqual(candidate, name)
                self.assertEqual(score, 100.0)

    def test_dataset_matcher_finds_best_company_from_noisy_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme = self._write_od_firme(temp_dir)
            matcher = DatasetMerchantMatcher(od_firme_csv_path=od_firme)

            candidate, score = matcher.find_best_match(
                "31/01/2026 21DRM1CB DRM TABEREI 7 AUCH C1 BUCURESTI RO"
            )

            self.assertIsNotNone(candidate)
            self.assertIn("AUCHAN", candidate or "")
            self.assertGreaterEqual(score, 70.0)

    def test_dataset_matcher_still_finds_auchan_when_multiple_auchan_rows_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme = self._write_od_firme(temp_dir)
            matcher = DatasetMerchantMatcher(od_firme_csv_path=od_firme)

            candidate, score = matcher.find_best_match(
                "POS 31/01/2026 TID AU014028 AUCHAN ROMANIA SA Brasov BUCURESTI"
            )

            self.assertEqual(candidate, "AUCHAN ROMANIA SA")
            self.assertGreaterEqual(score, 70.0)

    def test_dataset_matcher_returns_an_auchan_candidate_for_transaction_like_query(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme = self._write_od_firme(temp_dir)
            matcher = DatasetMerchantMatcher(od_firme_csv_path=od_firme)

            candidate, score = matcher.find_best_match(
                "POS 23/02/2026 TID AU031024 AUCHAN ROMANIA SA Strada Vasile Milea Bucuresti"
            )

            self.assertIsNotNone(candidate)
            self.assertIn("AUCHAN", candidate or "")
            self.assertGreaterEqual(score, 70.0)

    def test_dataset_matcher_prefers_carrefour_romania_for_pos_like_query(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme = self._write_od_firme(temp_dir)
            matcher = DatasetMerchantMatcher(od_firme_csv_path=od_firme)

            candidate, score = matcher.find_best_match(
                "31/01/2026 677661110 CARREFOUR MK CHILIA V BUCURESTI RO"
            )

            self.assertEqual(candidate, "CARREFOUR ROMANIA SA")
            self.assertGreaterEqual(score, 70.0)

    def test_normalizer_uses_dataset_match_when_alias_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme = self._write_od_firme(temp_dir)
            empty_aliases = os.path.join(temp_dir, "aliases.yaml")
            with open(empty_aliases, "w", encoding="utf-8") as handle:
                handle.write("{}\n")

            normalizer = MerchantNormalizer(
                aliases_yaml_path=empty_aliases,
                od_firme_csv_path=od_firme,
            )

            txn = Transaction(
                transaction_id="T1",
                booking_date="2026-02-03",
                amount=115.98,
                currency="RON",
                direction="debit",
                raw_description="x",
                source_section="booked_transactions",
                txn_type="card_purchase",
                merchant_raw="31/01/2026 21DRM1CB DRM TABEREI 7 AUCH C1 BUCURESTI RO",
            )

            normalized = normalizer.normalize(txn)
            self.assertEqual(normalized.merchant_canonical, "Auchan")
            self.assertEqual(normalized.mapping_method, "dataset_match")
            self.assertEqual(normalized.normalization_source, "dataset")


if __name__ == "__main__":
    unittest.main()

