import unittest
from unittest.mock import Mock
import os
import tempfile

from tests._brand_fixture import write_brand_fixture_csv
from src.classification.merchant_normalizer import MerchantNormalizer
from src.domain.models import Transaction


class MerchantNormalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls._aliases_yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src",
            "config",
            "merchant_aliases.yaml",
        )
        cls._od_firme_small_path = os.path.join(cls._tmp_dir.name, "od_firme_small.csv")
        write_brand_fixture_csv(cls._tmp_dir.name, filename="od_firme_small.csv")

        cls._empty_aliases_path = os.path.join(cls._tmp_dir.name, "aliases_empty.yaml")
        with open(cls._empty_aliases_path, "w", encoding="utf-8") as handle:
            handle.write("{}\n")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_dir.cleanup()

    def setUp(self) -> None:
        self.normalizer = MerchantNormalizer(
            od_firme_csv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "DatasetsCAEN", "__missing__.csv")
        )

    def _txn(self, merchant_raw: str) -> Transaction:
        return Transaction(
            transaction_id="T1",
            booking_date="2026-03-01",
            amount=10.0,
            currency="RON",
            direction="debit",
            raw_description=merchant_raw,
            source_section="booked_transactions",
            txn_type="card_purchase",
            merchant_raw=merchant_raw,
        )

    def test_exact_alias_mapping_is_deterministic(self):
        first = self.normalizer.normalize(self._txn("PAYU*EMAG.RO/MARKET"))
        second = self.normalizer.normalize(self._txn("PAYU*EMAG.RO/MARKET"))

        self.assertEqual(first.merchant_canonical, "eMAG")
        self.assertEqual(first.mapping_method, "exact_alias")
        self.assertEqual(first.merchant_canonical, second.merchant_canonical)

    def test_diacritics_and_case_are_normalized(self):
        normalized = self.normalizer.normalize(self._txn("páyu*émag.ro/market"))
        self.assertEqual(normalized.merchant_canonical, "eMAG")
        self.assertIn(normalized.mapping_method, {"exact_alias", "fuzzy_alias"})

    def test_payu_emag_with_symbol_separator_is_normalized_without_space(self):
        raw = "Plata la POS EPOS 064362ER PayU*eMAG Soseaua Virtutii Bucuresti"
        normalized = self.normalizer.normalize(self._txn(raw))
        self.assertEqual(normalized.merchant_canonical, "eMAG")
        self.assertIn(normalized.mapping_method, {"exact_alias", "regex_alias", "fuzzy_alias"})

    def test_unmapped_falls_back_to_raw_name(self):
        result = self.normalizer.normalize(self._txn("MERCHANT NOU TEST"))
        self.assertEqual(result.merchant_canonical, "MERCHANT NOU TEST")
        self.assertEqual(result.mapping_method, "unmapped")

    def test_noisy_string_containing_auchan_is_normalized(self):
        noisy = "PLATA LA 04/02/2026 AUCHAN ROMANIA SA BRASOV BUCURESTI ROM 55393082 SOLD FINAL ZI"
        result = self.normalizer.normalize(self._txn(noisy))
        self.assertEqual(result.merchant_canonical, "Auchan")
        self.assertIn(result.mapping_method, {"regex_alias", "exact_alias", "fuzzy_alias", "dataset_match"})

    def test_termene_fallback_is_used_only_when_enabled_and_unresolved(self):
        termene = Mock()
        termene.search_company.return_value = {"name": "COMPANIE TEST SRL"}

        normalizer = MerchantNormalizer(
            aliases_yaml_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "config", "merchant_aliases.yaml"),
            od_firme_csv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "DatasetsCAEN", "__missing__.csv"),
            termene_client=termene,
            enable_termene_fallback=True,
        )

        result = normalizer.normalize(self._txn("MERCHANT FARA MATCH LOCAL"))
        self.assertEqual(result.merchant_canonical, "COMPANIE TEST SRL")
        self.assertEqual(result.normalization_source, "termene")
        self.assertEqual(result.mapping_method, "termene_api")
        termene.search_company.assert_called_once()

    def test_dataset_match_is_normalized_to_carrefour_brand(self):
        normalizer = MerchantNormalizer(
            aliases_yaml_path=self._aliases_yaml_path,
            od_firme_csv_path=self._od_firme_small_path,
        )

        result = normalizer.normalize(
            self._txn("31/01/2026 677661110 CARREFOUR MK CHILIA V BUCURESTI RO")
        )

        self.assertEqual(result.merchant_canonical, "Carrefour")
        self.assertEqual(result.normalization_source, "dataset")

    def test_dataset_match_is_normalized_to_emag_brand(self):
        normalizer = MerchantNormalizer(
            aliases_yaml_path=self._aliases_yaml_path,
            od_firme_csv_path=self._od_firme_small_path,
        )

        result = normalizer.normalize(self._txn("PayU*eMAG.ro/Market"))
        self.assertEqual(result.merchant_canonical, "eMAG")

    def test_unknown_merchant_does_not_force_low_confidence_dataset_company(self):
        normalizer = MerchantNormalizer(
            aliases_yaml_path=self._aliases_yaml_path,
            od_firme_csv_path=self._od_firme_small_path,
            fuzzy_threshold=0.95,
        )

        result = normalizer.normalize(self._txn("MERCHANT FARA BRAND CLAR 123"))
        self.assertNotEqual(result.normalization_source, "dataset")

    def test_dataset_match_maps_auchan_brand_even_with_multiple_auchan_rows(self):
        normalizer = MerchantNormalizer(
            aliases_yaml_path=self._empty_aliases_path,
            od_firme_csv_path=self._od_firme_small_path,
        )

        result = normalizer.normalize(
            self._txn("POS 31/01/2026 TID AU014028 AUCHAN ROMANIA SA Brasov BUCURESTI")
        )

        self.assertEqual(result.normalization_source, "dataset")
        self.assertEqual(result.merchant_canonical, "Auchan")

    def test_dataset_match_maps_carrefour_brand_even_with_multiple_carrefour_rows(self):
        normalizer = MerchantNormalizer(
            aliases_yaml_path=self._empty_aliases_path,
            od_firme_csv_path=self._od_firme_small_path,
        )

        result = normalizer.normalize(
            self._txn("31/01/2026 677661110 CARREFOUR MK CHILIA V BUCURESTI RO")
        )

        self.assertEqual(result.normalization_source, "dataset")
        self.assertEqual(result.merchant_canonical, "Carrefour")


if __name__ == "__main__":
    unittest.main()

