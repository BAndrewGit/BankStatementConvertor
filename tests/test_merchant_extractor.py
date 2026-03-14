import unittest

from src.classification.merchant_extractor import MerchantExtractor
from src.domain.models import Transaction


class MerchantExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = MerchantExtractor()

    def _txn(self, txn_type: str, raw_description: str) -> Transaction:
        return Transaction(
            transaction_id="T1",
            booking_date="2026-03-01",
            amount=100.0,
            currency="RON",
            direction="debit",
            raw_description=raw_description,
            source_section="booked_transactions",
            txn_type=txn_type,
        )

    def test_extracts_expected_examples(self):
        cases = [
            (
                "POS TID:1234 RRN:XYZ PRANZOBYESS@FBP C3 BUCURESTI RO 49,99 RON",
                "PRANZOBYESS",
            ),
            (
                "EPOS PayU*eMAG.ro/Market REF:ABCD 249,90 RON",
                "PayU*eMAG.ro/Market",
            ),
            (
                "EPOS PPC ENERGIE SA HTTP://WWW.ENEL.RO 310,00 RON",
                "PPC ENERGIE SA",
            ),
            (
                "POS MEGAIMAGE 0156 MOGHIOR 77,20 RON",
                "MEGAIMAGE",
            ),
            (
                "UNKNOWN IULI AND ALI FOOD SRL",
                "IULI AND ALI FOOD SRL",
            ),
        ]

        for raw_description, expected in cases:
            txn = self._txn("card_purchase", raw_description)
            result = self.extractor.extract(txn)
            self.assertEqual(result.merchant_raw, expected)

    def test_non_eligible_types_keep_merchant_none(self):
        txn = self._txn("bank_fee", "Comision plata instant")
        result = self.extractor.extract(txn)
        self.assertIsNone(result.merchant_raw)

    def test_strips_pos_boilerplate_and_extracts_core_merchant(self):
        txn = self._txn(
            "card_purchase",
            "Plata la POS non-BT cu card MASTERCARD 7.60 POS 31/01/2026 TID:21593050 CARREFOUR MK. CHILIA V BUCURESTI RO",
        )
        result = self.extractor.extract(txn)
        self.assertEqual(result.merchant_raw, "CARREFOUR")

    def test_extracts_auchan_from_abbreviated_auch_token(self):
        txn = self._txn(
            "card_purchase",
            "Plata la POS non-BT cu card MASTERCARD 115.98 POS 31/01/2026 21DRM1CB TID:21DRM1CB DRM TABEREI 7 AUCH C1 BUCURESTI RO",
        )
        result = self.extractor.extract(txn)
        self.assertEqual(result.merchant_raw, "AUCHAN ROMANIA SA")


if __name__ == "__main__":
    unittest.main()

