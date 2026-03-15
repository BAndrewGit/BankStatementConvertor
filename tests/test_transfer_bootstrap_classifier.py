import unittest
import tempfile

from src.memory.entity_memory import EntityMemoryRepository
from src.classification.transfer_bootstrap_classifier import TransferBootstrapClassifier
from src.domain.models import Transaction


class TransferBootstrapClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = TransferBootstrapClassifier()

    def _txn(self, txn_id: str, txn_type: str, description: str, direction: str = "debit") -> Transaction:
        return Transaction(
            transaction_id=txn_id,
            booking_date="21/02/2026",
            amount=100.0,
            currency="RON",
            direction=direction,
            raw_description=description,
            source_section="booked_transactions",
            txn_type=txn_type,
        )

    def test_internal_transfer_is_excluded(self):
        txn = self._txn("1", "internal_transfer", "Transfer BT PAY REF: 123")
        out = self.classifier.classify(txn)
        self.assertEqual(out.state, "excluded")
        self.assertEqual(out.match_source, "internal_transfer_filter")

    def test_external_transfer_housing_exact_match(self):
        txn = self._txn(
            "2",
            "external_transfer",
            "Plata Instant 400.00 chirie+intretinere; Adina Miereanu; RO86INGB0000999911338663; REF: X",
        )
        out = self.classifier.classify(txn)
        self.assertEqual(out.state, "confirmed")
        self.assertEqual(out.match_source, "bootstrap_exact")
        self.assertEqual(out.category_area, "Housing")
        self.assertEqual(out.recipient_iban, "RO86INGB0000999911338663")

    def test_salary_income_remains_confirmed(self):
        txn = self._txn("3", "salary_income", "Incasare OP salariu luna ianuarie", direction="credit")
        out = self.classifier.classify(txn)
        self.assertEqual(out.state, "confirmed")
        self.assertEqual(out.match_source, "bootstrap_exact")

    def test_entity_memory_reuse_for_same_profile_and_entity_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_repo = EntityMemoryRepository(memory_dir=temp_dir)
            profile_id = "profile-test"

            classifier_first = TransferBootstrapClassifier(profile_id=profile_id, memory_repo=memory_repo)
            first = self._txn(
                "4",
                "external_transfer",
                "Plata Instant 500 chirie; Asociatia proprietarilor bloc X; RO48RNCB0068004387790001; REF: A",
            )
            first_out = classifier_first.classify_many([first])[0]
            self.assertEqual(first_out.state, "confirmed")
            self.assertEqual(first_out.match_source, "bootstrap_exact")

            classifier_second = TransferBootstrapClassifier(profile_id=profile_id, memory_repo=memory_repo)
            second = self._txn(
                "5",
                "external_transfer",
                "Plata Instant 510 random text fara semnal clar; Asociatia proprietarilor bloc X; RO48RNCB0068004387790001; REF: B",
            )
            second_out = classifier_second.classify_many([second])[0]
            self.assertEqual(second_out.state, "confirmed")
            self.assertEqual(second_out.match_source, "entity_memory")
            self.assertEqual(second_out.category_area, "Housing")

    def test_same_run_unknown_transfer_is_backfilled_from_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_repo = EntityMemoryRepository(memory_dir=temp_dir)
            profile_id = "profile-backfill"
            classifier = TransferBootstrapClassifier(profile_id=profile_id, memory_repo=memory_repo)

            unknown_first = self._txn(
                "6",
                "external_transfer",
                "Plata Instant 510 random text fara context; Beneficiar Generic; RO48RNCB0068004387790001; REF: U",
            )
            exact_second = self._txn(
                "7",
                "external_transfer",
                "Plata Instant 500 chirie+intretinere; Beneficiar Generic; RO48RNCB0068004387790001; REF: E",
            )

            out = classifier.classify_many([unknown_first, exact_second])
            by_id = {txn.transaction_id: txn for txn in out}

            self.assertEqual(by_id["7"].state, "confirmed")
            self.assertEqual(by_id["7"].match_source, "bootstrap_exact")
            self.assertEqual(by_id["6"].state, "confirmed")
            self.assertEqual(by_id["6"].match_source, "entity_memory_backfill")
            self.assertEqual(by_id["6"].category_area, "Housing")


if __name__ == "__main__":
    unittest.main()




