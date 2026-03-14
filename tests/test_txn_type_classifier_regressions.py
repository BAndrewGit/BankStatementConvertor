import unittest

from src.classification.txn_type_classifier import classify_transactions
from src.domain.models import Transaction


class TxnTypeClassifierRegressionTests(unittest.TestCase):
    def test_card_purchase_not_misclassified_as_fee_due_to_technical_fragment(self):
        rows = [
            Transaction(
                transaction_id="1",
                booking_date="2026-03-01",
                amount=79.67,
                currency="RON",
                direction="debit",
                raw_description=(
                    "Plata la POS 79.67 EPOS 06/02/2026 TID:064362ER PayU*eMAG.ro "
                    "valoare tranzactie: 79.67 RON RRN:288495921401 comision tranzactie 0.00 RON"
                ),
                source_section="booked_transactions",
            )
        ]

        classified, _ = classify_transactions(rows)
        self.assertEqual(classified[0].txn_type, "card_purchase")


if __name__ == "__main__":
    unittest.main()

