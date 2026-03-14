from __future__ import annotations

from dataclasses import replace
import re
from typing import Dict, List, Sequence, Tuple

from src.domain.enums import Channel, SourceSection, TransactionType
from src.domain.models import Transaction


ATM_HINTS = ["ATM", "RETRAGERE DE NUMERAR", "RETRAGERE NUMERAR"]
FEE_HINTS = ["TAXA", "COMISION", "TAXA SERVICIU SMS", "COMISION PLATA INSTANT"]
INTERNAL_TRANSFER_HINTS = ["TRANSFER BT PAY", "P2P BTPAY", "P2P BT PAY", "TRANSFER INTERN"]
EXTERNAL_TRANSFER_HINTS = ["PLATA INSTANT", "TRANSFER CATRE", "TRANSFER EXTERN", "IBAN", "BENEFICIAR"]
POS_HINTS = ["EPOS", "POS"]

TECHNICAL_FEE_FRAGMENT_PATTERN = re.compile(r"\bCOMISION\s+TRANZACTIE\b", re.IGNORECASE)


class TransactionTypeClassifier:
    """Classify transactions using strict stage-1 ordering rules."""

    def classify(self, transaction: Transaction) -> Transaction:
        raw_description = transaction.raw_description or ""
        description = raw_description.upper()
        description_for_fee = TECHNICAL_FEE_FRAGMENT_PATTERN.sub("", description)
        section = transaction.source_section

        # Rule order is intentional and must stay strict.
        if section == SourceSection.BLOCKED_AMOUNTS.value:
            return replace(
                transaction,
                channel=Channel.BLOCKED.value,
                txn_type=TransactionType.BLOCKED_AMOUNT.value,
                confidence=0.99,
            )

        if self._contains_any(description, ATM_HINTS):
            return replace(
                transaction,
                channel=Channel.ATM.value,
                txn_type=TransactionType.CASH_WITHDRAWAL.value,
                confidence=0.97,
            )

        if self._contains_any(description_for_fee, FEE_HINTS):
            return replace(
                transaction,
                channel=Channel.FEE.value,
                txn_type=TransactionType.BANK_FEE.value,
                confidence=0.96,
            )

        if self._contains_any(description, INTERNAL_TRANSFER_HINTS):
            return replace(
                transaction,
                channel=Channel.TRANSFER.value,
                txn_type=TransactionType.INTERNAL_TRANSFER.value,
                confidence=0.94,
            )

        if self._contains_any(description, EXTERNAL_TRANSFER_HINTS):
            return replace(
                transaction,
                channel=Channel.TRANSFER.value,
                txn_type=TransactionType.EXTERNAL_TRANSFER.value,
                confidence=0.93,
            )

        if self._contains_any(description, POS_HINTS):
            channel = Channel.EPOS.value if "EPOS" in description else Channel.POS.value
            return replace(
                transaction,
                channel=channel,
                txn_type=TransactionType.CARD_PURCHASE.value,
                confidence=0.92,
            )

        return replace(
            transaction,
            channel=Channel.OTHER.value,
            txn_type=TransactionType.UNKNOWN.value,
            confidence=0.55,
        )

    def classify_many(self, transactions: Sequence[Transaction]) -> Tuple[List[Transaction], Dict[str, float]]:
        classified = [self.classify(txn) for txn in transactions]
        total = len(classified)
        valid = sum(1 for txn in classified if txn.txn_type and txn.txn_type != TransactionType.UNKNOWN.value)
        summary: Dict[str, float] = {
            "total": float(total),
            "valid": float(valid),
            "valid_rate": round((valid / total), 4) if total else 0.0,
        }
        return classified, summary

    def _contains_any(self, description: str, hints: Sequence[str]) -> bool:
        return any(hint in description for hint in hints)


def classify_transactions(transactions: Sequence[Transaction]) -> Tuple[List[Transaction], Dict[str, float]]:
    return TransactionTypeClassifier().classify_many(transactions)

