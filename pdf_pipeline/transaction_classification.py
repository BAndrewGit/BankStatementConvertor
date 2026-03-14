from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import os
from typing import Dict, List, Optional, Tuple

from .transaction_normalization import TransactionNormalized
from .description_normalization import TransactionDescriptionNormalized


TRANSACTION_TYPES = {
    "income",
    "expense",
    "transfer",
    "cash_withdrawal",
    "fee",
    "debt_payment",
    "unknown",
}

FEE_KEYWORDS = ["COMISION", "TAXA", "ADMINISTRARE", "INTRETINERE CONT"]
CASH_KEYWORDS = ["RETRAGERE", "ATM", "NUMERAR", "CASH"]
TRANSFER_KEYWORDS = ["TRANSFER", "VIRAMENT", "P2P", "INTERN", "INTERCONT"]
DEBT_KEYWORDS = ["RATA", "CREDIT", "LOAN", "IPOTECA", "CARD CREDIT"]
INCOME_KEYWORDS = ["SALARIU", "INCASARE", "DOBANDA", "CASHBACK", "REFUND"]


@dataclass(frozen=True)
class TransactionClassified:
    candidate_id: str
    source_pdf: str
    source_page: int
    description_clean: str
    merchant_raw_candidate: Optional[str]
    direction: Optional[str]
    amount: Optional[float]
    transaction_type: str
    classification_confidence: float
    classification_reason: str
    classification_status: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _classify_type(description_clean: str, direction: Optional[str]) -> Tuple[str, float, str]:
    text = description_clean or ""

    if _contains_any(text, FEE_KEYWORDS):
        return "fee", 0.98, "rule:fee_keywords"

    if _contains_any(text, CASH_KEYWORDS):
        return "cash_withdrawal", 0.97, "rule:cash_keywords"

    if _contains_any(text, DEBT_KEYWORDS):
        return "debt_payment", 0.94, "rule:debt_keywords"

    if _contains_any(text, TRANSFER_KEYWORDS):
        return "transfer", 0.93, "rule:transfer_keywords"

    if _contains_any(text, INCOME_KEYWORDS):
        return "income", 0.91, "rule:income_keywords"

    if direction == "in":
        return "income", 0.75, "rule:direction_in_fallback"

    if direction == "out":
        return "expense", 0.72, "rule:direction_out_fallback"

    return "unknown", 0.40, "rule:unknown_fallback"


def classify_transaction_types(
    normalized_rows: List[TransactionNormalized],
    description_rows: List[TransactionDescriptionNormalized],
) -> Tuple[List[TransactionClassified], Dict[str, int]]:
    """Classify transactions into operational types for Story 6."""
    desc_by_candidate = {row.candidate_id: row for row in description_rows}

    classified_rows: List[TransactionClassified] = []

    for row in normalized_rows:
        desc_row = desc_by_candidate.get(row.candidate_id)
        description_clean = desc_row.description_clean if desc_row else (row.description or "")
        merchant_raw_candidate = desc_row.merchant_raw_candidate if desc_row else None

        tx_type, base_confidence, reason = _classify_type(description_clean, row.direction)
        confidence = min(base_confidence, desc_row.description_normalization_confidence) if desc_row else base_confidence
        confidence = min(confidence, row.normalization_confidence)
        confidence = round(max(0.0, confidence), 2)

        status = "classified_ok" if tx_type in TRANSACTION_TYPES and tx_type != "unknown" else "classified_with_warnings"

        classified_rows.append(
            TransactionClassified(
                candidate_id=row.candidate_id,
                source_pdf=row.source_pdf,
                source_page=row.source_page,
                description_clean=description_clean,
                merchant_raw_candidate=merchant_raw_candidate,
                direction=row.direction,
                amount=row.amount,
                transaction_type=tx_type,
                classification_confidence=confidence,
                classification_reason=reason,
                classification_status=status,
            )
        )

    summary = {
        "total_rows": len(classified_rows),
        "income": sum(1 for item in classified_rows if item.transaction_type == "income"),
        "expense": sum(1 for item in classified_rows if item.transaction_type == "expense"),
        "transfer": sum(1 for item in classified_rows if item.transaction_type == "transfer"),
        "cash_withdrawal": sum(1 for item in classified_rows if item.transaction_type == "cash_withdrawal"),
        "fee": sum(1 for item in classified_rows if item.transaction_type == "fee"),
        "debt_payment": sum(1 for item in classified_rows if item.transaction_type == "debt_payment"),
        "unknown": sum(1 for item in classified_rows if item.transaction_type == "unknown"),
    }

    return classified_rows, summary


def save_transactions_classified_csv(rows: List[TransactionClassified], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_id",
                "source_pdf",
                "source_page",
                "description_clean",
                "merchant_raw_candidate",
                "direction",
                "amount",
                "transaction_type",
                "classification_confidence",
                "classification_reason",
                "classification_status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


