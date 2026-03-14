from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence

from src.domain.enums import SourceSection, TransactionType
from src.domain.models import Transaction


MERCHANT_ELIGIBLE_TYPES = {
    TransactionType.CARD_PURCHASE.value,
    TransactionType.SUBSCRIPTION.value,
    TransactionType.UTILITY_PAYMENT.value,
}


@dataclass(frozen=True)
class QualityMetrics:
    pdf_parse_latency_ms: float
    transactions_extracted_count: int
    booked_transactions_count: int
    blocked_transactions_count: int
    merchant_extracted_count: int
    merchant_unknown_count: int
    category_unknown_count: int
    unknown_expense_percentage: float
    end_to_end_latency_ms: float
    txn_type_accuracy: Optional[float] = None
    merchant_extraction_accuracy: Optional[float] = None
    category_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_accuracy(matches: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return round(matches / total, 4)


def _manual_metric(
    transactions: Sequence[Transaction],
    manual_labels: Mapping[str, Mapping[str, object]],
    field_name: str,
) -> Optional[float]:
    matches = 0
    total = 0

    transactions_by_id = {txn.transaction_id: txn for txn in transactions}

    for txn_id in sorted(manual_labels.keys()):
        expected = manual_labels.get(txn_id, {})
        if field_name not in expected:
            continue

        transaction = transactions_by_id.get(txn_id)
        if transaction is None:
            continue

        predicted_value = getattr(transaction, field_name)
        expected_value = expected[field_name]
        total += 1
        if predicted_value == expected_value:
            matches += 1

    return _safe_accuracy(matches, total)


def compute_quality_metrics(
    transactions: Sequence[Transaction],
    unknown_expense_percentage: float,
    pdf_parse_latency_ms: float,
    end_to_end_latency_ms: float,
    manual_labels: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> QualityMetrics:
    booked_transactions_count = sum(
        1 for txn in transactions if txn.source_section == SourceSection.BOOKED_TRANSACTIONS.value
    )
    blocked_transactions_count = sum(
        1 for txn in transactions if txn.source_section == SourceSection.BLOCKED_AMOUNTS.value
    )

    merchant_eligible = [txn for txn in transactions if txn.txn_type in MERCHANT_ELIGIBLE_TYPES]
    merchant_extracted_count = sum(1 for txn in merchant_eligible if txn.merchant_raw)
    merchant_unknown_count = sum(1 for txn in merchant_eligible if not txn.merchant_raw)
    category_unknown_count = sum(1 for txn in merchant_eligible if txn.category_area == "Unknown")

    txn_type_accuracy = None
    merchant_extraction_accuracy = None
    category_accuracy = None
    if manual_labels:
        txn_type_accuracy = _manual_metric(transactions, manual_labels, "txn_type")
        merchant_extraction_accuracy = _manual_metric(transactions, manual_labels, "merchant_canonical")
        category_accuracy = _manual_metric(transactions, manual_labels, "category_area")

    return QualityMetrics(
        pdf_parse_latency_ms=round(pdf_parse_latency_ms, 2),
        transactions_extracted_count=len(transactions),
        booked_transactions_count=booked_transactions_count,
        blocked_transactions_count=blocked_transactions_count,
        merchant_extracted_count=merchant_extracted_count,
        merchant_unknown_count=merchant_unknown_count,
        category_unknown_count=category_unknown_count,
        unknown_expense_percentage=round(float(unknown_expense_percentage), 4),
        end_to_end_latency_ms=round(end_to_end_latency_ms, 2),
        txn_type_accuracy=txn_type_accuracy,
        merchant_extraction_accuracy=merchant_extraction_accuracy,
        category_accuracy=category_accuracy,
    )

