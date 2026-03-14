from __future__ import annotations

from typing import Dict, Sequence

from src.domain.enums import CategoryArea, TransactionType
from src.domain.models import Transaction


BASE_TOTALS = {
    "food_total": 0.0,
    "housing_total": 0.0,
    "transport_total": 0.0,
    "entertainment_total": 0.0,
    "health_total": 0.0,
    "personal_care_total": 0.0,
    "child_education_total": 0.0,
    "other_total": 0.0,
    "unknown_total": 0.0,
}

CATEGORY_TO_TOTAL_KEY = {
    CategoryArea.FOOD.value: "food_total",
    CategoryArea.HOUSING.value: "housing_total",
    CategoryArea.TRANSPORT.value: "transport_total",
    CategoryArea.ENTERTAINMENT.value: "entertainment_total",
    CategoryArea.HEALTH.value: "health_total",
    CategoryArea.PERSONAL_CARE.value: "personal_care_total",
    CategoryArea.CHILD_EDUCATION.value: "child_education_total",
    CategoryArea.OTHER.value: "other_total",
    CategoryArea.UNKNOWN.value: "unknown_total",
}

EXCLUDED_TYPES_DEFAULT = {
    TransactionType.INTERNAL_TRANSFER.value,
    TransactionType.EXTERNAL_TRANSFER.value,
    TransactionType.CASH_WITHDRAWAL.value,
    TransactionType.BLOCKED_AMOUNT.value,
    TransactionType.BANK_FEE.value,
}


def aggregate_expenses(transactions: Sequence[Transaction], include_bank_fees: bool = False) -> Dict[str, float]:
    totals = dict(BASE_TOTALS)

    excluded_types = set(EXCLUDED_TYPES_DEFAULT)
    if include_bank_fees:
        excluded_types.discard(TransactionType.BANK_FEE.value)

    for txn in transactions:
        if txn.direction != "debit":
            continue
        if txn.txn_type in excluded_types:
            continue

        category = txn.category_area or CategoryArea.UNKNOWN.value
        total_key = CATEGORY_TO_TOTAL_KEY.get(category, "unknown_total")
        totals[total_key] += abs(float(txn.amount))

    for key, value in list(totals.items()):
        totals[key] = round(value, 2)

    return totals

