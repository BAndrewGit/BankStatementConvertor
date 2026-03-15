from __future__ import annotations

import os
from typing import Dict, Sequence

import yaml

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
    "electronics_gadgets_total": 0.0,
    "income_total": 0.0,
    "outgoing_expense_total": 0.0,
    "essential_total": 0.0,
    "nonessential_total": 0.0,
    "outgoing_tx_count": 0.0,
    "impulse_candidate_tx_count": 0.0,
    "impulse_spend_clothing_personal_care": 0.0,
    "impulse_spend_electronics_gadgets": 0.0,
    "impulse_spend_entertainment": 0.0,
    "impulse_spend_food": 0.0,
    "impulse_spend_other": 0.0,
    "impulse_tx_count_clothing_personal_care": 0.0,
    "impulse_tx_count_electronics_gadgets": 0.0,
    "impulse_tx_count_entertainment": 0.0,
    "impulse_tx_count_food": 0.0,
    "impulse_tx_count_other": 0.0,
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
    TransactionType.CASH_WITHDRAWAL.value,
    TransactionType.BLOCKED_AMOUNT.value,
    TransactionType.BANK_FEE.value,
}

ESSENTIAL_CATEGORIES = {
    CategoryArea.FOOD.value,
    CategoryArea.HOUSING.value,
    CategoryArea.TRANSPORT.value,
    CategoryArea.HEALTH.value,
}

IMPULSE_EXCLUDED_TYPES = {
    TransactionType.INTERNAL_TRANSFER.value,
    TransactionType.EXTERNAL_TRANSFER.value,
    TransactionType.BANK_FEE.value,
    TransactionType.BLOCKED_AMOUNT.value,
    TransactionType.CASH_WITHDRAWAL.value,
}


def _is_transfer_expense_eligible(txn: Transaction, category: str) -> bool:
    if txn.txn_type != TransactionType.EXTERNAL_TRANSFER.value:
        return True
    if (txn.state or "") in {"excluded", "unknown"}:
        return False
    return category not in {CategoryArea.UNKNOWN.value, "", None}


def _impulse_bucket(txn: Transaction, category: str) -> str | None:
    canonical = (txn.merchant_canonical or "").strip()
    if canonical in ELECTRONICS_MERCHANTS:
        return "electronics_gadgets"
    if category == CategoryArea.PERSONAL_CARE.value:
        return "clothing_personal_care"
    if category == CategoryArea.ENTERTAINMENT.value:
        return "entertainment"
    if category == CategoryArea.FOOD.value:
        return "food"
    if category in {CategoryArea.OTHER.value, CategoryArea.UNKNOWN.value}:
        return "other"
    return None


def _load_impulse_food_merchants() -> set[str]:
    rules_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "category_rules.yaml",
    )
    try:
        with open(rules_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return {"Pranzo by ESS", "Taco Bell", "Froo", "JKC Restaurants"}

    merchants = (
        (payload.get("impulse_category_merchants") or {}).get("food_quick_consumption")
        or []
    )
    extracted = {str(item).strip() for item in merchants if str(item).strip()}
    return extracted or {"Pranzo by ESS", "Taco Bell", "Froo", "JKC Restaurants"}


def _load_impulse_max_amount() -> float:
    rules_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "category_rules.yaml",
    )
    try:
        with open(rules_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return 300.0

    return float((payload.get("impulse_rules") or {}).get("max_candidate_amount", 300.0))


def _load_electronics_merchants() -> set[str]:
    rules_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "category_rules.yaml",
    )
    try:
        with open(rules_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return set()

    merchants = (
        (payload.get("impulse_category_merchants") or {}).get("electronics_or_gadgets")
        or []
    )
    return {str(item).strip() for item in merchants if str(item).strip()}


ELECTRONICS_MERCHANTS = _load_electronics_merchants()
IMPULSE_FOOD_MERCHANTS = _load_impulse_food_merchants()
IMPULSE_MAX_AMOUNT = _load_impulse_max_amount()


def _is_impulse_candidate(txn: Transaction, category: str, amount: float, bucket: str | None) -> bool:
    if txn.txn_type in IMPULSE_EXCLUDED_TYPES:
        return False
    if category in {
        CategoryArea.HOUSING.value,
        CategoryArea.TRANSPORT.value,
        CategoryArea.HEALTH.value,
        CategoryArea.CHILD_EDUCATION.value,
    }:
        return False
    if amount > IMPULSE_MAX_AMOUNT:
        return False
    if bucket is None:
        return False

    # Food impulse is only for quick consumption, not household provisioning.
    if bucket == "food" and (txn.merchant_canonical or "") not in IMPULSE_FOOD_MERCHANTS:
        return False

    return True


def aggregate_expenses(transactions: Sequence[Transaction], include_bank_fees: bool = False) -> Dict[str, float]:
    totals = dict(BASE_TOTALS)

    excluded_types = set(EXCLUDED_TYPES_DEFAULT)
    if include_bank_fees:
        excluded_types.discard(TransactionType.BANK_FEE.value)

    for txn in transactions:
        amount = abs(float(txn.amount))

        if txn.direction == "credit" and txn.txn_type not in {
            TransactionType.INTERNAL_TRANSFER.value,
            TransactionType.BLOCKED_AMOUNT.value,
        }:
            totals["income_total"] += amount
            continue

        if txn.direction != "debit":
            continue
        if txn.txn_type in excluded_types:
            continue

        category = txn.category_area or CategoryArea.UNKNOWN.value
        if not _is_transfer_expense_eligible(txn, category):
            continue

        total_key = CATEGORY_TO_TOTAL_KEY.get(category, "unknown_total")
        totals[total_key] += amount
        totals["outgoing_expense_total"] += amount
        totals["outgoing_tx_count"] += 1.0

        if category in ESSENTIAL_CATEGORIES:
            totals["essential_total"] += amount
        else:
            totals["nonessential_total"] += amount

        if (txn.merchant_canonical or "") in ELECTRONICS_MERCHANTS:
            totals["electronics_gadgets_total"] += amount

        impulse_bucket = _impulse_bucket(txn, category)
        if _is_impulse_candidate(txn, category, amount, impulse_bucket):
            totals["impulse_candidate_tx_count"] += 1.0
            totals[f"impulse_spend_{impulse_bucket}"] += amount
            totals[f"impulse_tx_count_{impulse_bucket}"] += 1.0

    for key, value in list(totals.items()):
        totals[key] = round(value, 2)

    return totals

