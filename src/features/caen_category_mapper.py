from __future__ import annotations

import re
from typing import Optional, Sequence

from src.domain.enums import CategoryArea


# Prefix maps keep logic data-driven and avoid merchant-level hardcoding.
_EXPENSE_PREFIX_TO_CATEGORY = {
    "10": CategoryArea.FOOD.value,
    "11": CategoryArea.FOOD.value,
    "47": CategoryArea.FOOD.value,
    "56": CategoryArea.FOOD.value,
    "35": CategoryArea.HOUSING.value,
    "36": CategoryArea.HOUSING.value,
    "61": CategoryArea.HOUSING.value,
    "68": CategoryArea.HOUSING.value,
    "45": CategoryArea.TRANSPORT.value,
    "49": CategoryArea.TRANSPORT.value,
    "50": CategoryArea.TRANSPORT.value,
    "52": CategoryArea.TRANSPORT.value,
    "59": CategoryArea.ENTERTAINMENT.value,
    "90": CategoryArea.ENTERTAINMENT.value,
    "93": CategoryArea.ENTERTAINMENT.value,
    "21": CategoryArea.HEALTH.value,
    "86": CategoryArea.HEALTH.value,
    "32": CategoryArea.HEALTH.value,
    "96": CategoryArea.PERSONAL_CARE.value,
    "85": CategoryArea.CHILD_EDUCATION.value,
}

_EXPENSE_CODE_TO_CATEGORY = {
    "3250": CategoryArea.HEALTH.value,
    "4646": CategoryArea.HEALTH.value,
    "4773": CategoryArea.HEALTH.value,
    "4775": CategoryArea.PERSONAL_CARE.value,
    "4932": CategoryArea.TRANSPORT.value,
    "4939": CategoryArea.TRANSPORT.value,
    "5221": CategoryArea.TRANSPORT.value,
    "5510": CategoryArea.ENTERTAINMENT.value,
}

_IMPULSE_PREFIX_TO_CATEGORY = {
    "47": "Food",
    "56": "Food",
    "59": "Entertainment",
    "90": "Entertainment",
    "93": "Entertainment",
}

_IMPULSE_CODE_TO_CATEGORY = {
    "4741": "Electronics or gadgets",
    "4742": "Electronics or gadgets",
    "4743": "Electronics or gadgets",
    "4651": "Electronics or gadgets",
    "4652": "Electronics or gadgets",
    "9521": "Electronics or gadgets",
    "9522": "Electronics or gadgets",
    "4771": "Clothing or personal care products",
    "4772": "Clothing or personal care products",
    "4775": "Clothing or personal care products",
    "9602": "Clothing or personal care products",
    "9604": "Clothing or personal care products",
    "5914": "Entertainment",
    "9321": "Entertainment",
}

_PRODUCT_LIFETIME_CODE_TO_BUCKET = {
    "4511": "Cars",
    "4519": "Cars",
    "4520": "Cars",
    "4531": "Cars",
    "4532": "Cars",
    "4741": "Tech",
    "4742": "Tech",
    "4743": "Tech",
    "4651": "Tech",
    "4652": "Tech",
    "4754": "Appliances",
    "9522": "Appliances",
    "4771": "Clothing",
    "4772": "Clothing",
}


def _normalize_caen_code(code: str | int | None) -> Optional[str]:
    if code is None:
        return None
    digits = re.sub(r"\D", "", str(code))
    if not digits:
        return None
    return digits[:4].zfill(4)


def map_caen_to_expense_category(code: str | int | None) -> Optional[str]:
    normalized = _normalize_caen_code(code)
    if not normalized:
        return None
    if normalized in _EXPENSE_CODE_TO_CATEGORY:
        return _EXPENSE_CODE_TO_CATEGORY[normalized]
    return _EXPENSE_PREFIX_TO_CATEGORY.get(normalized[:2])


def map_caen_to_impulse_buying_category(code: str | int | None) -> str:
    normalized = _normalize_caen_code(code)
    if not normalized:
        return "Other"
    if normalized in _IMPULSE_CODE_TO_CATEGORY:
        return _IMPULSE_CODE_TO_CATEGORY[normalized]
    return _IMPULSE_PREFIX_TO_CATEGORY.get(normalized[:2], "Other")


def map_caen_to_product_lifetime_bucket(code: str | int | None) -> Optional[str]:
    normalized = _normalize_caen_code(code)
    if not normalized:
        return None
    return _PRODUCT_LIFETIME_CODE_TO_BUCKET.get(normalized)


def choose_primary_expense_category(codes: Sequence[str | int | None]) -> Optional[str]:
    if not codes:
        return None

    priority = [
        CategoryArea.FOOD.value,
        CategoryArea.HOUSING.value,
        CategoryArea.TRANSPORT.value,
        CategoryArea.HEALTH.value,
        CategoryArea.PERSONAL_CARE.value,
        CategoryArea.CHILD_EDUCATION.value,
        CategoryArea.ENTERTAINMENT.value,
        CategoryArea.OTHER.value,
    ]
    rank = {name: idx for idx, name in enumerate(priority)}

    counts: dict[str, int] = {}
    for code in codes:
        category = map_caen_to_expense_category(code)
        if not category:
            continue
        counts[category] = counts.get(category, 0) + 1

    if not counts:
        return None

    return sorted(counts.items(), key=lambda item: (-item[1], rank.get(item[0], 999)))[0][0]

