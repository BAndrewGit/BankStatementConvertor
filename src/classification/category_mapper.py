from __future__ import annotations

from dataclasses import replace
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from src.domain.enums import CategoryArea, TransactionType
from src.domain.models import Transaction
from src.infrastructure.cache import CacheRepository


ELIGIBLE_TYPES = {
    TransactionType.CARD_PURCHASE.value,
    TransactionType.SUBSCRIPTION.value,
    TransactionType.UTILITY_PAYMENT.value,
}


class CategoryMapper:
    def __init__(self, rules_yaml_path: Optional[str] = None, cache_repo: Optional[CacheRepository] = None) -> None:
        self._rules_yaml_path = rules_yaml_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "category_rules.yaml",
        )
        self._cache_repo = cache_repo
        self._merchant_categories: Dict[str, str] = {}
        self._regex_rules: List[Tuple[re.Pattern[str], str]] = []
        self._fallback_by_txn_type: Dict[str, str] = {}
        self._load_rules()

    def map(self, txn: Transaction) -> Transaction:
        if txn.txn_type not in ELIGIBLE_TYPES:
            return replace(txn, category_area=None)

        if not txn.merchant_canonical:
            return replace(
                txn,
                category_area=CategoryArea.UNKNOWN.value,
                mapping_method="unmapped",
            )

        merchant = txn.merchant_canonical.strip()
        cached = self._cache_repo.get("category", merchant) if self._cache_repo else None
        if cached:
            return replace(txn, category_area=str(cached), mapping_method=txn.mapping_method or "exact_alias")

        if merchant in self._merchant_categories:
            category = self._merchant_categories[merchant]
            if self._cache_repo:
                self._cache_repo.set("category", merchant, category)
            method = txn.mapping_method if txn.mapping_method in {"exact_alias", "fuzzy_alias"} else "exact_alias"
            return replace(txn, category_area=category, mapping_method=method)

        for pattern, category in self._regex_rules:
            if pattern.search(merchant):
                if self._cache_repo:
                    self._cache_repo.set("category", merchant, category)
                return replace(txn, category_area=category, mapping_method="regex_rule")

        fallback = self._fallback_by_txn_type.get(txn.txn_type)
        if fallback:
            if self._cache_repo:
                self._cache_repo.set("category", merchant, fallback)
            return replace(txn, category_area=fallback, mapping_method="fallback_rule")

        return replace(
            txn,
            category_area=CategoryArea.UNKNOWN.value,
            mapping_method="unmapped",
        )

    def map_many(self, transactions: Sequence[Transaction]) -> Tuple[List[Transaction], Dict[str, float]]:
        mapped = [self.map(txn) for txn in transactions]

        eligible = [txn for txn in mapped if txn.txn_type in ELIGIBLE_TYPES]
        unknown = [txn for txn in eligible if txn.category_area == CategoryArea.UNKNOWN.value]

        summary = {
            "eligible_for_category": float(len(eligible)),
            "unknown_category": float(len(unknown)),
            "unknown_rate": round((len(unknown) / len(eligible)), 4) if eligible else 0.0,
        }
        return mapped, summary

    def _load_rules(self) -> None:
        with open(self._rules_yaml_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        self._merchant_categories = {
            str(key).strip(): str(value).strip()
            for key, value in (payload.get("merchant_categories") or {}).items()
            if str(key).strip() and str(value).strip()
        }

        self._regex_rules = []
        for item in payload.get("regex_rules", []) or []:
            pattern_raw = str((item or {}).get("pattern", "")).strip()
            category = str((item or {}).get("category_area", "")).strip()
            if not pattern_raw or not category:
                continue
            self._regex_rules.append((re.compile(pattern_raw, re.IGNORECASE), category))

        self._fallback_by_txn_type = {
            str(key).strip(): str(value).strip()
            for key, value in (payload.get("fallback_by_txn_type") or {}).items()
            if str(key).strip() and str(value).strip()
        }


def map_categories(transactions: Sequence[Transaction], rules_yaml_path: Optional[str] = None) -> Tuple[List[Transaction], Dict[str, float]]:
    return CategoryMapper(rules_yaml_path=rules_yaml_path).map_many(transactions)

