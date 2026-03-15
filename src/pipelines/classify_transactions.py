from __future__ import annotations

from dataclasses import replace
import json
import os
from typing import Dict, List, Tuple

from src.classification.category_mapper import CategoryMapper
from src.classification.transfer_bootstrap_classifier import TransferBootstrapClassifier
from src.classification.txn_type_classifier import classify_transactions
from src.classification.merchant_extractor import MerchantExtractor
from src.classification.merchant_normalizer import MerchantNormalizer
from src.domain.models import Transaction
from src.infrastructure.cache import CacheRepository


ESSENTIAL_CATEGORIES = {"Food", "Housing", "Transport", "Health"}
UTILITY_HINTS: set[str] = set()
IMPULSE_QUICK_FOOD_HINTS: set[str] = set()


def _load_bootstrap_config() -> None:
    global UTILITY_HINTS
    global IMPULSE_QUICK_FOOD_HINTS

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "bootstrap_dictionary.json",
    )
    try:
        with open(config_path, encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    except Exception:
        payload = {}

    housing = payload.get("housing") if isinstance(payload.get("housing"), dict) else {}
    utilities = payload.get("utilities") if isinstance(payload.get("utilities"), dict) else {}

    util_terms = []
    util_terms.extend(housing.get("merchant_terms", []) or [])
    util_terms.extend(utilities.get("merchant_terms", []) or [])
    UTILITY_HINTS = {str(term).strip().lower() for term in util_terms if str(term).strip()}

    IMPULSE_QUICK_FOOD_HINTS = {
        "pranzo by ess",
        "taco bell",
        "froo",
        "jkc restaurants",
    }


def _normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def _infer_impulse_category(txn: Transaction) -> str | None:
    canonical = _normalize_text(txn.merchant_canonical or "")
    category = txn.category_area or ""

    if category == "Personal_Care":
        return "clothing_personal_care"
    if canonical in {"emag", "altex"}:
        return "electronics_gadgets"
    if category == "Entertainment":
        return "entertainment"
    if category == "Food" and canonical in IMPULSE_QUICK_FOOD_HINTS:
        return "food"
    if category in {"Other", "Unknown"}:
        return "other"
    return None


def _enrich_transaction(txn: Transaction) -> Transaction:
    final_category = txn.category_area or "Unknown"
    merchant_norm = _normalize_text(txn.merchant_canonical or "")

    is_internal_transfer = txn.txn_type == "internal_transfer"
    is_salary = txn.txn_type == "salary_income"
    is_housing = final_category == "Housing"
    is_utility = any(term and term in merchant_norm for term in UTILITY_HINTS)

    is_essential = final_category in ESSENTIAL_CATEGORIES
    is_nonessential = (
        txn.direction == "debit"
        and not is_internal_transfer
        and not is_salary
        and not is_essential
        and txn.txn_type not in {"bank_fee", "blocked_amount", "cash_withdrawal"}
    )

    impulse_category = _infer_impulse_category(txn)
    is_impulse_candidate = bool(
        is_nonessential
        and not is_housing
        and not is_utility
        and txn.txn_type not in {"external_transfer", "internal_transfer", "bank_fee"}
        and impulse_category
        and abs(float(txn.amount or 0.0)) <= 300.0
    )

    if not is_impulse_candidate:
        impulse_category = None

    return replace(
        txn,
        final_spend_category=final_category,
        is_essential=is_essential,
        is_nonessential=is_nonessential,
        is_internal_transfer=is_internal_transfer,
        is_salary=is_salary,
        is_housing=is_housing,
        is_utility=is_utility,
        is_impulse_candidate=is_impulse_candidate,
        impulse_category=impulse_category,
        classification_confidence=txn.confidence,
    )


_load_bootstrap_config()


def classify_parsed_transactions(
    transactions: List[Transaction],
    cache_repo: CacheRepository | None = None,
    profile_id: str | None = None,
) -> Tuple[List[Transaction], Dict[str, float]]:
    typed_transactions, summary = classify_transactions(transactions)

    extractor = MerchantExtractor()
    normalizer = MerchantNormalizer(cache_repo=cache_repo)
    category_mapper = CategoryMapper(cache_repo=cache_repo)
    transfer_classifier = TransferBootstrapClassifier(profile_id=profile_id)

    with_merchants = [extractor.extract(txn) for txn in typed_transactions]
    normalized = [normalizer.normalize(txn) for txn in with_merchants]
    categorized, category_summary = category_mapper.map_many(normalized)
    enriched = transfer_classifier.classify_many(categorized)

    extracted_count = sum(1 for txn in enriched if txn.merchant_raw)
    normalized_count = sum(1 for txn in enriched if txn.merchant_canonical)
    dataset_normalized_count = sum(1 for txn in enriched if txn.normalization_source == "dataset")
    termene_normalized_count = sum(1 for txn in enriched if txn.normalization_source == "termene")
    alias_normalized_count = sum(
        1 for txn in enriched if (txn.normalization_source or "").startswith("alias_")
    )
    unmapped_normalized_count = sum(1 for txn in enriched if txn.normalization_source == "unmapped")

    transfer_candidates = sum(1 for txn in enriched if (txn.state or "") == "candidate")
    transfer_bootstrap_exact = sum(1 for txn in enriched if (txn.match_source or "") == "bootstrap_exact")
    transfer_bootstrap_fuzzy = sum(1 for txn in enriched if (txn.match_source or "") == "bootstrap_fuzzy")

    enriched = [_enrich_transaction(txn) for txn in enriched]

    summary.update(
        {
            "merchant_extracted": float(extracted_count),
            "merchant_normalized": float(normalized_count),
            "merchant_normalized_from_dataset": float(dataset_normalized_count),
            "merchant_normalized_from_termene": float(termene_normalized_count),
            "merchant_normalized_from_alias": float(alias_normalized_count),
            "merchant_normalized_unmapped": float(unmapped_normalized_count),
            "transfer_candidate_count": float(transfer_candidates),
            "transfer_bootstrap_exact_hits": float(transfer_bootstrap_exact),
            "transfer_bootstrap_fuzzy_hits": float(transfer_bootstrap_fuzzy),
            "eligible_for_category": category_summary["eligible_for_category"],
            "unknown_category": category_summary["unknown_category"],
            "unknown_rate": category_summary["unknown_rate"],
        }
    )

    return enriched, summary

