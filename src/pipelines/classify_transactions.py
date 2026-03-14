from __future__ import annotations

from typing import Dict, List, Tuple

from src.classification.category_mapper import CategoryMapper
from src.classification.txn_type_classifier import classify_transactions
from src.classification.merchant_extractor import MerchantExtractor
from src.classification.merchant_normalizer import MerchantNormalizer
from src.domain.models import Transaction
from src.infrastructure.cache import CacheRepository


def classify_parsed_transactions(
    transactions: List[Transaction],
    cache_repo: CacheRepository | None = None,
) -> Tuple[List[Transaction], Dict[str, float]]:
    typed_transactions, summary = classify_transactions(transactions)

    extractor = MerchantExtractor()
    normalizer = MerchantNormalizer(cache_repo=cache_repo)
    category_mapper = CategoryMapper(cache_repo=cache_repo)

    with_merchants = [extractor.extract(txn) for txn in typed_transactions]
    normalized = [normalizer.normalize(txn) for txn in with_merchants]
    categorized, category_summary = category_mapper.map_many(normalized)

    extracted_count = sum(1 for txn in categorized if txn.merchant_raw)
    normalized_count = sum(1 for txn in categorized if txn.merchant_canonical)
    dataset_normalized_count = sum(1 for txn in categorized if txn.normalization_source == "dataset")
    termene_normalized_count = sum(1 for txn in categorized if txn.normalization_source == "termene")
    alias_normalized_count = sum(
        1 for txn in categorized if (txn.normalization_source or "").startswith("alias_")
    )
    unmapped_normalized_count = sum(1 for txn in categorized if txn.normalization_source == "unmapped")

    summary.update(
        {
            "merchant_extracted": float(extracted_count),
            "merchant_normalized": float(normalized_count),
            "merchant_normalized_from_dataset": float(dataset_normalized_count),
            "merchant_normalized_from_termene": float(termene_normalized_count),
            "merchant_normalized_from_alias": float(alias_normalized_count),
            "merchant_normalized_unmapped": float(unmapped_normalized_count),
            "eligible_for_category": category_summary["eligible_for_category"],
            "unknown_category": category_summary["unknown_category"],
            "unknown_rate": category_summary["unknown_rate"],
        }
    )

    return categorized, summary

