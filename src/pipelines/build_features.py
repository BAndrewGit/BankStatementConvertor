from __future__ import annotations

from typing import Dict, Sequence

from src.domain.models import Transaction
from src.features.expense_aggregator import aggregate_expenses
from src.features.feature_builder import build_feature_vector


def build_features(transactions: Sequence[Transaction], include_bank_fees: bool = False) -> Dict[str, float]:
    totals = aggregate_expenses(transactions, include_bank_fees=include_bank_fees)
    return build_feature_vector(totals)

