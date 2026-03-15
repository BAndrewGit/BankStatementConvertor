from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class Transaction:
    transaction_id: str
    booking_date: str
    amount: float
    currency: str
    direction: str  # debit | credit
    raw_description: str
    source_section: str  # booked_transactions | blocked_amounts

    channel: Optional[str] = None
    txn_type: Optional[str] = None
    merchant_raw: Optional[str] = None
    merchant_canonical: Optional[str] = None
    normalization_source: Optional[str] = None
    merchant_match_method: Optional[str] = None
    category_area: Optional[str] = None
    mapping_method: Optional[str] = None
    confidence: Optional[float] = None
    recipient_name: Optional[str] = None
    recipient_iban: Optional[str] = None
    entity_key: Optional[str] = None
    state: Optional[str] = None
    evidence_type: Optional[str] = None
    match_source: Optional[str] = None
    fuzzy_match_term: Optional[str] = None
    fuzzy_match_score: Optional[float] = None
    final_spend_category: Optional[str] = None
    is_essential: Optional[bool] = None
    is_nonessential: Optional[bool] = None
    is_internal_transfer: Optional[bool] = None
    is_salary: Optional[bool] = None
    is_housing: Optional[bool] = None
    is_utility: Optional[bool] = None
    is_impulse_candidate: Optional[bool] = None
    impulse_category: Optional[str] = None
    classification_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Transaction":
        return cls(
            transaction_id=str(payload.get("transaction_id", "")),
            booking_date=str(payload.get("booking_date", "")),
            amount=float(payload.get("amount", 0.0)),
            currency=str(payload.get("currency", "")),
            direction=str(payload.get("direction", "")),
            raw_description=str(payload.get("raw_description", "")),
            source_section=str(payload.get("source_section", "")),
            channel=payload.get("channel"),
            txn_type=payload.get("txn_type"),
            merchant_raw=payload.get("merchant_raw"),
            merchant_canonical=payload.get("merchant_canonical"),
            normalization_source=payload.get("normalization_source"),
            merchant_match_method=payload.get("merchant_match_method"),
            category_area=payload.get("category_area"),
            mapping_method=payload.get("mapping_method"),
            confidence=payload.get("confidence"),
            recipient_name=payload.get("recipient_name"),
            recipient_iban=payload.get("recipient_iban"),
            entity_key=payload.get("entity_key"),
            state=payload.get("state"),
            evidence_type=payload.get("evidence_type"),
            match_source=payload.get("match_source"),
            fuzzy_match_term=payload.get("fuzzy_match_term"),
            fuzzy_match_score=payload.get("fuzzy_match_score"),
            final_spend_category=payload.get("final_spend_category"),
            is_essential=payload.get("is_essential"),
            is_nonessential=payload.get("is_nonessential"),
            is_internal_transfer=payload.get("is_internal_transfer"),
            is_salary=payload.get("is_salary"),
            is_housing=payload.get("is_housing"),
            is_utility=payload.get("is_utility"),
            is_impulse_candidate=payload.get("is_impulse_candidate"),
            impulse_category=payload.get("impulse_category"),
            classification_confidence=payload.get("classification_confidence"),
        )

