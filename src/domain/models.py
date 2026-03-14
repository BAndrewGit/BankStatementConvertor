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
        )

