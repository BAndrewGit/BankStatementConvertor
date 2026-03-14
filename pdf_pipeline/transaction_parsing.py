from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .transaction_candidates import TransactionCandidate


DATE_PATTERN = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
AMOUNT_PATTERN = re.compile(r"([+-]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))")
BALANCE_WITH_KEYWORD_PATTERN = re.compile(
    r"\b(?:sold|balance)\b[^\d+-]{0,25}([+-]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))",
    re.IGNORECASE,
)
CURRENCY_PATTERN = re.compile(r"\b(RON|EUR|USD|GBP)\b", re.IGNORECASE)

INFLOW_HINTS = ["incasare", "alimentare", "credit", "virament intrat"]
OUTFLOW_HINTS = ["plata", "debit", "retragere", "comision", "taxa"]


@dataclass(frozen=True)
class TransactionRaw:
    candidate_id: str
    source_pdf: str
    source_page: int
    source_page_span: str
    source_text: str
    transaction_date_raw: Optional[str]
    description_raw: Optional[str]
    amount_raw: Optional[str]
    transaction_direction_raw: str
    balance_raw: Optional[str]
    currency_raw: Optional[str]
    parse_status: str
    parse_confidence: float
    parse_warnings: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _extract_currency(text: str) -> Optional[str]:
    match = CURRENCY_PATTERN.search(text)
    return match.group(1).upper() if match else None


def _extract_date(text: str) -> Optional[str]:
    match = DATE_PATTERN.search(text)
    return match.group(1) if match else None


def _extract_amounts(text: str) -> List[str]:
    return AMOUNT_PATTERN.findall(text)


def _extract_balance(text: str, amounts: Sequence[str]) -> Tuple[Optional[str], Optional[int]]:
    balance_match = BALANCE_WITH_KEYWORD_PATTERN.search(text)
    if balance_match:
        balance = balance_match.group(1)
        for idx, value in enumerate(amounts):
            if value == balance:
                return balance, idx
        return balance, None

    if len(amounts) >= 2:
        return amounts[-1], len(amounts) - 1

    return None, None


def _infer_direction(text: str, amount_raw: Optional[str]) -> str:
    lower_text = text.lower()

    if any(hint in lower_text for hint in OUTFLOW_HINTS):
        return "outflow"
    if any(hint in lower_text for hint in INFLOW_HINTS):
        return "inflow"

    if amount_raw:
        if amount_raw.startswith("-"):
            return "outflow"
        if amount_raw.startswith("+"):
            return "inflow"

    return "unknown"


def _build_description(text: str, date_raw: Optional[str], amount_raw: Optional[str], balance_raw: Optional[str]) -> Optional[str]:
    cleaned = text
    if date_raw:
        cleaned = cleaned.replace(date_raw, " ", 1)
    if amount_raw:
        cleaned = cleaned.replace(amount_raw, " ", 1)
    if balance_raw and balance_raw != amount_raw:
        cleaned = cleaned.replace(balance_raw, " ", 1)
    cleaned = " ".join(cleaned.split()).strip(" -:;")
    return cleaned or None


def _compute_confidence(
    candidate: TransactionCandidate,
    has_date: bool,
    has_amount: bool,
    has_balance: bool,
    direction: str,
) -> float:
    confidence = 1.0

    if not has_date:
        confidence -= 0.35
    if not has_amount:
        confidence -= 0.45
    if not has_balance:
        confidence -= 0.05
    if direction == "unknown":
        confidence -= 0.15

    if candidate.is_ambiguous:
        confidence -= 0.20

    for reason in candidate.ambiguity_reasons.split("|"):
        reason = reason.strip()
        if reason == "multiple_amounts":
            confidence -= 0.15
        elif reason == "missing_amount":
            confidence -= 0.35
        elif reason == "date_without_amount":
            confidence -= 0.25
        elif reason == "orphan_continuation":
            confidence -= 0.20

    return max(0.0, round(confidence, 2))


def _build_status(has_date: bool, has_amount: bool, confidence: float) -> str:
    if has_date and has_amount and confidence >= 0.75:
        return "parsed_ok"
    if has_date or has_amount:
        return "parsed_with_warnings"
    return "parse_failed"


def parse_transactions_raw(candidates: List[TransactionCandidate]) -> List[TransactionRaw]:
    """Parse Story 2 transaction candidates into Story 3 transactions_raw rows."""
    rows: List[TransactionRaw] = []

    for candidate in candidates:
        text = candidate.text
        amounts = _extract_amounts(text)
        balance_raw, balance_idx = _extract_balance(text, amounts)

        amount_raw = None
        for idx, value in enumerate(amounts):
            if balance_idx is not None and idx == balance_idx:
                continue
            amount_raw = value
            break

        if amount_raw is None and amounts:
            amount_raw = amounts[0]

        date_raw = _extract_date(text)
        currency_raw = _extract_currency(text)
        direction = _infer_direction(text, amount_raw)
        description_raw = _build_description(text, date_raw, amount_raw, balance_raw)

        confidence = _compute_confidence(
            candidate=candidate,
            has_date=bool(date_raw),
            has_amount=bool(amount_raw),
            has_balance=bool(balance_raw),
            direction=direction,
        )
        status = _build_status(bool(date_raw), bool(amount_raw), confidence)

        warnings = candidate.ambiguity_reasons
        if direction == "unknown":
            warnings = "|".join(filter(None, [warnings, "unknown_direction"]))

        rows.append(
            TransactionRaw(
                candidate_id=candidate.candidate_id,
                source_pdf=candidate.source_pdf,
                source_page=candidate.start_page,
                source_page_span=f"{candidate.start_page}-{candidate.end_page}",
                source_text=text,
                transaction_date_raw=date_raw,
                description_raw=description_raw,
                amount_raw=amount_raw,
                transaction_direction_raw=direction,
                balance_raw=balance_raw,
                currency_raw=currency_raw,
                parse_status=status,
                parse_confidence=confidence,
                parse_warnings=warnings,
            )
        )

    return rows


def save_transactions_raw_csv(rows: List[TransactionRaw], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_id",
                "source_pdf",
                "source_page",
                "source_page_span",
                "source_text",
                "transaction_date_raw",
                "description_raw",
                "amount_raw",
                "transaction_direction_raw",
                "balance_raw",
                "currency_raw",
                "parse_status",
                "parse_confidence",
                "parse_warnings",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


