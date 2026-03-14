from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

from .transaction_parsing import TransactionRaw


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizationIssue:
    candidate_id: str
    field_name: str
    issue_code: str
    raw_value: str
    message: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TransactionNormalized:
    candidate_id: str
    source_pdf: str
    source_page: int
    transaction_date: Optional[str]
    description: Optional[str]
    amount: Optional[float]
    balance: Optional[float]
    currency: Optional[str]
    direction: Optional[str]  # in | out
    is_valid: bool
    normalization_status: str
    normalization_confidence: float
    invalid_reasons: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalize_date(date_raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not date_raw:
        return None, "missing"

    try:
        date_obj = datetime.strptime(date_raw, "%d/%m/%Y")
    except ValueError:
        return None, "invalid_format"

    return date_obj.strftime("%Y-%m-%d"), None


def _normalize_decimal(value_raw: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not value_raw:
        return None, "missing"

    normalized = value_raw.strip().replace(" ", "")

    if "," in normalized and "." in normalized:
        normalized = normalized.replace(".", "")
        normalized = normalized.replace(",", ".")
    elif "," in normalized:
        normalized = normalized.replace(",", ".")

    try:
        value = Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None, "invalid_number"

    return float(value), None


def _normalize_currency(currency_raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not currency_raw:
        return None, "missing"

    normalized = currency_raw.strip().upper()
    if len(normalized) != 3 or not normalized.isalpha():
        return None, "invalid_currency"

    return normalized, None


def _normalize_direction(direction_raw: Optional[str], amount: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    if direction_raw == "inflow":
        return "in", None
    if direction_raw == "outflow":
        return "out", None

    if amount is not None:
        if amount > 0:
            return "out", "inferred_from_positive_amount"
        if amount < 0:
            return "in", "inferred_from_negative_amount"

    return None, "unknown_direction"


def _add_issue(
    issues: List[NormalizationIssue],
    candidate_id: str,
    field_name: str,
    issue_code: str,
    raw_value: Optional[str],
    message: str,
) -> None:
    issue = NormalizationIssue(
        candidate_id=candidate_id,
        field_name=field_name,
        issue_code=issue_code,
        raw_value=raw_value or "",
        message=message,
    )
    issues.append(issue)
    LOGGER.warning("Normalization issue [%s] %s.%s: %s", candidate_id, field_name, issue_code, message)


def normalize_transactions_raw(
    rows: List[TransactionRaw],
) -> Tuple[List[TransactionNormalized], List[NormalizationIssue], Dict[str, int]]:
    """Normalize Story 3 raw values into canonical date/amount/currency/direction fields."""
    normalized_rows: List[TransactionNormalized] = []
    issues: List[NormalizationIssue] = []

    for row in rows:
        invalid_reasons: List[str] = []

        date_value, date_issue = _normalize_date(row.transaction_date_raw)
        if date_issue:
            invalid_reasons.append(f"date_{date_issue}")
            _add_issue(
                issues,
                row.candidate_id,
                "transaction_date",
                f"date_{date_issue}",
                row.transaction_date_raw,
                "Data nu poate fi normalizata in format ISO.",
            )

        amount_value, amount_issue = _normalize_decimal(row.amount_raw)
        if amount_issue:
            invalid_reasons.append(f"amount_{amount_issue}")
            _add_issue(
                issues,
                row.candidate_id,
                "amount",
                f"amount_{amount_issue}",
                row.amount_raw,
                "Suma tranzactiei lipseste sau nu este numerica valida.",
            )

        balance_value, balance_issue = _normalize_decimal(row.balance_raw)
        if balance_issue == "invalid_number":
            invalid_reasons.append("balance_invalid_number")
            _add_issue(
                issues,
                row.candidate_id,
                "balance",
                "balance_invalid_number",
                row.balance_raw,
                "Soldul nu este numeric valid.",
            )

        currency_value, currency_issue = _normalize_currency(row.currency_raw)
        if currency_issue:
            invalid_reasons.append(f"currency_{currency_issue}")
            _add_issue(
                issues,
                row.candidate_id,
                "currency",
                f"currency_{currency_issue}",
                row.currency_raw,
                "Moneda lipseste sau nu respecta codul ISO de 3 litere.",
            )

        direction_value, direction_issue = _normalize_direction(row.transaction_direction_raw, amount_value)
        if direction_issue:
            invalid_reasons.append(direction_issue)
            _add_issue(
                issues,
                row.candidate_id,
                "direction",
                direction_issue,
                row.transaction_direction_raw,
                "Sensul tranzactiei nu a putut fi standardizat la in/out.",
            )

        is_valid = len(invalid_reasons) == 0
        status = "normalized_ok" if is_valid else "normalized_with_issues"

        confidence = 1.0
        confidence -= 0.30 if date_issue else 0.0
        confidence -= 0.35 if amount_issue else 0.0
        confidence -= 0.10 if currency_issue else 0.0
        confidence -= 0.20 if direction_issue else 0.0
        confidence -= 0.05 if balance_issue == "invalid_number" else 0.0
        confidence = max(0.0, round(confidence, 2))

        normalized_rows.append(
            TransactionNormalized(
                candidate_id=row.candidate_id,
                source_pdf=row.source_pdf,
                source_page=row.source_page,
                transaction_date=date_value,
                description=row.description_raw,
                amount=amount_value,
                balance=balance_value,
                currency=currency_value,
                direction=direction_value,
                is_valid=is_valid,
                normalization_status=status,
                normalization_confidence=confidence,
                invalid_reasons="|".join(sorted(set(invalid_reasons))),
            )
        )

    summary = {
        "total_rows": len(rows),
        "valid_rows": sum(1 for item in normalized_rows if item.is_valid),
        "rows_with_issues": sum(1 for item in normalized_rows if not item.is_valid),
        "issues_count": len(issues),
    }

    return normalized_rows, issues, summary


def save_transactions_normalized_csv(rows: List[TransactionNormalized], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_id",
                "source_pdf",
                "source_page",
                "transaction_date",
                "description",
                "amount",
                "balance",
                "currency",
                "direction",
                "is_valid",
                "normalization_status",
                "normalization_confidence",
                "invalid_reasons",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def save_normalization_issues_csv(issues: List[NormalizationIssue], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["candidate_id", "field_name", "issue_code", "raw_value", "message"],
        )
        writer.writeheader()
        for row in issues:
            writer.writerow(row.to_dict())


