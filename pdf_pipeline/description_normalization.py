from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from .transaction_normalization import TransactionNormalized


LOGGER = logging.getLogger(__name__)

TECHNICAL_PATTERNS = [
    re.compile(r"\bTID[:=]?\s*[A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bRRN[:=]?\s*[A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bREF[:=]?\s*[A-Z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\bTERMINAL[:=]?\s*[A-Z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\bCARD\b\s*[*X0-9]{4,}", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{8,}\b", re.IGNORECASE),
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\bWWW\.[A-Z0-9.-]+\b", re.IGNORECASE),
]

NOISE_TOKENS: Set[str] = {
    "POS",
    "NONBT",
    "NON",
    "BT",
    "BTPAY",
    "ONLINE",
    "WWW",
    "PAYMENT",
    "TRANZACTIE",
    "TRNZ",
    "NR",
    "NO",
    "TERM",
    "TERMINAL",
    "CARD",
    "VISA",
    "MASTERCARD",
    "CONTACTLESS",
    "DEBIT",
    "CREDIT",
    "DATA",
    "ORA",
}

TRANSACTION_TYPE_TOKENS: Set[str] = {
    "PLATA",
    "COMISION",
    "RETRAGERE",
    "TRANSFER",
    "ALIMENTARE",
    "INCASARE",
    "NUMERAR",
    "ATM",
    "INTERN",
    "EXTERN",
    "CANAL",
    "ELECTRONIC",
    "P2P",
}

MERCHANT_CONNECTOR_TOKENS: Set[str] = {"LA", "DE", "DIN", "PE", "CATRE", "CU", "SI"}

NON_MERCHANT_MARKERS = [
    "COMISION",
    "RETRAGERE",
    "ATM",
    "TRANSFER",
    "P2P",
    "TAXA",
]


@dataclass(frozen=True)
class DescriptionNormalizationIssue:
    candidate_id: str
    issue_code: str
    raw_value: str
    message: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TransactionDescriptionNormalized:
    candidate_id: str
    source_pdf: str
    source_page: int
    description_clean: str
    merchant_raw_candidate: Optional[str]
    description_normalization_confidence: float
    description_status: str
    description_warnings: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _remove_technical_fragments(text: str) -> str:
    cleaned = text
    for pattern in TECHNICAL_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def _normalize_description_text(text: Optional[str]) -> str:
    if not text:
        return ""

    cleaned = _strip_diacritics(text)
    cleaned = cleaned.upper()
    cleaned = _remove_technical_fragments(cleaned)
    cleaned = re.sub(r"[^A-Z0-9\s&.-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    kept_tokens: List[str] = []
    for token in cleaned.split():
        token_compact = token.strip(".-")
        if not token_compact:
            continue
        if token_compact in NOISE_TOKENS:
            continue
        if token_compact.isdigit() and len(token_compact) >= 2:
            continue
        # Drop technical identifiers, not regular long words.
        if re.fullmatch(r"(?=.*[A-Z])(?=.*\d)[A-Z0-9]{8,}", token_compact):
            continue
        kept_tokens.append(token_compact)

    return " ".join(kept_tokens)


def _extract_merchant_candidate(description_clean: str) -> Tuple[Optional[str], List[str]]:
    warnings: List[str] = []

    if not description_clean:
        warnings.append("missing_description")
        return None, warnings

    if any(marker in description_clean for marker in NON_MERCHANT_MARKERS):
        warnings.append("non_merchant_transaction")
        return None, warnings

    merchant_tokens: List[str] = []
    for token in description_clean.split():
        if token in TRANSACTION_TYPE_TOKENS:
            continue
        if token in MERCHANT_CONNECTOR_TOKENS:
            continue
        if re.fullmatch(r"\d+[.,]?\d*", token):
            continue
        merchant_tokens.append(token)

    if not merchant_tokens:
        warnings.append("merchant_not_found")
        return None, warnings

    merchant = " ".join(merchant_tokens[:5]).strip()
    if len(merchant) < 3:
        warnings.append("merchant_too_short")
        return None, warnings

    return merchant, warnings


def _compute_confidence(base_confidence: float, description_clean: str, merchant: Optional[str], warnings: List[str]) -> float:
    confidence = base_confidence

    if not description_clean:
        confidence -= 0.60
    if not merchant and "non_merchant_transaction" not in warnings:
        confidence -= 0.25
    if "merchant_too_short" in warnings:
        confidence -= 0.20
    if "missing_description" in warnings:
        confidence -= 0.25
    if "merchant_not_found" in warnings:
        confidence -= 0.15

    return max(0.0, round(confidence, 2))


def normalize_transaction_descriptions(
    rows: List[TransactionNormalized],
) -> Tuple[List[TransactionDescriptionNormalized], List[DescriptionNormalizationIssue], Dict[str, int]]:
    """Clean transaction descriptions and extract merchant candidates for Story 5."""
    normalized_rows: List[TransactionDescriptionNormalized] = []
    issues: List[DescriptionNormalizationIssue] = []

    for row in rows:
        clean_description = _normalize_description_text(row.description)
        merchant, warnings_list = _extract_merchant_candidate(clean_description)
        warnings = "|".join(sorted(set(warnings_list)))

        if "missing_description" in warnings_list:
            issues.append(
                DescriptionNormalizationIssue(
                    candidate_id=row.candidate_id,
                    issue_code="missing_description",
                    raw_value=row.description or "",
                    message="Descrierea lipseste si nu poate fi normalizata.",
                )
            )
            LOGGER.warning("Description issue [%s] missing_description", row.candidate_id)

        if "merchant_not_found" in warnings_list:
            issues.append(
                DescriptionNormalizationIssue(
                    candidate_id=row.candidate_id,
                    issue_code="merchant_not_found",
                    raw_value=clean_description,
                    message="Nu a fost identificat un merchant valid din descriere.",
                )
            )
            LOGGER.warning("Description issue [%s] merchant_not_found", row.candidate_id)

        confidence = _compute_confidence(
            base_confidence=row.normalization_confidence,
            description_clean=clean_description,
            merchant=merchant,
            warnings=warnings_list,
        )

        status = "description_ok"
        if "missing_description" in warnings_list:
            status = "description_failed"
        elif warnings:
            status = "description_with_warnings"

        normalized_rows.append(
            TransactionDescriptionNormalized(
                candidate_id=row.candidate_id,
                source_pdf=row.source_pdf,
                source_page=row.source_page,
                description_clean=clean_description,
                merchant_raw_candidate=merchant,
                description_normalization_confidence=confidence,
                description_status=status,
                description_warnings=warnings,
            )
        )

    summary = {
        "total_rows": len(rows),
        "rows_with_merchant": sum(1 for item in normalized_rows if item.merchant_raw_candidate),
        "rows_without_merchant": sum(1 for item in normalized_rows if not item.merchant_raw_candidate),
        "issues_count": len(issues),
    }

    return normalized_rows, issues, summary


def save_transactions_description_normalized_csv(
    rows: List[TransactionDescriptionNormalized], output_csv_path: str
) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_id",
                "source_pdf",
                "source_page",
                "description_clean",
                "merchant_raw_candidate",
                "description_normalization_confidence",
                "description_status",
                "description_warnings",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def save_description_normalization_issues_csv(
    issues: List[DescriptionNormalizationIssue], output_csv_path: str
) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["candidate_id", "issue_code", "raw_value", "message"],
        )
        writer.writeheader()
        for row in issues:
            writer.writerow(row.to_dict())


