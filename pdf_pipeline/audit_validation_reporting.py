from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import os
from typing import Dict, List, Set, Tuple

from .transaction_parsing import TransactionRaw
from .merchant_resolution import MerchantResolution
from .caen_enrichment import (
    MerchantCaenEnriched,
    load_merchant_resolution_csv as _load_merchant_resolution_csv,
)
from .budget_categorization import (
    BudgetCategorizedTransaction,
    load_budget_categorized_csv as _load_budget_categorized_csv,
    load_caen_enriched_csv as _load_caen_enriched_csv,
)


@dataclass(frozen=True)
class TransactionAuditRow:
    candidate_id: str
    source_pdf: str
    source_page: int
    parse_status: str
    parse_confidence: float
    match_method: str
    match_score: float
    caen_enrichment_status: str
    budget_category: str
    categorization_status: str
    classification_reason: str
    review_required: bool
    audit_reasons: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentQualityReport:
    source_pdf: str
    total_transactions: int
    parse_failed_count: int
    weak_parse_count: int
    merchant_not_found_count: int
    low_match_score_count: int
    caen_missing_count: int
    category_unknown_count: int
    fallback_count: int
    review_required_count: int
    quality_score: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Story12AuditSummary:
    total_transactions: int
    review_required_count: int
    parse_failed_count: int
    weak_parse_count: int
    merchant_not_found_count: int
    low_match_score_count: int
    caen_missing_count: int
    category_unknown_count: int
    fallback_count: int
    documents_count: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def _contains_marker(value: str, marker: str) -> bool:
    value_upper = (value or "").upper()
    return marker.upper() in value_upper


def build_story12_audit(
    raw_rows: List[TransactionRaw],
    resolution_rows: List[MerchantResolution],
    caen_rows: List[MerchantCaenEnriched],
    categorized_rows: List[BudgetCategorizedTransaction],
    parse_confidence_threshold: float = 0.75,
    match_score_threshold: float = 0.75,
) -> Tuple[List[TransactionAuditRow], List[DocumentQualityReport], Story12AuditSummary]:
    """Story 12: build audit rows and minimal quality report per processed document."""
    resolution_by_candidate = {row.candidate_id: row for row in resolution_rows}
    categorized_by_candidate = {row.candidate_id: row for row in categorized_rows}

    caen_status_by_candidate: Dict[str, str] = {}
    for row in caen_rows:
        existing = caen_status_by_candidate.get(row.candidate_id)
        if row.caen_enrichment_status == "caen_attached":
            caen_status_by_candidate[row.candidate_id] = "caen_attached"
        elif not existing:
            caen_status_by_candidate[row.candidate_id] = row.caen_enrichment_status

    audit_rows: List[TransactionAuditRow] = []

    for raw in raw_rows:
        resolution = resolution_by_candidate.get(raw.candidate_id)
        categorized = categorized_by_candidate.get(raw.candidate_id)
        caen_status = caen_status_by_candidate.get(raw.candidate_id, "caen_unknown")

        reasons: Set[str] = set()

        if raw.parse_status == "parse_failed":
            reasons.add("parse_failed")

        if not raw.amount_raw or _contains_marker(raw.parse_warnings, "missing_amount"):
            reasons.add("amount_missing")

        if raw.parse_confidence < parse_confidence_threshold and raw.parse_status != "parse_failed":
            reasons.add("low_parse_confidence")

        if not resolution or resolution.match_method == "no_match":
            reasons.add("merchant_not_found")
        elif resolution.match_score < match_score_threshold:
            reasons.add("low_match_score")

        if resolution and resolution.match_method != "no_match" and caen_status != "caen_attached":
            reasons.add("caen_missing")

        budget_category = categorized.budget_category if categorized else "Unknown"
        classification_reason = categorized.classification_reason if categorized else ""

        if budget_category == "Unknown":
            reasons.add("category_unknown")

        if classification_reason.startswith("rule:fallback:"):
            reasons.add("category_fallback")

        review_required = bool(reasons)
        audit_reasons = "|".join(sorted(reasons))

        audit_rows.append(
            TransactionAuditRow(
                candidate_id=raw.candidate_id,
                source_pdf=raw.source_pdf,
                source_page=raw.source_page,
                parse_status=raw.parse_status,
                parse_confidence=round(raw.parse_confidence, 2),
                match_method=resolution.match_method if resolution else "no_match",
                match_score=round(resolution.match_score, 2) if resolution else 0.0,
                caen_enrichment_status=caen_status,
                budget_category=budget_category,
                categorization_status=(categorized.categorization_status if categorized else "missing_category"),
                classification_reason=classification_reason,
                review_required=review_required,
                audit_reasons=audit_reasons,
            )
        )

    reports_by_pdf: Dict[str, List[TransactionAuditRow]] = {}
    for row in audit_rows:
        reports_by_pdf.setdefault(row.source_pdf, []).append(row)

    document_reports: List[DocumentQualityReport] = []
    for source_pdf in sorted(reports_by_pdf.keys()):
        rows = reports_by_pdf[source_pdf]
        total = len(rows)

        parse_failed_count = sum(1 for item in rows if "parse_failed" in item.audit_reasons)
        weak_parse_count = sum(1 for item in rows if "low_parse_confidence" in item.audit_reasons)
        merchant_not_found_count = sum(1 for item in rows if "merchant_not_found" in item.audit_reasons)
        low_match_score_count = sum(1 for item in rows if "low_match_score" in item.audit_reasons)
        caen_missing_count = sum(1 for item in rows if "caen_missing" in item.audit_reasons)
        category_unknown_count = sum(1 for item in rows if "category_unknown" in item.audit_reasons)
        fallback_count = sum(1 for item in rows if "category_fallback" in item.audit_reasons)
        review_required_count = sum(1 for item in rows if item.review_required)

        quality_score = 1.0
        if total > 0:
            quality_score = round(max(0.0, 1.0 - (review_required_count / total)), 2)

        document_reports.append(
            DocumentQualityReport(
                source_pdf=source_pdf,
                total_transactions=total,
                parse_failed_count=parse_failed_count,
                weak_parse_count=weak_parse_count,
                merchant_not_found_count=merchant_not_found_count,
                low_match_score_count=low_match_score_count,
                caen_missing_count=caen_missing_count,
                category_unknown_count=category_unknown_count,
                fallback_count=fallback_count,
                review_required_count=review_required_count,
                quality_score=quality_score,
            )
        )

    summary = Story12AuditSummary(
        total_transactions=len(audit_rows),
        review_required_count=sum(1 for item in audit_rows if item.review_required),
        parse_failed_count=sum(1 for item in audit_rows if "parse_failed" in item.audit_reasons),
        weak_parse_count=sum(1 for item in audit_rows if "low_parse_confidence" in item.audit_reasons),
        merchant_not_found_count=sum(1 for item in audit_rows if "merchant_not_found" in item.audit_reasons),
        low_match_score_count=sum(1 for item in audit_rows if "low_match_score" in item.audit_reasons),
        caen_missing_count=sum(1 for item in audit_rows if "caen_missing" in item.audit_reasons),
        category_unknown_count=sum(1 for item in audit_rows if "category_unknown" in item.audit_reasons),
        fallback_count=sum(1 for item in audit_rows if "category_fallback" in item.audit_reasons),
        documents_count=len(document_reports),
    )

    return audit_rows, document_reports, summary


def load_transactions_raw_csv(csv_path: str) -> List[TransactionRaw]:
    rows: List[TransactionRaw] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                TransactionRaw(
                    candidate_id=(row.get("candidate_id") or "").strip(),
                    source_pdf=(row.get("source_pdf") or "").strip(),
                    source_page=int((row.get("source_page") or "0").strip() or 0),
                    source_page_span=(row.get("source_page_span") or "").strip(),
                    source_text=(row.get("source_text") or "").strip(),
                    transaction_date_raw=((row.get("transaction_date_raw") or "").strip() or None),
                    description_raw=((row.get("description_raw") or "").strip() or None),
                    amount_raw=((row.get("amount_raw") or "").strip() or None),
                    transaction_direction_raw=(row.get("transaction_direction_raw") or "unknown").strip(),
                    balance_raw=((row.get("balance_raw") or "").strip() or None),
                    currency_raw=((row.get("currency_raw") or "").strip() or None),
                    parse_status=(row.get("parse_status") or "").strip(),
                    parse_confidence=float((row.get("parse_confidence") or "0").strip() or 0.0),
                    parse_warnings=(row.get("parse_warnings") or "").strip(),
                )
            )
    return rows


def load_merchant_resolution_csv(csv_path: str) -> List[MerchantResolution]:
    # Compatibility wrapper kept in Story 12 API.
    return _load_merchant_resolution_csv(csv_path)


def load_caen_enriched_csv(csv_path: str) -> List[MerchantCaenEnriched]:
    # Compatibility wrapper kept in Story 12 API.
    return _load_caen_enriched_csv(csv_path)


def load_budget_categorized_csv(csv_path: str) -> List[BudgetCategorizedTransaction]:
    # Compatibility wrapper kept in Story 12 API.
    return _load_budget_categorized_csv(csv_path)


def save_story12_audit_csv(rows: List[TransactionAuditRow], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_id",
                "source_pdf",
                "source_page",
                "parse_status",
                "parse_confidence",
                "match_method",
                "match_score",
                "caen_enrichment_status",
                "budget_category",
                "categorization_status",
                "classification_reason",
                "review_required",
                "audit_reasons",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def save_story12_document_report_csv(rows: List[DocumentQualityReport], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_pdf",
                "total_transactions",
                "parse_failed_count",
                "weak_parse_count",
                "merchant_not_found_count",
                "low_match_score_count",
                "caen_missing_count",
                "category_unknown_count",
                "fallback_count",
                "review_required_count",
                "quality_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


