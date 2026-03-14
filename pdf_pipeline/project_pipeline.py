from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Dict, Optional, Tuple

from .pdf_extraction import extract_raw_pdf_lines, save_raw_pdf_lines_csv
from .transaction_candidates import (
    identify_transaction_candidates,
    save_candidate_lines_csv,
    save_transaction_candidates_csv,
)
from .transaction_parsing import parse_transactions_raw, save_transactions_raw_csv
from .transaction_normalization import (
    normalize_transactions_raw,
    save_normalization_issues_csv,
    save_transactions_normalized_csv,
)
from .description_normalization import (
    normalize_transaction_descriptions,
    save_description_normalization_issues_csv,
    save_transactions_description_normalized_csv,
)
from .transaction_classification import classify_transaction_types, save_transactions_classified_csv
from .external_datasets import prepare_external_datasets
from .merchant_resolution import resolve_merchants_to_onrc, save_merchant_resolution_csv
from .caen_enrichment import enrich_resolved_merchants_with_caen, save_merchants_caen_enriched_csv
from .budget_categorization import map_transactions_to_budget_categories, save_budget_categorized_csv
from .monthly_budget_aggregation import (
    aggregate_monthly_budget,
    save_monthly_budget_overview_csv,
    save_monthly_category_aggregates_csv,
)
from .audit_validation_reporting import (
    build_story12_audit,
    save_story12_audit_csv,
    save_story12_document_report_csv,
)


@dataclass(frozen=True)
class PipelineRunResult:
    pdf_path: str
    export_dir: str
    reference_db_path: str
    output_files: Dict[str, str]
    summaries: Dict[str, Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _workspace_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _default_datasets_dir() -> str:
    return os.path.join(_workspace_root(), "DatasetsCAEN")


def build_output_paths(export_dir: str) -> Dict[str, str]:
    return {
        "raw_pdf_pages_json": os.path.join(export_dir, "raw_pdf_pages.json"),
        "raw_pdf_lines_csv": os.path.join(export_dir, "raw_pdf_lines.csv"),
        "candidate_lines_csv": os.path.join(export_dir, "candidate_lines.csv"),
        "transaction_candidates_csv": os.path.join(export_dir, "transaction_candidates.csv"),
        "transactions_raw_csv": os.path.join(export_dir, "transactions_raw.csv"),
        "transactions_normalized_csv": os.path.join(export_dir, "transactions_normalized.csv"),
        "normalization_issues_csv": os.path.join(export_dir, "normalization_issues.csv"),
        "transactions_description_normalized_csv": os.path.join(export_dir, "transactions_description_normalized.csv"),
        "description_normalization_issues_csv": os.path.join(export_dir, "description_normalization_issues.csv"),
        "transactions_classified_csv": os.path.join(export_dir, "transactions_classified.csv"),
        "merchant_resolution_csv": os.path.join(export_dir, "merchant_resolution.csv"),
        "merchant_caen_enriched_csv": os.path.join(export_dir, "merchant_caen_enriched.csv"),
        "transactions_budget_categorized_csv": os.path.join(export_dir, "transactions_budget_categorized.csv"),
        "monthly_budget_overview_csv": os.path.join(export_dir, "monthly_budget_overview.csv"),
        "monthly_budget_categories_csv": os.path.join(export_dir, "monthly_budget_categories.csv"),
        "audit_csv": os.path.join(export_dir, "pipeline_audit.csv"),
        "quality_report_csv": os.path.join(export_dir, "pipeline_quality_report.csv"),
        "summary_json": os.path.join(export_dir, "pipeline_summary.json"),
    }


def _resolve_dataset_paths(datasets_dir: str) -> Tuple[str, str, str]:
    od_firme = os.path.join(datasets_dir, "od_firme.csv")
    od_caen = os.path.join(datasets_dir, "od_caen_autorizat.csv")
    n_caen = os.path.join(datasets_dir, "n_caen.csv")

    missing = [path for path in (od_firme, od_caen, n_caen) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Lipsesc dataseturi externe necesare: " + ", ".join(missing)
        )

    return od_firme, od_caen, n_caen


def run_full_pipeline(
    pdf_path: str,
    export_dir: str,
    datasets_dir: Optional[str] = None,
    reference_db_path: Optional[str] = None,
) -> PipelineRunResult:
    """Run the full extraction-to-audit pipeline and write all CSV outputs to export_dir."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Fisierul PDF nu exista: {pdf_path}")

    os.makedirs(export_dir, exist_ok=True)
    output_paths = build_output_paths(export_dir)

    datasets_folder = datasets_dir or _default_datasets_dir()
    od_firme_path, od_caen_path, n_caen_path = _resolve_dataset_paths(datasets_folder)

    db_path = reference_db_path or os.path.join(export_dir, "reference_data.db")

    pages, raw_lines = extract_raw_pdf_lines(pdf_path)
    save_raw_pdf_lines_csv(raw_lines, output_paths["raw_pdf_lines_csv"])

    with open(output_paths["raw_pdf_pages_json"], "w", encoding="utf-8") as handle:
        json.dump([page.to_dict() for page in pages], handle, indent=2, ensure_ascii=False)

    candidate_lines, candidates = identify_transaction_candidates(raw_lines)
    save_candidate_lines_csv(candidate_lines, output_paths["candidate_lines_csv"])
    save_transaction_candidates_csv(candidates, output_paths["transaction_candidates_csv"])

    transactions_raw = parse_transactions_raw(candidates)
    save_transactions_raw_csv(transactions_raw, output_paths["transactions_raw_csv"])

    transactions_normalized, normalization_issues, normalization_summary = normalize_transactions_raw(transactions_raw)
    save_transactions_normalized_csv(transactions_normalized, output_paths["transactions_normalized_csv"])
    save_normalization_issues_csv(normalization_issues, output_paths["normalization_issues_csv"])

    (
        transactions_description_normalized,
        description_normalization_issues,
        description_normalization_summary,
    ) = normalize_transaction_descriptions(transactions_normalized)
    save_transactions_description_normalized_csv(
        transactions_description_normalized,
        output_paths["transactions_description_normalized_csv"],
    )
    save_description_normalization_issues_csv(
        description_normalization_issues,
        output_paths["description_normalization_issues_csv"],
    )

    transactions_classified, classification_summary = classify_transaction_types(
        transactions_normalized,
        transactions_description_normalized,
    )
    save_transactions_classified_csv(transactions_classified, output_paths["transactions_classified_csv"])

    external_summary = prepare_external_datasets(
        od_firme_csv_path=od_firme_path,
        od_caen_autorizat_csv_path=od_caen_path,
        n_caen_csv_path=n_caen_path,
        output_db_path=db_path,
    )

    merchant_resolution_rows, merchant_resolution_summary = resolve_merchants_to_onrc(
        transactions_description_normalized,
        db_path=db_path,
    )
    save_merchant_resolution_csv(merchant_resolution_rows, output_paths["merchant_resolution_csv"])

    caen_rows, caen_summary = enrich_resolved_merchants_with_caen(merchant_resolution_rows, db_path)
    save_merchants_caen_enriched_csv(caen_rows, output_paths["merchant_caen_enriched_csv"])

    categorized_rows, categorization_summary = map_transactions_to_budget_categories(
        transactions_classified,
        caen_rows,
        db_path=db_path,
    )
    save_budget_categorized_csv(categorized_rows, output_paths["transactions_budget_categorized_csv"])

    monthly_overview_rows, monthly_category_rows, monthly_summary = aggregate_monthly_budget(
        transactions_classified,
        categorized_rows,
        transactions_normalized,
    )
    save_monthly_budget_overview_csv(monthly_overview_rows, output_paths["monthly_budget_overview_csv"])
    save_monthly_category_aggregates_csv(monthly_category_rows, output_paths["monthly_budget_categories_csv"])

    audit_rows, document_report_rows, audit_summary = build_story12_audit(
        transactions_raw,
        merchant_resolution_rows,
        caen_rows,
        categorized_rows,
    )
    save_story12_audit_csv(audit_rows, output_paths["audit_csv"])
    save_story12_document_report_csv(document_report_rows, output_paths["quality_report_csv"])

    summaries: Dict[str, Dict[str, object]] = {
        "normalization": normalization_summary,
        "description_normalization": description_normalization_summary,
        "classification": classification_summary,
        "external_datasets": external_summary.to_dict(),
        "merchant_resolution": merchant_resolution_summary.to_dict(),
        "caen_enrichment": caen_summary.to_dict(),
        "categorization": categorization_summary.to_dict(),
        "monthly_aggregation": monthly_summary.to_dict(),
        "audit": audit_summary.to_dict(),
    }

    result = PipelineRunResult(
        pdf_path=pdf_path,
        export_dir=export_dir,
        reference_db_path=db_path,
        output_files=output_paths,
        summaries=summaries,
    )

    with open(output_paths["summary_json"], "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)

    return result


