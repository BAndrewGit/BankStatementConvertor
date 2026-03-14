from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from time import perf_counter
from typing import Dict, Mapping, Optional

from src.features.feature_builder import FEATURE_COLUMNS
from src.features.quality_metrics import QualityMetrics, compute_quality_metrics
from src.infrastructure.cache import CacheRepository, FileCacheRepository
from src.pipelines.build_features import build_features
from src.pipelines.classify_transactions import classify_parsed_transactions
from src.pipelines.parse_statement import parse_statement


@dataclass(frozen=True)
class EndToEndRunResult:
    pdf_path: str
    export_dir: str
    run_report_path: str
    transactions_csv_path: str
    final_dataset_csv_path: str
    quality_metrics: QualityMetrics
    features: Dict[str, float]
    classification_summary: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["quality_metrics"] = self.quality_metrics.to_dict()
        return payload


def run_end_to_end(
    pdf_path: str,
    export_dir: str,
    cache_repo: Optional[CacheRepository] = None,
    manual_labels: Optional[Mapping[str, Mapping[str, object]]] = None,
    max_transactions_in_report: int = 500,
) -> EndToEndRunResult:
    os.makedirs(export_dir, exist_ok=True)
    cache = cache_repo or FileCacheRepository(persist_every_n_writes=200)

    run_started = perf_counter()

    parse_started = perf_counter()
    parsed_transactions = parse_statement(pdf_path, cache_repo=cache)
    parse_latency_ms = (perf_counter() - parse_started) * 1000.0

    classified_transactions, classification_summary = classify_parsed_transactions(
        parsed_transactions,
        cache_repo=cache,
    )

    feature_vector = build_features(classified_transactions)
    unknown_expense_percentage = feature_vector.get("Unknown_Expense_Percentage", 0.0)

    end_to_end_latency_ms = (perf_counter() - run_started) * 1000.0

    metrics = compute_quality_metrics(
        transactions=classified_transactions,
        unknown_expense_percentage=unknown_expense_percentage,
        pdf_parse_latency_ms=parse_latency_ms,
        end_to_end_latency_ms=end_to_end_latency_ms,
        manual_labels=manual_labels,
    )

    transactions_csv_path = os.path.join(export_dir, "transactions_classified.csv")
    if classified_transactions:
        transaction_fieldnames = list(classified_transactions[0].to_dict().keys())
    else:
        transaction_fieldnames = [
            "transaction_id",
            "booking_date",
            "amount",
            "currency",
            "direction",
            "raw_description",
            "source_section",
            "channel",
            "txn_type",
            "merchant_raw",
            "merchant_canonical",
            "category_area",
            "mapping_method",
            "confidence",
        ]
    with open(transactions_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=transaction_fieldnames)
        writer.writeheader()
        for txn in classified_transactions:
            writer.writerow(txn.to_dict())

    final_dataset_csv_path = os.path.join(export_dir, "final_dataset.csv")
    with open(final_dataset_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerow({column: float(feature_vector.get(column, 0.0)) for column in FEATURE_COLUMNS})

    run_report_path = os.path.join(export_dir, "run_report.json")
    serialized_transactions = [txn.to_dict() for txn in classified_transactions]
    report_transactions = serialized_transactions
    omitted_transactions = 0
    if max_transactions_in_report >= 0 and len(serialized_transactions) > max_transactions_in_report:
        report_transactions = serialized_transactions[:max_transactions_in_report]
        omitted_transactions = len(serialized_transactions) - len(report_transactions)

    report_payload: Dict[str, object] = {
        "pdf_path": pdf_path,
        "output_files": {
            "run_report": run_report_path,
            "transactions_classified": transactions_csv_path,
            "final_dataset": final_dataset_csv_path,
        },
        "run_summary": {
            "transaction_count": metrics.transactions_extracted_count,
            "unknown_count": metrics.category_unknown_count,
            "runtime_ms": metrics.end_to_end_latency_ms,
        },
        "transactions": report_transactions,
        "transactions_omitted_from_report": omitted_transactions,
        "classification_summary": classification_summary,
        "features": feature_vector,
        "quality_metrics": metrics.to_dict(),
    }
    with open(run_report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False, sort_keys=True)

    if isinstance(cache, FileCacheRepository):
        cache.flush()

    return EndToEndRunResult(
        pdf_path=pdf_path,
        export_dir=export_dir,
        run_report_path=run_report_path,
        transactions_csv_path=transactions_csv_path,
        final_dataset_csv_path=final_dataset_csv_path,
        quality_metrics=metrics,
        features=feature_vector,
        classification_summary=classification_summary,
    )


