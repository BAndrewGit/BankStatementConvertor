from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
import os
import re
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.domain.inference_contracts import InferenceInputRow
from src.features.feature_builder import FINAL_DATASET_COLUMNS, build_final_dataset_row, classify_behavior_risk_level
from src.features.expense_aggregator import aggregate_expenses
from src.features.quality_metrics import QualityMetrics, compute_quality_metrics
from src.infrastructure.cache import CacheRepository, FileCacheRepository
from src.memory.entity_memory import EntityMemoryRepository
from src.pipelines.build_features import build_features
from src.pipelines.classify_transactions import classify_parsed_transactions
from src.pipelines.parse_statement import parse_statement
from src.pipelines.program2_adapter import build_file_traceability


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


@dataclass(frozen=True)
class BatchEndToEndRunResult:
    pdf_paths: List[str]
    export_dir: str
    run_report_path: str
    transactions_csv_path: str
    final_dataset_csv_path: str
    quality_metrics: QualityMetrics
    features: Dict[str, float]
    monthly_features: Dict[str, Dict[str, float]]
    classification_summary: Dict[str, float]
    deduplicated_transactions_count: int

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["quality_metrics"] = self.quality_metrics.to_dict()
        return payload


def _month_key(booking_date: str) -> str:
    raw = (booking_date or "").strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.strftime("%Y-%m")
        except ValueError:
            continue

    if len(raw) >= 7 and raw[4] == "-":
        return raw[:7]
    return "unknown"


REF_PATTERN = re.compile(r"\bREF\s*[:=]?\s*([A-Z0-9-]+)", re.IGNORECASE)
TRANSFER_TYPES = {"internal_transfer", "external_transfer"}
SALARY_TXN_TYPE = "salary_income"
IBAN_PATTERN = re.compile(r"\bRO\d{2}[A-Z0-9]{10,30}\b", re.IGNORECASE)


def _salary_income_metrics(transactions: Sequence[object]) -> Dict[str, object]:
    salary_transactions = [
        txn for txn in transactions if txn.txn_type == SALARY_TXN_TYPE and txn.direction == "credit"
    ]
    salary_total = round(sum(abs(float(txn.amount)) for txn in salary_transactions), 2)
    salary_count = len(salary_transactions)
    return {
        "salary_income_detected": salary_count > 0,
        "salary_income_count": salary_count,
        "salary_income_total": salary_total,
    }


def _transfer_entity_metrics(transactions: Sequence[object]) -> Dict[str, float]:
    transfer_like = [
        txn
        for txn in transactions
        if txn.txn_type in {"internal_transfer", "external_transfer", "salary_income"}
    ]
    entity_memory_direct_hits = float(
        sum(1 for txn in transfer_like if (txn.match_source or "") == "entity_memory")
    )
    entity_memory_backfill_hits = float(
        sum(1 for txn in transfer_like if (txn.match_source or "") == "entity_memory_backfill")
    )

    return {
        "entity_memory_hits": entity_memory_direct_hits + entity_memory_backfill_hits,
        "entity_memory_direct_hits": entity_memory_direct_hits,
        "entity_memory_backfill_hits": entity_memory_backfill_hits,
        "bootstrap_exact_hits": float(sum(1 for txn in transfer_like if (txn.match_source or "") == "bootstrap_exact")),
        "bootstrap_fuzzy_hits": float(sum(1 for txn in transfer_like if (txn.match_source or "") == "bootstrap_fuzzy")),
        "candidate_count": float(sum(1 for txn in transfer_like if (txn.state or "") == "candidate")),
        "confirmed_by_memory": float(sum(1 for txn in transfer_like if (txn.match_source or "") in {"entity_memory", "entity_memory_backfill"} and (txn.state or "") == "confirmed")),
        "confirmed_by_recurrence": float(sum(1 for txn in transfer_like if (txn.match_source or "") == "recurrence" and (txn.state or "") == "confirmed")),
        "backfilled_transactions": entity_memory_backfill_hits,
        "remaining_unknown_transfers": float(sum(1 for txn in transfer_like if (txn.state or "") in {"unknown", "candidate"})),
    }


def _extended_block1_kpis(transactions: Sequence[object], feature_vector: Dict[str, float]) -> Dict[str, object]:
    totals = aggregate_expenses(transactions)
    outgoing_tx_count = float(totals.get("outgoing_tx_count", 0.0))
    impulse_candidate_tx_count = float(totals.get("impulse_candidate_tx_count", 0.0))

    impulse_frequency = 0.0
    if outgoing_tx_count > 0:
        impulse_frequency = round(impulse_candidate_tx_count / outgoing_tx_count, 4)

    unknown_transfer_count = sum(
        1
        for txn in transactions
        if txn.txn_type in {"external_transfer", "internal_transfer"}
        and (txn.state or "") in {"unknown", "candidate"}
    )

    impulse_category_counts = {
        "clothing_personal_care": int(totals.get("impulse_tx_count_clothing_personal_care", 0.0)),
        "electronics_gadgets": int(totals.get("impulse_tx_count_electronics_gadgets", 0.0)),
        "entertainment": int(totals.get("impulse_tx_count_entertainment", 0.0)),
        "food": int(totals.get("impulse_tx_count_food", 0.0)),
        "other": int(totals.get("impulse_tx_count_other", 0.0)),
    }

    return {
        "save_money_computed": bool(
            "Save_Money_Yes" in feature_vector and "Save_Money_No" in feature_vector
        ),
        "impulse_candidates_count": int(impulse_candidate_tx_count),
        "impulse_frequency": impulse_frequency,
        "impulse_category_counts": impulse_category_counts,
        "housing_detected_amount": round(float(totals.get("housing_total", 0.0)), 2),
        "unknown_transfer_count": int(unknown_transfer_count),
        "other_spend_share": round(float(feature_vector.get("Expense_Distribution_Other", 0.0)), 4),
    }


def _derive_profile_id(transactions: Sequence[object], pdf_paths: Sequence[str]) -> str:
    ibans: set[str] = set()
    for txn in transactions:
        for match in IBAN_PATTERN.findall(txn.raw_description or ""):
            ibans.add(match.upper())

    if ibans:
        raw = "|".join(sorted(ibans))
    else:
        raw = "|".join(sorted(str(path) for path in pdf_paths))

    return f"profile-{sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _local_storage_paths(profile_id: str) -> Dict[str, str]:
    bootstrap_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "bootstrap_dictionary.json",
    )
    memory_repo = EntityMemoryRepository()
    return {
        "bootstrap_dictionary_path": os.path.abspath(bootstrap_path),
        "entity_memory_path": os.path.abspath(memory_repo.get_profile_path(profile_id)),
    }


def _deduplicate_transactions(transactions: Sequence[object]) -> Tuple[List[object], int, List[str]]:
    kept: List[object] = []
    exact_seen: set[Tuple[str, float, str, str, str]] = set()
    transfer_index_by_ref_amount_type: Dict[Tuple[str, float, str], int] = {}
    removed = 0
    removed_ids: List[str] = []

    for txn in transactions:
        normalized_desc = re.sub(r"\s+", " ", (txn.raw_description or "").strip().upper())
        exact_key = (
            txn.booking_date,
            round(float(txn.amount), 2),
            txn.direction,
            txn.source_section,
            normalized_desc,
        )
        if exact_key in exact_seen:
            removed += 1
            removed_ids.append(str(txn.transaction_id))
            continue

        exact_seen.add(exact_key)

        if txn.txn_type in TRANSFER_TYPES:
            ref_match = REF_PATTERN.search(txn.raw_description or "")
            if ref_match:
                ref_amount_type_key = (
                    ref_match.group(1).upper(),
                    round(float(txn.amount), 2),
                    str(txn.txn_type),
                )
                existing_index = transfer_index_by_ref_amount_type.get(ref_amount_type_key)
                if existing_index is not None:
                    existing = kept[existing_index]
                    # Keep credit side when available; otherwise keep first seen instance.
                    if txn.direction == "credit" and existing.direction == "debit":
                        removed_ids.append(str(existing.transaction_id))
                        kept[existing_index] = txn
                    else:
                        removed_ids.append(str(txn.transaction_id))
                    removed += 1
                    continue
                else:
                    transfer_index_by_ref_amount_type[ref_amount_type_key] = len(kept)

        kept.append(txn)

    return kept, removed, removed_ids


def _write_transactions_csv(transactions_csv_path: str, transactions: Sequence[object]) -> None:
    if transactions:
        transaction_fieldnames = list(transactions[0].to_dict().keys())
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
        for txn in transactions:
            writer.writerow(txn.to_dict())


def _write_single_row_dataset(final_dataset_csv_path: str, feature_vector: Dict[str, float]) -> None:
    with open(final_dataset_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_DATASET_COLUMNS)
        writer.writeheader()
        writer.writerow(build_final_dataset_row(feature_vector))


def _build_final_dataset_row_with_income_category(
    feature_vector: Dict[str, float],
    income_category_value: float,
    profile_answers: Optional[Mapping[str, object]] = None,
) -> Dict[str, float]:
    row = build_final_dataset_row(feature_vector)
    row["Income_Category"] = float(income_category_value)

    for column, raw_value in (profile_answers or {}).items():
        # Keep salary-derived Income_Category authoritative for model input consistency.
        if column == "Income_Category":
            continue
        if column in row:
            row[column] = float(raw_value)

    return _coerce_final_dataset_integer_columns(row)


def _final_dataset_integer_columns() -> set[str]:
    return {
        "Debt_Level",
        "Bank_Account_Analysis_Frequency",
        "Save_Money_No",
        "Save_Money_Yes",
        "Impulse_Buying_Frequency",
        "Savings_Goal_Major_Purchases",
        "Savings_Goal_Retirement",
        "Savings_Goal_Emergency_Fund",
        "Savings_Goal_Child_Education",
        "Savings_Goal_Vacation",
        "Savings_Goal_Other",
        "Savings_Obstacle_Other",
        "Savings_Obstacle_Insufficient_Income",
        "Savings_Obstacle_Other_Expenses",
        "Savings_Obstacle_Not_Priority",
        "Credit_Usage_Essential_Needs",
        "Credit_Usage_Major_Purchases",
        "Credit_Usage_Unexpected_Expenses",
        "Credit_Usage_Personal_Needs",
        "Credit_Usage_Never_Used",
        "Family_Status_Another",
        "Family_Status_In a relationship/married with children",
        "Family_Status_In a relationship/married without children",
        "Family_Status_Single, no children",
        "Family_Status_Single, with children",
        "Gender_Female",
        "Gender_Male",
        "Gender_Prefer not to say",
        "Financial_Attitude_I am disciplined in saving",
        "Financial_Attitude_I try to find a balance",
        "Financial_Attitude_Spend more than I earn",
        "Budget_Planning_Don't plan at all",
        "Budget_Planning_Plan budget in detail",
        "Budget_Planning_Plan only essentials",
        "Impulse_Buying_Category_Clothing or personal care products",
        "Impulse_Buying_Category_Electronics or gadgets",
        "Impulse_Buying_Category_Entertainment",
        "Impulse_Buying_Category_Food",
        "Impulse_Buying_Category_Other",
        "Impulse_Buying_Reason_Discounts or promotions",
        "Impulse_Buying_Reason_Other",
        "Impulse_Buying_Reason_Self-reward",
        "Impulse_Buying_Reason_Social pressure",
        "Financial_Investments_No, but interested",
        "Financial_Investments_No, not interested",
        "Financial_Investments_Yes, occasionally",
        "Financial_Investments_Yes, regularly",
    }


def _coerce_final_dataset_integer_columns(row: Dict[str, float]) -> Dict[str, float]:
    integer_columns = _final_dataset_integer_columns()
    coerced = dict(row)
    for column in integer_columns:
        if column not in coerced or coerced[column] is None:
            continue
        try:
            value = float(coerced[column])
        except Exception:
            continue
        if value.is_integer():
            coerced[column] = int(value)
    return coerced


def _predict_risk_score_for_row(
    row: Dict[str, float],
    export_scaler: Optional[Tuple[List[str], Any]],
) -> Optional[float]:
    if export_scaler is None:
        return None

    ordered_columns, predictor = export_scaler
    if not hasattr(predictor, "predict"):
        return None

    try:
        inference_values = {column: float(row.get(column, 0.0)) for column in ordered_columns}
        inference_row = InferenceInputRow.from_values(inference_values, ordered_columns)
        prediction = predictor.predict(inference_row)
        risk_score = getattr(prediction, "risk_score", None)
        if risk_score is None:
            return None
        return float(risk_score)
    except Exception:
        return None


def _build_prediction_source(
    feature_values: Mapping[str, float],
    income_category_value: float,
    profile_answers: Optional[Mapping[str, object]] = None,
) -> Dict[str, float]:
    source: Dict[str, float] = {}
    for column, value in feature_values.items():
        try:
            source[str(column)] = float(value)
        except Exception:
            continue

    source["Income_Category"] = float(income_category_value)
    for column, raw_value in (profile_answers or {}).items():
        if column == "Income_Category":
            continue
        try:
            source[str(column)] = float(raw_value)
        except Exception:
            continue
    return source


def _build_export_scaler(artifacts_dir: Optional[str]) -> Optional[Tuple[List[str], Any]]:
    if not artifacts_dir:
        return None

    try:
        from src.inference.model_artifact_loader import ModelArtifactLoader
        from src.inference.predictor import Predictor

        artifacts = ModelArtifactLoader(artifacts_dir).load(require_multitask=False)
    except Exception:
        return None

    return list(artifacts.feature_columns), Predictor(artifacts)


def _scale_final_dataset_row(
    row: Dict[str, float],
    export_scaler: Optional[Tuple[List[str], Any]],
) -> Dict[str, float]:
    if export_scaler is None:
        return _coerce_final_dataset_integer_columns(row)

    ordered_columns, predictor = export_scaler
    ordered_values: List[float] = []
    for column in ordered_columns:
        try:
            ordered_values.append(float(row.get(column, 0.0)))
        except Exception:
            ordered_values.append(0.0)

    try:
        scaled_values = predictor.scale_ordered_values(ordered_values)
    except Exception:
        return row

    scaled_row = dict(row)
    for index, column in enumerate(ordered_columns):
        if index < len(scaled_values):
            scaled_row[column] = float(scaled_values[index])
    return _coerce_final_dataset_integer_columns(scaled_row)


def _combine_classification_summaries(
    summaries: Iterable[Mapping[str, float]],
) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    for summary in summaries:
        for key, value in summary.items():
            if key.endswith("_rate"):
                continue
            combined[key] = round(float(combined.get(key, 0.0)) + float(value), 6)

    total = float(combined.get("total", 0.0))
    valid = float(combined.get("valid", 0.0))
    unknown = float(combined.get("unknown_category", 0.0))
    if total > 0:
        combined["valid_rate"] = round(valid / total, 4)
        combined["unknown_rate"] = round(unknown / total, 4)
    return combined


def run_end_to_end(
    pdf_path: str,
    export_dir: str,
    cache_repo: Optional[CacheRepository] = None,
    manual_labels: Optional[Mapping[str, Mapping[str, object]]] = None,
    max_transactions_in_report: int = 500,
    profile_answers: Optional[Mapping[str, object]] = None,
    artifacts_dir: Optional[str] = None,
) -> EndToEndRunResult:
    os.makedirs(export_dir, exist_ok=True)
    cache = cache_repo or FileCacheRepository(persist_every_n_writes=200)

    run_started = perf_counter()

    parse_started = perf_counter()
    parsed_transactions = parse_statement(pdf_path, cache_repo=cache)
    parse_latency_ms = (perf_counter() - parse_started) * 1000.0
    profile_id = _derive_profile_id(parsed_transactions, [pdf_path])

    classified_transactions, classification_summary = classify_parsed_transactions(
        parsed_transactions,
        cache_repo=cache,
        profile_id=profile_id,
    )

    salary_metrics = _salary_income_metrics(classified_transactions)
    transfer_metrics = _transfer_entity_metrics(classified_transactions)
    local_storage = _local_storage_paths(profile_id)

    feature_vector = build_features(classified_transactions)
    block1_kpis = _extended_block1_kpis(classified_transactions, feature_vector)
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
    _write_transactions_csv(transactions_csv_path, classified_transactions)

    final_dataset_csv_path = os.path.join(export_dir, "final_dataset.csv")
    export_scaler = _build_export_scaler(artifacts_dir)
    with open(final_dataset_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_DATASET_COLUMNS)
        writer.writeheader()
        prediction_source = _build_prediction_source(
            feature_values=feature_vector,
            income_category_value=float(salary_metrics["salary_income_total"]),
            profile_answers=profile_answers,
        )
        row = _build_final_dataset_row_with_income_category(
            feature_vector=feature_vector,
            income_category_value=float(salary_metrics["salary_income_total"]),
            profile_answers=profile_answers,
        )
        risk_score = _predict_risk_score_for_row(row=prediction_source, export_scaler=export_scaler)
        if risk_score is not None:
            row["Risk_Score"] = risk_score
            row["Behavior_Risk_Level"] = classify_behavior_risk_level(risk_score)
        writer.writerow(
            _scale_final_dataset_row(
                row=row,
                export_scaler=export_scaler,
            )
        )

    run_report_path = os.path.join(export_dir, "run_report.json")
    serialized_transactions = [txn.to_dict() for txn in classified_transactions]
    report_transactions = serialized_transactions
    omitted_transactions = 0
    if max_transactions_in_report >= 0 and len(serialized_transactions) > max_transactions_in_report:
        report_transactions = serialized_transactions[:max_transactions_in_report]
        omitted_transactions = len(serialized_transactions) - len(report_transactions)

    report_payload: Dict[str, object] = {
        "pdf_path": pdf_path,
        "per_file_traceability": build_file_traceability([pdf_path]),
        "output_files": {
            "run_report": run_report_path,
            "transactions_classified": transactions_csv_path,
            "final_dataset": final_dataset_csv_path,
        },
        "run_summary": {
            "transaction_count": metrics.transactions_extracted_count,
            "unknown_count": metrics.category_unknown_count,
            "runtime_ms": metrics.end_to_end_latency_ms,
            **salary_metrics,
            **transfer_metrics,
            **block1_kpis,
        },
        "transactions": report_transactions,
        "transactions_omitted_from_report": omitted_transactions,
        "classification_summary": classification_summary,
        "features": feature_vector,
        "salary_income": salary_metrics,
        "transfer_entity_metrics": transfer_metrics,
        "block1_kpis": block1_kpis,
        "profile_id": profile_id,
        "local_storage": local_storage,
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


def run_end_to_end_many(
    pdf_paths: Sequence[str],
    export_dir: str,
    cache_repo: Optional[CacheRepository] = None,
    manual_labels: Optional[Mapping[str, Mapping[str, object]]] = None,
    max_transactions_in_report: int = 500,
    profile_answers: Optional[Mapping[str, object]] = None,
    artifacts_dir: Optional[str] = None,
) -> BatchEndToEndRunResult:
    if not pdf_paths:
        raise ValueError("run_end_to_end_many requires at least one PDF path")

    os.makedirs(export_dir, exist_ok=True)
    cache = cache_repo or FileCacheRepository(persist_every_n_writes=200)

    run_started = perf_counter()
    total_parse_latency_ms = 0.0
    all_parsed_transactions = []
    parsed_per_pdf: List[Tuple[str, List[object]]] = []
    all_classified_transactions = []
    classification_summaries = []

    for pdf_path in pdf_paths:
        parse_started = perf_counter()
        parsed_transactions = parse_statement(pdf_path, cache_repo=cache)
        total_parse_latency_ms += (perf_counter() - parse_started) * 1000.0
        parsed_per_pdf.append((pdf_path, parsed_transactions))
        all_parsed_transactions.extend(parsed_transactions)

    profile_id = _derive_profile_id(all_parsed_transactions, pdf_paths)

    for _pdf_path, parsed_transactions in parsed_per_pdf:
        classified_transactions, classification_summary = classify_parsed_transactions(
            parsed_transactions,
            cache_repo=cache,
            profile_id=profile_id,
        )
        all_classified_transactions.extend(classified_transactions)
        classification_summaries.append(classification_summary)

    deduped_transactions, deduped_count, deduped_transaction_ids = _deduplicate_transactions(all_classified_transactions)
    salary_metrics = _salary_income_metrics(deduped_transactions)
    transfer_metrics = _transfer_entity_metrics(deduped_transactions)
    local_storage = _local_storage_paths(profile_id)

    feature_vector = build_features(deduped_transactions)
    block1_kpis = _extended_block1_kpis(deduped_transactions, feature_vector)
    unknown_expense_percentage = feature_vector.get("Unknown_Expense_Percentage", 0.0)
    end_to_end_latency_ms = (perf_counter() - run_started) * 1000.0

    metrics = compute_quality_metrics(
        transactions=deduped_transactions,
        unknown_expense_percentage=unknown_expense_percentage,
        pdf_parse_latency_ms=total_parse_latency_ms,
        end_to_end_latency_ms=end_to_end_latency_ms,
        manual_labels=manual_labels,
    )

    monthly_transactions: Dict[str, List[object]] = {}
    for txn in deduped_transactions:
        key = _month_key(txn.booking_date)
        monthly_transactions.setdefault(key, []).append(txn)

    monthly_features: Dict[str, Dict[str, float]] = {}
    for month_key in sorted(monthly_transactions.keys()):
        monthly_features[month_key] = build_features(monthly_transactions[month_key])

    transactions_csv_path = os.path.join(export_dir, "transactions_classified.csv")
    _write_transactions_csv(transactions_csv_path, deduped_transactions)

    final_dataset_csv_path = os.path.join(export_dir, "final_dataset.csv")
    export_scaler = _build_export_scaler(artifacts_dir)
    with open(final_dataset_csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["statement_month", *FINAL_DATASET_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for month_key in sorted(monthly_features.keys()):
            month_salary_metrics = _salary_income_metrics(monthly_transactions[month_key])
            prediction_source = _build_prediction_source(
                feature_values=monthly_features[month_key],
                income_category_value=float(month_salary_metrics["salary_income_total"]),
                profile_answers=profile_answers,
            )
            row = _build_final_dataset_row_with_income_category(
                feature_vector=monthly_features[month_key],
                income_category_value=float(month_salary_metrics["salary_income_total"]),
                profile_answers=profile_answers,
            )
            risk_score = _predict_risk_score_for_row(row=prediction_source, export_scaler=export_scaler)
            if risk_score is not None:
                row["Risk_Score"] = risk_score
                row["Behavior_Risk_Level"] = classify_behavior_risk_level(risk_score)
            scaled_row = _scale_final_dataset_row(row=row, export_scaler=export_scaler)
            writer.writerow({"statement_month": month_key, **scaled_row})

    classification_summary = _combine_classification_summaries(classification_summaries)

    run_report_path = os.path.join(export_dir, "run_report.json")
    serialized_transactions = [txn.to_dict() for txn in deduped_transactions]
    report_transactions = serialized_transactions
    omitted_transactions = 0
    if max_transactions_in_report >= 0 and len(serialized_transactions) > max_transactions_in_report:
        report_transactions = serialized_transactions[:max_transactions_in_report]
        omitted_transactions = len(serialized_transactions) - len(report_transactions)

    report_payload: Dict[str, object] = {
        "pdf_paths": list(pdf_paths),
        "per_file_traceability": build_file_traceability(pdf_paths),
        "output_files": {
            "run_report": run_report_path,
            "transactions_classified": transactions_csv_path,
            "final_dataset": final_dataset_csv_path,
        },
        "run_summary": {
            "transaction_count": metrics.transactions_extracted_count,
            "unknown_count": metrics.category_unknown_count,
            "runtime_ms": metrics.end_to_end_latency_ms,
            "months_count": len(monthly_features),
            "deduplicated_count": deduped_count,
            **salary_metrics,
            **transfer_metrics,
            **block1_kpis,
        },
        "transactions": report_transactions,
        "transactions_omitted_from_report": omitted_transactions,
        "classification_summary": classification_summary,
        "features": feature_vector,
        "features_by_month": monthly_features,
        "deduplicated_transactions_count": deduped_count,
        "deduplicated_transaction_ids": deduped_transaction_ids,
        "salary_income": salary_metrics,
        "transfer_entity_metrics": transfer_metrics,
        "block1_kpis": block1_kpis,
        "profile_id": profile_id,
        "local_storage": local_storage,
        "quality_metrics": metrics.to_dict(),
    }
    with open(run_report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False, sort_keys=True)

    if isinstance(cache, FileCacheRepository):
        cache.flush()

    return BatchEndToEndRunResult(
        pdf_paths=list(pdf_paths),
        export_dir=export_dir,
        run_report_path=run_report_path,
        transactions_csv_path=transactions_csv_path,
        final_dataset_csv_path=final_dataset_csv_path,
        quality_metrics=metrics,
        features=feature_vector,
        monthly_features=monthly_features,
        classification_summary=classification_summary,
        deduplicated_transactions_count=deduped_count,
    )


