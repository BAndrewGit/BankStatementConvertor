from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, Mapping, Sequence

from src.domain.inference_contracts import (
    ProcessedStatementFeatures,
    SOURCE_PROGRAM,
    SOURCE_QUESTIONNAIRE,
    build_feature_source_map,
)


PROGRAM2_FIELD_ALIASES: Dict[str, str] = {
    "food_total_pct": "Expense_Distribution_Food",
    "housing_total_pct": "Expense_Distribution_Housing",
    "transport_total_pct": "Expense_Distribution_Transport",
    "entertainment_total_pct": "Expense_Distribution_Entertainment",
    "health_total_pct": "Expense_Distribution_Health",
    "personal_care_total_pct": "Expense_Distribution_Personal_Care",
    "child_education_total_pct": "Expense_Distribution_Child_Education",
    "other_total_pct": "Expense_Distribution_Other",
}


def _normalize_feature_payload(payload: Mapping[str, object]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}

    for key, value in payload.items():
        if value is None:
            continue
        mapped_key = PROGRAM2_FIELD_ALIASES.get(str(key), str(key))
        try:
            normalized[mapped_key] = float(value)
        except (TypeError, ValueError):
            continue

    return normalized


def adapt_program2_output_to_processed_features(program2_output: object) -> ProcessedStatementFeatures:
    """Build ProcessedStatementFeatures from run_end_to_end outputs or raw feature dicts."""

    if isinstance(program2_output, Mapping):
        if "features" in program2_output and isinstance(program2_output["features"], Mapping):
            source_payload = program2_output["features"]
        else:
            source_payload = program2_output
        return ProcessedStatementFeatures.from_mapping(_normalize_feature_payload(source_payload))

    if hasattr(program2_output, "features"):
        source_payload = getattr(program2_output, "features")
        if isinstance(source_payload, Mapping):
            return ProcessedStatementFeatures.from_mapping(_normalize_feature_payload(source_payload))

    if is_dataclass(program2_output):
        payload = asdict(program2_output)
        features = payload.get("features")
        if isinstance(features, Mapping):
            return ProcessedStatementFeatures.from_mapping(_normalize_feature_payload(features))

    raise ValueError("Unsupported program 2 output payload for feature adaptation")


def detect_profile_required_features(
    feature_columns: Sequence[str],
    processed_features: ProcessedStatementFeatures,
) -> Dict[str, Sequence[str]]:
    """Return which features are still required from profile answers."""

    source_map = build_feature_source_map(feature_columns)
    statement_map = processed_features.to_feature_map()

    missing_program_features = [
        feature
        for feature in feature_columns
        if source_map.get(feature) == SOURCE_PROGRAM and feature not in statement_map
    ]
    profile_features = [
        feature for feature in feature_columns if source_map.get(feature) == SOURCE_QUESTIONNAIRE
    ]

    return {
        "missing_program_features": missing_program_features,
        "required_profile_features": profile_features,
    }


def build_file_traceability(pdf_paths: Iterable[str], processed_ok: bool = True) -> Sequence[Dict[str, object]]:
    """Simple per-file trace payload used by batch report and UI results."""

    return [
        {
            "pdf_path": path,
            "processed": bool(processed_ok),
        }
        for path in pdf_paths
    ]

