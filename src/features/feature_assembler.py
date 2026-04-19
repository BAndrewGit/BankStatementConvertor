from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from src.domain.inference_contracts import (
    InferenceInputRow,
    ProcessedStatementFeatures,
    ProfileAnswers,
    SOURCE_IGNORE,
    SOURCE_PROGRAM,
    SOURCE_QUESTIONNAIRE,
    build_feature_source_map,
)
from src.features.feature_builder import EXPENSE_DISTRIBUTION_BINARY_THRESHOLD


@dataclass(frozen=True)
class FeatureAssemblyResult:
    row: InferenceInputRow
    source_map: Dict[str, str]


class FeatureAssembler:
    """Build final inference row from statement features + profile answers."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        source_map: Mapping[str, str] | None = None,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty")
        self._feature_columns = list(feature_columns)
        self._source_map = dict(source_map or build_feature_source_map(self._feature_columns))

        missing = [column for column in self._feature_columns if column not in self._source_map]
        if missing:
            raise ValueError(f"Missing source mapping for features: {missing}")

    @property
    def feature_columns(self) -> Sequence[str]:
        return tuple(self._feature_columns)

    @property
    def source_map(self) -> Dict[str, str]:
        return dict(self._source_map)

    def assemble(
        self,
        statement_features: ProcessedStatementFeatures,
        profile_answers: ProfileAnswers,
    ) -> FeatureAssemblyResult:
        statement_map = statement_features.to_feature_map()
        profile_map = profile_answers.to_feature_map()
        force_zero_savings_obstacles = float(statement_map.get("Save_Money_Yes", 0.0)) >= 0.5

        merged: Dict[str, float] = {}
        for column in self._feature_columns:
            source = self._source_map[column]
            if source == SOURCE_PROGRAM:
                value = statement_map.get(column)
                if column.startswith("Expense_Distribution_") and value is not None:
                    value = 1.0 if float(value) >= EXPENSE_DISTRIBUTION_BINARY_THRESHOLD else 0.0
            elif source == SOURCE_QUESTIONNAIRE:
                if column.startswith("Savings_Obstacle_"):
                    if force_zero_savings_obstacles:
                        value = 0.0
                    else:
                        # Savings obstacles are optional: missing answer means keep all obstacle flags at 0.
                        value = profile_map.get(column, 0.0)
                else:
                    value = profile_map.get(column)
            elif source == SOURCE_IGNORE:
                value = 0.0
            else:
                raise ValueError(f"Unknown feature source '{source}' for column '{column}'")

            if value is None:
                raise ValueError(f"Missing required value for feature '{column}' from source '{source}'")
            merged[column] = float(value)

        row = InferenceInputRow.from_projected_values(merged, ordered_columns=self._feature_columns)
        return FeatureAssemblyResult(row=row, source_map=dict(self._source_map))

    def preview(self, row: InferenceInputRow, limit: int = 12) -> Dict[str, float]:
        sample_limit = max(1, int(limit))
        keys = row.ordered_columns[:sample_limit]
        return {key: row.values[key] for key in keys}


