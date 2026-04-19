from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence

from src.domain.inference_contracts import InferenceInputRow
from src.inference.model_artifact_loader import ModelArtifacts
from src.pipelines.training_governance import GateCheck, assert_quality_gate, evaluate_quality_gate


SCALED_FEATURE_COLUMNS = {
    "Age",
    "Income_Category",
    "Essential_Needs_Percentage",
    "Product_Lifetime_Clothing",
    "Product_Lifetime_Tech",
    "Product_Lifetime_Appliances",
    "Product_Lifetime_Cars",
}


@dataclass(frozen=True)
class PredictionResult:
    risk_score: float
    saving_probability: float
    risk_level: str
    inputs_scaled: bool
    scaled_feature_columns: List[str]
    top_factors: List[Dict[str, float]]
    risk_factors: List[Dict[str, float]]
    healthy_factors: List[Dict[str, float]]
    alerts: List[str]
    raw_output: Dict[str, float]


class Predictor:
    def __init__(self, artifacts: ModelArtifacts) -> None:
        self._artifacts = artifacts
        self._saving_probability_threshold = float(
            artifacts.thresholds.get("saving_probability_threshold", 0.5)
        )
        self._top_k_factors = int(artifacts.thresholds.get("top_k_factors", 5))
        self._healthy_threshold = float(artifacts.thresholds.get("risk_score_healthy_threshold", 0.33))
        self._risky_threshold = float(artifacts.thresholds.get("risk_score_risky_threshold", 0.67))
        if not (0.0 <= self._healthy_threshold < self._risky_threshold <= 1.0):
            raise ValueError(
                "Invalid risk band thresholds: expected 0.0 <= healthy < risky <= 1.0"
            )

    def predict(self, inference_row: InferenceInputRow) -> PredictionResult:
        checks = [
            GateCheck(
                name="feature_count_matches_model",
                passed=len(inference_row.ordered_columns) == len(self._artifacts.feature_columns),
                details={
                    "row_feature_count": len(inference_row.ordered_columns),
                    "model_feature_count": len(self._artifacts.feature_columns),
                },
            ),
            GateCheck(
                name="feature_order_matches_model",
                passed=list(inference_row.ordered_columns) == list(self._artifacts.feature_columns),
            ),
        ]
        assert_quality_gate(evaluate_quality_gate(checks))

        ordered_values = inference_row.as_ordered_list()
        scaled_values = self._scale_row(ordered_values)
        raw_output = self._run_model(scaled_values)

        risk_score, saving_probability = self._extract_scores(raw_output)
        risk_level = self._classify_risk_level(risk_score)
        scaled_feature_columns = [
            column for column in inference_row.ordered_columns if column in SCALED_FEATURE_COLUMNS
        ]
        inputs_scaled = bool(scaled_feature_columns) and self._artifacts.scaler is not None
        top_factors, risk_factors, healthy_factors = self._compute_factor_groups(
            inference_row.ordered_columns,
            scaled_values,
        )
        alerts = self._build_alerts(risk_score=risk_score, saving_probability=saving_probability)

        return PredictionResult(
            risk_score=risk_score,
            saving_probability=saving_probability,
            risk_level=risk_level,
            inputs_scaled=inputs_scaled,
            scaled_feature_columns=scaled_feature_columns,
            top_factors=top_factors,
            risk_factors=risk_factors,
            healthy_factors=healthy_factors,
            alerts=alerts,
            raw_output=raw_output,
        )

    def scale_ordered_values(self, row_values: Sequence[float]) -> List[float]:
        # Reuse the same scaler pipeline for export-time and inference-time processing.
        return self._scale_row(row_values)

    def _scale_row(self, row_values: Sequence[float]) -> List[float]:
        if not row_values:
            raise ValueError("Inference row is empty")
        numeric_row = [float(value) for value in row_values]
        for index, numeric in enumerate(numeric_row):
            if not math.isfinite(numeric):
                raise ValueError(f"Non-finite pre-scaler value at index {index}")

        scaler = self._artifacts.scaler
        if scaler is None:
            return list(numeric_row)

        if hasattr(scaler, "transform"):
            feature_columns = list(self._artifacts.feature_columns)
            if len(feature_columns) != len(numeric_row):
                raise ValueError(
                    "Feature columns width does not match row width "
                    f"({len(feature_columns)} != {len(numeric_row)})"
                )

            scale_indices = [
                index for index, column in enumerate(feature_columns) if column in SCALED_FEATURE_COLUMNS
            ]
            if not scale_indices:
                return list(numeric_row)

            expected_width = getattr(scaler, "n_features_in_", None)

            income_index = feature_columns.index("Income_Category") if "Income_Category" in feature_columns else None
            has_income_in_scaled_subset = income_index is not None and income_index in scale_indices

            def _to_float_list(values: Sequence[float]) -> List[float]:
                return [float(value) for value in values]

            merged: List[float] | None = None

            # Case 1: scaler trained on full model width.
            if isinstance(expected_width, int) and expected_width == len(numeric_row):
                transformed_full = _to_float_list(scaler.transform([numeric_row])[0])
                if len(transformed_full) != len(numeric_row):
                    raise ValueError(
                        "Scaler output width does not match model feature count "
                        f"({len(transformed_full)} != {len(numeric_row)})"
                    )
                merged = list(numeric_row)
                for index in scale_indices:
                    merged[index] = transformed_full[index]

            # Case 2: legacy full-width scaler trained with Income_Category excluded.
            elif (
                isinstance(expected_width, int)
                and expected_width == len(numeric_row) - 1
                and has_income_in_scaled_subset
                and income_index is not None
            ):
                reduced_full = [value for idx, value in enumerate(numeric_row) if idx != income_index]
                transformed_reduced = _to_float_list(scaler.transform([reduced_full])[0])
                if len(transformed_reduced) != len(reduced_full):
                    raise ValueError(
                        "Scaler output width does not match reduced feature count "
                        f"({len(transformed_reduced)} != {len(reduced_full)})"
                    )
                transformed_full: List[float] = []
                cursor = 0
                for idx in range(len(numeric_row)):
                    if idx == income_index:
                        transformed_full.append(numeric_row[idx])
                    else:
                        transformed_full.append(transformed_reduced[cursor])
                        cursor += 1
                merged = list(numeric_row)
                for index in scale_indices:
                    merged[index] = transformed_full[index]

            # Case 2b: legacy scaler trained on a 54-column projection that excludes
            # Age and Income_Category from the scaler input, while still keeping them
            # available in the model row.
            elif (
                isinstance(expected_width, int)
                and expected_width == len(numeric_row) - 2
                and "Age" in feature_columns
                and "Income_Category" in feature_columns
            ):
                excluded_from_scaler = {"Age", "Income_Category"}
                reduced_indices = [
                    idx for idx, column in enumerate(feature_columns) if column not in excluded_from_scaler
                ]
                reduced_row = [numeric_row[idx] for idx in reduced_indices]
                transformed_reduced = _to_float_list(scaler.transform([reduced_row])[0])
                if len(transformed_reduced) != len(reduced_row):
                    raise ValueError(
                        "Scaler output width does not match reduced 54-column feature count "
                        f"({len(transformed_reduced)} != {len(reduced_row)})"
                    )

                merged = list(numeric_row)
                for reduced_index, original_index in enumerate(reduced_indices):
                    column_name = feature_columns[original_index]
                    if column_name in SCALED_FEATURE_COLUMNS and column_name not in excluded_from_scaler:
                        merged[original_index] = transformed_reduced[reduced_index]

            # Case 3: scaler trained only on selected subset.
            elif isinstance(expected_width, int) and expected_width == len(scale_indices):
                subset = [numeric_row[idx] for idx in scale_indices]
                transformed_subset = _to_float_list(scaler.transform([subset])[0])
                if len(transformed_subset) != len(subset):
                    raise ValueError(
                        "Scaler output width does not match selected feature count "
                        f"({len(transformed_subset)} != {len(subset)})"
                    )
                merged = list(numeric_row)
                for subset_index, row_index in enumerate(scale_indices):
                    merged[row_index] = transformed_subset[subset_index]

            # Case 4: legacy subset scaler excludes Income_Category.
            elif (
                isinstance(expected_width, int)
                and expected_width == len(scale_indices) - 1
                and has_income_in_scaled_subset
                and income_index is not None
            ):
                subset_without_income = [idx for idx in scale_indices if idx != income_index]
                values_for_scaler = [numeric_row[idx] for idx in subset_without_income]
                transformed_subset = _to_float_list(scaler.transform([values_for_scaler])[0])
                if len(transformed_subset) != len(values_for_scaler):
                    raise ValueError(
                        "Scaler output width does not match reduced selected feature count "
                        f"({len(transformed_subset)} != {len(values_for_scaler)})"
                    )
                merged = list(numeric_row)
                for subset_index, row_index in enumerate(subset_without_income):
                    merged[row_index] = transformed_subset[subset_index]
                merged[income_index] = numeric_row[income_index]

            # Unknown width metadata: probe full width first, then subset width.
            elif not isinstance(expected_width, int):
                try:
                    transformed_full = _to_float_list(scaler.transform([numeric_row])[0])
                    if len(transformed_full) == len(numeric_row):
                        merged = list(numeric_row)
                        for index in scale_indices:
                            merged[index] = transformed_full[index]
                except Exception:
                    merged = None

                if merged is None:
                    subset = [numeric_row[idx] for idx in scale_indices]
                    transformed_subset = _to_float_list(scaler.transform([subset])[0])
                    if len(transformed_subset) != len(subset):
                        raise ValueError(
                            "Scaler output width does not match selected feature count "
                            f"({len(transformed_subset)} != {len(subset)})"
                        )
                    merged = list(numeric_row)
                    for subset_index, row_index in enumerate(scale_indices):
                        merged[row_index] = transformed_subset[subset_index]

            else:
                raise ValueError(
                    "Scaler expects unsupported input feature width "
                    f"({expected_width}) for row width {len(numeric_row)} and selected subset {len(scale_indices)}"
                )

            if merged is None:
                raise ValueError("Scaler transform failed to produce scaled values")

            for index, value in enumerate(merged):
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite post-scaler value at index {index}")
            return merged


        raise ValueError("Loaded scaler does not implement transform")

    def _run_model(self, scaled_values: Sequence[float]) -> Dict[str, float]:
        model = self._artifacts.model

        if self._is_torch_module(model):
            return self._run_torch_module(model, scaled_values)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([list(scaled_values)])
            proba_row = probabilities[0]
            if len(proba_row) == 1:
                saving_probability = float(proba_row[0])
            else:
                saving_probability = float(proba_row[-1])
            return {
                "saving_probability": saving_probability,
                "risk_score": round(1.0 - saving_probability, 6),
            }

        if hasattr(model, "predict"):
            predicted = model.predict([list(scaled_values)])
            output = predicted[0]
            if isinstance(output, Mapping):
                return {
                    key: float(value)
                    for key, value in output.items()
                    if isinstance(value, (int, float))
                }
            if isinstance(output, (list, tuple)) and len(output) >= 2:
                return {
                    "risk_score": float(output[0]),
                    "saving_probability": float(output[1]),
                }
            return {
                "risk_score": float(output),
                "saving_probability": round(1.0 - self._sigmoid(float(output)), 6),
            }

        if callable(model):
            output = model(list(scaled_values))
            if isinstance(output, Mapping):
                converted: Dict[str, float] = {}
                for key, value in output.items():
                    numeric = self._to_float_scalar(value)
                    if numeric is None:
                        continue
                    converted[str(key)] = numeric
                if converted:
                    return converted
            if isinstance(output, (list, tuple)) and len(output) >= 2:
                risk = self._to_float_scalar(output[0])
                saving = self._to_float_scalar(output[1])
                if risk is not None and saving is not None:
                    return {
                        "risk_score": risk,
                        "saving_probability": saving,
                    }
            if isinstance(output, (int, float)):
                score = float(output)
                return {
                    "risk_score": score,
                    "saving_probability": round(1.0 - self._sigmoid(score), 6),
                }

        raise ValueError("Loaded model cannot be executed: unsupported interface")

    @staticmethod
    def _is_torch_module(model: object) -> bool:
        try:
            import torch.nn as nn  # type: ignore

            return isinstance(model, nn.Module)
        except Exception:
            return False

    def _run_torch_module(self, model: object, scaled_values: Sequence[float]) -> Dict[str, float]:
        try:
            import torch  # type: ignore

            values_for_model = list(scaled_values)
            feature_columns = list(self._artifacts.feature_columns)
            trunk = getattr(model, "shared_trunk", None)
            first_linear = None
            try:
                if trunk is not None and hasattr(trunk, "__getitem__"):
                    first_linear = trunk[0]
            except Exception:
                first_linear = None
            model_in_features = getattr(first_linear, "in_features", None)
            if (
                isinstance(model_in_features, int)
                and model_in_features == len(values_for_model) - 1
                and "Income_Category" in feature_columns
                and len(feature_columns) == len(values_for_model)
            ):
                income_index = feature_columns.index("Income_Category")
                values_for_model = [value for idx, value in enumerate(values_for_model) if idx != income_index]
            elif (
                isinstance(model_in_features, int)
                and model_in_features == len(values_for_model) - 2
                and "Age" in feature_columns
                and "Income_Category" in feature_columns
                and len(feature_columns) == len(values_for_model)
            ):
                excluded_from_model = {"Age", "Income_Category"}
                values_for_model = [
                    value
                    for idx, value in enumerate(values_for_model)
                    if feature_columns[idx] not in excluded_from_model
                ]

            with torch.no_grad():
                tensor = torch.tensor([values_for_model], dtype=torch.float32)
                output = model(tensor)
        except Exception as exc:
            raise ValueError(f"Torch model execution failed: {exc}") from exc

        if isinstance(output, Mapping):
            converted: Dict[str, float] = {}
            for key, value in output.items():
                numeric = self._to_float_scalar(value)
                if numeric is None:
                    continue
                converted[str(key)] = numeric
            if converted:
                return converted

        if isinstance(output, (list, tuple)) and len(output) >= 2:
            risk_value = self._to_float_scalar(output[0])
            saving_value = self._to_float_scalar(output[1])
            if risk_value is None or saving_value is None:
                raise ValueError("Torch model returned non-scalar outputs")
            return {
                "risk_score": round(self._sigmoid(risk_value), 6),
                "saving_probability": round(self._sigmoid(saving_value), 6),
            }

        scalar = self._to_float_scalar(output)
        if scalar is not None:
            return {
                "risk_score": round(self._sigmoid(scalar), 6),
                "saving_probability": round(1.0 - self._sigmoid(scalar), 6),
            }

        raise ValueError("Torch model returned unsupported output payload")

    @staticmethod
    def _to_float_scalar(value: object) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)

        try:
            import torch  # type: ignore

            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return float(value.detach().cpu().item())
                return None
        except Exception:
            pass

        try:
            if hasattr(value, "item"):
                return float(value.item())
        except Exception:
            return None

        return None

    @staticmethod
    def _sigmoid(value: float) -> float:
        # Numerically stable sigmoid approximation for large magnitudes.
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _extract_scores(self, raw_output: Mapping[str, float]) -> tuple[float, float]:
        risk_score = float(raw_output.get("risk_score", 0.0))
        saving_probability = float(raw_output.get("saving_probability", 0.0))

        # Clamp into [0, 1] because business thresholds are probability-based.
        risk_score = min(1.0, max(0.0, risk_score))
        saving_probability = min(1.0, max(0.0, saving_probability))
        return risk_score, saving_probability

    def _classify_risk_level(self, risk_score: float) -> str:
        if risk_score < self._healthy_threshold:
            return "healthy"
        if risk_score < self._risky_threshold:
            return "moderate"
        return "risky"

    def _compute_factor_groups(
        self,
        feature_columns: Sequence[str],
        scaled_values: Sequence[float],
    ) -> tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]]]:
        model = self._artifacts.model
        coefficients = getattr(model, "coef_", None)

        contributions: List[tuple[str, float]] = []
        if coefficients is not None:
            coeff_row = coefficients[0] if hasattr(coefficients, "__getitem__") else coefficients
            for index, feature in enumerate(feature_columns):
                coef = float(coeff_row[index])
                contribution = coef * float(scaled_values[index])
                contributions.append((feature, contribution))
        else:
            for index, feature in enumerate(feature_columns):
                contributions.append((feature, float(scaled_values[index])))

        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        top = contributions[: self._top_k_factors]

        positive = sorted((item for item in contributions if item[1] > 0), key=lambda item: item[1], reverse=True)
        negative = sorted((item for item in contributions if item[1] < 0), key=lambda item: item[1])

        return (
            [{"feature": feature, "contribution": round(value, 6)} for feature, value in top],
            [{"feature": feature, "contribution": round(value, 6)} for feature, value in positive[: self._top_k_factors]],
            [{"feature": feature, "contribution": round(value, 6)} for feature, value in negative[: self._top_k_factors]],
        )

    def _build_alerts(self, risk_score: float, saving_probability: float) -> List[str]:
        alerts: List[str] = []
        rules = self._artifacts.bank_mapping_rules or {}

        if risk_score >= float(rules.get("risk_score_high_threshold", 0.7)):
            alerts.append("high_risk_score")
        if saving_probability < self._saving_probability_threshold:
            alerts.append("low_saving_probability")

        for rule in rules.get("alerts", []) if isinstance(rules.get("alerts"), list) else []:
            if not isinstance(rule, Mapping):
                continue
            metric_name = str(rule.get("metric", "")).strip()
            operator = str(rule.get("operator", "")).strip()
            threshold = rule.get("value")
            message = str(rule.get("message", "")).strip() or f"rule:{metric_name}"
            if metric_name not in {"risk_score", "saving_probability"}:
                continue
            if threshold is None:
                continue

            current_value = risk_score if metric_name == "risk_score" else saving_probability
            threshold_value = float(threshold)
            if operator == ">=" and current_value >= threshold_value:
                alerts.append(message)
            elif operator == ">" and current_value > threshold_value:
                alerts.append(message)
            elif operator == "<=" and current_value <= threshold_value:
                alerts.append(message)
            elif operator == "<" and current_value < threshold_value:
                alerts.append(message)
            elif operator == "==" and current_value == threshold_value:
                alerts.append(message)

        # Keep deterministic output order.
        unique: List[str] = []
        seen = set()
        for alert in alerts:
            if alert in seen:
                continue
            seen.add(alert)
            unique.append(alert)
        return unique

