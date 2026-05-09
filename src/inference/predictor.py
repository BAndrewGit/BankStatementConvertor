from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence

from src.domain.inference_contracts import InferenceInputRow, MODEL_SCALED_FEATURE_COLUMNS
from src.inference.model_artifact_loader import ModelArtifacts
from src.pipelines.training_governance import GateCheck, assert_quality_gate, evaluate_quality_gate


SCALED_FEATURE_COLUMNS = set(MODEL_SCALED_FEATURE_COLUMNS)


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
        thresholds = artifacts.thresholds or {}
        bank_rules = artifacts.bank_mapping_rules or {}

        self._saving_probability_threshold = self._resolve_threshold(
            thresholds,
            ("saving_probability_threshold", "savings_classification_threshold", "decision_boundary_savings"),
            0.5,
        )
        self._top_k_factors = max(1, int(self._resolve_threshold(thresholds, ("top_k_factors",), 5.0)))
        self._healthy_threshold = self._resolve_threshold(
            thresholds,
            ("risk_score_healthy_threshold", "risk_score_medium"),
            0.33,
        )
        self._risky_threshold = self._resolve_threshold(
            thresholds,
            ("risk_score_risky_threshold", "risk_score_high", "decision_boundary_risk"),
            0.67,
        )
        self._risk_score_high_threshold = self._resolve_threshold(
            bank_rules,
            ("risk_score_high_threshold",),
            self._resolve_threshold(
                thresholds,
                ("risk_score_high_threshold", "risk_score_high", "decision_boundary_risk"),
                0.7,
            ),
        )
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

        model_feature_columns = list(inference_row.ordered_columns)
        model_scaled_values = list(scaled_values)
        if self._is_torch_module(self._artifacts.model):
            model_feature_columns, model_scaled_values = self._align_torch_input(
                model_feature_columns,
                model_scaled_values,
            )

        raw_output = self._run_model(model_scaled_values)

        risk_score, saving_probability = self._extract_scores(raw_output)
        risk_level = self._classify_risk_level(risk_score)
        scaled_feature_columns = [
            column for column in inference_row.ordered_columns if column in SCALED_FEATURE_COLUMNS
        ]
        inputs_scaled = bool(scaled_feature_columns) and self._artifacts.scaler is not None
        top_factors, risk_factors, healthy_factors = self._compute_factor_groups(
            model_feature_columns,
            model_scaled_values,
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

        if not hasattr(scaler, "transform"):
            raise ValueError("Loaded scaler does not implement transform")

        feature_columns = list(self._artifacts.feature_columns)
        if len(feature_columns) != len(numeric_row):
            raise ValueError(
                "Feature columns width does not match row width "
                f"({len(feature_columns)} != {len(numeric_row)})"
            )

        metadata = getattr(self._artifacts, "model_metadata", {}) or {}
        scaled_feature_columns = metadata.get("scaled_feature_columns", list(MODEL_SCALED_FEATURE_COLUMNS))
        scaler_mode = metadata.get("scaler_mode")

        scale_indices = [
            index for index, column in enumerate(feature_columns) if column in scaled_feature_columns
        ]

        if not scale_indices:
            return list(numeric_row)

        expected_width = getattr(scaler, "n_features_in_", None)

        # Enforce exact selected_columns mapping if set
        if scaler_mode == "selected_columns":
            if isinstance(expected_width, int) and expected_width != len(scale_indices):
                raise ValueError(
                    f"Scaler expects {expected_width} selected features, but found {len(scale_indices)}"
                )

            subset = [numeric_row[idx] for idx in scale_indices]
            transformed_subset = [float(val) for val in scaler.transform([subset])[0]]

            merged = list(numeric_row)
            for subset_index, row_index in enumerate(scale_indices):
                merged[row_index] = transformed_subset[subset_index]

            for index, value in enumerate(merged):
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite post-scaler value at index {index}")
            return merged

        raise ValueError("Predictor missing explicit scaler_mode='selected_columns' from artifacts. Extrapolating is forbidden.")

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

    def _align_torch_input(
        self,
        feature_columns: Sequence[str],
        scaled_values: Sequence[float],
    ) -> tuple[List[str], List[float]]:
        model_in_features = self._infer_torch_input_dim(self._artifacts.model)
        values = list(scaled_values)
        columns = list(feature_columns)

        if not isinstance(model_in_features, int):
            return columns, values

        if model_in_features == len(values):
            return columns, values

        if model_in_features == len(values) - 1:
            # Legacy 54-wide multitask checkpoints drop Essential_Needs_Percentage.
            drop_column = "Essential_Needs_Percentage"
            if drop_column in columns:
                drop_index = columns.index(drop_column)
                projected_columns = [column for column in columns if column != drop_column]
                projected_values = [
                    value for index, value in enumerate(values) if index != drop_index
                ]
                if len(projected_values) == model_in_features:
                    return projected_columns, projected_values

        raise ValueError(f"Model expects {model_in_features} features, got {len(values)}")

    @staticmethod
    def _infer_torch_input_dim(model: object) -> int | None:
        trunk = getattr(model, "shared_trunk", None)
        first_linear = None
        try:
            if trunk is not None and hasattr(trunk, "__getitem__"):
                first_linear = trunk[0]
        except Exception:
            first_linear = None
        model_in_features = getattr(first_linear, "in_features", None)
        return model_in_features if isinstance(model_in_features, int) else None

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

            if isinstance(model_in_features, int) and model_in_features != len(values_for_model):
                raise ValueError(
                    f"Model expects {model_in_features} features, got {len(values_for_model)}"
                )

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
                "risk_score": round(risk_value, 6),
                "saving_probability": round(self._sigmoid(saving_value), 6),
            }

        scalar = self._to_float_scalar(output)
        if scalar is not None:
            return {
                "risk_score": round(scalar, 6),
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

        # Saving probability is probability-based; risk_score may be a raw regression output.
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
            # For neural networks without coefficients: normalize scaled values
            # Prevents one feature from dominating due to different scales
            scaled_list = [float(v) for v in scaled_values]
            scaled_abs = [abs(v) for v in scaled_list]
            max_abs = max(scaled_abs) if scaled_abs else 1.0
            max_abs = max(max_abs, 1.0)

            for index, feature in enumerate(feature_columns):
                normalized_contribution = scaled_list[index] / max_abs
                contributions.append((feature, normalized_contribution))

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

        if risk_score >= self._risk_score_high_threshold:
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

    @staticmethod
    def _resolve_threshold(payload: Mapping[str, object], keys: Sequence[str], default: float) -> float:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                return numeric
        return float(default)

