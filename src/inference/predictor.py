from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence

from src.domain.inference_contracts import InferenceInputRow
from src.inference.model_artifact_loader import ModelArtifacts
from src.pipelines.training_governance import GateCheck, assert_quality_gate, evaluate_quality_gate


@dataclass(frozen=True)
class PredictionResult:
    risk_score: float
    saving_probability: float
    top_factors: List[Dict[str, float]]
    alerts: List[str]
    raw_output: Dict[str, float]


class Predictor:
    def __init__(self, artifacts: ModelArtifacts) -> None:
        self._artifacts = artifacts
        self._saving_probability_threshold = float(
            artifacts.thresholds.get("saving_probability_threshold", 0.5)
        )
        self._top_k_factors = int(artifacts.thresholds.get("top_k_factors", 5))

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
        top_factors = self._compute_top_factors(inference_row.ordered_columns, scaled_values)
        alerts = self._build_alerts(risk_score=risk_score, saving_probability=saving_probability)

        return PredictionResult(
            risk_score=risk_score,
            saving_probability=saving_probability,
            top_factors=top_factors,
            alerts=alerts,
            raw_output=raw_output,
        )

    def _scale_row(self, row_values: Sequence[float]) -> List[float]:
        if not row_values:
            raise ValueError("Inference row is empty")
        for index, value in enumerate(row_values):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Non-finite pre-scaler value at index {index}")

        scaler = self._artifacts.scaler
        if scaler is None:
            return list(row_values)

        if hasattr(scaler, "transform"):
            feature_columns = list(self._artifacts.feature_columns)
            expected_width = getattr(scaler, "n_features_in_", None)

            # Some legacy bundles were trained with Income_Category excluded from scaler fit.
            if (
                isinstance(expected_width, int)
                and expected_width == len(row_values) - 1
                and "Income_Category" in feature_columns
                and len(feature_columns) == len(row_values)
            ):
                income_index = feature_columns.index("Income_Category")
                reduced_row = [float(value) for index, value in enumerate(row_values) if index != income_index]
                transformed = scaler.transform([reduced_row])
                scaled_reduced = [float(value) for value in transformed[0]]
                if len(scaled_reduced) != len(reduced_row):
                    raise ValueError(
                        "Scaler output width does not match reduced feature count "
                        f"({len(scaled_reduced)} != {len(reduced_row)})"
                    )
                merged: List[float] = []
                reduced_cursor = 0
                for index, value in enumerate(row_values):
                    if index == income_index:
                        merged.append(float(value))
                    else:
                        merged.append(scaled_reduced[reduced_cursor])
                        reduced_cursor += 1
                for index, value in enumerate(merged):
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite post-scaler value at index {index}")
                return merged

            transformed = scaler.transform([list(row_values)])
            first = transformed[0]
            scaled = [float(value) for value in first]
            if len(scaled) != len(row_values):
                raise ValueError(
                    "Scaler output width does not match model feature count "
                    f"({len(scaled)} != {len(row_values)})"
                )
            for index, value in enumerate(scaled):
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite post-scaler value at index {index}")
            return scaled

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

    def _compute_top_factors(
        self,
        feature_columns: Sequence[str],
        scaled_values: Sequence[float],
    ) -> List[Dict[str, float]]:
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
        return [{"feature": feature, "contribution": round(value, 6)} for feature, value in top]

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

