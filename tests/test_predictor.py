import unittest

from src.domain.inference_contracts import InferenceInputRow
from src.inference.model_artifact_loader import ModelArtifacts
from src.inference.predictor import Predictor


class _ScaleByTwo:
    def transform(self, rows):
        return [[float(value) * 2.0 for value in rows[0]]]


class _ProbaModel:
    coef_ = [[0.2, -0.4, 0.1]]

    def predict_proba(self, rows):
        return [[0.35, 0.65] for _ in rows]


class _BadWidthScaler:
    def transform(self, rows):
        return [[1.0]]


class _NanScaler:
    def transform(self, rows):
        return [[float("nan") for _ in rows[0]]]


class _LegacyIncomeExcludedScaler:
    n_features_in_ = 2

    def transform(self, rows):
        # Scale only non-income features.
        return [[float(rows[0][0]) * 2.0, float(rows[0][1]) * 2.0]]


class PredictorTests(unittest.TestCase):
    def test_predictor_runs_scaler_model_thresholds_and_alert_rules(self):
        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_ProbaModel(),
            scaler=_ScaleByTwo(),
            feature_columns=["f1", "f2", "f3"],
            thresholds={"saving_probability_threshold": 0.7, "top_k_factors": 2},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={
                "risk_score_high_threshold": 0.3,
                "alerts": [
                    {
                        "metric": "saving_probability",
                        "operator": "<",
                        "value": 0.7,
                        "message": "custom_low_saving",
                    }
                ],
            },
        )

        predictor = Predictor(artifacts)
        input_row = InferenceInputRow.from_values({"f1": 1.0, "f2": 2.0, "f3": 3.0}, ["f1", "f2", "f3"])

        result = predictor.predict(input_row)

        self.assertAlmostEqual(result.saving_probability, 0.65, places=6)
        self.assertAlmostEqual(result.risk_score, 0.35, places=6)
        self.assertEqual(len(result.top_factors), 2)
        self.assertIn("high_risk_score", result.alerts)
        self.assertIn("low_saving_probability", result.alerts)
        self.assertIn("custom_low_saving", result.alerts)

    def test_predictor_quality_gate_rejects_feature_order_mismatch(self):
        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_ProbaModel(),
            scaler=_ScaleByTwo(),
            feature_columns=["f1", "f2", "f3"],
            thresholds={},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)
        wrong_order = InferenceInputRow.from_values(
            {"f2": 2.0, "f1": 1.0, "f3": 3.0},
            ["f2", "f1", "f3"],
        )

        with self.assertRaises(ValueError):
            predictor.predict(wrong_order)

    def test_predictor_rejects_bad_scaler_output_shape_or_values(self):
        base_artifacts = {
            "artifacts_dir": ".",
            "model": _ProbaModel(),
            "feature_columns": ["f1", "f2", "f3"],
            "thresholds": {},
            "model_metadata": {"model_type": "multitask_net", "multitask": True},
            "bank_mapping_rules": {},
        }
        row = InferenceInputRow.from_values({"f1": 1.0, "f2": 2.0, "f3": 3.0}, ["f1", "f2", "f3"])

        bad_width_predictor = Predictor(ModelArtifacts(scaler=_BadWidthScaler(), **base_artifacts))
        with self.assertRaises(ValueError):
            bad_width_predictor.predict(row)

        nan_predictor = Predictor(ModelArtifacts(scaler=_NanScaler(), **base_artifacts))
        with self.assertRaises(ValueError):
            nan_predictor.predict(row)

    def test_predictor_supports_legacy_scaler_missing_income_category(self):
        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_ProbaModel(),
            scaler=_LegacyIncomeExcludedScaler(),
            feature_columns=["f1", "Income_Category", "f3"],
            thresholds={},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)
        row = InferenceInputRow.from_values(
            {"f1": 1.0, "Income_Category": 5700.0, "f3": 3.0},
            ["f1", "Income_Category", "f3"],
        )

        result = predictor.predict(row)
        self.assertGreaterEqual(result.risk_score, 0.0)
        self.assertLessEqual(result.risk_score, 1.0)

    def test_predictor_supports_legacy_torch_model_missing_income_category(self):
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self.skipTest("torch is not available in this environment")

        class _TorchLegacyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_trunk = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
                self.risk_head = nn.Linear(2, 1)
                self.savings_head = nn.Linear(2, 1)

            def forward(self, x):
                hidden = self.shared_trunk(x)
                return self.risk_head(hidden), self.savings_head(hidden)

        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_TorchLegacyModel(),
            scaler=None,
            feature_columns=["f1", "Income_Category", "f3"],
            thresholds={},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)
        row = InferenceInputRow.from_values(
            {"f1": 1.0, "Income_Category": 5700.0, "f3": 3.0},
            ["f1", "Income_Category", "f3"],
        )

        result = predictor.predict(row)
        self.assertGreaterEqual(result.risk_score, 0.0)
        self.assertLessEqual(result.risk_score, 1.0)


if __name__ == "__main__":
    unittest.main()

