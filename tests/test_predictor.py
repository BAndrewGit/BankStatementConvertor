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


class _LowRiskModel:
    def predict_proba(self, rows):
        return [[0.1, 0.9] for _ in rows]


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


class _FullWidthScaler:
    n_features_in_ = 5

    def transform(self, rows):
        return [[float(value) * 2.0 for value in rows[0]]]


class _Legacy54WidthScaler:
    n_features_in_ = 54

    def transform(self, rows):
        return [[float(value) * 10.0 for value in rows[0]]]


class PredictorTests(unittest.TestCase):
    def test_predictor_runs_scaler_model_thresholds_and_alert_rules(self):
        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_ProbaModel(),
            scaler=_ScaleByTwo(),
            feature_columns=["Age", "Income_Category", "Essential_Needs_Percentage"],
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
        input_row = InferenceInputRow.from_values(
            {"Age": 30.0, "Income_Category": 5700.0, "Essential_Needs_Percentage": 55.0},
            ["Age", "Income_Category", "Essential_Needs_Percentage"],
        )

        result = predictor.predict(input_row)

        self.assertAlmostEqual(result.saving_probability, 0.65, places=6)
        self.assertAlmostEqual(result.risk_score, 0.35, places=6)
        self.assertEqual(result.risk_level, "moderate")
        self.assertTrue(result.inputs_scaled)
        self.assertEqual(
            result.scaled_feature_columns,
            ["Age", "Income_Category", "Essential_Needs_Percentage"],
        )
        self.assertEqual(len(result.top_factors), 2)
        self.assertIn("high_risk_score", result.alerts)
        self.assertIn("low_saving_probability", result.alerts)
        self.assertIn("custom_low_saving", result.alerts)

    def test_predictor_exposes_split_risk_and_healthy_factors(self):
        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_ProbaModel(),
            scaler=_ScaleByTwo(),
            feature_columns=["Age", "Income_Category", "Essential_Needs_Percentage"],
            thresholds={"top_k_factors": 5},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)
        row = InferenceInputRow.from_values(
            {"Age": 30.0, "Income_Category": 5700.0, "Essential_Needs_Percentage": 55.0},
            ["Age", "Income_Category", "Essential_Needs_Percentage"],
        )

        result = predictor.predict(row)

        self.assertEqual([item["feature"] for item in result.risk_factors], ["Age", "Essential_Needs_Percentage"])
        self.assertEqual([item["feature"] for item in result.healthy_factors], ["Income_Category"])
        self.assertEqual(result.risk_factors[0]["contribution"], 12.0)
        self.assertEqual(result.healthy_factors[0]["contribution"], -4560.0)

    def test_predictor_returns_empty_healthy_factors_when_no_negative_contributions(self):
        class _AllPositiveCoefModel:
            coef_ = [[0.2, 0.4, 0.1]]

            def predict_proba(self, rows):
                return [[0.35, 0.65] for _ in rows]

        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_AllPositiveCoefModel(),
            scaler=_ScaleByTwo(),
            feature_columns=["Age", "Income_Category", "Essential_Needs_Percentage"],
            thresholds={"top_k_factors": 5},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)
        row = InferenceInputRow.from_values(
            {"Age": 30.0, "Income_Category": 5700.0, "Essential_Needs_Percentage": 55.0},
            ["Age", "Income_Category", "Essential_Needs_Percentage"],
        )

        result = predictor.predict(row)

        self.assertEqual([item["feature"] for item in result.risk_factors], ["Income_Category", "Age", "Essential_Needs_Percentage"])
        self.assertEqual(result.healthy_factors, [])

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
            "feature_columns": ["Age", "Income_Category", "Gender_Male"],
            "thresholds": {},
            "model_metadata": {"model_type": "multitask_net", "multitask": True},
            "bank_mapping_rules": {},
        }
        row = InferenceInputRow.from_values(
            {"Age": 30.0, "Income_Category": 5700.0, "Gender_Male": 1.0},
            ["Age", "Income_Category", "Gender_Male"],
        )

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
            feature_columns=["Age", "Income_Category", "Product_Lifetime_Tech"],
            thresholds={},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)
        row = InferenceInputRow.from_values(
            {"Age": 30.0, "Income_Category": 5700.0, "Product_Lifetime_Tech": 24.0},
            ["Age", "Income_Category", "Product_Lifetime_Tech"],
        )

        result = predictor.predict(row)
        self.assertGreaterEqual(result.risk_score, 0.0)
        self.assertLessEqual(result.risk_score, 1.0)

    def test_predictor_marks_healthy_predictions_and_records_scaled_inputs(self):
        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_LowRiskModel(),
            scaler=_ScaleByTwo(),
            feature_columns=["Age", "Income_Category", "Essential_Needs_Percentage"],
            thresholds={
                "risk_score_healthy_threshold": 0.33,
                "risk_score_risky_threshold": 0.67,
            },
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)
        row = InferenceInputRow.from_values(
            {"Age": 30.0, "Income_Category": 5700.0, "Essential_Needs_Percentage": 55.0},
            ["Age", "Income_Category", "Essential_Needs_Percentage"],
        )

        result = predictor.predict(row)

        self.assertAlmostEqual(result.risk_score, 0.1, places=6)
        self.assertEqual(result.risk_level, "healthy")
        self.assertTrue(result.inputs_scaled)
        self.assertEqual(
            result.scaled_feature_columns,
            ["Age", "Income_Category", "Essential_Needs_Percentage"],
        )
        self.assertEqual(len(result.top_factors), 3)
        self.assertEqual(result.top_factors[0]["feature"], "Income_Category")
        self.assertEqual(result.alerts, [])

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

    def test_predictor_supports_legacy_torch_model_missing_age_and_income_category(self):
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self.skipTest("torch is not available in this environment")

        class _TorchLegacy54Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_trunk = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
                self.risk_head = nn.Linear(64, 1)
                self.savings_head = nn.Linear(64, 1)

            def forward(self, x):
                hidden = self.shared_trunk(x)
                return self.risk_head(hidden), self.savings_head(hidden)

        feature_columns = [
            "Age",
            "Income_Category",
            "Essential_Needs_Percentage",
            "Product_Lifetime_Clothing",
            "Product_Lifetime_Tech",
            "Product_Lifetime_Appliances",
            "Product_Lifetime_Cars",
            "Savings_Goal_Major_Purchases",
            "Savings_Goal_Retirement",
            "Savings_Goal_Child_Education",
            "Savings_Goal_Vacation",
            "Savings_Goal_Other",
            "Savings_Obstacle_Other",
            "Savings_Obstacle_Insufficient_Income",
            "Savings_Obstacle_Other_Expenses",
            "Savings_Obstacle_Not_Priority",
            "Expense_Distribution_Food",
            "Expense_Distribution_Housing",
            "Expense_Distribution_Transport",
            "Expense_Distribution_Entertainment",
            "Expense_Distribution_Health",
            "Expense_Distribution_Personal_Care",
            "Expense_Distribution_Child_Education",
            "Expense_Distribution_Other",
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
        ]

        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_TorchLegacy54Model(),
            scaler=None,
            feature_columns=feature_columns,
            thresholds={},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)
        row = InferenceInputRow.from_values(
            {column: float(index) for index, column in enumerate(feature_columns)},
            feature_columns,
        )

        result = predictor.predict(row)
        self.assertGreaterEqual(result.risk_score, 0.0)
        self.assertLessEqual(result.risk_score, 1.0)

    def test_scale_ordered_values_scales_only_target_columns_with_full_width_scaler(self):
        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_ProbaModel(),
            scaler=_FullWidthScaler(),
            feature_columns=[
                "Age",
                "Gender_Male",
                "Income_Category",
                "Essential_Needs_Percentage",
                "Risk_Score",
            ],
            thresholds={},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)

        scaled = predictor.scale_ordered_values([30.0, 1.0, 5700.0, 55.0, 0.3])

        self.assertEqual(scaled[0], 60.0)
        self.assertEqual(scaled[1], 1.0)
        self.assertEqual(scaled[2], 11400.0)
        self.assertEqual(scaled[3], 110.0)
        self.assertEqual(scaled[4], 0.3)

    def test_scale_ordered_values_supports_legacy_54_width_scaler_projection(self):
        artifacts = ModelArtifacts(
            artifacts_dir=".",
            model=_ProbaModel(),
            scaler=_Legacy54WidthScaler(),
            feature_columns=[
                "Age",
                "Income_Category",
                "Essential_Needs_Percentage",
                "Product_Lifetime_Clothing",
                "Product_Lifetime_Tech",
                "Product_Lifetime_Appliances",
                "Product_Lifetime_Cars",
                "Gender_Male",
            ]
            + [f"Binary_{index}" for index in range(48)],
            thresholds={},
            model_metadata={"model_type": "multitask_net", "multitask": True},
            bank_mapping_rules={},
        )
        predictor = Predictor(artifacts)

        row = [
            30.0,
            5700.0,
            55.0,
            12.0,
            24.0,
            48.0,
            120.0,
            1.0,
        ] + [0.0 for _ in range(48)]

        scaled = predictor.scale_ordered_values(row)

        self.assertEqual(scaled[0], 30.0)
        self.assertEqual(scaled[1], 5700.0)
        self.assertEqual(scaled[2], 550.0)
        self.assertEqual(scaled[3], 120.0)
        self.assertEqual(scaled[4], 240.0)
        self.assertEqual(scaled[5], 480.0)
        self.assertEqual(scaled[6], 1200.0)
        self.assertEqual(scaled[7], 1.0)
        self.assertTrue(all(value == 0.0 for value in scaled[8:]))


if __name__ == "__main__":
    unittest.main()

