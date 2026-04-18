import unittest

from src.domain.inference_contracts import ProcessedStatementFeatures
from src.pipelines.program2_adapter import (
    adapt_program2_output_to_processed_features,
    build_file_traceability,
    detect_profile_required_features,
)


class _FakeRunResult:
    def __init__(self, features):
        self.features = features


class Program2AdapterTests(unittest.TestCase):
    def test_adapter_supports_run_result_and_raw_mapping(self):
        payload = {
            "features": {
                "Expense_Distribution_Food": 10.0,
                "Save_Money_Yes": 1.0,
                "food_total_pct": 15.0,
            }
        }
        adapted = adapt_program2_output_to_processed_features(payload)
        self.assertIsInstance(adapted, ProcessedStatementFeatures)
        # alias should map and override same semantic key if both are present in input
        self.assertEqual(adapted.to_feature_map()["Expense_Distribution_Food"], 15.0)

        run_result = _FakeRunResult(features={"Expense_Distribution_Housing": 20.0})
        adapted_result = adapt_program2_output_to_processed_features(run_result)
        self.assertEqual(adapted_result.to_feature_map()["Expense_Distribution_Housing"], 20.0)

    def test_detect_profile_required_features(self):
        feature_columns = [
            "Expense_Distribution_Food",
            "Gender_Male",
            "Save_Money_Yes",
        ]
        processed = ProcessedStatementFeatures.from_mapping({"Expense_Distribution_Food": 11.0})

        missing = detect_profile_required_features(feature_columns, processed)
        self.assertIn("Save_Money_Yes", missing["missing_program_features"])
        self.assertIn("Gender_Male", missing["required_profile_features"])

    def test_build_file_traceability(self):
        trace = build_file_traceability(["a.pdf", "b.pdf"], processed_ok=True)
        self.assertEqual(len(trace), 2)
        self.assertTrue(all(item["processed"] for item in trace))


if __name__ == "__main__":
    unittest.main()

