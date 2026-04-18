import unittest

from src.domain.inference_contracts import (
    InferenceInputRow,
    SOURCE_IGNORE,
    SOURCE_PROGRAM,
    SOURCE_QUESTIONNAIRE,
    build_feature_source_map,
)


class InferenceContractsTests(unittest.TestCase):
    def test_build_feature_source_map_marks_program_questionnaire_and_ignore(self):
        feature_columns = [
            "Expense_Distribution_Food",
            "Gender_Male",
            "Totally_Unknown_Field",
        ]

        source_map = build_feature_source_map(feature_columns)

        self.assertEqual(source_map["Expense_Distribution_Food"], SOURCE_PROGRAM)
        self.assertEqual(source_map["Gender_Male"], SOURCE_QUESTIONNAIRE)
        self.assertEqual(source_map["Totally_Unknown_Field"], SOURCE_IGNORE)

    def test_inference_input_row_requires_exact_columns_and_non_null_values(self):
        ordered = ["a", "b"]

        row = InferenceInputRow.from_values({"a": 1.0, "b": 2}, ordered)
        self.assertEqual(row.as_ordered_list(), [1.0, 2.0])

        with self.assertRaises(ValueError):
            InferenceInputRow.from_values({"a": 1.0}, ordered)

        with self.assertRaises(ValueError):
            InferenceInputRow.from_values({"a": 1.0, "b": None}, ordered)

        with self.assertRaises(ValueError):
            InferenceInputRow.from_values({"a": 1.0, "b": 2.0, "c": 3.0}, ordered)

    def test_inference_input_row_projection_keeps_only_model_columns(self):
        ordered = ["a", "b"]
        row = InferenceInputRow.from_projected_values(
            {"a": 1.0, "b": 2.0, "extra": 9.0},
            ordered,
        )
        self.assertEqual(row.as_ordered_list(), [1.0, 2.0])

        with self.assertRaises(ValueError):
            InferenceInputRow.from_projected_values({"a": 1.0}, ordered)


if __name__ == "__main__":
    unittest.main()

