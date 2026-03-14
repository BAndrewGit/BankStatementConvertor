import unittest

from src.features.feature_builder import FEATURE_COLUMNS, build_feature_vector


class FeatureBuilderTests(unittest.TestCase):
    def test_feature_vector_has_fixed_columns_and_order(self):
        totals = {
            "food_total": 100.0,
            "housing_total": 100.0,
            "transport_total": 50.0,
            "entertainment_total": 50.0,
            "health_total": 0.0,
            "personal_care_total": 0.0,
            "child_education_total": 0.0,
            "other_total": 100.0,
            "unknown_total": 100.0,
        }

        features = build_feature_vector(totals)
        self.assertEqual(list(features.keys()), FEATURE_COLUMNS)
        self.assertAlmostEqual(features["Unknown_Expense_Percentage"], 20.0, places=4)
        self.assertAlmostEqual(features["Essential_Needs_Percentage"], 50.0, places=4)
        self.assertAlmostEqual(features["Discretionary_Spending_Percentage"], 30.0, places=4)

    def test_zero_total_returns_zeroes(self):
        features = build_feature_vector({})
        self.assertTrue(all(value == 0.0 for value in features.values()))


if __name__ == "__main__":
    unittest.main()

