import unittest

from src.features.feature_builder import (
    FEATURE_COLUMNS,
    FINAL_DATASET_COLUMNS,
    build_feature_vector,
    build_final_dataset_row,
    classify_behavior_risk_level,
)


class FeatureBuilderTests(unittest.TestCase):
    def test_final_dataset_column_order_matches_processing_contract(self):
        expected_order = [
            "Age",
            "Income_Category",
            "Essential_Needs_Percentage",
            "Product_Lifetime_Clothing",
            "Product_Lifetime_Tech",
            "Product_Lifetime_Appliances",
            "Product_Lifetime_Cars",
            "Impulse_Buying_Frequency",
            "Debt_Level",
            "Bank_Account_Analysis_Frequency",
            "Savings_Goal_Major_Purchases",
            "Savings_Goal_Retirement",
            "Savings_Goal_Emergency_Fund",
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
            "Save_Money_No",
            "Save_Money_Yes",
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
            "Risk_Score",
            "Behavior_Risk_Level",
        ]

        self.assertEqual(FINAL_DATASET_COLUMNS, expected_order)

    def test_behavior_risk_level_uses_english_labels(self):
        self.assertEqual(classify_behavior_risk_level(-0.11), "Healthy")
        self.assertEqual(classify_behavior_risk_level(-0.10), "Healthy")
        self.assertEqual(classify_behavior_risk_level(-0.09), "Moderate")
        self.assertEqual(classify_behavior_risk_level(0.20), "Moderate")
        self.assertEqual(classify_behavior_risk_level(0.21), "Risky")

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
        self.assertAlmostEqual(features["Essential_Needs_Percentage"], 62.5, places=4)
        self.assertAlmostEqual(features["Discretionary_Spending_Percentage"], 37.5, places=4)

        distribution_sum = sum(
            features[column]
            for column in FEATURE_COLUMNS
            if column.startswith("Expense_Distribution_")
        )
        self.assertAlmostEqual(distribution_sum, 100.0, places=4)

    def test_rounding_edge_case_distribution_stays_at_100(self):
        totals = {
            "food_total": 1.0,
            "housing_total": 1.0,
            "transport_total": 1.0,
            "entertainment_total": 1.0,
            "health_total": 1.0,
            "personal_care_total": 1.0,
            "child_education_total": 1.0,
            "other_total": 1.0,
            "unknown_total": 0.0,
        }

        features = build_feature_vector(totals)
        distribution_sum = sum(
            features[column]
            for column in FEATURE_COLUMNS
            if column.startswith("Expense_Distribution_")
        )
        self.assertAlmostEqual(distribution_sum, 100.0, places=4)

    def test_negative_totals_are_clamped_to_zero(self):
        totals = {
            "food_total": -10.0,
            "housing_total": 50.0,
            "unknown_total": -5.0,
        }
        features = build_feature_vector(totals)
        self.assertGreaterEqual(features["Expense_Distribution_Housing"], 0.0)
        self.assertEqual(features["Unknown_Expense_Percentage"], 0.0)

    def test_build_final_dataset_row_has_target_columns(self):
        features = build_feature_vector(
            {
                "food_total": 100.0,
                "housing_total": 50.0,
                "transport_total": 50.0,
                "entertainment_total": 0.0,
                "health_total": 0.0,
                "personal_care_total": 0.0,
                "child_education_total": 0.0,
                "other_total": 0.0,
                "electronics_gadgets_total": 25.0,
                "unknown_total": 0.0,
            }
        )
        row = build_final_dataset_row(features)
        self.assertEqual(list(row.keys()), FINAL_DATASET_COLUMNS)
        self.assertEqual(row["Expense_Distribution_Food"], 1)
        self.assertEqual(row["Expense_Distribution_Housing"], 1)
        self.assertEqual(row["Expense_Distribution_Transport"], 1)
        self.assertAlmostEqual(row["Essential_Needs_Percentage"], 100.0, places=4)
        self.assertEqual(row["Impulse_Buying_Category_Electronics or gadgets"], 1)
        self.assertIsNone(row["Age"])

    def test_electronics_impulse_feature_defaults_to_zero_without_signal(self):
        features = build_feature_vector(
            {
                "food_total": 10.0,
                "other_total": 10.0,
                "unknown_total": 0.0,
            }
        )
        self.assertEqual(features["Impulse_Buying_Category_Electronics or gadgets"], 0.0)

    def test_save_money_and_impulse_outputs_are_computed(self):
        features = build_feature_vector(
            {
                "food_total": 120.0,
                "housing_total": 200.0,
                "entertainment_total": 80.0,
                "other_total": 50.0,
                "unknown_total": 0.0,
                "income_total": 3000.0,
                "outgoing_expense_total": 450.0,
                "outgoing_tx_count": 10.0,
                "impulse_candidate_tx_count": 4.0,
                "impulse_spend_clothing_personal_care": 20.0,
                "impulse_spend_electronics_gadgets": 200.0,
                "impulse_spend_entertainment": 80.0,
                "impulse_spend_food": 30.0,
                "impulse_spend_other": 10.0,
                "impulse_tx_count_clothing_personal_care": 1.0,
                "impulse_tx_count_electronics_gadgets": 2.0,
                "impulse_tx_count_entertainment": 1.0,
                "impulse_tx_count_food": 0.0,
                "impulse_tx_count_other": 0.0,
            }
        )

        self.assertEqual(features["Save_Money_Yes"], 1)
        self.assertEqual(features["Save_Money_No"], 0)
        self.assertEqual(features["Impulse_Buying_Frequency"], 2)
        self.assertEqual(features["Impulse_Buying_Category_Electronics or gadgets"], 1.0)
        self.assertEqual(features["Impulse_Buying_Category_Food"], 0.0)
        self.assertEqual(features["Impulse_Buying_Category_Entertainment"], 0.0)
        impulse_category_columns = [
            "Impulse_Buying_Category_Clothing or personal care products",
            "Impulse_Buying_Category_Electronics or gadgets",
            "Impulse_Buying_Category_Entertainment",
            "Impulse_Buying_Category_Food",
            "Impulse_Buying_Category_Other",
        ]
        self.assertEqual(sum(features[column] for column in impulse_category_columns), 1.0)

    def test_impulse_category_is_one_hot_with_highest_signal_winner(self):
        features = build_feature_vector(
            {
                "food_total": 100.0,
                "other_total": 50.0,
                "unknown_total": 0.0,
                "impulse_tx_count_clothing_personal_care": 2.0,
                "impulse_tx_count_electronics_gadgets": 2.0,
                "impulse_spend_clothing_personal_care": 60.0,
                "impulse_spend_electronics_gadgets": 120.0,
            }
        )

        self.assertEqual(features["Impulse_Buying_Category_Electronics or gadgets"], 1.0)
        self.assertEqual(features["Impulse_Buying_Category_Clothing or personal care products"], 0.0)
        self.assertEqual(features["Impulse_Buying_Category_Entertainment"], 0.0)
        self.assertEqual(features["Impulse_Buying_Category_Food"], 0.0)
        self.assertEqual(features["Impulse_Buying_Category_Other"], 0.0)

    def test_impulse_frequency_is_encoded_on_0_to_4_scale(self):
        cases = [
            (0.0, 10.0, 0.0),  # Never
            (2.0, 10.0, 1.0),  # Rarely
            (4.0, 10.0, 2.0),  # Sometimes
            (7.0, 10.0, 3.0),  # Often
            (9.0, 10.0, 4.0),  # Always
        ]

        for impulse_count, outgoing_count, expected in cases:
            features = build_feature_vector(
                {
                    "food_total": 100.0,
                    "other_total": 100.0,
                    "unknown_total": 0.0,
                    "outgoing_tx_count": outgoing_count,
                    "impulse_candidate_tx_count": impulse_count,
                }
            )
            self.assertEqual(features["Impulse_Buying_Frequency"], expected)

    def test_save_money_no_when_cashflow_is_negative(self):
        features = build_feature_vector(
            {
                "food_total": 100.0,
                "other_total": 100.0,
                "unknown_total": 0.0,
                "income_total": 50.0,
                "outgoing_expense_total": 250.0,
            }
        )
        self.assertEqual(features["Save_Money_Yes"], 0.0)
        self.assertEqual(features["Save_Money_No"], 1.0)

    def test_zero_total_returns_zeroes(self):
        features = build_feature_vector({})
        self.assertTrue(all(value == 0.0 for value in features.values()))

    def test_behavior_risk_level_uses_requested_thresholds(self):
        self.assertEqual(classify_behavior_risk_level(-0.11), "Healthy")
        self.assertEqual(classify_behavior_risk_level(-0.10), "Healthy")
        self.assertEqual(classify_behavior_risk_level(-0.09), "Moderate")
        self.assertEqual(classify_behavior_risk_level(0.20), "Moderate")
        self.assertEqual(classify_behavior_risk_level(0.21), "Risky")


if __name__ == "__main__":
    unittest.main()

