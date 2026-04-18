import unittest

from src.profile.questionnaire import (
    PROFILE_MULTI_FEATURE_GROUPS,
    PROFILE_ORDINAL_FEATURES,
    REQUIRED_NUMERIC_FEATURES,
    QUESTION_GROUPS,
    PROFILE_FEATURE_GROUPS,
    map_raw_profile_inputs_to_one_hot,
    map_questionnaire_answers_to_one_hot,
    questionnaire_answers_complete,
    validate_questionnaire_groups_against_features,
)


class QuestionnaireTests(unittest.TestCase):
    @staticmethod
    def _build_complete_raw_payload():
        return {
            "single_choice": {
                "Family status": "Single, no children",
                "Gender": "Male",
                "Financial attitude": "I try to find a balance",
                "Budget planning": "I plan only essentials",
                "Impulse buying reason": "Self-reward",
                "Financial investments": "Yes, occasionally",
                "Credit usage": "Essential needs",
                "Savings obstacles": "Insufficient income",
            },
            "ordinal_choice": {
                "Debt level": "Manageable",
                "Bank account analysis frequency": "Weekly",
            },
            "multi_select": {
                "Savings goals": ["Major purchases", "Vacation"],
            },
            "numeric": {
                "Age": 30,
                "Product_Lifetime_Clothing": 12,
                "Product_Lifetime_Tech": 24,
                "Product_Lifetime_Appliances": 48,
                "Product_Lifetime_Cars": 120,
            },
        }

    def test_8_questions_produce_full_one_hot_groups(self):
        answers = {
            "Family status": "Single, no children",
            "Gender": "Male",
            "Financial attitude": "I try to find a balance",
            "Budget planning": "I plan only essentials",
            "Impulse buying reason": "Self-reward",
            "Financial investments": "Yes, occasionally",
            "Credit usage": "Essential needs",
            "Savings obstacles": "Insufficient income",
        }

        one_hot = map_questionnaire_answers_to_one_hot(answers)

        for group_name, options in QUESTION_GROUPS.items():
            values = [one_hot[column] for column in options.values()]
            self.assertEqual(sum(values), 1.0, msg=f"Group {group_name} must be one-hot")

    def test_schema_validation_detects_missing_questionnaire_columns(self):
        full_columns = [col for options in PROFILE_FEATURE_GROUPS.values() for col in options]
        full_columns += [col for options in PROFILE_MULTI_FEATURE_GROUPS.values() for col in options]
        full_columns += list(PROFILE_ORDINAL_FEATURES.values())
        full_columns += list(REQUIRED_NUMERIC_FEATURES)
        self.assertEqual(validate_questionnaire_groups_against_features(full_columns), {})

        # Models may not include questionnaire columns at all; this is valid.
        self.assertEqual(validate_questionnaire_groups_against_features(["Expense_Distribution_Food"]), {})

        partial = [column for column in full_columns if column != "Gender_Male"]
        missing = validate_questionnaire_groups_against_features(partial)
        self.assertIn("Gender", missing)
        self.assertGreaterEqual(len(missing["Gender"]), 1)

    def test_questionnaire_completion_requires_one_answer_per_group(self):
        one_hot = map_raw_profile_inputs_to_one_hot(self._build_complete_raw_payload())
        self.assertTrue(questionnaire_answers_complete(one_hot))

        incomplete = dict(one_hot)
        incomplete["Gender_Male"] = 0.0
        self.assertFalse(questionnaire_answers_complete(incomplete))

        missing_numeric = dict(one_hot)
        del missing_numeric["Age"]
        self.assertFalse(questionnaire_answers_complete(missing_numeric))

    def test_raw_input_encoding_supports_multiselect_and_optional_income(self):
        encoded = map_raw_profile_inputs_to_one_hot(self._build_complete_raw_payload())

        self.assertEqual(encoded["Savings_Goal_Major_Purchases"], 1.0)
        self.assertEqual(encoded["Savings_Goal_Vacation"], 1.0)
        self.assertEqual(encoded["Savings_Goal_Emergency_Fund"], 0.0)
        self.assertEqual(encoded["Savings_Goal_Retirement"], 0.0)
        self.assertEqual(encoded["Credit_Usage_Essential_Needs"], 1.0)
        self.assertEqual(encoded["Credit_Usage_Major_Purchases"], 0.0)
        self.assertEqual(encoded["Debt_Level"], 2.0)
        self.assertEqual(encoded["Bank_Account_Analysis_Frequency"], 2.0)
        self.assertNotIn("Income_Category", encoded)

        for field_name in REQUIRED_NUMERIC_FEATURES:
            self.assertIn(field_name, encoded)

    def test_raw_input_encoding_accepts_semicolon_multiselect_strings(self):
        payload = self._build_complete_raw_payload()
        payload["multi_select"] = {
            "Savings goals": "Major purchases;Vacation;Emergency fund",
        }
        encoded = map_raw_profile_inputs_to_one_hot(payload)
        self.assertEqual(encoded["Savings_Goal_Major_Purchases"], 1.0)
        self.assertEqual(encoded["Savings_Goal_Vacation"], 1.0)
        self.assertEqual(encoded["Savings_Goal_Emergency_Fund"], 1.0)
        self.assertEqual(encoded["Savings_Goal_Other"], 0.0)

    def test_savings_obstacles_can_be_left_empty_and_remain_all_zero(self):
        payload = self._build_complete_raw_payload()
        payload["single_choice"]["Savings obstacles"] = ""
        encoded = map_raw_profile_inputs_to_one_hot(payload)

        self.assertEqual(encoded["Savings_Obstacle_Other"], 0.0)
        self.assertEqual(encoded["Savings_Obstacle_Insufficient_Income"], 0.0)
        self.assertEqual(encoded["Savings_Obstacle_Other_Expenses"], 0.0)
        self.assertEqual(encoded["Savings_Obstacle_Not_Priority"], 0.0)
        self.assertTrue(questionnaire_answers_complete(encoded))


if __name__ == "__main__":
    unittest.main()










