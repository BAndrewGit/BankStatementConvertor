import unittest

from src.domain.inference_contracts import ProcessedStatementFeatures, ProfileAnswers
from src.features.feature_assembler import FeatureAssembler


class FeatureAssemblerTests(unittest.TestCase):
    def test_assembler_keeps_model_feature_order_and_rejects_missing_required_values(self):
        feature_columns = [
            "Expense_Distribution_Food",
            "Gender_Male",
            "Ignored_Field",
        ]

        assembler = FeatureAssembler(feature_columns)

        statement = ProcessedStatementFeatures.from_mapping({"Expense_Distribution_Food": 25.5})
        profile = ProfileAnswers.from_mapping({"Gender_Male": 1.0})

        result = assembler.assemble(statement_features=statement, profile_answers=profile)
        self.assertEqual(result.row.ordered_columns, feature_columns)
        self.assertEqual(result.row.as_ordered_list(), [25.5, 1.0, 0.0])

        with self.assertRaises(ValueError):
            assembler.assemble(
                statement_features=statement,
                profile_answers=ProfileAnswers.from_mapping({}),
            )

    def test_savings_obstacles_are_zero_when_save_money_yes_or_unanswered(self):
        feature_columns = [
            "Save_Money_Yes",
            "Savings_Obstacle_Insufficient_Income",
            "Savings_Obstacle_Other_Expenses",
        ]
        assembler = FeatureAssembler(feature_columns)

        # When Save_Money_Yes is active, obstacle flags are forced to zero.
        statement_yes = ProcessedStatementFeatures.from_mapping({"Save_Money_Yes": 1.0})
        profile_answered = ProfileAnswers.from_mapping(
            {
                "Savings_Obstacle_Insufficient_Income": 1.0,
                "Savings_Obstacle_Other_Expenses": 0.0,
            }
        )
        result_yes = assembler.assemble(statement_features=statement_yes, profile_answers=profile_answered)
        self.assertEqual(result_yes.row.as_ordered_list(), [1.0, 0.0, 0.0])

        # When Save_Money_Yes is not active and obstacles are unanswered, keep all zeros.
        statement_no = ProcessedStatementFeatures.from_mapping({"Save_Money_Yes": 0.0})
        profile_empty = ProfileAnswers.from_mapping({})
        result_no = assembler.assemble(statement_features=statement_no, profile_answers=profile_empty)
        self.assertEqual(result_no.row.as_ordered_list(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()


