from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Mapping, Sequence


SOURCE_PROGRAM = "program_2"
SOURCE_QUESTIONNAIRE = "questionnaire"
SOURCE_IGNORE = "ignore"

# Features produced by statement processing.
PROGRAM_FEATURES = {
    "Income_Category",
    "Essential_Needs_Percentage",
    "Expense_Distribution_Food",
    "Expense_Distribution_Housing",
    "Expense_Distribution_Transport",
    "Expense_Distribution_Entertainment",
    "Expense_Distribution_Health",
    "Expense_Distribution_Personal_Care",
    "Expense_Distribution_Child_Education",
    "Expense_Distribution_Other",
    "Save_Money_No",
    "Save_Money_Yes",
    "Impulse_Buying_Frequency",
    "Impulse_Buying_Category_Clothing or personal care products",
    "Impulse_Buying_Category_Electronics or gadgets",
    "Impulse_Buying_Category_Entertainment",
    "Impulse_Buying_Category_Food",
    "Impulse_Buying_Category_Other",
}

# Features filled from explicit profile questionnaire answers.
QUESTIONNAIRE_FEATURES = {
    "Age",
    "Product_Lifetime_Clothing",
    "Product_Lifetime_Tech",
    "Product_Lifetime_Appliances",
    "Product_Lifetime_Cars",
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
    "Impulse_Buying_Reason_Discounts or promotions",
    "Impulse_Buying_Reason_Other",
    "Impulse_Buying_Reason_Self-reward",
    "Impulse_Buying_Reason_Social pressure",
    "Financial_Investments_No, but interested",
    "Financial_Investments_No, not interested",
    "Financial_Investments_Yes, occasionally",
    "Financial_Investments_Yes, regularly",
}


@dataclass(frozen=True)
class ProfileAnswers:
    """Normalized questionnaire payload used for inference assembly."""

    values: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ProfileAnswers":
        normalized: Dict[str, float] = {}
        for key, value in payload.items():
            if value is None:
                continue
            normalized[str(key)] = float(value)
        return cls(values=normalized)

    def to_feature_map(self) -> Dict[str, float]:
        return dict(self.values)


@dataclass(frozen=True)
class ProcessedStatementFeatures:
    """Features produced by statement pipeline before model inference."""

    values: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ProcessedStatementFeatures":
        normalized: Dict[str, float] = {}
        for key, value in payload.items():
            if value is None:
                continue
            normalized[str(key)] = float(value)
        return cls(values=normalized)

    def to_feature_map(self) -> Dict[str, float]:
        return dict(self.values)


@dataclass(frozen=True)
class InferenceInputRow:
    """Strict model-ready row with deterministic feature ordering."""

    values: Dict[str, float]
    ordered_columns: List[str]

    @classmethod
    def from_values(cls, values: Mapping[str, object], ordered_columns: Sequence[str]) -> "InferenceInputRow":
        normalized: Dict[str, float] = {}
        for column in ordered_columns:
            if column not in values:
                raise ValueError(f"Missing required feature: {column}")
            raw = values[column]
            if raw is None:
                raise ValueError(f"Null value for feature: {column}")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite value for feature: {column}")
            normalized[column] = value

        extras = sorted(set(values.keys()) - set(ordered_columns))
        if extras:
            raise ValueError(f"Unexpected extra features: {extras}")

        return cls(values=normalized, ordered_columns=list(ordered_columns))

    @classmethod
    def from_projected_values(
        cls,
        values: Mapping[str, object],
        ordered_columns: Sequence[str],
    ) -> "InferenceInputRow":
        """Build strict model row from a wider mapping by projecting only required model columns."""

        projected: Dict[str, object] = {}
        for column in ordered_columns:
            if column not in values:
                raise ValueError(f"Missing required feature: {column}")
            projected[column] = values[column]
        return cls.from_values(projected, ordered_columns=ordered_columns)

    def as_ordered_list(self) -> List[float]:
        return [self.values[column] for column in self.ordered_columns]


def build_feature_source_map(feature_columns: Sequence[str]) -> Dict[str, str]:
    """Return explicit source for each model feature: program_2/questionnaire/ignore."""

    mapping: Dict[str, str] = {}
    for column in feature_columns:
        if column in PROGRAM_FEATURES:
            mapping[column] = SOURCE_PROGRAM
        elif column in QUESTIONNAIRE_FEATURES:
            mapping[column] = SOURCE_QUESTIONNAIRE
        else:
            mapping[column] = SOURCE_IGNORE
    return mapping

