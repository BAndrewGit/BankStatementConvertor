from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence


QUESTION_GROUPS: Dict[str, Dict[str, str]] = {
    "Family status": {
        "In a relationship / married with children": "Family_Status_In a relationship/married with children",
        "In a relationship / married without children": "Family_Status_In a relationship/married without children",
        "Single, no children": "Family_Status_Single, no children",
        "Single, with children": "Family_Status_Single, with children",
        "Other": "Family_Status_Another",
    },
    "Gender": {
        "Female": "Gender_Female",
        "Male": "Gender_Male",
        "Prefer not to say": "Gender_Prefer not to say",
    },
    "Financial attitude": {
        "I am disciplined in saving": "Financial_Attitude_I am disciplined in saving",
        "I try to find a balance": "Financial_Attitude_I try to find a balance",
        "I spend more than I earn": "Financial_Attitude_Spend more than I earn",
    },
    "Budget planning": {
        "I do not plan at all": "Budget_Planning_Don't plan at all",
        "I plan my budget in detail": "Budget_Planning_Plan budget in detail",
        "I plan only essentials": "Budget_Planning_Plan only essentials",
    },
    "Impulse buying reason": {
        "Discounts or promotions": "Impulse_Buying_Reason_Discounts or promotions",
        "Self-reward": "Impulse_Buying_Reason_Self-reward",
        "Social pressure": "Impulse_Buying_Reason_Social pressure",
        "Other": "Impulse_Buying_Reason_Other",
    },
    "Financial investments": {
        "No, but interested": "Financial_Investments_No, but interested",
        "No, not interested": "Financial_Investments_No, not interested",
        "Yes, occasionally": "Financial_Investments_Yes, occasionally",
        "Yes, regularly": "Financial_Investments_Yes, regularly",
    },
    "Credit usage": {
        "Essential needs": "Credit_Usage_Essential_Needs",
        "Major purchases": "Credit_Usage_Major_Purchases",
        "Unexpected expenses": "Credit_Usage_Unexpected_Expenses",
        "Personal needs": "Credit_Usage_Personal_Needs",
        "I never use credit": "Credit_Usage_Never_Used",
    },
    "Savings obstacles": {
        "Insufficient income": "Savings_Obstacle_Insufficient_Income",
        "Other expenses": "Savings_Obstacle_Other_Expenses",
        "Not a priority": "Savings_Obstacle_Not_Priority",
        "Other": "Savings_Obstacle_Other",
    },
}

MULTI_SELECT_GROUPS: Dict[str, Dict[str, str]] = {
    "Savings goals": {
        "Major purchases": "Savings_Goal_Major_Purchases",
        "Retirement": "Savings_Goal_Retirement",
        "Emergency fund": "Savings_Goal_Emergency_Fund",
        "Child education": "Savings_Goal_Child_Education",
        "Vacation": "Savings_Goal_Vacation",
        "Other": "Savings_Goal_Other",
    },
}

ORDINAL_CHOICE_GROUPS: Dict[str, Dict[str, float]] = {
    "Debt level": {
        "Absent": 0.0,
        "Low": 1.0,
        "Manageable": 2.0,
        "High": 3.0,
    },
    "Bank account analysis frequency": {
        "Daily": 3.0,
        "Weekly": 2.0,
        "Monthly": 1.0,
        "Rarely or never": 0.0,
    },
}

ORDINAL_CHOICE_FEATURES: Dict[str, str] = {
    "Debt level": "Debt_Level",
    "Bank account analysis frequency": "Bank_Account_Analysis_Frequency",
}

OPTIONAL_SINGLE_CHOICE_GROUPS = {"Savings obstacles"}

NUMERIC_FIELDS: Dict[str, str] = {
    "Age": "Age",
    "Product lifetime - clothing": "Product_Lifetime_Clothing",
    "Product lifetime - tech": "Product_Lifetime_Tech",
    "Product lifetime - appliances": "Product_Lifetime_Appliances",
    "Product lifetime - cars": "Product_Lifetime_Cars",
}

REQUIRED_NUMERIC_FEATURES = (
    "Age",
    "Product_Lifetime_Clothing",
    "Product_Lifetime_Tech",
    "Product_Lifetime_Appliances",
    "Product_Lifetime_Cars",
)

PROFILE_FEATURE_GROUPS: Dict[str, List[str]] = {
    group_name: list(option_map.values()) for group_name, option_map in QUESTION_GROUPS.items()
}

PROFILE_MULTI_FEATURE_GROUPS: Dict[str, List[str]] = {
    group_name: list(option_map.values()) for group_name, option_map in MULTI_SELECT_GROUPS.items()
}

PROFILE_ORDINAL_FEATURES: Dict[str, str] = dict(ORDINAL_CHOICE_FEATURES)

OPTIONAL_GROUP_FEATURES: Dict[str, set[str]] = {
    "Savings goals": {"Savings_Goal_Emergency_Fund"},
}


def map_questionnaire_answers_to_one_hot(answers: Mapping[str, str]) -> Dict[str, float]:
    """Convert selected options into one-hot model features for the 8 questionnaire groups."""

    output: Dict[str, float] = {}
    for group_name, option_map in QUESTION_GROUPS.items():
        selected = answers.get(group_name)
        if selected not in option_map:
            if group_name in OPTIONAL_SINGLE_CHOICE_GROUPS and selected in {None, ""}:
                for feature_name in option_map.values():
                    output[feature_name] = 0.0
                continue
            raise ValueError(f"Missing or invalid answer for '{group_name}'")

        for option_label, feature_name in option_map.items():
            output[feature_name] = 1.0 if option_label == selected else 0.0

    return output


def selected_options_from_one_hot(one_hot_values: Mapping[str, float]) -> Dict[str, str]:
    """Recover UI selected options from one-hot values persisted in profile."""

    selected: Dict[str, str] = {}
    for group_name, option_map in QUESTION_GROUPS.items():
        best_option = None
        best_value = float("-inf")
        for option_label, feature_name in option_map.items():
            value = float(one_hot_values.get(feature_name, 0.0))
            if value > best_value:
                best_value = value
                best_option = option_label
        if best_option is not None and best_value >= 0.5:
            selected[group_name] = best_option
    return selected


def selected_multi_options_from_one_hot(one_hot_values: Mapping[str, float]) -> Dict[str, List[str]]:
    """Recover selected multi-choice options from one-hot values persisted in profile."""

    selected: Dict[str, List[str]] = {}
    for group_name, option_map in MULTI_SELECT_GROUPS.items():
        group_selected: List[str] = []
        for option_label, feature_name in option_map.items():
            if float(one_hot_values.get(feature_name, 0.0)) >= 0.5:
                group_selected.append(option_label)
        selected[group_name] = group_selected
    return selected


def selected_numeric_values_from_one_hot(one_hot_values: Mapping[str, float]) -> Dict[str, float]:
    """Recover numeric profile fields from one-hot values persisted in profile."""

    selected: Dict[str, float] = {}
    for feature_name in NUMERIC_FIELDS.values():
        if feature_name in one_hot_values:
            selected[feature_name] = float(one_hot_values[feature_name])
    return selected


def selected_ordinal_options_from_values(one_hot_values: Mapping[str, float]) -> Dict[str, str]:
    """Recover selected ordinal options from persisted profile values."""

    selected: Dict[str, str] = {}
    for group_name, option_map in ORDINAL_CHOICE_GROUPS.items():
        feature_name = ORDINAL_CHOICE_FEATURES[group_name]
        if feature_name not in one_hot_values:
            continue
        value = float(one_hot_values.get(feature_name, 0.0))
        for label, encoded_value in option_map.items():
            if abs(value - float(encoded_value)) < 1e-6:
                selected[group_name] = label
                break
    return selected


def _normalize_multi_selected(raw_value: object) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(";") if item.strip()]
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    raise ValueError("Multi-select answers must be a ';'-separated string or a list of options")


def _parse_float(raw_value: object, field_name: str) -> float:
    if raw_value is None:
        raise ValueError(f"Missing numeric value for '{field_name}'")
    if isinstance(raw_value, str) and not raw_value.strip():
        raise ValueError(f"Missing numeric value for '{field_name}'")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"Invalid numeric value for '{field_name}'")
    return value


def _parse_lifetime(raw_value: object, field_name: str) -> float:
    if isinstance(raw_value, str) and raw_value.strip().lower() in {"not purchased yet", "not_purchased"}:
        return 0.0
    return _parse_float(raw_value, field_name)


def map_raw_profile_inputs_to_one_hot(raw_inputs: Mapping[str, object]) -> Dict[str, float]:
    """Map raw human inputs into encoded model features (numeric/one-hot/multi-hot)."""

    single_choice = raw_inputs.get("single_choice") or {}
    ordinal_choice = raw_inputs.get("ordinal_choice") or {}
    numeric = raw_inputs.get("numeric") or {}
    multi_select = raw_inputs.get("multi_select") or {}

    if not isinstance(single_choice, Mapping):
        raise ValueError("single_choice payload must be a mapping")
    if not isinstance(numeric, Mapping):
        raise ValueError("numeric payload must be a mapping")
    if not isinstance(multi_select, Mapping):
        raise ValueError("multi_select payload must be a mapping")
    if not isinstance(ordinal_choice, Mapping):
        raise ValueError("ordinal_choice payload must be a mapping")

    output: Dict[str, float] = {}

    for group_name, option_map in QUESTION_GROUPS.items():
        selected = single_choice.get(group_name)
        if selected not in option_map:
            if group_name in OPTIONAL_SINGLE_CHOICE_GROUPS and selected in {None, ""}:
                for feature_name in option_map.values():
                    output[feature_name] = 0.0
                continue
            raise ValueError(f"Missing or invalid answer for '{group_name}'")
        for option_label, feature_name in option_map.items():
            output[feature_name] = 1.0 if option_label == selected else 0.0

    for group_name, option_map in MULTI_SELECT_GROUPS.items():
        selected_options = set(_normalize_multi_selected(multi_select.get(group_name)))
        if not selected_options:
            raise ValueError(f"Select at least one option for '{group_name}'")

        unknown = sorted(selected_options - set(option_map.keys()))
        if unknown:
            raise ValueError(f"Invalid options for '{group_name}': {unknown}")

        for option_label, feature_name in option_map.items():
            output[feature_name] = 1.0 if option_label in selected_options else 0.0

    for group_name, option_map in ORDINAL_CHOICE_GROUPS.items():
        selected = ordinal_choice.get(group_name)
        if selected not in option_map:
            raise ValueError(f"Missing or invalid answer for '{group_name}'")
        target_feature = ORDINAL_CHOICE_FEATURES[group_name]
        output[target_feature] = float(option_map[str(selected)])

    output["Age"] = _parse_float(numeric.get("Age"), "Age")
    output["Product_Lifetime_Clothing"] = _parse_lifetime(
        numeric.get("Product_Lifetime_Clothing"), "Product_Lifetime_Clothing"
    )
    output["Product_Lifetime_Tech"] = _parse_lifetime(
        numeric.get("Product_Lifetime_Tech"), "Product_Lifetime_Tech"
    )
    output["Product_Lifetime_Appliances"] = _parse_lifetime(
        numeric.get("Product_Lifetime_Appliances"), "Product_Lifetime_Appliances"
    )
    output["Product_Lifetime_Cars"] = _parse_lifetime(
        numeric.get("Product_Lifetime_Cars"), "Product_Lifetime_Cars"
    )

    return output


def validate_questionnaire_groups_against_features(feature_columns: Sequence[str]) -> Dict[str, List[str]]:
    """Return missing columns only for partially represented questionnaire groups in model schema."""

    available = set(feature_columns)
    missing: Dict[str, List[str]] = {}
    for group_name, columns in PROFILE_FEATURE_GROUPS.items():
        present_count = sum(1 for column in columns if column in available)
        if present_count == 0:
            continue
        optional_columns = OPTIONAL_GROUP_FEATURES.get(group_name, set())
        required_columns = [column for column in columns if column not in optional_columns]
        missing_columns = [column for column in required_columns if column not in available]
        if missing_columns:
            missing[group_name] = missing_columns
    for group_name, columns in PROFILE_MULTI_FEATURE_GROUPS.items():
        present_count = sum(1 for column in columns if column in available)
        if present_count == 0:
            continue
        optional_columns = OPTIONAL_GROUP_FEATURES.get(group_name, set())
        required_columns = [column for column in columns if column not in optional_columns]
        missing_columns = [column for column in required_columns if column not in available]
        if missing_columns:
            missing[group_name] = missing_columns
    ordinal_features = list(PROFILE_ORDINAL_FEATURES.values())
    ordinal_present = [feature for feature in ordinal_features if feature in available]
    if ordinal_present and len(ordinal_present) != len(ordinal_features):
        missing["Ordinal fields"] = [feature for feature in ordinal_features if feature not in available]

    numeric_present = [name for name in REQUIRED_NUMERIC_FEATURES if name in available]
    if numeric_present and len(numeric_present) != len(REQUIRED_NUMERIC_FEATURES):
        missing["Numeric fields"] = [name for name in REQUIRED_NUMERIC_FEATURES if name not in available]
    return missing


def questionnaire_answers_complete(one_hot_values: Mapping[str, float]) -> bool:
    """Return True only if single/multi/numeric profile fields are complete for model inference."""

    for group_name, columns in PROFILE_FEATURE_GROUPS.items():
        selected_count = 0
        for column in columns:
            if float(one_hot_values.get(column, 0.0)) >= 0.5:
                selected_count += 1
        if group_name in OPTIONAL_SINGLE_CHOICE_GROUPS:
            if selected_count > 1:
                return False
        elif selected_count != 1:
            return False

    for columns in PROFILE_MULTI_FEATURE_GROUPS.values():
        selected_count = 0
        for column in columns:
            if float(one_hot_values.get(column, 0.0)) >= 0.5:
                selected_count += 1
        if selected_count < 1:
            return False

    for feature_name in PROFILE_ORDINAL_FEATURES.values():
        if feature_name not in one_hot_values:
            return False
        value = float(one_hot_values.get(feature_name, 0.0))
        if not math.isfinite(value):
            return False

    for feature_name in REQUIRED_NUMERIC_FEATURES:
        if feature_name not in one_hot_values:
            return False
        value = float(one_hot_values.get(feature_name, 0.0))
        if not math.isfinite(value):
            return False
    return True






