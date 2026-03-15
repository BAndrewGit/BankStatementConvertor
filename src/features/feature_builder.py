from __future__ import annotations

import logging
from typing import Dict


LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Expense_Distribution_Food",
    "Expense_Distribution_Housing",
    "Expense_Distribution_Transport",
    "Expense_Distribution_Entertainment",
    "Expense_Distribution_Health",
    "Expense_Distribution_Personal_Care",
    "Expense_Distribution_Child_Education",
    "Expense_Distribution_Other",
    "Unknown_Expense_Percentage",
    "Essential_Needs_Percentage",
    "Discretionary_Spending_Percentage",
    "Save_Money_No",
    "Save_Money_Yes",
    "Impulse_Buying_Frequency",
    "Impulse_Buying_Category_Clothing or personal care products",
    "Impulse_Buying_Category_Electronics or gadgets",
    "Impulse_Buying_Category_Entertainment",
    "Impulse_Buying_Category_Food",
    "Impulse_Buying_Category_Other",
]

FINAL_DATASET_COLUMNS = [
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


def _safe_non_negative(value: float) -> float:
    return max(0.0, float(value))


def _normalize_distribution_percentages(amounts: Dict[str, float]) -> Dict[str, float]:
    total = sum(amounts.values())
    if total <= 0:
        return {key: 0.0 for key in amounts}

    raw = {key: (value / total) * 100.0 for key, value in amounts.items()}
    rounded = {key: round(value, 4) for key, value in raw.items()}

    delta = round(100.0 - sum(rounded.values()), 4)
    if abs(delta) > 0:
        # Keep output deterministic and sum at exactly 100 after rounding.
        target_key = max(amounts.keys(), key=lambda key: (amounts[key], key))
        rounded[target_key] = round(max(0.0, rounded[target_key] + delta), 4)

    return rounded


def _impulse_frequency_band(impulse_ratio: float) -> float:
    # Ordinal mapping requested by model input:
    # Never=0, Rarely=1, Sometimes=2, Often=3, Always=4
    if impulse_ratio <= 0.0:
        return 0.0
    if impulse_ratio <= 0.20:
        return 1.0
    if impulse_ratio <= 0.45:
        return 2.0
    if impulse_ratio <= 0.75:
        return 3.0
    return 4.0


def _build_single_impulse_category_flags(
    expense_totals: Dict[str, float],
    impulse_spend: Dict[str, float],
    electronics_total: float,
) -> Dict[str, float]:
    categories = {
        "Impulse_Buying_Category_Clothing or personal care products": "impulse_tx_count_clothing_personal_care",
        "Impulse_Buying_Category_Electronics or gadgets": "impulse_tx_count_electronics_gadgets",
        "Impulse_Buying_Category_Entertainment": "impulse_tx_count_entertainment",
        "Impulse_Buying_Category_Food": "impulse_tx_count_food",
        "Impulse_Buying_Category_Other": "impulse_tx_count_other",
    }

    scored = []
    for index, (category, tx_count_key) in enumerate(categories.items()):
        tx_count = _safe_non_negative(expense_totals.get(tx_count_key, 0.0))
        spend = _safe_non_negative(impulse_spend.get(category, 0.0))
        if category == "Impulse_Buying_Category_Electronics or gadgets" and electronics_total > 0:
            tx_count = max(tx_count, 1.0)
            spend = max(spend, electronics_total)
        scored.append((category, tx_count, spend, index))

    has_signal = any(tx_count > 0 or spend > 0 for _, tx_count, spend, _ in scored)
    if not has_signal:
        return {category: 0.0 for category in categories}

    winner, _, _, _ = max(scored, key=lambda item: (item[1], item[2], -item[3]))
    return {category: (1.0 if category == winner else 0.0) for category in categories}


def build_feature_vector(expense_totals: Dict[str, float]) -> Dict[str, float]:
    known_amounts = {
        "Expense_Distribution_Food": _safe_non_negative(expense_totals.get("food_total", 0.0)),
        "Expense_Distribution_Housing": _safe_non_negative(expense_totals.get("housing_total", 0.0)),
        "Expense_Distribution_Transport": _safe_non_negative(expense_totals.get("transport_total", 0.0)),
        "Expense_Distribution_Entertainment": _safe_non_negative(expense_totals.get("entertainment_total", 0.0)),
        "Expense_Distribution_Health": _safe_non_negative(expense_totals.get("health_total", 0.0)),
        "Expense_Distribution_Personal_Care": _safe_non_negative(expense_totals.get("personal_care_total", 0.0)),
        "Expense_Distribution_Child_Education": _safe_non_negative(expense_totals.get("child_education_total", 0.0)),
        "Expense_Distribution_Other": _safe_non_negative(expense_totals.get("other_total", 0.0)),
    }
    unknown = _safe_non_negative(expense_totals.get("unknown_total", 0.0))
    electronics_total = _safe_non_negative(expense_totals.get("electronics_gadgets_total", 0.0))
    income_total = _safe_non_negative(expense_totals.get("income_total", 0.0))
    outgoing_total = _safe_non_negative(expense_totals.get("outgoing_expense_total", 0.0))
    impulse_candidate_tx_count = _safe_non_negative(expense_totals.get("impulse_candidate_tx_count", 0.0))
    outgoing_tx_count = _safe_non_negative(expense_totals.get("outgoing_tx_count", 0.0))
    impulse_spend = {
        "Impulse_Buying_Category_Clothing or personal care products": _safe_non_negative(
            expense_totals.get("impulse_spend_clothing_personal_care", 0.0)
        ),
        "Impulse_Buying_Category_Electronics or gadgets": _safe_non_negative(
            expense_totals.get("impulse_spend_electronics_gadgets", 0.0)
        ),
        "Impulse_Buying_Category_Entertainment": _safe_non_negative(
            expense_totals.get("impulse_spend_entertainment", 0.0)
        ),
        "Impulse_Buying_Category_Food": _safe_non_negative(
            expense_totals.get("impulse_spend_food", 0.0)
        ),
        "Impulse_Buying_Category_Other": _safe_non_negative(
            expense_totals.get("impulse_spend_other", 0.0)
        ),
    }

    known_total = sum(known_amounts.values())
    total_including_unknown = known_total + unknown

    if total_including_unknown <= 0:
        LOGGER.warning("consumer_expense_total is zero; setting feature distributions to 0.0")
        return {column: 0.0 for column in FEATURE_COLUMNS}

    distribution = _normalize_distribution_percentages(known_amounts)

    essential_needs = (
        distribution["Expense_Distribution_Food"]
        + distribution["Expense_Distribution_Housing"]
        + distribution["Expense_Distribution_Transport"]
        + distribution["Expense_Distribution_Health"]
        + distribution["Expense_Distribution_Personal_Care"]
        + distribution["Expense_Distribution_Child_Education"]
    )
    discretionary = (
        distribution["Expense_Distribution_Entertainment"] + distribution["Expense_Distribution_Other"]
    )

    net_cashflow = income_total - outgoing_total
    save_money_yes = 1.0 if net_cashflow > 0 else 0.0
    save_money_no = 0.0 if net_cashflow > 0 else 1.0

    impulse_flags = _build_single_impulse_category_flags(
        expense_totals=expense_totals,
        impulse_spend=impulse_spend,
        electronics_total=electronics_total,
    )

    impulse_frequency = 0.0
    if outgoing_tx_count > 0:
        impulse_frequency = _impulse_frequency_band(
            impulse_candidate_tx_count / outgoing_tx_count
        )

    feature_vector = {
        **distribution,
        "Unknown_Expense_Percentage": round((unknown / total_including_unknown) * 100.0, 4),
        "Essential_Needs_Percentage": round(essential_needs, 4),
        "Discretionary_Spending_Percentage": round(discretionary, 4),
        "Save_Money_No": save_money_no,
        "Save_Money_Yes": save_money_yes,
        "Impulse_Buying_Frequency": impulse_frequency,
        **impulse_flags,
    }

    return feature_vector


def build_final_dataset_row(feature_vector: Dict[str, float]) -> Dict[str, float | None]:
    row = {column: None for column in FINAL_DATASET_COLUMNS}

    direct_mappings = [
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
    ]
    for column in direct_mappings:
        row[column] = float(feature_vector.get(column, 0.0))

    return row


