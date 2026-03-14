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
]


def build_feature_vector(expense_totals: Dict[str, float]) -> Dict[str, float]:
    food = float(expense_totals.get("food_total", 0.0))
    housing = float(expense_totals.get("housing_total", 0.0))
    transport = float(expense_totals.get("transport_total", 0.0))
    entertainment = float(expense_totals.get("entertainment_total", 0.0))
    health = float(expense_totals.get("health_total", 0.0))
    personal_care = float(expense_totals.get("personal_care_total", 0.0))
    child_education = float(expense_totals.get("child_education_total", 0.0))
    other = float(expense_totals.get("other_total", 0.0))
    unknown = float(expense_totals.get("unknown_total", 0.0))

    consumer_expense_total = (
        food
        + housing
        + transport
        + entertainment
        + health
        + personal_care
        + child_education
        + other
        + unknown
    )

    if consumer_expense_total <= 0:
        LOGGER.warning("consumer_expense_total is zero; setting feature distributions to 0.0")
        return {column: 0.0 for column in FEATURE_COLUMNS}

    essential_needs = food + housing + transport + health + personal_care + child_education
    discretionary = entertainment + other

    feature_vector = {
        "Expense_Distribution_Food": round((food / consumer_expense_total) * 100.0, 4),
        "Expense_Distribution_Housing": round((housing / consumer_expense_total) * 100.0, 4),
        "Expense_Distribution_Transport": round((transport / consumer_expense_total) * 100.0, 4),
        "Expense_Distribution_Entertainment": round((entertainment / consumer_expense_total) * 100.0, 4),
        "Expense_Distribution_Health": round((health / consumer_expense_total) * 100.0, 4),
        "Expense_Distribution_Personal_Care": round((personal_care / consumer_expense_total) * 100.0, 4),
        "Expense_Distribution_Child_Education": round((child_education / consumer_expense_total) * 100.0, 4),
        "Expense_Distribution_Other": round((other / consumer_expense_total) * 100.0, 4),
        "Unknown_Expense_Percentage": round((unknown / consumer_expense_total) * 100.0, 4),
        "Essential_Needs_Percentage": round((essential_needs / consumer_expense_total) * 100.0, 4),
        "Discretionary_Spending_Percentage": round((discretionary / consumer_expense_total) * 100.0, 4),
    }

    return feature_vector

