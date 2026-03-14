from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import os
from typing import Dict, List, Optional, Set, Tuple

from .transaction_normalization import TransactionNormalized
from .transaction_classification import TransactionClassified
from .budget_categorization import BudgetCategorizedTransaction


ESSENTIAL_CATEGORIES = {"Food", "Housing", "Transport", "Health", "Education", "Debt", "Fees"}
DISCRETIONARY_CATEGORIES = {"Entertainment", "Personal_Care", "Cash", "Transfers", "Other"}


@dataclass(frozen=True)
class MonthlyCategoryAggregate:
    month: str
    budget_category: str
    amount_total: float
    transaction_count: int
    expense_percentage: float
    spending_bucket: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MonthlyBudgetOverview:
    month: str
    income_total: float
    expense_total: float
    essential_total: float
    discretionary_total: float
    unknown_total: float
    unknown_transaction_count: int
    total_transactions: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MonthlyAggregationSummary:
    total_input_rows: int
    rows_with_month: int
    rows_missing_month: int
    months_count: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def _derive_month(transaction_date: Optional[str]) -> Optional[str]:
    if not transaction_date:
        return None

    value = transaction_date.strip()
    if not value:
        return None

    if len(value) >= 7:
        candidate = value[:7]
        try:
            datetime.strptime(candidate, "%Y-%m")
            return candidate
        except ValueError:
            pass

    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        return parsed.strftime("%Y-%m")
    except ValueError:
        return None


def aggregate_monthly_budget(
    classified_rows: List[TransactionClassified],
    categorized_rows: List[BudgetCategorizedTransaction],
    normalized_rows: List[TransactionNormalized],
) -> Tuple[List[MonthlyBudgetOverview], List[MonthlyCategoryAggregate], MonthlyAggregationSummary]:
    """Story 11: aggregate categorized transactions monthly for budget analytics."""
    category_by_candidate = {
        row.candidate_id: row
        for row in categorized_rows
    }
    month_by_candidate = {
        row.candidate_id: _derive_month(row.transaction_date)
        for row in normalized_rows
    }

    overview_acc: Dict[str, Dict[str, float]] = {}
    category_acc: Dict[Tuple[str, str], Dict[str, float]] = {}

    rows_with_month = 0
    rows_missing_month = 0

    for row in classified_rows:
        month = month_by_candidate.get(row.candidate_id)
        if not month:
            rows_missing_month += 1
            continue

        rows_with_month += 1
        amount = abs(float(row.amount)) if row.amount is not None else 0.0

        categorized = category_by_candidate.get(row.candidate_id)
        budget_category = categorized.budget_category if categorized else "Unknown"
        if not budget_category:
            budget_category = "Unknown"

        month_totals = overview_acc.setdefault(
            month,
            {
                "income_total": 0.0,
                "expense_total": 0.0,
                "essential_total": 0.0,
                "discretionary_total": 0.0,
                "unknown_total": 0.0,
                "unknown_transaction_count": 0.0,
                "total_transactions": 0.0,
            },
        )
        month_totals["total_transactions"] += 1

        if row.transaction_type == "income":
            month_totals["income_total"] += amount
            continue

        month_totals["expense_total"] += amount

        if budget_category in ESSENTIAL_CATEGORIES:
            month_totals["essential_total"] += amount
            spending_bucket = "essential"
        elif budget_category in DISCRETIONARY_CATEGORIES:
            month_totals["discretionary_total"] += amount
            spending_bucket = "discretionary"
        else:
            month_totals["unknown_total"] += amount
            month_totals["unknown_transaction_count"] += 1
            spending_bucket = "unknown"

        category_key = (month, budget_category)
        category_totals = category_acc.setdefault(
            category_key,
            {
                "amount_total": 0.0,
                "transaction_count": 0.0,
                "spending_bucket": spending_bucket,
            },
        )
        category_totals["amount_total"] += amount
        category_totals["transaction_count"] += 1

    overview_rows: List[MonthlyBudgetOverview] = []
    for month in sorted(overview_acc.keys()):
        values = overview_acc[month]
        overview_rows.append(
            MonthlyBudgetOverview(
                month=month,
                income_total=round(values["income_total"], 2),
                expense_total=round(values["expense_total"], 2),
                essential_total=round(values["essential_total"], 2),
                discretionary_total=round(values["discretionary_total"], 2),
                unknown_total=round(values["unknown_total"], 2),
                unknown_transaction_count=int(values["unknown_transaction_count"]),
                total_transactions=int(values["total_transactions"]),
            )
        )

    category_rows: List[MonthlyCategoryAggregate] = []
    for (month, budget_category) in sorted(category_acc.keys()):
        values = category_acc[(month, budget_category)]
        expense_total = overview_acc.get(month, {}).get("expense_total", 0.0)
        expense_percentage = 0.0
        if expense_total > 0:
            expense_percentage = round((values["amount_total"] / expense_total) * 100.0, 2)

        category_rows.append(
            MonthlyCategoryAggregate(
                month=month,
                budget_category=budget_category,
                amount_total=round(values["amount_total"], 2),
                transaction_count=int(values["transaction_count"]),
                expense_percentage=expense_percentage,
                spending_bucket=str(values["spending_bucket"]),
            )
        )

    summary = MonthlyAggregationSummary(
        total_input_rows=len(classified_rows),
        rows_with_month=rows_with_month,
        rows_missing_month=rows_missing_month,
        months_count=len(overview_rows),
    )

    return overview_rows, category_rows, summary


def recalculate_monthly_budget(
    classified_rows: List[TransactionClassified],
    categorized_rows: List[BudgetCategorizedTransaction],
    normalized_rows: List[TransactionNormalized],
    months: Optional[Set[str]] = None,
) -> Tuple[List[MonthlyBudgetOverview], List[MonthlyCategoryAggregate], MonthlyAggregationSummary]:
    """Recalculate all or selected months after category corrections."""
    overview_rows, category_rows, summary = aggregate_monthly_budget(
        classified_rows,
        categorized_rows,
        normalized_rows,
    )
    if not months:
        return overview_rows, category_rows, summary

    month_filter = set(months)
    return (
        [row for row in overview_rows if row.month in month_filter],
        [row for row in category_rows if row.month in month_filter],
        summary,
    )


def load_transactions_normalized_csv(csv_path: str) -> List[TransactionNormalized]:
    rows: List[TransactionNormalized] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            amount_raw = (row.get("amount") or "").strip()
            balance_raw = (row.get("balance") or "").strip()
            rows.append(
                TransactionNormalized(
                    candidate_id=(row.get("candidate_id") or "").strip(),
                    source_pdf=(row.get("source_pdf") or "").strip(),
                    source_page=int((row.get("source_page") or "0").strip() or 0),
                    transaction_date=((row.get("transaction_date") or "").strip() or None),
                    description=((row.get("description") or "").strip() or None),
                    amount=float(amount_raw) if amount_raw else None,
                    balance=float(balance_raw) if balance_raw else None,
                    currency=((row.get("currency") or "").strip() or None),
                    direction=((row.get("direction") or "").strip() or None),
                    is_valid=(row.get("is_valid") or "").strip().lower() == "true",
                    normalization_status=(row.get("normalization_status") or "").strip(),
                    normalization_confidence=float((row.get("normalization_confidence") or "0").strip() or 0.0),
                    invalid_reasons=(row.get("invalid_reasons") or "").strip(),
                )
            )
    return rows


def save_monthly_budget_overview_csv(rows: List[MonthlyBudgetOverview], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "month",
                "income_total",
                "expense_total",
                "essential_total",
                "discretionary_total",
                "unknown_total",
                "unknown_transaction_count",
                "total_transactions",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def save_monthly_category_aggregates_csv(rows: List[MonthlyCategoryAggregate], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "month",
                "budget_category",
                "amount_total",
                "transaction_count",
                "expense_percentage",
                "spending_bucket",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


