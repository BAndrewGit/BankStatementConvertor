from __future__ import annotations

from enum import Enum


class SourceSection(str, Enum):
    BOOKED_TRANSACTIONS = "booked_transactions"
    BLOCKED_AMOUNTS = "blocked_amounts"


class Channel(str, Enum):
    POS = "POS"
    EPOS = "EPOS"
    ATM = "ATM"
    TRANSFER = "TRANSFER"
    FEE = "FEE"
    BLOCKED = "BLOCKED"
    OTHER = "OTHER"


class TransactionType(str, Enum):
    CARD_PURCHASE = "card_purchase"
    CASH_WITHDRAWAL = "cash_withdrawal"
    INTERNAL_TRANSFER = "internal_transfer"
    EXTERNAL_TRANSFER = "external_transfer"
    SALARY_INCOME = "salary_income"
    BANK_FEE = "bank_fee"
    SUBSCRIPTION = "subscription"
    UTILITY_PAYMENT = "utility_payment"
    BLOCKED_AMOUNT = "blocked_amount"
    UNKNOWN = "unknown"


class CategoryArea(str, Enum):
    FOOD = "Food"
    HOUSING = "Housing"
    TRANSPORT = "Transport"
    ENTERTAINMENT = "Entertainment"
    HEALTH = "Health"
    PERSONAL_CARE = "Personal_Care"
    CHILD_EDUCATION = "Child_Education"
    OTHER = "Other"
    UNKNOWN = "Unknown"

