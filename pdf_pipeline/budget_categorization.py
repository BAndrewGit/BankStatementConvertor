from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
import csv
import os
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from .transaction_classification import TransactionClassified
from .caen_enrichment import MerchantCaenEnriched


BUDGET_CATEGORIES = {
    "Food",
    "Housing",
    "Transport",
    "Health",
    "Entertainment",
    "Personal_Care",
    "Education",
    "Debt",
    "Fees",
    "Cash",
    "Transfers",
    "Other",
    "Unknown",
}

CAEN_MAP_REQUIRED_COLUMNS = ("COD_CAEN", "BUDGET_CATEGORY")
MERCHANT_BUDGET_ALIAS_REQUIRED_COLUMNS = ("ALIAS", "BUDGET_CATEGORY")

TRANSACTION_TYPE_DIRECT_CATEGORY = {
    "fee": "Fees",
    "cash_withdrawal": "Cash",
    "transfer": "Transfers",
    "debt_payment": "Debt",
}

DESCRIPTION_HARD_RULES: List[Tuple[str, List[str]]] = [
    ("Food", ["SUPERMARKET", "MEGA IMAGE", "LIDL", "KAUFLAND", "CARREFOUR", "AUCHAN", "PROFI", "PENNY"]),
    ("Transport", ["UBER", "BOLT", "METROU", "TAXI", "BENZINA", "MOTORINA", "STB", "OMV", "PETROM", "MOL"]),
    ("Housing", ["CHIRIE", "INTRETINERE", "UTILITATI", "ELECTRICA", "GAZ", "APA", "RENT"]),
    ("Health", ["FARMACIE", "MEDICAL", "SPITAL", "CLINICA", "DENT", "SANADOR", "REGINA MARIA"]),
    ("Education", ["CURS", "TRAINING", "SCOALA", "UNIVERSITATE", "EDUCATIE", "MANUAL"]),
    ("Personal_Care", ["COSMETIC", "FRIZERIE", "SALON", "SPA", "BARBER"]),
    ("Entertainment", ["NETFLIX", "CINEMA", "THEATER", "SPOTIFY", "HBO", "GAME", "CONCERT"]),
]

DEFAULT_CAEN_TO_BUDGET_CATEGORY = {
    "4711": "Food",
    "4721": "Food",
    "4932": "Transport",
    "4939": "Transport",
    "6820": "Housing",
    "8621": "Health",
    "8622": "Health",
    "8559": "Education",
    "9602": "Personal_Care",
    "5630": "Entertainment",
}

DEFAULT_MERCHANT_BUDGET_ALIASES = {
    "MEGA IMAGE": "Food",
    "LIDL": "Food",
    "UBER": "Transport",
    "BOLT": "Transport",
    "NETFLIX": "Entertainment",
}


@dataclass(frozen=True)
class BudgetCategorizedTransaction:
    candidate_id: str
    source_pdf: str
    source_page: int
    transaction_type: str
    description_clean: str
    merchant_raw_candidate: Optional[str]
    selected_caen_code: Optional[str]
    selected_caen_description: Optional[str]
    budget_category: str
    classification_reason: str
    classification_confidence: float
    categorization_status: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetCategorizationSummary:
    total_rows: int
    categorized_ok: int
    categorized_with_warnings: int
    via_type_rules: int
    via_description_rules: int
    via_alias_rules: int
    via_caen_rules: int
    via_fallback: int
    food: int
    housing: int
    transport: int
    health: int
    entertainment: int
    personal_care: int
    education: int
    debt: int
    fees: int
    cash: int
    transfers: int
    other: int
    unknown: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class Story10SchemaError(ValueError):
    """Raised when Story 10 mapping CSV files are invalid."""


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.upper().split())


def _normalize_caen_code(value: Optional[str]) -> str:
    clean = (value or "").strip()
    if not clean:
        return ""
    return clean.zfill(4)


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _ensure_story10_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS caen_to_budget_category (
            cod_caen TEXT PRIMARY KEY,
            budget_category TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS merchant_budget_aliases (
            alias_norm TEXT PRIMARY KEY,
            alias_original TEXT,
            budget_category TEXT,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_story10_caen_category
            ON caen_to_budget_category (budget_category);
        CREATE INDEX IF NOT EXISTS idx_story10_alias_category
            ON merchant_budget_aliases (budget_category);
        """
    )


def _validate_columns(csv_path: str, required_columns: Sequence[str], delimiter: str = ",") -> None:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        header = next(reader, None)

    if not header:
        raise Story10SchemaError(f"File {csv_path} has no header row.")

    missing = [column for column in required_columns if column not in header]
    if missing:
        raise Story10SchemaError(
            f"File {csv_path} is missing required columns: {', '.join(missing)}"
        )


def load_caen_to_budget_category_csv(db_path: str, csv_path: str) -> int:
    _validate_columns(csv_path, CAEN_MAP_REQUIRED_COLUMNS)

    rows: List[Tuple[str, str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            cod_caen = _normalize_caen_code(row.get("COD_CAEN"))
            budget_category = (row.get("BUDGET_CATEGORY") or "").strip()
            notes = (row.get("NOTES") or "").strip()
            if not cod_caen or budget_category not in BUDGET_CATEGORIES:
                continue
            rows.append((cod_caen, budget_category, notes))

    with closing(sqlite3.connect(db_path)) as connection:
        _ensure_story10_tables(connection)
        connection.execute("DELETE FROM caen_to_budget_category")
        if rows:
            connection.executemany(
                """
                INSERT INTO caen_to_budget_category (cod_caen, budget_category, notes)
                VALUES (?, ?, ?)
                """,
                rows,
            )
        connection.commit()

    return len(rows)


def load_merchant_budget_aliases_csv(db_path: str, csv_path: str) -> int:
    _validate_columns(csv_path, MERCHANT_BUDGET_ALIAS_REQUIRED_COLUMNS)

    rows: List[Tuple[str, str, str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            alias_original = (row.get("ALIAS") or "").strip()
            alias_norm = _normalize_text(alias_original)
            budget_category = (row.get("BUDGET_CATEGORY") or "").strip()
            notes = (row.get("NOTES") or "").strip()
            if not alias_norm or budget_category not in BUDGET_CATEGORIES:
                continue
            rows.append((alias_norm, alias_original, budget_category, notes))

    with closing(sqlite3.connect(db_path)) as connection:
        _ensure_story10_tables(connection)
        connection.execute("DELETE FROM merchant_budget_aliases")
        if rows:
            connection.executemany(
                """
                INSERT INTO merchant_budget_aliases (alias_norm, alias_original, budget_category, notes)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        connection.commit()

    return len(rows)


def _fetch_caen_category_map_from_db(db_path: str) -> Dict[str, str]:
    if not db_path or not os.path.exists(db_path):
        return {}

    with closing(sqlite3.connect(db_path)) as connection:
        _ensure_story10_tables(connection)
        cursor = connection.execute("SELECT cod_caen, budget_category FROM caen_to_budget_category")
        return {
            _normalize_caen_code(str(row[0])): str(row[1])
            for row in cursor.fetchall()
            if row[0] and row[1] in BUDGET_CATEGORIES
        }


def _fetch_merchant_alias_map_from_db(db_path: str) -> Dict[str, str]:
    if not db_path or not os.path.exists(db_path):
        return {}

    with closing(sqlite3.connect(db_path)) as connection:
        _ensure_story10_tables(connection)
        cursor = connection.execute("SELECT alias_norm, budget_category FROM merchant_budget_aliases")
        return {
            _normalize_text(str(row[0])): str(row[1])
            for row in cursor.fetchall()
            if row[0] and row[1] in BUDGET_CATEGORIES
        }


def _group_caen_rows_by_candidate(
    caen_rows: List[MerchantCaenEnriched],
) -> Dict[str, List[MerchantCaenEnriched]]:
    grouped: Dict[str, List[MerchantCaenEnriched]] = {}
    for row in caen_rows:
        grouped.setdefault(row.candidate_id, []).append(row)

    for candidate_id in grouped:
        grouped[candidate_id].sort(
            key=lambda item: (
                _normalize_caen_code(item.caen_code),
                item.caen_version or "",
            )
        )

    return grouped


def _category_from_direct_type(tx_type: str) -> Optional[Tuple[str, float, str]]:
    if tx_type in TRANSACTION_TYPE_DIRECT_CATEGORY:
        return TRANSACTION_TYPE_DIRECT_CATEGORY[tx_type], 0.99, f"rule:type:{tx_type}"
    return None


def _category_from_description(description_clean: str) -> Optional[Tuple[str, float, str]]:
    text = _normalize_text(description_clean)
    for category, keywords in DESCRIPTION_HARD_RULES:
        if _contains_any(text, keywords):
            return category, 0.95, f"rule:description:{category}"
    return None


def _category_from_alias(
    merchant_raw_candidate: Optional[str],
    alias_map: Dict[str, str],
) -> Optional[Tuple[str, float, str]]:
    merchant_norm = _normalize_text(merchant_raw_candidate)
    if not merchant_norm:
        return None

    if merchant_norm in alias_map:
        category = alias_map[merchant_norm]
        return category, 0.92, f"rule:merchant_alias:{merchant_norm}"

    return None


def _category_from_caen(
    candidate_caen_rows: List[MerchantCaenEnriched],
    caen_map: Dict[str, str],
) -> Optional[Tuple[str, float, str, Optional[str], Optional[str]]]:
    for row in candidate_caen_rows:
        code = _normalize_caen_code(row.caen_code)
        if not code:
            continue
        category = caen_map.get(code)
        if not category:
            continue
        return category, 0.85, f"rule:caen:{code}", code, row.caen_description

    return None


def _fallback_category(transaction_type: str) -> Tuple[str, float, str]:
    if transaction_type == "expense":
        return "Other", 0.6, "rule:fallback:expense"
    return "Unknown", 0.45, "rule:fallback:unknown"


def map_transactions_to_budget_categories(
    classified_rows: List[TransactionClassified],
    caen_rows: List[MerchantCaenEnriched],
    db_path: Optional[str] = None,
    caen_to_category_overrides: Optional[Dict[str, str]] = None,
    merchant_alias_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[List[BudgetCategorizedTransaction], BudgetCategorizationSummary]:
    """Story 10: map transactions to final budget categories with explicit priority rules."""
    caen_map = dict(DEFAULT_CAEN_TO_BUDGET_CATEGORY)
    caen_map.update(_fetch_caen_category_map_from_db(db_path or ""))
    if caen_to_category_overrides:
        caen_map.update(
            {
                _normalize_caen_code(code): category
                for code, category in caen_to_category_overrides.items()
                if category in BUDGET_CATEGORIES
            }
        )

    alias_map = dict(DEFAULT_MERCHANT_BUDGET_ALIASES)
    alias_map.update(_fetch_merchant_alias_map_from_db(db_path or ""))
    if merchant_alias_overrides:
        alias_map.update(
            {
                _normalize_text(alias): category
                for alias, category in merchant_alias_overrides.items()
                if category in BUDGET_CATEGORIES
            }
        )

    caen_by_candidate = _group_caen_rows_by_candidate(caen_rows)

    categorized_rows: List[BudgetCategorizedTransaction] = []

    for row in classified_rows:
        selected_caen_code: Optional[str] = None
        selected_caen_description: Optional[str] = None

        direct_type = _category_from_direct_type(row.transaction_type)
        if direct_type:
            category, confidence, reason = direct_type
            source = "type"
        else:
            direct_desc = _category_from_description(row.description_clean)
            if direct_desc:
                category, confidence, reason = direct_desc
                source = "description"
            else:
                alias_rule = _category_from_alias(row.merchant_raw_candidate, alias_map)
                if alias_rule:
                    category, confidence, reason = alias_rule
                    source = "alias"
                else:
                    caen_rule = _category_from_caen(caen_by_candidate.get(row.candidate_id, []), caen_map)
                    if caen_rule:
                        category, confidence, reason, selected_caen_code, selected_caen_description = caen_rule
                        source = "caen"
                    else:
                        category, confidence, reason = _fallback_category(row.transaction_type)
                        source = "fallback"

        status = "categorized_ok" if source != "fallback" else "categorized_with_warnings"

        categorized_rows.append(
            BudgetCategorizedTransaction(
                candidate_id=row.candidate_id,
                source_pdf=row.source_pdf,
                source_page=row.source_page,
                transaction_type=row.transaction_type,
                description_clean=row.description_clean,
                merchant_raw_candidate=row.merchant_raw_candidate,
                selected_caen_code=selected_caen_code,
                selected_caen_description=selected_caen_description,
                budget_category=category,
                classification_reason=reason,
                classification_confidence=round(max(0.0, min(1.0, confidence)), 2),
                categorization_status=status,
            )
        )

    summary = BudgetCategorizationSummary(
        total_rows=len(categorized_rows),
        categorized_ok=sum(1 for item in categorized_rows if item.categorization_status == "categorized_ok"),
        categorized_with_warnings=sum(
            1 for item in categorized_rows if item.categorization_status == "categorized_with_warnings"
        ),
        via_type_rules=sum(1 for item in categorized_rows if item.classification_reason.startswith("rule:type:")),
        via_description_rules=sum(
            1 for item in categorized_rows if item.classification_reason.startswith("rule:description:")
        ),
        via_alias_rules=sum(
            1 for item in categorized_rows if item.classification_reason.startswith("rule:merchant_alias:")
        ),
        via_caen_rules=sum(1 for item in categorized_rows if item.classification_reason.startswith("rule:caen:")),
        via_fallback=sum(1 for item in categorized_rows if item.classification_reason.startswith("rule:fallback:")),
        food=sum(1 for item in categorized_rows if item.budget_category == "Food"),
        housing=sum(1 for item in categorized_rows if item.budget_category == "Housing"),
        transport=sum(1 for item in categorized_rows if item.budget_category == "Transport"),
        health=sum(1 for item in categorized_rows if item.budget_category == "Health"),
        entertainment=sum(1 for item in categorized_rows if item.budget_category == "Entertainment"),
        personal_care=sum(1 for item in categorized_rows if item.budget_category == "Personal_Care"),
        education=sum(1 for item in categorized_rows if item.budget_category == "Education"),
        debt=sum(1 for item in categorized_rows if item.budget_category == "Debt"),
        fees=sum(1 for item in categorized_rows if item.budget_category == "Fees"),
        cash=sum(1 for item in categorized_rows if item.budget_category == "Cash"),
        transfers=sum(1 for item in categorized_rows if item.budget_category == "Transfers"),
        other=sum(1 for item in categorized_rows if item.budget_category == "Other"),
        unknown=sum(1 for item in categorized_rows if item.budget_category == "Unknown"),
    )

    return categorized_rows, summary


def load_transactions_classified_csv(csv_path: str) -> List[TransactionClassified]:
    rows: List[TransactionClassified] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            amount_raw = (row.get("amount") or "").strip()
            rows.append(
                TransactionClassified(
                    candidate_id=(row.get("candidate_id") or "").strip(),
                    source_pdf=(row.get("source_pdf") or "").strip(),
                    source_page=int((row.get("source_page") or "0").strip() or 0),
                    description_clean=(row.get("description_clean") or "").strip(),
                    merchant_raw_candidate=((row.get("merchant_raw_candidate") or "").strip() or None),
                    direction=((row.get("direction") or "").strip() or None),
                    amount=float(amount_raw) if amount_raw else None,
                    transaction_type=(row.get("transaction_type") or "unknown").strip(),
                    classification_confidence=float((row.get("classification_confidence") or "0").strip() or 0.0),
                    classification_reason=(row.get("classification_reason") or "").strip(),
                    classification_status=(row.get("classification_status") or "").strip(),
                )
            )
    return rows


def load_caen_enriched_csv(csv_path: str) -> List[MerchantCaenEnriched]:
    rows: List[MerchantCaenEnriched] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                MerchantCaenEnriched(
                    candidate_id=(row.get("candidate_id") or "").strip(),
                    source_pdf=(row.get("source_pdf") or "").strip(),
                    source_page=int((row.get("source_page") or "0").strip() or 0),
                    merchant_raw_candidate=((row.get("merchant_raw_candidate") or "").strip() or None),
                    matched_denumire=((row.get("matched_denumire") or "").strip() or None),
                    matched_cui=((row.get("matched_cui") or "").strip() or None),
                    matched_cod_inmatriculare=((row.get("matched_cod_inmatriculare") or "").strip() or None),
                    match_method=(row.get("match_method") or "").strip(),
                    match_score=float((row.get("match_score") or "0").strip() or 0.0),
                    match_reason=(row.get("match_reason") or "").strip(),
                    resolution_status=(row.get("resolution_status") or "").strip(),
                    caen_code=((row.get("caen_code") or "").strip() or None),
                    caen_version=((row.get("caen_version") or "").strip() or None),
                    caen_description=((row.get("caen_description") or "").strip() or None),
                    caen_enrichment_status=(row.get("caen_enrichment_status") or "").strip(),
                )
            )
    return rows


def load_budget_categorized_csv(csv_path: str) -> List[BudgetCategorizedTransaction]:
    rows: List[BudgetCategorizedTransaction] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                BudgetCategorizedTransaction(
                    candidate_id=(row.get("candidate_id") or "").strip(),
                    source_pdf=(row.get("source_pdf") or "").strip(),
                    source_page=int((row.get("source_page") or "0").strip() or 0),
                    transaction_type=(row.get("transaction_type") or "unknown").strip(),
                    description_clean=(row.get("description_clean") or "").strip(),
                    merchant_raw_candidate=((row.get("merchant_raw_candidate") or "").strip() or None),
                    selected_caen_code=((row.get("selected_caen_code") or "").strip() or None),
                    selected_caen_description=((row.get("selected_caen_description") or "").strip() or None),
                    budget_category=(row.get("budget_category") or "Unknown").strip(),
                    classification_reason=(row.get("classification_reason") or "").strip(),
                    classification_confidence=float((row.get("classification_confidence") or "0").strip() or 0.0),
                    categorization_status=(row.get("categorization_status") or "").strip(),
                )
            )
    return rows


def save_budget_categorized_csv(rows: List[BudgetCategorizedTransaction], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_id",
                "source_pdf",
                "source_page",
                "transaction_type",
                "description_clean",
                "merchant_raw_candidate",
                "selected_caen_code",
                "selected_caen_description",
                "budget_category",
                "classification_reason",
                "classification_confidence",
                "categorization_status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


