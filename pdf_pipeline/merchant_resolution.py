from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
import csv
from difflib import SequenceMatcher
import os
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from .description_normalization import TransactionDescriptionNormalized


ALIAS_REQUIRED_COLUMNS = ("ALIAS", "COD_INMATRICULARE")


@dataclass(frozen=True)
class MerchantResolution:
    candidate_id: str
    source_pdf: str
    source_page: int
    merchant_raw_candidate: Optional[str]
    matched_denumire: Optional[str]
    matched_cui: Optional[str]
    matched_cod_inmatriculare: Optional[str]
    matched_caen_codes: str
    match_method: str
    match_score: float
    match_reason: str
    resolution_status: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MerchantResolutionSummary:
    total_rows: int
    accepted: int
    review: int
    rejected: int
    alias_exact: int
    exact_name: int
    fuzzy_name: int
    no_match: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class AliasSchemaError(ValueError):
    """Raised when alias CSV misses required columns."""


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.upper().split())


def _ensure_alias_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS merchant_aliases (
            alias_norm TEXT PRIMARY KEY,
            alias_original TEXT,
            target_cod_inmatriculare TEXT,
            target_cui TEXT,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_merchant_aliases_target_code
            ON merchant_aliases (target_cod_inmatriculare);
        """
    )


def validate_alias_columns(alias_csv_path: str, required_columns: Sequence[str] = ALIAS_REQUIRED_COLUMNS) -> None:
    with open(alias_csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)

    if not header:
        raise AliasSchemaError(f"Alias file {alias_csv_path} has no header row.")

    missing = [column for column in required_columns if column not in header]
    if missing:
        raise AliasSchemaError(
            f"Alias file {alias_csv_path} is missing required columns: {', '.join(missing)}"
        )


def load_merchant_aliases_csv(db_path: str, alias_csv_path: str) -> int:
    """Load manual merchant aliases into Story 7 SQLite DB."""
    validate_alias_columns(alias_csv_path)

    with closing(sqlite3.connect(db_path)) as connection:
        _ensure_alias_table(connection)
        connection.execute("DELETE FROM merchant_aliases")

        rows: List[Tuple[str, str, str, str, str]] = []
        with open(alias_csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                alias_original = (row.get("ALIAS") or "").strip()
                alias_norm = _normalize_text(alias_original)
                target_code = _normalize_text(row.get("COD_INMATRICULARE"))
                target_cui = (row.get("CUI") or "").strip()
                notes = (row.get("NOTES") or "").strip()
                if not alias_norm or not target_code:
                    continue
                rows.append((alias_norm, alias_original, target_code, target_cui, notes))

        if rows:
            connection.executemany(
                """
                INSERT INTO merchant_aliases (
                    alias_norm,
                    alias_original,
                    target_cod_inmatriculare,
                    target_cui,
                    notes
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        connection.commit()

    return len(rows)


def _fetch_caen_codes(connection: sqlite3.Connection, cod_inmatriculare: str) -> str:
    cursor = connection.execute(
        """
        SELECT DISTINCT cod_caen_autorizat
        FROM od_caen_autorizat
        WHERE cod_inmatriculare = ?
        ORDER BY cod_caen_autorizat ASC
        """,
        (cod_inmatriculare,),
    )
    codes = [str(row[0]) for row in cursor.fetchall() if row[0]]
    return ",".join(codes)


def _resolve_by_alias(
    connection: sqlite3.Connection,
    merchant_norm: str,
) -> Optional[Tuple[str, str, str, str, float, str]]:
    cursor = connection.execute(
        """
        SELECT ma.target_cod_inmatriculare, ofi.denumire, ofi.cui, ma.notes
        FROM merchant_aliases ma
        JOIN od_firme ofi ON ofi.cod_inmatriculare = ma.target_cod_inmatriculare
        WHERE ma.alias_norm = ?
        LIMIT 1
        """,
        (merchant_norm,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    code, denumire, cui, notes = row
    reason = "alias_exact"
    if notes:
        reason = f"alias_exact:{notes}"
    return denumire, cui, code, _fetch_caen_codes(connection, code), 1.0, reason


def _resolve_by_exact_name(
    connection: sqlite3.Connection,
    merchant_norm: str,
) -> Optional[Tuple[str, str, str, str, float, str]]:
    cursor = connection.execute(
        """
        SELECT denumire, cui, cod_inmatriculare
        FROM od_firme
        WHERE denumire_norm = ?
        LIMIT 1
        """,
        (merchant_norm,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    denumire, cui, code = row
    return denumire, cui, code, _fetch_caen_codes(connection, code), 0.99, "exact_name"


def _fetch_fuzzy_candidates(
    connection: sqlite3.Connection,
    merchant_norm: str,
    max_candidates: int,
) -> List[Tuple[str, str, str]]:
    first_token = merchant_norm.split()[0] if merchant_norm else ""
    if first_token:
        cursor = connection.execute(
            """
            SELECT denumire, cui, cod_inmatriculare
            FROM od_firme
            WHERE denumire_norm LIKE ?
            LIMIT ?
            """,
            (f"%{first_token}%", max_candidates),
        )
        rows = cursor.fetchall()
        if rows:
            return [(str(r[0]), str(r[1] or ""), str(r[2])) for r in rows]

    cursor = connection.execute(
        """
        SELECT denumire, cui, cod_inmatriculare
        FROM od_firme
        LIMIT ?
        """,
        (max_candidates,),
    )
    rows = cursor.fetchall()
    return [(str(r[0]), str(r[1] or ""), str(r[2])) for r in rows]


def _resolve_by_fuzzy_name(
    connection: sqlite3.Connection,
    merchant_norm: str,
    max_candidates: int,
) -> Optional[Tuple[str, str, str, str, float, str]]:
    candidates = _fetch_fuzzy_candidates(connection, merchant_norm, max_candidates=max_candidates)

    best: Optional[Tuple[str, str, str, float]] = None
    for denumire, cui, code in candidates:
        score = SequenceMatcher(None, merchant_norm, _normalize_text(denumire)).ratio()
        if best is None or score > best[3]:
            best = (denumire, cui, code, score)

    if not best:
        return None

    denumire, cui, code, score = best
    return denumire, cui, code, _fetch_caen_codes(connection, code), round(score, 2), "fuzzy_name"


def _decide_status(match_score: float, accept_threshold: float, review_threshold: float) -> str:
    if match_score >= accept_threshold:
        return "accepted"
    if match_score >= review_threshold:
        return "review"
    return "rejected"


def resolve_merchants_to_onrc(
    description_rows: List[TransactionDescriptionNormalized],
    db_path: str,
    accept_threshold: float = 0.9,
    review_threshold: float = 0.75,
    fuzzy_max_candidates: int = 500,
) -> Tuple[List[MerchantResolution], MerchantResolutionSummary]:
    """Story 8: resolve normalized merchants to ONRC firms with exact-first strategy."""
    if review_threshold > accept_threshold:
        raise ValueError("review_threshold cannot be greater than accept_threshold")

    results: List[MerchantResolution] = []

    with closing(sqlite3.connect(db_path)) as connection:
        _ensure_alias_table(connection)

        for row in description_rows:
            merchant_raw = row.merchant_raw_candidate
            merchant_norm = _normalize_text(merchant_raw)

            if not merchant_norm:
                results.append(
                    MerchantResolution(
                        candidate_id=row.candidate_id,
                        source_pdf=row.source_pdf,
                        source_page=row.source_page,
                        merchant_raw_candidate=merchant_raw,
                        matched_denumire=None,
                        matched_cui=None,
                        matched_cod_inmatriculare=None,
                        matched_caen_codes="",
                        match_method="no_match",
                        match_score=0.0,
                        match_reason="missing_merchant_candidate",
                        resolution_status="review",
                    )
                )
                continue

            alias_hit = _resolve_by_alias(connection, merchant_norm)
            if alias_hit:
                denumire, cui, code, caen_codes, score, reason = alias_hit
                method = "alias_exact"
            else:
                exact_hit = _resolve_by_exact_name(connection, merchant_norm)
                if exact_hit:
                    denumire, cui, code, caen_codes, score, reason = exact_hit
                    method = "exact_name"
                else:
                    fuzzy_hit = _resolve_by_fuzzy_name(
                        connection,
                        merchant_norm,
                        max_candidates=max(10, fuzzy_max_candidates),
                    )
                    if fuzzy_hit:
                        denumire, cui, code, caen_codes, score, reason = fuzzy_hit
                        method = "fuzzy_name"
                    else:
                        denumire, cui, code, caen_codes, score, reason = None, None, None, "", 0.0, "no_match"
                        method = "no_match"

            status = _decide_status(score, accept_threshold, review_threshold)
            if method == "no_match":
                status = "review"

            results.append(
                MerchantResolution(
                    candidate_id=row.candidate_id,
                    source_pdf=row.source_pdf,
                    source_page=row.source_page,
                    merchant_raw_candidate=merchant_raw,
                    matched_denumire=denumire,
                    matched_cui=cui,
                    matched_cod_inmatriculare=code,
                    matched_caen_codes=caen_codes,
                    match_method=method,
                    match_score=round(score, 2),
                    match_reason=reason,
                    resolution_status=status,
                )
            )

    summary = MerchantResolutionSummary(
        total_rows=len(results),
        accepted=sum(1 for item in results if item.resolution_status == "accepted"),
        review=sum(1 for item in results if item.resolution_status == "review"),
        rejected=sum(1 for item in results if item.resolution_status == "rejected"),
        alias_exact=sum(1 for item in results if item.match_method == "alias_exact"),
        exact_name=sum(1 for item in results if item.match_method == "exact_name"),
        fuzzy_name=sum(1 for item in results if item.match_method == "fuzzy_name"),
        no_match=sum(1 for item in results if item.match_method == "no_match"),
    )

    return results, summary


def save_merchant_resolution_csv(rows: List[MerchantResolution], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_id",
                "source_pdf",
                "source_page",
                "merchant_raw_candidate",
                "matched_denumire",
                "matched_cui",
                "matched_cod_inmatriculare",
                "matched_caen_codes",
                "match_method",
                "match_score",
                "match_reason",
                "resolution_status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def load_description_normalized_csv(csv_path: str) -> List[TransactionDescriptionNormalized]:
    rows: List[TransactionDescriptionNormalized] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                TransactionDescriptionNormalized(
                    candidate_id=(row.get("candidate_id") or "").strip(),
                    source_pdf=(row.get("source_pdf") or "").strip(),
                    source_page=int((row.get("source_page") or "0").strip() or 0),
                    description_clean=(row.get("description_clean") or "").strip(),
                    merchant_raw_candidate=((row.get("merchant_raw_candidate") or "").strip() or None),
                    description_normalization_confidence=float(
                        (row.get("description_normalization_confidence") or "0").strip() or 0.0
                    ),
                    description_status=(row.get("description_status") or "").strip(),
                    description_warnings=(row.get("description_warnings") or "").strip(),
                )
            )
    return rows



