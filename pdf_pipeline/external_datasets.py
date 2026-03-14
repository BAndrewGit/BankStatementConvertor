from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from contextlib import closing
import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CSV_DELIMITER = "^"

OD_FIRME_REQUIRED_COLUMNS = ("DENUMIRE", "CUI", "COD_INMATRICULARE")
OD_CAEN_AUTORIZAT_REQUIRED_COLUMNS = ("COD_INMATRICULARE", "COD_CAEN_AUTORIZAT", "VER_CAEN_AUTORIZAT")
N_CAEN_REQUIRED_COLUMNS = ("CLASA", "DENUMIRE", "VERSIUNE_CAEN")


@dataclass(frozen=True)
class DatasetPreparationStats:
    dataset_name: str
    rows_read: int
    rows_inserted: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalDatasetsPreparationSummary:
    output_db_path: str
    od_firme: DatasetPreparationStats
    od_caen_autorizat: DatasetPreparationStats
    n_caen: DatasetPreparationStats

    def to_dict(self) -> Dict[str, object]:
        return {
            "output_db_path": self.output_db_path,
            "od_firme": self.od_firme.to_dict(),
            "od_caen_autorizat": self.od_caen_autorizat.to_dict(),
            "n_caen": self.n_caen.to_dict(),
        }


class DatasetSchemaError(ValueError):
    """Raised when a dataset misses required columns."""


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.upper().split())


def _normalize_registry_code(value: Optional[str]) -> str:
    return _normalize_text(value)


def _normalize_caen_code(value: Optional[str]) -> str:
    clean = (value or "").strip()
    if not clean:
        return ""
    return clean.zfill(4)


def _iter_csv_rows(csv_path: str) -> Iterable[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=CSV_DELIMITER)
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items()}


def validate_required_columns(csv_path: str, required_columns: Sequence[str]) -> None:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMITER)
        header = next(reader, None)

    if not header:
        raise DatasetSchemaError(f"Dataset {csv_path} has no header row.")

    missing = [column for column in required_columns if column not in header]
    if missing:
        raise DatasetSchemaError(
            f"Dataset {csv_path} is missing required columns: {', '.join(missing)}"
        )


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS od_firme (
            denumire TEXT,
            denumire_norm TEXT,
            cui TEXT,
            cod_inmatriculare TEXT
        );

        CREATE TABLE IF NOT EXISTS od_caen_autorizat (
            cod_inmatriculare TEXT,
            cod_caen_autorizat TEXT,
            ver_caen_autorizat TEXT
        );

        CREATE TABLE IF NOT EXISTS n_caen (
            clasa TEXT,
            denumire TEXT,
            versiune_caen TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_od_firme_denumire_norm ON od_firme (denumire_norm);
        CREATE INDEX IF NOT EXISTS idx_od_firme_cod_inmatriculare ON od_firme (cod_inmatriculare);
        CREATE INDEX IF NOT EXISTS idx_od_caen_cod_inmatriculare ON od_caen_autorizat (cod_inmatriculare);
        CREATE INDEX IF NOT EXISTS idx_od_caen_cod_caen ON od_caen_autorizat (cod_caen_autorizat);
        CREATE INDEX IF NOT EXISTS idx_n_caen_clasa ON n_caen (clasa);
        """
    )


def _insert_rows_in_batches(
    connection: sqlite3.Connection,
    insert_sql: str,
    rows: Iterable[Tuple[str, ...]],
    batch_size: int = 5000,
) -> int:
    inserted = 0
    batch: List[Tuple[str, ...]] = []

    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            connection.executemany(insert_sql, batch)
            inserted += len(batch)
            batch.clear()

    if batch:
        connection.executemany(insert_sql, batch)
        inserted += len(batch)

    return inserted


def _prepare_od_firme(connection: sqlite3.Connection, csv_path: str) -> DatasetPreparationStats:
    rows_read = 0

    def row_generator() -> Iterable[Tuple[str, ...]]:
        nonlocal rows_read
        for row in _iter_csv_rows(csv_path):
            rows_read += 1
            cod_inmatriculare = _normalize_registry_code(row.get("COD_INMATRICULARE"))
            if not cod_inmatriculare:
                continue

            denumire = (row.get("DENUMIRE") or "").strip()
            cui = (row.get("CUI") or "").strip()
            denumire_norm = _normalize_text(denumire)
            yield (denumire, denumire_norm, cui, cod_inmatriculare)

    connection.execute("DELETE FROM od_firme")
    rows_inserted = _insert_rows_in_batches(
        connection,
        """
        INSERT INTO od_firme (denumire, denumire_norm, cui, cod_inmatriculare)
        VALUES (?, ?, ?, ?)
        """,
        row_generator(),
    )
    return DatasetPreparationStats("od_firme", rows_read, rows_inserted)


def _prepare_od_caen_autorizat(connection: sqlite3.Connection, csv_path: str) -> DatasetPreparationStats:
    rows_read = 0

    def row_generator() -> Iterable[Tuple[str, ...]]:
        nonlocal rows_read
        for row in _iter_csv_rows(csv_path):
            rows_read += 1
            cod_inmatriculare = _normalize_registry_code(row.get("COD_INMATRICULARE"))
            cod_caen = _normalize_caen_code(row.get("COD_CAEN_AUTORIZAT"))
            if not cod_inmatriculare or not cod_caen:
                continue

            ver = (row.get("VER_CAEN_AUTORIZAT") or "").strip()
            yield (cod_inmatriculare, cod_caen, ver)

    connection.execute("DELETE FROM od_caen_autorizat")
    rows_inserted = _insert_rows_in_batches(
        connection,
        """
        INSERT INTO od_caen_autorizat (cod_inmatriculare, cod_caen_autorizat, ver_caen_autorizat)
        VALUES (?, ?, ?)
        """,
        row_generator(),
    )
    return DatasetPreparationStats("od_caen_autorizat", rows_read, rows_inserted)


def _prepare_n_caen(connection: sqlite3.Connection, csv_path: str) -> DatasetPreparationStats:
    rows_read = 0

    def row_generator() -> Iterable[Tuple[str, ...]]:
        nonlocal rows_read
        for row in _iter_csv_rows(csv_path):
            rows_read += 1
            clasa = _normalize_caen_code(row.get("CLASA"))
            if not clasa:
                continue

            denumire = (row.get("DENUMIRE") or "").strip()
            versiune = (row.get("VERSIUNE_CAEN") or "").strip()
            yield (clasa, denumire, versiune)

    connection.execute("DELETE FROM n_caen")
    rows_inserted = _insert_rows_in_batches(
        connection,
        """
        INSERT INTO n_caen (clasa, denumire, versiune_caen)
        VALUES (?, ?, ?)
        """,
        row_generator(),
    )
    return DatasetPreparationStats("n_caen", rows_read, rows_inserted)


def prepare_external_datasets(
    od_firme_csv_path: str,
    od_caen_autorizat_csv_path: str,
    n_caen_csv_path: str,
    output_db_path: str,
) -> ExternalDatasetsPreparationSummary:
    """Story 7: ingest and prepare external reference datasets in local SQLite tables."""
    validate_required_columns(od_firme_csv_path, OD_FIRME_REQUIRED_COLUMNS)
    validate_required_columns(od_caen_autorizat_csv_path, OD_CAEN_AUTORIZAT_REQUIRED_COLUMNS)
    validate_required_columns(n_caen_csv_path, N_CAEN_REQUIRED_COLUMNS)

    os.makedirs(os.path.dirname(output_db_path) or ".", exist_ok=True)

    with closing(sqlite3.connect(output_db_path)) as connection:
        _create_tables(connection)
        od_firme_stats = _prepare_od_firme(connection, od_firme_csv_path)
        od_caen_stats = _prepare_od_caen_autorizat(connection, od_caen_autorizat_csv_path)
        n_caen_stats = _prepare_n_caen(connection, n_caen_csv_path)
        connection.commit()

    return ExternalDatasetsPreparationSummary(
        output_db_path=output_db_path,
        od_firme=od_firme_stats,
        od_caen_autorizat=od_caen_stats,
        n_caen=n_caen_stats,
    )


def lookup_od_firme_by_cod_inmatriculare(db_path: str, cod_inmatriculare: str) -> List[Dict[str, str]]:
    normalized_code = _normalize_registry_code(cod_inmatriculare)
    if not normalized_code:
        return []

    with closing(sqlite3.connect(db_path)) as connection:
        cursor = connection.execute(
            """
            SELECT denumire, cui, cod_inmatriculare
            FROM od_firme
            WHERE cod_inmatriculare = ?
            """,
            (normalized_code,),
        )
        return [
            {
                "denumire": row[0],
                "cui": row[1],
                "cod_inmatriculare": row[2],
            }
            for row in cursor.fetchall()
        ]


def lookup_od_firme_by_name(db_path: str, denumire: str, limit: int = 20) -> List[Dict[str, str]]:
    normalized_name = _normalize_text(denumire)
    if not normalized_name:
        return []

    with closing(sqlite3.connect(db_path)) as connection:
        cursor = connection.execute(
            """
            SELECT denumire, cui, cod_inmatriculare
            FROM od_firme
            WHERE denumire_norm LIKE ?
            LIMIT ?
            """,
            (f"%{normalized_name}%", max(1, limit)),
        )
        return [
            {
                "denumire": row[0],
                "cui": row[1],
                "cod_inmatriculare": row[2],
            }
            for row in cursor.fetchall()
        ]


def get_caen_description(db_path: str, cod_caen: str) -> Optional[str]:
    normalized_caen = _normalize_caen_code(cod_caen)
    if not normalized_caen:
        return None

    with closing(sqlite3.connect(db_path)) as connection:
        cursor = connection.execute(
            """
            SELECT denumire
            FROM n_caen
            WHERE clasa = ?
            LIMIT 1
            """,
            (normalized_caen,),
        )
        row = cursor.fetchone()

    return row[0] if row else None


def get_company_caen_descriptions(db_path: str, cod_inmatriculare: str) -> List[Dict[str, str]]:
    normalized_code = _normalize_registry_code(cod_inmatriculare)
    if not normalized_code:
        return []

    with closing(sqlite3.connect(db_path)) as connection:
        cursor = connection.execute(
            """
            SELECT oc.cod_caen_autorizat, nc.denumire
            FROM od_caen_autorizat oc
            LEFT JOIN n_caen nc ON nc.clasa = oc.cod_caen_autorizat
            WHERE oc.cod_inmatriculare = ?
            ORDER BY oc.cod_caen_autorizat ASC
            """,
            (normalized_code,),
        )
        return [
            {
                "cod_caen_autorizat": row[0],
                "caen_description": row[1] or "",
            }
            for row in cursor.fetchall()
        ]


