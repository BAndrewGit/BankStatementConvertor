from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
import csv
import os
import sqlite3
from typing import Dict, List, Optional, Set, Tuple

from .merchant_resolution import MerchantResolution


@dataclass(frozen=True)
class MerchantCaenEnriched:
    candidate_id: str
    source_pdf: str
    source_page: int
    merchant_raw_candidate: Optional[str]
    matched_denumire: Optional[str]
    matched_cui: Optional[str]
    matched_cod_inmatriculare: Optional[str]
    match_method: str
    match_score: float
    match_reason: str
    resolution_status: str
    caen_code: Optional[str]
    caen_version: Optional[str]
    caen_description: Optional[str]
    caen_enrichment_status: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CaenEnrichmentSummary:
    total_input_rows: int
    total_output_rows: int
    resolved_companies: int
    unresolved_companies: int
    rows_with_caen: int
    rows_without_caen: int
    unique_caen_codes: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def _fetch_company_caen_rows(
    connection: sqlite3.Connection,
    cod_inmatriculare: str,
) -> List[Tuple[str, str, str, str]]:
    cursor = connection.execute(
        """
        SELECT
            oc.cod_caen_autorizat,
            oc.ver_caen_autorizat,
            nc.denumire,
            nc.versiune_caen
        FROM od_caen_autorizat oc
        LEFT JOIN n_caen nc ON nc.clasa = oc.cod_caen_autorizat
        WHERE oc.cod_inmatriculare = ?
        ORDER BY oc.cod_caen_autorizat ASC, oc.ver_caen_autorizat ASC
        """,
        (cod_inmatriculare,),
    )
    return [
        (
            str(row[0] or ""),
            str(row[1] or ""),
            str(row[2] or ""),
            str(row[3] or ""),
        )
        for row in cursor.fetchall()
    ]


def enrich_resolved_merchants_with_caen(
    merchant_resolution_rows: List[MerchantResolution],
    db_path: str,
) -> Tuple[List[MerchantCaenEnriched], CaenEnrichmentSummary]:
    """Story 9: attach CAEN code(s), version, and description for resolved firms."""
    output_rows: List[MerchantCaenEnriched] = []
    unique_caen_codes: Set[str] = set()

    resolved_companies = 0
    unresolved_companies = 0

    with closing(sqlite3.connect(db_path)) as connection:
        for row in merchant_resolution_rows:
            cod_inmatriculare = (row.matched_cod_inmatriculare or "").strip()

            if not cod_inmatriculare:
                unresolved_companies += 1
                output_rows.append(
                    MerchantCaenEnriched(
                        candidate_id=row.candidate_id,
                        source_pdf=row.source_pdf,
                        source_page=row.source_page,
                        merchant_raw_candidate=row.merchant_raw_candidate,
                        matched_denumire=row.matched_denumire,
                        matched_cui=row.matched_cui,
                        matched_cod_inmatriculare=row.matched_cod_inmatriculare,
                        match_method=row.match_method,
                        match_score=row.match_score,
                        match_reason=row.match_reason,
                        resolution_status=row.resolution_status,
                        caen_code=None,
                        caen_version=None,
                        caen_description=None,
                        caen_enrichment_status="no_company_resolved",
                    )
                )
                continue

            resolved_companies += 1
            caen_rows = _fetch_company_caen_rows(connection, cod_inmatriculare)

            if not caen_rows:
                output_rows.append(
                    MerchantCaenEnriched(
                        candidate_id=row.candidate_id,
                        source_pdf=row.source_pdf,
                        source_page=row.source_page,
                        merchant_raw_candidate=row.merchant_raw_candidate,
                        matched_denumire=row.matched_denumire,
                        matched_cui=row.matched_cui,
                        matched_cod_inmatriculare=row.matched_cod_inmatriculare,
                        match_method=row.match_method,
                        match_score=row.match_score,
                        match_reason=row.match_reason,
                        resolution_status=row.resolution_status,
                        caen_code=None,
                        caen_version=None,
                        caen_description=None,
                        caen_enrichment_status="no_caen_for_company",
                    )
                )
                continue

            seen_for_candidate: Set[Tuple[str, str]] = set()
            for caen_code, ver_auth, caen_description, ver_n_caen in caen_rows:
                caen_code_clean = caen_code.strip() or None
                if not caen_code_clean:
                    continue

                # Deduplicate repeated rows from source datasets.
                dedupe_key = (caen_code_clean, ver_auth.strip() or ver_n_caen.strip())
                if dedupe_key in seen_for_candidate:
                    continue
                seen_for_candidate.add(dedupe_key)

                unique_caen_codes.add(caen_code_clean)
                caen_version = (ver_auth.strip() or ver_n_caen.strip() or None)
                caen_description_clean = caen_description.strip() or None

                output_rows.append(
                    MerchantCaenEnriched(
                        candidate_id=row.candidate_id,
                        source_pdf=row.source_pdf,
                        source_page=row.source_page,
                        merchant_raw_candidate=row.merchant_raw_candidate,
                        matched_denumire=row.matched_denumire,
                        matched_cui=row.matched_cui,
                        matched_cod_inmatriculare=row.matched_cod_inmatriculare,
                        match_method=row.match_method,
                        match_score=row.match_score,
                        match_reason=row.match_reason,
                        resolution_status=row.resolution_status,
                        caen_code=caen_code_clean,
                        caen_version=caen_version,
                        caen_description=caen_description_clean,
                        caen_enrichment_status="caen_attached",
                    )
                )

            if not seen_for_candidate:
                output_rows.append(
                    MerchantCaenEnriched(
                        candidate_id=row.candidate_id,
                        source_pdf=row.source_pdf,
                        source_page=row.source_page,
                        merchant_raw_candidate=row.merchant_raw_candidate,
                        matched_denumire=row.matched_denumire,
                        matched_cui=row.matched_cui,
                        matched_cod_inmatriculare=row.matched_cod_inmatriculare,
                        match_method=row.match_method,
                        match_score=row.match_score,
                        match_reason=row.match_reason,
                        resolution_status=row.resolution_status,
                        caen_code=None,
                        caen_version=None,
                        caen_description=None,
                        caen_enrichment_status="no_caen_for_company",
                    )
                )

    summary = CaenEnrichmentSummary(
        total_input_rows=len(merchant_resolution_rows),
        total_output_rows=len(output_rows),
        resolved_companies=resolved_companies,
        unresolved_companies=unresolved_companies,
        rows_with_caen=sum(1 for item in output_rows if item.caen_enrichment_status == "caen_attached"),
        rows_without_caen=sum(1 for item in output_rows if item.caen_enrichment_status != "caen_attached"),
        unique_caen_codes=len(unique_caen_codes),
    )

    return output_rows, summary


def save_merchants_caen_enriched_csv(rows: List[MerchantCaenEnriched], output_csv_path: str) -> None:
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
                "match_method",
                "match_score",
                "match_reason",
                "resolution_status",
                "caen_code",
                "caen_version",
                "caen_description",
                "caen_enrichment_status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def load_merchant_resolution_csv(csv_path: str) -> List[MerchantResolution]:
    rows: List[MerchantResolution] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                MerchantResolution(
                    candidate_id=(row.get("candidate_id") or "").strip(),
                    source_pdf=(row.get("source_pdf") or "").strip(),
                    source_page=int((row.get("source_page") or "0").strip() or 0),
                    merchant_raw_candidate=((row.get("merchant_raw_candidate") or "").strip() or None),
                    matched_denumire=((row.get("matched_denumire") or "").strip() or None),
                    matched_cui=((row.get("matched_cui") or "").strip() or None),
                    matched_cod_inmatriculare=((row.get("matched_cod_inmatriculare") or "").strip() or None),
                    matched_caen_codes=(row.get("matched_caen_codes") or "").strip(),
                    match_method=(row.get("match_method") or "").strip(),
                    match_score=float((row.get("match_score") or "0").strip() or 0.0),
                    match_reason=(row.get("match_reason") or "").strip(),
                    resolution_status=(row.get("resolution_status") or "").strip(),
                )
            )
    return rows


