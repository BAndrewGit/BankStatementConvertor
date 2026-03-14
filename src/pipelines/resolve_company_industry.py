from __future__ import annotations

import csv
from dataclasses import dataclass
import os
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.domain.company_lookup import CaenIndustryEntry, CompanyIndustryResult
from src.infrastructure.termene_client import TermeneClient


INDUSTRY_OVERRIDES = {
    "5610": "Restaurante",
    "6201": "Activitati software",
    "4711": "Comert cu amanuntul",
}


@dataclass(frozen=True)
class _FirmRecord:
    row: Dict[str, str]
    den_norm: str
    den_tokens: Set[str]
    den_prefixes: Set[str]


@dataclass(frozen=True)
class _OnrcContext:
    firms: List[_FirmRecord]
    token_index: Dict[str, List[int]]
    prefix_index: Dict[str, List[int]]
    caen_rows_by_cod: Dict[str, List[Dict[str, str]]]
    nomenclature: Dict[str, str]


_ONRC_CONTEXT_CACHE: Dict[Tuple[str, str, str, Optional[float], Optional[float], Optional[float]], _OnrcContext] = {}


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    no_diacritics = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    upper = no_diacritics.upper()
    upper = re.sub(r"[^A-Z0-9\s&.-]", " ", upper)
    return re.sub(r"\s+", " ", upper).strip()


def _tokens(text: str) -> Set[str]:
    out: Set[str] = set()
    for token in _normalize(text).split():
        compact = token.strip(".-")
        if len(compact) < 3 or compact.isdigit():
            continue
        out.add(compact)
    return out


def _read_csv(path: str, delimiter: str = "^") -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


def _field(row: Dict[str, str], key: str) -> str:
    if key in row:
        return (row.get(key) or "").strip()
    bom_key = f"\ufeff{key}"
    if bom_key in row:
        return (row.get(bom_key) or "").strip()
    for existing_key, value in row.items():
        if existing_key.lstrip("\ufeff").strip().upper() == key.upper():
            return (value or "").strip()
    return ""


def _best_onrc_match(company_name: str, context: _OnrcContext) -> Optional[Dict[str, str]]:
    query_norm = _normalize(company_name)
    query_tokens = _tokens(company_name)
    if not query_tokens:
        return None

    candidate_indices: Set[int] = set()
    query_prefixes = {token[:4] for token in query_tokens if len(token) >= 4}
    for token in query_tokens:
        for idx in context.token_index.get(token, []):
            candidate_indices.add(idx)
    for prefix in query_prefixes:
        for idx in context.prefix_index.get(prefix, []):
            candidate_indices.add(idx)
    if not candidate_indices:
        return None

    best_row = None
    best_score = 0.0

    for idx in candidate_indices:
        record = context.firms[idx]
        den_tokens = record.den_tokens

        overlap = len(query_tokens & den_tokens)
        if overlap == 0:
            if query_prefixes & record.den_prefixes:
                overlap = 1

        if overlap == 0:
            continue

        coverage_query = overlap / max(1, len(query_tokens))
        coverage_den = overlap / max(1, len(den_tokens))
        score = 0.65 * coverage_query + 0.35 * coverage_den

        if query_norm in record.den_norm:
            score = max(score, 0.97)

        if score > best_score:
            best_score = score
            best_row = record.row

    if best_row is None or best_score < 0.5:
        return None

    return best_row


def _context_cache_key(od_firme_path: str, od_caen_path: str, n_caen_path: str) -> Tuple[str, str, str, Optional[float], Optional[float], Optional[float]]:
    firme_abs = os.path.abspath(od_firme_path)
    caen_abs = os.path.abspath(od_caen_path)
    nomenclature_abs = os.path.abspath(n_caen_path)
    firme_mtime = os.path.getmtime(firme_abs) if os.path.exists(firme_abs) else None
    caen_mtime = os.path.getmtime(caen_abs) if os.path.exists(caen_abs) else None
    nomenclature_mtime = os.path.getmtime(nomenclature_abs) if os.path.exists(nomenclature_abs) else None
    return firme_abs, caen_abs, nomenclature_abs, firme_mtime, caen_mtime, nomenclature_mtime


def _load_context(od_firme_path: str, od_caen_path: str, n_caen_path: str) -> _OnrcContext:
    key = _context_cache_key(od_firme_path, od_caen_path, n_caen_path)
    cached = _ONRC_CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached

    firms_rows = _read_csv(od_firme_path)
    firms: List[_FirmRecord] = []
    token_index: Dict[str, List[int]] = {}
    prefix_index: Dict[str, List[int]] = {}
    for row in firms_rows:
        den = _field(row, "DENUMIRE")
        if not den:
            continue

        den_norm = _normalize(den)
        den_tokens = _tokens(den)
        if not den_tokens:
            continue

        den_prefixes = {token[:4] for token in den_tokens if len(token) >= 4}
        record = _FirmRecord(row=row, den_norm=den_norm, den_tokens=den_tokens, den_prefixes=den_prefixes)
        idx = len(firms)
        firms.append(record)
        for token in den_tokens:
            token_index.setdefault(token, []).append(idx)
        for prefix in den_prefixes:
            prefix_index.setdefault(prefix, []).append(idx)

    caen_rows = _read_csv(od_caen_path)
    caen_rows_by_cod: Dict[str, List[Dict[str, str]]] = {}
    for row in caen_rows:
        cod_inmatriculare = _field(row, "COD_INMATRICULARE")
        if cod_inmatriculare:
            caen_rows_by_cod.setdefault(cod_inmatriculare, []).append(row)

    context = _OnrcContext(
        firms=firms,
        token_index=token_index,
        prefix_index=prefix_index,
        caen_rows_by_cod=caen_rows_by_cod,
        nomenclature=_caen_map(_read_csv(n_caen_path)),
    )
    _ONRC_CONTEXT_CACHE[key] = context
    return context


def _caen_map(nomenclature_rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in nomenclature_rows:
        raw = _field(row, "CLASA")
        if not raw:
            continue
        code = raw.zfill(4)
        den = _field(row, "DENUMIRE")
        if den and code not in mapping:
            mapping[code] = den
    return mapping


def _build_entries(
    cod_inmatriculare: str,
    caen_rows: Sequence[Dict[str, str]],
    nomenclature: Dict[str, str],
) -> List[CaenIndustryEntry]:
    entries: List[CaenIndustryEntry] = []
    seen: Set[str] = set()

    for row in caen_rows:
        if _field(row, "COD_INMATRICULARE") != cod_inmatriculare:
            continue

        code = _field(row, "COD_CAEN_AUTORIZAT").zfill(4)
        if not code or code in seen:
            continue
        seen.add(code)

        description = nomenclature.get(code)
        industry = INDUSTRY_OVERRIDES.get(code, description or "Necunoscut")

        entries.append(
            CaenIndustryEntry(
                caen_code=code,
                caen_description=description,
                industry=industry,
            )
        )

    return sorted(entries, key=lambda item: item.caen_code)


def resolve_company_industry(
    company_name: str,
    od_firme_csv_path: Optional[str] = None,
    od_caen_autorizat_csv_path: Optional[str] = None,
    n_caen_csv_path: Optional[str] = None,
    termene_client: Optional[TermeneClient] = None,
) -> CompanyIndustryResult:
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "DatasetsCAEN")

    od_firme_path = od_firme_csv_path or os.path.join(datasets_dir, "od_firme.csv")
    od_caen_path = od_caen_autorizat_csv_path or os.path.join(datasets_dir, "od_caen_autorizat.csv")
    n_caen_path = n_caen_csv_path or os.path.join(datasets_dir, "n_caen.csv")

    context = _load_context(od_firme_path, od_caen_path, n_caen_path)
    onrc_match = _best_onrc_match(company_name, context)

    if onrc_match:
        cod_inmatriculare = _field(onrc_match, "COD_INMATRICULARE")
        entries = _build_entries(
            cod_inmatriculare=cod_inmatriculare,
            caen_rows=context.caen_rows_by_cod.get(cod_inmatriculare, []),
            nomenclature=context.nomenclature,
        )
        return CompanyIndustryResult(
            input_name=company_name,
            source="onrc",
            company_name=_field(onrc_match, "DENUMIRE") or None,
            cui=_field(onrc_match, "CUI") or None,
            cod_inmatriculare=cod_inmatriculare or None,
            entries=entries,
        )

    client = termene_client or TermeneClient()
    termene_payload = client.search_company(company_name)
    if termene_payload:
        return CompanyIndustryResult(
            input_name=company_name,
            source="termene",
            company_name=(termene_payload.get("name") or termene_payload.get("denumire") or "").strip() or None,
            cui=str(termene_payload.get("cui") or "").strip() or None,
            cod_inmatriculare=(termene_payload.get("cod_inmatriculare") or termene_payload.get("nr_reg_com") or "").strip() or None,
            entries=[],
        )

    return CompanyIndustryResult(input_name=company_name, source="not_found")

