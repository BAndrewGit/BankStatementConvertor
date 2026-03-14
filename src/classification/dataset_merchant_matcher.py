from __future__ import annotations

import csv
import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

try:
    from rapidfuzz import fuzz as _rapidfuzz
except Exception:  # pragma: no cover - exercised only when dependency is missing
    _rapidfuzz = None


LEGAL_SUFFIXES = {
    "SRL",
    "SA",
    "SRL-D",
    "PFA",
    "PF",
    "IF",
    "II",
    "SNC",
    "SCS",
    "SCA",
}

STOPWORDS = {
    "by",
    "ess",
    "romania",
    "roumanie",
    "import",
    "export",
    "plata",
    "pos",
    "epos",
    "non",
    "bt",
    "card",
    "mastercard",
    "ron",
    "rrn",
    "tid",
    "ref",
    "valoare",
    "tranzactie",
    "comision",
    "sold",
    "final",
    "zi",
    "transfer",
    "intern",
    "canal",
    "electronic",
    "p2p",
    "ro",
    "rom",
    "bucuresti",
    "unknown",
    "merchant",
    "company",
    "concept",
    "consulting",
    "services",
    "service",
    "invest",
    "international",
    "group",
    "holding",
}

BRAND_PREFIXES = {
    "AUCH",
    "CARR",
    "MEGA",
    "PRAN",
    "EMAG",
    "ALTE",
    "METR",
    "TACO",
    "MOVI",
    "FROO",
    "DIGI",
    "PPC",
    "ORAN",
    "YOXO",
    "JKC",
    "FARM",
}

EPSILON = 1e-9


@dataclass(frozen=True)
class CompanyRecord:
    display_name: str
    normalized_name: str
    normalized_core: str
    tokens: Set[str]
    token_prefixes: Set[str]


class DatasetMerchantMatcher:
    def __init__(
        self,
        od_firme_csv_path: Optional[str] = None,
        max_candidates: Optional[int] = None,
        min_score_brand: Optional[float] = None,
        min_score_generic: Optional[float] = None,
    ) -> None:
        self._od_firme_csv_path = od_firme_csv_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "..",
            "DatasetsCAEN",
            "od_firme.csv",
        )
        self._max_candidates = int(os.getenv("ONRC_MAX_CANDIDATES", str(max_candidates or 150)))
        self._min_score_brand = float(os.getenv("ONRC_MIN_SCORE_BRAND", str(min_score_brand or 60.0)))
        self._min_score_generic = float(os.getenv("ONRC_MIN_SCORE_GENERIC", str(min_score_generic or 60.0)))
        self._records: List[CompanyRecord] = []
        self._token_index: Dict[str, List[int]] = {}
        self._token_prefix_index: Dict[str, List[int]] = {}
        self._exact_name_index: Dict[str, str] = {}
        self._exact_core_index: Dict[str, str] = {}
        self._match_cache: Dict[str, Tuple[Optional[str], float]] = {}
        self._loaded = False

    def _token_set_ratio(self, left: str, right: str) -> float:
        if _rapidfuzz is not None:
            return float(_rapidfuzz.token_set_ratio(left, right))

        # Fallback for environments where rapidfuzz is not installed.
        left_tokens = sorted(set(left.split()))
        right_tokens = sorted(set(right.split()))
        left_norm = " ".join(left_tokens)
        right_norm = " ".join(right_tokens)
        return SequenceMatcher(a=left_norm, b=right_norm).ratio() * 100.0

    def _foreign_token_count(self, rec: CompanyRecord, query_tokens: Set[str], query_prefixes: Set[str]) -> int:
        foreign = 0
        for token in rec.tokens:
            if token in query_tokens:
                continue
            if len(token) >= 4 and token[:4] in query_prefixes:
                continue
            foreign += 1
        return foreign

    def _specificity_adjustment(
        self,
        rec: CompanyRecord,
        query_tokens: Set[str],
        query_prefixes: Set[str],
        query_brand_prefixes: Set[str],
    ) -> float:
        foreign_tokens = self._foreign_token_count(rec, query_tokens, query_prefixes)
        adjustment = -2.2 * foreign_tokens

        if query_brand_prefixes:
            brand_hits = sum(1 for prefix in rec.token_prefixes if prefix in query_brand_prefixes)
            if brand_hits:
                # Prefer candidates centered on the requested brand when multiple brand-like names exist.
                adjustment += min(6.0, 8.0 * (brand_hits / max(1, len(rec.tokens))))

        return adjustment

    def find_best_match(self, merchant_text: str) -> Tuple[Optional[str], float]:
        if not merchant_text:
            return None, 0.0

        self._ensure_loaded()
        if not self._records:
            return None, 0.0

        normalized_query = self._normalize_text(merchant_text)
        cached = self._match_cache.get(normalized_query)
        if cached is not None:
            return cached

        query_tokens = self._tokenize(normalized_query)
        important_tokens = self._extract_important_tokens(normalized_query)
        if not query_tokens:
            return None, 0.0
        important_text = " ".join(important_tokens)
        query_prefixes = {token[:4] for token in query_tokens if len(token) >= 4}
        query_brand_prefixes = {prefix for prefix in query_prefixes if prefix in BRAND_PREFIXES}
        has_brand_query_token = bool(query_brand_prefixes)

        # exact match fast-path
        exact_name = self._exact_name_index.get(normalized_query)
        if exact_name:
            result = (exact_name, 100.0)
            self._match_cache[normalized_query] = result
            return result

        normalized_core_query = " ".join(sorted(query_tokens))
        exact_core = self._exact_core_index.get(normalized_core_query)
        if exact_core:
            result = (exact_core, 100.0)
            self._match_cache[normalized_query] = result
            return result

        candidate_indices = self._candidate_indices(important_tokens)
        if not candidate_indices:
            return None, 0.0

        best_idx = -1
        best_score = 0.0

        for idx in candidate_indices:
            rec = self._records[idx]
            score = max(
                self._token_set_ratio(normalized_query, rec.normalized_name),
                self._token_set_ratio(important_text, rec.normalized_name) if important_text else 0.0,
            )

            bonus = 0.0
            overlap_count = 0
            has_brand_prefix_hit = False
            for token in important_tokens:
                if token in rec.tokens:
                    bonus += 8.0
                    overlap_count += 1
                elif len(token) >= 4 and token[:4] in rec.token_prefixes:
                    bonus += 6.0
                    overlap_count += 1
                    if token[:4] in BRAND_PREFIXES:
                        has_brand_prefix_hit = True

            if overlap_count == 0:
                continue

            score += bonus
            score += self._specificity_adjustment(
                rec=rec,
                query_tokens=query_tokens,
                query_prefixes=query_prefixes,
                query_brand_prefixes=query_brand_prefixes,
            )
            score = max(0.0, min(100.0, score))
            if has_brand_prefix_hit:
                score = max(score, 82.0)

            if score > best_score + EPSILON:
                best_score = score
                best_idx = idx
                continue

            if abs(score - best_score) <= 0.5 and best_idx >= 0:
                best_rec = self._records[best_idx]
                current_foreign = self._foreign_token_count(rec, query_tokens, query_prefixes)
                best_foreign = self._foreign_token_count(best_rec, query_tokens, query_prefixes)
                if current_foreign < best_foreign or (
                    current_foreign == best_foreign and rec.display_name < best_rec.display_name
                ):
                    best_score = score
                    best_idx = idx

        threshold = self._min_score_brand if has_brand_query_token else self._min_score_generic
        if best_idx < 0 or best_score < threshold:
            self._match_cache[normalized_query] = (None, 0.0)
            return None, 0.0

        result = (self._records[best_idx].display_name, round(best_score, 4))
        self._match_cache[normalized_query] = result
        return result

    def _candidate_indices(self, important_tokens: List[str]) -> List[int]:
        candidate_scores: Dict[int, int] = {}
        for token in important_tokens:
            for idx in self._token_index.get(token, []):
                candidate_scores[idx] = candidate_scores.get(idx, 0) + 3
            if len(token) >= 4:
                for idx in self._token_prefix_index.get(token[:4], []):
                    candidate_scores[idx] = candidate_scores.get(idx, 0) + 1

        if not candidate_scores:
            return []

        ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
        return [idx for idx, _ in ranked[: self._max_candidates]]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not os.path.exists(self._od_firme_csv_path):
            self._loaded = True
            return

        with open(self._od_firme_csv_path, encoding="utf-8-sig", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="^")
            for row in reader:
                company_name = self._get_value(row, "DENUMIRE").strip()
                if not company_name:
                    continue

                normalized = self._normalize_text(company_name)
                tokens = self._tokenize(normalized)
                if not tokens:
                    continue

                normalized_core = " ".join(sorted(tokens))

                record = CompanyRecord(
                    display_name=company_name,
                    normalized_name=normalized,
                    normalized_core=normalized_core,
                    tokens=tokens,
                    token_prefixes={token[:4] for token in tokens if len(token) >= 4},
                )
                index = len(self._records)
                self._records.append(record)
                self._exact_name_index.setdefault(normalized, company_name)
                self._exact_core_index.setdefault(normalized_core, company_name)

                for token in tokens:
                    self._token_index.setdefault(token, []).append(index)
                    if len(token) >= 4:
                        self._token_prefix_index.setdefault(token[:4], []).append(index)


        self._loaded = True

    def _get_value(self, row: Dict[str, str], key: str) -> str:
        if key in row:
            return row.get(key) or ""
        bom_key = f"\ufeff{key}"
        if bom_key in row:
            return row.get(bom_key) or ""
        for existing_key, value in row.items():
            if existing_key.lstrip("\ufeff").strip().upper() == key.upper():
                return value or ""
        return ""

    def _normalize_text(self, text: str) -> str:
        without_diacritics = unicodedata.normalize("NFKD", text)
        without_diacritics = "".join(ch for ch in without_diacritics if not unicodedata.combining(ch))
        upper = without_diacritics.upper()
        upper = re.sub(r"[^A-Z0-9\s&.-]", " ", upper)
        return re.sub(r"\s+", " ", upper).strip()

    def _tokenize(self, normalized_text: str) -> Set[str]:
        tokens: Set[str] = set()
        for token in normalized_text.split():
            compact = token.strip(".-")
            if len(compact) < 3:
                continue
            if compact.isdigit():
                continue
            if compact in LEGAL_SUFFIXES:
                continue
            if compact.lower() in STOPWORDS:
                continue
            tokens.add(compact)
        return tokens

    def _extract_important_tokens(self, normalized_text: str) -> List[str]:
        tokens_in_order = []
        seen: Set[str] = set()
        for token in normalized_text.split():
            compact = token.strip(".-")
            if len(compact) < 3:
                continue
            if compact.isdigit():
                continue
            if compact in LEGAL_SUFFIXES:
                continue
            if compact.lower() in STOPWORDS:
                continue
            if compact not in seen:
                tokens_in_order.append(compact)
                seen.add(compact)

        def priority(tok: str) -> Tuple[int, int]:
            prefix = tok[:4]
            brand_priority = 0 if prefix in BRAND_PREFIXES else 1
            return (brand_priority, -len(tok))

        ordered = sorted(tokens_in_order, key=priority)
        return ordered[:3]



