from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher
import os
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from src.classification.dataset_merchant_matcher import DatasetMerchantMatcher
from src.domain.models import Transaction
from src.infrastructure.cache import CacheRepository
from src.infrastructure.termene_client import TermeneClient


class MerchantNormalizer:
    def __init__(
        self,
        aliases_yaml_path: Optional[str] = None,
        od_firme_csv_path: Optional[str] = None,
        fuzzy_threshold: float = 0.88,
        cache_repo: Optional[CacheRepository] = None,
        termene_client: Optional[TermeneClient] = None,
        enable_termene_fallback: Optional[bool] = None,
    ) -> None:
        self._aliases_yaml_path = aliases_yaml_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "merchant_aliases.yaml",
        )
        self._fuzzy_threshold = fuzzy_threshold
        self._cache_repo = cache_repo
        self._dataset_matcher = DatasetMerchantMatcher(od_firme_csv_path=od_firme_csv_path)
        if enable_termene_fallback is None:
            enable_termene_fallback = os.getenv("TERMENE_ENABLE_FALLBACK", "0") == "1"
        self._enable_termene_fallback = enable_termene_fallback
        self._termene_client = termene_client or (TermeneClient() if self._enable_termene_fallback else None)
        self._dataset_min_confidence = float(os.getenv("ONRC_ACCEPT_MIN_CONFIDENCE", "60"))

        self._exact_aliases: Dict[str, str] = {}
        self._regex_aliases: List[Tuple[re.Pattern[str], str]] = []
        self._all_aliases: List[Tuple[str, str]] = []
        self._all_aliases_by_length: List[Tuple[str, str]] = []
        self._alias_token_index: Dict[str, List[Tuple[str, str]]] = {}
        self._canonicalized_text_cache: Dict[str, str] = {}
        self._contains_alias_cache: Dict[str, Optional[str]] = {}
        self._fuzzy_cache: Dict[str, Optional[Tuple[str, float]]] = {}
        self._normalized_result_cache: Dict[str, Dict[str, object]] = {}
        self._load_aliases()

    def _dataset_name_to_brand(self, dataset_name: str, normalized_raw: str) -> str:
        normalized_dataset = self._canonicalize(dataset_name)
        candidate_text = f"{normalized_dataset} {normalized_raw}"

        # Prefer canonical names configured in merchant_aliases.yaml.
        alias_match = self._contains_alias_match(candidate_text)
        if alias_match:
            return alias_match

        exact_match = self._exact_aliases.get(normalized_dataset)
        if exact_match:
            return exact_match

        # Generic fallback when aliases/bootstrap do not define a brand: strip legal suffixes/noise.
        legal_noise = {
            "SA",
            "SRL",
            "SRL.",
            "S.A",
            "S.R.L",
            "ROMANIA",
            "THE",
            "COMPANY",
        }
        tokens = [token for token in normalized_dataset.split() if token not in legal_noise]
        if tokens:
            candidate_tokens = tokens[: min(2, len(tokens))]
            candidate = " ".join(candidate_tokens)
            if candidate:
                return " ".join(word.capitalize() for word in candidate.split())

        return dataset_name


    def _prefer_alias_for_raw_pattern(self, normalized_raw: str) -> bool:
        # Processor-based merchants are better handled by alias map than ONRC fuzzy company names.
        return "PAYU*" in normalized_raw

    def _is_dataset_match_trustworthy(self, dataset_name: str, normalized_raw: str, score: float) -> bool:
        _ = dataset_name
        _ = normalized_raw
        return score >= self._dataset_min_confidence

    def _dataset_result(self, normalized_raw: str) -> Optional[Dict[str, object]]:
        dataset_match, dataset_score = self._dataset_matcher.find_best_match(normalized_raw)
        if not dataset_match:
            return None
        if not self._is_dataset_match_trustworthy(dataset_match, normalized_raw, dataset_score):
            return None

        canonical_dataset_name = self._dataset_name_to_brand(dataset_match, normalized_raw)
        confidence = round(min(0.99, max(0.7, dataset_score / 100.0)), 2)
        return {
            "merchant_canonical": canonical_dataset_name,
            "normalization_source": "dataset",
            "merchant_match_method": "dataset_token_set_ratio",
            "mapping_method": "dataset_match",
            "confidence": confidence,
        }

    def normalize(self, txn: Transaction) -> Transaction:
        if not txn.merchant_raw:
            return replace(txn, merchant_canonical=None, mapping_method=None)

        normalized_raw = self._canonicalize(txn.merchant_raw)
        cached_result = self._normalized_result_cache.get(normalized_raw)
        if cached_result is not None:
            return replace(txn, **cached_result)

        if not self._prefer_alias_for_raw_pattern(normalized_raw):
            dataset_first = self._dataset_result(normalized_raw)
            if dataset_first:
                self._normalized_result_cache[normalized_raw] = dataset_first
                return replace(txn, **dataset_first)

        # Keep deterministic exact alias mapping for known processors/brands.
        exact = self._exact_aliases.get(normalized_raw)
        if exact:
            result = {
                "merchant_canonical": exact,
                "normalization_source": "alias_exact",
                "merchant_match_method": "alias_exact",
                "mapping_method": "exact_alias",
                "confidence": 1.0,
            }
            self._normalized_result_cache[normalized_raw] = result
            return replace(txn, **result)

        if self._prefer_alias_for_raw_pattern(normalized_raw):
            contains_match = self._contains_alias_match(normalized_raw)
            if contains_match:
                result = {
                    "merchant_canonical": contains_match,
                    "normalization_source": "alias_contains",
                    "merchant_match_method": "alias_contains",
                    "mapping_method": "regex_alias",
                    "confidence": 0.98,
                }
                self._normalized_result_cache[normalized_raw] = result
                return replace(txn, **result)

            for pattern, canonical in self._regex_aliases:
                if pattern.search(normalized_raw):
                    result = {
                        "merchant_canonical": canonical,
                        "normalization_source": "alias_regex",
                        "merchant_match_method": "alias_regex",
                        "mapping_method": "regex_alias",
                        "confidence": 0.96,
                    }
                    self._normalized_result_cache[normalized_raw] = result
                    return replace(txn, **result)

        dataset_result = self._dataset_result(normalized_raw)
        if dataset_result:
            self._normalized_result_cache[normalized_raw] = dataset_result
            return replace(txn, **dataset_result)


        contains_match = self._contains_alias_match(normalized_raw)
        if contains_match:
            result = {
                "merchant_canonical": contains_match,
                "normalization_source": "alias_contains",
                "merchant_match_method": "alias_contains",
                "mapping_method": "regex_alias",
                "confidence": 0.98,
            }
            self._normalized_result_cache[normalized_raw] = result
            return replace(txn, **result)

        for pattern, canonical in self._regex_aliases:
            if pattern.search(normalized_raw):
                result = {
                    "merchant_canonical": canonical,
                    "normalization_source": "alias_regex",
                    "merchant_match_method": "alias_regex",
                    "mapping_method": "regex_alias",
                    "confidence": 0.96,
                }
                self._normalized_result_cache[normalized_raw] = result
                return replace(txn, **result)

        if self._cache_repo:
            cached_canonical = self._cache_repo.get("merchant_normalization", normalized_raw)
            if cached_canonical:
                result = {
                    "merchant_canonical": str(cached_canonical),
                    "normalization_source": "alias_fuzzy",
                    "merchant_match_method": "alias_fuzzy_cached",
                    "mapping_method": "fuzzy_alias",
                    "confidence": round(self._fuzzy_threshold, 2),
                }
                self._normalized_result_cache[normalized_raw] = result
                return replace(txn, **result)

        fuzzy = self._fuzzy_match(normalized_raw)
        if fuzzy:
            canonical, score = fuzzy
            if self._cache_repo:
                self._cache_repo.set("merchant_normalization", normalized_raw, canonical)
            result = {
                "merchant_canonical": canonical,
                "normalization_source": "alias_fuzzy",
                "merchant_match_method": "alias_fuzzy",
                "mapping_method": "fuzzy_alias",
                "confidence": round(score, 2),
            }
            self._normalized_result_cache[normalized_raw] = result
            return replace(txn, **result)

        if self._enable_termene_fallback:
            cached_termene = self._cache_repo.get("termene_company", normalized_raw) if self._cache_repo else None
            if cached_termene:
                result = {
                    "merchant_canonical": str(cached_termene),
                    "normalization_source": "termene",
                    "merchant_match_method": "termene_api_cached",
                    "mapping_method": "termene_api",
                    "confidence": 0.85,
                }
                self._normalized_result_cache[normalized_raw] = result
                return replace(txn, **result)

            termene_match = self._termene_client.search_company(txn.merchant_raw) if self._termene_client else None
            if termene_match:
                termene_name = (termene_match.get("name") or termene_match.get("denumire") or "").strip()
                if termene_name:
                    if self._cache_repo:
                        self._cache_repo.set("termene_company", normalized_raw, termene_name)
                    result = {
                        "merchant_canonical": termene_name,
                        "normalization_source": "termene",
                        "merchant_match_method": "termene_api",
                        "mapping_method": "termene_api",
                        "confidence": 0.85,
                    }
                    self._normalized_result_cache[normalized_raw] = result
                    return replace(txn, **result)

        result = {
            "merchant_canonical": txn.merchant_raw,
            "normalization_source": "unmapped",
            "merchant_match_method": "unmapped",
            "mapping_method": "unmapped",
            "confidence": 0.5,
        }
        self._normalized_result_cache[normalized_raw] = result
        return replace(txn, **result)

    def _load_aliases(self) -> None:
        with open(self._aliases_yaml_path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        for key in sorted(payload.keys()):
            entry = payload.get(key) or {}
            canonical = str(entry.get("canonical", "")).strip()
            if not canonical:
                continue

            aliases = entry.get("aliases", []) or []
            for alias in aliases:
                normalized_alias = self._canonicalize(str(alias))
                if not normalized_alias:
                    continue
                self._exact_aliases[normalized_alias] = canonical
                self._all_aliases.append((normalized_alias, canonical))

                for token in normalized_alias.split():
                    if len(token) < 4:
                        continue
                    self._alias_token_index.setdefault(token, []).append((normalized_alias, canonical))

            for regex_alias in entry.get("regex_aliases", []) or []:
                pattern = re.compile(str(regex_alias), re.IGNORECASE)
                self._regex_aliases.append((pattern, canonical))

        self._all_aliases.sort(key=lambda item: (item[1], item[0]))
        self._all_aliases_by_length = sorted(
            self._all_aliases,
            key=lambda item: (-len(item[0]), item[0], item[1]),
        )

    def _contains_alias_match(self, normalized_raw: str) -> Optional[str]:
        cached = self._contains_alias_cache.get(normalized_raw)
        if normalized_raw in self._contains_alias_cache:
            return cached

        candidate_aliases: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for token in normalized_raw.split():
            if len(token) < 4:
                continue
            for candidate in self._alias_token_index.get(token, []):
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidate_aliases.append(candidate)

        candidate_aliases.sort(key=lambda item: (-len(item[0]), item[0], item[1]))
        scan_list = candidate_aliases or self._all_aliases_by_length
        compact_raw: Optional[str] = None
        for alias, canonical in scan_list:
            if len(alias) < 4:
                continue
            if alias in normalized_raw:
                self._contains_alias_cache[normalized_raw] = canonical
                return canonical

            # Processor aliases often include separators like '*', '.', '/'.
            # Match on compact alnum form as fallback when separators vary.
            if any(ch in alias for ch in "*./-"):
                if compact_raw is None:
                    compact_raw = re.sub(r"[^A-Z0-9]", "", normalized_raw)
                compact_alias = re.sub(r"[^A-Z0-9]", "", alias)
                if compact_alias and compact_alias in compact_raw:
                    self._contains_alias_cache[normalized_raw] = canonical
                    return canonical

        self._contains_alias_cache[normalized_raw] = None
        return None

    def _fuzzy_match(self, normalized_raw: str) -> Optional[Tuple[str, float]]:
        if normalized_raw in self._fuzzy_cache:
            return self._fuzzy_cache[normalized_raw]

        best_canonical = None
        best_score = 0.0

        candidate_aliases = self._all_aliases
        rough_candidates: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for token in normalized_raw.split():
            if len(token) < 4:
                continue
            for item in self._alias_token_index.get(token, []):
                if item in seen:
                    continue
                seen.add(item)
                rough_candidates.append(item)
        if len(rough_candidates) >= 8:
            candidate_aliases = rough_candidates

        for alias, canonical in candidate_aliases:
            score = SequenceMatcher(a=normalized_raw, b=alias).ratio()
            if score > best_score or (score == best_score and canonical < (best_canonical or canonical)):
                best_score = score
                best_canonical = canonical

        if best_canonical is None or best_score < self._fuzzy_threshold:
            self._fuzzy_cache[normalized_raw] = None
            return None

        result = (best_canonical, best_score)
        self._fuzzy_cache[normalized_raw] = result
        return result

    def _canonicalize(self, text: str) -> str:
        cached = self._canonicalized_text_cache.get(text)
        if cached is not None:
            return cached

        without_diacritics = unicodedata.normalize("NFKD", text)
        without_diacritics = "".join(ch for ch in without_diacritics if not unicodedata.combining(ch))
        upper = without_diacritics.upper()
        upper = re.sub(r"[^A-Z0-9*./&\s-]", " ", upper)
        normalized = re.sub(r"\s+", " ", upper).strip()
        self._canonicalized_text_cache[text] = normalized
        return normalized


def normalize_merchant(
    transactions: Sequence[Transaction],
    aliases_yaml_path: Optional[str] = None,
    od_firme_csv_path: Optional[str] = None,
    cache_repo: Optional[CacheRepository] = None,
    termene_client: Optional[TermeneClient] = None,
    enable_termene_fallback: Optional[bool] = None,
) -> List[Transaction]:
    normalizer = MerchantNormalizer(
        aliases_yaml_path=aliases_yaml_path,
        od_firme_csv_path=od_firme_csv_path,
        cache_repo=cache_repo,
        termene_client=termene_client,
        enable_termene_fallback=enable_termene_fallback,
    )
    return [normalizer.normalize(txn) for txn in transactions]

