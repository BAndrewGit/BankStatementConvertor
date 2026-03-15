from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.domain.enums import TransactionType
from src.domain.models import Transaction
from src.memory.entity_memory import EntityMemoryRepository

from .fuzzy_matcher import find_best_category_match
from .text_normalization import normalize_text


IBAN_PATTERN = re.compile(r"\bRO\d{2}[A-Z0-9]{10,30}\b", re.IGNORECASE)


class TransferBootstrapClassifier:
    def __init__(
        self,
        bootstrap_dictionary_path: Optional[str] = None,
        profile_id: Optional[str] = None,
        memory_repo: Optional[EntityMemoryRepository] = None,
    ) -> None:
        self._bootstrap_dictionary_path = bootstrap_dictionary_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "bootstrap_dictionary.json",
        )
        self._profile_id = profile_id or "default"
        self._memory_repo = memory_repo or EntityMemoryRepository()
        self._memory: Dict[str, Dict[str, object]] = self._memory_repo.load(self._profile_id)
        self._memory_dirty = False
        self._rules = self._load_rules()

    def classify_many(self, transactions: Sequence[Transaction]) -> List[Transaction]:
        classified = [self.classify(txn) for txn in transactions]
        classified = [self._backfill_from_memory(txn) for txn in classified]
        if self._memory_dirty:
            self._memory_repo.save(self._profile_id, self._memory)
            self._memory_dirty = False
        return classified

    def classify(self, txn: Transaction) -> Transaction:
        if txn.txn_type not in {
            TransactionType.INTERNAL_TRANSFER.value,
            TransactionType.EXTERNAL_TRANSFER.value,
            TransactionType.SALARY_INCOME.value,
        }:
            return txn

        recipient_iban = self._extract_iban(txn.raw_description)
        recipient_name = self._extract_recipient_name(txn.raw_description)
        normalized_recipient = normalize_text(recipient_name or "")
        normalized_description = normalize_text(txn.raw_description or "")
        entity_key = recipient_iban or normalized_recipient or normalized_description[:64] or None

        base_update = {
            "recipient_iban": recipient_iban,
            "recipient_name": recipient_name,
            "entity_key": entity_key,
        }

        if txn.txn_type == TransactionType.INTERNAL_TRANSFER.value:
            return replace(
                txn,
                **base_update,
                state="excluded",
                evidence_type="internal_transfer_filter",
                match_source="internal_transfer_filter",
            )

        if entity_key and entity_key in self._memory:
            memory_item = self._memory.get(entity_key) or {}
            return replace(
                txn,
                **base_update,
                category_area=str(memory_item.get("category", txn.category_area or "Unknown")),
                state="confirmed",
                evidence_type=str(memory_item.get("evidence_type", "entity_memory")),
                match_source="entity_memory",
                confidence=max(float(txn.confidence or 0.0), float(memory_item.get("confidence", 0.95))),
                fuzzy_match_term=None,
                fuzzy_match_score=None,
            )

        if txn.txn_type == TransactionType.SALARY_INCOME.value:
            self._remember(
                entity_key=entity_key,
                canonical_name=recipient_name or "salary_income",
                category_area="Other",
                confidence=max(float(txn.confidence or 0.0), 0.98),
                evidence_type="explicit_text",
                booking_date=txn.booking_date,
            )
            return replace(
                txn,
                **base_update,
                state="confirmed",
                evidence_type="explicit_text",
                match_source="bootstrap_exact",
                category_area="Other",
            )

        exact = self._exact_match(normalized_recipient, normalized_description)
        if exact:
            category, term = exact
            category_area = self._category_to_area(category)
            self._remember(
                entity_key=entity_key,
                canonical_name=recipient_name or term,
                category_area=category_area,
                confidence=max(float(txn.confidence or 0.0), 0.97),
                evidence_type="exact_text",
                booking_date=txn.booking_date,
            )
            return replace(
                txn,
                **base_update,
                category_area=category_area,
                state="confirmed",
                evidence_type="exact_text",
                match_source="bootstrap_exact",
                confidence=max(float(txn.confidence or 0.0), 0.97),
                fuzzy_match_term=term,
                fuzzy_match_score=100.0,
            )

        fuzzy = self._fuzzy_match(normalized_recipient, normalized_description)
        if fuzzy:
            category, term, score = fuzzy
            state = "confirmed" if score >= 95 else "candidate"
            category_area = self._category_to_area(category)
            if state == "confirmed":
                self._remember(
                    entity_key=entity_key,
                    canonical_name=recipient_name or term,
                    category_area=category_area,
                    confidence=max(float(txn.confidence or 0.0), round(min(0.99, score / 100.0), 2)),
                    evidence_type="fuzzy_text",
                    booking_date=txn.booking_date,
                )
            return replace(
                txn,
                **base_update,
                category_area=category_area,
                state=state,
                evidence_type="fuzzy_text",
                match_source="bootstrap_fuzzy",
                confidence=max(float(txn.confidence or 0.0), round(min(0.99, score / 100.0), 2)),
                fuzzy_match_term=term,
                fuzzy_match_score=round(score, 2),
            )

        return replace(
            txn,
            **base_update,
            state="unknown",
            evidence_type="no_signal",
            match_source="unknown",
            fuzzy_match_term=None,
            fuzzy_match_score=None,
        )

    def _load_rules(self) -> Dict[str, Dict[str, object]]:
        if not os.path.exists(self._bootstrap_dictionary_path):
            return {}
        try:
            with open(self._bootstrap_dictionary_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    def _extract_iban(self, text: str) -> Optional[str]:
        match = IBAN_PATTERN.search(text or "")
        return match.group(0).upper() if match else None

    def _extract_recipient_name(self, text: str) -> Optional[str]:
        if not text:
            return None
        chunks = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
        if len(chunks) >= 2:
            return chunks[1]
        return chunks[0] if chunks else None

    def _exact_match(self, normalized_recipient: str, normalized_description: str) -> Optional[Tuple[str, str]]:
        haystack = f"{normalized_recipient} {normalized_description}".strip()
        for category, config in self._rules.items():
            terms = []
            for key in ("exact_terms", "transfer_terms", "recipient_terms"):
                terms.extend(config.get(key, []) or [])
            negative_terms = [normalize_text(str(t)) for t in (config.get("negative_terms", []) or [])]
            if any(term and term in haystack for term in negative_terms):
                continue
            for term in terms:
                term_norm = normalize_text(str(term))
                if term_norm and term_norm in haystack:
                    return category, str(term)
        return None

    def _fuzzy_match(self, normalized_recipient: str, normalized_description: str) -> Optional[Tuple[str, str, float]]:
        fields = [normalized_recipient, normalized_description]
        terms_by_category: Dict[str, List[str]] = {}
        min_scores: Dict[str, float] = {}
        for category, config in self._rules.items():
            if not bool(config.get("fuzzy_enabled", False)):
                continue
            terms = []
            for key in ("exact_terms", "transfer_terms", "recipient_terms"):
                terms.extend(config.get(key, []) or [])
            terms_by_category[category] = [normalize_text(str(term)) for term in terms if str(term).strip()]
            min_scores[category] = float(config.get("min_fuzzy_score", 95))

        best_category: Optional[str] = None
        best_term = ""
        best_score = -1.0
        for field in fields:
            result = find_best_category_match(field, terms_by_category)
            if result and result.score > best_score:
                best_category = result.category
                best_term = result.matched_term
                best_score = result.score

        if not best_category:
            return None
        if best_score < min_scores.get(best_category, 95):
            return None
        return best_category, best_term, best_score

    def _category_to_area(self, category: str) -> str:
        mapping = {
            "housing": "Housing",
            "utilities": "Housing",
            "salary_income": "Other",
            "internal_transfer": "Other",
        }
        return mapping.get(category, "Unknown")

    def _backfill_from_memory(self, txn: Transaction) -> Transaction:
        if txn.txn_type not in {
            TransactionType.EXTERNAL_TRANSFER.value,
            TransactionType.SALARY_INCOME.value,
        }:
            return txn
        if (txn.state or "") not in {"candidate", "unknown"}:
            return txn

        entity_key = (txn.entity_key or "").strip()
        if not entity_key:
            return txn
        memory_item = self._memory.get(entity_key) or {}
        if str(memory_item.get("status", "")).lower() != "confirmed":
            return txn

        return replace(
            txn,
            category_area=str(memory_item.get("category", txn.category_area or "Unknown")),
            state="confirmed",
            evidence_type="entity_memory_backfill",
            match_source="entity_memory_backfill",
            confidence=max(float(txn.confidence or 0.0), float(memory_item.get("confidence", 0.95))),
            fuzzy_match_term=txn.fuzzy_match_term,
            fuzzy_match_score=txn.fuzzy_match_score,
        )

    def _remember(
        self,
        entity_key: Optional[str],
        canonical_name: str,
        category_area: str,
        confidence: float,
        evidence_type: str,
        booking_date: str,
    ) -> None:
        if not entity_key:
            return
        month_key = self._month_key(booking_date)
        self._memory[entity_key] = {
            "canonical_name": canonical_name,
            "category": category_area,
            "confidence": round(float(confidence), 4),
            "status": "confirmed",
            "first_seen": month_key,
            "last_seen": month_key,
            "months_seen": 1,
            "evidence_type": evidence_type,
        }
        self._memory_dirty = True

    def _month_key(self, booking_date: str) -> str:
        raw = (booking_date or "").strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                parsed = datetime.strptime(raw, fmt)
                return parsed.strftime("%Y-%m")
            except ValueError:
                continue
        return datetime.utcnow().strftime("%Y-%m")




