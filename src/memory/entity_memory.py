from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from typing import Dict, Optional


DEFAULT_MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..",
    "data",
    "entity_memory",
)


@dataclass
class EntityMemoryRecord:
    canonical_name: str
    category: str
    confidence: float
    status: str
    first_seen: str
    last_seen: str
    months_seen: int
    evidence_type: str


class EntityMemoryRepository:
    """Local per-profile memory persisted as JSON under data/entity_memory/."""

    def __init__(self, memory_dir: Optional[str] = None) -> None:
        self._memory_dir = memory_dir or DEFAULT_MEMORY_DIR

    def get_profile_path(self, profile_id: str) -> str:
        safe = profile_id.strip() or "default"
        return os.path.join(self._memory_dir, f"{safe}.json")

    def load(self, profile_id: str) -> Dict[str, Dict[str, object]]:
        path = self.get_profile_path(profile_id)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    def save(self, profile_id: str, memory: Dict[str, Dict[str, object]]) -> str:
        os.makedirs(self._memory_dir, exist_ok=True)
        path = self.get_profile_path(profile_id)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(memory, handle, ensure_ascii=False, indent=2, sort_keys=True)
        return path

    def upsert(
        self,
        profile_id: str,
        entity_key: str,
        canonical_name: str,
        category: str,
        confidence: float,
        status: str,
        evidence_type: str,
        month_key: Optional[str] = None,
    ) -> str:
        if not entity_key:
            return self.get_profile_path(profile_id)

        memory = self.load(profile_id)
        now_month = month_key or datetime.utcnow().strftime("%Y-%m")
        existing = memory.get(entity_key)

        if existing:
            first_seen = str(existing.get("first_seen", now_month))
            months_seen = int(existing.get("months_seen", 1))
            if str(existing.get("last_seen", "")) != now_month:
                months_seen += 1
        else:
            first_seen = now_month
            months_seen = 1

        memory[entity_key] = {
            "canonical_name": canonical_name,
            "category": category,
            "confidence": round(float(confidence), 4),
            "status": status,
            "first_seen": first_seen,
            "last_seen": now_month,
            "months_seen": months_seen,
            "evidence_type": evidence_type,
        }
        return self.save(profile_id, memory)

