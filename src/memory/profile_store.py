from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from typing import Dict, List, Optional
from uuid import uuid4


DEFAULT_PROFILE_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..",
    "data",
    "profiles",
    "profiles.json",
)


@dataclass(frozen=True)
class ProfileRecord:
    profile_id: str
    profile_name: str
    questionnaire_answers: Dict[str, float]
    entity_memory: Dict[str, Dict[str, object]]
    model_artifacts_path: str
    last_run: Optional[Dict[str, object]]
    export_preferences: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "questionnaire_answers": dict(self.questionnaire_answers),
            "entity_memory": dict(self.entity_memory),
            "model_artifacts_path": self.model_artifacts_path,
            "last_run": self.last_run,
            "export_preferences": dict(self.export_preferences),
        }


class ProfileStore:
    """Simple JSON-backed profile store with CRUD and active-profile selection."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self._storage_path = os.path.abspath(storage_path or DEFAULT_PROFILE_STORE_PATH)

    def list_profiles(self) -> List[ProfileRecord]:
        payload = self._load_payload()
        return [self._to_record(item) for item in payload.get("profiles", [])]

    def get_profile(self, profile_id: str) -> Optional[ProfileRecord]:
        for profile in self.list_profiles():
            if profile.profile_id == profile_id:
                return profile
        return None

    def get_active_profile(self) -> Optional[ProfileRecord]:
        payload = self._load_payload()
        active = payload.get("active_profile_id")
        if not active:
            return None
        return self.get_profile(str(active))

    def create_profile(
        self,
        profile_name: str,
        questionnaire_answers: Optional[Dict[str, float]] = None,
        entity_memory: Optional[Dict[str, Dict[str, object]]] = None,
        model_artifacts_path: str = "",
        export_preferences: Optional[Dict[str, object]] = None,
    ) -> ProfileRecord:
        payload = self._load_payload()
        profile_id = f"profile-{uuid4().hex[:12]}"
        record = ProfileRecord(
            profile_id=profile_id,
            profile_name=profile_name.strip() or "Default Profile",
            questionnaire_answers=dict(questionnaire_answers or {}),
            entity_memory=dict(entity_memory or {}),
            model_artifacts_path=str(model_artifacts_path or ""),
            last_run=None,
            export_preferences=dict(export_preferences or {}),
        )
        payload.setdefault("profiles", []).append(record.to_dict())
        if not payload.get("active_profile_id"):
            payload["active_profile_id"] = profile_id
        self._save_payload(payload)
        return record

    def update_profile(self, profile_id: str, **changes: object) -> ProfileRecord:
        payload = self._load_payload()
        profiles = payload.get("profiles", [])
        for index, item in enumerate(profiles):
            if str(item.get("profile_id")) != profile_id:
                continue

            updated = dict(item)
            for key in [
                "profile_name",
                "questionnaire_answers",
                "entity_memory",
                "model_artifacts_path",
                "last_run",
                "export_preferences",
            ]:
                if key in changes:
                    updated[key] = changes[key]

            if "last_run" in changes and isinstance(changes["last_run"], dict):
                enriched_last_run = dict(changes["last_run"])
                enriched_last_run.setdefault("updated_at", datetime.utcnow().isoformat())
                updated["last_run"] = enriched_last_run

            profiles[index] = updated
            payload["profiles"] = profiles
            self._save_payload(payload)
            return self._to_record(updated)

        raise KeyError(f"Profile not found: {profile_id}")

    def delete_profile(self, profile_id: str) -> None:
        payload = self._load_payload()
        profiles = payload.get("profiles", [])
        remaining = [item for item in profiles if str(item.get("profile_id")) != profile_id]
        if len(remaining) == len(profiles):
            raise KeyError(f"Profile not found: {profile_id}")

        payload["profiles"] = remaining
        active_id = payload.get("active_profile_id")
        if active_id == profile_id:
            payload["active_profile_id"] = remaining[0]["profile_id"] if remaining else None
        self._save_payload(payload)

    def set_active_profile(self, profile_id: str) -> ProfileRecord:
        profile = self.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"Profile not found: {profile_id}")
        payload = self._load_payload()
        payload["active_profile_id"] = profile_id
        self._save_payload(payload)
        return profile

    def _load_payload(self) -> Dict[str, object]:
        if not os.path.exists(self._storage_path):
            return {"active_profile_id": None, "profiles": []}

        try:
            with open(self._storage_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                payload.setdefault("active_profile_id", None)
                payload.setdefault("profiles", [])
                return payload
        except Exception:
            pass

        return {"active_profile_id": None, "profiles": []}

    def _save_payload(self, payload: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        with open(self._storage_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    @staticmethod
    def _to_record(payload: Dict[str, object]) -> ProfileRecord:
        return ProfileRecord(
            profile_id=str(payload.get("profile_id", "")),
            profile_name=str(payload.get("profile_name", "")),
            questionnaire_answers={
                str(key): float(value)
                for key, value in (payload.get("questionnaire_answers") or {}).items()
            },
            entity_memory=dict(payload.get("entity_memory") or {}),
            model_artifacts_path=str(payload.get("model_artifacts_path", "")),
            last_run=payload.get("last_run") if isinstance(payload.get("last_run"), dict) else None,
            export_preferences=dict(payload.get("export_preferences") or {}),
        )

