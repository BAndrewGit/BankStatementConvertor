from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


_UNSET = object()


class TermeneClient:
    """Minimal Termene API client used only as ONRC fallback."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] | object = _UNSET,
        password: Optional[str] | object = _UNSET,
        schema_key: Optional[str] | object = _UNSET,
    ) -> None:
        load_dotenv()

        self._base_url = (base_url or os.getenv("TERMENE_API_URL") or "https://api.termene.ro/v2").rstrip("/")

        # Explicit None means "disable credential", while omitted uses environment fallback.
        self._username = os.getenv("TERMENE_USERNAME") if username is _UNSET else username
        self._password = os.getenv("TERMENE_PASSWORD") if password is _UNSET else password
        self._schema_key = os.getenv("TERMENE_SCHEMA_KEY") if schema_key is _UNSET else schema_key

    def search_company(self, company_name: str) -> Optional[Dict[str, Any]]:
        if not company_name.strip():
            return None
        if not self._username or not self._password or not self._schema_key:
            return None

        # Endpoint can be configured because Termene free-plan routes may vary by account.
        endpoint = os.getenv("TERMENE_API_SEARCH_ENDPOINT", "")
        url = f"{self._base_url}{endpoint}"

        payload: Dict[str, Any] = {
            "query": company_name.strip(),
            "schemaKey": self._schema_key,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                auth=(self._username, self._password),
                timeout=30,
            )
            response.raise_for_status()
            data = response.json() if response.content else {}
        except Exception:
            return None

        return self._extract_first_company(data)

    def _extract_first_company(self, payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            for key in ("results", "companies", "data"):
                value = payload.get(key)
                if isinstance(value, list) and value:
                    first = value[0]
                    return first if isinstance(first, dict) else None
            if "name" in payload or "denumire" in payload:
                return payload
        if isinstance(payload, list) and payload:
            first = payload[0]
            return first if isinstance(first, dict) else None
        return None

