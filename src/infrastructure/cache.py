from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from threading import RLock
from typing import Any, Dict, Optional


class CacheRepository(ABC):
    @abstractmethod
    def get(self, namespace: str, key: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def set(self, namespace: str, key: str, value: Any) -> None:
        raise NotImplementedError


class InMemoryCacheRepository(CacheRepository):
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, namespace: str, key: str) -> Any:
        return self._store.get(namespace, {}).get(key)

    def set(self, namespace: str, key: str, value: Any) -> None:
        self._store.setdefault(namespace, {})[key] = value


class FileCacheRepository(CacheRepository):
    def __init__(self, cache_file_path: Optional[str] = None, persist_every_n_writes: int = 1) -> None:
        self._cache_file_path = cache_file_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "..",
            ".cache",
            "pipeline_cache.json",
        )
        self._persist_every_n_writes = max(1, int(persist_every_n_writes))
        self._lock = RLock()
        self._store: Dict[str, Dict[str, Any]] = {}
        self._pending_writes = 0
        self._load()

    def get(self, namespace: str, key: str) -> Any:
        with self._lock:
            return self._store.get(namespace, {}).get(key)

    def set(self, namespace: str, key: str, value: Any) -> None:
        with self._lock:
            self._store.setdefault(namespace, {})[key] = value
            self._pending_writes += 1
            if self._pending_writes >= self._persist_every_n_writes:
                self._persist_locked()

    def flush(self) -> None:
        with self._lock:
            if self._pending_writes > 0:
                self._persist_locked()

    def _load(self) -> None:
        with self._lock:
            if not os.path.exists(self._cache_file_path):
                self._store = {}
                return

            try:
                with open(self._cache_file_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    self._store = payload
                else:
                    self._store = {}
            except Exception:
                self._store = {}

    def _persist(self) -> None:
        with self._lock:
            self._persist_locked()

    def _persist_locked(self) -> None:
        # Caller must hold self._lock.
        os.makedirs(os.path.dirname(self._cache_file_path) or ".", exist_ok=True)
        with open(self._cache_file_path, "w", encoding="utf-8") as handle:
            json.dump(self._store, handle, ensure_ascii=False, sort_keys=True, indent=2)
        self._pending_writes = 0

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass

