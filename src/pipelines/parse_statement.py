from __future__ import annotations

from hashlib import sha256
from typing import List

from src.domain.models import Transaction
from src.ingestion.bt_statement_parser import BTStatementParser
from src.infrastructure.cache import CacheRepository, FileCacheRepository


PARSE_CACHE_VERSION = "v3"


def _sha256_file(file_path: str) -> str:
    hasher = sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_statement(pdf_path: str, cache_repo: CacheRepository | None = None) -> List[Transaction]:
    cache = cache_repo or FileCacheRepository()
    cache_key = f"{PARSE_CACHE_VERSION}:{_sha256_file(pdf_path)}"

    cached = cache.get("parse", cache_key)
    if isinstance(cached, list):
        return [Transaction.from_dict(item) for item in cached]

    parsed = BTStatementParser().parse(pdf_path)
    cache.set("parse", cache_key, [txn.to_dict() for txn in parsed])
    return parsed

