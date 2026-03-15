from __future__ import annotations

import re
import unicodedata


NOISE_TOKENS = {
    "REF",
    "RRN",
    "TID",
    "POS",
    "EPOS",
    "PLATA",
    "TRANSFER",
    "INSTANT",
    "CANAL",
    "ELECTRONIC",
}


def normalize_text(value: str) -> str:
    text = value or ""
    deaccented = unicodedata.normalize("NFKD", text)
    deaccented = "".join(ch for ch in deaccented if not unicodedata.combining(ch))
    lowered = deaccented.lower()
    lowered = re.sub(r"\b(?:ref|rrn|tid)\s*:?\s*[a-z0-9-]+\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return ""

    kept_tokens = [token for token in lowered.split() if token.upper() not in NOISE_TOKENS]
    return " ".join(kept_tokens)

