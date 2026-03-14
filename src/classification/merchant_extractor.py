from __future__ import annotations

from dataclasses import replace
import re
from typing import Optional

from src.domain.enums import TransactionType
from src.domain.models import Transaction


ELIGIBLE_TYPES = {
    TransactionType.CARD_PURCHASE.value,
    TransactionType.SUBSCRIPTION.value,
    TransactionType.UTILITY_PAYMENT.value,
}

REF_PATTERNS = [
    re.compile(r"\bTID\s*[:=]?\s*[A-Z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\bRRN\s*[:=]?\s*[A-Z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\bREF\s*[:=]?\s*[A-Z0-9-]+\b", re.IGNORECASE),
]

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"[+-]?\d+(?:[.,]\d{3})*[.,]\d{2}\s*(?:RON|EUR|USD|GBP)?", re.IGNORECASE)
TECHNICAL_TOKENS = {
    "POS",
    "EPOS",
    "UNKNOWN",
    "TRANZACTIE",
    "TRANZACTIA",
    "COMISION",
    "VALOARE",
    "CARD",
    "BT",
}

LOCATION_TAIL_TOKENS = {
    "RO",
    "ROMANIA",
    "BUCURESTI",
    "BUCURESTIULUI",
    "CLUJ",
    "TIMISOARA",
    "IASI",
    "MOGHIOR",
}

LEADING_BOILERPLATE_PATTERN = re.compile(
    r"^(?:PLATA\s+LA\s+POS(?:\s+NON-BT)?(?:\s+CU\s+CARD)?\s+MASTERCARD\s*|PLATA\s+LA\s+POS\s*)",
    re.IGNORECASE,
)
TRAILING_NOISE_PATTERN = re.compile(
    r"\b(?:SOLD\s+FINAL\s+ZI|SOLD\s+FINAL\s+CONT|SOLICITANT\s*:?[\sA-Z0-9-]*TIPARIT\s*:?.*)$",
    re.IGNORECASE,
)


class MerchantExtractor:
    def extract(self, txn: Transaction) -> Transaction:
        if txn.txn_type not in ELIGIBLE_TYPES:
            return replace(txn, merchant_raw=None)

        merchant = self._extract_merchant(txn.raw_description)
        return replace(txn, merchant_raw=merchant)

    def _extract_merchant(self, description: str) -> Optional[str]:
        cleaned = description or ""

        cleaned = URL_PATTERN.sub(" ", cleaned)
        for pattern in REF_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)

        cleaned = re.sub(r"\bBALAN\s+ANDREI\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bCOMISION\s+TRANZACTIE\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bVALOARE\s+TRANZACTIE\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = AMOUNT_PATTERN.sub(" ", cleaned)
        cleaned = LEADING_BOILERPLATE_PATTERN.sub("", cleaned)
        cleaned = TRAILING_NOISE_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,")

        if not cleaned:
            return None

        # High-precision merchant patterns from real BT statement variants.
        payu_match = re.search(r"PAYU\*[A-Z0-9./-]+", cleaned, re.IGNORECASE)
        if payu_match:
            return payu_match.group(0)

        pranz_match = re.search(r"PRANZOBYESS(?:@FBP\s*C\d+)?", cleaned, re.IGNORECASE)
        if pranz_match:
            return "PRANZOBYESS"

        ppc_match = re.search(r"PPC\s+ENERGIE\s+SA", cleaned, re.IGNORECASE)
        if ppc_match:
            return "PPC ENERGIE SA"

        mega_match = re.search(r"MEGAIMAGE(?:\s+\d{2,6}(?:\s+[A-Z]+)*)?", cleaned, re.IGNORECASE)
        if mega_match:
            return re.sub(r"\s+\d{2,6}.*$", "", mega_match.group(0), flags=re.IGNORECASE).strip()

        for pattern, canonical in [
            (r"\bCARREFOUR\b", "CARREFOUR"),
            (r"\bAUCHAN(?:\s+ROMANIA\s+SA)?\b", "AUCHAN ROMANIA SA"),
            (r"\bAUCH\b", "AUCHAN ROMANIA SA"),
            (r"\bMETROREX\b", "METROREX VALIDATOARE"),
            (r"\bTACO\s+BELL\b", "TACO BELL"),
            (r"\bMOVIEPLEX\b", "MOVIEPLEX"),
            (r"\bDM\s+DROGERIE\s+MARKT\b", "DM DROGERIE MARKT"),
            (r"\bFROO\b", "FROO"),
            (r"\bDIGI\s+RO\b", "DIGI"),
            (r"\bJKC\s+RESTAURANTS\b", "JKC RESTAURANTS"),
            (r"\bFARMACIA\b", "FARMACIA"),
        ]:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return canonical

        compact = re.sub(r"[^A-Za-z0-9*./&@\s-]", " ", cleaned)
        compact = re.sub(r"\s+", " ", compact).strip()
        tokens = compact.split(" ")
        if not tokens:
            return None

        filtered = [token for token in tokens if token.upper() not in TECHNICAL_TOKENS]
        while filtered and (filtered[-1].upper() in LOCATION_TAIL_TOKENS or filtered[-1].isdigit()):
            filtered.pop()

        if not filtered:
            return None

        return " ".join(filtered)

