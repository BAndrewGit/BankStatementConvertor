from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from rapidfuzz import fuzz


@dataclass(frozen=True)
class FuzzyMatchResult:
    category: str
    matched_term: str
    score: float


def find_best_category_match(
    normalized_text: str,
    category_terms: Dict[str, Iterable[str]],
) -> Optional[FuzzyMatchResult]:
    if not normalized_text:
        return None

    best: Optional[FuzzyMatchResult] = None
    for category, terms in category_terms.items():
        for term in terms:
            term_norm = (term or "").strip().lower()
            if not term_norm:
                continue
            score = float(fuzz.token_set_ratio(normalized_text, term_norm))
            if best is None or score > best.score:
                best = FuzzyMatchResult(
                    category=category,
                    matched_term=term,
                    score=score,
                )
    return best

