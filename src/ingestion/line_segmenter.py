from __future__ import annotations

import re
from collections import Counter
from typing import List


BT_NOISE_PATTERNS = [
    re.compile(r"\bBANCA TRANSILVANIA\b", re.IGNORECASE),
    re.compile(r"\bEXTRAS DE CONT\b", re.IGNORECASE),
    re.compile(r"\bPAGINA\b", re.IGNORECASE),
    re.compile(r"\bIBAN\b", re.IGNORECASE),
    re.compile(r"\bSWIFT\b", re.IGNORECASE),
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_lines(page_text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in (page_text or "").splitlines():
        normalized = normalize_whitespace(raw_line)
        if normalized:
            lines.append(normalized)
    return lines


def _line_key(line: str) -> str:
    lowered = line.lower()
    lowered = re.sub(r"\d+", "#", lowered)
    return lowered


def _looks_like_bt_noise(line: str) -> bool:
    return any(pattern.search(line) for pattern in BT_NOISE_PATTERNS)


def remove_repetitive_headers_and_footers(pages_lines: List[List[str]], window_size: int = 4) -> List[List[str]]:
    if not pages_lines:
        return []

    top_bottom_keys: List[str] = []
    for lines in pages_lines:
        if not lines:
            continue
        top_bottom = lines[:window_size] + lines[max(0, len(lines) - window_size) :]
        top_bottom_keys.extend(_line_key(line) for line in top_bottom)

    repeated_keys = {
        key
        for key, count in Counter(top_bottom_keys).items()
        if count >= 2
    }

    cleaned_pages: List[List[str]] = []
    for lines in pages_lines:
        cleaned = [
            line
            for line in lines
            if _line_key(line) not in repeated_keys and not _looks_like_bt_noise(line)
        ]
        cleaned_pages.append(cleaned)

    return cleaned_pages


def segment_pages_to_lines(pages_text: List[str]) -> List[List[str]]:
    split = [split_lines(page_text) for page_text in pages_text]
    return remove_repetitive_headers_and_footers(split)

