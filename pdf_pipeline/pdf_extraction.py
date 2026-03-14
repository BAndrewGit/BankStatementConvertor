from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import os
import re
from typing import Dict, Iterable, List, Set, Tuple

import pdfplumber


class PdfProcessingError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed as text."""


@dataclass(frozen=True)
class RawPdfLine:
    source_pdf: str
    page_number: int
    line_number: int
    section: str  # header | body | footer
    text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PageExtraction:
    source_pdf: str
    page_number: int
    header_lines: List[str]
    body_lines: List[str]
    footer_lines: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalize_line_for_repetition(line: str) -> str:
    compact = " ".join(line.split()).strip().lower()
    compact = re.sub(r"\d+", "#", compact)
    return compact


def _safe_extract_page_lines(page: pdfplumber.page.Page, page_index: int, source_pdf: str) -> List[str]:
    try:
        text = page.extract_text() or ""
    except Exception as exc:  # pragma: no cover - defensive branch
        raise PdfProcessingError(
            f"Nu pot extrage text din pagina {page_index} pentru fisierul '{source_pdf}': {exc}"
        ) from exc

    lines = []
    for line in text.splitlines():
        cleaned = " ".join(line.split()).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _build_repetitive_candidates(
    pages_lines: List[List[str]],
    top_window_size: int,
    bottom_window_size: int,
    min_occurrences: int,
) -> Set[str]:
    occurrences: Dict[str, Set[int]] = {}

    for page_index, lines in enumerate(pages_lines):
        if not lines:
            continue

        top_candidates = lines[:top_window_size]
        bottom_candidates = lines[max(0, len(lines) - bottom_window_size):]

        for line in top_candidates + bottom_candidates:
            key = _normalize_line_for_repetition(line)
            if not key:
                continue
            occurrences.setdefault(key, set()).add(page_index)

    return {key for key, seen_pages in occurrences.items() if len(seen_pages) >= min_occurrences}


def _split_page_sections(
    lines: List[str],
    repetitive_keys: Set[str],
    top_window_size: int,
    bottom_window_size: int,
) -> Tuple[List[str], List[str], List[str]]:
    header_lines: List[str] = []
    body_lines: List[str] = []
    footer_lines: List[str] = []

    for idx, line in enumerate(lines):
        key = _normalize_line_for_repetition(line)
        in_top = idx < top_window_size
        in_bottom = idx >= max(0, len(lines) - bottom_window_size)

        if key in repetitive_keys and in_top:
            header_lines.append(line)
        elif key in repetitive_keys and in_bottom:
            footer_lines.append(line)
        else:
            body_lines.append(line)

    return header_lines, body_lines, footer_lines


def _flatten_raw_lines(pages: Iterable[PageExtraction]) -> List[RawPdfLine]:
    rows: List[RawPdfLine] = []

    for page in pages:
        line_number = 1
        for line in page.header_lines:
            rows.append(
                RawPdfLine(
                    source_pdf=page.source_pdf,
                    page_number=page.page_number,
                    line_number=line_number,
                    section="header",
                    text=line,
                )
            )
            line_number += 1

        for line in page.body_lines:
            rows.append(
                RawPdfLine(
                    source_pdf=page.source_pdf,
                    page_number=page.page_number,
                    line_number=line_number,
                    section="body",
                    text=line,
                )
            )
            line_number += 1

        for line in page.footer_lines:
            rows.append(
                RawPdfLine(
                    source_pdf=page.source_pdf,
                    page_number=page.page_number,
                    line_number=line_number,
                    section="footer",
                    text=line,
                )
            )
            line_number += 1

    return rows


def extract_raw_pdf_lines(
    pdf_path: str,
    top_window_size: int = 4,
    bottom_window_size: int = 4,
    min_repetitive_pages: int = 2,
) -> Tuple[List[PageExtraction], List[RawPdfLine]]:
    """Extract raw lines for each page and mark likely header/footer/body sections."""
    if not os.path.exists(pdf_path):
        raise PdfProcessingError(f"Fisierul PDF nu exista: '{pdf_path}'")

    if top_window_size <= 0 or bottom_window_size <= 0:
        raise ValueError("top_window_size si bottom_window_size trebuie sa fie > 0")

    source_pdf = os.path.basename(pdf_path)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_lines = [
                _safe_extract_page_lines(page, page_index + 1, source_pdf)
                for page_index, page in enumerate(pdf.pages)
            ]
    except PdfProcessingError:
        raise
    except Exception as exc:
        raise PdfProcessingError(f"Nu pot deschide PDF-ul '{pdf_path}': {exc}") from exc

    if not pages_lines:
        raise PdfProcessingError(f"PDF-ul '{pdf_path}' nu contine pagini.")

    repetitive_keys = _build_repetitive_candidates(
        pages_lines=pages_lines,
        top_window_size=top_window_size,
        bottom_window_size=bottom_window_size,
        min_occurrences=min_repetitive_pages,
    )

    pages: List[PageExtraction] = []
    for page_index, lines in enumerate(pages_lines, start=1):
        header_lines, body_lines, footer_lines = _split_page_sections(
            lines=lines,
            repetitive_keys=repetitive_keys,
            top_window_size=top_window_size,
            bottom_window_size=bottom_window_size,
        )

        pages.append(
            PageExtraction(
                source_pdf=source_pdf,
                page_number=page_index,
                header_lines=header_lines,
                body_lines=body_lines,
                footer_lines=footer_lines,
            )
        )

    raw_lines = _flatten_raw_lines(pages)
    return pages, raw_lines


def save_raw_pdf_lines_csv(raw_lines: List[RawPdfLine], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["source_pdf", "page_number", "line_number", "section", "text"],
        )
        writer.writeheader()
        for row in raw_lines:
            writer.writerow(row.to_dict())

