from __future__ import annotations

from typing import List

import pdfplumber


def read_pdf_pages(pdf_path: str) -> List[str]:
    """Extract plain text from each PDF page without OCR."""
    pages_text: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return pages_text

