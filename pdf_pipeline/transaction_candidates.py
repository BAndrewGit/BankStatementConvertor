from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import os
import re
from typing import Dict, List, Sequence, Tuple

from .pdf_extraction import RawPdfLine


DATE_PREFIX_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}\b")
AMOUNT_PATTERN = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))")

EXCLUDE_PATTERNS = [
    re.compile(r"\brulaj\b", re.IGNORECASE),
    re.compile(r"\bsold\b", re.IGNORECASE),
    re.compile(r"\btotal\b", re.IGNORECASE),
    re.compile(r"\bdisponibil\b", re.IGNORECASE),
    re.compile(r"\bsume blocate\b", re.IGNORECASE),
    re.compile(r"\bextras de cont\b", re.IGNORECASE),
    re.compile(r"\bacest extras\b", re.IGNORECASE),
    re.compile(r"\binfo clienti\b", re.IGNORECASE),
    re.compile(r"\btiparit\b", re.IGNORECASE),
]

TRANSACTION_HINT_PATTERNS = [
    re.compile(r"plata la pos", re.IGNORECASE),
    re.compile(r"transfer", re.IGNORECASE),
    re.compile(r"retragere", re.IGNORECASE),
    re.compile(r"p2p", re.IGNORECASE),
    re.compile(r"comision", re.IGNORECASE),
    re.compile(r"incasare", re.IGNORECASE),
]


@dataclass(frozen=True)
class CandidateLine:
    source_pdf: str
    page_number: int
    line_number: int
    section: str
    text: str
    is_transaction_candidate: bool
    classification: str
    ambiguity_reason: str
    candidate_id: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TransactionCandidate:
    candidate_id: str
    source_pdf: str
    start_page: int
    end_page: int
    start_line: int
    end_line: int
    text: str
    has_date: bool
    amount_count: int
    is_ambiguous: bool
    ambiguity_reasons: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _is_excluded_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in EXCLUDE_PATTERNS)


def _has_transaction_hint(text: str) -> bool:
    return any(pattern.search(text) for pattern in TRANSACTION_HINT_PATTERNS)


def _classify_body_line(line_text: str) -> Tuple[str, List[str]]:
    has_date = bool(DATE_PREFIX_PATTERN.search(line_text))
    amount_count = len(AMOUNT_PATTERN.findall(line_text))
    has_hint = _has_transaction_hint(line_text)

    flags: List[str] = []
    if amount_count > 1:
        flags.append("multiple_amounts")

    if has_date and (amount_count > 0 or has_hint):
        return "transaction_start", flags

    if has_date and amount_count == 0:
        flags.append("date_without_amount")
        return "transaction_start", flags

    if amount_count > 0 or has_hint:
        return "transaction_continuation", flags

    return "not_transaction", flags


def _build_transaction_candidate(candidate_id: str, lines: Sequence[RawPdfLine], flags: List[str]) -> TransactionCandidate:
    text = " ".join(line.text for line in lines)
    amount_count = len(AMOUNT_PATTERN.findall(text))
    has_date = bool(DATE_PREFIX_PATTERN.search(lines[0].text))

    merged_flags = list(flags)
    if amount_count == 0:
        merged_flags.append("missing_amount")

    unique_flags = sorted(set(merged_flags))

    return TransactionCandidate(
        candidate_id=candidate_id,
        source_pdf=lines[0].source_pdf,
        start_page=lines[0].page_number,
        end_page=lines[-1].page_number,
        start_line=lines[0].line_number,
        end_line=lines[-1].line_number,
        text=text,
        has_date=has_date,
        amount_count=amount_count,
        is_ambiguous=bool(unique_flags),
        ambiguity_reasons="|".join(unique_flags),
    )


def identify_transaction_candidates(
    raw_lines: List[RawPdfLine],
) -> Tuple[List[CandidateLine], List[TransactionCandidate]]:
    """Mark candidate transaction lines and merge multi-line transactions."""
    sorted_lines = sorted(raw_lines, key=lambda line: (line.source_pdf, line.page_number, line.line_number))

    classified_lines: List[CandidateLine] = []
    candidates: List[TransactionCandidate] = []

    current_lines: List[RawPdfLine] = []
    current_flags: List[str] = []
    current_candidate_id = ""
    candidate_seq = 1

    for line in sorted_lines:
        if line.section != "body":
            classified_lines.append(
                CandidateLine(
                    source_pdf=line.source_pdf,
                    page_number=line.page_number,
                    line_number=line.line_number,
                    section=line.section,
                    text=line.text,
                    is_transaction_candidate=False,
                    classification="ignored_non_body",
                    ambiguity_reason="",
                    candidate_id="",
                )
            )
            continue

        if _is_excluded_text(line.text):
            classified_lines.append(
                CandidateLine(
                    source_pdf=line.source_pdf,
                    page_number=line.page_number,
                    line_number=line.line_number,
                    section=line.section,
                    text=line.text,
                    is_transaction_candidate=False,
                    classification="excluded_info_total",
                    ambiguity_reason="",
                    candidate_id="",
                )
            )
            continue

        line_type, line_flags = _classify_body_line(line.text)

        if line_type == "transaction_start":
            if current_lines:
                candidates.append(
                    _build_transaction_candidate(current_candidate_id, current_lines, current_flags)
                )

            current_candidate_id = f"C{candidate_seq:06d}"
            candidate_seq += 1
            current_lines = [line]
            current_flags = list(line_flags)

            classified_lines.append(
                CandidateLine(
                    source_pdf=line.source_pdf,
                    page_number=line.page_number,
                    line_number=line.line_number,
                    section=line.section,
                    text=line.text,
                    is_transaction_candidate=True,
                    classification="candidate_start",
                    ambiguity_reason="|".join(line_flags),
                    candidate_id=current_candidate_id,
                )
            )
            continue

        if line_type == "transaction_continuation":
            if current_lines:
                current_lines.append(line)
                current_flags.extend(line_flags)

                classified_lines.append(
                    CandidateLine(
                        source_pdf=line.source_pdf,
                        page_number=line.page_number,
                        line_number=line.line_number,
                        section=line.section,
                        text=line.text,
                        is_transaction_candidate=True,
                        classification="candidate_continuation",
                        ambiguity_reason="|".join(line_flags),
                        candidate_id=current_candidate_id,
                    )
                )
            else:
                orphan_id = f"C{candidate_seq:06d}"
                candidate_seq += 1
                orphan_flags = list(line_flags) + ["orphan_continuation"]
                candidates.append(_build_transaction_candidate(orphan_id, [line], orphan_flags))

                classified_lines.append(
                    CandidateLine(
                        source_pdf=line.source_pdf,
                        page_number=line.page_number,
                        line_number=line.line_number,
                        section=line.section,
                        text=line.text,
                        is_transaction_candidate=True,
                        classification="candidate_orphan",
                        ambiguity_reason="|".join(orphan_flags),
                        candidate_id=orphan_id,
                    )
                )
            continue

        classified_lines.append(
            CandidateLine(
                source_pdf=line.source_pdf,
                page_number=line.page_number,
                line_number=line.line_number,
                section=line.section,
                text=line.text,
                is_transaction_candidate=False,
                classification="ignored_non_transaction",
                ambiguity_reason="",
                candidate_id="",
            )
        )

    if current_lines:
        candidates.append(_build_transaction_candidate(current_candidate_id, current_lines, current_flags))

    return classified_lines, candidates


def save_candidate_lines_csv(candidate_lines: List[CandidateLine], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_pdf",
                "page_number",
                "line_number",
                "section",
                "text",
                "is_transaction_candidate",
                "classification",
                "ambiguity_reason",
                "candidate_id",
            ],
        )
        writer.writeheader()
        for row in candidate_lines:
            writer.writerow(row.to_dict())


def save_transaction_candidates_csv(candidates: List[TransactionCandidate], output_csv_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_id",
                "source_pdf",
                "start_page",
                "end_page",
                "start_line",
                "end_line",
                "text",
                "has_date",
                "amount_count",
                "is_ambiguous",
                "ambiguity_reasons",
            ],
        )
        writer.writeheader()
        for row in candidates:
            writer.writerow(row.to_dict())


