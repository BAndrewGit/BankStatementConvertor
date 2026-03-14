from __future__ import annotations

from hashlib import sha1
import re
from typing import List, Optional, Sequence, Tuple

from src.domain.enums import SourceSection
from src.domain.models import Transaction

from .line_segmenter import segment_pages_to_lines
from .pdf_reader import read_pdf_pages


DATE_PATTERN = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
DATE_START_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}\b")
CURRENCY_PATTERN = re.compile(r"\b(RON|EUR|USD|GBP)\b", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"([+-]?\d+(?:[.,]\d{3})*(?:[.,]\d{2}))")

BOOKED_SECTION_HINTS = ["TRANZACTII EFECTUATE", "TRANZACTII CONTABILIZATE", "OPERATIUNI EFECTUATE"]
BLOCKED_SECTION_HINTS = ["SUME BLOCATE", "BLOCARE SUME", "OPERATIUNI BLOCATE"]
BOOKED_TABLE_HINTS = ["DATA DESCRIERE DEBIT CREDIT", "DATA DESCRIERE DEBIT", "DATA DESCRIERE CREDIT"]

NON_TRANSACTION_DATE_HINTS = [
    "RULAJ ZI",
    "SOLD FINAL ZI",
    "SOLD ANTERIOR",
    "SOLD FINAL",
    "EXTRAS CONT",
    "RULAJ TOTAL CONT",
]

BOOKED_BOUNDARY_HINTS = [
    "PLATA LA POS",
    "TRANSFER INTERN",
    "P2P BTPAY",
    "RETRAGERE DE NUMERAR",
    "TAXA SERVICIU",
    "COMISION ",
    "PLATA INSTANT",
]


class BTStatementParser:
    def parse(self, pdf_path: str) -> List[Transaction]:
        pages_text = read_pdf_pages(pdf_path)
        pages_lines = segment_pages_to_lines(pages_text)

        candidates = self._collect_candidates(pages_lines)
        transactions: List[Transaction] = []

        for index, (section, candidate_text) in enumerate(candidates, start=1):
            transaction = self._parse_candidate(section, candidate_text, index)
            if transaction is not None:
                transactions.append(transaction)

        return transactions

    def _collect_candidates(self, pages_lines: Sequence[Sequence[str]]) -> List[Tuple[str, str]]:
        current_section: Optional[str] = None
        current_candidate_lines: List[str] = []
        current_candidate_section: Optional[str] = None
        last_seen_booking_date: Optional[str] = None
        collected: List[Tuple[str, str]] = []

        for page_lines in pages_lines:
            for line in page_lines:
                section = self._detect_section(line)
                if section is not None:
                    if current_candidate_lines and current_candidate_section:
                        collected.append((current_candidate_section, " ".join(current_candidate_lines)))
                    current_section = section
                    current_candidate_lines = []
                    current_candidate_section = None
                    if section == SourceSection.BLOCKED_AMOUNTS.value:
                        last_seen_booking_date = None
                    continue

                if current_section is None:
                    # Some BT statement variants do not print an explicit booked section title.
                    # If a dated transactional row appears first, start in booked section.
                    if DATE_START_PATTERN.search(line) and not self._is_non_transaction_date_line(line):
                        current_section = SourceSection.BOOKED_TRANSACTIONS.value
                    else:
                        continue

                starts_with_date = bool(DATE_START_PATTERN.search(line))
                blocked_entry_start = (
                    current_section == SourceSection.BLOCKED_AMOUNTS.value
                    and self._is_blocked_entry_start(line)
                )
                booked_boundary_start = (
                    current_section == SourceSection.BOOKED_TRANSACTIONS.value
                    and self._is_booked_boundary_line(line)
                )

                if starts_with_date or blocked_entry_start or booked_boundary_start:
                    if self._is_non_transaction_date_line(line):
                        continue
                    if current_candidate_lines and current_candidate_section:
                        collected.append((current_candidate_section, " ".join(current_candidate_lines)))

                    if starts_with_date:
                        date_match = DATE_PATTERN.search(line)
                        last_seen_booking_date = date_match.group(1) if date_match else last_seen_booking_date
                        current_candidate_lines = [line]
                    elif booked_boundary_start and last_seen_booking_date and not DATE_PATTERN.search(line):
                        current_candidate_lines = [f"{last_seen_booking_date} {line}"]
                    else:
                        current_candidate_lines = [line]

                    current_candidate_section = current_section
                    continue

                if current_candidate_lines:
                    current_candidate_lines.append(line)

        if current_candidate_lines and current_candidate_section:
            collected.append((current_candidate_section, " ".join(current_candidate_lines)))

        return collected

    def _detect_section(self, line: str) -> Optional[str]:
        upper = line.upper()
        if any(hint in upper for hint in BLOCKED_SECTION_HINTS):
            return SourceSection.BLOCKED_AMOUNTS.value
        if any(hint in upper for hint in BOOKED_TABLE_HINTS):
            return SourceSection.BOOKED_TRANSACTIONS.value
        if any(hint in upper for hint in BOOKED_SECTION_HINTS):
            return SourceSection.BOOKED_TRANSACTIONS.value
        return None

    def _is_non_transaction_date_line(self, line: str) -> bool:
        upper = line.upper()
        return any(hint in upper for hint in NON_TRANSACTION_DATE_HINTS)

    def _is_blocked_entry_start(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith("-"):
            return False
        upper = stripped.upper()
        return "AFERENTA TRANZACTIEI" in upper and bool(DATE_PATTERN.search(stripped))

    def _is_booked_boundary_line(self, line: str) -> bool:
        upper = line.strip().upper()
        return any(upper.startswith(hint) for hint in BOOKED_BOUNDARY_HINTS)

    def _parse_candidate(self, section: str, text: str, sequence: int) -> Optional[Transaction]:
        date_match = DATE_PATTERN.search(text)
        if not date_match:
            return None

        booking_date = date_match.group(1)
        currency_match = CURRENCY_PATTERN.search(text)
        currency = currency_match.group(1).upper() if currency_match else "RON"

        amount, direction = self._extract_amount_and_direction(text, section)
        if amount is None:
            return None

        raw_description = self._build_description(text, booking_date)
        transaction_id = self._build_transaction_id(
            sequence=sequence,
            booking_date=booking_date,
            amount=amount,
            direction=direction,
            raw_description=raw_description,
            section=section,
        )

        return Transaction(
            transaction_id=transaction_id,
            booking_date=booking_date,
            amount=amount,
            currency=currency,
            direction=direction,
            raw_description=raw_description,
            source_section=section,
        )

    def _extract_amount_and_direction(self, text: str, section: str) -> Tuple[Optional[float], str]:
        if section == SourceSection.BLOCKED_AMOUNTS.value:
            amount = self._first_amount(text)
            return amount, "debit"

        debit_marked = self._amount_after_marker(text, ["DEBIT", "DB"])
        if debit_marked is not None:
            return debit_marked, "debit"

        credit_marked = self._amount_after_marker(text, ["CREDIT", "CR"])
        if credit_marked is not None:
            return credit_marked, "credit"

        cleaned = re.sub(r"\b(?:SOLD|BALANCE)\b.*$", "", text, flags=re.IGNORECASE)
        amount_strings = AMOUNT_PATTERN.findall(cleaned)
        if not amount_strings:
            return None, "debit"

        first_amount_raw = amount_strings[0]
        first_amount = self._to_float(first_amount_raw)
        if first_amount is None:
            return None, "debit"

        if first_amount_raw.strip().startswith("+"):
            return abs(first_amount), "credit"
        if first_amount_raw.strip().startswith("-"):
            return abs(first_amount), "debit"

        text_upper = text.upper()
        if any(token in text_upper for token in ["INCASARE", "ALIMENTARE", "DOBANDA", "REFUND"]):
            return abs(first_amount), "credit"

        return abs(first_amount), "debit"

    def _amount_after_marker(self, text: str, markers: Sequence[str]) -> Optional[float]:
        markers_pattern = "|".join(re.escape(marker) for marker in markers)
        pattern = re.compile(
            rf"\b(?:{markers_pattern})\b[^\d+-]{{0,12}}([+-]?\d+(?:[.,]\d{{3}})*(?:[.,]\d{{2}}))",
            re.IGNORECASE,
        )
        matched = pattern.search(text)
        if not matched:
            return None
        value = self._to_float(matched.group(1))
        if value is None:
            return None
        return abs(value)

    def _first_amount(self, text: str) -> Optional[float]:
        match = AMOUNT_PATTERN.search(text)
        if not match:
            return None
        value = self._to_float(match.group(1))
        return abs(value) if value is not None else None

    def _to_float(self, amount_raw: str) -> Optional[float]:
        normalized = amount_raw.strip().replace(" ", "")

        if "," in normalized and "." in normalized:
            # Use the right-most separator as decimal marker and strip the other.
            last_comma = normalized.rfind(",")
            last_dot = normalized.rfind(".")
            if last_dot > last_comma:
                normalized = normalized.replace(",", "")
            else:
                normalized = normalized.replace(".", "")
                normalized = normalized.replace(",", ".")
        elif "," in normalized:
            parts = normalized.split(",")
            # Treat comma as thousands separator when grouping length is 3 (e.g., 2,900).
            if len(parts) == 2 and len(parts[1]) == 3:
                normalized = "".join(parts)
            else:
                normalized = normalized.replace(",", ".")

        try:
            return float(normalized)
        except ValueError:
            return None

    def _build_description(self, text: str, booking_date: str) -> str:
        description = text.replace(booking_date, " ", 1)
        description = re.sub(r"\s+", " ", description).strip(" -:;")
        return description

    def _build_transaction_id(
        self,
        sequence: int,
        booking_date: str,
        amount: float,
        direction: str,
        raw_description: str,
        section: str,
    ) -> str:
        raw = "|".join(
            [
                str(sequence),
                booking_date,
                f"{amount:.2f}",
                direction,
                raw_description,
                section,
            ]
        )
        digest = sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"TXN-{sequence:06d}-{digest}"



