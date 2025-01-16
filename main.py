import tkinter as tk
from tkinter import filedialog
import os
import re
import csv
import pdfplumber


DATE_LINE_PATTERN = re.compile(r"^(\d{2}/\d{2}/\d{4})(.*)$")
AMOUNT_PATTERN = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))")


KEYWORDS_EXCLUDE = [
    "BANCA TRANSILVANIA S.A.",
    "Capitalul social:",
    "Clasificare BT:",
    "RULAJ ZI",
    "SOLD FINAL ZI",
    "RULAJ TOTAL CONT",
    "SOLD FINAL CONT",
    "TOTAL DISPONIBIL",
    "SUME BLOCATE",
    "din care Fonduri proprii",
    "Credit neutilizat",
    "Acest extras de cont este valabil",
    "Tiparit:",
    "Info clienti:",
]

TRANSACTION_TRIGGERS = [
    r"plata la pos non-bt cu card mastercard",
    r"plata la pos non-bt",
    r"plata la pos",
    r"transfer intern - canal electronic",
    r"retragere de numerar",
    r"p2p btpay",
    r"p2p",
]

KEYWORDS_CREDIT = [
    "transfer",
    "p2p",
    "alimentare",
]


MERCHANT_PATTERN = re.compile(
    r"TID[:=\s](\w+)\s+(.*?)(?=\s+(valoare tranzactie|comision|RRN:|Data Descriere Debit Credit|$))",
    re.IGNORECASE | re.DOTALL
)

TID_PATTERN = re.compile(r"TID[:=\s](\w+)", re.IGNORECASE)


def normalize_amount(num_str: str) -> float:
    num_str = num_str.strip()
    if '.' in num_str and ',' in num_str:
        num_str = num_str.replace('.', '')
        num_str = num_str.replace(',', '.')
    elif ',' in num_str and '.' not in num_str:
        num_str = num_str.replace(',', '.')
    return float(num_str)


def should_exclude_line(line: str) -> bool:
    up = line.upper()
    for kw in KEYWORDS_EXCLUDE:
        if kw.upper() in up:
            return True
    return False


def is_likely_credit(fragment: str) -> bool:
    frag_lower = fragment.lower()
    for kw in KEYWORDS_CREDIT:
        if kw in frag_lower:
            return True
    return False


def clean_merchant_name(merchant: str) -> str:

    text = merchant


    text = re.sub(r"BANCA TRANSILVANIA.*", "", text, flags=re.IGNORECASE)

    text = re.sub(r"18\.02\.1999.*", "", text, flags=re.IGNORECASE)

    text = ' '.join(text.split())

    return text.strip()


def parse_description(fragment: str) -> str:
    f_lower = fragment.lower()
    if "plata la pos non-bt cu card mastercard" in f_lower:
        desc = "Plata la POS non-BT cu card MASTERCARD"
    elif "transfer intern - canal electronic" in f_lower:
        desc = "Transfer intern - canal electronic"
    elif "retragere de numerar" in f_lower:
        desc = "Retragere de numerar de la ATM"
    elif "plata la pos non-bt" in f_lower:
        desc = "Plata la POS non-BT cu card MASTERCARD"
    elif "plata la pos" in f_lower:
        desc = "Plata la POS"
    elif "p2p" in f_lower:
        desc = "P2P BTPay"
    else:
        desc = "Tranzacție"

    # Find TID + merchant
    m = MERCHANT_PATTERN.search(fragment)
    if m:
        tid_val = m.group(1)
        merchant_raw = m.group(2)
        merchant_clean = clean_merchant_name(merchant_raw)
        desc += f" - TID:{tid_val}"
        if merchant_clean:
            desc += f" - {merchant_clean}"
    else:
        tid_only = TID_PATTERN.search(fragment)
        if tid_only:
            desc += f" - TID:{tid_only.group(1)}"

    return desc


def split_into_subtransactions(full_text: str):
    text_lower = full_text.lower()
    matches = []
    for kw in TRANSACTION_TRIGGERS:
        pattern = re.compile(kw, re.IGNORECASE)
        for mm in pattern.finditer(text_lower):
            matches.append((mm.start(), mm.group(0)))
    matches.sort(key=lambda x: x[0])

    if not matches:
        amounts = AMOUNT_PATTERN.findall(full_text)
        if amounts:
            amt = normalize_amount(amounts[0])
            credit = is_likely_credit(full_text)
            return [(full_text, amt, credit)]
        else:
            return []

    sub_txs = []
    for i in range(len(matches)):
        start_idx = matches[i][0]
        if i < len(matches) - 1:
            end_idx = matches[i + 1][0]
        else:
            end_idx = len(full_text)

        fragment = full_text[start_idx:end_idx].strip()
        amounts = AMOUNT_PATTERN.findall(fragment)
        if not amounts:
            continue
        amt = normalize_amount(amounts[0])
        credit = is_likely_credit(fragment)
        sub_txs.append((fragment, amt, credit))

    return sub_txs


def extract_transactions_from_pdf(pdf_path: str) -> list:
    transactions = []

    with pdfplumber.open(pdf_path) as pdf:
        current_date = None
        buffer_lines = []

        def process_buffer(date_val, lines):
            text_for_date = " ".join(lines)
            sub_ts = split_into_subtransactions(text_for_date)
            results = []
            for (frag, amt, credit_bool) in sub_ts:
                desc = parse_description(frag)
                if credit_bool:
                    results.append({
                        "data": date_val,
                        "descriere": desc,
                        "debit": 0.0,
                        "credit": amt
                    })
                else:
                    results.append({
                        "data": date_val,
                        "descriere": desc,
                        "debit": amt,
                        "credit": 0.0
                    })
            return results

        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if should_exclude_line(line):
                    continue

                match_d = DATE_LINE_PATTERN.match(line)
                if match_d:
                    if current_date and buffer_lines:
                        transactions.extend(process_buffer(current_date, buffer_lines))
                    current_date = match_d.group(1)
                    rest_text = match_d.group(2).strip()
                    buffer_lines = []
                    if rest_text:
                        buffer_lines.append(rest_text)
                else:
                    if current_date:
                        buffer_lines.append(line)

        if current_date and buffer_lines:
            transactions.extend(process_buffer(current_date, buffer_lines))

    return transactions


def export_to_csv(transactions: list, csv_path: str) -> None:
    if not transactions:
        print("Nu există tranzacții de exportat.")
        return
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Data", "Descriere", "Debit", "Credit"])
        for tx in transactions:
            writer.writerow([
                tx["data"],
                tx["descriere"],
                f"{tx['debit']:.2f}",
                f"{tx['credit']:.2f}"
            ])
    print(f"Tranzacțiile au fost scrise în: {csv_path}")


def select_pdfs_and_save_single_csv():
    root = tk.Tk()
    root.withdraw()

    pdf_paths = filedialog.askopenfilenames(
        title="Selectează fișiere PDF",
        filetypes=[("PDF files", "*.pdf"), ("Toate fișierele", "*.*")]
    )
    if not pdf_paths:
        print("Nu ai selectat niciun fișier PDF.")
        return

    csv_path = filedialog.asksaveasfilename(
        title="Selectează locația și numele pentru fișierul CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("Toate fișierele", "*.*")]
    )
    if not csv_path:
        print("Nu ai selectat fișierul CSV de destinație.")
        return

    all_transactions = []
    for pdf_file in pdf_paths:
        txs = extract_transactions_from_pdf(pdf_file)
        all_transactions.extend(txs)

    # Sortăm opțional după dată
    def parse_date(dmy):
        dd, mm, yyyy = dmy.split('/')
        return (int(yyyy), int(mm), int(dd))

    all_transactions.sort(key=lambda t: parse_date(t["data"]))

    export_to_csv(all_transactions, csv_path)

    print("Fișiere PDF procesate:")
    for p in pdf_paths:
        print("  ", p)
    print("CSV salvat în:", csv_path)


if __name__ == "__main__":
    select_pdfs_and_save_single_csv()
