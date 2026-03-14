import os
import tempfile
import unittest

from pdf_pipeline.description_normalization import TransactionDescriptionNormalized
from pdf_pipeline.external_datasets import prepare_external_datasets
from pdf_pipeline.merchant_resolution import (
    load_merchant_aliases_csv,
    resolve_merchants_to_onrc,
)


class Story8MerchantOnrcResolutionTests(unittest.TestCase):
    def test_alias_exact_priority_over_exact_and_fuzzy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = _prepare_reference_db(temp_dir)

            alias_path = os.path.join(temp_dir, "merchant_aliases.csv")
            with open(alias_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("ALIAS,COD_INMATRICULARE,CUI,NOTES\n")
                handle.write("MEGA SHOP,J40/1000/2020,123456,manual alias\n")

            inserted = load_merchant_aliases_csv(db_path, alias_path)
            self.assertEqual(inserted, 1)

            input_rows = [
                _desc_row("C80001", "MEGA SHOP"),
                _desc_row("C80002", "LIDL DISCOUNT SRL"),
            ]

            rows, summary = resolve_merchants_to_onrc(input_rows, db_path)
            by_id = {item.candidate_id: item for item in rows}

            self.assertEqual(by_id["C80001"].match_method, "alias_exact")
            self.assertEqual(by_id["C80001"].match_score, 1.0)
            self.assertEqual(by_id["C80001"].matched_cod_inmatriculare, "J40/1000/2020")
            self.assertIn("4711", by_id["C80001"].matched_caen_codes)

            self.assertEqual(by_id["C80002"].match_method, "exact_name")
            self.assertEqual(by_id["C80002"].resolution_status, "accepted")
            self.assertEqual(summary.alias_exact, 1)
            self.assertEqual(summary.exact_name, 1)

    def test_fuzzy_fallback_and_review_thresholds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = _prepare_reference_db(temp_dir)

            input_rows = [
                _desc_row("C80003", "MEGA IMAG SRL"),
                _desc_row("C80004", None),
            ]

            rows, summary = resolve_merchants_to_onrc(
                input_rows,
                db_path,
                accept_threshold=0.98,
                review_threshold=0.7,
                fuzzy_max_candidates=50,
            )
            by_id = {item.candidate_id: item for item in rows}

            self.assertEqual(by_id["C80003"].match_method, "fuzzy_name")
            self.assertGreaterEqual(by_id["C80003"].match_score, 0.7)
            self.assertEqual(by_id["C80003"].resolution_status, "review")

            self.assertEqual(by_id["C80004"].match_method, "no_match")
            self.assertEqual(by_id["C80004"].resolution_status, "review")
            self.assertEqual(summary.review, 2)
            self.assertEqual(summary.no_match, 1)


def _prepare_reference_db(temp_dir: str) -> str:
    od_firme_path = os.path.join(temp_dir, "od_firme.csv")
    od_caen_path = os.path.join(temp_dir, "od_caen_autorizat.csv")
    n_caen_path = os.path.join(temp_dir, "n_caen.csv")
    db_path = os.path.join(temp_dir, "story7_refs.db")

    with open(od_firme_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("DENUMIRE^CUI^COD_INMATRICULARE\n")
        handle.write("MEGA IMAGE SRL^123456^J40/1000/2020\n")
        handle.write("LIDL DISCOUNT SRL^654321^J40/2000/2020\n")

    with open(od_caen_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("COD_INMATRICULARE^COD_CAEN_AUTORIZAT^VER_CAEN_AUTORIZAT\n")
        handle.write("J40/1000/2020^4711^2\n")
        handle.write("J40/2000/2020^4719^2\n")

    with open(n_caen_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("SECTIUNEA^SUBSECTIUNEA^DIVIZIUNEA^GRUPA^CLASA^DENUMIRE^VERSIUNE_CAEN\n")
        handle.write("G^^47^471^4711^Comert cu amanuntul in magazine nespecializate^2\n")
        handle.write("G^^47^471^4719^Comert cu amanuntul in alte magazine nespecializate^2\n")

    prepare_external_datasets(od_firme_path, od_caen_path, n_caen_path, db_path)
    return db_path


def _desc_row(candidate_id: str, merchant: str | None) -> TransactionDescriptionNormalized:
    return TransactionDescriptionNormalized(
        candidate_id=candidate_id,
        source_pdf="extras.pdf",
        source_page=1,
        description_clean="PAYMENT",
        merchant_raw_candidate=merchant,
        description_normalization_confidence=0.9,
        description_status="description_ok",
        description_warnings="",
    )


if __name__ == "__main__":
    unittest.main()



