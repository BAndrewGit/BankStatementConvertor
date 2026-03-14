import os
import tempfile
import unittest

from pdf_pipeline.external_datasets import prepare_external_datasets
from pdf_pipeline.merchant_resolution import MerchantResolution
from pdf_pipeline.caen_enrichment import enrich_resolved_merchants_with_caen


class Story9CaenEnrichmentTests(unittest.TestCase):
    def test_attaches_multiple_caen_codes_for_resolved_company(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = _prepare_reference_db(temp_dir)
            input_rows = [
                MerchantResolution(
                    candidate_id="C90001",
                    source_pdf="extras.pdf",
                    source_page=1,
                    merchant_raw_candidate="MEGA IMAGE",
                    matched_denumire="MEGA IMAGE SRL",
                    matched_cui="123456",
                    matched_cod_inmatriculare="J40/1000/2020",
                    matched_caen_codes="",
                    match_method="exact_name",
                    match_score=0.99,
                    match_reason="exact_name",
                    resolution_status="accepted",
                )
            ]

            rows, summary = enrich_resolved_merchants_with_caen(input_rows, db_path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(summary.total_input_rows, 1)
            self.assertEqual(summary.total_output_rows, 2)
            self.assertEqual(summary.rows_with_caen, 2)
            self.assertEqual(summary.unique_caen_codes, 2)

            codes = [row.caen_code for row in rows]
            self.assertEqual(codes, ["4711", "4719"])
            self.assertTrue(all(row.caen_version == "2" for row in rows))
            self.assertTrue(all("Comert" in (row.caen_description or "") for row in rows))

    def test_unresolved_company_does_not_block_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = _prepare_reference_db(temp_dir)
            input_rows = [
                MerchantResolution(
                    candidate_id="C90002",
                    source_pdf="extras.pdf",
                    source_page=1,
                    merchant_raw_candidate="UNKNOWN SHOP",
                    matched_denumire=None,
                    matched_cui=None,
                    matched_cod_inmatriculare=None,
                    matched_caen_codes="",
                    match_method="no_match",
                    match_score=0.0,
                    match_reason="no_match",
                    resolution_status="review",
                )
            ]

            rows, summary = enrich_resolved_merchants_with_caen(input_rows, db_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].caen_enrichment_status, "no_company_resolved")
            self.assertIsNone(rows[0].caen_code)
            self.assertEqual(summary.unresolved_companies, 1)
            self.assertEqual(summary.rows_without_caen, 1)


def _prepare_reference_db(temp_dir: str) -> str:
    od_firme_path = os.path.join(temp_dir, "od_firme.csv")
    od_caen_path = os.path.join(temp_dir, "od_caen_autorizat.csv")
    n_caen_path = os.path.join(temp_dir, "n_caen.csv")
    db_path = os.path.join(temp_dir, "story7_refs.db")

    with open(od_firme_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("DENUMIRE^CUI^COD_INMATRICULARE\n")
        handle.write("MEGA IMAGE SRL^123456^J40/1000/2020\n")

    with open(od_caen_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("COD_INMATRICULARE^COD_CAEN_AUTORIZAT^VER_CAEN_AUTORIZAT\n")
        handle.write("J40/1000/2020^4711^2\n")
        handle.write("J40/1000/2020^4719^2\n")

    with open(n_caen_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("SECTIUNEA^SUBSECTIUNEA^DIVIZIUNEA^GRUPA^CLASA^DENUMIRE^VERSIUNE_CAEN\n")
        handle.write("G^^47^471^4711^Comert cu amanuntul in magazine nespecializate^2\n")
        handle.write("G^^47^471^4719^Comert cu amanuntul in alte magazine nespecializate^2\n")

    prepare_external_datasets(od_firme_path, od_caen_path, n_caen_path, db_path)
    return db_path


if __name__ == "__main__":
    unittest.main()



