import os
import tempfile
import unittest

from src.classification.dataset_merchant_matcher import DatasetMerchantMatcher
from src.pipelines.resolve_company_industry import resolve_company_industry


class DatasetBomHeaderTests(unittest.TestCase):
    def test_bom_header_is_handled_for_dataset_matching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme = os.path.join(temp_dir, "od_firme.csv")
            with open(od_firme, "w", encoding="utf-8-sig", newline="") as handle:
                handle.write("DENUMIRE^CUI^COD_INMATRICULARE\n")
                handle.write("AUCHAN ROMANIA SA^456^J40/2/2020\n")

            matcher = DatasetMerchantMatcher(od_firme_csv_path=od_firme)
            match, score = matcher.find_best_match("AUCHAN")

        self.assertEqual(match, "AUCHAN ROMANIA SA")
        self.assertGreaterEqual(score, 70.0)

    def test_resolver_handles_bom_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme = os.path.join(temp_dir, "od_firme.csv")
            od_caen = os.path.join(temp_dir, "od_caen_autorizat.csv")
            n_caen = os.path.join(temp_dir, "n_caen.csv")

            with open(od_firme, "w", encoding="utf-8-sig", newline="") as handle:
                handle.write("DENUMIRE^CUI^COD_INMATRICULARE\n")
                handle.write("AUCHAN ROMANIA SA^456^J40/2/2020\n")
            with open(od_caen, "w", encoding="utf-8-sig", newline="") as handle:
                handle.write("COD_INMATRICULARE^COD_CAEN_AUTORIZAT^VER_CAEN_AUTORIZAT\n")
                handle.write("J40/2/2020^4711^2\n")
            with open(n_caen, "w", encoding="utf-8-sig", newline="") as handle:
                handle.write("SECTIUNEA^SUBSECTIUNEA^DIVIZIUNEA^GRUPA^CLASA^DENUMIRE^VERSIUNE_CAEN\n")
                handle.write("G^^47^471^4711^Comert cu amanuntul in magazine nespecializate^0\n")

            result = resolve_company_industry(
                "AUCHAN",
                od_firme_csv_path=od_firme,
                od_caen_autorizat_csv_path=od_caen,
                n_caen_csv_path=n_caen,
            )

        self.assertEqual(result.source, "onrc")
        self.assertEqual(result.company_name, "AUCHAN ROMANIA SA")
        self.assertEqual(result.cui, "456")
        self.assertEqual(result.entries[0].caen_code, "4711")


if __name__ == "__main__":
    unittest.main()


