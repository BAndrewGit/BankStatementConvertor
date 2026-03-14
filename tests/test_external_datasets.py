import os
import tempfile
import unittest

from pdf_pipeline.external_datasets import (
    DatasetSchemaError,
    get_caen_description,
    get_company_caen_descriptions,
    lookup_od_firme_by_cod_inmatriculare,
    lookup_od_firme_by_name,
    prepare_external_datasets,
)


class Story7ExternalDatasetsTests(unittest.TestCase):
    def test_prepares_datasets_and_exposes_lookups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme_path = os.path.join(temp_dir, "od_firme.csv")
            od_caen_path = os.path.join(temp_dir, "od_caen_autorizat.csv")
            n_caen_path = os.path.join(temp_dir, "n_caen.csv")
            db_path = os.path.join(temp_dir, "story7_refs.db")

            with open(od_firme_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("DENUMIRE^CUI^COD_INMATRICULARE\n")
                handle.write("Mega Image SRL^123456^J40/1000/2020\n")
                handle.write("Lidl Discount SRL^654321^J40/2000/2020\n")

            with open(od_caen_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("COD_INMATRICULARE^COD_CAEN_AUTORIZAT^VER_CAEN_AUTORIZAT\n")
                handle.write("J40/1000/2020^4711^2\n")
                handle.write("J40/1000/2020^4719^2\n")

            with open(n_caen_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("SECTIUNEA^SUBSECTIUNEA^DIVIZIUNEA^GRUPA^CLASA^DENUMIRE^VERSIUNE_CAEN\n")
                handle.write("G^^47^471^4711^Comert cu amanuntul in magazine nespecializate^2\n")
                handle.write("G^^47^471^4719^Comert cu amanuntul in alte magazine nespecializate^2\n")

            summary = prepare_external_datasets(od_firme_path, od_caen_path, n_caen_path, db_path)

            self.assertEqual(summary.od_firme.rows_inserted, 2)
            self.assertEqual(summary.od_caen_autorizat.rows_inserted, 2)
            self.assertEqual(summary.n_caen.rows_inserted, 2)

            by_code = lookup_od_firme_by_cod_inmatriculare(db_path, "j40/1000/2020")
            self.assertEqual(len(by_code), 1)
            self.assertEqual(by_code[0]["denumire"], "Mega Image SRL")

            by_name = lookup_od_firme_by_name(db_path, "mega image")
            self.assertEqual(len(by_name), 1)
            self.assertEqual(by_name[0]["cod_inmatriculare"], "J40/1000/2020")

            self.assertEqual(
                get_caen_description(db_path, "4711"),
                "Comert cu amanuntul in magazine nespecializate",
            )

            company_caens = get_company_caen_descriptions(db_path, "J40/1000/2020")
            self.assertEqual(len(company_caens), 2)
            self.assertEqual(company_caens[0]["cod_caen_autorizat"], "4711")
            self.assertIn("Comert", company_caens[0]["caen_description"])

    def test_fails_when_required_columns_are_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme_path = os.path.join(temp_dir, "od_firme.csv")
            od_caen_path = os.path.join(temp_dir, "od_caen_autorizat.csv")
            n_caen_path = os.path.join(temp_dir, "n_caen.csv")
            db_path = os.path.join(temp_dir, "story7_refs.db")

            with open(od_firme_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("DENUMIRE^CUI\n")
                handle.write("Mega Image SRL^123456\n")

            with open(od_caen_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("COD_INMATRICULARE^COD_CAEN_AUTORIZAT^VER_CAEN_AUTORIZAT\n")
                handle.write("J40/1000/2020^4711^2\n")

            with open(n_caen_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("SECTIUNEA^SUBSECTIUNEA^DIVIZIUNEA^GRUPA^CLASA^DENUMIRE^VERSIUNE_CAEN\n")
                handle.write("G^^47^471^4711^Comert cu amanuntul in magazine nespecializate^2\n")

            with self.assertRaises(DatasetSchemaError):
                prepare_external_datasets(od_firme_path, od_caen_path, n_caen_path, db_path)


if __name__ == "__main__":
    unittest.main()



