import os
import tempfile
import unittest
from unittest.mock import Mock

from src.infrastructure.termene_client import TermeneClient
from src.pipelines.resolve_company_industry import resolve_company_industry


class ResolveCompanyIndustryPipelineTests(unittest.TestCase):
    def _write_datasets(self, folder: str):
        od_firme = os.path.join(folder, "od_firme.csv")
        od_caen = os.path.join(folder, "od_caen_autorizat.csv")
        n_caen = os.path.join(folder, "n_caen.csv")

        with open(od_firme, "w", encoding="utf-8", newline="") as handle:
            handle.write("DENUMIRE^CUI^COD_INMATRICULARE\n")
            handle.write("PRANZZO ESS SRL^123^J40/1/2020\n")
            handle.write("AUCHAN ROMANIA SA^456^J40/2/2020\n")

        with open(od_caen, "w", encoding="utf-8", newline="") as handle:
            handle.write("COD_INMATRICULARE^COD_CAEN_AUTORIZAT^VER_CAEN_AUTORIZAT\n")
            handle.write("J40/1/2020^5610^2\n")
            handle.write("J40/2/2020^4711^2\n")

        with open(n_caen, "w", encoding="utf-8", newline="") as handle:
            handle.write("SECTIUNEA^SUBSECTIUNEA^DIVIZIUNEA^GRUPA^CLASA^DENUMIRE^VERSIUNE_CAEN\n")
            handle.write("I^^56^561^5610^Restaurante si servicii mobile^0\n")
            handle.write("G^^47^471^4711^Comert cu amanuntul in magazine nespecializate^0\n")

        return od_firme, od_caen, n_caen

    def test_onrc_join_maps_caen_and_industry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme, od_caen, n_caen = self._write_datasets(temp_dir)
            result = resolve_company_industry(
                company_name="Pranzzo",
                od_firme_csv_path=od_firme,
                od_caen_autorizat_csv_path=od_caen,
                n_caen_csv_path=n_caen,
            )

        self.assertEqual(result.source, "onrc")
        self.assertEqual(result.company_name, "PRANZZO ESS SRL")
        self.assertEqual(result.cui, "123")
        self.assertEqual(result.cod_inmatriculare, "J40/1/2020")
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].caen_code, "5610")
        self.assertEqual(result.entries[0].industry, "Restaurante")

    def test_termene_is_used_when_onrc_has_no_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            od_firme, od_caen, n_caen = self._write_datasets(temp_dir)
            fake_client = Mock(spec=TermeneClient)
            fake_client.search_company.return_value = {
                "name": "NO MATCH SRL",
                "cui": "999",
                "nr_reg_com": "J40/9/2020",
            }

            result = resolve_company_industry(
                company_name="Companie Inexistenta",
                od_firme_csv_path=od_firme,
                od_caen_autorizat_csv_path=od_caen,
                n_caen_csv_path=n_caen,
                termene_client=fake_client,
            )

        self.assertEqual(result.source, "termene")
        self.assertEqual(result.company_name, "NO MATCH SRL")
        self.assertEqual(result.cui, "999")


if __name__ == "__main__":
    unittest.main()


