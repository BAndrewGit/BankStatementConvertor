import os
import tempfile
import unittest

from pdf_pipeline.budget_categorization import (
    load_caen_to_budget_category_csv,
    load_merchant_budget_aliases_csv,
    map_transactions_to_budget_categories,
)
from pdf_pipeline.transaction_classification import TransactionClassified
from pdf_pipeline.caen_enrichment import MerchantCaenEnriched


class Story10BudgetCategorizationTests(unittest.TestCase):
    def test_priority_type_then_alias_then_caen_then_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "story10.db")
            _write_caen_map_csv(os.path.join(temp_dir, "caen_map.csv"))
            _write_alias_map_csv(os.path.join(temp_dir, "alias_map.csv"))

            load_caen_to_budget_category_csv(db_path, os.path.join(temp_dir, "caen_map.csv"))
            load_merchant_budget_aliases_csv(db_path, os.path.join(temp_dir, "alias_map.csv"))

            classified_rows = [
                _classified("C100001", "fee", "COMISION ADMINISTRARE", "UBER"),
                _classified("C100002", "expense", "PLATA LUNARA", "NETFLIX"),
                _classified("C100003", "expense", "ACHIZITIE SERVICII", "NO_ALIAS"),
                _classified("C100004", "expense", "FARA INDICII", None),
            ]

            caen_rows = [
                _caen_row("C100003", "6201", "Activitati de realizare a soft-ului"),
                _caen_row("C100003", "4711", "Comert alimentar"),
            ]

            rows, summary = map_transactions_to_budget_categories(
                classified_rows,
                caen_rows,
                db_path=db_path,
            )
            by_id = {row.candidate_id: row for row in rows}

            self.assertEqual(by_id["C100001"].budget_category, "Fees")
            self.assertIn("rule:type", by_id["C100001"].classification_reason)

            self.assertEqual(by_id["C100002"].budget_category, "Entertainment")
            self.assertIn("merchant_alias", by_id["C100002"].classification_reason)

            self.assertEqual(by_id["C100003"].budget_category, "Food")
            self.assertEqual(by_id["C100003"].selected_caen_code, "4711")
            self.assertIn("rule:caen", by_id["C100003"].classification_reason)

            self.assertEqual(by_id["C100004"].budget_category, "Other")
            self.assertTrue(by_id["C100004"].classification_reason.startswith("rule:fallback:"))

            self.assertEqual(summary.via_type_rules, 1)
            self.assertEqual(summary.via_alias_rules, 1)
            self.assertEqual(summary.via_caen_rules, 1)
            self.assertEqual(summary.via_fallback, 1)

    def test_direct_description_rule_beats_caen(self):
        classified_rows = [
            _classified("C100005", "expense", "PLATA UBER CURSA", "NO_ALIAS"),
        ]
        caen_rows = [
            _caen_row("C100005", "4711", "Comert alimentar"),
        ]

        rows, summary = map_transactions_to_budget_categories(classified_rows, caen_rows)

        self.assertEqual(rows[0].budget_category, "Transport")
        self.assertIn("rule:description", rows[0].classification_reason)
        self.assertEqual(rows[0].selected_caen_code, None)
        self.assertEqual(summary.transport, 1)


def _classified(candidate_id: str, tx_type: str, description: str, merchant: str | None) -> TransactionClassified:
    return TransactionClassified(
        candidate_id=candidate_id,
        source_pdf="extras.pdf",
        source_page=1,
        description_clean=description,
        merchant_raw_candidate=merchant,
        direction="out",
        amount=100.0,
        transaction_type=tx_type,
        classification_confidence=0.9,
        classification_reason="seed",
        classification_status="classified_ok",
    )


def _caen_row(candidate_id: str, code: str, description: str) -> MerchantCaenEnriched:
    return MerchantCaenEnriched(
        candidate_id=candidate_id,
        source_pdf="extras.pdf",
        source_page=1,
        merchant_raw_candidate="X",
        matched_denumire="Firma",
        matched_cui="123",
        matched_cod_inmatriculare="J40/1/2020",
        match_method="exact_name",
        match_score=0.99,
        match_reason="exact",
        resolution_status="accepted",
        caen_code=code,
        caen_version="2",
        caen_description=description,
        caen_enrichment_status="caen_attached",
    )


def _write_caen_map_csv(path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("COD_CAEN,BUDGET_CATEGORY,NOTES\n")
        handle.write("4711,Food,retail alimentar\n")


def _write_alias_map_csv(path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("ALIAS,BUDGET_CATEGORY,NOTES\n")
        handle.write("NETFLIX,Entertainment,subscription\n")


if __name__ == "__main__":
    unittest.main()



