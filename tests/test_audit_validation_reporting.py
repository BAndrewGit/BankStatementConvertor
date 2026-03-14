import unittest

from pdf_pipeline.transaction_parsing import TransactionRaw
from pdf_pipeline.merchant_resolution import MerchantResolution
from pdf_pipeline.caen_enrichment import MerchantCaenEnriched
from pdf_pipeline.budget_categorization import BudgetCategorizedTransaction
from pdf_pipeline.audit_validation_reporting import build_story12_audit


class Story12AuditValidationReportingTests(unittest.TestCase):
    def test_marks_parse_merchant_caen_and_category_issues(self):
        raw_rows = [
            TransactionRaw(
                candidate_id="C120001",
                source_pdf="doc1.pdf",
                source_page=1,
                source_page_span="1-1",
                source_text="x",
                transaction_date_raw="01/02/2026",
                description_raw="Plata",
                amount_raw=None,
                transaction_direction_raw="outflow",
                balance_raw="100",
                currency_raw="RON",
                parse_status="parse_failed",
                parse_confidence=0.3,
                parse_warnings="missing_amount",
            )
        ]

        resolution_rows = [
            MerchantResolution(
                candidate_id="C120001",
                source_pdf="doc1.pdf",
                source_page=1,
                merchant_raw_candidate="X",
                matched_denumire=None,
                matched_cui=None,
                matched_cod_inmatriculare=None,
                matched_caen_codes="",
                match_method="no_match",
                match_score=0.2,
                match_reason="no_match",
                resolution_status="review",
            )
        ]

        caen_rows = [
            MerchantCaenEnriched(
                candidate_id="C120001",
                source_pdf="doc1.pdf",
                source_page=1,
                merchant_raw_candidate="X",
                matched_denumire=None,
                matched_cui=None,
                matched_cod_inmatriculare=None,
                match_method="no_match",
                match_score=0.2,
                match_reason="no_match",
                resolution_status="review",
                caen_code=None,
                caen_version=None,
                caen_description=None,
                caen_enrichment_status="no_company_resolved",
            )
        ]

        categorized_rows = [
            BudgetCategorizedTransaction(
                candidate_id="C120001",
                source_pdf="doc1.pdf",
                source_page=1,
                transaction_type="unknown",
                description_clean="X",
                merchant_raw_candidate="X",
                selected_caen_code=None,
                selected_caen_description=None,
                budget_category="Unknown",
                classification_reason="rule:fallback:unknown",
                classification_confidence=0.4,
                categorization_status="categorized_with_warnings",
            )
        ]

        audit_rows, report_rows, summary = build_story12_audit(
            raw_rows,
            resolution_rows,
            caen_rows,
            categorized_rows,
        )

        self.assertEqual(len(audit_rows), 1)
        audit = audit_rows[0]
        self.assertTrue(audit.review_required)
        self.assertIn("parse_failed", audit.audit_reasons)
        self.assertIn("amount_missing", audit.audit_reasons)
        self.assertIn("merchant_not_found", audit.audit_reasons)
        self.assertIn("category_unknown", audit.audit_reasons)
        self.assertIn("category_fallback", audit.audit_reasons)

        self.assertEqual(len(report_rows), 1)
        report = report_rows[0]
        self.assertEqual(report.source_pdf, "doc1.pdf")
        self.assertEqual(report.total_transactions, 1)
        self.assertEqual(report.review_required_count, 1)
        self.assertEqual(summary.review_required_count, 1)

    def test_preserves_low_match_and_caen_missing_reasons(self):
        raw_rows = [
            TransactionRaw(
                candidate_id="C120002",
                source_pdf="doc2.pdf",
                source_page=1,
                source_page_span="1-1",
                source_text="x",
                transaction_date_raw="02/02/2026",
                description_raw="Plata",
                amount_raw="50,00",
                transaction_direction_raw="outflow",
                balance_raw="100",
                currency_raw="RON",
                parse_status="parsed_with_warnings",
                parse_confidence=0.7,
                parse_warnings="",
            )
        ]

        resolution_rows = [
            MerchantResolution(
                candidate_id="C120002",
                source_pdf="doc2.pdf",
                source_page=1,
                merchant_raw_candidate="Merchant",
                matched_denumire="Merchant SRL",
                matched_cui="123",
                matched_cod_inmatriculare="J40/1/2020",
                matched_caen_codes="",
                match_method="fuzzy_name",
                match_score=0.6,
                match_reason="fuzzy_name",
                resolution_status="review",
            )
        ]

        caen_rows = [
            MerchantCaenEnriched(
                candidate_id="C120002",
                source_pdf="doc2.pdf",
                source_page=1,
                merchant_raw_candidate="Merchant",
                matched_denumire="Merchant SRL",
                matched_cui="123",
                matched_cod_inmatriculare="J40/1/2020",
                match_method="fuzzy_name",
                match_score=0.6,
                match_reason="fuzzy_name",
                resolution_status="review",
                caen_code=None,
                caen_version=None,
                caen_description=None,
                caen_enrichment_status="no_caen_for_company",
            )
        ]

        categorized_rows = [
            BudgetCategorizedTransaction(
                candidate_id="C120002",
                source_pdf="doc2.pdf",
                source_page=1,
                transaction_type="expense",
                description_clean="x",
                merchant_raw_candidate="Merchant",
                selected_caen_code=None,
                selected_caen_description=None,
                budget_category="Other",
                classification_reason="rule:fallback:expense",
                classification_confidence=0.6,
                categorization_status="categorized_with_warnings",
            )
        ]

        audit_rows, report_rows, summary = build_story12_audit(
            raw_rows,
            resolution_rows,
            caen_rows,
            categorized_rows,
            parse_confidence_threshold=0.75,
            match_score_threshold=0.75,
        )

        audit = audit_rows[0]
        self.assertIn("low_parse_confidence", audit.audit_reasons)
        self.assertIn("low_match_score", audit.audit_reasons)
        self.assertIn("caen_missing", audit.audit_reasons)
        self.assertIn("category_fallback", audit.audit_reasons)

        self.assertEqual(report_rows[0].low_match_score_count, 1)
        self.assertEqual(report_rows[0].caen_missing_count, 1)
        self.assertEqual(summary.low_match_score_count, 1)
        self.assertEqual(summary.caen_missing_count, 1)


if __name__ == "__main__":
    unittest.main()



