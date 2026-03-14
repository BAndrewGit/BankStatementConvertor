import tempfile
import unittest
from pathlib import Path

from pdf_pipeline.pdf_extraction import (
    PdfProcessingError,
    _build_repetitive_candidates,
    _split_page_sections,
    extract_raw_pdf_lines,
)


class Story1ExtractionTests(unittest.TestCase):
    def test_repetitive_candidates_detected_in_top_and_bottom(self):
        pages = [
            ["BANCA TRANSILVANIA", "client X", "linie body 1", "Page 1/2"],
            ["BANCA TRANSILVANIA", "client Y", "linie body 2", "Page 2/2"],
        ]

        repetitive = _build_repetitive_candidates(
            pages_lines=pages,
            top_window_size=2,
            bottom_window_size=1,
            min_occurrences=2,
        )

        self.assertIn("banca transilvania", repetitive)
        self.assertIn("page #/#", repetitive)

    def test_split_page_sections_marks_header_footer(self):
        repetitive_keys = {"header fix", "footer fix"}
        lines = ["Header Fix", "Body A", "Body B", "Footer Fix"]

        header, body, footer = _split_page_sections(
            lines=lines,
            repetitive_keys=repetitive_keys,
            top_window_size=1,
            bottom_window_size=1,
        )

        self.assertEqual(header, ["Header Fix"])
        self.assertEqual(body, ["Body A", "Body B"])
        self.assertEqual(footer, ["Footer Fix"])

    def test_missing_file_raises_clear_error(self):
        with self.assertRaises(PdfProcessingError) as context:
            extract_raw_pdf_lines("F:/2026/ProcesareExtrasCont/tests/does_not_exist.pdf")

        self.assertIn("nu exista", str(context.exception).lower())

    def test_empty_invalid_file_raises_clear_open_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_pdf = Path(tmp_dir) / "invalid.pdf"
            fake_pdf.write_text("not a real pdf", encoding="utf-8")

            with self.assertRaises(PdfProcessingError) as context:
                extract_raw_pdf_lines(str(fake_pdf))

            self.assertIn("nu pot deschide", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main()



