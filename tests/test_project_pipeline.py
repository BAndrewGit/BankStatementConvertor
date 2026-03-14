import os
import glob
import tempfile
import unittest

from pdf_pipeline import run_full_pipeline
from pdf_pipeline.project_pipeline import build_output_paths


class ProjectPipelineTests(unittest.TestCase):
    def test_build_output_paths_uses_target_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = build_output_paths(temp_dir)

            self.assertIn("summary_json", paths)
            self.assertIn("transactions_classified_csv", paths)
            self.assertTrue(paths["summary_json"].startswith(temp_dir))
            self.assertTrue(paths["transactions_classified_csv"].startswith(temp_dir))
            self.assertEqual(os.path.basename(paths["summary_json"]), "pipeline_summary.json")

    def test_natural_entrypoint_is_exported(self):
        self.assertTrue(callable(run_full_pipeline))

    def test_legacy_story_runners_are_removed_from_root(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        legacy_runner_paths = glob.glob(os.path.join(repo_root, "run_story*.py"))
        self.assertEqual(legacy_runner_paths, [])


if __name__ == "__main__":
    unittest.main()


