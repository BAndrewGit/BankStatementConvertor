import json
import os
import pickle
import tempfile
import unittest
import warnings
from unittest.mock import patch

from src.domain.inference_contracts import build_feature_source_map
from src.inference.model_artifact_loader import ModelArtifactLoader
from src.profile.questionnaire import (
    PROFILE_FEATURE_GROUPS,
    PROFILE_MULTI_FEATURE_GROUPS,
    PROFILE_ORDINAL_FEATURES,
    REQUIRED_NUMERIC_FEATURES,
)


class _FakeScaler:
    def transform(self, rows):
        return rows


class _FakeModel:
    def predict_proba(self, rows):
        return [[0.2, 0.8] for _ in rows]


class InconsistentVersionWarning(UserWarning):
    pass


class ModelArtifactLoaderTests(unittest.TestCase):
    def _write_bundle(self, root: str, multitask: bool = True):
        questionnaire_columns = [column for options in PROFILE_FEATURE_GROUPS.values() for column in options]
        questionnaire_columns.extend(
            [column for options in PROFILE_MULTI_FEATURE_GROUPS.values() for column in options]
        )
        questionnaire_columns.extend(list(PROFILE_ORDINAL_FEATURES.values()))
        questionnaire_columns.extend(list(REQUIRED_NUMERIC_FEATURES))
        with open(os.path.join(root, "feature_columns.json"), "w", encoding="utf-8") as handle:
            json.dump(["Expense_Distribution_Food", "Gender_Male", *questionnaire_columns], handle)

        with open(os.path.join(root, "thresholds.json"), "w", encoding="utf-8") as handle:
            json.dump({"saving_probability_threshold": 0.5, "top_k_factors": 5}, handle)

        with open(os.path.join(root, "model_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({"model_type": "multitask_net", "multitask": multitask}, handle)

        with open(os.path.join(root, "bank_mapping_rules.yaml"), "w", encoding="utf-8") as handle:
            handle.write("risk_score_high_threshold: 0.7\n")

        with open(os.path.join(root, "scaler.pkl"), "wb") as handle:
            pickle.dump(_FakeScaler(), handle)

        # Pickle fallback is supported for tests when torch artifact is not available.
        with open(os.path.join(root, "model.pt"), "wb") as handle:
            pickle.dump(_FakeModel(), handle)

    def test_loader_reads_all_artifacts_and_reports_compatibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_bundle(temp_dir)

            loader = ModelArtifactLoader(temp_dir)
            artifacts = loader.load()

            self.assertEqual(artifacts.feature_columns[0], "Expense_Distribution_Food")
            source_map = build_feature_source_map(artifacts.feature_columns)
            status = loader.compatibility_status(source_map)
            self.assertTrue(status["compatible"])
            self.assertTrue(status["is_multitask"])

    def test_loader_fails_when_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_bundle(temp_dir)
            os.remove(os.path.join(temp_dir, "model.pt"))

            loader = ModelArtifactLoader(temp_dir)
            with self.assertRaises(FileNotFoundError):
                loader.load()

    def test_loader_rejects_non_multitask_model_when_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_bundle(temp_dir, multitask=False)

            loader = ModelArtifactLoader(temp_dir)
            with self.assertRaises(ValueError):
                loader.load(require_multitask=True)

    def test_loader_fails_on_inconsistent_sklearn_warning_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_bundle(temp_dir)
            loader = ModelArtifactLoader(temp_dir)

            original_pickle_load = pickle.load

            def _warn_and_load(handle):
                warnings.warn(
                    "Trying to unpickle estimator StandardScaler from version 1.8.0 when using version 1.6.1.",
                    InconsistentVersionWarning,
                )
                return original_pickle_load(handle)

            with patch.dict("os.environ", {"ALLOW_SKLEARN_VERSION_MISMATCH": "0"}, clear=False):
                with patch("pickle.load", side_effect=_warn_and_load):
                    with self.assertRaises(ValueError) as context:
                        loader.load()

            self.assertIn("Incompatible sklearn version", str(context.exception))

    def test_loader_allows_inconsistent_sklearn_warning_with_explicit_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_bundle(temp_dir)
            loader = ModelArtifactLoader(temp_dir)

            original_pickle_load = pickle.load

            def _warn_and_load(handle):
                warnings.warn(
                    "Trying to unpickle estimator StandardScaler from version 1.8.0 when using version 1.6.1.",
                    InconsistentVersionWarning,
                )
                return original_pickle_load(handle)

            with patch.dict("os.environ", {"ALLOW_SKLEARN_VERSION_MISMATCH": "1"}, clear=False):
                with patch("pickle.load", side_effect=_warn_and_load):
                    artifacts = loader.load()

            self.assertGreater(len(artifacts.feature_columns), 0)

    def test_loader_preserves_value_error_from_joblib_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_bundle(temp_dir)
            loader = ModelArtifactLoader(temp_dir)

            with patch("pickle.load", side_effect=RuntimeError("pickle decode failed")):
                with patch(
                    "src.inference.model_artifact_loader.ModelArtifactLoader._load_with_warning_capture",
                    side_effect=[RuntimeError("pickle decode failed"), ValueError("Incompatible sklearn version for serialized scaler artifact.")],
                ):
                    with self.assertRaises(ValueError) as context:
                        loader.load()

        self.assertIn("Incompatible sklearn version", str(context.exception))

    def test_loader_rebuilds_torch_state_dict_model_artifact(self):
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self.skipTest("torch is not available in this environment")

        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_bundle(temp_dir)

            class _TinyNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.shared_trunk = nn.Sequential(
                        nn.Linear(3, 2),
                        nn.ReLU(),
                        nn.Dropout(0.0),
                        nn.Linear(2, 2),
                    )
                    self.risk_head = nn.Linear(2, 1)
                    self.savings_head = nn.Linear(2, 1)

            model = _TinyNet()
            torch.save(model.state_dict(), os.path.join(temp_dir, "model.pt"))
            with open(os.path.join(temp_dir, "model_metadata.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "model_type": "multitask_net",
                        "multitask": True,
                        "model_config": {
                            "input_dim": 3,
                            "hidden_dims": [2, 2],
                            "dropout": 0.0,
                            "activation": "relu",
                        },
                    },
                    handle,
                )
            with open(os.path.join(temp_dir, "feature_columns.json"), "w", encoding="utf-8") as handle:
                json.dump(["f1", "f2", "f3", "Gender_Male", "Age"], handle)

            loader = ModelArtifactLoader(temp_dir)
            artifacts = loader.load(require_multitask=True)
            self.assertTrue(callable(artifacts.model))


if __name__ == "__main__":
    unittest.main()





