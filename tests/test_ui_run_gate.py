import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

from src.ui.app import _MainWindow, validate_artifacts_folder_status


class _DummyMessageBox:
    def __init__(self) -> None:
        self.critical = Mock()
        self.question = Mock()

        class _Buttons:
            Yes = 1
            No = 2
            Cancel = 4

        self.StandardButton = _Buttons


class _DummyProfileStore:
    def __init__(self, active_profile) -> None:
        self._active_profile = active_profile

    def get_active_profile(self):
        return self._active_profile


class DesktopRunGateTests(unittest.TestCase):
    def test_run_is_blocked_when_no_active_profile(self):
        message_box = _DummyMessageBox()
        status_label = Mock()
        fake_window = SimpleNamespace(
            _current_run_thread=None,
            _profile_store=_DummyProfileStore(active_profile=None),
            _set_processing=Mock(),
            _status_label=status_label,
            _message_box_cls=message_box,
            _qt_window=object(),
        )

        typed_window = cast(_MainWindow, cast(object, fake_window))
        _MainWindow._on_run_requested(typed_window, [r"C:\in\statement.pdf"], r"C:\out")

        fake_window._set_processing.assert_not_called()
        message_box.critical.assert_called_once()
        status_label.setText.assert_called_with("Run failed")

    def test_run_is_blocked_when_questionnaire_is_incomplete(self):
        incomplete_profile = SimpleNamespace(
            questionnaire_answers={
                "Gender_Male": 1.0,
                "Gender_Female": 0.0,
            }
        )
        message_box = _DummyMessageBox()
        status_label = Mock()
        fake_window = SimpleNamespace(
            _current_run_thread=None,
            _profile_store=_DummyProfileStore(active_profile=incomplete_profile),
            _set_processing=Mock(),
            _status_label=status_label,
            _message_box_cls=message_box,
            _qt_window=object(),
        )

        typed_window = cast(_MainWindow, cast(object, fake_window))
        _MainWindow._on_run_requested(typed_window, [r"C:\in\statement.pdf"], r"C:\out")

        fake_window._set_processing.assert_not_called()
        message_box.critical.assert_called_once()
        status_label.setText.assert_called_with("Run failed")

    def test_validate_artifacts_folder_status_returns_invalid_instead_of_crashing(self):
        with patch("src.ui.app.os.path.isdir", return_value=True):
            with patch("src.ui.app.ModelArtifactLoader.missing_artifacts", return_value=[]):
                with patch(
                    "src.ui.app.ModelArtifactLoader.load",
                    side_effect=ValueError("Incompatible sklearn version for serialized scaler artifact."),
                ):
                    status = validate_artifacts_folder_status(r"F:\2026\ProcesareExtrasCont\model_artifacts")

        self.assertFalse(status["valid"])
        self.assertIn("Incompatible sklearn version", str(status.get("error", "")))

    def test_handle_run_payload_keeps_ui_alive_when_inference_fails(self):
        results_tab = Mock()
        status_label = Mock()
        fake_window = SimpleNamespace(
            _last_run_payload={},
            _results_tab=results_tab,
            _active_profile_id=None,
            _active_artifacts_dir=r"F:\2026\ProcesareExtrasCont\model_artifacts",
            _run_model_inference=Mock(side_effect=RuntimeError("model failed")),
            _status_label=status_label,
            _profile_store=Mock(),
        )

        typed_window = cast(_MainWindow, cast(object, fake_window))
        _MainWindow._handle_run_payload(typed_window, {"features": {"x": 1.0}})

        results_tab.render.assert_called_once()
        results_tab.append_text.assert_called_once()
        self.assertIn("Inference failed", results_tab.append_text.call_args[0][0])
        status_label.setText.assert_called_with("Run finished")

    def test_prompt_output_strategy_returns_reuse_when_user_selects_no(self):
        message_box = _DummyMessageBox()
        message_box.question.return_value = message_box.StandardButton.No
        fake_window = SimpleNamespace(_message_box_cls=message_box, _qt_window=object())

        with patch("src.ui.app.os.path.exists", return_value=True):
            typed_window = cast(_MainWindow, cast(object, fake_window))
            strategy = _MainWindow._prompt_output_strategy(typed_window, r"C:\out")

        self.assertEqual(strategy, "reuse")

    def test_prompt_output_strategy_returns_cancel_when_user_selects_cancel(self):
        message_box = _DummyMessageBox()
        message_box.question.return_value = message_box.StandardButton.Cancel
        fake_window = SimpleNamespace(_message_box_cls=message_box, _qt_window=object())

        with patch("src.ui.app.os.path.exists", return_value=True):
            typed_window = cast(_MainWindow, cast(object, fake_window))
            strategy = _MainWindow._prompt_output_strategy(typed_window, r"C:\out")

        self.assertEqual(strategy, "cancel")

    def test_load_existing_run_report_raises_when_missing(self):
        with patch("src.ui.app.os.path.exists", return_value=False):
            with self.assertRaises(ValueError):
                _MainWindow._load_existing_run_report(r"C:\out")

    def test_run_without_pdf_reuses_existing_report_from_output_folder(self):
        complete_profile = SimpleNamespace(questionnaire_answers={"ok": 1.0})
        message_box = _DummyMessageBox()
        status_label = Mock()
        fake_window = SimpleNamespace(
            _current_run_thread=None,
            _profile_store=_DummyProfileStore(active_profile=complete_profile),
            _set_processing=Mock(),
            _status_label=status_label,
            _message_box_cls=message_box,
            _qt_window=object(),
            _on_output_dir_selected=Mock(),
            _load_existing_run_report=Mock(return_value={"features": {"x": 1.0}}),
            _handle_run_payload=Mock(),
        )

        with patch("src.ui.app.questionnaire_answers_complete", return_value=True):
            with patch("src.ui.app.os.path.exists", return_value=True):
                typed_window = cast(_MainWindow, cast(object, fake_window))
                _MainWindow._on_run_requested(typed_window, [], r"C:\out")

        fake_window._load_existing_run_report.assert_called_once_with(r"C:\out")
        fake_window._handle_run_payload.assert_called_once()
        fake_window._set_processing.assert_not_called()
        message_box.critical.assert_not_called()

    def test_profile_change_restores_last_output_folder(self):
        profile = SimpleNamespace(export_preferences={"last_output_dir": r"C:\existing"})
        statements_tab = SimpleNamespace(set_output_dir=Mock())
        status_label = Mock()
        fake_window = SimpleNamespace(
            _active_profile_id=None,
            _questions_tab=SimpleNamespace(reload=Mock()),
            _statements_tab=statements_tab,
            _status_label=status_label,
            _profile_store=SimpleNamespace(get_profile=Mock(return_value=profile)),
            _sync_output_dir_from_active_profile=Mock(),
        )

        typed_window = cast(_MainWindow, cast(object, fake_window))
        _MainWindow._on_profile_changed(typed_window, "profile-1")

        fake_window._sync_output_dir_from_active_profile.assert_called_once_with()
        fake_window._questions_tab.reload.assert_called_once_with()

    def test_extract_prediction_payload_returns_values_for_valid_json(self):
        payload = _MainWindow._extract_prediction_payload('{"risk_score": 0.31, "saving_probability": 0.69}')
        self.assertIsNotNone(payload)
        self.assertEqual(payload["risk_score"], 0.31)
        self.assertEqual(payload["saving_probability"], 0.69)

        self.assertIsNone(_MainWindow._extract_prediction_payload("Skipped: model artifacts unavailable"))

    def test_persist_inference_to_final_dataset_writes_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "final_dataset.csv")
            with open(dataset_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["f1", "f2"])
                writer.writeheader()
                writer.writerow({"f1": "1", "f2": "2"})

            report_payload = {"output_files": {"final_dataset": dataset_path}}
            message = _MainWindow._persist_inference_to_final_dataset(
                report_payload,
                {"risk_score": 0.42, "saving_probability": 0.58},
            )

            self.assertIn("Inference persisted", str(message))
            with open(dataset_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["risk_score"], "0.42")
            self.assertEqual(rows[0]["saving_probability"], "0.58")

    def test_build_inference_feature_payload_injects_income_category_from_salary(self):
        payload = _MainWindow._build_inference_feature_payload(
            {
                "features": {"Expense_Distribution_Food": 12.0},
                "salary_income": {"salary_income_total": 5773.0},
            }
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["Expense_Distribution_Food"], 12.0)
        self.assertEqual(payload["Income_Category"], 5773.0)

    def test_build_inference_feature_payload_returns_none_without_features(self):
        self.assertIsNone(_MainWindow._build_inference_feature_payload({"run_summary": {}}))

    def test_extract_monthly_prediction_payloads_reads_valid_entries(self):
        monthly = _MainWindow._extract_monthly_prediction_payloads(
            '{"monthly_predictions": {"2026-02": {"risk_score": 0.11, "saving_probability": 0.89}, "bad": {}}}'
        )
        self.assertIn("2026-02", monthly)
        self.assertEqual(monthly["2026-02"]["risk_score"], 0.11)
        self.assertEqual(monthly["2026-02"]["saving_probability"], 0.89)
        self.assertNotIn("bad", monthly)

    def test_persist_inference_to_final_dataset_writes_monthly_values_by_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "final_dataset.csv")
            with open(dataset_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["statement_month", "Income_Category"])
                writer.writeheader()
                writer.writerow({"statement_month": "2026-02", "Income_Category": "5773"})
                writer.writerow({"statement_month": "2026-03", "Income_Category": "5773"})

            report_payload = {"output_files": {"final_dataset": dataset_path}}
            message = _MainWindow._persist_inference_to_final_dataset(
                report_payload,
                prediction_payload={"risk_score": 0.5, "saving_probability": 0.5},
                monthly_prediction_payloads={
                    "2026-02": {"risk_score": 0.2, "saving_probability": 0.8},
                    "2026-03": {"risk_score": 0.7, "saving_probability": 0.3},
                },
            )

            self.assertIn("Monthly inference persisted", str(message))
            with open(dataset_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["risk_score"], "0.2")
            self.assertEqual(rows[0]["saving_probability"], "0.8")
            self.assertEqual(rows[1]["risk_score"], "0.7")
            self.assertEqual(rows[1]["saving_probability"], "0.3")


if __name__ == "__main__":
    unittest.main()




