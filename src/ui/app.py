from __future__ import annotations

import csv
import json
import os
import traceback
from typing import Dict, List, Optional

from src.domain.inference_contracts import (
    ProcessedStatementFeatures,
    ProfileAnswers,
    build_feature_source_map,
)
from src.features.feature_assembler import FeatureAssembler
from src.inference.model_artifact_loader import ModelArtifactLoader
from src.inference.predictor import Predictor
from src.memory.profile_store import ProfileStore
from src.pipelines.program2_adapter import adapt_program2_output_to_processed_features
from src.pipelines.run_end_to_end import run_end_to_end, run_end_to_end_many
from src.profile.questionnaire import (
    MULTI_SELECT_GROUPS,
    ORDINAL_CHOICE_GROUPS,
    QUESTION_GROUPS,
    map_raw_profile_inputs_to_one_hot,
    questionnaire_answers_complete,
    selected_multi_options_from_one_hot,
    selected_numeric_values_from_one_hot,
    selected_ordinal_options_from_values,
    selected_options_from_one_hot,
    validate_questionnaire_groups_against_features,
)


DEFAULT_ARTIFACTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model_artifacts")
)


def get_default_artifacts_dir() -> str:
    return DEFAULT_ARTIFACTS_DIR


def validate_artifacts_folder_status(folder: str) -> Dict[str, object]:
    candidate = os.path.abspath(folder or "")
    if not candidate or not os.path.isdir(candidate):
        return {
            "valid": False,
            "folder": candidate,
            "error": "Artifacts folder does not exist",
            "missing_artifacts": [],
        }

    loader = ModelArtifactLoader(candidate)
    missing = loader.missing_artifacts()
    if missing:
        return {
            "valid": False,
            "folder": candidate,
            "error": "Missing required artifacts",
            "missing_artifacts": missing,
        }

    try:
        artifacts = loader.load(require_multitask=False)
        compatibility = loader.compatibility_status(build_feature_source_map(artifacts.feature_columns))
    except Exception as exc:
        return {
            "valid": False,
            "folder": candidate,
            "error": str(exc),
            "missing_artifacts": [],
        }

    return {
        "valid": bool(compatibility.get("compatible", False)),
        "folder": candidate,
        "compatibility": compatibility,
        "metadata": artifacts.model_metadata,
        "missing_artifacts": [],
    }


def launch_desktop_app() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        raise RuntimeError("PySide6 is required for desktop UI. Install PySide6 first.") from exc

    # Ensure the hardcoded artifacts folder exists before UI starts.
    os.makedirs(DEFAULT_ARTIFACTS_DIR, exist_ok=True)

    app = QApplication([])
    window = _MainWindow()
    window.show()
    return app.exec()


class _MainWindow:  # pragma: no cover - covered by integration usage, not unit tests.
    def __init__(self) -> None:
        from PySide6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        self._qt_window = QMainWindow()
        self._qt_window.setWindowTitle("Procesare Extras Cont - Desktop")
        self._qt_window.resize(1180, 760)

        self._profile_store = ProfileStore()
        self._active_profile_id: Optional[str] = None
        self._active_artifacts_dir: str = ""
        self._last_run_payload: Dict[str, object] = {}
        self._current_run_thread = None
        self._current_run_worker = None

        central = QWidget()
        layout = QVBoxLayout(central)

        header = QLabel("Single shell: Profile | Questions | Statements | Results | Model Artifacts")
        layout.addWidget(header)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._profile_tab = _ProfileManagerTab(self._profile_store, self._on_profile_changed)
        self._questions_tab = _ProfileQuestionsTab(self._profile_store, self._get_active_profile_id)
        self._statements_tab = _StatementProcessorTab(self._on_run_requested, self._on_output_dir_selected)
        self._results_tab = _ResultsTab()
        self._artifacts_tab = _ModelArtifactsTab(self._on_artifacts_selected)

        self._tabs.addTab(self._profile_tab.widget, "Profile")
        self._tabs.addTab(self._questions_tab.widget, "Questions")
        self._tabs.addTab(self._statements_tab.widget, "Statements")
        self._tabs.addTab(self._results_tab.widget, "Results")
        self._tabs.addTab(self._artifacts_tab.widget, "Model Artifacts")

        quick_bar = QWidget()
        quick_layout = QHBoxLayout(quick_bar)
        self._status_label = QLabel("Ready")
        refresh_btn = QPushButton("Refresh Profile Data")
        refresh_btn.clicked.connect(self._refresh_profile_views)
        quick_layout.addWidget(self._status_label)
        quick_layout.addStretch()
        quick_layout.addWidget(refresh_btn)
        layout.addWidget(quick_bar)

        self._qt_window.setCentralWidget(central)

        self._message_box_cls = QMessageBox
        self._refresh_profile_views()
        self._artifacts_tab.set_initial_folder(self._active_artifacts_dir)

    def show(self) -> None:
        self._qt_window.show()

    def _get_active_profile_id(self) -> Optional[str]:
        return self._active_profile_id

    def _on_profile_changed(self, profile_id: Optional[str]) -> None:
        self._active_profile_id = profile_id
        self._questions_tab.reload()
        self._sync_output_dir_from_active_profile()
        self._status_label.setText(f"Active profile: {profile_id or 'none'}")

    def _on_artifacts_selected(self, artifacts_dir: str) -> None:
        self._active_artifacts_dir = artifacts_dir
        self._status_label.setText(f"Artifacts: {artifacts_dir}")

    def _refresh_profile_views(self) -> None:
        active = self._profile_store.get_active_profile()
        self._active_profile_id = active.profile_id if active else None
        self._profile_tab.reload()
        self._questions_tab.reload()
        self._sync_output_dir_from_active_profile()
        self._status_label.setText(f"Active profile: {self._active_profile_id or 'none'}")

    def _on_output_dir_selected(self, output_dir: str) -> None:
        folder = (output_dir or "").strip()
        if not folder or not self._active_profile_id:
            return

        active = self._profile_store.get_profile(self._active_profile_id)
        existing_preferences = dict(active.export_preferences) if active else {}
        existing_preferences["last_output_dir"] = folder
        self._profile_store.update_profile(self._active_profile_id, export_preferences=existing_preferences)

    def _sync_output_dir_from_active_profile(self) -> None:
        if not self._active_profile_id:
            self._statements_tab.set_output_dir("")
            return

        active = self._profile_store.get_profile(self._active_profile_id)
        if not active:
            self._statements_tab.set_output_dir("")
            return
        folder = str((active.export_preferences or {}).get("last_output_dir", "") or "").strip()
        self._statements_tab.set_output_dir(folder)

    def _on_run_requested(self, pdf_paths: List[str], output_dir: str) -> None:
        try:
            if not output_dir:
                raise ValueError("Select output folder")
            if self._current_run_thread is not None:
                raise ValueError("A processing run is already in progress")

            active_profile = self._profile_store.get_active_profile()
            if active_profile is None:
                raise ValueError("Set an active profile before Analyze")
            if not questionnaire_answers_complete(active_profile.questionnaire_answers):
                raise ValueError("Complete all 8 questionnaire answers in Questions tab before Analyze")

            self._on_output_dir_selected(output_dir)

            if not pdf_paths:
                report_path = os.path.join(output_dir, "run_report.json")
                if not os.path.exists(report_path):
                    raise ValueError("Select at least one PDF file or choose output folder with existing run_report.json")
                report_payload = self._load_existing_run_report(output_dir)
                self._handle_run_payload(report_payload)
                return

            output_strategy = self._prompt_output_strategy(output_dir)
            if output_strategy == "cancel":
                self._status_label.setText("Run cancelled")
                return
            if output_strategy == "reuse":
                report_payload = self._load_existing_run_report(output_dir)
                self._handle_run_payload(report_payload)
                return

            profile_answers = dict(active_profile.questionnaire_answers)

            self._set_processing(True)
            self._status_label.setText("Running...")

            from PySide6.QtCore import QObject, QThread, Signal

            class _RunWorker(QObject):
                finished = Signal(dict)
                failed = Signal(str)
                done = Signal()

                def __init__(
                    self,
                    paths: List[str],
                    export_path: str,
                    questionnaire_answers: Dict[str, float],
                ) -> None:
                    super().__init__()
                    self._paths = paths
                    self._export_path = export_path
                    self._questionnaire_answers = questionnaire_answers

                def run(self) -> None:
                    try:
                        if len(self._paths) == 1:
                            run_result = run_end_to_end(
                                pdf_path=self._paths[0],
                                export_dir=self._export_path,
                                profile_answers=self._questionnaire_answers,
                            )
                        else:
                            run_result = run_end_to_end_many(
                                pdf_paths=self._paths,
                                export_dir=self._export_path,
                                profile_answers=self._questionnaire_answers,
                            )

                        report_data = run_result.to_dict()
                        report_path = report_data.get("run_report_path")
                        if isinstance(report_path, str) and os.path.exists(report_path):
                            with open(report_path, encoding="utf-8") as handle:
                                report_payload = json.load(handle)
                        else:
                            report_payload = report_data

                        self.finished.emit(report_payload)
                    except Exception as exc:
                        self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=2)}")
                    finally:
                        self.done.emit()

            thread = QThread(self._qt_window)
            worker = _RunWorker(list(pdf_paths), output_dir, profile_answers)
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_run_payload)
            worker.failed.connect(self._handle_run_failure)
            worker.done.connect(thread.quit)
            worker.done.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._on_run_thread_finished)

            self._current_run_thread = thread
            self._current_run_worker = worker
            thread.start()
        except Exception as exc:
            self._message_box_cls.critical(
                self._qt_window,
                "Processing error",
                f"Run failed:\n{exc}",
            )
            self._status_label.setText("Run failed")

    def _prompt_output_strategy(self, output_dir: str) -> str:
        report_path = os.path.join(output_dir, "run_report.json")
        if not os.path.exists(report_path):
            return "overwrite"

        buttons = self._message_box_cls.StandardButton
        reply = self._message_box_cls.question(
            self._qt_window,
            "Existing output detected",
            "The selected output folder already contains run_report.json.\n\n"
            "Yes = overwrite and reprocess statements\n"
            "No = reuse existing output and run model inference\n"
            "Cancel = abort",
            buttons.Yes | buttons.No | buttons.Cancel,
            buttons.No,
        )
        if reply == buttons.Yes:
            return "overwrite"
        if reply == buttons.No:
            return "reuse"
        return "cancel"

    @staticmethod
    def _load_existing_run_report(output_dir: str) -> Dict[str, object]:
        report_path = os.path.join(output_dir, "run_report.json")
        if not os.path.exists(report_path):
            raise ValueError("Cannot reuse output: run_report.json is missing")
        with open(report_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Cannot reuse output: run_report.json is invalid")
        return payload

    def _handle_run_payload(self, report_payload: Dict[str, object]) -> None:
        self._last_run_payload = report_payload
        self._results_tab.render(report_payload)

        if self._active_profile_id:
            self._profile_store.update_profile(
                self._active_profile_id,
                last_run={
                    "run_report_path": report_payload.get("output_files", {}).get("run_report"),
                    "transaction_count": report_payload.get("run_summary", {}).get("transaction_count"),
                },
            )

        # Optional model inference after statement run if artifacts are configured.
        if self._active_artifacts_dir:
            try:
                inference_summary = self._run_model_inference(report_payload)
            except Exception as exc:
                inference_summary = f"Inference failed: {exc}"
            self._results_tab.append_text("\n\nInference summary:\n" + inference_summary)

            prediction_payload = _MainWindow._extract_prediction_payload(inference_summary)
            monthly_prediction_payloads = _MainWindow._extract_monthly_prediction_payloads(inference_summary)
            if prediction_payload or monthly_prediction_payloads:
                persist_message = _MainWindow._persist_inference_to_final_dataset(
                    report_payload,
                    prediction_payload=prediction_payload,
                    monthly_prediction_payloads=monthly_prediction_payloads,
                )
                if persist_message:
                    self._results_tab.append_text("\n" + persist_message)

        self._status_label.setText("Run finished")

    @staticmethod
    def _extract_prediction_payload(inference_summary: str) -> Optional[Dict[str, float]]:
        try:
            parsed = json.loads(inference_summary)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None

        if "risk_score" not in parsed:
            return None

        payload: Dict[str, float] = {}
        try:
            payload["risk_score"] = float(parsed["risk_score"])
            if "saving_probability" in parsed:
                payload["saving_probability"] = float(parsed["saving_probability"])
        except Exception:
            return None
        return payload

    @staticmethod
    def _extract_monthly_prediction_payloads(inference_summary: str) -> Dict[str, Dict[str, float]]:
        try:
            parsed = json.loads(inference_summary)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}

        monthly = parsed.get("monthly_predictions")
        if not isinstance(monthly, dict):
            return {}

        normalized: Dict[str, Dict[str, float]] = {}
        for month_key, raw_payload in monthly.items():
            if not isinstance(raw_payload, dict):
                continue
            try:
                month_prediction = {
                    "risk_score": float(raw_payload["risk_score"]),
                    "saving_probability": float(raw_payload["saving_probability"]),
                }
            except Exception:
                continue
            normalized[str(month_key)] = month_prediction
        return normalized

    @staticmethod
    def _persist_inference_to_final_dataset(
        report_payload: Dict[str, object],
        prediction_payload: Optional[Dict[str, float]] = None,
        monthly_prediction_payloads: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[str]:
        output_files = report_payload.get("output_files")
        if not isinstance(output_files, dict):
            return None

        final_dataset_path = output_files.get("final_dataset")
        if not isinstance(final_dataset_path, str) or not final_dataset_path.strip():
            return "Skipped persisting inference: final_dataset path missing"
        if not os.path.exists(final_dataset_path):
            return f"Skipped persisting inference: file not found ({final_dataset_path})"

        with open(final_dataset_path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            existing_columns = list(reader.fieldnames or [])

        if not existing_columns:
            return "Skipped persisting inference: final_dataset has no header"

        monthly_payloads = dict(monthly_prediction_payloads or {})
        fallback_payload = dict(prediction_payload or {})
        if not fallback_payload and not monthly_payloads:
            return None

        target_columns = set(fallback_payload.keys())
        for payload in monthly_payloads.values():
            target_columns.update(payload.keys())

        for column in sorted(target_columns):
            if column not in existing_columns:
                existing_columns.append(column)

        for row in rows:
            month_key = str(row.get("statement_month", "") or "").strip()
            row_payload = monthly_payloads.get(month_key) or fallback_payload
            for column in target_columns:
                if column in row_payload:
                    row[column] = str(row_payload[column])

        with open(final_dataset_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing_columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        persisted_columns = ", ".join(sorted(target_columns))
        if monthly_payloads:
            return f"Monthly inference persisted to final_dataset.csv: {persisted_columns}"
        return f"Inference persisted to final_dataset.csv: {persisted_columns}"

    def _handle_run_failure(self, message: str) -> None:
        self._message_box_cls.critical(
            self._qt_window,
            "Processing error",
            f"Run failed:\n{message}",
        )
        self._status_label.setText("Run failed")

    def _on_run_thread_finished(self) -> None:
        self._current_run_worker = None
        self._current_run_thread = None
        self._set_processing(False)

    def _set_processing(self, is_running: bool) -> None:
        self._statements_tab.set_processing(is_running)

    def _run_model_inference(self, report_payload: Dict[str, object]) -> str:
        raw_features = report_payload.get("features")
        if not isinstance(raw_features, dict):
            return "Skipped: report has no feature payload"

        loader = ModelArtifactLoader(self._active_artifacts_dir)
        try:
            artifacts = loader.load(require_multitask=True)
        except Exception as exc:
            return f"Skipped: model artifacts unavailable ({exc})"

        missing_groups = validate_questionnaire_groups_against_features(artifacts.feature_columns)
        if missing_groups:
            return f"Skipped: questionnaire groups missing in model schema: {missing_groups}"

        active = self._profile_store.get_active_profile()
        questionnaire_one_hot = active.questionnaire_answers if active else {}

        assembler = FeatureAssembler(feature_columns=artifacts.feature_columns)
        predictor = Predictor(artifacts)

        feature_values = _MainWindow._build_inference_feature_payload(report_payload)
        if feature_values is None:
            return "Skipped: report has no feature payload"
        assembly = assembler.assemble(
            statement_features=ProcessedStatementFeatures.from_mapping(
                adapt_program2_output_to_processed_features({"features": feature_values}).to_feature_map()
            ),
            profile_answers=ProfileAnswers.from_mapping(questionnaire_one_hot),
        )
        try:
            prediction = predictor.predict(assembly.row)
        except Exception as exc:
            return f"Skipped: model inference failed ({exc})"

        monthly_predictions: Dict[str, Dict[str, float]] = {}
        raw_features_by_month = report_payload.get("features_by_month")
        if isinstance(raw_features_by_month, dict):
            income_by_month = _MainWindow._load_income_by_month_from_final_dataset(report_payload)
            for month_key, monthly_features in raw_features_by_month.items():
                if not isinstance(monthly_features, dict):
                    continue
                income_override = income_by_month.get(str(month_key))
                monthly_payload = _MainWindow._build_inference_feature_payload(
                    {"features": monthly_features, "salary_income": report_payload.get("salary_income")},
                    income_override=income_override,
                )
                if monthly_payload is None:
                    continue
                monthly_assembly = assembler.assemble(
                    statement_features=ProcessedStatementFeatures.from_mapping(
                        adapt_program2_output_to_processed_features(
                            {"features": monthly_payload}
                        ).to_feature_map()
                    ),
                    profile_answers=ProfileAnswers.from_mapping(questionnaire_one_hot),
                )
                try:
                    monthly_prediction = predictor.predict(monthly_assembly.row)
                except Exception:
                    continue
                monthly_predictions[str(month_key)] = {
                    "risk_score": monthly_prediction.risk_score,
                    "saving_probability": monthly_prediction.saving_probability,
                }

        return json.dumps(
            {
                "risk_score": prediction.risk_score,
                "saving_probability": prediction.saving_probability,
                "alerts": prediction.alerts,
                "top_factors": prediction.top_factors,
                "monthly_predictions": monthly_predictions,
            },
            indent=2,
            ensure_ascii=False,
        )

    @staticmethod
    def _build_inference_feature_payload(
        report_payload: Dict[str, object],
        income_override: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        raw_features = report_payload.get("features")
        if not isinstance(raw_features, dict):
            return None

        merged: Dict[str, float] = {}
        for key, value in raw_features.items():
            try:
                merged[str(key)] = float(value)
            except Exception:
                continue

        if "Income_Category" in merged:
            return merged

        if income_override is not None:
            merged["Income_Category"] = float(income_override)
            return merged

        salary_total = None
        salary_income = report_payload.get("salary_income")
        if isinstance(salary_income, dict):
            salary_total = salary_income.get("salary_income_total")
        if salary_total is None:
            run_summary = report_payload.get("run_summary")
            if isinstance(run_summary, dict):
                salary_total = run_summary.get("salary_income_total")

        if salary_total is not None:
            try:
                merged["Income_Category"] = float(salary_total)
            except Exception:
                pass

        return merged

    @staticmethod
    def _load_income_by_month_from_final_dataset(report_payload: Dict[str, object]) -> Dict[str, float]:
        output_files = report_payload.get("output_files")
        if not isinstance(output_files, dict):
            return {}
        final_dataset_path = output_files.get("final_dataset")
        if not isinstance(final_dataset_path, str) or not os.path.exists(final_dataset_path):
            return {}

        by_month: Dict[str, float] = {}
        with open(final_dataset_path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                month_key = str(row.get("statement_month", "") or "").strip()
                raw_income = row.get("Income_Category")
                if not month_key or raw_income in (None, ""):
                    continue
                try:
                    by_month[month_key] = float(raw_income)
                except Exception:
                    continue
        return by_month


class _ProfileManagerTab:  # pragma: no cover
    def __init__(self, profile_store: ProfileStore, on_active_changed) -> None:
        from PySide6.QtWidgets import (
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QListWidget,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self._profile_store = profile_store
        self._on_active_changed = on_active_changed
        self._QInputDialog = QInputDialog
        self._QMessageBox = QMessageBox

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel("Profiles"))

        self._list = QListWidget()
        layout.addWidget(self._list)

        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        add_btn = QPushButton("Add")
        rename_btn = QPushButton("Edit name")
        delete_btn = QPushButton("Delete")
        activate_btn = QPushButton("Set active")

        add_btn.clicked.connect(self._add)
        rename_btn.clicked.connect(self._rename)
        delete_btn.clicked.connect(self._delete)
        activate_btn.clicked.connect(self._activate)

        actions_layout.addWidget(add_btn)
        actions_layout.addWidget(rename_btn)
        actions_layout.addWidget(delete_btn)
        actions_layout.addWidget(activate_btn)
        layout.addWidget(actions)

    def reload(self) -> None:
        self._list.clear()
        active = self._profile_store.get_active_profile()
        active_id = active.profile_id if active else None

        for profile in self._profile_store.list_profiles():
            label = f"{profile.profile_name} ({profile.profile_id})"
            if profile.profile_id == active_id:
                label = "* " + label
            self._list.addItem(label)

    def _extract_selected_profile_id(self) -> Optional[str]:
        selected = self._list.currentItem()
        if not selected:
            return None
        text = selected.text().lstrip("* ")
        start = text.rfind("(")
        end = text.rfind(")")
        if start < 0 or end < 0:
            return None
        return text[start + 1 : end]

    def _add(self) -> None:
        name, ok = self._QInputDialog.getText(self.widget, "Add profile", "Profile name")
        if not ok:
            return
        created = self._profile_store.create_profile(profile_name=name)
        self.reload()
        self._on_active_changed(created.profile_id)

    def _rename(self) -> None:
        profile_id = self._extract_selected_profile_id()
        if not profile_id:
            return
        name, ok = self._QInputDialog.getText(self.widget, "Edit profile", "Profile name")
        if not ok:
            return
        self._profile_store.update_profile(profile_id, profile_name=name)
        self.reload()

    def _delete(self) -> None:
        profile_id = self._extract_selected_profile_id()
        if not profile_id:
            return
        reply = self._QMessageBox.question(
            self.widget,
            "Delete profile",
            "Delete selected profile?",
        )
        if reply != self._QMessageBox.StandardButton.Yes:
            return
        self._profile_store.delete_profile(profile_id)
        self.reload()
        active = self._profile_store.get_active_profile()
        self._on_active_changed(active.profile_id if active else None)

    def _activate(self) -> None:
        profile_id = self._extract_selected_profile_id()
        if not profile_id:
            return
        self._profile_store.set_active_profile(profile_id)
        self.reload()
        self._on_active_changed(profile_id)


class _ProfileQuestionsTab:  # pragma: no cover
    def __init__(self, profile_store: ProfileStore, get_active_profile_id) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox,
            QGridLayout,
            QComboBox,
            QFormLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )

        self._profile_store = profile_store
        self._get_active_profile_id = get_active_profile_id
        self._QMessageBox = QMessageBox
        self._Qt = Qt

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel("Profile inputs (raw) - encoded to model one-hot automatically"))

        self._numeric_form = QFormLayout()
        self._numeric_inputs: Dict[str, QLineEdit] = {}
        age_input = QLineEdit()
        age_input.setPlaceholderText("numeric")
        self._numeric_inputs["Age"] = age_input
        self._numeric_form.addRow("Age", age_input)
        layout.addLayout(self._numeric_form)

        self._lifetime_controls: Dict[str, Dict[str, object]] = {}
        lifetime_form = QFormLayout()
        lifetime_fields = [
            ("Product_Lifetime_Clothing", "Product lifetime - clothing"),
            ("Product_Lifetime_Tech", "Product lifetime - tech"),
            ("Product_Lifetime_Appliances", "Product lifetime - appliances"),
            ("Product_Lifetime_Cars", "Product lifetime - cars"),
        ]
        for feature_name, label in lifetime_fields:
            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            slider = QSlider(self._Qt.Orientation.Horizontal)
            slider.setMinimum(1)
            slider.setMaximum(240)
            slider.setValue(12)
            status = QLabel("1 year")
            not_purchased = QCheckBox("Not purchased yet")

            def _update_status(value: int, status_label=status) -> None:
                if value < 12:
                    status_label.setText(f"{value} months")
                else:
                    years = value // 12
                    months = value % 12
                    if months == 0:
                        status_label.setText(f"{years} years")
                    else:
                        status_label.setText(f"{years} years {months} months")

            def _toggle_state(is_checked: bool, bound_slider=slider, bound_status=status) -> None:
                bound_slider.setEnabled(not is_checked)
                if is_checked:
                    bound_status.setText("Not purchased yet")
                else:
                    _update_status(bound_slider.value(), bound_status)

            slider.valueChanged.connect(_update_status)
            not_purchased.toggled.connect(_toggle_state)
            _update_status(slider.value(), status)

            row_layout.addWidget(slider)
            row_layout.addWidget(status)
            row_layout.addWidget(not_purchased)
            lifetime_form.addRow(label, row_widget)
            self._lifetime_controls[feature_name] = {
                "slider": slider,
                "status": status,
                "not_purchased": not_purchased,
            }
        layout.addLayout(lifetime_form)

        self._form_layout = QFormLayout()
        self._combos: Dict[str, QComboBox] = {}
        for group_name, options in QUESTION_GROUPS.items():
            combo = QComboBox()
            if group_name == "Savings obstacles":
                combo.addItem("")
            combo.addItems(list(options.keys()))
            self._combos[group_name] = combo
            self._form_layout.addRow(group_name, combo)
        layout.addLayout(self._form_layout)

        self._ordinal_form_layout = QFormLayout()
        self._ordinal_combos: Dict[str, QComboBox] = {}
        for group_name, options in ORDINAL_CHOICE_GROUPS.items():
            combo = QComboBox()
            combo.addItems(list(options.keys()))
            self._ordinal_combos[group_name] = combo
            self._ordinal_form_layout.addRow(group_name, combo)
        layout.addLayout(self._ordinal_form_layout)

        self._multi_layout = QVBoxLayout()
        self._multi_checks: Dict[str, Dict[str, QCheckBox]] = {}
        for group_name, options in MULTI_SELECT_GROUPS.items():
            self._multi_layout.addWidget(QLabel(group_name))
            grid = QWidget()
            grid_layout = QGridLayout(grid)
            checks_for_group: Dict[str, QCheckBox] = {}
            for index, option_label in enumerate(options.keys()):
                checkbox = QCheckBox(option_label)
                checks_for_group[option_label] = checkbox
                row = index // 2
                col = index % 2
                grid_layout.addWidget(checkbox, row, col)
            self._multi_checks[group_name] = checks_for_group
            self._multi_layout.addWidget(grid)
        layout.addLayout(self._multi_layout)

        save_btn = QPushButton("Save answers")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

    def reload(self) -> None:
        active_id = self._get_active_profile_id()
        if not active_id:
            return
        profile = self._profile_store.get_profile(active_id)
        if not profile:
            return

        selected = selected_options_from_one_hot(profile.questionnaire_answers)
        for group_name, combo in self._combos.items():
            option = selected.get(group_name)
            if option and option in QUESTION_GROUPS[group_name]:
                combo.setCurrentText(option)
            elif group_name == "Savings obstacles":
                combo.setCurrentText("")

        selected_multi = selected_multi_options_from_one_hot(profile.questionnaire_answers)
        for group_name, checks in self._multi_checks.items():
            group_selected = set(selected_multi.get(group_name, []))
            for option_label, checkbox in checks.items():
                checkbox.setChecked(option_label in group_selected)

        numeric_values = selected_numeric_values_from_one_hot(profile.questionnaire_answers)
        for feature_name, line in self._numeric_inputs.items():
            if feature_name in numeric_values:
                line.setText(str(numeric_values[feature_name]))
            else:
                line.setText("")

        for feature_name, controls in self._lifetime_controls.items():
            slider = controls["slider"]
            status = controls["status"]
            not_purchased = controls["not_purchased"]
            if feature_name in numeric_values and float(numeric_values[feature_name]) <= 0.0:
                not_purchased.setChecked(True)
                status.setText("Not purchased yet")
            else:
                not_purchased.setChecked(False)
                if feature_name in numeric_values:
                    slider.setValue(max(1, int(float(numeric_values[feature_name]))))

        ordinal_values = selected_ordinal_options_from_values(profile.questionnaire_answers)
        for group_name, combo in self._ordinal_combos.items():
            option = ordinal_values.get(group_name)
            if option and option in ORDINAL_CHOICE_GROUPS[group_name]:
                combo.setCurrentText(option)

    def _save(self) -> None:
        active_id = self._get_active_profile_id()
        if not active_id:
            self._QMessageBox.warning(self.widget, "No active profile", "Select an active profile first")
            return

        single_choice = {group: combo.currentText() for group, combo in self._combos.items()}
        multi_select = {
            group_name: [
                option_label
                for option_label, checkbox in checks.items()
                if checkbox.isChecked()
            ]
            for group_name, checks in self._multi_checks.items()
        }
        numeric: Dict[str, object] = {
            feature_name: line.text().strip()
            for feature_name, line in self._numeric_inputs.items()
        }
        for feature_name, controls in self._lifetime_controls.items():
            slider = controls["slider"]
            not_purchased = controls["not_purchased"]
            numeric[feature_name] = "not_purchased" if bool(not_purchased.isChecked()) else int(slider.value())
        ordinal_choice = {group: combo.currentText() for group, combo in self._ordinal_combos.items()}

        try:
            encoded = map_raw_profile_inputs_to_one_hot(
                {
                    "single_choice": single_choice,
                    "ordinal_choice": ordinal_choice,
                    "multi_select": multi_select,
                    "numeric": numeric,
                }
            )
        except ValueError as exc:
            self._QMessageBox.warning(self.widget, "Invalid profile data", str(exc))
            return

        self._profile_store.update_profile(active_id, questionnaire_answers=encoded)
        self._QMessageBox.information(self.widget, "Saved", "Profile data was saved and encoded for model input")


class _StatementProcessorTab:  # pragma: no cover
    def __init__(self, on_run_requested, on_output_selected=None) -> None:
        from PySide6.QtWidgets import (
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self._on_run_requested = on_run_requested
        self._on_output_selected = on_output_selected
        self._QFileDialog = QFileDialog

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel("Statements processing"))

        self._files_list = QListWidget()
        layout.addWidget(self._files_list)

        self._output_label = QLabel("Output: not selected")
        layout.addWidget(self._output_label)

        file_actions = QWidget()
        file_actions_layout = QHBoxLayout(file_actions)
        select_files_btn = QPushButton("Select PDF files")
        select_output_btn = QPushButton("Select output folder")
        run_btn = QPushButton("Process")

        select_files_btn.clicked.connect(self._select_files)
        select_output_btn.clicked.connect(self._select_output)
        run_btn.clicked.connect(self._run)

        file_actions_layout.addWidget(select_files_btn)
        file_actions_layout.addWidget(select_output_btn)
        file_actions_layout.addWidget(run_btn)
        layout.addWidget(file_actions)

        self._select_files_btn = select_files_btn
        self._select_output_btn = select_output_btn
        self._run_btn = run_btn

        self._pdf_paths: List[str] = []
        self._output_dir: str = ""

    def _select_files(self) -> None:
        files, _ = self._QFileDialog.getOpenFileNames(
            self.widget,
            "Select statements",
            "",
            "PDF files (*.pdf)",
        )
        if not files:
            return
        self._pdf_paths = files
        self._files_list.clear()
        self._files_list.addItems(files)

    def _select_output(self) -> None:
        folder = self._QFileDialog.getExistingDirectory(self.widget, "Select output folder")
        if not folder:
            return
        self.set_output_dir(folder)
        if callable(self._on_output_selected):
            self._on_output_selected(folder)

    def set_output_dir(self, folder: str) -> None:
        self._output_dir = str(folder or "")
        if self._output_dir:
            self._output_label.setText(f"Output: {self._output_dir}")
        else:
            self._output_label.setText("Output: not selected")

    def _run(self) -> None:
        self._on_run_requested(self._pdf_paths, self._output_dir)

    def set_processing(self, is_running: bool) -> None:
        self._select_files_btn.setEnabled(not is_running)
        self._select_output_btn.setEnabled(not is_running)
        self._run_btn.setEnabled(not is_running)
        if is_running:
            self._run_btn.setText("Processing...")
        else:
            self._run_btn.setText("Process")


class _ResultsTab:  # pragma: no cover
    def __init__(self) -> None:
        from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel("Results / warnings / alerts"))
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    def render(self, payload: Dict[str, object]) -> None:
        self._text.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))

    def append_text(self, text: str) -> None:
        self._text.append(text)


class _ModelArtifactsTab:  # pragma: no cover
    def __init__(self, on_artifacts_selected) -> None:
        from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

        self._on_artifacts_selected = on_artifacts_selected
        self._QFileDialog = QFileDialog
        self._QMessageBox = QMessageBox

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel("Model artifacts bundle"))

        browse_btn = QPushButton("Browse artifacts folder")
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)

        self._status = QTextEdit()
        self._status.setReadOnly(True)
        layout.addWidget(self._status)
        self._initial_folder = get_default_artifacts_dir()

    def set_initial_folder(self, folder: str) -> None:
        self._initial_folder = folder or get_default_artifacts_dir()
        self._apply_validation(self._initial_folder, show_popup=False, notify_selection=True)

    def _browse(self) -> None:
        folder = self._QFileDialog.getExistingDirectory(
            self.widget,
            "Select artifacts folder",
            self._initial_folder,
        )
        if not folder:
            # Keep deterministic hardcoded fallback when user cancels browse.
            folder = self._initial_folder
            if not os.path.isdir(folder):
                return

        self._apply_validation(folder, show_popup=True, notify_selection=True)

    def _apply_validation(self, folder: str, show_popup: bool, notify_selection: bool) -> None:
        status = validate_artifacts_folder_status(folder)
        self._initial_folder = status.get("folder") or folder

        if not status.get("valid"):
            message = status.get("error") or "Invalid artifacts bundle"
            missing = status.get("missing_artifacts") or []
            if missing:
                message = f"{message}. Missing: {missing}"
            self._status.setPlainText(message)
            if show_popup:
                self._QMessageBox.warning(self.widget, "Invalid artifacts", message)
            return

        compatibility = status.get("compatibility") or {}
        metadata = status.get("metadata") or {}
        display_payload = {
            "valid": True,
            "folder": self._initial_folder,
            "model_type": compatibility.get("model_type"),
            "is_multitask": compatibility.get("is_multitask"),
            "feature_count": compatibility.get("feature_count"),
            "metadata": metadata,
        }
        self._status.setPlainText(json.dumps(display_payload, indent=2, ensure_ascii=False))
        if notify_selection:
            self._on_artifacts_selected(str(self._initial_folder))






