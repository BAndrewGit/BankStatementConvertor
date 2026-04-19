from __future__ import annotations

import csv
import json
import logging
import os
import traceback
from typing import Dict, List, Optional

from src.domain.inference_contracts import (
    ProcessedStatementFeatures,
    ProfileAnswers,
    build_feature_source_map,
)
from src.features.feature_assembler import FeatureAssembler
from src.features.feature_builder import classify_behavior_risk_level
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
_DESKTOP_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "desktop_ui_runtime.log")
)


def _desktop_logger() -> logging.Logger:
    logger = logging.getLogger("desktop_ui")
    if logger.handlers:
        return logger

    os.makedirs(os.path.dirname(_DESKTOP_LOG_PATH), exist_ok=True)
    handler = logging.FileHandler(_DESKTOP_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


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
        from PySide6.QtCore import qInstallMessageHandler
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        raise RuntimeError("PySide6 is required for desktop UI. Install PySide6 first.") from exc

    logger = _desktop_logger()

    # Ensure the hardcoded artifacts folder exists before UI starts.
    os.makedirs(DEFAULT_ARTIFACTS_DIR, exist_ok=True)

    app = QApplication([])

    # Capture Qt warnings/errors in a deterministic log file for post-mortem debugging.
    def _qt_message_handler(_msg_type, _context, message):
        logger.error("Qt message: %s", message)

    qInstallMessageHandler(_qt_message_handler)
    app.aboutToQuit.connect(lambda: logger.info("Qt event: aboutToQuit"))

    window = _MainWindow()
    window.show()
    logger.info("Desktop UI started")
    exit_code = app.exec()
    logger.info("Desktop UI finished with exit code %s", exit_code)
    return exit_code


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
        self._current_run_bridge = None

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

            from PySide6.QtCore import QObject, QThread, Signal, Slot

            class _RunWorker(QObject):
                finished = Signal(dict)
                failed = Signal(str)
                done = Signal()

                def __init__(
                    self,
                    paths: List[str],
                    export_path: str,
                    questionnaire_answers: Dict[str, float],
                    artifacts_dir: str,
                ) -> None:
                    super().__init__()
                    self._paths = paths
                    self._export_path = export_path
                    self._questionnaire_answers = questionnaire_answers
                    self._artifacts_dir = artifacts_dir

                def run(self) -> None:
                    try:
                        if len(self._paths) == 1:
                            run_result = run_end_to_end(
                                pdf_path=self._paths[0],
                                export_dir=self._export_path,
                                profile_answers=self._questionnaire_answers,
                                artifacts_dir=self._artifacts_dir or None,
                            )
                        else:
                            run_result = run_end_to_end_many(
                                pdf_paths=self._paths,
                                export_dir=self._export_path,
                                profile_answers=self._questionnaire_answers,
                                artifacts_dir=self._artifacts_dir or None,
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

            class _UiBridge(QObject):
                finished_on_ui = Signal(dict)
                failed_on_ui = Signal(str)

                @Slot(dict)
                def forward_finished(self, payload: dict) -> None:
                    self.finished_on_ui.emit(payload)

                @Slot(str)
                def forward_failed(self, message: str) -> None:
                    self.failed_on_ui.emit(message)

            thread = QThread(self._qt_window)
            worker = _RunWorker(
                list(pdf_paths),
                output_dir,
                profile_answers,
                getattr(self, "_active_artifacts_dir", ""),
            )
            bridge = _UiBridge(self._qt_window)
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.finished.connect(bridge.forward_finished)
            worker.failed.connect(bridge.forward_failed)
            bridge.finished_on_ui.connect(self._handle_run_payload)
            bridge.failed_on_ui.connect(self._handle_run_failure)
            worker.done.connect(thread.quit)
            worker.done.connect(worker.deleteLater)
            thread.finished.connect(bridge.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._on_run_thread_finished)

            self._current_run_thread = thread
            self._current_run_worker = worker
            self._current_run_bridge = bridge
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
        try:
            self._last_run_payload = report_payload
            self._results_tab.render(report_payload)

            if self._active_profile_id:
                output_files = report_payload.get("output_files")
                run_summary = report_payload.get("run_summary")
                run_report_path = output_files.get("run_report") if isinstance(output_files, dict) else None
                transaction_count = run_summary.get("transaction_count") if isinstance(run_summary, dict) else None
                self._profile_store.update_profile(
                    self._active_profile_id,
                    last_run={
                        "run_report_path": run_report_path,
                        "transaction_count": transaction_count,
                    },
                )

            # Optional model inference after statement run if artifacts are configured.
            if self._active_artifacts_dir:
                try:
                    inference_summary = self._run_model_inference(report_payload)
                except Exception as exc:
                    inference_summary = f"Inference failed: {exc}"
                self._results_tab.append_text("\n\nInference summary:\n" + inference_summary)
                formatted_factors = _MainWindow._format_inference_top_factors_summary(inference_summary)
                if formatted_factors:
                    self._results_tab.append_text("\n" + formatted_factors)

                factors_export_message = _MainWindow._persist_inference_factors_to_file(
                    report_payload,
                    inference_summary,
                )
                if factors_export_message:
                    self._results_tab.append_text("\n" + factors_export_message)

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
        except Exception as exc:
            # Keep desktop shell alive even if the run payload shape is unexpected.
            self._results_tab.append_text(f"\n\nRun payload handling failed: {exc}")
            self._status_label.setText("Run finished with errors")

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
    def _normalize_factor_list(raw_factors: object) -> List[Dict[str, float]]:
        if not isinstance(raw_factors, list):
            return []

        normalized: List[Dict[str, float]] = []
        for item in raw_factors:
            if not isinstance(item, dict):
                continue
            feature = item.get("feature")
            contribution = item.get("contribution")
            if feature is None or contribution is None:
                continue
            try:
                normalized.append({"feature": str(feature), "contribution": float(contribution)})
            except Exception:
                continue
        return normalized

    @staticmethod
    def _split_risk_and_healthy_factors(raw_payload: object) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        if not isinstance(raw_payload, dict):
            return [], []

        if any(key in raw_payload for key in ("risk_factors", "healthy_factors", "top_risk_factors", "top_healthy_factors")):
            risk_factors = _MainWindow._normalize_factor_list(
                raw_payload.get("risk_factors") if "risk_factors" in raw_payload else raw_payload.get("top_risk_factors")
            )
            healthy_factors = _MainWindow._normalize_factor_list(
                raw_payload.get("healthy_factors") if "healthy_factors" in raw_payload else raw_payload.get("top_healthy_factors")
            )
            return risk_factors[:5], healthy_factors[:5]

        combined = _MainWindow._normalize_factor_list(raw_payload.get("top_factors"))
        risk_candidates = [item for item in combined if item["contribution"] > 0]
        risk_candidates.sort(key=lambda item: item["contribution"], reverse=True)
        return risk_candidates[:5], []

    @staticmethod
    def _average_monthly_prediction_section(monthly: object) -> Optional[Dict[str, object]]:
        if not isinstance(monthly, dict):
            return None

        risk_score_total = 0.0
        saving_probability_total = 0.0
        month_count = 0
        risk_totals: Dict[str, List[float]] = {}
        healthy_totals: Dict[str, List[float]] = {}

        for raw_payload in monthly.values():
            if not isinstance(raw_payload, dict):
                continue
            try:
                risk_score = float(raw_payload.get("risk_score"))
                saving_probability = float(raw_payload.get("saving_probability"))
            except Exception:
                continue

            month_count += 1
            risk_score_total += risk_score
            saving_probability_total += saving_probability

            risk_factors, healthy_factors = _MainWindow._split_risk_and_healthy_factors(raw_payload)
            for item in risk_factors:
                bucket = risk_totals.setdefault(item["feature"], [0.0, 0.0])
                bucket[0] += float(item["contribution"])
                bucket[1] += 1.0
            for item in healthy_factors:
                bucket = healthy_totals.setdefault(item["feature"], [0.0, 0.0])
                bucket[0] += float(item["contribution"])
                bucket[1] += 1.0

        if month_count == 0:
            return None

        averaged_risk = [
            {"feature": feature, "contribution": round(total / count, 6)}
            for feature, (total, count) in risk_totals.items()
            if count > 0
        ]
        averaged_healthy = [
            {"feature": feature, "contribution": round(total / count, 6)}
            for feature, (total, count) in healthy_totals.items()
            if count > 0
        ]
        averaged_risk.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        averaged_healthy.sort(key=lambda item: item["contribution"])

        average_risk_score = risk_score_total / month_count
        average_saving_probability = saving_probability_total / month_count
        if average_risk_score < 0.33:
            risk_level = "healthy"
        elif average_risk_score < 0.67:
            risk_level = "moderate"
        else:
            risk_level = "risky"

        return {
            "risk_score": round(average_risk_score, 6),
            "saving_probability": round(average_saving_probability, 6),
            "risk_level": risk_level,
            "risk_factors": averaged_risk[:5],
            "healthy_factors": averaged_healthy[:5],
        }

    @staticmethod
    def _average_monthly_prediction_section(monthly: object) -> Optional[Dict[str, object]]:
        if not isinstance(monthly, dict):
            return None

        risk_score_total = 0.0
        saving_probability_total = 0.0
        month_count = 0

        risk_totals: Dict[str, List[float]] = {}
        healthy_totals: Dict[str, List[float]] = {}

        for raw_payload in monthly.values():
            if not isinstance(raw_payload, dict):
                continue
            try:
                risk_score_total += float(raw_payload.get("risk_score"))
                saving_probability_total += float(raw_payload.get("saving_probability"))
            except Exception:
                continue

            month_count += 1
            risk_factors, healthy_factors = _MainWindow._split_risk_and_healthy_factors(raw_payload)
            for item in risk_factors:
                bucket = risk_totals.setdefault(item["feature"], [0.0, 0.0])
                bucket[0] += float(item["contribution"])
                bucket[1] += 1.0
            for item in healthy_factors:
                bucket = healthy_totals.setdefault(item["feature"], [0.0, 0.0])
                bucket[0] += float(item["contribution"])
                bucket[1] += 1.0

        if month_count == 0:
            return None

        averaged_risk = [
            {"feature": feature, "contribution": round(total / count, 6)}
            for feature, (total, count) in risk_totals.items()
            if count > 0
        ]
        averaged_healthy = [
            {"feature": feature, "contribution": round(total / count, 6)}
            for feature, (total, count) in healthy_totals.items()
            if count > 0
        ]
        averaged_risk.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        averaged_healthy.sort(key=lambda item: item["contribution"])

        average_risk_score = risk_score_total / month_count
        average_saving_probability = saving_probability_total / month_count

        if average_risk_score < 0.33:
            risk_level = "healthy"
        elif average_risk_score < 0.67:
            risk_level = "moderate"
        else:
            risk_level = "risky"

        return {
            "risk_score": round(average_risk_score, 6),
            "saving_probability": round(average_saving_probability, 6),
            "risk_level": risk_level,
            "risk_factors": averaged_risk[:5],
            "healthy_factors": averaged_healthy[:5],
        }

    @staticmethod
    def _format_factor_lines(title: str, factors: List[Dict[str, float]]) -> List[str]:
        lines = [title]
        if not factors:
            lines.append("  (none)")
            return lines

        for index, factor in enumerate(factors[:5], start=1):
            lines.append(f"  {index}. {factor['feature']}: {float(factor['contribution']):.6f}")
        return lines

    @staticmethod
    def _build_inference_factor_export_payload(inference_summary: str) -> Optional[Dict[str, object]]:
        try:
            parsed = json.loads(inference_summary)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None

        current_risk, current_healthy = _MainWindow._split_risk_and_healthy_factors(parsed)
        monthly_payloads: Dict[str, Dict[str, object]] = {}
        monthly = parsed.get("monthly_predictions")
        if isinstance(monthly, dict):
            for month_key, raw_payload in monthly.items():
                if not isinstance(raw_payload, dict):
                    continue
                month_risk, month_healthy = _MainWindow._split_risk_and_healthy_factors(raw_payload)
                monthly_payloads[str(month_key)] = {
                    "risk_score": raw_payload.get("risk_score"),
                    "saving_probability": raw_payload.get("saving_probability"),
                    "risk_level": raw_payload.get("risk_level"),
                    "risk_factors": month_risk,
                    "healthy_factors": month_healthy,
                }

        return {
            "current_prediction": {
                "risk_score": parsed.get("risk_score"),
                "saving_probability": parsed.get("saving_probability"),
                "risk_level": parsed.get("risk_level"),
                "risk_factors": current_risk,
                "healthy_factors": current_healthy,
            },
            "monthly_predictions": monthly_payloads,
        }

    @staticmethod
    def _persist_inference_factors_to_file(
        report_payload: Dict[str, object],
        inference_summary: str,
    ) -> Optional[str]:
        payload = _MainWindow._build_inference_factor_export_payload(inference_summary)
        if not payload:
            return None

        output_files = report_payload.get("output_files")
        if not isinstance(output_files, dict):
            return None

        final_dataset_path = output_files.get("final_dataset")
        if not isinstance(final_dataset_path, str) or not final_dataset_path.strip():
            return None

        export_dir = os.path.dirname(final_dataset_path)
        if not export_dir:
            return None

        factors_path = os.path.join(export_dir, "inference_factors.txt")
        factors_text = _MainWindow._format_inference_factor_export_text(payload)
        with open(factors_path, "w", encoding="utf-8") as handle:
            handle.write(factors_text)

        output_files["inference_factors"] = factors_path

        run_report_path = output_files.get("run_report")
        if isinstance(run_report_path, str) and run_report_path.strip():
            try:
                with open(run_report_path, "w", encoding="utf-8") as handle:
                    json.dump(report_payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            except Exception:
                pass

        return f"Inference factors exported to {factors_path}"

    @staticmethod
    def _format_inference_factor_export_text(payload: Dict[str, object]) -> str:
        lines: List[str] = []

        def _append_prediction_block(title: str, section: object) -> None:
            if not isinstance(section, dict):
                return
            lines.append(title)
            lines.append(f"  risk_score: {section.get('risk_score')}")
            lines.append(f"  saving_probability: {section.get('saving_probability')}")
            lines.append(f"  risk_level: {section.get('risk_level')}")
            risk_factors, healthy_factors = _MainWindow._split_risk_and_healthy_factors(section)
            lines.extend(_MainWindow._format_factor_lines("  Risk factors:", risk_factors))
            lines.extend(_MainWindow._format_factor_lines("  Healthy/stable factors:", healthy_factors))
            lines.append("")

        if not isinstance(payload, dict):
            return ""

        monthly = payload.get("monthly_predictions")
        general_section = _MainWindow._average_monthly_prediction_section(monthly)
        if general_section is None:
            _append_prediction_block("Current prediction", payload.get("current_prediction"))
        else:
            _append_prediction_block("Average monthly prediction", general_section)
            lines.append("Note: the general section is the arithmetic mean of the monthly predictions.")
            lines.append("")

        if isinstance(monthly, dict) and monthly:
            lines.append("Monthly predictions")
            for month_key in sorted(monthly.keys()):
                _append_prediction_block(f"- {month_key}", monthly.get(month_key))

        text = "\n".join(lines).strip()
        return text + ("\n" if text else "")

    @staticmethod
    def _format_inference_top_factors_summary(inference_summary: str) -> str:
        try:
            parsed = json.loads(inference_summary)
        except Exception:
            return ""
        if not isinstance(parsed, dict):
            return ""

        sections: List[str] = []

        monthly = parsed.get("monthly_predictions")
        general_section = _MainWindow._average_monthly_prediction_section(monthly)
        if general_section is None:
            current_risk, current_healthy = _MainWindow._split_risk_and_healthy_factors(parsed)
            sections.extend(_MainWindow._format_factor_lines("Top 5 risk factors - current prediction:", current_risk))
            sections.extend(
                _MainWindow._format_factor_lines(
                    "Top 5 healthy behavior factors - current prediction:",
                    current_healthy,
                )
            )
        else:
            sections.extend(_MainWindow._format_factor_lines("Top 5 risk factors - average monthly prediction:", general_section.get("risk_factors", [])))
            sections.extend(
                _MainWindow._format_factor_lines(
                    "Top 5 healthy behavior factors - average monthly prediction:",
                    general_section.get("healthy_factors", []),
                )
            )

        if isinstance(monthly, dict):
            for month_key in sorted(monthly.keys()):
                raw_payload = monthly.get(month_key)
                if not isinstance(raw_payload, dict):
                    continue
                month_risk, month_healthy = _MainWindow._split_risk_and_healthy_factors(raw_payload)
                sections.extend(
                    _MainWindow._format_factor_lines(
                        f"Top 5 risk factors - {month_key}:",
                        month_risk,
                    )
                )
                sections.extend(
                    _MainWindow._format_factor_lines(
                        f"Top 5 healthy behavior factors - {month_key}:",
                        month_healthy,
                    )
                )

        return "\n".join(sections)

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

        def _canonicalize_prediction_keys(payload: Mapping[str, float]) -> Dict[str, float]:
            canonical: Dict[str, float] = {}
            for key, value in payload.items():
                canonical_key = "Risk_Score" if str(key) == "risk_score" else str(key)
                if canonical_key in {"Risk_Score", "saving_probability"}:
                    canonical[canonical_key] = float(value)
            return canonical

        fallback_payload = _canonicalize_prediction_keys(fallback_payload)
        monthly_payloads = {
            month_key: _canonicalize_prediction_keys(payload)
            for month_key, payload in monthly_payloads.items()
        }

        target_columns = set(fallback_payload.keys())
        for payload in monthly_payloads.values():
            target_columns.update(payload.keys())
        if "Risk_Score" in target_columns:
            target_columns.add("Behavior_Risk_Level")

        for column in sorted(target_columns):
            if column not in existing_columns:
                existing_columns.append(column)

        for row in rows:
            month_key = str(row.get("statement_month", "") or "").strip()
            row_payload = monthly_payloads.get(month_key) or fallback_payload
            for column in target_columns:
                if column in row_payload:
                    row[column] = str(row_payload[column])
            if "Risk_Score" in row:
                try:
                    row["Behavior_Risk_Level"] = classify_behavior_risk_level(float(row["Risk_Score"]))
                except Exception:
                    pass

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
        self._current_run_bridge = None
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
                    "risk_level": monthly_prediction.risk_level,
                    "inputs_scaled": monthly_prediction.inputs_scaled,
                    "top_factors": monthly_prediction.top_factors,
                    "risk_factors": monthly_prediction.risk_factors,
                    "healthy_factors": monthly_prediction.healthy_factors,
                }

        return json.dumps(
            {
                "risk_score": prediction.risk_score,
                "saving_probability": prediction.saving_probability,
                "risk_level": prediction.risk_level,
                "inputs_scaled": prediction.inputs_scaled,
                "scaled_feature_columns": prediction.scaled_feature_columns,
                "alerts": prediction.alerts,
                "top_factors": prediction.top_factors,
                "risk_factors": prediction.risk_factors,
                "healthy_factors": prediction.healthy_factors,
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






