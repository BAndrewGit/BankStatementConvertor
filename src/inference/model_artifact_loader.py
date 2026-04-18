from __future__ import annotations

from dataclasses import dataclass
import json
import os
import pickle
import warnings
from typing import Dict, List, Mapping, Sequence

import yaml
from src.profile.questionnaire import validate_questionnaire_groups_against_features


REQUIRED_ARTIFACT_FILES = (
    "model.pt",
    "scaler.pkl",
    "feature_columns.json",
    "thresholds.json",
    "model_metadata.json",
    "bank_mapping_rules.yaml",
)


@dataclass(frozen=True)
class ModelArtifacts:
    artifacts_dir: str
    model: object
    scaler: object
    feature_columns: List[str]
    thresholds: Dict[str, object]
    model_metadata: Dict[str, object]
    bank_mapping_rules: Dict[str, object]


class ModelArtifactLoader:
    def __init__(self, artifacts_dir: str) -> None:
        self._artifacts_dir = os.path.abspath(artifacts_dir)

    def required_paths(self) -> Dict[str, str]:
        return {
            filename: os.path.join(self._artifacts_dir, filename)
            for filename in REQUIRED_ARTIFACT_FILES
        }

    def missing_artifacts(self) -> List[str]:
        missing: List[str] = []
        for filename, path in self.required_paths().items():
            if not os.path.exists(path):
                missing.append(filename)
        return missing

    def compatibility_status(self, feature_source_map: Mapping[str, str]) -> Dict[str, object]:
        try:
            artifacts = self.load(require_multitask=False)
        except Exception as exc:
            return {
                "compatible": False,
                "error": str(exc),
                "missing_artifacts": self.missing_artifacts(),
            }

        missing_mappings = [
            feature for feature in artifacts.feature_columns if feature not in feature_source_map
        ]
        metadata = artifacts.model_metadata
        model_kind = str(
            metadata.get("model_type")
            or metadata.get("model_name")
            or metadata.get("model_family")
            or "unknown"
        )
        is_multitask = bool(metadata.get("multitask", "multitask" in model_kind.lower()))

        missing_questionnaire_groups = validate_questionnaire_groups_against_features(
            artifacts.feature_columns
        )

        return {
            "compatible": len(missing_mappings) == 0 and len(missing_questionnaire_groups) == 0,
            "missing_artifacts": self.missing_artifacts(),
            "missing_feature_mappings": missing_mappings,
            "missing_questionnaire_groups": missing_questionnaire_groups,
            "model_type": model_kind,
            "is_multitask": is_multitask,
            "feature_count": len(artifacts.feature_columns),
        }

    def load(self, require_multitask: bool = True) -> ModelArtifacts:
        missing = self.missing_artifacts()
        if missing:
            raise FileNotFoundError(f"Missing model artifacts: {missing}")

        paths = self.required_paths()

        feature_columns = self._read_json_list(paths["feature_columns.json"])
        thresholds = self._read_json_dict(paths["thresholds.json"])
        model_metadata = self._read_json_dict(paths["model_metadata.json"])
        bank_mapping_rules = self._read_yaml_dict(paths["bank_mapping_rules.yaml"])
        scaler = self._read_pickle(paths["scaler.pkl"])
        model = self._load_model(paths["model.pt"], model_metadata=model_metadata)

        model_kind = str(
            model_metadata.get("model_type")
            or model_metadata.get("model_name")
            or model_metadata.get("model_family")
            or ""
        )
        is_multitask = bool(model_metadata.get("multitask", "multitask" in model_kind.lower()))
        if require_multitask and not is_multitask:
            raise ValueError("Loaded model artifact is not marked as multitask in model_metadata.json")

        return ModelArtifacts(
            artifacts_dir=self._artifacts_dir,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            thresholds=thresholds,
            model_metadata=model_metadata,
            bank_mapping_rules=bank_mapping_rules,
        )

    @staticmethod
    def _read_json_dict(path: str) -> Dict[str, object]:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return payload

    @staticmethod
    def _read_json_list(path: str) -> List[str]:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError(f"Expected JSON string list in {path}")
        return list(payload)

    @staticmethod
    def _read_yaml_dict(path: str) -> Dict[str, object]:
        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Expected YAML object in {path}")
        return payload

    @staticmethod
    def _read_pickle(path: str) -> object:
        try:
            with open(path, "rb") as handle:
                return ModelArtifactLoader._load_with_warning_capture(lambda: pickle.load(handle), path)
        except ValueError:
            raise
        except Exception:
            # Many sklearn scalers are persisted with joblib.
            try:
                import joblib  # type: ignore

                return ModelArtifactLoader._load_with_warning_capture(lambda: joblib.load(path), path)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"Could not load serialized artifact: {path}") from exc

    @staticmethod
    def _load_with_warning_capture(loader_fn, artifact_path: str) -> object:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            loaded = loader_fn()

        for item in captured_warnings:
            category_name = getattr(item.category, "__name__", "")
            if category_name != "InconsistentVersionWarning":
                continue

            if os.getenv("ALLOW_SKLEARN_VERSION_MISMATCH", "0") == "1":
                break

            message = str(item.message)
            raise ValueError(
                "Incompatible sklearn version for serialized scaler artifact. "
                f"Artifact: {artifact_path}. Details: {message}. "
                "Install the sklearn version used at training time or set "
                "ALLOW_SKLEARN_VERSION_MISMATCH=1 to override at your own risk."
            )

        return loaded

    @staticmethod
    def _load_model(path: str, model_metadata: Mapping[str, object] | None = None) -> object:
        # First attempt torch.load for real .pt artifacts. Fallback to pickle for tests/local stubs.
        try:
            import torch  # type: ignore

            loaded = torch.load(path, map_location="cpu")
            if isinstance(loaded, dict) and all(isinstance(key, str) for key in loaded.keys()):
                rebuilt = ModelArtifactLoader._rebuild_torch_multitask_model_if_needed(
                    loaded,
                    model_metadata=model_metadata or {},
                )
                if rebuilt is not None:
                    return rebuilt
            return loaded
        except Exception:
            with open(path, "rb") as handle:
                return pickle.load(handle)

    @staticmethod
    def _rebuild_torch_multitask_model_if_needed(
        state_dict: Mapping[str, object],
        model_metadata: Mapping[str, object],
    ) -> object | None:
        if "shared_trunk.0.weight" not in state_dict:
            return None
        if "risk_head.weight" not in state_dict or "savings_head.weight" not in state_dict:
            return None

        model_config = model_metadata.get("model_config")
        if not isinstance(model_config, Mapping):
            return None

        try:
            import torch
            import torch.nn as nn

            # Infer dimensions directly from checkpoint weights to tolerate stale metadata.
            linear_weight_keys = sorted(
                [
                    key
                    for key in state_dict.keys()
                    if isinstance(key, str) and key.startswith("shared_trunk.") and key.endswith(".weight")
                ],
                key=lambda item: int(str(item).split(".")[1]),
            )
            if not linear_weight_keys:
                return None

            first_weight = state_dict[linear_weight_keys[0]]
            if not hasattr(first_weight, "shape") or len(first_weight.shape) != 2:
                return None
            input_dim = int(first_weight.shape[1])

            hidden_dims = []
            for key in linear_weight_keys:
                weight = state_dict[key]
                if not hasattr(weight, "shape") or len(weight.shape) != 2:
                    return None
                hidden_dims.append(int(weight.shape[0]))
            activation_name = str(model_config.get("activation", "relu")).strip().lower()
            dropout = float(model_config.get("dropout", 0.0))
            if input_dim <= 0 or not hidden_dims:
                return None

            activation_cls = {
                "relu": nn.ReLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
            }.get(activation_name, nn.ReLU)

            class _LoadedMultitaskNet(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    layers: list[nn.Module] = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(activation_cls())
                        if dropout > 0:
                            layers.append(nn.Dropout(dropout))
                        prev_dim = hidden_dim
                    self.shared_trunk = nn.Sequential(*layers)
                    self.risk_head = nn.Linear(prev_dim, 1)
                    self.savings_head = nn.Linear(prev_dim, 1)

                def forward(self, x):
                    hidden = self.shared_trunk(x)
                    return self.risk_head(hidden), self.savings_head(hidden)

            model = _LoadedMultitaskNet()
            model.load_state_dict(dict(state_dict), strict=False)
            model.eval()
            return model
        except Exception:
            return None



