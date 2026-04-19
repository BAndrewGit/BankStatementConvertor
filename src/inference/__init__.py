from .model_artifact_loader import ModelArtifactLoader, ModelArtifacts

__all__ = [
    "ModelArtifactLoader",
    "ModelArtifacts",
    "Predictor",
    "PredictionResult",
]


def __getattr__(name: str):
    if name in {"Predictor", "PredictionResult"}:
        from .predictor import PredictionResult, Predictor

        return {"Predictor": Predictor, "PredictionResult": PredictionResult}[name]
    raise AttributeError(name)


