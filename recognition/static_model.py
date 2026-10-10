"""Random-forest artifact handling for app-trained bilateral static classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
from config.settings import FEATURE_VERSION
from vision.landmarks import FEATURE_DIM

ARTIFACT_VERSION = 4


class StaticClassifier:
    def __init__(self, estimator, classes: list[str], threshold: float = 0.65,
                 mirrored_input: bool = False, swap_handedness: bool = True):
        self.estimator = estimator
        self.classes = list(classes)
        self.threshold = float(threshold)
        self.mirrored_input = bool(mirrored_input)
        self.swap_handedness = bool(swap_handedness)

    def predict(self, feature) -> tuple[str | None, float, list[tuple[str, float]]]:
        x = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
            raise ValueError(f"Input features do not match the supported {FEATURE_DIM}-value bilateral hand format.")
        probabilities = self.estimator.predict_proba(x)[0]
        ranking = sorted(zip(self.estimator.classes_, probabilities), key=lambda item: float(item[1]), reverse=True)
        candidates = [(str(label), float(score)) for label, score in ranking[:3]]
        label, confidence = candidates[0]
        return (label if confidence >= self.threshold else None, confidence, candidates)

    @staticmethod
    def save(path: str | Path, estimator, classes: list[str], metrics: dict,
             mirrored_input: bool = False, swap_handedness: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"artifact_version": ARTIFACT_VERSION, "feature_version": FEATURE_VERSION,
                   "feature_dim": FEATURE_DIM, "mirrored_input": bool(mirrored_input),
                   "swap_handedness": bool(swap_handedness),
                   "classes": list(classes), "estimator": estimator}
        joblib.dump(payload, path)
        metadata = {"artifact_version": ARTIFACT_VERSION, "feature_version": FEATURE_VERSION,
                    "feature_dim": FEATURE_DIM, "mirrored_input": bool(mirrored_input),
                    "swap_handedness": bool(swap_handedness),
                    "classes": list(classes), "metrics": metrics,
                    "warning": "Only load model files trained and exported by your trusted SignBridge installation."}
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path, threshold: float = 0.65,
             mirrored_input: bool = False, swap_handedness: bool = True) -> "StaticClassifier":
        path = Path(path)
        if path.suffix.lower() != ".joblib" or not path.is_file():
            raise ValueError("Select an existing .joblib model created by SignBridge.")
        meta_path = path.with_suffix(path.suffix + ".json")
        if not meta_path.is_file():
            raise ValueError("Model metadata sidecar is missing; this model cannot be verified.")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Model metadata is unreadable or corrupted.") from exc
        if meta.get("artifact_version") != ARTIFACT_VERSION or meta.get("feature_version") != FEATURE_VERSION or meta.get("feature_dim") != FEATURE_DIM:
            raise ValueError("Model version or feature dimensions are incompatible with this application.")
        if bool(meta.get("mirrored_input", False)) != bool(mirrored_input):
            raise ValueError("Model input mirroring setting differs from the current handedness setting. Restore the training setting or retrain the model.")
        if bool(meta.get("swap_handedness", False)) != bool(swap_handedness):
            raise ValueError("Model Left/Right correction differs from the current setting. Restore the training correction or retrain the model.")
        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ValueError(f"The model file is corrupted or cannot be loaded: {exc}") from exc
        if (not isinstance(payload, dict) or payload.get("feature_version") != FEATURE_VERSION
                or payload.get("feature_dim") != FEATURE_DIM
                or bool(payload.get("mirrored_input", False)) != bool(mirrored_input)
                or bool(payload.get("swap_handedness", False)) != bool(swap_handedness)):
            raise ValueError("Model contents do not match the metadata format.")
        classes = payload.get("classes")
        estimator = payload.get("estimator")
        if not isinstance(classes, list) or not classes or not hasattr(estimator, "predict_proba"):
            raise ValueError("Model artifact is missing a valid classifier or class mapping.")
        if list(classes) != list(meta.get("classes", [])):
            raise ValueError("Model class mapping does not match its metadata sidecar.")
        if list(getattr(estimator, "classes_", [])) != classes or getattr(estimator, "n_features_in_", None) != FEATURE_DIM:
            raise ValueError("Estimator class mapping or feature count is incompatible with the model metadata.")
        return StaticClassifier(estimator, classes, threshold, mirrored_input, swap_handedness)
