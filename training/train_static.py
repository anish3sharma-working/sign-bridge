"""Train and evaluate a static hand-pose Random Forest using local labeled media."""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from config.settings import app_data_dir, FEATURE_VERSION
from database.db import Database
from recognition.static_model import StaticClassifier
from training.dataset_splitter import split_by_group
from vision.landmarks import LandmarkExtractor


def train_static(database_path: str | Path, model_dir: str | Path | None = None,
                 progress=None, cancel_event=None, mirrored_input: bool = False,
                 min_hand_confidence: float = 0.45, swap_handedness: bool = True,
                 model_mode: str = "unified", allow_non_independent: bool = False,
                 grayscale_input: bool = True, target_label: str | None = None) -> dict:
    if model_mode not in {"alphabet", "commands", "unified"}:
        raise ValueError("Training mode must be alphabet, commands, or unified.")
    def report(message):
        if progress:
            progress(message)
    def cancelled():
        return bool(cancel_event and cancel_event.is_set())

    database = Database(database_path)
    samples = database.list_samples()
    class_rows = {row["label"].casefold(): row for row in database.list_classes()}
    if model_mode != "unified":
        samples = [sample for sample in samples
                   if class_rows.get(sample["label"].casefold(), {}).get("recognition_mode") == model_mode]
    if target_label:
        target_row = class_rows.get(target_label.casefold())
        if not target_row or (model_mode != "unified" and target_row.get("recognition_mode") != model_mode):
            raise ValueError(f"Target class '{target_label}' is not available in the {model_mode} mode.")
    if not samples:
        raise ValueError(f"No labeled samples found for the {model_mode} mode. Add and label compatible samples first.")
    report(f"Checking {len(samples)} local sample files…")
    features, labels, groups, failures = [], [], [], []
    sample_ids = []
    try:
        extractor = LandmarkExtractor(static_image_mode=True, max_hands=2,
                                      mirrored_input=mirrored_input,
                                      min_hand_confidence=min_hand_confidence,
                                      swap_handedness=swap_handedness,
                                      grayscale_input=grayscale_input)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    try:
        for index, sample in enumerate(samples, 1):
            if cancelled():
                raise InterruptedError("Training was cancelled.")
            try:
                feature = extractor.extract_file(sample["source_path"])
                features.append(feature)
                labels.append(sample["label"] if not target_label or sample["label"].casefold() == target_label.casefold() else "__other__")
                sample_ids.append(sample["id"])
                # Participant group, when provided, takes precedence; otherwise keep each recording together.
                groups.append(sample["participant_group"].strip() or sample["recording_group"])
            except Exception as exc:
                failures.append({"file": Path(sample["source_path"]).name, "label": sample["label"], "reason": str(exc)})
            if index % 5 == 0 or index == len(samples):
                report(f"Extracted usable landmarks from {len(features)} of {index} samples…")
    finally:
        extractor.close()
    if len(set(labels)) < 2:
        raise ValueError(f"At least two classes need usable hand-landmark samples. Usable: {dict(Counter(labels))}; failures: {len(failures)}")
    if not features:
        raise ValueError("No usable hand landmarks were found. Review the sample files and lighting.")
    report("Creating participant/recording-independent train, validation, and test sets…")
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=str)
    group_array = np.asarray(groups, dtype=str)
    evaluation_warning = ""
    try:
        split = split_by_group(y, group_array)
    except ValueError:
        if not allow_non_independent:
            raise
        report("No independent session metadata found; using an exploratory sample-level split…")
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val, test = next(outer.split(X, y))
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=143)
        train_rel, val_rel = next(inner.split(X[train_val], y[train_val]))
        split = {"train": train_val[train_rel], "validation": train_val[val_rel], "test": test}
        evaluation_warning = "Exploratory sample-level split only: participant/recording groups were unavailable, so held-out metrics are not independent generalization evidence."
    database.assign_dataset_splits({sample_ids[int(index)]: name for name, indices in split.items() for index in indices})
    if cancelled():
        raise InterruptedError("Training was cancelled.")
    model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1)
    report(f"Training on {len(split['train'])} samples across {len(set(group_array[split['train']]))} groups…")
    model.fit(X[split["train"]], y[split["train"]])
    def metrics_for(indices):
        actual = y[indices]
        pred = model.predict(X[indices])
        classes = sorted(set(y))
        return {
            "sample_count": int(len(indices)),
            "accuracy": float(accuracy_score(actual, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, pred)),
            "classification_report": classification_report(actual, pred, labels=classes,
                output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(actual, pred, labels=classes).tolist(),
            "class_order": classes,
        }
    validation_metrics = metrics_for(split["validation"])
    report("Evaluating once on the held-out test set…")
    test_metrics = metrics_for(split["test"])
    record = {
        "feature_version": FEATURE_VERSION,
        "model_mode": model_mode,
        "target_label": target_label,
        "classes": sorted(set(y)),
        "sample_counts": dict(Counter(y)),
        "usable_samples": int(len(y)),
        "failed_samples": failures,
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "validation": validation_metrics,
        "test": test_metrics,
        "confidence_note": "Random Forest vote fractions are uncalibrated scores, not probabilities of correctness.",
        "evaluation_warning": evaluation_warning,
    }
    folder = Path(model_dir) if model_dir else app_data_dir() / "models"
    safe_target = re.sub(r"[^a-zA-Z0-9]+", "_", target_label.casefold()).strip("_") if target_label else "all"
    path = folder / f"signbridge_{model_mode}_{safe_target}_v5.joblib"
    StaticClassifier.save(path, model, sorted(set(y)), record, mirrored_input=mirrored_input,
                          swap_handedness=swap_handedness, model_mode=model_mode,
                          target_label=target_label)
    database.record_model(f"Local static hand-gesture RF ({model_mode})", str(path), FEATURE_VERSION,
                          sorted(set(y)), record, model_mode=model_mode, target_label=target_label)
    report(f"Done. Held-out test accuracy: {test_metrics['accuracy']:.3f}; see saved report for full metrics.")
    return {"model_path": str(path), "report": record}


if __name__ == "__main__":
    from config.settings import app_data_dir
    result = train_static(app_data_dir() / "signbridge.sqlite3")
    print(json.dumps(result, indent=2))
