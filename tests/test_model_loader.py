import json
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from recognition.static_model import StaticClassifier
from vision.landmarks import FEATURE_DIM


def test_model_artifact_roundtrip_and_prediction(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, FEATURE_DIM)).astype(np.float32)
    y = np.array(["A"] * 6 + ["B"] * 6)
    estimator = RandomForestClassifier(n_estimators=5, random_state=1).fit(x, y)
    path = StaticClassifier.save(tmp_path / "toy.joblib", estimator, ["A", "B"], {"test_fixture_only": True})
    loaded = StaticClassifier.load(path, threshold=0.0)
    label, score, candidates = loaded.predict(x[0])
    assert label in {"A", "B"}
    assert 0 <= score <= 1
    assert candidates


def test_incompatible_metadata_is_rejected(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, FEATURE_DIM)).astype(np.float32)
    estimator = RandomForestClassifier(n_estimators=2, random_state=1).fit(x, ["A", "A", "B", "B"])
    path = StaticClassifier.save(tmp_path / "toy.joblib", estimator, ["A", "B"], {})
    meta_path = path.with_suffix(path.suffix + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")); meta["feature_dim"] = 3
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        StaticClassifier.load(path)


def test_model_rejects_different_mirrored_input_orientation(tmp_path):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(8, FEATURE_DIM)).astype(np.float32)
    y = np.array(["A"] * 4 + ["B"] * 4)
    estimator = RandomForestClassifier(n_estimators=3, random_state=3).fit(x, y)
    path = StaticClassifier.save(tmp_path / "mirrored.joblib", estimator, ["A", "B"], {}, mirrored_input=True)
    loaded = StaticClassifier.load(path, mirrored_input=True)
    assert loaded.mirrored_input is True
    with pytest.raises(ValueError, match="mirroring setting"):
        StaticClassifier.load(path, mirrored_input=False)


def test_model_rejects_different_left_right_correction(tmp_path):
    rng = np.random.default_rng(13)
    x = rng.normal(size=(8, FEATURE_DIM)).astype(np.float32)
    y = np.array(["A"] * 4 + ["B"] * 4)
    estimator = RandomForestClassifier(n_estimators=3, random_state=5).fit(x, y)
    path = StaticClassifier.save(tmp_path / "corrected.joblib", estimator, ["A", "B"], {},
                                 mirrored_input=True, swap_handedness=True)
    loaded = StaticClassifier.load(path, mirrored_input=True, swap_handedness=True)
    assert loaded.swap_handedness is True
    with pytest.raises(ValueError, match="Left/Right correction"):
        StaticClassifier.load(path, mirrored_input=True, swap_handedness=False)
