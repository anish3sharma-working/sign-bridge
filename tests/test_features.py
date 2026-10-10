import numpy as np
import pytest
from vision.landmarks import normalize_landmarks


def test_normalization_is_finite_wrist_anchored_and_scaled():
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0, 20, 21)
    points[:, 1] = np.linspace(0, 10, 21)
    result = normalize_landmarks(points)
    assert result.shape == (63,)
    normalized = result.reshape(21, 3)
    assert np.allclose(normalized[0], 0)
    assert np.isfinite(result).all()
    assert np.isclose(np.max(np.linalg.norm(normalized[:, :2], axis=1)), 1.0)


def test_invalid_landmark_inputs_are_rejected():
    with pytest.raises(ValueError): normalize_landmarks(np.zeros((5, 3)))
    with pytest.raises(ValueError): normalize_landmarks(np.full((21, 3), np.nan))
    with pytest.raises(ValueError): normalize_landmarks(np.zeros((21, 3)))
