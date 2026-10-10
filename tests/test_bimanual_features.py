import numpy as np
import pytest
from vision.landmarks import (
    FEATURE_DIM, canonical_handedness, pack_bimanual_features,
)


def hand(side, wrist_x, score=0.9):
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = wrist_x + np.linspace(0.0, 0.12, 21)
    points[:, 1] = 0.25 + np.linspace(0.0, 0.1, 21)
    points[:, 2] = np.linspace(0.0, 0.02, 21)
    points[0] = [wrist_x, 0.25, 0.0]
    return {"side": side, "score": score, "points": points}


def test_non_mirrored_mediapipe_handedness_is_swapped_to_person_side():
    assert canonical_handedness("Left", mirrored_input=False) == "Right"
    assert canonical_handedness("Right", mirrored_input=False) == "Left"
    assert canonical_handedness("Left", mirrored_input=True) == "Left"
    assert canonical_handedness("Right", mirrored_input=True) == "Right"
    assert canonical_handedness("Left", mirrored_input=True, swap_labels=True) == "Right"
    assert canonical_handedness("Right", mirrored_input=True, swap_labels=True) == "Left"


def test_right_only_hand_is_stored_in_right_slot_and_keeps_image_position():
    feature, hands = pack_bimanual_features([hand("Right", 0.72)], (480, 640, 3))
    assert feature.shape == (FEATURE_DIM,)
    assert hands[0]["side"] == "Right"
    assert feature[0] == 0.0
    assert feature[68] == 1.0
    assert feature[68 + 64] == pytest.approx(0.72)
    assert np.count_nonzero(feature[:68]) == 0


def test_bilateral_pose_keeps_both_sides_and_relative_hand_position():
    feature, hands = pack_bimanual_features(
        [hand("Right", 0.7), hand("Left", 0.2)], (480, 640, 3)
    )
    assert [item["side"] for item in hands] == ["Left", "Right"]
    assert feature[0] == feature[68] == 1.0
    assert feature[64] == pytest.approx(0.2)
    assert feature[68 + 64] == pytest.approx(0.7)
    assert feature.dtype == np.float32
    assert np.isfinite(feature).all()


def test_duplicate_side_keeps_higher_confidence_detection():
    low, high = hand("Left", 0.25, 0.4), hand("Left", 0.8, 0.95)
    feature, hands = pack_bimanual_features([low, high], (480, 640, 3))
    assert len(hands) == 1 and hands[0]["score"] == pytest.approx(0.95)
    assert feature[64] == pytest.approx(0.8)
    assert feature[68] == 0.0


def test_no_hands_returns_no_feature_and_bad_frame_is_rejected():
    assert pack_bimanual_features([], (480, 640, 3)) == (None, [])
    with pytest.raises(ValueError, match="dimensions"):
        pack_bimanual_features([hand("Left", 0.2)], (0, 640, 3))
