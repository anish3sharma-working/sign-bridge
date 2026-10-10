from pathlib import Path
import numpy as np
import pytest

mp = pytest.importorskip("mediapipe")
pytest.importorskip("cv2")
from vision.landmarks import LandmarkExtractor


def test_official_local_task_asset_initializes_and_handles_no_hand():
    project_root = Path(__file__).resolve().parents[1]
    assert (project_root / "models" / "hand_landmarker.task").is_file()
    extractor = LandmarkExtractor(static_image_mode=True)
    try:
        feature, annotated, hands = extractor.process(np.zeros((96, 96, 3), dtype=np.uint8), draw=True)
        assert feature is None
        assert annotated.shape == (96, 96, 3)
        assert hands == []
    finally:
        extractor.close()
