from types import SimpleNamespace
import numpy as np
from vision.landmarks import FEATURE_DIM, LandmarkExtractor


class FakeImage:
    def __init__(self, **kwargs): self.data = kwargs["data"]


class FakeLandmarker:
    def __init__(self, result): self.result = result
    def detect(self, image): self.last_image = image; return self.result


def _hand(x_offset):
    return [SimpleNamespace(x=x_offset + (i % 5) * 0.018,
                            y=0.2 + (i // 5) * 0.025,
                            z=i * 0.0005) for i in range(21)]


def test_two_hand_results_are_labeled_by_anatomical_side_and_packed():
    extractor = LandmarkExtractor.__new__(LandmarkExtractor)
    extractor._mp = SimpleNamespace(Image=lambda **kwargs: FakeImage(**kwargs),
                                    ImageFormat=SimpleNamespace(SRGB=1))
    extractor._static = True
    extractor.mirrored_input = False
    extractor.swap_handedness = False
    extractor._timestamp = 0
    extractor._landmarker = FakeLandmarker(SimpleNamespace(
        hand_landmarks=[_hand(0.1), _hand(0.7)],
        handedness=[
            [SimpleNamespace(category_name="Left", score=0.91)],
            [SimpleNamespace(category_name="Right", score=0.88)],
        ],
    ))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feature, annotated, hands = extractor.process(frame, draw=True)
    assert feature.shape == (FEATURE_DIM,)
    assert np.isfinite(feature).all()
    assert [(hand["side"], hand["score"]) for hand in hands] == [
        ("Left", 0.88), ("Right", 0.91)
    ]
    # MediaPipe handedness is reversed for ordinary, non-mirrored camera input.
    assert feature[0] == feature[68] == 1.0
    assert annotated.shape == frame.shape
    assert extractor.last_hands == [
        {"side": "Left", "score": 0.88}, {"side": "Right", "score": 0.91}
    ]


def test_mirrored_input_physically_flips_pixels_before_detection():
    extractor = LandmarkExtractor.__new__(LandmarkExtractor)
    extractor._mp = SimpleNamespace(Image=lambda **kwargs: FakeImage(**kwargs),
                                    ImageFormat=SimpleNamespace(SRGB=1))
    extractor._static = True
    extractor.mirrored_input = True
    extractor.swap_handedness = False
    extractor._timestamp = 0
    result = SimpleNamespace(
        hand_landmarks=[_hand(0.2)],
        handedness=[[SimpleNamespace(category_name="Left", score=0.9)]],
    )
    extractor._landmarker = FakeLandmarker(result)
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[:, -1] = [0, 0, 255]
    feature, _, hands = extractor.process(frame)
    assert feature is not None
    assert hands[0]["side"] == "Left"
    # BGR red from the original right edge becomes RGB red at the mirrored left edge.
    assert extractor._landmarker.last_image.data[0, 0].tolist() == [255, 0, 0]
