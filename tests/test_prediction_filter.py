from recognition.prediction_filter import PredictionFilter


def test_low_confidence_resets_history():
    f = PredictionFilter(0.7, stable_frames=3)
    assert not f.update("A", 0.69)["accepted"]
    assert not f.update("A", 0.99)["accepted"]
    assert not f.update(None, 0.99)["accepted"]
    assert f.update("A", 0.95)["status"] == "stabilizing"


def test_stability_requires_repeated_same_label():
    f = PredictionFilter(0.5, stable_frames=3, window=5)
    assert f.update("A", 0.8)["status"] == "stabilizing"
    assert f.update("B", 0.8)["status"] == "stabilizing"
    assert f.update("A", 0.8)["status"] == "stabilizing"
    assert f.update("A", 0.8)["accepted"] is True
    assert f.update("B", 0.8)["accepted"] is False
