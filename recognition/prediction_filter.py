"""Suppress unstable and low-confidence outputs without auto-editing transcripts."""
from collections import Counter, deque


class PredictionFilter:
    def __init__(self, threshold: float = 0.65, stable_frames: int = 5, window: int = 8):
        self.threshold = float(threshold)
        self.stable_frames = max(1, int(stable_frames))
        self.window = max(self.stable_frames, int(window))
        self._history = deque(maxlen=self.window)

    def reset(self) -> None:
        self._history.clear()

    def update(self, label: str | None, confidence: float) -> dict:
        confidence = float(confidence)
        if not label or confidence < self.threshold:
            self._history.clear()
            return {"accepted": False, "label": None, "confidence": confidence, "status": "uncertain"}
        self._history.append(label)
        counts = Counter(self._history)
        winner, count = counts.most_common(1)[0]
        stable = count >= self.stable_frames and self._history[-1] == winner
        return {"accepted": stable, "label": winner if stable else None,
                "confidence": confidence, "status": "stable" if stable else "stabilizing"}
