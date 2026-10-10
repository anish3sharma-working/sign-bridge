"""Two-hand MediaPipe Tasks extraction with side-aware static features."""
from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np

FEATURE_VERSION = "hands2-mirrored-swap-side-local63-wristxyz-scale-v4"
HAND_FEATURE_DIM = 68
FEATURE_DIM = HAND_FEATURE_DIM * 2
SIDES = ("Left", "Right")
HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)]
SIDE_COLORS = {"Left": (65, 210, 170), "Right": (80, 160, 255)}


def normalize_landmarks(points) -> np.ndarray:
    """Return a 63-value wrist-anchored, hand-scale-normalized descriptor."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape == (63,):
        arr = arr.reshape(21, 3)
    if arr.shape != (21, 3) or not np.isfinite(arr).all():
        raise ValueError("Expected 21 finite hand landmarks with x, y, z coordinates.")
    shifted = arr - arr[0]
    scale = float(np.max(np.linalg.norm(shifted[:, :2], axis=1)))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("Hand landmarks are degenerate; please reposition your hand.")
    shifted /= scale
    return shifted.reshape(63).astype(np.float32)


def canonical_handedness(label: str, mirrored_input: bool = False,
                         swap_labels: bool = False) -> str:
    """Map MediaPipe handedness to the person’s anatomical side.

    The Tasks handedness convention assumes mirrored/selfie input. This app
    physically flips input pixels when mirrored_input=True, so detector labels
    then match the signer’s anatomical sides. Unmirrored pixels are swapped.
    """
    side = str(label).strip().title()
    if side not in SIDES:
        raise ValueError(f"Unknown hand side {label!r}; expected Left or Right.")
    if not mirrored_input:
        side = "Right" if side == "Left" else "Left"
    if swap_labels:
        side = "Right" if side == "Left" else "Left"
    return side


def pack_bimanual_features(hand_records, frame_shape) -> tuple[np.ndarray | None, list[dict]]:
    """Pack detected hands into canonical Left/Right 68-value slots.

    Each slot stores presence, 63 local pose values, frame-relative wrist
    x/y/z, and apparent hand scale. Position/scale preserve useful two-hand
    spatial information discarded by local normalization.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Frame dimensions must be positive.")
    selected: dict[str, dict] = {}
    for record in hand_records:
        side = str(record.get("side", "")).title()
        if side not in SIDES:
            continue
        score = float(record.get("score", 0.0))
        if not np.isfinite(score):
            continue
        if side not in selected or score > selected[side]["score"]:
            selected[side] = {**record, "side": side, "score": score}
    if not selected:
        return None, []

    packed = np.zeros(FEATURE_DIM, dtype=np.float32)
    output_records = []
    for side_index, side in enumerate(SIDES):
        record = selected.get(side)
        if record is None:
            continue
        points = np.asarray(record["points"], dtype=np.float32)
        pose = normalize_landmarks(points)
        wrist = points[0]
        scale = float(np.max(np.linalg.norm(points[:, :2] - wrist[:2], axis=1)))
        if not np.isfinite(scale):
            raise ValueError("Hand scale contains invalid coordinates.")
        offset = side_index * HAND_FEATURE_DIM
        packed[offset] = 1.0
        packed[offset + 1:offset + 64] = pose
        packed[offset + 64:offset + 67] = wrist
        packed[offset + 67] = scale
        output_records.append(record)
    if not np.isfinite(packed).all():
        raise ValueError("Two-hand feature vector contains invalid values.")
    return packed, output_records


class LandmarkExtractor:
    """Lazy local MediaPipe Tasks wrapper; detects and labels up to two hands."""
    def __init__(self, static_image_mode: bool = True, max_hands: int = 2,
                 mirrored_input: bool = False, min_hand_confidence: float = 0.45,
                 swap_handedness: bool = True):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("MediaPipe Tasks is unavailable. Install project requirements.") from exc
        if getattr(sys, "frozen", False):
            root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        else:
            root = Path(__file__).resolve().parents[1]
        asset = root / "models" / "hand_landmarker.task"
        if not asset.is_file():
            raise RuntimeError(f"The local hand landmark detector is missing: {asset}. Restore the official model asset listed in models/NOTICE.md; no model is downloaded at runtime.")
        confidence = float(min_hand_confidence)
        if not 0.1 <= confidence <= 0.95:
            raise ValueError("Hand detector confidence must be between 0.10 and 0.95.")
        try:
            running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
            options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(asset)),
                running_mode=running_mode,
                num_hands=max(1, min(2, int(max_hands))),
                min_hand_detection_confidence=confidence,
                min_hand_presence_confidence=confidence,
                min_tracking_confidence=confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize the local MediaPipe hand detector: {exc}") from exc
        self._mp = mp
        self._static = static_image_mode
        self.mirrored_input = bool(mirrored_input)
        self.swap_handedness = bool(swap_handedness)
        self._timestamp = 0
        self.last_hands: list[dict] = []

    def process(self, frame_bgr, draw: bool = False):
        """Return (bilateral_feature_or_None, annotated_frame, hand summaries)."""
        import cv2
        if self.mirrored_input:
            frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._static:
            result = self._landmarker.detect(image)
        else:
            self._timestamp = max(self._timestamp + 1, int(time.monotonic() * 1000))
            result = self._landmarker.detect_for_video(image, self._timestamp)

        raw_landmarks = getattr(result, "hand_landmarks", None) or []
        handedness = getattr(result, "handedness", None) or []
        records = []
        for index, landmarks in enumerate(raw_landmarks):
            classifications = handedness[index] if index < len(handedness) else []
            if not classifications:
                continue
            classification = classifications[0]
            try:
                side = canonical_handedness(classification.category_name, self.mirrored_input,
                                            self.swap_handedness)
            except ValueError:
                continue
            points = np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
            records.append({
                "side": side,
                "score": float(getattr(classification, "score", 0.0)),
                "points": points,
                "landmarks": landmarks,
            })

        feature, selected = pack_bimanual_features(records, frame_bgr.shape)
        if draw:
            height, width = frame_bgr.shape[:2]
            for record in selected:
                landmarks = record["landmarks"]
                pixels = [(int(np.clip(p.x * width, 0, width - 1)),
                           int(np.clip(p.y * height, 0, height - 1))) for p in landmarks]
                color = SIDE_COLORS[record["side"]]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame_bgr, pixels[a], pixels[b], color, 2, cv2.LINE_AA)
                for x, y in pixels:
                    cv2.circle(frame_bgr, (x, y), 3, (245, 250, 255), -1, cv2.LINE_AA)
                x, y = pixels[0]
                cv2.putText(frame_bgr, f"{record['side']} hand {record['score']:.0%}",
                            (max(4, x - 8), max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, color, 2, cv2.LINE_AA)
        summaries = [{"side": item["side"], "score": item["score"]} for item in selected]
        self.last_hands = summaries
        return feature, frame_bgr, summaries

    def extract_file(self, path: str | Path):
        import cv2
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"Sample file is missing: {p}")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            frame = cv2.imread(str(p))
            if frame is None:
                raise ValueError(f"Could not decode image: {p.name}")
        elif p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {p.name}")
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if count > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                raise ValueError(f"Could not read a usable frame from: {p.name}")
        else:
            raise ValueError(f"Unsupported sample format: {p.suffix}")
        feature, _, _hands = self.process(frame)
        if feature is None:
            raise ValueError(f"No hand landmarks found in: {p.name}")
        return feature

    def close(self):
        self._landmarker.close()
