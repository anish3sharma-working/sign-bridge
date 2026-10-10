"""Validation helpers for local image/video samples; no media is transmitted."""
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def validate_media(path: str | Path, label: str) -> tuple[bool, str, str | None]:
    p = Path(path)
    if not label or not label.strip():
        return False, "A non-empty gesture label is required.", None
    if not p.is_file():
        return False, "File does not exist or is not a regular file.", None
    if p.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return False, f"Unsupported format: {p.suffix or '(no extension)'}", None
    try:
        import cv2
    except ImportError:
        return False, "OpenCV is not installed.", None
    if p.suffix.lower() in IMAGE_EXTENSIONS:
        image = cv2.imread(str(p))
        if image is None:
            return False, "Image could not be decoded.", None
        return True, "Valid image.", "image"
    capture = cv2.VideoCapture(str(p))
    try:
        if not capture.isOpened():
            return False, "Video could not be opened.", None
        ok, frame = capture.read()
        if not ok or frame is None:
            return False, "Video opened but no readable frame was found.", None
        return True, "Valid video.", "video"
    finally:
        capture.release()
