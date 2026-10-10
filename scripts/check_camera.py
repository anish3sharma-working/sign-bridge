"""Probe local camera indices and immediately release each device."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.capture import open_capture


def main():
    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed. Install requirements first.")
        return 2
    found = []
    for index in range(5):
        cap, frame, backend = open_capture(cv2, index)
        if cap is None:
            continue
        try:
            height, width = frame.shape[:2]
            found.append((index, width, height))
            print(f"Camera {index}: frame read succeeded ({width}×{height}) via {backend}")
        finally:
            cap.release()
    if not found:
        print("No usable camera found in indices 0–4 using DirectShow, Media Foundation, or OpenCV default capture.")
        print("Check Windows camera privacy permissions and close other camera apps; the application can still be run without a camera.")
        return 1
    print("Camera handles released. Choose one of the listed indices in SignBridge settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
