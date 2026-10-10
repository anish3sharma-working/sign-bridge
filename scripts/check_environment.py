"""Report runtime/dependency availability; this script does not test a webcam."""
import importlib
import sys


def main():
    print(f"Python: {sys.version.split()[0]} ({'64-bit' if sys.maxsize > 2**32 else '32-bit'})")
    if sys.version_info[:2] != (3, 14):
        print("WARNING: the current project pins are validated for CPython 3.14 on this device.")
    packages = [("PySide6", "PySide6"), ("OpenCV", "cv2"), ("MediaPipe", "mediapipe"),
                ("NumPy", "numpy"), ("scikit-learn", "sklearn"), ("pyttsx3", "pyttsx3")]
    missing = []
    for label, module in packages:
        try:
            loaded = importlib.import_module(module)
            version = getattr(loaded, "__version__", "installed")
            print(f"{label}: {version}")
        except Exception as exc:
            missing.append(label)
            print(f"{label}: unavailable ({exc})")
    if missing:
        print("\nInstall missing dependencies with: python -m pip install -r requirements.txt")
        return 1
    print("\nDependencies are importable. Webcam access and OS speech voices require separate hardware checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
