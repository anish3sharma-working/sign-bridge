"""Application paths and user settings for the local Windows desktop app."""
from __future__ import annotations
import json
import os
from pathlib import Path

APP_NAME = "SignBridge"
FEATURE_VERSION = "hands2-mirrored-swap-side-local63-wristxyz-scale-v4"


def app_data_dir() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def defaults() -> dict:
    return {
        "camera_index": 0,
        "camera_width": 640,
        "camera_height": 480,
        "confidence_threshold": 0.65,
        "stability_frames": 5,
        "inference_fps": 15,
        "min_hand_confidence": 0.45,
        "mirrored_input": True,
        "swap_handedness": True,
        "settings_version": 3,
        "speech_enabled": True,
        "speech_rate": 165,
        "theme": "dark",
        "manual_confirmation": True,
        "show_landmarks": True,
    }


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = path or (app_data_dir() / "settings.json")
        self.data = defaults()
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(stored, dict):
                    self.data.update({k: v for k, v in stored.items() if k in self.data})
                    settings_version = int(stored.get("settings_version", 1))
                    if settings_version < 2:
                        # Previous versions changed only handedness labels.
                        # The new setting physically mirrors pixels before
                        # detection, so invert the old orientation convention.
                        self.data["mirrored_input"] = not bool(stored.get("mirrored_input", False))
                    if settings_version < 3:
                        self.data["swap_handedness"] = True
                    if settings_version < 3:
                        self.data["settings_version"] = 3
                        self.save()
            except (OSError, json.JSONDecodeError):
                pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value) -> None:
        if key not in defaults():
            raise KeyError(f"Unknown setting: {key}")
        self.data[key] = value
        self.save()
