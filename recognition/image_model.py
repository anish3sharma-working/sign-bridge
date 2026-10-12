"""PyTorch CNN artifact handling for grayscale alphabet image recognition."""
from __future__ import annotations

import json
from pathlib import Path

IMAGE_MODEL_VERSION = 1
IMAGE_SIZE = 128


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the CNN image model. Install the optional deep-learning dependencies.") from exc
    return torch, nn


def build_alphabet_cnn(num_classes: int):
    torch, nn = _torch()

    class AlphabetCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(192 * 16, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, num_classes))

        def forward(self, x):
            return self.classifier(self.features(x))

    return AlphabetCNN()


def save_image_model(path: str | Path, model, classes: list[str], metrics: dict) -> Path:
    torch, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_version": IMAGE_MODEL_VERSION, "image_size": IMAGE_SIZE, "channels": 1, "classes": list(classes), "state_dict": model.state_dict()}, path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps({"model_version": IMAGE_MODEL_VERSION, "image_size": IMAGE_SIZE, "channels": 1, "classes": list(classes), "metrics": metrics, "preprocessing": "BGR/RGB input converted to grayscale, resized to 128x128, scaled to [0,1]"}, indent=2), encoding="utf-8")
    return path


def load_image_model(path: str | Path):
    torch, _ = _torch()
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_version") != IMAGE_MODEL_VERSION or payload.get("image_size") != IMAGE_SIZE or payload.get("channels") != 1:
        raise ValueError("Image model version or preprocessing contract is incompatible.")
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Image model is missing its class mapping.")
    model = build_alphabet_cnn(len(classes))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, classes
