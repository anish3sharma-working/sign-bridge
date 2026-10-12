"""PyTorch GRU artifact handling for temporal hand-landmark sequences."""
from __future__ import annotations

import json
from pathlib import Path

SEQUENCE_MODEL_VERSION = 1
SEQUENCE_LENGTH = 32
SEQUENCE_FEATURE_DIM = 136


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the dynamic sequence model. Install the optional deep-learning dependencies.") from exc
    return torch, nn


def build_sequence_gru(num_classes: int):
    _, nn = _torch()

    class CommandGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(SEQUENCE_FEATURE_DIM, 128, num_layers=2, batch_first=True, dropout=0.2)
            self.classifier = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.25), nn.Linear(128, num_classes))

        def forward(self, x):
            _, hidden = self.gru(x)
            return self.classifier(hidden[-1])

    return CommandGRU()


def save_sequence_model(path: str | Path, model, classes: list[str], metrics: dict) -> Path:
    torch, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_version": SEQUENCE_MODEL_VERSION, "sequence_length": SEQUENCE_LENGTH, "feature_dim": SEQUENCE_FEATURE_DIM, "classes": list(classes), "state_dict": model.state_dict()}, path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps({"model_version": SEQUENCE_MODEL_VERSION, "sequence_length": SEQUENCE_LENGTH, "feature_dim": SEQUENCE_FEATURE_DIM, "classes": list(classes), "metrics": metrics, "preprocessing": "all video frames decoded; bilateral landmarks extracted in temporal order; sequences uniformly resampled/padded to 32 frames"}, indent=2), encoding="utf-8")
    return path


def load_sequence_model(path: str | Path):
    torch, _ = _torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("model_version") != SEQUENCE_MODEL_VERSION or payload.get("sequence_length") != SEQUENCE_LENGTH or payload.get("feature_dim") != SEQUENCE_FEATURE_DIM:
        raise ValueError("Sequence model version or temporal preprocessing contract is incompatible.")
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Sequence model is missing its class mapping.")
    model = build_sequence_gru(len(classes))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, classes
