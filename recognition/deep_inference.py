"""Preprocessing and prediction helpers for the separate CNN and GRU models."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from recognition.image_model import IMAGE_SIZE
from recognition.sequence_model import SEQUENCE_LENGTH


def predict_image_frame(model, classes: list[str], frame_bgr) -> tuple[str | None, float, list[tuple[str, float]]]:
    import torch
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    ranking = sorted(((classes[i], float(score)) for i, score in enumerate(probabilities)), key=lambda item: item[1], reverse=True)
    candidates = ranking[:3]
    label, confidence = candidates[0]
    return label, confidence, candidates


def make_sequence_window(buffer: deque[np.ndarray] | list[np.ndarray]) -> np.ndarray | None:
    if len(buffer) < SEQUENCE_LENGTH:
        return None
    values = list(buffer)[-SEQUENCE_LENGTH:]
    return np.asarray(values, dtype=np.float32)


def predict_sequence_window(model, classes: list[str], sequence: np.ndarray) -> tuple[str | None, float, list[tuple[str, float]]]:
    import torch
    tensor = torch.from_numpy(np.asarray(sequence, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    ranking = sorted(((classes[i], float(score)) for i, score in enumerate(probabilities)), key=lambda item: item[1], reverse=True)
    candidates = ranking[:3]
    label, confidence = candidates[0]
    return label, confidence, candidates
