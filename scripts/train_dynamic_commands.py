"""Train a separate GRU on temporal hand-landmark sequences from command videos."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import app_data_dir
from database.db import Database
from recognition.sequence_model import SEQUENCE_LENGTH, build_sequence_gru, save_sequence_model
from vision.landmarks import FEATURE_DIM, LandmarkExtractor


def resample_sequence(sequence: list[np.ndarray]) -> np.ndarray:
    if not sequence:
        raise ValueError("No frames with detectable hands")
    if len(sequence) == SEQUENCE_LENGTH:
        return np.asarray(sequence, dtype=np.float32)
    indices = np.linspace(0, len(sequence) - 1, SEQUENCE_LENGTH).round().astype(int)
    return np.asarray([sequence[index] for index in indices], dtype=np.float32)


def extract_video(path: Path, extractor: LandmarkExtractor) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path.name}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            feature, _, _ = extractor.process(frame)
            if feature is not None:
                frames.append(feature)
    finally:
        cap.release()
    return resample_sequence(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all command videos")
    parser.add_argument("--min-hand-confidence", type=float, default=0.35)
    parser.add_argument("--allow-exploratory-split", action="store_true")
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    database = Database(args.database or (app_data_dir() / "signbridge.sqlite3"))
    rows = database.list_samples()
    allowed = {row["label"].casefold() for row in database.list_classes() if row.get("recognition_mode") == "commands"}
    rows = [row for row in rows if row["label"].casefold() in allowed and Path(row["source_path"]).suffix.casefold() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}]
    if args.max_videos:
        rows = rows[:args.max_videos]
    if not rows:
        raise ValueError("No command videos are registered in the local database.")
    classes = sorted({row["label"] for row in rows})
    mapping = {label: index for index, label in enumerate(classes)}
    extractor = LandmarkExtractor(static_image_mode=False, max_hands=2, min_hand_confidence=args.min_hand_confidence, grayscale_input=True)
    sequences, labels, groups, failures = [], [], [], []
    try:
        for index, row in enumerate(rows, 1):
            try:
                sequences.append(extract_video(Path(row["source_path"]), extractor)); labels.append(mapping[row["label"]]); groups.append(row["participant_group"].strip() or row["recording_group"])
            except Exception as exc:
                failures.append({"file": row["source_path"], "label": row["label"], "reason": str(exc)})
            if index % 10 == 0 or index == len(rows):
                print(f"processed {index}/{len(rows)} videos; usable={len(sequences)} failures={len(failures)}", flush=True)
    finally:
        extractor.close()
    if len(set(labels)) < 2:
        raise ValueError(f"At least two command classes need usable sequences. failures={len(failures)}")
    X = torch.tensor(np.asarray(sequences), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    try:
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=np.asarray(labels)[train_idx], random_state=43)
    except ValueError as exc:
        if not args.allow_exploratory_split:
            raise ValueError("Video-level split could not be created. Use --allow-exploratory-split only for a technical check.") from exc
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=np.asarray(labels)[train_idx], random_state=43)
    model = build_sequence_gru(len(classes))
    loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train(); seen = 0; correct = 0; loss_total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(); logits = model(batch_x); loss = criterion(logits, batch_y); loss.backward(); optimizer.step()
            loss_total += float(loss) * len(batch_y); correct += int((logits.argmax(1) == batch_y).sum()); seen += len(batch_y)
        model.eval()
        with torch.no_grad(): val_acc = float((model(X[val_idx]).argmax(1) == y[val_idx]).float().mean())
        print(f"epoch {epoch}/{args.epochs}: train_accuracy={correct/max(seen,1):.4f} val_accuracy={val_acc:.4f} loss={loss_total/max(seen,1):.4f}", flush=True)
    model.eval()
    with torch.no_grad(): test_acc = float((model(X[test_idx]).argmax(1) == y[test_idx]).float().mean())
    metrics = {"model_type": "dynamic_command_gru", "classes": classes, "usable_videos": len(sequences), "failed_videos": failures, "train_count": len(train_idx), "validation_count": len(val_idx), "test_count": len(test_idx), "test_accuracy": test_acc, "warning": "Video-level labels are retained, but archive participant/session metadata is unavailable; metrics are exploratory."}
    output = args.output or (app_data_dir() / "models" / "signbridge_commands_dynamic_gru_v1.pt")
    save_sequence_model(output, model, classes, metrics)
    print(f"saved {output}")
    print(f"test_accuracy={test_acc:.4f}; warning={metrics['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
