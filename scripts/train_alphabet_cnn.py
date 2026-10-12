"""Train a standalone grayscale CNN on the organized a-z image dataset.

This model is intentionally separate from the MediaPipe landmark Random Forest.
It learns image appearance and uses the same grayscale/128x128 preprocessing at
inference time.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognition.image_model import IMAGE_SIZE, build_alphabet_cnn, save_image_model


def load_dataset(root: Path):
    paths, labels = [], []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        label = folder.name.casefold()
        if len(label) != 1 or not label.isalpha():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                paths.append(path); labels.append(label)
    classes = sorted(set(labels))
    if classes != list("abcdefghijklmnopqrstuvwxyz"):
        raise ValueError(f"Expected a-z classes; found {classes}")
    return paths, labels, classes


def image_tensor(path: Path):
    import torch
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".work/organized-dataset/alphabet"))
    parser.add_argument("--output", type=Path, default=Path("%LOCALAPPDATA%/SignBridge/models/signbridge_alphabet_cnn_v1.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root = args.root.resolve()
    paths, labels, classes = load_dataset(root)
    mapping = {label: index for index, label in enumerate(classes)}
    usable, y = [], []
    failures = []
    for path, label in zip(paths, labels):
        try:
            usable.append(image_tensor(path)); y.append(mapping[label])
        except Exception as exc:
            failures.append({"path": str(path), "reason": str(exc)})
    if len(set(y)) != len(classes):
        raise ValueError("At least one alphabet class has no decodable images.")
    X = torch.stack(usable)
    y_tensor = torch.tensor(y, dtype=torch.long)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=y_tensor[train_idx].numpy(), random_state=args.seed + 1)
    train_loader = DataLoader(TensorDataset(X[train_idx], y_tensor[train_idx]), batch_size=args.batch_size, shuffle=True)
    model = build_alphabet_cnn(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train(); seen = 0; correct = 0; loss_total = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(); logits = model(batch_x); loss = criterion(logits, batch_y); loss.backward(); optimizer.step()
            loss_total += float(loss) * len(batch_y); correct += int((logits.argmax(1) == batch_y).sum()); seen += len(batch_y)
        model.eval()
        with torch.no_grad():
            val_pred = model(X[val_idx]).argmax(1)
            val_acc = float((val_pred == y_tensor[val_idx]).float().mean())
        print(f"epoch {epoch}/{args.epochs}: train_accuracy={correct/max(seen,1):.4f} val_accuracy={val_acc:.4f} loss={loss_total/max(seen,1):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        test_pred = model(X[test_idx]).argmax(1)
        test_acc = float((test_pred == y_tensor[test_idx]).float().mean())
    metrics = {"model_type": "alphabet_cnn", "classes": classes, "usable_images": len(usable), "failed_images": failures, "train_count": len(train_idx), "validation_count": len(val_idx), "test_count": len(test_idx), "test_accuracy": test_acc, "warning": "Random image split; no participant/session metadata was available, so this is not independent generalization evidence."}
    output = Path(str(args.output).replace("%LOCALAPPDATA%", str(Path.home() / "AppData/Local"))).resolve()
    save_image_model(output, model, classes, metrics)
    print(f"saved {output}")
    print(f"test_accuracy={test_acc:.4f}; warning={metrics['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
