"""Measure image/video decode and MediaPipe landmark usability before training."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import app_data_dir
from database.db import Database
from vision.landmarks import LandmarkExtractor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def check_image(path: Path, extractor: LandmarkExtractor) -> tuple[bool, str]:
    image = cv2.imread(str(path))
    if image is None:
        return False, "decode_failed"
    try:
        feature, _, _ = extractor.process(image)
        return (feature is not None, "ok" if feature is not None else "no_hands")
    except Exception as exc:
        return False, type(exc).__name__


def check_video(path: Path, extractor: LandmarkExtractor) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "open_failed"
    frames = 0; detected = 0
    try:
        while frames < 300:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            feature, _, _ = extractor.process(frame)
            if feature is not None:
                detected += 1
    finally:
        cap.release()
    return (detected > 0, f"frames={frames};detected={detected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--alphabet-root", type=Path, default=Path(".work/organized-dataset/alphabet"))
    parser.add_argument("--output", type=Path, default=Path(".work/diagnostics/dataset_report.json"))
    parser.add_argument("--max-per-class", type=int, default=30)
    args = parser.parse_args()
    report = {"alphabet": {"classes": {}, "total_checked": 0, "usable": 0}, "commands": {"classes": {}, "total_checked": 0, "usable": 0}}
    extractor = LandmarkExtractor(static_image_mode=True, max_hands=2, min_hand_confidence=0.35, grayscale_input=True)
    try:
        for folder in sorted(path for path in args.alphabet_root.resolve().iterdir() if path.is_dir()):
            files = [p for p in sorted(folder.iterdir()) if p.suffix.casefold() in IMAGE_EXTENSIONS][:args.max_per_class]
            counts = Counter()
            for path in files:
                ok, reason = check_image(path, extractor); counts[reason] += 1
            report["alphabet"]["classes"][folder.name] = dict(counts)
            report["alphabet"]["total_checked"] += len(files); report["alphabet"]["usable"] += counts["ok"]
    finally:
        extractor.close()
    database = Database(args.database or (app_data_dir() / "signbridge.sqlite3"))
    rows = [row for row in database.list_samples() if Path(row["source_path"]).suffix.casefold() in VIDEO_EXTENSIONS]
    extractor = LandmarkExtractor(static_image_mode=False, max_hands=2, min_hand_confidence=0.35, grayscale_input=True)
    try:
        grouped = defaultdict(list)
        for row in rows:
            if len(grouped[row["label"]]) < args.max_per_class: grouped[row["label"]].append(row)
        for label, items in sorted(grouped.items()):
            counts = Counter()
            for row in items:
                ok, reason = check_video(Path(row["source_path"]), extractor); counts["usable" if ok else "failed"] += 1; counts[reason] += 1
            report["commands"]["classes"][label] = dict(counts)
            report["commands"]["total_checked"] += len(items); report["commands"]["usable"] += counts["usable"]
    finally:
        extractor.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "alphabet": report["alphabet"], "commands": report["commands"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
