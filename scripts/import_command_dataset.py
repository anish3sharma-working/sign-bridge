"""Organize and import the labeled command videos from archive (5).zip.

Only Video_Dataset entries are used; the small Sample Videos preview folder is
excluded to avoid mixing it into the training set. Videos are extracted into a
local ignored working directory, then referenced by SignBridge metadata. The
original ZIP and raw data remain unchanged.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import app_data_dir
from database.db import Database

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SOURCE_GROUP = "commands-archive5-no-session-metadata"
PREFIX = "Video_Dataset/Video_Dataset/"


def canonical_label(raw: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", raw.strip().casefold()).strip("_")
    if not clean:
        raise ValueError(f"Could not create a safe command label from {raw!r}")
    return clean


def indexed_entries(zip_path: Path) -> list[tuple[zipfile.ZipInfo, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        result = []
        for entry in archive.infolist():
            name = entry.filename.replace("\\", "/")
            if entry.is_dir() or not name.startswith(PREFIX):
                continue
            relative = name[len(PREFIX):]
            parts = relative.split("/")
            if len(parts) != 2 or Path(parts[1]).suffix.casefold() not in VIDEO_EXTENSIONS:
                continue
            result.append((entry, canonical_label(parts[0])))
        return result


def organize(zip_path: Path, output_root: Path, dry_run: bool = False) -> Counter[str]:
    entries = indexed_entries(zip_path)
    counts: Counter[str] = Counter(label for _, label in entries)
    if dry_run:
        return counts
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for entry, label in entries:
            folder = output_root / label
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / Path(entry.filename).name
            if target.exists() and target.stat().st_size == entry.file_size:
                continue
            with archive.open(entry) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    return counts


def import_metadata(root: Path, database_path: Path | None = None) -> Counter[str]:
    database = Database(database_path or (app_data_dir() / "signbridge.sqlite3"))
    existing = {row["label"].casefold(): row for row in database.list_classes()}
    counts: Counter[str] = Counter()
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        label = canonical_label(folder.name)
        row = existing.get(label.casefold())
        if row is None:
            database.add_class(label, "Static command class imported from archive (5).zip", recognition_mode="commands")
        elif row.get("recognition_mode", "commands") != "commands":
            raise ValueError(f"Existing class '{label}' is not marked as commands; refusing to change it automatically.")
        for video in sorted(folder.iterdir(), key=lambda path: path.name.casefold()):
            if not video.is_file() or video.suffix.casefold() not in VIDEO_EXTENSIONS:
                continue
            try:
                database.add_sample(label, video, "video", SOURCE_GROUP)
                counts[label] += 1
            except Exception as exc:
                if "UNIQUE constraint failed" in str(exc):
                    continue
                raise
    return counts


def print_counts(title: str, counts: Counter[str]) -> None:
    print(f"{title} {sum(counts.values())} videos across {len(counts)} command classes.")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("data/raw/archive (5).zip"))
    parser.add_argument("--root", type=Path, default=Path(".work/organized-dataset/commands"))
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Inspect entries without extraction or database changes.")
    parser.add_argument("--extract-only", action="store_true", help="Extract command videos but do not write database metadata.")
    args = parser.parse_args()
    zip_path = args.zip.resolve()
    if not zip_path.is_file():
        raise ValueError(f"Command archive does not exist: {zip_path}")
    counts = organize(zip_path, args.root.resolve(), args.dry_run)
    print_counts("Validated" if args.dry_run else "Organized", counts)
    if not args.dry_run and not args.extract_only:
        imported = import_metadata(args.root.resolve(), args.database)
        print_counts("Imported", imported)
    print(f"Source group: {SOURCE_GROUP}")
    print("Warning: archive (5).zip has no participant/session metadata; independent evaluation remains unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
