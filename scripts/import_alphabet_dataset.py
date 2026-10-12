"""Bulk-import the organized a-z alphabet dataset into local SignBridge metadata.

The source images are referenced in place; they are not copied into the app
folder and the source ZIP/raw data are never modified. This importer records a
single provenance group because the archive contains no participant or session
metadata. The current leakage-safe trainer will therefore refuse to claim
independent validation until genuine group metadata is supplied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import app_data_dir
from database.db import Database

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_GROUP = "alphabet-archive6-no-session-metadata"


def import_alphabet(root: Path, database_path: Path | None = None, dry_run: bool = False) -> dict[str, int]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Alphabet dataset folder does not exist: {root}")
    folders = sorted((path for path in root.iterdir() if path.is_dir() and len(path.name) == 1 and path.name.isalpha()), key=lambda p: p.name.casefold())
    if [path.name.casefold() for path in folders] != list("abcdefghijklmnopqrstuvwxyz"):
        raise ValueError("Expected exactly the 26 alphabet folders a through z.")

    counts = {path.name.casefold(): sum(1 for item in path.iterdir() if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS) for path in folders}
    if not all(counts.values()):
        missing = [label for label, count in counts.items() if not count]
        raise ValueError("Alphabet folders without supported images: " + ", ".join(missing))
    if dry_run:
        return counts

    database = Database(database_path or (app_data_dir() / "signbridge.sqlite3"))
    existing = {row["label"].casefold(): row for row in database.list_classes()}
    imported = {label: 0 for label in counts}
    for folder in folders:
        label = folder.name.casefold()
        row = existing.get(label)
        if row is None:
            database.add_class(label, "Alphabet class imported from archive (6).zip", recognition_mode="alphabet")
        elif row.get("recognition_mode", "commands") != "alphabet":
            raise ValueError(f"Existing class '{label}' is not marked as alphabet; refusing to change it automatically.")
        for image in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
            if not image.is_file() or image.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            try:
                database.add_sample(label, image, "image", SOURCE_GROUP)
                imported[label] += 1
            except Exception as exc:
                if "UNIQUE constraint failed" in str(exc):
                    continue
                raise
    return imported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".work/organized-dataset/alphabet"))
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate folders and counts without changing the local database.")
    args = parser.parse_args()
    counts = import_alphabet(args.root, args.database, args.dry_run)
    total = sum(counts.values())
    action = "Validated" if args.dry_run else "Imported"
    print(f"{action} {total} alphabet images across {len(counts)} classes.")
    print(f"Source group: {SOURCE_GROUP}")
    print("Warning: the source archive has no participant/session metadata; independent evaluation remains unavailable.")
    for label, count in counts.items():
        print(f"{label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
