"""SQLite persistence for SignBridge. All user data remains on this computer."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS gesture_classes (
                    id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    description TEXT NOT NULL DEFAULT '',
                    gesture_type TEXT NOT NULL DEFAULT 'static',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_samples (
                    id INTEGER PRIMARY KEY,
                    class_id INTEGER NOT NULL REFERENCES gesture_classes(id) ON DELETE CASCADE,
                    source_path TEXT NOT NULL,
                    sample_type TEXT NOT NULL CHECK(sample_type IN ('image','video')),
                    recording_group TEXT NOT NULL,
                    participant_group TEXT NOT NULL DEFAULT '',
                    dataset_split TEXT NOT NULL DEFAULT 'unassigned',
                    created_at TEXT NOT NULL,
                    UNIQUE(source_path, class_id)
                );
                CREATE INDEX IF NOT EXISTS idx_samples_class ON dataset_samples(class_id);
                CREATE TABLE IF NOT EXISTS transcript_sessions (
                    id INTEGER PRIMARY KEY,
                    transcript_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    class_mapping TEXT NOT NULL,
                    evaluation_summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                PRAGMA user_version = 1;
            """)

    def add_class(self, label: str, description: str = "", gesture_type: str = "static") -> int:
        clean = " ".join(label.strip().split())
        if not clean or len(clean) > 60:
            raise ValueError("Enter a gesture label between 1 and 60 characters.")
        if clean in {".", ".."} or any(ch in clean for ch in '\\/:<>|?*"'):
            raise ValueError("Gesture labels cannot contain path separators or filesystem-reserved characters.")
        if gesture_type not in {"static", "dynamic"}:
            raise ValueError("Gesture type must be static or dynamic.")
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO gesture_classes(label, description, gesture_type, created_at) VALUES(?,?,?,?)",
                (clean, description.strip(), gesture_type, utc_now()),
            )
            return int(cur.lastrowid)

    def delete_class(self, class_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM gesture_classes WHERE id = ?", (class_id,))

    def list_classes(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("""SELECT c.id, c.label, c.description, c.gesture_type, c.created_at,
                COUNT(s.id) AS sample_count FROM gesture_classes c
                LEFT JOIN dataset_samples s ON s.class_id=c.id
                GROUP BY c.id ORDER BY c.label COLLATE NOCASE""").fetchall()
        return [dict(row) for row in rows]

    def add_sample(self, label: str, source_path: str | Path, sample_type: str,
                   recording_group: str, participant_group: str = "") -> int:
        if sample_type not in {"image", "video"}:
            raise ValueError("Sample type must be image or video.")
        path = str(Path(source_path).resolve())
        group = recording_group.strip()
        if not group:
            raise ValueError("A recording group is required to prevent split leakage.")
        with self.connect() as conn:
            cls = conn.execute("SELECT id FROM gesture_classes WHERE label = ? COLLATE NOCASE", (label.strip(),)).fetchone()
            if not cls:
                raise ValueError(f"Gesture class not found: {label}")
            cur = conn.execute("""INSERT INTO dataset_samples
                (class_id, source_path, sample_type, recording_group, participant_group, created_at)
                VALUES(?,?,?,?,?,?)""", (cls["id"], path, sample_type, group, participant_group.strip(), utc_now()))
            return int(cur.lastrowid)

    def list_samples(self, class_id: int | None = None) -> list[dict[str, Any]]:
        query = """SELECT s.*, c.label FROM dataset_samples s JOIN gesture_classes c ON c.id=s.class_id"""
        params: tuple[Any, ...] = ()
        if class_id is not None:
            query += " WHERE s.class_id = ?"
            params = (class_id,)
        query += " ORDER BY s.created_at DESC, s.id DESC"
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def delete_sample(self, sample_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM dataset_samples WHERE id = ?", (sample_id,))

    def assign_dataset_splits(self, assignments: dict[int, str]) -> None:
        allowed = {"train", "validation", "test"}
        if any(partition not in allowed for partition in assignments.values()):
            raise ValueError("Dataset split must be train, validation, or test.")
        with self.connect() as conn:
            conn.execute("UPDATE dataset_samples SET dataset_split = 'unassigned'")
            conn.executemany("UPDATE dataset_samples SET dataset_split = ? WHERE id = ?",
                             [(partition, int(sample_id)) for sample_id, partition in assignments.items()])

    def save_transcript(self, text: str) -> int:
        with self.connect() as conn:
            cur = conn.execute("INSERT INTO transcript_sessions(transcript_text, created_at) VALUES(?,?)", (text, utc_now()))
            return int(cur.lastrowid)

    def list_transcripts(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM transcript_sessions ORDER BY id DESC").fetchall()]

    def delete_transcript(self, transcript_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM transcript_sessions WHERE id = ?", (transcript_id,))

    def clear_transcripts(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM transcript_sessions")

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO app_settings(setting_key,setting_value) VALUES(?,?) "
                         "ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value",
                         (key, json.dumps(value)))

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute("SELECT setting_value FROM app_settings WHERE setting_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def record_model(self, name: str, path: str, feature_version: str,
                     class_mapping: list[str], metrics: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO models(model_name,model_path,feature_version,class_mapping,evaluation_summary,created_at) VALUES(?,?,?,?,?,?)",
                         (name, path, feature_version, json.dumps(class_mapping), json.dumps(metrics), utc_now()))

    def latest_model(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM models ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None
