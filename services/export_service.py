"""Local-only text export helpers."""
from pathlib import Path
from services.speech_service import normalize_transcript


def export_transcript(text: str, destination: str | Path) -> Path:
    path = Path(destination)
    if path.suffix.lower() != ".txt":
        raise ValueError("Transcript export currently supports plain .txt files only.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalize_transcript(text), encoding="utf-8")
    return path
