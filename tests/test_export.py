import pytest
from services.export_service import export_transcript


def test_export_text(tmp_path):
    path = export_transcript(" hello   , world ! ", tmp_path / "transcript.txt")
    assert path.read_text(encoding="utf-8") == "hello, world!"
    with pytest.raises(ValueError): export_transcript("hi", tmp_path / "transcript.pdf")
