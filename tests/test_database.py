from database.db import Database


def test_database_crud_and_cascade(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    class_id = db.add_class("Hello", "reviewed label")
    assert db.list_classes()[0]["label"] == "Hello"
    assert db.list_classes()[0]["sample_count"] == 0
    sample = tmp_path / "sample.jpg"; sample.write_bytes(b"image")
    db.add_sample("Hello", sample, "image", "recording-1", "participant-a")
    assert db.list_classes()[0]["sample_count"] == 1
    assert db.list_samples()[0]["participant_group"] == "participant-a"
    sample_id = db.list_samples()[0]["id"]
    db.assign_dataset_splits({sample_id: "train"})
    assert db.list_samples()[0]["dataset_split"] == "train"
    db.delete_class(class_id)
    assert db.list_classes() == []
    assert db.list_samples() == []


def test_transcript_and_parameterized_settings(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    ident = db.save_transcript("it's a test")
    assert db.list_transcripts()[0]["transcript_text"] == "it's a test"
    db.set_setting("example", {"safe": "value"})
    assert db.get_setting("example") == {"safe": "value"}
    db.delete_transcript(ident)
    assert not db.list_transcripts()


def test_class_label_rejects_path_separators(tmp_path):
    import pytest
    db = Database(tmp_path / "test.sqlite3")
    with pytest.raises(ValueError, match="path separators"):
        db.add_class("../outside")
