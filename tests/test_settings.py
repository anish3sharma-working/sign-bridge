import pytest
from config.settings import Settings


def test_settings_persist_and_keep_defaults(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(path)
    settings.set("camera_index", 2)
    loaded = Settings(path)
    assert loaded.get("camera_index") == 2
    assert loaded.get("theme") == "dark"
    assert loaded.get("inference_fps") == 15
    assert loaded.get("mirrored_input") is True
    assert loaded.get("swap_handedness") is True
    assert loaded.get("settings_version") == 3
    loaded.set("inference_fps", 20)
    assert Settings(path).get("inference_fps") == 20
    loaded.set("mirrored_input", True)
    assert Settings(path).get("mirrored_input") is True
    loaded.set("swap_handedness", False)
    assert Settings(path).get("swap_handedness") is False
    with pytest.raises(KeyError): loaded.set("unexpected", True)


def test_legacy_handedness_setting_is_migrated_to_pixel_flip_semantics(tmp_path):
    path = tmp_path / "legacy-settings.json"
    path.write_text('{"mirrored_input": false}', encoding="utf-8")
    loaded = Settings(path)
    assert loaded.get("mirrored_input") is True
    assert loaded.get("swap_handedness") is True
    assert loaded.get("settings_version") == 3
    assert '"settings_version": 3' in path.read_text(encoding="utf-8")
