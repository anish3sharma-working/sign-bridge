import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")
from training.dataset_validation import validate_media


def test_validate_image_and_bad_extension(tmp_path):
    image_path = tmp_path / "ok.png"
    cv2.imwrite(str(image_path), np.zeros((24, 24, 3), dtype=np.uint8))
    assert validate_media(image_path, "A")[0] is True
    bad = tmp_path / "data.bin"; bad.write_bytes(b"no")
    valid, message, kind = validate_media(bad, "A")
    assert not valid and "Unsupported" in message and kind is None


def test_validate_requires_label_and_file(tmp_path):
    valid, message, _ = validate_media(tmp_path / "missing.png", "")
    assert not valid and "label" in message
