from vision.capture import open_capture


class _FakeFrame:
    pass


class _FakeCapture:
    def __init__(self, opened, frame_ok=True):
        self.opened = opened
        self.frame_ok = frame_ok
        self.released = False
        self.properties = []

    def isOpened(self): return self.opened
    def set(self, name, value): self.properties.append((name, value)); return True
    def read(self): return self.frame_ok, _FakeFrame() if self.frame_ok else None
    def release(self): self.released = True


class _FakeCV:
    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_ANY = 0
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 5

    def __init__(self, captures):
        self.captures = iter(captures)
        self.backends = []

    def VideoCapture(self, index, backend):
        self.backends.append((index, backend))
        return next(self.captures)


def test_failed_camera_backends_are_released_before_fallback():
    first = _FakeCapture(False)
    second = _FakeCapture(True)
    cv = _FakeCV([first, second])
    cap, frame, backend = open_capture(cv, 2, 800, 600)
    assert cap is second
    assert frame is not None
    assert backend == "MSMF"
    assert first.released
    assert not second.released
    assert cv.backends == [(2, cv.CAP_DSHOW), (2, cv.CAP_MSMF)]
    cap.release()


def test_open_backend_without_frame_is_released_and_rejected():
    capture = _FakeCapture(True, frame_ok=False)
    cv = _FakeCV([capture, _FakeCapture(False), _FakeCapture(False)])
    assert open_capture(cv, 0) == (None, None, None)
    assert capture.released
