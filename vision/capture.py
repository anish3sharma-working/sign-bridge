"""Windows camera backend fallback with an initial frame check."""


def open_capture(cv2, index: int, width: int = 640, height: int = 480):
    """Return (capture, first_frame, backend_name), or (None, None, None).

    Each failed capture handle is released before trying the next backend.
    """
    candidates = []
    for name in ("CAP_DSHOW", "CAP_MSMF"):
        backend = getattr(cv2, name, None)
        if backend is not None and backend not in [entry[1] for entry in candidates]:
            candidates.append((name.removeprefix("CAP_"), backend))
    default = getattr(cv2, "CAP_ANY", 0)
    if default not in [entry[1] for entry in candidates]:
        candidates.append(("default", default))

    for name, backend in candidates:
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame, name
        except Exception:
            pass
        if cap is not None:
            try: cap.release()
            except Exception: pass
    return None, None, None
