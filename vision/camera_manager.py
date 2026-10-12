"""Webcam capture and inference in a stoppable worker with a one-frame UI handoff."""
from __future__ import annotations
import threading
import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from recognition.prediction_filter import PredictionFilter
from vision.landmarks import LandmarkExtractor
from vision.capture import open_capture


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    prediction_ready = Signal(object, float, object, str, float, float)
    error = Signal(str)
    status = Signal(str)
    hand_status = Signal(str)

    def __init__(self, camera_index=0, width=640, height=480, show_landmarks=True,
                 model=None, threshold=0.65, stable_frames=5, inference_fps=15, parent=None,
                 mirrored_input=False, min_hand_confidence=0.45, swap_handedness=True,
                 grayscale_input=True):
        super().__init__(parent)
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.show_landmarks = bool(show_landmarks)
        self.model = model
        self.filter = PredictionFilter(threshold, stable_frames)
        self.inference_interval = 1.0 / max(1, int(inference_fps))
        self.mirrored_input = bool(mirrored_input)
        self.min_hand_confidence = float(min_hand_confidence)
        self.swap_handedness = bool(swap_handedness)
        self.grayscale_input = bool(grayscale_input)
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._display_pending = False

    def stop(self):
        self._stop_event.set()
        self.wait(4000)

    def acknowledge_frame(self):
        with self._frame_lock:
            self._display_pending = False

    def latest_frame_copy(self):
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def run(self):
        cap = None
        first_frame = None
        extractor = None
        try:
            import cv2
            self.status.emit("Opening camera…")
            cap, first_frame, backend = open_capture(cv2, self.camera_index, self.width, self.height)
            if cap is None:
                raise RuntimeError(f"Camera {self.camera_index} could not provide a frame through DirectShow, Media Foundation, or the default backend. Check Windows camera permission and other apps.")
            try:
                extractor = LandmarkExtractor(static_image_mode=False, max_hands=2,
                                              mirrored_input=self.mirrored_input,
                                              min_hand_confidence=self.min_hand_confidence,
                                              swap_handedness=self.swap_handedness,
                                              grayscale_input=self.grayscale_input)
            except RuntimeError as exc:
                self.status.emit("Camera active; landmark recognition unavailable")
                self.error.emit(str(exc))
            self.status.emit(f"Camera active via {backend} — all processing stays on this computer")
            tick = time.perf_counter()
            last_inference = 0.0
            frames = 0
            while not self._stop_event.is_set():
                if first_frame is not None:
                    frame, first_frame = first_frame, None
                    ok = True
                else:
                    ok, frame = cap.read()
                if not ok:
                    if not self._stop_event.is_set(): self.error.emit("The camera stopped delivering frames. Check whether another app is using it.")
                    break
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                raw_frame = frame
                display_frame = cv2.flip(raw_frame, 1) if self.mirrored_input else raw_frame.copy()
                inference_ms = 0.0
                label, confidence, candidates, state = None, 0.0, [], "No trained model"
                if extractor is not None:
                    now = time.perf_counter()
                    if now - last_inference >= self.inference_interval:
                        last_inference = now
                        begin = time.perf_counter()
                        try:
                            feature, display_frame, hands = extractor.process(raw_frame, draw=self.show_landmarks)
                            inference_ms = (time.perf_counter() - begin) * 1000
                            labels = {item["side"]: item["score"] for item in hands}
                            if labels:
                                found = " · ".join(f"{side}: detected ({score:.0%})" for side, score in sorted(labels.items()))
                                missing = [side for side in ("Left", "Right") if side not in labels]
                                if missing: found += " · " + " · ".join(f"{side}: not detected" for side in missing)
                                self.hand_status.emit(found)
                            else:
                                self.hand_status.emit("No hands detected — move hands into view and improve lighting")
                            if feature is None:
                                self.filter.reset(); state = "Show one or both hands clearly"
                            elif self.model is None:
                                state = "No trained model — collect data and train first"
                            else:
                                label, confidence, candidates = self.model.predict(feature)
                                filtered = self.filter.update(label, confidence)
                                state = "Stable — confirm to add" if filtered["accepted"] else ("Sign uncertain — please repeat" if not label else "Stabilizing")
                                if not filtered["accepted"]: label = None
                        except Exception as exc:
                            self.filter.reset(); state = "Landmark processing error"; self.error.emit(f"Landmark processing failed: {exc}")
                    else:
                        state = "Waiting for next inference frame"
                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                with self._frame_lock:
                    if not self._display_pending:
                        self._display_pending = True
                        self.frame_ready.emit(image)
                frames += 1
                now = time.perf_counter()
                fps = frames / max(now - tick, 1e-3)
                self.prediction_ready.emit(label, float(confidence), candidates, state, inference_ms, fps)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if extractor is not None:
                try: extractor.close()
                except Exception: pass
            if cap is not None: cap.release()
            self.hand_status.emit("Camera stopped — hand detection is off")
            self.status.emit("Camera stopped")
