"""Optional offline speech through an installed operating-system voice."""
from __future__ import annotations
import re
import threading


def normalize_transcript(text: str) -> str:
    """Light formatting only; does not translate, infer, or add words."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


class SpeechService:
    def __init__(self, rate: int = 165):
        self.rate = int(rate)
        self._engine = None
        self._thread = None
        self._lock = threading.Lock()

    @staticmethod
    def availability() -> tuple[bool, str]:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices") or []
            engine.stop()
            if not voices:
                return False, "No local operating-system speech voice is available."
            return True, "Local speech engine and voice are available."
        except Exception as exc:
            return False, f"Local speech is unavailable: {exc}"

    def speak(self, text: str, on_error=None) -> bool:
        content = normalize_transcript(text)
        if not content:
            if on_error: on_error("The transcript is empty.")
            return False
        if self._thread and self._thread.is_alive():
            if on_error: on_error("Speech is already in progress. Stop it before speaking again.")
            return False

        def run():
            engine = None
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                if not engine.getProperty("voices"):
                    raise RuntimeError("No local system voice is installed.")
                with self._lock: self._engine = engine
                engine.say(content)
                engine.runAndWait()
            except Exception as exc:
                if on_error: on_error(f"Local speech failed: {exc}")
            finally:
                with self._lock:
                    if self._engine is engine: self._engine = None

        self._thread = threading.Thread(target=run, name="SignBridgeSpeech", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        with self._lock: engine = self._engine
        if engine is not None:
            try: engine.stop()
            except Exception: pass
