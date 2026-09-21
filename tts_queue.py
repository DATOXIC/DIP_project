"""Bộ đọc âm thanh tuần tự, không chặn luồng chính."""

import queue
import threading
from typing import Callable, Optional, Tuple


_QueueItem = Tuple[str, str, Optional[str]]


class QueuedSpeaker:
    """Xếp lời đọc và WAV vào một worker duy nhất theo đúng thứ tự."""

    def __init__(
        self,
        rate: int = 175,
        max_queue: int = 4,
        speak_fn: Optional[Callable[[str], None]] = None,
    ):
        self.rate = rate
        self.max_queue = max(1, int(max_queue))
        self._speak_fn = speak_fn
        self._q: "queue.Queue[Optional[_QueueItem]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = False
        self._engine = None

    def speak(self, text: str, voice_id: Optional[str] = None) -> bool:
        if not isinstance(text, str):
            return False
        text = text.strip()
        if not text or self._closed:
            return False
        return self._enqueue(("text", text, voice_id))

    def play_wav(self, path: str) -> bool:
        if not isinstance(path, str):
            return False
        path = path.strip()
        if not path or self._closed:
            return False
        return self._enqueue(("wav", path, None))

    def _enqueue(self, item: _QueueItem) -> bool:
        with self._lock:
            if self._closed:
                return False
            while self._q.qsize() >= self.max_queue:
                try:
                    old_item = self._q.get_nowait()
                except queue.Empty:
                    break
                if old_item is None:
                    continue
            self._q.put(item)
            self._ensure_thread_locked()
        return True

    def _ensure_thread_locked(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _speak_one(self, text: str, voice_id: Optional[str]) -> None:
        try:
            import pyttsx3

            if self._engine is None:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
            if voice_id:
                self._engine.setProperty("voice", voice_id)
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            self._reset_engine()
            raise

    def _reset_engine(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None:
            try:
                if hasattr(engine, "_inLoop") and engine._inLoop:
                    engine._inLoop = False
                engine.stop()
            except Exception:
                pass
        try:
            import pyttsx3

            if hasattr(pyttsx3, "_activeEngines"):
                pyttsx3._activeEngines.clear()
        except Exception:
            pass

    @staticmethod
    def _play_wav(path: str) -> None:
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME)

    def _worker(self) -> None:
        com_ok = False
        try:
            try:
                import pythoncom

                pythoncom.CoInitialize()
                com_ok = True
            except Exception:
                pass
            while True:
                item = self._q.get()
                if item is None:
                    break
                kind, value, voice_id = item
                try:
                    if kind == "wav":
                        self._play_wav(value)
                    elif self._speak_fn is not None:
                        self._speak_fn(value)
                    else:
                        self._speak_one(value, voice_id)
                except Exception as err:
                    print(f"[TTS] Lỗi phát âm '{value}': {err}")
        finally:
            self._reset_engine()
            if com_ok:
                try:
                    import pythoncom

                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def close(self, wait: bool = True, timeout: float = 3.0) -> None:
        with self._lock:
            if self._closed:
                thread = self._thread
            else:
                self._closed = True
                self._q.put(None)
                thread = self._thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
