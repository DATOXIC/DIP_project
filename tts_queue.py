"""Bộ đọc âm thanh tuần tự, không chặn luồng chính."""

import queue
import io
import math
import os
import threading
import time
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
        self._voice = None
        self._stats = {"enqueued": 0, "played": 0, "dropped": 0, "errors": 0}
        self._debug_started = time.perf_counter()

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
                self._stats["dropped"] += 1
                self._debug("bỏ từ cũ do hàng đầy", f"con_lai={self._q.qsize()}")
            self._q.put(item)
            self._stats["enqueued"] += 1
            self._ensure_thread_locked()
            self._debug("xếp hàng", f"dài={self._q.qsize()}")
        return True

    def _ensure_thread_locked(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _speak_one(self, text: str, voice_id: Optional[str]) -> None:
        try:
            import win32com.client

            if self._voice is None:
                self._voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._voice.Rate = int(math.log(max(1, self.rate) / 200.0, 1.1))
            if "DIP_TTS_VOLUME" in os.environ:
                self._voice.Volume = max(0, min(100, int(os.environ["DIP_TTS_VOLUME"])))
            if voice_id:
                for token in self._voice.GetVoices():
                    if str(getattr(token, "Id", "")) == voice_id:
                        self._voice.Voice = token
                        break
            self._voice.Speak(text, 0)
        except Exception:
            self._voice = None
            raise

    def _reset_engine(self) -> None:
        self._voice = None

    @staticmethod
    def _play_wav(path: str) -> None:
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _debug(self, event: str, detail: str = "") -> None:
        if os.environ.get("DIP_TTS_DEBUG") == "1":
            elapsed = time.perf_counter() - self._debug_started
            suffix = f" {detail}" if detail else ""
            print(f"[TTS-dbg] t={elapsed:.3f} {event}{suffix}")

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
                started = time.perf_counter()
                self._debug("bắt đầu phát", f"loại={kind} giá trị={value}")
                try:
                    if kind == "wav":
                        self._play_wav(value)
                    elif self._speak_fn is not None:
                        self._speak_fn(value)
                    else:
                        self._speak_one(value, voice_id)
                    with self._lock:
                        self._stats["played"] += 1
                    self._debug("phát xong", f"thời_lượng={time.perf_counter() - started:.3f}")
                except Exception as err:
                    with self._lock:
                        self._stats["errors"] += 1
                    self._debug("lỗi", str(err))
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
