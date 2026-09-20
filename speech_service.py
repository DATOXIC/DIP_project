"""
DỊCH VỤ GIỌNG NÓI THEO NGÔN NGỮ (SPEECH SERVICE)
================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

speak_word(key, lang, text) chọn cách phát âm cho từ vừa nhận được:
  1. Có sẵn file  audio_cache/<lang>/<KEY>.wav  -> phát bất đồng bộ bằng winsound (không chặn vòng lặp).
  2. lang == "en" -> đọc bằng bộ TTS tiếng Anh hiện có (predict_sentence_webcam.speak_async, đăng ký qua
                     set_en_speaker; mặc định dùng worker nội bộ của module này).
  3. lang == "vi" -> tìm giọng SAPI tiếng Việt (culture vi-VN / tên chứa "Vietnam" hoặc "An") bằng pyttsx3
                     trong luồng nền. Không có giọng -> KHÔNG phát âm, trả về "no_vi_voice".

Module KHÔNG tạo file âm thanh, KHÔNG gọi mạng, KHÔNG làm chậm vòng lặp khung hình (mọi việc nặng nằm ở luồng nền).
Mọi lỗi được bắt lại và chỉ cảnh báo MỘT lần bằng tiếng Việt.
"""

import os
import re
import threading
from typing import Callable, Optional

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_CACHE_DIR = os.path.join(_BASE_DIR, "audio_cache")

_muted = False
_en_speaker: Optional[Callable[[str], bool]] = None
_vi_voice_id: Optional[str] = None
_vi_checked = False
_discovery_started = False
_vi_wav_any: Optional[bool] = None
_state_lock = threading.Lock()
_warned = set()
_worker_lock = threading.Lock()


def _warn_once(tag: str, msg: str) -> None:
    if tag not in _warned:
        _warned.add(tag)
        try:
            print(f"[Speech] {msg}")
        except Exception:  # console không hỗ trợ Unicode (vd. cp1252) -> in bản không dấu, không được sập
            print("[Speech] " + msg.encode("ascii", "replace").decode("ascii"))


# ---- điều khiển loa --------------------------------------------------------

def set_muted(value: bool) -> None:
    global _muted
    _muted = bool(value)


def is_muted() -> bool:
    return _muted


def toggle_muted() -> bool:
    """Đảo trạng thái tắt loa, trả về trạng thái mới (True = đang tắt)."""
    global _muted
    _muted = not _muted
    return _muted


def set_en_speaker(fn: Optional[Callable[[str], bool]]) -> None:
    """Đăng ký hàm đọc tiếng Anh (thường là speak_async của predict_sentence_webcam)."""
    global _en_speaker
    _en_speaker = fn


# ---- tìm giọng tiếng Việt ---------------------------------------------------

def _is_vi_voice(voice) -> bool:
    name = str(getattr(voice, "name", "") or "")
    vid = str(getattr(voice, "id", "") or "")
    langs = getattr(voice, "languages", None) or []
    for lg in langs:
        if isinstance(lg, bytes):
            lg = lg.decode("utf-8", "ignore")
        if str(lg).lower().replace("_", "-").startswith("vi"):
            return True
    if "vietnam" in name.lower() or "vietnam" in vid.lower():
        return True
    return re.search(r"\bAn\b", name) is not None or "VI-VN" in vid.upper()


def _discover_vi_voice() -> None:
    """Chạy trong luồng nền: liệt kê giọng SAPI và ghi nhớ giọng Việt (nếu có)."""
    global _vi_voice_id, _vi_checked
    engine = None
    com_ok = False
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_ok = True
        except Exception:
            pass
        import pyttsx3
        engine = pyttsx3.init()
        for v in engine.getProperty("voices"):
            if _is_vi_voice(v):
                _vi_voice_id = v.id
                break
    except Exception as err:
        _warn_once("discover", f"Không liệt kê được giọng nói hệ thống ({err}).")
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
            try:
                import pyttsx3
                if hasattr(pyttsx3, "_activeEngines"):
                    pyttsx3._activeEngines.clear()
            except Exception:
                pass
        if com_ok:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
        _vi_checked = True


def init_speech() -> None:
    """Khởi động dò giọng Việt ở luồng nền (không chặn). Gọi nhiều lần vẫn chỉ chạy một lần."""
    global _discovery_started
    with _state_lock:
        if _discovery_started or _vi_checked:
            return
        _discovery_started = True
    threading.Thread(target=_discover_vi_voice, daemon=True).start()


def _has_wav(key: str, lang: str) -> Optional[str]:
    path = os.path.join(AUDIO_CACHE_DIR, lang, f"{key}.wav")
    return path if os.path.isfile(path) else None


def get_speech_status(lang: str) -> str:
    """'muted' | 'no_vi_voice' | 'on' — dùng cho nhãn loa trên HUD."""
    if _muted:
        return "muted"
    if lang == "vi" and _vi_checked and _vi_voice_id is None:
        global _vi_wav_any
        if _vi_wav_any is None:  # chỉ quét thư mục một lần (hàm này được gọi mỗi khung hình)
            wav_dir = os.path.join(AUDIO_CACHE_DIR, "vi")
            _vi_wav_any = os.path.isdir(wav_dir) and any(f.lower().endswith(".wav") for f in os.listdir(wav_dir))
        if not _vi_wav_any:
            return "no_vi_voice"
    return "on"


# ---- phát âm ---------------------------------------------------------------

def _voice_worker(text: str, voice_id: Optional[str]) -> None:
    engine = None
    com_ok = False
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_ok = True
        except Exception:
            pass
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.say(text)
        engine.runAndWait()
    except Exception as err:
        _warn_once("speak", f"Lỗi phát giọng nói ({err}).")
    finally:
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
        if com_ok:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
        _worker_lock.release()


def _speak_thread(text: str, voice_id: Optional[str]) -> bool:
    """Nói bằng worker nội bộ; nếu đang bận thì bỏ qua (không xếp hàng, không chặn)."""
    if not _worker_lock.acquire(blocking=False):
        return False
    try:
        threading.Thread(target=_voice_worker, args=(text, voice_id), daemon=True).start()
        return True
    except Exception:
        _worker_lock.release()
        return False


def speak_word(key: str, lang: str, text: str) -> str:
    """
    Phát âm từ vừa nhận. Trả về trạng thái:
      'muted' | 'wav' | 'tts' | 'busy' | 'no_vi_voice' | 'empty' | 'error'
    Không bao giờ ném ngoại lệ.
    """
    try:
        if _muted:
            return "muted"
        if not text or not str(text).strip():
            return "empty"

        wav = _has_wav(key, lang)
        if wav is not None:
            try:
                import winsound
                winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return "wav"
            except Exception as err:
                _warn_once("wav", f"Không phát được file âm thanh '{wav}' ({err}); chuyển sang đọc bằng TTS.")

        if lang == "vi":
            if not _vi_checked:
                init_speech()  # chưa dò xong: bắt đầu dò ở nền, lần nói này bỏ qua (không chờ)
                return "busy"
            if _vi_voice_id is None:
                _warn_once("no_vi", "Máy chưa có giọng đọc tiếng Việt (SAPI vi-VN) và không có file audio_cache/vi/*.wav "
                                    "-> chỉ hiển thị chữ, không phát âm.")
                return "no_vi_voice"
            return "tts" if _speak_thread(str(text), _vi_voice_id) else "busy"

        if _en_speaker is not None:
            return "tts" if _en_speaker(str(text)) else "busy"
        return "tts" if _speak_thread(str(text), None) else "busy"
    except Exception as err:
        _warn_once("speak_word", f"Lỗi giọng nói ({err}); tiếp tục chạy không có tiếng.")
        return "error"
