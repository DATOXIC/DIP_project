"""
GIAO DIỆN DEMO THỜI GIAN THỰC (DEMO HUD) — SONG NGỮ ANH / VIỆT
================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

Module này TÁCH BIỆT khỏi vòng lặp nhận diện: chỉ nhận một `HudState` (dữ liệu thuần) và vẽ lên khung hình,
nên có thể gọi offscreen với khung giả để kiểm thử.

Thành phần chính:
  - UI_STRINGS / tr()        : bảng chuỗi giao diện song ngữ.
  - strip_diacritics()       : bỏ dấu tiếng Việt (dùng cho fallback khi không có font Unicode).
  - TextRenderer             : vẽ chữ Unicode bằng Pillow theo hàng đợi. Mỗi chuỗi được render thành MẶT NẠ xám
                               (cache theo (chuỗi, cỡ, đậm)) rồi trộn trực tiếp vào ROI của khung hình
                               => KHÔNG phải chuyển đổi BGR<->PIL cho cả khung hình, chi phí rất thấp.
  - DemoHUD.render()         : chip trạng thái, vòng quanh bàn tay, toast từ nhận được, thanh phụ đề song ngữ,
                               FPS + trạng thái loa, overlay trợ giúp.
  - draw_soft_trail()        : biến thể vệt chuyển động dịu hơn cho giao diện demo (không đổi hàm gốc).
"""

import math
import os
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # Pillow không có -> luôn dùng fallback cv2.putText
    _PIL_OK = False


# =============================================================================
# 1. BẢNG CHUỖI GIAO DIỆN SONG NGỮ
# =============================================================================

UI_STRINGS: Dict[str, Dict[str, str]] = {
    "vi": {
        "chip_no_hand": "Đưa tay vào khung",
        "chip_hold": "Giữ yên tay…",
        "chip_ready": "Sẵn sàng — hãy ký",
        "chip_chain_ready": "Ký tiếp được rồi",
        "chip_recording": "Đang ghi…",
        "chip_recognising": "Đang nhận dạng…",
        "match": "Độ khớp {q}%",
        "uncertain": "Ý bạn là: {a} hay {b}?",
        "rejected": "Chưa nhận ra cử chỉ",
        "suppressed": "Bỏ qua (trùng trong cùng lần ký)",
        "deleted": "Đã xóa: {word}",
        "empty": "Câu đang trống",
        "committed": "Đã chốt câu",
        "cleared": "Đã xóa câu",
        "placeholder": "Hãy ký từ đầu tiên…",
        "speech_on": "Loa bật",
        "speech_muted": "Loa tắt",
        "speech_no_vi": "Chưa có giọng Việt",
        "hint_help": "[H] Trợ giúp",
        "vocab_empty": "Chưa có từ điển! Hãy chạy collect_dynamic_templates.py",
        "help_title": "Hướng dẫn sử dụng",
        "help_step1": "Giữ yên tay cho đến khi vòng tròn đầy, ký cử chỉ, rồi dừng nhẹ để chốt từ.",
        "help_step2": "Vuốt tay nhanh sang trái để xóa từ cuối cùng.",
        "help_step3": "Hạ tay khỏi khung hình hơn 1,8 giây để chốt câu.",
        "key_h": "Trợ giúp",
        "key_j": "Ký nhanh (nối từ)",
        "key_k": "Cắt dẫn đầu thích nghi (lead-trim)",
        "key_l": "Đổi ngôn ngữ",
        "key_t": "Bật/tắt loa",
        "key_u": "Giao diện gỡ lỗi",
        "key_c": "Xóa câu",
        "key_f": "Lật camera",
        "key_p": "Cài đặt camera",
        "key_q": "Thoát",
        "help_close": "Nhấn H để đóng",
        "help_diag": "Camera {cam}  ·  Mốc khuôn mặt: {face}",
        "face_ok": "đã thấy",
        "face_searching": "đang tìm",
        "face_off": "tắt",
        "cam_fps": "Camera {n} khung/s",
        "cam_warn_low_light": "Thiếu sáng — hãy bật thêm đèn",
        "cam_warn_slow_camera": "Camera đang chậm — thêm ánh sáng, tránh ngược sáng",
        "cam_warn_too_bright": "Ánh sáng quá chói — giảm bớt hoặc xoay đèn",
    },
    "en": {
        "chip_no_hand": "Show your hand",
        "chip_hold": "Hold still…",
        "chip_ready": "Ready — sign now",
        "chip_chain_ready": "Ready — keep signing",
        "chip_recording": "Recording…",
        "chip_recognising": "Recognising…",
        "match": "Match {q}%",
        "uncertain": "Did you mean: {a} or {b}?",
        "rejected": "Gesture not recognised",
        "suppressed": "Skipped (duplicate in same sign)",
        "deleted": "Removed: {word}",
        "empty": "Sentence is empty",
        "committed": "Sentence committed",
        "cleared": "Sentence cleared",
        "placeholder": "Sign your first word…",
        "speech_on": "Speaker on",
        "speech_muted": "Speaker off",
        "speech_no_vi": "No Vietnamese voice",
        "hint_help": "[H] Help",
        "vocab_empty": "No vocabulary! Run collect_dynamic_templates.py",
        "help_title": "How to use",
        "help_step1": "Hold your hand still until the ring fills, sign, then pause briefly to confirm the word.",
        "help_step2": "Swipe your hand quickly to the left to delete the last word.",
        "help_step3": "Lower your hand out of view for over 1.8 s to finish the sentence.",
        "key_h": "Help",
        "key_j": "Fast chain on/off",
        "key_k": "Lead trim on/off",
        "key_l": "Switch language",
        "key_t": "Speaker on/off",
        "key_u": "Debug view",
        "key_c": "Clear sentence",
        "key_f": "Flip camera",
        "key_p": "Camera settings",
        "key_q": "Quit",
        "help_close": "Press H to close",
        "help_diag": "Camera {cam}  ·  Face anchor: {face}",
        "face_ok": "found",
        "face_searching": "searching",
        "face_off": "off",
        "cam_fps": "Camera {n} fps",
        "cam_warn_low_light": "Low light — turn on more lights",
        "cam_warn_slow_camera": "Camera is slow — add light, avoid backlight",
        "cam_warn_too_bright": "Light too harsh — dim or turn the lamp",
    },
}


def tr(lang: str, key: str, **fmt) -> str:
    """Tra chuỗi giao diện theo ngôn ngữ; thiếu khóa -> rơi về 'en' -> rơi về chính tên khóa."""
    text = UI_STRINGS.get(lang, {}).get(key)
    if text is None:
        text = UI_STRINGS["en"].get(key)
    if text is None:
        text = key
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt (xử lý riêng đ/Đ) và các ký tự ngoài ASCII để cv2.putText vẽ được."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    text = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    for src, dst in (("…", "..."), ("—", "-"), ("–", "-"), ("·", "-"), ("→", "->"), ("“", '"'), ("”", '"')):
        text = text.replace(src, dst)
    return text.encode("ascii", "ignore").decode("ascii")


# =============================================================================
# 2. BẢNG MÀU (BGR) & TIỆN ÍCH TRỘN ẢNH
# =============================================================================

WHITE = (255, 255, 255)
TEXT_MAIN = (245, 245, 245)
TEXT_OLD = (200, 200, 200)
TEXT_DIM = (170, 170, 170)
TEXT_MUTED = (140, 140, 140)
PANEL_BG = (22, 22, 22)
GREEN = (110, 214, 84)        # xanh lá = OK
AMBER = (0, 190, 255)         # hổ phách = đang xử lý / nghi vấn
GRAY = (150, 150, 150)        # xám = từ chối / trung tính
RED_SOFT = (120, 120, 240)    # đỏ nhạt = FPS rất thấp
RED_DEEP = (40, 40, 175)      # nền cảnh báo thiếu từ điển


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


_CAM_WARN_KEYS = ("low_light", "slow_camera", "too_bright")   # khóa cảnh báo camera (camera_health.py)

_BLEND_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


def blend_mask(frame: np.ndarray, x: int, y: int, mask: np.ndarray,
               color: Tuple[int, int, int], alpha: float = 1.0, cache_key=None) -> None:
    """
    Trộn màu `color` vào `frame` tại (x, y) theo mặt nạ xám `mask` (0..255) nhân `alpha`. Tự cắt theo biên.
    Nếu có `cache_key` (định danh ổn định của mặt nạ) và mặt nạ nằm trọn trong khung: cache sẵn cặp
    (255-a, màu*a) để mỗi khung hình chỉ tốn 2 phép toán uint8 SIMD của OpenCV.
    """
    if alpha <= 0.0:
        return
    alpha = _clamp(alpha, 0.0, 1.0)
    fh, fw = frame.shape[:2]
    mh, mw = mask.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + mw), min(fh, y + mh)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    inside = (x0 == x and y0 == y and x1 == x + mw and y1 == y + mh)
    entry = None
    ck = None
    if cache_key is not None and inside:
        ck = (cache_key, color, int(round(alpha * 64)))
        entry = _BLEND_CACHE.get(ck)
        if entry is not None and entry[0].shape[:2] != roi.shape[:2]:
            entry = None
    if entry is None:
        m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        a1 = m if alpha >= 0.999 else cv2.convertScaleAbs(m, alpha=alpha)
        a3 = cv2.cvtColor(a1, cv2.COLOR_GRAY2BGR)
        inv = cv2.bitwise_not(a3)
        add = cv2.multiply(a3, (float(color[0]), float(color[1]), float(color[2]), 0.0), scale=1.0 / 255.0)
        entry = (inv, add)
        if ck is not None:
            if len(_BLEND_CACHE) > 160:
                _BLEND_CACHE.clear()
            _BLEND_CACHE[ck] = entry
    # kết quả = roi*(255-a)/255 + màu*a/255
    roi[:] = cv2.add(cv2.multiply(roi, entry[0], scale=1.0 / 255.0), entry[1])


_SHAPE_CACHE: Dict[Tuple, np.ndarray] = {}


def _rrect_mask(w: int, h: int, r: int) -> np.ndarray:
    """Mặt nạ hình chữ nhật bo góc, khử răng cưa bằng siêu lấy mẫu 4x (có cache)."""
    key = ("rr", w, h, r)
    m = _SHAPE_CACHE.get(key)
    if m is not None:
        return m
    if len(_SHAPE_CACHE) > 300:
        _SHAPE_CACHE.clear()
    k = 4
    r = max(0, min(r, h // 2, w // 2))
    big = np.zeros((h * k, w * k), dtype=np.uint8)
    rk = r * k
    if rk > 0:
        cv2.rectangle(big, (rk, 0), (w * k - 1 - rk, h * k - 1), 255, -1)
        cv2.rectangle(big, (0, rk), (w * k - 1, h * k - 1 - rk), 255, -1)
        for cx, cy in ((rk, rk), (w * k - 1 - rk, rk), (rk, h * k - 1 - rk), (w * k - 1 - rk, h * k - 1 - rk)):
            cv2.circle(big, (cx, cy), rk, 255, -1)
    else:
        big[:] = 255
    m = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    _SHAPE_CACHE[key] = m
    return m


def _ring_mask(w: int, h: int, r: int, b: int) -> np.ndarray:
    """Mặt nạ viền dày b của hình chữ nhật bo góc (có cache)."""
    key = ("rg", w, h, r, b)
    m = _SHAPE_CACHE.get(key)
    if m is not None:
        return m
    outer = _rrect_mask(w, h, r)
    ring = outer.copy()
    if w - 2 * b > 0 and h - 2 * b > 0:
        inner = _rrect_mask(w - 2 * b, h - 2 * b, max(0, r - b))
        ring[b:h - b, b:w - b] = cv2.subtract(outer[b:h - b, b:w - b], inner)
    _SHAPE_CACHE[key] = ring
    return ring


def draw_panel(frame: np.ndarray, x: int, y: int, w: int, h: int, r: int,
               color: Tuple[int, int, int] = PANEL_BG, alpha: float = 0.62,
               border_color: Optional[Tuple[int, int, int]] = None, border_w: int = 0,
               border_alpha: float = 1.0) -> None:
    """Vẽ bảng nền bo góc bán trong suốt, tuỳ chọn có viền."""
    if w <= 1 or h <= 1:
        return
    fh, fw = frame.shape[:2]
    r = max(0, min(r, h // 2, w // 2))
    if (w * h > 50000 and r >= 2 and color[0] == color[1] == color[2]
            and x >= 0 and y >= 0 and x + w <= fw and y + h <= fh and alpha > 0.0):
        # Đường nhanh cho bảng lớn màu xám (vd. overlay trợ giúp): phần lõi chữ nhật chỉ cần 1 phép
        # roi*(1-a)+màu*a (convertScaleAbs, SIMD); chỉ 4 góc bo mới cần trộn theo mặt nạ.
        a = _clamp(alpha, 0.0, 1.0)
        beta = float(color[0]) * a
        for rx, ry, rw, rh in ((x + r, y, w - 2 * r, h), (x, y + r, r, h - 2 * r), (x + w - r, y + r, r, h - 2 * r)):
            if rw > 0 and rh > 0:
                sub = frame[ry:ry + rh, rx:rx + rw]
                cv2.convertScaleAbs(sub, dst=sub, alpha=1.0 - a, beta=beta)
        full = _rrect_mask(w, h, r)
        for ci, (cx0, cy0) in enumerate(((0, 0), (w - r, 0), (0, h - r), (w - r, h - r))):
            blend_mask(frame, x + cx0, y + cy0, full[cy0:cy0 + r, cx0:cx0 + r], color, alpha,
                       cache_key=("rrc", w, h, r, ci))
    else:
        blend_mask(frame, x, y, _rrect_mask(w, h, r), color, alpha, cache_key=("rr", w, h, r))
    if border_color is not None and border_w > 0:
        blend_mask(frame, x, y, _ring_mask(w, h, r, border_w), border_color, border_alpha,
                   cache_key=("rg", w, h, r, border_w))


# =============================================================================
# 3. BỘ VẼ CHỮ UNICODE (TextRenderer)
# =============================================================================

_FONT_DIR = r"C:\Windows\Fonts"
DEFAULT_FONT_CANDIDATES: List[Tuple[str, str]] = [
    (os.path.join(_FONT_DIR, "segoeui.ttf"), os.path.join(_FONT_DIR, "segoeuib.ttf")),
    (os.path.join(_FONT_DIR, "arial.ttf"), os.path.join(_FONT_DIR, "arialbd.ttf")),
    (os.path.join(_FONT_DIR, "calibri.ttf"), os.path.join(_FONT_DIR, "calibrib.ttf")),
]


class TextRenderer:
    """
    Vẽ chữ Unicode qua hàng đợi: queue() ghi lệnh + trả về kích thước, flush(frame) trộn tất cả vào khung hình.
    Không tìm được font nào -> tự động dùng cv2.putText với chuỗi đã bỏ dấu (không bao giờ ném lỗi).
    """

    _counter = 0

    def __init__(self, font_candidates: Optional[Sequence[Tuple[str, str]]] = None):
        TextRenderer._counter += 1
        self._uid = TextRenderer._counter   # phân biệt cache trộn giữa các bộ vẽ (font thật / fallback)
        self._candidates = list(DEFAULT_FONT_CANDIDATES if font_candidates is None else font_candidates)
        self._fonts: Dict[Tuple[str, int], object] = {}
        self._masks: Dict[Tuple[str, int, bool], np.ndarray] = {}
        self._queue: List[Tuple[np.ndarray, int, int, Tuple[int, int, int], float, Tuple]] = []
        self._regular: Optional[str] = None
        self._bold: Optional[str] = None
        self.using_fallback = True
        self._resolve_fonts()

    def _resolve_fonts(self) -> None:
        if not _PIL_OK:
            return
        for reg, bold in self._candidates:
            try:
                ImageFont.truetype(reg, 16)
            except Exception:
                continue
            self._regular = reg
            try:
                ImageFont.truetype(bold, 16)
                self._bold = bold
            except Exception:
                self._bold = reg
            self.using_fallback = False
            return

    def _font(self, size: int, bold: bool):
        path = (self._bold if bold else self._regular) or self._regular
        key = (path, size)
        f = self._fonts.get(key)
        if f is None:
            f = ImageFont.truetype(path, size)
            self._fonts[key] = f
        return f

    def _mask(self, text: str, size: int, bold: bool) -> np.ndarray:
        key = (text, size, bold)
        m = self._masks.get(key)
        if m is not None:
            return m
        if len(self._masks) > 700:
            self._masks.clear()
        if not self.using_fallback:
            try:
                font = self._font(size, bold)
                asc, desc = font.getmetrics()
                w = int(math.ceil(font.getlength(text))) + 3
                img = Image.new("L", (max(1, w), asc + desc + 2), 0)
                ImageDraw.Draw(img).text((1, 0), text, font=font, fill=255)
                m = np.array(img, dtype=np.uint8)
            except Exception:
                m = None
        if m is None:
            m = self._fallback_mask(text, size, bold)
        self._masks[key] = m
        return m

    @staticmethod
    def _fallback_mask(text: str, size: int, bold: bool) -> np.ndarray:
        plain = strip_diacritics(text)
        scale = size / 30.0
        thick = 2 if bold else 1
        (tw, th), base = cv2.getTextSize(plain, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        h = int(size * 1.3) + 2
        m = np.zeros((h, tw + 4), dtype=np.uint8)
        cv2.putText(m, plain, (1, int(size * 1.0)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
        return m

    def text_size(self, text: str, size: float, bold: bool = False) -> Tuple[int, int]:
        m = self._mask(text, max(6, int(round(size))), bold)
        return m.shape[1], m.shape[0]

    def fit_size(self, text: str, size: float, max_w: float, bold: bool = False, min_size: float = 8) -> int:
        """Giảm dần cỡ chữ cho đến khi chuỗi vừa độ rộng max_w."""
        sz = max(6, int(round(size)))
        while sz > min_size and self.text_size(text, sz, bold)[0] > max_w:
            sz -= 1
        return sz

    def queue(self, text: str, x: int, y: int, size: float, color: Tuple[int, int, int],
              bold: bool = False, anchor: str = "l", alpha: float = 1.0) -> Tuple[int, int]:
        """Đưa lệnh vẽ vào hàng đợi. (x, y) = mép TRÊN của dòng; anchor l/c/r theo trục x. Trả về (rộng, cao)."""
        if not text:
            return 0, 0
        m = self._mask(text, max(6, int(round(size))), bold)
        w = m.shape[1]
        if anchor == "c":
            x -= w // 2
        elif anchor == "r":
            x -= w
        self._queue.append((m, int(x), int(y), color, alpha, ("tx", self._uid, text, max(6, int(round(size))), bold)))
        return w, m.shape[0]

    def flush(self, frame: np.ndarray) -> None:
        for m, x, y, color, alpha, ck in self._queue:
            blend_mask(frame, x, y, m, color, alpha, cache_key=ck)
        self._queue.clear()


# =============================================================================
# 4. DỮ LIỆU TRẠNG THÁI HUD
# =============================================================================

@dataclass
class HudToast:
    """Sự kiện phản hồi có cấu trúc. kind: accepted | uncertain | rejected | pending | deleted | empty | committed | cleared."""
    kind: str
    keys: List[str]
    quality: Optional[float]
    t_start: float
    t_expire: float


@dataclass
class HudState:
    lang: str = "en"
    seg_state: str = "IDLE"                      # IDLE | READY | RECORDING | PENDING_COMMIT
    hand_found: bool = False
    hand_center: Optional[Tuple[float, float]] = None
    palm_len: float = 80.0
    v_norm: float = 0.0
    settle_progress: Optional[float] = None      # 0..1 hoặc None (không dùng cung "giữ yên")
    record_progress: float = 0.0                 # 0..1
    fps: float = 0.0
    sentence_keys: List[str] = field(default_factory=list)
    sentence_finalized: bool = False
    toast: Optional[HudToast] = None
    vocabulary_empty: bool = False
    face_state: str = "off"                      # ok | searching | off
    speech_status: str = "on"                    # on | muted | no_vi_voice
    show_help: bool = False
    camera_index: int = 0
    now: Optional[float] = None                  # None -> dùng time.time()
    # --- sức khỏe camera (camera_health.py); mặc định = "không có dữ liệu" ---
    unique_fps: float = -1.0                     # khung MỚI THẬT/giây; < 0 = không đo -> hiển thị FPS vòng lặp như cũ
    loop_fps: float = 0.0                        # số lần đọc/giây (chỉ để chẩn đoán)
    camera_warning: Optional[str] = None         # None | "low_light" | "slow_camera" | "too_bright"
    camera_warning_level: str = "warn"           # "warn" | "bad"
    dup_ratio: float = 0.0
    near_dup_ratio: float = 0.0
    luma_mean: float = 0.0
    # --- chế độ "ký nhanh" (SPEC 08): True khi cửa sổ nối đang mở (vừa chốt xong một từ, có thể ký tiếp ngay).
    #     Mặc định False => giao diện y hệt cũ. ---
    chain_ready: bool = False


def _seg_name(seg_state) -> str:
    return str(getattr(seg_state, "value", seg_state))


# =============================================================================
# 5. LỚP DemoHUD
# =============================================================================

class DemoHUD:
    """Giao diện demo tối giản. Mọi kích thước nhân với s = chiều cao khung / 480."""

    FADE_IN_SEC = 0.15
    FADE_OUT_SEC = 0.40

    def __init__(self, phrase_lookup: Optional[Callable[[str, str], str]] = None,
                 text_renderer: Optional[TextRenderer] = None):
        # phrase_lookup(key, lang) -> cụm từ ("" nếu không có). Mặc định: dùng chính khóa.
        self._lookup = phrase_lookup
        self.text = text_renderer if text_renderer is not None else TextRenderer()

    # ---- tiện ích ---------------------------------------------------------
    def phrase(self, key: str, lang: str) -> str:
        if self._lookup is not None:
            for lg in (lang, "en"):
                try:
                    p = self._lookup(key, lg)
                except Exception:
                    p = ""
                if p:
                    return p
        return str(key)

    @staticmethod
    def _mode(st: HudState) -> str:
        if not st.hand_found:
            return "no_hand"
        name = _seg_name(st.seg_state)
        if st.chain_ready and name in ("IDLE", "READY"):
            return "ready"   # cửa sổ nối đang mở: vòng giữ trạng thái sẵn sàng (đầy, xanh), không chạy lại cung chờ
        if name == "RECORDING":
            return "recording"
        if name == "PENDING_COMMIT":
            return "pending"
        if name == "READY":
            return "ready"
        return "hold"

    # ---- điểm vào chính ---------------------------------------------------
    def render(self, frame: np.ndarray, state: HudState) -> np.ndarray:
        h, w = frame.shape[:2]
        s = h / 480.0
        now = state.now if state.now is not None else time.time()
        lang = state.lang if state.lang in UI_STRINGS else "en"
        mode = self._mode(state)

        self._draw_hand_ring(frame, state, s, now, mode)
        self._draw_status_chip(frame, state, s, lang, mode)
        self._draw_top_right(frame, state, s, lang, w)
        top_shift = 0
        if not state.show_help:  # overlay trợ giúp che phủ: không vẽ toast / phụ đề bên dưới cho đỡ rối
            if state.camera_warning in _CAM_WARN_KEYS:
                top_shift = self._draw_camera_warning(frame, state, s, lang, w)
            if state.vocabulary_empty:
                self._draw_vocab_error(frame, s, lang, w, top_shift)
            elif state.toast is not None:
                self._draw_toast(frame, state.toast, s, lang, now, w, top_shift)
            self._draw_subtitle_bar(frame, state, s, lang, w, h)
        self.text.flush(frame)

        if state.show_help:
            self._draw_help(frame, state, s, lang, w, h)
            self.text.flush(frame)
        return frame

    # ---- vòng quanh bàn tay ----------------------------------------------
    def _draw_hand_ring(self, frame, st: HudState, s: float, now: float, mode: str) -> None:
        if mode == "no_hand" or st.hand_center is None:
            return
        fh, fw = frame.shape[:2]
        cx, cy = int(round(st.hand_center[0])), int(round(st.hand_center[1]))
        r = int(_clamp(0.9 * st.palm_len, 30 * s, 140 * s))
        pad = r + int(16 * s)
        x0, y0 = max(0, cx - pad), max(0, cy - pad)
        x1, y1 = min(fw, cx + pad), min(fh, cy + pad)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        layer = roi.copy()
        c = (cx - x0, cy - y0)
        thin = max(1, int(round(2 * s)))
        AA = cv2.LINE_AA
        if mode == "hold":
            cv2.circle(layer, c, r, (215, 215, 215), thin, AA)
            p = st.settle_progress
            if p is not None and p > 0.0:
                cv2.ellipse(layer, c, (r, r), 0, -90, -90 + 360.0 * _clamp(p, 0.0, 1.0),
                            AMBER, max(2, int(round(4 * s))), AA)
        elif mode == "ready":
            cv2.circle(layer, c, r, GREEN, max(2, int(round(4 * s))), AA)
        elif mode == "recording":
            breath = 0.5 + 0.5 * math.sin(2.0 * math.pi * 1.1 * now)
            th = max(2, int(round((3.0 + 2.5 * breath) * s)))
            cv2.circle(layer, c, r, GREEN, th, AA)
            prog = _clamp(st.record_progress, 0.0, 1.0)
            if prog > 0.0:
                cv2.ellipse(layer, c, (r + int(9 * s), r + int(9 * s)), 0, -90, -90 + 360.0 * prog,
                            (190, 245, 170), max(2, int(round(3 * s))), AA)
        else:  # pending
            cv2.circle(layer, c, r, AMBER, max(2, int(round(4 * s))), AA)
        roi[:] = cv2.addWeighted(layer, 0.88, roi, 0.12, 0)

    # ---- chip trạng thái --------------------------------------------------
    def _draw_status_chip(self, frame, st: HudState, s: float, lang: str, mode: str) -> None:
        key = {"no_hand": "chip_no_hand", "hold": "chip_hold", "ready": "chip_ready",
               "recording": "chip_recording", "pending": "chip_recognising"}[mode]
        if mode == "ready" and st.chain_ready:
            key = "chip_chain_ready"
        dot_color = {"no_hand": GRAY, "hold": AMBER, "ready": GREEN, "recording": GREEN, "pending": AMBER}[mode]
        text = tr(lang, key)
        fs = 16 * s
        tw, th = self.text.text_size(text, fs)
        m = int(round(16 * s))
        ch = int(round(34 * s))
        pad = int(round(14 * s))
        dot_r = max(3, int(round(5 * s)))
        gap = int(round(9 * s))
        cw = pad + dot_r * 2 + gap + tw + pad
        draw_panel(frame, m, m, cw, ch, ch // 2, PANEL_BG, 0.70)
        cy = m + ch // 2
        cv2.circle(frame, (m + pad + dot_r, cy), dot_r, dot_color, -1, cv2.LINE_AA)
        self.text.queue(text, m + pad + dot_r * 2 + gap, cy - th // 2, fs, TEXT_MAIN)

    # ---- góc trên-phải: loa + FPS ----------------------------------------
    def _draw_top_right(self, frame, st: HudState, s: float, lang: str, w: int) -> None:
        sp = {"on": ("speech_on", GREEN), "muted": ("speech_muted", GRAY),
              "no_vi_voice": ("speech_no_vi", AMBER)}.get(st.speech_status, ("speech_on", GREEN))
        sp_text = tr(lang, sp[0])
        if st.unique_fps >= 0.0:
            # Khung MỚI THẬT/giây của camera (không phải tốc độ vòng lặp — vòng lặp vẫn quay khi camera trả khung cũ)
            fps_text = tr(lang, "cam_fps", n=f"{st.unique_fps:.0f}")
            fps_col = GREEN if st.unique_fps >= 20.0 else (AMBER if st.unique_fps >= 12.0 else RED_SOFT)
        else:  # không có dữ liệu đo camera: giữ chỉ số FPS vòng lặp như cũ
            fps_text = f"FPS {st.fps:.0f}"
            fps_col = RED_SOFT if st.fps < 20.0 else (AMBER if st.fps < 24.0 else TEXT_DIM)
        fs = 14 * s
        sw, sh = self.text.text_size(sp_text, fs)
        fw_, fh_ = self.text.text_size(fps_text, fs)
        m = int(round(16 * s))
        ch = int(round(34 * s))
        pad = int(round(13 * s))
        dot_r = max(2, int(round(4 * s)))
        gap = int(round(7 * s))
        sep = int(round(14 * s))
        cw = pad + dot_r * 2 + gap + sw + sep + fw_ + pad
        x0 = w - m - cw
        draw_panel(frame, x0, m, cw, ch, ch // 2, PANEL_BG, 0.62)
        cy = m + ch // 2
        cv2.circle(frame, (x0 + pad + dot_r, cy), dot_r, sp[1], -1, cv2.LINE_AA)
        x = x0 + pad + dot_r * 2 + gap
        self.text.queue(sp_text, x, cy - sh // 2, fs, TEXT_DIM)
        self.text.queue(fps_text, x + sw + sep, cy - fh_ // 2, fs, fps_col)

    # ---- toast ------------------------------------------------------------
    def _toast_alpha(self, t: HudToast, now: float) -> float:
        if now < t.t_start or now > t.t_expire:
            return 0.0
        a = 1.0
        if now - t.t_start < self.FADE_IN_SEC:
            a = min(a, (now - t.t_start) / self.FADE_IN_SEC)
        if t.t_expire - now < self.FADE_OUT_SEC:
            a = min(a, (t.t_expire - now) / self.FADE_OUT_SEC)
        return _clamp(a, 0.0, 1.0)

    def _draw_toast(self, frame, t: HudToast, s: float, lang: str, now: float, w: int, top_shift: int = 0) -> None:
        fade = self._toast_alpha(t, now)
        if fade <= 0.0:
            return
        cx = w // 2
        top = int(round(60 * s)) + top_shift     # top_shift > 0 khi chip cảnh báo camera đang chiếm hàng trên
        max_w = w - 2 * int(round(24 * s))
        kind = t.kind
        keys = list(t.keys or [])

        if kind == "pending":
            word = self.phrase(keys[0], lang) if keys else ""
            text = f"{word}…"
            fs = self.text.fit_size(text, 19 * s, max_w - 30 * s)
            tw, th = self.text.text_size(text, fs)
            pw, ph = tw + int(round(28 * s)), th + int(round(12 * s))
            draw_panel(frame, cx - pw // 2, top, pw, ph, ph // 2, PANEL_BG, 0.40 * fade)
            self.text.queue(text, cx, top + (ph - th) // 2, fs, TEXT_OLD, anchor="c", alpha=0.85 * fade)
            return

        if kind in ("rejected", "suppressed", "empty", "cleared", "deleted", "committed"):
            if kind == "rejected":
                text, accent = tr(lang, "rejected"), None
            elif kind == "suppressed":
                text, accent = tr(lang, "suppressed"), None
            elif kind == "empty":
                text, accent = tr(lang, "empty"), None
            elif kind == "cleared":
                text, accent = tr(lang, "cleared"), None
            elif kind == "deleted":
                word = self.phrase(keys[0], lang) if keys else ""
                text, accent = tr(lang, "deleted", word=word), AMBER
            else:
                text, accent = tr(lang, "committed"), GREEN
            fs = self.text.fit_size(text, 18 * s, max_w - 40 * s)
            tw, th = self.text.text_size(text, fs)
            pw, ph = tw + int(round(36 * s)), th + int(round(16 * s))
            draw_panel(frame, cx - pw // 2, top, pw, ph, ph // 2, PANEL_BG, 0.55 * fade,
                       border_color=accent, border_w=max(1, int(round(1.5 * s))), border_alpha=0.9 * fade)
            self.text.queue(text, cx, top + (ph - th) // 2, fs, TEXT_MAIN if accent else (205, 205, 205),
                            anchor="c", alpha=fade)
            return

        if kind == "uncertain":
            a = self.phrase(keys[0], lang) if len(keys) > 0 else "?"
            b = self.phrase(keys[1], lang) if len(keys) > 1 else "?"
            main = tr(lang, "uncertain", a=a, b=b)
            fs = self.text.fit_size(main, 28 * s, max_w - 48 * s, bold=True)
            tw, th = self.text.text_size(main, fs, bold=True)
            pw, ph = tw + int(round(48 * s)), th + int(round(24 * s))
            draw_panel(frame, cx - pw // 2, top, pw, ph, int(round(14 * s)), PANEL_BG, 0.68 * fade,
                       border_color=AMBER, border_w=max(1, int(round(2 * s))), border_alpha=0.95 * fade)
            self.text.queue(main, cx, top + (ph - th) // 2, fs, WHITE, bold=True, anchor="c", alpha=fade)
            return

        # accepted
        word = self.phrase(keys[0], lang) if keys else ""
        fs = self.text.fit_size(word, 40 * s, max_w - 56 * s, bold=True)
        tw, th = self.text.text_size(word, fs, bold=True)
        sub = ""
        if t.quality is not None:
            sub = tr(lang, "match", q=int(round(_clamp(float(t.quality), 0.0, 100.0))))
        sub_fs = 15 * s
        sw, sh = self.text.text_size(sub, sub_fs) if sub else (0, 0)
        pad_x, pad_y = int(round(28 * s)), int(round(12 * s))
        gap = int(round(2 * s)) if sub else 0
        pw = max(tw, sw) + 2 * pad_x
        ph = th + gap + sh + 2 * pad_y
        draw_panel(frame, cx - pw // 2, top, pw, ph, int(round(16 * s)), PANEL_BG, 0.68 * fade,
                   border_color=GREEN, border_w=max(1, int(round(2 * s))), border_alpha=0.95 * fade)
        self.text.queue(word, cx, top + pad_y, fs, WHITE, bold=True, anchor="c", alpha=fade)
        if sub:
            self.text.queue(sub, cx, top + pad_y + th + gap, sub_fs, TEXT_DIM, anchor="c", alpha=fade)

    def _draw_vocab_error(self, frame, s: float, lang: str, w: int, top_shift: int = 0) -> None:
        text = tr(lang, "vocab_empty")
        max_w = w - 2 * int(round(16 * s)) - int(round(32 * s))
        fs = self.text.fit_size(text, 16 * s, max_w, bold=True)
        tw, th = self.text.text_size(text, fs, bold=True)
        pw, ph = tw + int(round(32 * s)), th + int(round(18 * s))
        top = int(round(60 * s)) + top_shift
        draw_panel(frame, (w - pw) // 2, top, pw, ph, int(round(12 * s)), RED_DEEP, 0.88)
        self.text.queue(text, w // 2, top + (ph - th) // 2, fs, WHITE, bold=True, anchor="c")

    # ---- chip cảnh báo camera (thiếu sáng / chậm / quá chói) ----------------
    @staticmethod
    def _draw_cam_icon(frame, kind: str, cx: int, cy: int, r: int, color: Tuple[int, int, int], s: float) -> None:
        """Biểu tượng vẽ bằng hình (không emoji), bán kính r quanh (cx, cy)."""
        AA = cv2.LINE_AA
        th = max(1, int(round(1.6 * s)))
        if kind == "too_bright":       # mặt trời: đĩa + 8 tia
            cv2.circle(frame, (cx, cy), max(2, int(r * 0.42)), color, -1, AA)
            for k in range(8):
                a = k * math.pi / 4.0
                p0 = (int(round(cx + math.cos(a) * r * 0.66)), int(round(cy + math.sin(a) * r * 0.66)))
                p1 = (int(round(cx + math.cos(a) * r)), int(round(cy + math.sin(a) * r)))
                cv2.line(frame, p0, p1, color, th, AA)
        elif kind == "low_light":      # bóng đèn: bầu tròn + đui + dấu gạch
            bc = (cx, cy - int(r * 0.15))
            cv2.circle(frame, bc, int(r * 0.62), color, th, AA)
            cv2.line(frame, (cx - int(r * 0.3), cy + int(r * 0.62)), (cx + int(r * 0.3), cy + int(r * 0.62)), color, th, AA)
            cv2.line(frame, (cx - int(r * 0.2), cy + int(r * 0.95)), (cx + int(r * 0.2), cy + int(r * 0.95)), color, th, AA)
        else:                          # slow_camera: đồng hồ (vòng + hai kim)
            cv2.circle(frame, (cx, cy), int(r * 0.92), color, th, AA)
            cv2.line(frame, (cx, cy), (cx, cy - int(r * 0.6)), color, th, AA)
            cv2.line(frame, (cx, cy), (cx + int(r * 0.45), cy), color, th, AA)

    def _draw_camera_warning(self, frame, st: HudState, s: float, lang: str, w: int) -> int:
        """Vẽ chip dưới chip trạng thái (góc trên-trái). Trả về độ dịch xuống (px) cho toast để không chồng lấn."""
        kind = st.camera_warning
        bad = st.camera_warning_level == "bad"
        text = tr(lang, "cam_warn_" + str(kind))
        m = int(round(16 * s))
        ch = int(round(32 * s))
        pad = int(round(12 * s))
        icon_r = max(5, int(round(8 * s)))
        gap = int(round(9 * s))
        y0 = m + int(round(34 * s)) + int(round(8 * s))          # ngay dưới chip trạng thái (cao 34*s)
        max_w = w - 2 * m - pad * 2 - icon_r * 2 - gap
        fs = self.text.fit_size(text, 14 * s, max_w, bold=bad)
        tw, th = self.text.text_size(text, fs, bold=bad)
        cw = pad + icon_r * 2 + gap + tw + pad
        if bad:
            draw_panel(frame, m, y0, cw, ch, ch // 2, RED_DEEP, 0.90,
                       border_color=RED_SOFT, border_w=max(1, int(round(1.5 * s))), border_alpha=0.95)
            fg = WHITE
        else:
            draw_panel(frame, m, y0, cw, ch, ch // 2, PANEL_BG, 0.72,
                       border_color=AMBER, border_w=max(1, int(round(1.2 * s))), border_alpha=0.85)
            fg = AMBER
        self._draw_cam_icon(frame, str(kind), m + pad + icon_r, y0 + ch // 2, icon_r, fg, s)
        self.text.queue(text, m + pad + icon_r * 2 + gap, y0 + (ch - th) // 2, fs,
                        WHITE if bad else TEXT_MAIN, bold=bad)
        # Chỉ dịch toast xuống khi chip thật sự lấn vào vùng toast căn giữa (toast điển hình rộng ~220*s)
        if m + cw > w // 2 - int(round(110 * s)):
            return ch + int(round(8 * s))
        return 0

    # ---- thanh phụ đề ------------------------------------------------------
    def _sentence_tokens(self, st: HudState, lang: str) -> List[Tuple[str, bool]]:
        """Chuỗi (từ, là-từ-mới-nhất). Chỉ từ đầu câu viết hoa, các từ sau viết thường (chỉ khi hiển thị)."""
        keys = list(st.sentence_keys)
        tokens: List[Tuple[str, bool]] = []
        for i, k in enumerate(keys):
            ph = self.phrase(k, lang)
            ph = ph[:1].upper() + ph[1:].lower() if i == 0 else ph.lower()
            latest = (i == len(keys) - 1)
            for piece in ph.split():
                tokens.append((piece, latest))
        if st.sentence_finalized and tokens:
            last, latest = tokens[-1]
            tokens[-1] = (last + ".", latest)
        return tokens

    def _wrap(self, tokens: List[Tuple[str, bool]], size: int, max_w: int):
        space = max(3, int(round(size * 0.18)))
        lines: List[List[Tuple[str, bool, int]]] = [[]]
        x = 0
        for text, latest in tokens:
            wd = self.text.text_size(text, size, latest)[0]
            if x > 0 and x + space + wd > max_w:
                lines.append([])
                x = 0
            elif x > 0:
                x += space
            lines[-1].append((text, latest, x))
            x += wd
        return lines

    def _draw_subtitle_bar(self, frame, st: HudState, s: float, lang: str, w: int, h: int) -> None:
        m = int(round(16 * s))
        # gợi ý chân trang
        hint = tr(lang, "hint_help")
        hint_fs = 13 * s
        hw, hh = self.text.text_size(hint, hint_fs)
        hint_y = h - int(round(10 * s)) - hh
        hp_x, hp_y = m, hint_y - int(round(2 * s))
        draw_panel(frame, hp_x, hp_y, hw + int(round(16 * s)), hh + int(round(4 * s)),
                   (hh + int(round(4 * s))) // 2, PANEL_BG, 0.55)
        self.text.queue(hint, hp_x + int(round(8 * s)), hint_y, hint_fs, TEXT_DIM)

        bar_x0, bar_x1 = m, w - m
        bar_y1 = hint_y - int(round(5 * s))
        pill_w, pill_h = int(round(84 * s)), int(round(30 * s))
        pad_x = int(round(18 * s))
        pad_y = int(round(11 * s))
        text_x = bar_x0 + pad_x
        text_max_w = (bar_x1 - int(round(12 * s)) - pill_w) - int(round(14 * s)) - text_x

        tokens = self._sentence_tokens(st, lang)
        lines: List = []
        size = int(round(26 * s))
        placeholder = not tokens
        if placeholder:
            size = self.text.fit_size(tr(lang, "placeholder"), 22 * s, text_max_w)
            lines = [[(tr(lang, "placeholder"), False, 0)]]
        else:
            fitted = False
            for base in (26, 23, 20, 18):
                size = int(round(base * s))
                lines = self._wrap(tokens, size, text_max_w)
                if len(lines) <= 2:
                    fitted = True
                    break
            if not fitted:  # vẫn tràn: bỏ dần từ cũ nhất, giữ từ mới nhất, thêm "…" đầu câu
                toks = list(tokens)
                while len(toks) > 1:
                    toks.pop(0)
                    lines = self._wrap([("…", False)] + toks, size, text_max_w)
                    if len(lines) <= 2:
                        break
        line_h = int(round(size * 1.32))
        n_lines = len(lines)
        bar_h = max(int(round(52 * s)), 2 * pad_y + n_lines * line_h)
        bar_y0 = bar_y1 - bar_h
        bar_w = bar_x1 - bar_x0
        radius = int(round(14 * s))
        if st.sentence_finalized and not placeholder:
            draw_panel(frame, bar_x0, bar_y0, bar_w, bar_h, radius, PANEL_BG, 0.76,
                       border_color=GREEN, border_w=max(1, int(round(2 * s))), border_alpha=0.95)
        else:
            draw_panel(frame, bar_x0, bar_y0, bar_w, bar_h, radius, PANEL_BG, 0.72)

        y_text = bar_y0 + (bar_h - n_lines * line_h) // 2
        for li, line in enumerate(lines):
            y = y_text + li * line_h
            for text, latest, x in line:
                if placeholder:
                    self.text.queue(text, text_x, y, size, TEXT_MUTED)
                else:
                    self.text.queue(text, text_x + x, y, size, WHITE if latest else TEXT_OLD, bold=latest)

        # viên thuốc chọn ngôn ngữ EN | VI
        px = bar_x1 - int(round(12 * s)) - pill_w
        py = bar_y0 + (bar_h - pill_h) // 2
        draw_panel(frame, px, py, pill_w, pill_h, pill_h // 2, (70, 70, 70), 0.85)
        seg_w = pill_w // 2
        sel_x = px if lang == "en" else px + seg_w
        inset = max(2, int(round(3 * s)))
        draw_panel(frame, sel_x + inset, py + inset, seg_w - 2 * inset, pill_h - 2 * inset,
                   (pill_h - 2 * inset) // 2, (240, 240, 240), 0.96)
        lfs = 14 * s
        for i, (label, active) in enumerate((("EN", lang == "en"), ("VI", lang == "vi"))):
            tw, th = self.text.text_size(label, lfs, True)
            self.text.queue(label, px + seg_w * i + seg_w // 2, py + (pill_h - th) // 2, lfs,
                            (35, 35, 35) if active else TEXT_DIM, bold=True, anchor="c")

    # ---- overlay trợ giúp --------------------------------------------------
    def _draw_help(self, frame, st: HudState, s: float, lang: str, w: int, h: int) -> None:
        cv2.convertScaleAbs(frame, dst=frame, alpha=0.25)
        pw = min(w - 2 * int(round(16 * s)), int(round(640 * s)))
        pad = int(round(22 * s))
        title = tr(lang, "help_title")
        steps = [tr(lang, "help_step1"), tr(lang, "help_step2"), tr(lang, "help_step3")]
        keys = [("H", "key_h"), ("L", "key_l"), ("T", "key_t"), ("U", "key_u"),
                ("C", "key_c"), ("F", "key_f"), ("P", "key_p"), ("J", "key_j"),
                ("K", "key_k"), ("Q", "key_q")]
        n_rows = (len(keys) + 1) // 2     # 2 cột; 9 phím => 5 hàng
        title_fs = 22 * s
        step_fs = 15 * s
        row_h = int(round(34 * s))
        step_h = int(round(28 * s))
        _, title_h = self.text.text_size(title, title_fs, True)
        ph = (pad + title_h + int(round(14 * s)) + 3 * step_h + int(round(16 * s))
              + n_rows * row_h + int(round(10 * s)) + int(round(18 * s)) + pad)
        px, py = (w - pw) // 2, max(int(round(8 * s)), (h - ph) // 2)
        draw_panel(frame, px, py, pw, ph, int(round(18 * s)), PANEL_BG, 0.96,
                   border_color=(90, 90, 90), border_w=max(1, int(round(1.5 * s))), border_alpha=0.9)

        y = py + pad
        self.text.queue(title, px + pad, y, title_fs, WHITE, bold=True)
        y += title_h + int(round(14 * s))

        inner_w = pw - 2 * pad
        dot_r = max(2, int(round(3.5 * s)))
        for step in steps:
            fs = self.text.fit_size(step, step_fs, inner_w - 22 * s)
            tw, th = self.text.text_size(step, fs)
            cv2.circle(frame, (px + pad + dot_r, y + step_h // 2 - int(round(2 * s))), dot_r, GREEN, -1, cv2.LINE_AA)
            self.text.queue(step, px + pad + int(round(18 * s)), y + (step_h - th) // 2 - int(round(2 * s)), fs, TEXT_MAIN)
            y += step_h
        y += int(round(16 * s))

        col_w = inner_w // 2
        badge = int(round(26 * s))
        label_fs = 15 * s
        for idx, (letter, label_key) in enumerate(keys):
            col, row = idx // n_rows, idx % n_rows
            bx = px + pad + col * col_w
            by = y + row * row_h
            draw_panel(frame, bx, by, badge, badge, int(round(6 * s)), (75, 75, 75), 0.95)
            lw, lh = self.text.text_size(letter, 14 * s, True)
            self.text.queue(letter, bx + badge // 2, by + (badge - lh) // 2, 14 * s, WHITE, bold=True, anchor="c")
            label = tr(lang, label_key)
            fs = self.text.fit_size(label, label_fs, col_w - badge - 20 * s)
            tw, th = self.text.text_size(label, fs)
            self.text.queue(label, bx + badge + int(round(12 * s)), by + (badge - th) // 2, fs, TEXT_OLD)
        y += n_rows * row_h + int(round(10 * s))

        face = tr(lang, {"ok": "face_ok", "searching": "face_searching"}.get(st.face_state, "face_off"))
        diag = tr(lang, "help_diag", cam=st.camera_index, face=face)
        diag_fs = 13 * s
        self.text.queue(diag, px + pad, y, diag_fs, TEXT_MUTED)
        self.text.queue(tr(lang, "help_close"), px + pw - pad, y, diag_fs, TEXT_DIM, anchor="r")


# =============================================================================
# 6. VỆT CHUYỂN ĐỘNG DỊU CHO GIAO DIỆN DEMO
# =============================================================================

def draw_soft_trail(frame: np.ndarray, trail_points: List[Tuple[int, int, float]], curr_time: float,
                    max_age_sec: float = 1.2) -> None:
    """Biến thể dịu của vệt chuyển động: cùng đầu vào với draw_fading_neon_trail nhưng màu trắng-lam nhạt, nét mảnh
    hơn, tỉ lệ theo s = h/480. Không chỉnh sửa hàm gốc."""
    n = len(trail_points)
    if n < 2:
        return
    s = frame.shape[0] / 480.0
    for i in range(1, n):
        age = max(0.0, curr_time - trail_points[i][2])
        if age > max_age_sec:
            continue
        f = _clamp(1.0 - age / max_age_sec, 0.1, 1.0)
        color = (int(235 * f), int(225 * f), int(150 * f))
        thick = max(1, int(round(3.5 * s * f)))
        cv2.line(frame, (trail_points[i - 1][0], trail_points[i - 1][1]),
                 (trail_points[i][0], trail_points[i][1]), color, thick, lineType=cv2.LINE_AA)
