"""
GIAO DIỆN DEMO THỜI GIAN THỰC (DEMO HUD) — SONG NGỮ ANH / VIỆT
================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

Module này TÁCH BIỆT khỏi vòng lặp nhận diện: chứa bảng chuỗi song ngữ, bộ vẽ chữ Unicode và vệt chuyển động.
(Lớp DemoHUD/HudState của giao diện demo cũ đã bị gỡ ngày 2026-10-14; giao diện chạy thật là ui_side_panel.py.)

Thành phần chính:
  - UI_STRINGS / tr()        : bảng chuỗi giao diện song ngữ.
  - strip_diacritics()       : bỏ dấu tiếng Việt (dùng cho fallback khi không có font Unicode).
  - TextRenderer             : vẽ chữ Unicode bằng Pillow theo hàng đợi. Mỗi chuỗi được render thành MẶT NẠ xám
                               (cache theo (chuỗi, cỡ, đậm)) rồi trộn trực tiếp vào ROI của khung hình
                               => KHÔNG phải chuyển đổi BGR<->PIL cho cả khung hình, chi phí rất thấp.
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

# =============================================================================
# 5. VỆT CHUYỂN ĐỘNG DỊU CHO GIAO DIỆN DEMO
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
