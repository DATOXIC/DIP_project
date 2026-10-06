# -*- coding: utf-8 -*-
"""
GIAO DIỆN PANEL BÊN (phải) + PANEL DƯỚI cho predict_sentence_webcam.py.

Bản ĐƠN GIẢN HÓA: chỉ dùng cv2 (không PIL/font TrueType/thẻ bo góc) để vẽ nhanh, ít phụ
thuộc, dễ đọc code. Mục tiêu KHÔNG đổi: khung camera hoàn toàn sạch (không overlay nào đè
lên), mọi trạng thái/kết quả/log được chuyển ra panel bên phải + panel dưới.

Panel bên phải: trạng thái hệ thống -> kết quả hiện tại -> demo DTW vs DL -> top-K ứng viên
DTW -> đặc trưng 12D -> log sự kiện (thay terminal).
Panel dưới: câu đang ghép + phím tắt.
"""
from __future__ import annotations

import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

import cv2
import numpy as np

PANEL_W = 480
BOTTOM_H = 140
BG_COLOR = (22, 22, 22)

COL_TEXT = (225, 225, 225)
COL_MUTED = (140, 140, 140)
COL_OK = (0, 255, 0)
COL_WARN = (0, 165, 255)
COL_BAD = (0, 0, 255)
COL_DTW = (255, 180, 60)     # xanh dương nhạt (BGR) — nguồn DTW
COL_DL = (60, 170, 255)      # cam nhạt (BGR) — nguồn DL
COL_DIVIDER = (70, 70, 70)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Bảng màu giao diện demo (BGR)
COL_CARD = (34, 34, 34)
COL_ACCENT = (200, 200, 60)
COL_CHIP_IDLE = (110, 110, 110)
COL_CHIP_SEEN = (230, 170, 80)


def _ascii(text: Any) -> str:
    """Bỏ dấu tiếng Việt: font Hershey của cv2.putText không vẽ được ký tự có dấu (ra '???')."""
    s = unicodedata.normalize("NFD", str(text).replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.encode("ascii", "ignore").decode("ascii")


def _put(canvas: np.ndarray, text: Any, org: Tuple[int, int], font: int, scale: float,
         color: Tuple[int, int, int], thickness: int = 1, line_type: int = cv2.LINE_AA) -> None:
    cv2.putText(canvas, _ascii(text), org, font, scale, color, thickness, line_type)


def _fit_text(canvas: np.ndarray, text: Any, x: int, y: int, max_w: int, scale: float,
              color: Tuple[int, int, int], thickness: int = 1, min_scale: float = 0.4) -> int:
    """Vẽ một dòng, co cỡ chữ cho vừa max_w (không co dưới min_scale thì cắt bằng '...'). Trả về chiều rộng đã vẽ."""
    t = _ascii(text)
    sc = scale
    (tw, _), _ = cv2.getTextSize(t, FONT, sc, thickness)
    if tw > max_w and tw > 0:
        sc = max(min_scale, sc * max_w / tw)
    while len(t) > 4 and cv2.getTextSize(t, FONT, sc, thickness)[0][0] > max_w:
        t = t[:-4].rstrip() + "..."
    cv2.putText(canvas, t, (x, y), FONT, sc, color, thickness, cv2.LINE_AA)
    return cv2.getTextSize(t, FONT, sc, thickness)[0][0]


def status_color(status: str) -> Tuple[int, int, int]:
    if status == "ACCEPTED":
        return COL_OK
    if status == "UNCERTAIN":
        return COL_WARN
    return COL_MUTED


class EventLog:
    """Hàng đợi sự kiện gần nhất, thay thế các print() rải rác trong vòng lặp chính."""

    def __init__(self, maxlen: int = 12):
        self._items: Deque[Tuple[float, str, Tuple[int, int, int]]] = deque(maxlen=maxlen)

    def push(self, text: str, color: Tuple[int, int, int] = COL_TEXT) -> None:
        self._items.append((time.time(), text, color))

    def items(self) -> List[Tuple[float, str, Tuple[int, int, int]]]:
        return list(self._items)


@dataclass
class SidePanelState:
    fps: float = 0.0
    camera_index: int = 0
    cam_diag: Optional[str] = None
    face_badge: str = "MAT: TAT"
    face_color: Tuple[int, int, int] = COL_MUTED
    seg_text: str = "CHO TAY"
    seg_color: Tuple[int, int, int] = COL_MUTED
    v_norm: float = 0.0
    vocabulary_empty: bool = False
    hand_found: bool = False
    match_result: Optional[Any] = None
    dl_enabled: bool = False
    dl_conf_threshold: float = 0.38
    sentence_words: List[str] = field(default_factory=list)
    sentence_finalized: bool = False
    show_features: bool = True
    angles: Optional[np.ndarray] = None
    palm_len: float = 80.0
    palm_normal: Optional[np.ndarray] = None
    event_log: Optional[EventLog] = None
    help_line: str = ""
    ui_mode: str = "debug"
    ui_lang: str = "vi"
    profile_lines: List[str] = field(default_factory=list)
    speech_status: str = "on"
    # SPEC 14: câu tự nhiên sinh từ chuỗi từ (sentence_composer). Chỉ hiện khi câu đã chốt; None = không hiện gì thêm.
    # Phải là chuỗi ASCII (cv2.putText không vẽ được chữ có dấu) — người gọi bỏ dấu tiếng Việt trước khi truyền vào.
    composed_sentence: Optional[str] = None


_CANVAS_BASE: dict = {}


def compose_canvas(frame: np.ndarray, panel_w: int = PANEL_W, bottom_h: int = BOTTOM_H,
                    bg: Tuple[int, int, int] = BG_COLOR) -> Tuple[np.ndarray, int, int]:
    """Canvas lớn hơn khung camera, dán khung camera NGUYÊN VẸN ở góc trên-trái."""
    h_f, w_f = frame.shape[:2]
    # np.full với màu 3 kênh tốn ~10 ms/khung (đo trên 1120x620); dựng nền MỘT lần rồi chỉ sao chép (~0,3 ms), kết quả y hệt.
    key = (h_f + bottom_h, w_f + panel_w, tuple(int(c) for c in bg))
    base = _CANVAS_BASE.get(key)
    if base is None:
        base = np.empty((key[0], key[1], 3), dtype=np.uint8)
        base[:] = np.array(key[2], dtype=np.uint8)
        _CANVAS_BASE.clear()          # chỉ giữ một kích thước (kích thước camera hiếm khi đổi)
        _CANVAS_BASE[key] = base
    canvas = base.copy()
    canvas[0:h_f, 0:w_f] = frame
    return canvas, w_f, h_f


def _title(canvas: np.ndarray, x: int, y: int, text: str, w: int) -> int:
    cv2.line(canvas, (x, y), (x + w, y), COL_DIVIDER, 1, cv2.LINE_AA)
    _put(canvas, text, (x, y + 16), FONT, 0.42, (0, 255, 200), 1, cv2.LINE_AA)
    return y + 24


def _bar(canvas: np.ndarray, x: int, y: int, w: int, h: int, frac: float,
         color: Tuple[int, int, int], bg: Tuple[int, int, int] = (55, 55, 55)) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    fw = int(max(0.0, min(1.0, float(frac))) * w)
    if fw > 0:
        cv2.rectangle(canvas, (x, y), (x + fw, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 1)


def _draw_right_debug(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), BG_COLOR, -1)
    pad = 14
    x = x0 + pad
    inner_w = w - 2 * pad
    y = y0 + 10

    # 1. Tiêu đề + trạng thái hệ thống ---------------------------------------
    _put(canvas, "VSL RECOGNITION MONITOR", (x, y + 16), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += 26
    fps_color = COL_OK if st.fps >= 24.0 else (COL_WARN if st.fps >= 20.0 else COL_BAD)
    _put(canvas, f"FPS:{st.fps:5.1f}  CAM {st.camera_index}  {st.ui_mode}/{st.ui_lang}  loa:{st.speech_status}",
                (x, y + 12), FONT, 0.38, fps_color, 1, cv2.LINE_AA)
    y += 17
    if st.cam_diag:
        _put(canvas, st.cam_diag, (x, y + 10), FONT, 0.32, (180, 180, 130), 1, cv2.LINE_AA)
        y += 15
    for pline in st.profile_lines:
        _put(canvas, pline, (x, y + 10), FONT, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
        y += 14
    y += 2
    _put(canvas, st.face_badge, (x, y + 10), FONT, 0.36, st.face_color, 1, cv2.LINE_AA)
    _put(canvas, st.seg_text, (x + 130, y + 10), FONT, 0.36, st.seg_color, 1, cv2.LINE_AA)
    y += 17
    _put(canvas, f"v:{st.v_norm * 100:4.1f}%", (x, y + 10), FONT, 0.34, (0, 255, 255), 1, cv2.LINE_AA)
    _bar(canvas, x + 60, y, inner_w - 60, 11, min(1.0, st.v_norm / 0.15), (0, 255, 255))
    y += 18

    if st.vocabulary_empty:
        cv2.rectangle(canvas, (x, y), (x + inner_w, y + 26), (0, 0, 150), -1)
        _put(canvas, "CHUA CO TU DIEN! Chay collect_dynamic_templates.py", (x + 6, y + 17),
                    FONT, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        y += 32

    mr = st.match_result

    # 2. Kết quả hiện tại -----------------------------------------------------
    y = _title(canvas, x, y, "KET QUA NHAN DIEN", inner_w)
    if mr is None or not (mr.top_word or mr.top_display):
        _put(canvas, "(dang cho cu chi...)", (x, y + 12), FONT, 0.36, COL_MUTED, 1, cv2.LINE_AA)
        y += 20
    else:
        word = mr.top_display or mr.top_word
        s_col = status_color(mr.status)
        _put(canvas, word, (x, y + 20), FONT, 0.70, s_col, 2, cv2.LINE_AA)
        y += 28
        src_col = COL_DL if mr.result_source == "DL" else COL_DTW
        _put(canvas, f"[{mr.status}]  nguon: {mr.result_source}", (x, y + 11), FONT, 0.36, src_col, 1, cv2.LINE_AA)
        y += 16
        _put(canvas, f"dist={mr.best_distance:.3f}  conf={mr.best_confidence*100:.0f}%  chat_luong={mr.match_quality:.0f}",
                    (x, y + 10), FONT, 0.32, (210, 210, 210), 1, cv2.LINE_AA)
        y += 15
        if mr.fusion_note:
            _put(canvas, f"fusion: {mr.fusion_note}"[:52], (x, y + 10), FONT, 0.30, COL_DL, 1, cv2.LINE_AA)
            y += 14
    y += 2

    # 3. Demo DTW vs DL --------------------------------------------------------
    y = _title(canvas, x, y, "QUA TRINH: DTW vs DL", inner_w)
    if mr is None or not mr.dtw_status:
        _put(canvas, "(chua co du lieu)", (x, y + 12), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        y += 18
    else:
        _put(canvas, f"DTW -> {mr.dtw_word or '(khong)'}  [{mr.dtw_status}]", (x, y + 11), FONT, 0.34, COL_DTW, 1, cv2.LINE_AA)
        y += 16
        if not st.dl_enabled:
            _put(canvas, "DL   -> TAT (chua cau hinh --dl-model)", (x, y + 11), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        elif mr.dtw_status == "ACCEPTED":
            _put(canvas, "DL   -> khong duoc hoi (DTW da ACCEPTED)", (x, y + 11), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        elif mr.dl_confidence < 0:
            _put(canvas, "DL   -> khong du du lieu de suy luan", (x, y + 11), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        else:
            below = mr.dl_confidence < st.dl_conf_threshold
            tag = "duoi nguong" if below else "DAT nguong"
            _put(canvas, f"DL   -> {mr.dl_word} ({mr.dl_confidence*100:.0f}%, {tag})",
                        (x, y + 11), FONT, 0.34, (COL_MUTED if below else COL_DL), 1, cv2.LINE_AA)
        y += 16
        decided = "DL da GHI DE DTW" if mr.result_source == "DL" else "DTW giu quyet dinh"
        _put(canvas, f"=> {decided}", (x, y + 11), FONT, 0.34, (COL_DL if mr.result_source == "DL" else COL_DTW), 1, cv2.LINE_AA)
        y += 16
    y += 2

    # 4. Top-K ứng viên DTW ------------------------------------------------
    y = _title(canvas, x, y, "TOP-K UNG VIEN DTW", inner_w)
    if mr is None or not mr.candidates:
        _put(canvas, "(chua co du lieu)", (x, y + 12), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        y += 16
    else:
        for cand in mr.candidates[:4]:
            label = (cand.display_name or cand.word_key)[:14]
            _put(canvas, label, (x, y + 11), FONT, 0.32, (210, 210, 210), 1, cv2.LINE_AA)
            bar_x = x + 96
            bar_w = inner_w - 96 - 44
            _bar(canvas, bar_x, y, bar_w, 11, cand.confidence, (0, 200, 120))
            _put(canvas, f"{cand.distance:4.2f}", (bar_x + bar_w + 4, y + 10), FONT, 0.30, (200, 200, 200), 1, cv2.LINE_AA)
            y += 15
    y += 4

    # 5. Đặc trưng 12D ----------------------------------------------------
    y = _title(canvas, x, y, "DAC TRUNG 12D", inner_w)
    if not st.show_features or st.angles is None or len(st.angles) < 5:
        _put(canvas, "(bam [d] de bat, can thay tay)", (x, y + 12), FONT, 0.32, COL_MUTED, 1, cv2.LINE_AA)
        y += 16
    else:
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        bar_x = x + 56
        bar_w = inner_w - 56 - 36
        for name, val in zip(finger_names, st.angles):
            val_c = max(0.0, min(1.0, float(val)))
            _put(canvas, name, (x, y + 10), FONT, 0.30, (220, 220, 220), 1, cv2.LINE_AA)
            r_c = int(255 * min(1.0, val_c * 1.5))
            g_c = int(255 * max(0.0, 1.0 - val_c * 1.2))
            _bar(canvas, bar_x, y, bar_w, 11, val_c, (40, g_c, r_c))
            _put(canvas, f"{int(val_c*100)}%", (bar_x + bar_w + 4, y + 10), FONT, 0.30, (200, 200, 200), 1, cv2.LINE_AA)
            y += 15
        if st.palm_normal is not None and len(st.palm_normal) >= 3:
            _put(canvas, f"Normal:[{st.palm_normal[0]:+.2f},{st.palm_normal[1]:+.2f},{st.palm_normal[2]:+.2f}] Palm:{st.palm_len:.0f}px",
                        (x, y + 10), FONT, 0.30, (0, 220, 255), 1, cv2.LINE_AA)
            y += 14
    y += 4

    # 6. Log sự kiện (thay terminal) -------------------------------------
    y = _title(canvas, x, y, "LOG SU KIEN (thay terminal)", inner_w)
    log_bottom = y0 + h - 8
    if st.event_log is not None:
        items = st.event_log.items()
        max_lines = max(0, (log_bottom - y) // 15)
        for t_evt, text, color in items[-max_lines:]:
            age = max(0.0, time.time() - t_evt)
            ts = time.strftime("%H:%M:%S", time.localtime(t_evt))
            line = f"[{ts}] {text}"
            if len(line) > 56:
                line = line[:53] + "..."
            fade = color if age < 6.0 else tuple(int(c * 0.55) for c in color)
            _put(canvas, line, (x, y + 10), FONT, 0.30, fade, 1, cv2.LINE_AA)
            y += 15


def _draw_bottom_debug(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (14, 14, 14), -1)
    border_col = COL_OK if st.sentence_finalized else COL_WARN
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), border_col, 2)

    pad = 14
    x = x0 + pad
    if st.sentence_words:
        sentence_str = _ascii(" ".join(st.sentence_words))
        if st.sentence_finalized:
            sentence_str += " ."
        sentence_display = f"CAU: \"{sentence_str}\""
        # Câu quá dài: bỏ dần từ CŨ NHẤT (giữ từ mới) và thêm "..." đầu câu, thay vì co chữ tới mức tràn khỏi panel.
        _tail = [_ascii(w) for w in st.sentence_words]
        while len(_tail) > 1 and cv2.getTextSize(sentence_display, FONT, 0.55, 2)[0][0] > w - 2 * pad:
            _tail.pop(0)
            sentence_display = f"CAU: \"... {' '.join(_tail)}{' .' if st.sentence_finalized else ''}\""
    else:
        sentence_display = "CAU: (dang cho cu chi dau tien...)"

    text_col = COL_OK if st.sentence_finalized else (255, 255, 255)
    font_scale = 0.68
    (tw, _), _ = cv2.getTextSize(sentence_display, FONT, font_scale, 2)
    max_w = w - 2 * pad
    if tw > max_w and tw > 0:
        font_scale = max(0.40, font_scale * (max_w / tw))
    _put(canvas, sentence_display, (x, y0 + 40), FONT, font_scale, text_col, 2, cv2.LINE_AA)

    # SPEC 14: dòng câu tự nhiên bên dưới dòng gloss gốc (không che, không thêm khung).
    if st.sentence_finalized and st.composed_sentence:
        natural = _ascii(st.composed_sentence)
        nat_scale = 0.55
        (nw, _), _ = cv2.getTextSize(natural, FONT, nat_scale, 1)
        if nw > max_w and nw > 0:
            nat_scale = max(0.38, nat_scale * (max_w / nw))
        while len(natural) > 4 and cv2.getTextSize(natural, FONT, nat_scale, 1)[0][0] > max_w:
            natural = natural[:-4].rstrip() + "..."
        _put(canvas, natural, (x, y0 + 75), FONT, nat_scale, COL_OK, 1, cv2.LINE_AA)

    if st.help_line:
        _put(canvas, st.help_line, (x, y0 + h - 16), FONT, 0.34, (190, 190, 190), 1, cv2.LINE_AA)


# =============================================================================
# GIAO DIỆN DEMO (cho người ký / người xem): ít chữ, chữ to, một thứ nổi bật mỗi vùng.
# Bảng kỹ thuật (DTW vs DL, top-K, đặc trưng 12D...) chỉ hiện ở chế độ debug (phím [u] để chuyển).
# =============================================================================
def _chip_state(st: SidePanelState) -> Tuple[str, str, Tuple[int, int, int]]:
    """(dòng chính, dòng phụ, màu) mô tả hệ thống đang ở đâu để người ký biết phải làm gì."""
    if st.vocabulary_empty:
        return "CHUA CO TU DIEN", "Can thu mau cu chi truoc khi nhan dien", COL_BAD
    if st.sentence_finalized and st.sentence_words:
        return "DA CHOT CAU", "Ky tu moi de bat dau cau moi", COL_OK
    seg = (st.seg_text or "").upper()
    if seg == "DANG GHI":
        return "DANG GHI CU CHI", "Dang ghi nhan cu chi cua ban", COL_OK
    if seg == "SAN SANG":
        return "SAN SANG", "Bat dau ky khi ban muon", COL_OK
    if st.hand_found:
        return "DA THAY TAY", "San sang nhan cu chi", COL_CHIP_SEEN
    return "DUA TAY VAO KHUNG HINH", "He thong dang cho tay cua ban", COL_CHIP_IDLE


def _draw_right_demo(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), BG_COLOR, -1)
    pad = 18
    x = x0 + pad
    inner_w = w - 2 * pad
    y = y0 + 16

    # 1. Tiêu đề + thông số hệ thống (nhỏ, không tranh chú ý) ---------------------
    _put(canvas, "VSL", (x, y + 18), FONT, 0.8, (255, 255, 255), 2)
    fps_color = COL_OK if st.fps >= 24.0 else (COL_WARN if st.fps >= 20.0 else COL_BAD)
    meta = f"{st.fps:.0f} FPS  |  CAM {st.camera_index}"
    (mw, _), _ = cv2.getTextSize(meta, FONT, 0.45, 1)
    _put(canvas, meta, (x0 + w - pad - mw, y + 16), FONT, 0.45, fps_color, 1)
    y += 30
    sub = f"{st.face_badge}   loa: {st.speech_status}"
    if st.profile_lines:
        sub = f"{st.profile_lines[0]}   {sub}"
    _fit_text(canvas, sub, x, y + 10, inner_w, 0.4, COL_MUTED, 1, min_scale=0.32)
    y += 26

    # 2. Chip trạng thái ------------------------------------------------------------
    main, hint, chip_col = _chip_state(st)
    chip_h = 64
    cv2.rectangle(canvas, (x, y), (x + inner_w, y + chip_h), COL_CARD, -1)
    cv2.rectangle(canvas, (x, y), (x + 7, y + chip_h), chip_col, -1)
    _fit_text(canvas, main, x + 22, y + 28, inner_w - 34, 0.8, chip_col, 2, min_scale=0.5)
    _fit_text(canvas, hint, x + 22, y + 50, inner_w - 34, 0.46, COL_TEXT, 1, min_scale=0.34)
    y += chip_h + 22

    # 3. Kết quả -------------------------------------------------------------------
    mr = st.match_result
    _put(canvas, "KET QUA", (x, y + 10), FONT, 0.45, COL_MUTED, 1)
    y += 20
    word = (mr.top_display or mr.top_word) if mr is not None else None
    if not word:
        _fit_text(canvas, "...", x, y + 52, inner_w, 1.5, COL_DIVIDER, 3)
        _put(canvas, "Chua co cu chi nao duoc nhan ra", (x, y + 82), FONT, 0.5, COL_MUTED, 1)
        y += 100
    else:
        s_col = status_color(mr.status)
        _fit_text(canvas, word, x, y + 52, inner_w, 1.5, s_col, 3, min_scale=0.7)
        label = {"ACCEPTED": "CHAC CHAN", "UNCERTAIN": "CHUA CHAC"}.get(mr.status, "KHONG NHAN RA")
        _put(canvas, label, (x, y + 82), FONT, 0.55, s_col, 2)
        conf = max(0.0, min(1.0, float(mr.best_confidence)))
        pct = f"{conf * 100:.0f}%"
        (pw, _), _ = cv2.getTextSize(pct, FONT, 0.55, 1)
        _bar(canvas, x, y + 94, inner_w - pw - 12, 12, conf, s_col)
        _put(canvas, pct, (x + inner_w - pw, y + 106), FONT, 0.55, COL_TEXT, 1)
        y += 124
        if mr.status == "UNCERTAIN" and mr.candidates:
            _put(canvas, "Co the la:", (x, y + 10), FONT, 0.45, COL_MUTED, 1)
            y += 24
            for cand in mr.candidates[:3]:
                name = cand.display_name or cand.word_key
                _fit_text(canvas, f"{name}   {cand.confidence * 100:.0f}%", x + 8, y + 10, inner_w - 8, 0.5,
                          COL_TEXT, 1, min_scale=0.38)
                y += 22
            y += 4
    y += 6

    # 4. Hoạt động gần đây -----------------------------------------------------------
    cv2.line(canvas, (x, y), (x + inner_w, y), COL_DIVIDER, 1, cv2.LINE_AA)
    _put(canvas, "HOAT DONG GAN DAY", (x, y + 24), FONT, 0.45, COL_MUTED, 1)
    y += 36
    foot_y = y0 + h - 14
    if st.event_log is not None:
        items = st.event_log.items()
        max_lines = max(0, (foot_y - 22 - y) // 22)
        for t_evt, text, color in (items[-max_lines:] if max_lines else []):
            age = max(0.0, time.time() - t_evt)
            fade = color if age < 6.0 else tuple(int(c * 0.55) for c in color)
            _fit_text(canvas, text, x, y + 12, inner_w, 0.46, fade, 1, min_scale=0.36)
            y += 22
    _put(canvas, "[u] bang ky thuat", (x, foot_y), FONT, 0.4, COL_DIVIDER, 1)


def _draw_bottom_demo(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (14, 14, 14), -1)
    border_col = COL_OK if st.sentence_finalized else COL_WARN
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), border_col, 2)
    pad = 18
    x = x0 + pad
    max_w = w - 2 * pad

    _put(canvas, "CAU DA CHOT" if st.sentence_finalized else "CAU DANG GHEP", (x, y0 + 26), FONT, 0.45,
         border_col, 1)
    if st.sentence_words:
        words = [_ascii(wd) for wd in st.sentence_words]
        dot = "." if st.sentence_finalized else ""
        text = " ".join(words) + dot
        # Câu quá dài: bỏ dần từ CŨ NHẤT (giữ từ mới), thêm "..." đầu câu.
        while len(words) > 1 and cv2.getTextSize(text, FONT, 0.8, 2)[0][0] > max_w:
            words.pop(0)
            text = "... " + " ".join(words) + dot
        _fit_text(canvas, text, x, y0 + 66, max_w, 1.0, COL_OK if st.sentence_finalized else (255, 255, 255), 2,
                  min_scale=0.6)
    else:
        _fit_text(canvas, "Dang cho cu chi dau tien...", x, y0 + 66, max_w, 0.8, COL_MUTED, 1)

    if st.sentence_finalized and st.composed_sentence:
        _fit_text(canvas, st.composed_sentence, x, y0 + 98, max_w, 0.7, COL_OK, 1, min_scale=0.4)

    if st.help_line:
        _fit_text(canvas, st.help_line, x, y0 + h - 14, max_w, 0.42, (190, 190, 190), 1, min_scale=0.3)


def draw_right_panel(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    """Panel phải: ui_mode 'demo' = gọn cho người dùng cuối; mọi giá trị khác = bảng kỹ thuật đầy đủ (debug)."""
    if st.ui_mode == "demo":
        _draw_right_demo(canvas, x0, y0, w, h, st)
    else:
        _draw_right_debug(canvas, x0, y0, w, h, st)


def draw_bottom_panel(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    """Panel dưới: ui_mode 'demo' = câu to, rõ; mọi giá trị khác = bố cục cũ (debug)."""
    if st.ui_mode == "demo":
        _draw_bottom_demo(canvas, x0, y0, w, h, st)
    else:
        _draw_bottom_debug(canvas, x0, y0, w, h, st)
