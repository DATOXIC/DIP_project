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


def compose_canvas(frame: np.ndarray, panel_w: int = PANEL_W, bottom_h: int = BOTTOM_H,
                    bg: Tuple[int, int, int] = BG_COLOR) -> Tuple[np.ndarray, int, int]:
    """Canvas lớn hơn khung camera, dán khung camera NGUYÊN VẸN ở góc trên-trái."""
    h_f, w_f = frame.shape[:2]
    canvas = np.full((h_f + bottom_h, w_f + panel_w, 3), bg, dtype=np.uint8)
    canvas[0:h_f, 0:w_f] = frame
    return canvas, w_f, h_f


def _title(canvas: np.ndarray, x: int, y: int, text: str, w: int) -> int:
    cv2.line(canvas, (x, y), (x + w, y), COL_DIVIDER, 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y + 16), FONT, 0.42, (0, 255, 200), 1, cv2.LINE_AA)
    return y + 24


def _bar(canvas: np.ndarray, x: int, y: int, w: int, h: int, frac: float,
         color: Tuple[int, int, int], bg: Tuple[int, int, int] = (55, 55, 55)) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    fw = int(max(0.0, min(1.0, float(frac))) * w)
    if fw > 0:
        cv2.rectangle(canvas, (x, y), (x + fw, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_right_panel(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), BG_COLOR, -1)
    pad = 14
    x = x0 + pad
    inner_w = w - 2 * pad
    y = y0 + 10

    # 1. Tiêu đề + trạng thái hệ thống ---------------------------------------
    cv2.putText(canvas, "VSL RECOGNITION MONITOR", (x, y + 16), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += 26
    fps_color = COL_OK if st.fps >= 24.0 else (COL_WARN if st.fps >= 20.0 else COL_BAD)
    cv2.putText(canvas, f"FPS:{st.fps:5.1f}  CAM {st.camera_index}  {st.ui_mode}/{st.ui_lang}  loa:{st.speech_status}",
                (x, y + 12), FONT, 0.38, fps_color, 1, cv2.LINE_AA)
    y += 17
    if st.cam_diag:
        cv2.putText(canvas, st.cam_diag, (x, y + 10), FONT, 0.32, (180, 180, 130), 1, cv2.LINE_AA)
        y += 15
    for pline in st.profile_lines:
        cv2.putText(canvas, pline, (x, y + 10), FONT, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
        y += 14
    y += 2
    cv2.putText(canvas, st.face_badge, (x, y + 10), FONT, 0.36, st.face_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, st.seg_text, (x + 130, y + 10), FONT, 0.36, st.seg_color, 1, cv2.LINE_AA)
    y += 17
    cv2.putText(canvas, f"v:{st.v_norm * 100:4.1f}%", (x, y + 10), FONT, 0.34, (0, 255, 255), 1, cv2.LINE_AA)
    _bar(canvas, x + 60, y, inner_w - 60, 11, min(1.0, st.v_norm / 0.15), (0, 255, 255))
    y += 18

    if st.vocabulary_empty:
        cv2.rectangle(canvas, (x, y), (x + inner_w, y + 26), (0, 0, 150), -1)
        cv2.putText(canvas, "CHUA CO TU DIEN! Chay collect_dynamic_templates.py", (x + 6, y + 17),
                    FONT, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        y += 32

    mr = st.match_result

    # 2. Kết quả hiện tại -----------------------------------------------------
    y = _title(canvas, x, y, "KET QUA NHAN DIEN", inner_w)
    if mr is None or not (mr.top_word or mr.top_display):
        cv2.putText(canvas, "(dang cho cu chi...)", (x, y + 12), FONT, 0.36, COL_MUTED, 1, cv2.LINE_AA)
        y += 20
    else:
        word = mr.top_display or mr.top_word
        s_col = status_color(mr.status)
        cv2.putText(canvas, word, (x, y + 20), FONT, 0.70, s_col, 2, cv2.LINE_AA)
        y += 28
        src_col = COL_DL if mr.result_source == "DL" else COL_DTW
        cv2.putText(canvas, f"[{mr.status}]  nguon: {mr.result_source}", (x, y + 11), FONT, 0.36, src_col, 1, cv2.LINE_AA)
        y += 16
        cv2.putText(canvas, f"dist={mr.best_distance:.3f}  conf={mr.best_confidence*100:.0f}%  chat_luong={mr.match_quality:.0f}",
                    (x, y + 10), FONT, 0.32, (210, 210, 210), 1, cv2.LINE_AA)
        y += 15
        if mr.fusion_note:
            cv2.putText(canvas, f"fusion: {mr.fusion_note}"[:52], (x, y + 10), FONT, 0.30, COL_DL, 1, cv2.LINE_AA)
            y += 14
    y += 2

    # 3. Demo DTW vs DL --------------------------------------------------------
    y = _title(canvas, x, y, "QUA TRINH: DTW vs DL", inner_w)
    if mr is None or not mr.dtw_status:
        cv2.putText(canvas, "(chua co du lieu)", (x, y + 12), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        y += 18
    else:
        cv2.putText(canvas, f"DTW -> {mr.dtw_word or '(khong)'}  [{mr.dtw_status}]", (x, y + 11), FONT, 0.34, COL_DTW, 1, cv2.LINE_AA)
        y += 16
        if not st.dl_enabled:
            cv2.putText(canvas, "DL   -> TAT (chua cau hinh --dl-model)", (x, y + 11), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        elif mr.dtw_status == "ACCEPTED":
            cv2.putText(canvas, "DL   -> khong duoc hoi (DTW da ACCEPTED)", (x, y + 11), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        elif mr.dl_confidence < 0:
            cv2.putText(canvas, "DL   -> khong du du lieu de suy luan", (x, y + 11), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        else:
            below = mr.dl_confidence < st.dl_conf_threshold
            tag = "duoi nguong" if below else "DAT nguong"
            cv2.putText(canvas, f"DL   -> {mr.dl_word} ({mr.dl_confidence*100:.0f}%, {tag})",
                        (x, y + 11), FONT, 0.34, (COL_MUTED if below else COL_DL), 1, cv2.LINE_AA)
        y += 16
        decided = "DL da GHI DE DTW" if mr.result_source == "DL" else "DTW giu quyet dinh"
        cv2.putText(canvas, f"=> {decided}", (x, y + 11), FONT, 0.34, (COL_DL if mr.result_source == "DL" else COL_DTW), 1, cv2.LINE_AA)
        y += 16
    y += 2

    # 4. Top-K ứng viên DTW ------------------------------------------------
    y = _title(canvas, x, y, "TOP-K UNG VIEN DTW", inner_w)
    if mr is None or not mr.candidates:
        cv2.putText(canvas, "(chua co du lieu)", (x, y + 12), FONT, 0.34, COL_MUTED, 1, cv2.LINE_AA)
        y += 16
    else:
        for cand in mr.candidates[:4]:
            label = (cand.display_name or cand.word_key)[:14]
            cv2.putText(canvas, label, (x, y + 11), FONT, 0.32, (210, 210, 210), 1, cv2.LINE_AA)
            bar_x = x + 96
            bar_w = inner_w - 96 - 44
            _bar(canvas, bar_x, y, bar_w, 11, cand.confidence, (0, 200, 120))
            cv2.putText(canvas, f"{cand.distance:4.2f}", (bar_x + bar_w + 4, y + 10), FONT, 0.30, (200, 200, 200), 1, cv2.LINE_AA)
            y += 15
    y += 4

    # 5. Đặc trưng 12D ----------------------------------------------------
    y = _title(canvas, x, y, "DAC TRUNG 12D", inner_w)
    if not st.show_features or st.angles is None or len(st.angles) < 5:
        cv2.putText(canvas, "(bam [d] de bat, can thay tay)", (x, y + 12), FONT, 0.32, COL_MUTED, 1, cv2.LINE_AA)
        y += 16
    else:
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        bar_x = x + 56
        bar_w = inner_w - 56 - 36
        for name, val in zip(finger_names, st.angles):
            val_c = max(0.0, min(1.0, float(val)))
            cv2.putText(canvas, name, (x, y + 10), FONT, 0.30, (220, 220, 220), 1, cv2.LINE_AA)
            r_c = int(255 * min(1.0, val_c * 1.5))
            g_c = int(255 * max(0.0, 1.0 - val_c * 1.2))
            _bar(canvas, bar_x, y, bar_w, 11, val_c, (40, g_c, r_c))
            cv2.putText(canvas, f"{int(val_c*100)}%", (bar_x + bar_w + 4, y + 10), FONT, 0.30, (200, 200, 200), 1, cv2.LINE_AA)
            y += 15
        if st.palm_normal is not None and len(st.palm_normal) >= 3:
            cv2.putText(canvas, f"Normal:[{st.palm_normal[0]:+.2f},{st.palm_normal[1]:+.2f},{st.palm_normal[2]:+.2f}] Palm:{st.palm_len:.0f}px",
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
            cv2.putText(canvas, line, (x, y + 10), FONT, 0.30, fade, 1, cv2.LINE_AA)
            y += 15


def draw_bottom_panel(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (14, 14, 14), -1)
    border_col = COL_OK if st.sentence_finalized else COL_WARN
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), border_col, 2)

    pad = 14
    x = x0 + pad
    if st.sentence_words:
        sentence_str = " ".join(st.sentence_words)
        if st.sentence_finalized:
            sentence_str += " ."
        sentence_display = f"CAU: \"{sentence_str}\""
    else:
        sentence_display = "CAU: (dang cho cu chi dau tien...)"

    text_col = COL_OK if st.sentence_finalized else (255, 255, 255)
    font_scale = 0.68
    (tw, _), _ = cv2.getTextSize(sentence_display, FONT, font_scale, 2)
    max_w = w - 2 * pad
    if tw > max_w and tw > 0:
        font_scale = max(0.40, font_scale * (max_w / tw))
    cv2.putText(canvas, sentence_display, (x, y0 + 40), FONT, font_scale, text_col, 2, cv2.LINE_AA)

    if st.help_line:
        cv2.putText(canvas, st.help_line, (x, y0 + h - 16), FONT, 0.34, (190, 190, 190), 1, cv2.LINE_AA)
