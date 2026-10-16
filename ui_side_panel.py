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
_BANNER_CACHE: dict = {}


def draw_top_banner(canvas: np.ndarray, w: int, h: int = 38) -> None:
    """Vẽ banner thông tin môn học, thành viên nhóm và MSSV ở cạnh trên cùng khung camera."""
    if w <= 0 or h <= 0:
        return

    banner_img = _BANNER_CACHE.get((w, h))
    if banner_img is None:
        try:
            import os
            from PIL import Image, ImageDraw, ImageFont

            # Ưu tiên font Segoe UI Bold / Arial Bold để chữ nét đậm, rõ ràng
            font_bold_path = "C:/Windows/Fonts/segoeuib.ttf"
            if not os.path.exists(font_bold_path):
                font_bold_path = "C:/Windows/Fonts/arialbd.ttf"
            if not os.path.exists(font_bold_path):
                font_bold_path = "C:/Windows/Fonts/segoeui.ttf"
            if not os.path.exists(font_bold_path):
                font_bold_path = "C:/Windows/Fonts/arial.ttf"

            if w >= 960:
                f_size = 16
                f_title_size = 12
                title_text = "ĐỒ ÁN DIP | NHẬN DIỆN VSL & ASL"
            elif w >= 760:
                f_size = 15
                f_title_size = 11
                title_text = "ĐỒ ÁN DIP | NHẬN DIỆN VSL & ASL"
            elif w >= 600:
                f_size = 15
                f_title_size = 11
                title_text = "DIP | VSL"
            else:
                f_size = 12
                f_title_size = 10
                title_text = "DIP"

            f_members = ImageFont.truetype(font_bold_path, f_size)
            f_title = ImageFont.truetype(font_bold_path, f_title_size)

            pil_img = Image.new("RGB", (w, h), (13, 16, 22))
            draw = ImageDraw.Draw(pil_img)

            # 1. Thẻ thành viên nổi bật (Pill Badge)
            members_text = "Tô Phương Đạt - 24110086   |   Trương Khắc Nhật Phi - 24110120"
            m_bbox = draw.textbbox((0, 0), members_text, font=f_members)
            m_tw = m_bbox[2] - m_bbox[0]

            dot_gap = 22
            pill_w = m_tw + dot_gap + 14
            pill_h = min(h - 8, 30)
            pill_y = int((h - pill_h) / 2)
            pill_x = max(8, w - pill_w - 8)

            # Khung thẻ nền tối cao cấp với viền Cyan nổi bật
            draw.rounded_rectangle(
                [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                radius=7,
                fill=(24, 32, 46),
                outline=(0, 220, 185),
                width=1,
            )

            # Đèn LED trạng thái phát sáng (Cyan Glow Dot)
            dot_cy = pill_y + int(pill_h / 2)
            dot_cx = pill_x + 11
            draw.ellipse([dot_cx - 4, dot_cy - 4, dot_cx + 4, dot_cy + 4], fill=(0, 140, 120))
            draw.ellipse([dot_cx - 2, dot_cy - 2, dot_cx + 2, dot_cy + 2], fill=(0, 255, 200))

            # Chữ thành viên & MSSV màu trắng sáng nổi bật, căn giữa tuyệt đối
            ty = pill_y + int((pill_h - (m_bbox[1] + m_bbox[3])) / 2)
            draw.text((pill_x + dot_gap, ty), members_text, font=f_members, fill=(255, 255, 255))

            # 2. Tiêu đề môn học / đồ án (bên trái)
            t_bbox = draw.textbbox((0, 0), title_text, font=f_title)
            t_tw = t_bbox[2] - t_bbox[0]
            t_pill_w = t_tw + 16
            t_pill_x = 8
            if t_pill_x + t_pill_w < pill_x - 10:
                draw.rounded_rectangle(
                    [t_pill_x, pill_y, t_pill_x + t_pill_w, pill_y + pill_h],
                    radius=6,
                    fill=(18, 22, 30),
                    outline=(45, 55, 70),
                    width=1,
                )
                t_ty = pill_y + int((pill_h - (t_bbox[1] + t_bbox[3])) / 2)
                draw.text((t_pill_x + 8, t_ty), title_text, font=f_title, fill=(0, 225, 185))

            # Chuyển đổi sang BGR cho OpenCV
            banner_bgr = np.array(pil_img)[:, :, ::-1].copy()

            # Đường viền đáy tinh tế kèm dải màu Cyan nhấn mạnh dưới thẻ thành viên
            banner_bgr[-1, :] = (38, 44, 54)
            banner_bgr[-1, pill_x:pill_x + pill_w] = (0, 220, 185)

            banner_img = banner_bgr
            _BANNER_CACHE[(w, h)] = banner_img
        except Exception:
            # Fallback thuần OpenCV nếu không có PIL hoặc font hệ thống
            banner_bgr = np.full((h, w, 3), (13, 16, 22), dtype=np.uint8)
            title = "DIP PROJECT - VSL" if w >= 800 else "DIP"
            _put(banner_bgr, title, (14, int(h * 0.65)), FONT, 0.40, (0, 220, 180), 1)
            members = "To Phuong Dat - 24110086  |  Truong Khac Nhat Phi - 24110120"
            (tw, _), _ = cv2.getTextSize(members, FONT, 0.45, 1)
            mx = max(160, w - tw - 16)
            cv2.rectangle(banner_bgr, (mx - 8, 4), (w - 6, h - 5), (24, 32, 46), -1)
            cv2.rectangle(banner_bgr, (mx - 8, 4), (w - 6, h - 5), (0, 220, 185), 1)
            _put(banner_bgr, members, (mx, int(h * 0.65)), FONT, 0.45, (255, 255, 255), 1)
            banner_bgr[-1, :] = (38, 44, 54)
            banner_bgr[-1, mx - 8:w - 6] = (0, 220, 185)
            banner_img = banner_bgr
            _BANNER_CACHE[(w, h)] = banner_img

    canvas[0:h, 0:w] = banner_img


def compose_canvas(frame: np.ndarray, panel_w: int = PANEL_W, bottom_h: int = BOTTOM_H,
                    bg: Tuple[int, int, int] = BG_COLOR,
                    draw_banner: bool = True) -> Tuple[np.ndarray, int, int]:
    """Canvas lớn hơn khung camera, dán khung camera ở góc trên-trái kèm banner thông tin nhóm trên đầu."""
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
    if draw_banner:
        draw_top_banner(canvas, w_f)
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


# Bộ đệm lịch sử vận tốc cho đồ thị sóng động học (kinematic waveform sparkline)
_VEL_HISTORY: Deque[float] = deque(maxlen=64)


def _draw_card(canvas: np.ndarray, x: int, y: int, w: int, h: int,
               bg: Tuple[int, int, int] = (24, 24, 24),
               border: Tuple[int, int, int] = (45, 45, 45)) -> None:
    """Vẽ thẻ card nền tối viền thanh lịch."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)


def _draw_hand_visual(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                      st: SidePanelState) -> None:
    """Tầng 1: Trực quan hóa hình dáng 5 ngón tay và la bàn hướng bàn tay 3D."""
    _draw_card(canvas, x, y, w, h)
    _put(canvas, "[1] DANG TAY & HUONG 3D (VISION)", (x + 10, y + 16), FONT, 0.38, (0, 240, 200), 1)

    # 1. 5 Cột ngón tay (Hand Silhouette)
    finger_labels = ["T", "I", "M", "R", "P"]
    finger_max_h = [28, 38, 44, 38, 30]  # Chiều cao theo giải phẫu bàn tay (ngón giữa cao nhất)
    base_y = y + h - 16
    col_w = 12
    start_fx = x + 16

    for i in range(5):
        fx = start_fx + i * 20
        max_fh = finger_max_h[i]
        val = 0.0
        if st.angles is not None and len(st.angles) > i:
            val = max(0.0, min(1.0, float(st.angles[i])))

        # Khung viền ngón tay
        cv2.rectangle(canvas, (fx, base_y - max_fh), (fx + col_w, base_y), (50, 50, 50), 1)
        fill_fh = int(val * max_fh)

        # Màu theo độ duỗi: duỗi thẳng = Xanh Neon, co vừa = Vàng cam, gập chặt = Đỏ cam
        if val >= 0.65:
            col = (0, 240, 140)
        elif val >= 0.35:
            col = (0, 200, 255)
        else:
            col = (50, 70, 240)

        if fill_fh > 0:
            cv2.rectangle(canvas, (fx + 1, base_y - fill_fh), (fx + col_w - 1, base_y), col, -1)

        # Nhãn ngón tay
        _put(canvas, finger_labels[i], (fx + 2, base_y + 11), FONT, 0.30, (200, 200, 200), 1)

    # 2. La bàn Hướng Lòng bàn tay (3D Palm Orientation Gyro)
    cx = x + 180
    cy = y + 44
    r = 21
    cv2.circle(canvas, (cx, cy), r, (40, 40, 40), -1)
    cv2.circle(canvas, (cx, cy), r, (65, 65, 65), 1)
    cv2.line(canvas, (cx - r, cy), (cx + r, cy), (50, 50, 50), 1)
    cv2.line(canvas, (cx, cy - r), (cx, cy + r), (50, 50, 50), 1)

    if st.palm_normal is not None and len(st.palm_normal) >= 3:
        nx, ny, nz = float(st.palm_normal[0]), float(st.palm_normal[1]), float(st.palm_normal[2])
        dx = int(nx * 16)
        dy = int(ny * 16)
        # Mũi tên hướng
        cv2.arrowedLine(canvas, (cx, cy), (cx + dx, cy + dy), (0, 220, 255), 2, tipLength=0.35)
        # Chấm tròn biểu thị độ hướng thẳng vào camera (Nz âm)
        dot_r = max(2, min(6, int(3 + max(0.0, -nz) * 3)))
        cv2.circle(canvas, (cx, cy), dot_r, (0, 255, 120), -1)
        _put(canvas, "HUONG LONG TAY 3D", (x + 212, y + 36), FONT, 0.32, (220, 220, 220), 1)
        _put(canvas, f"Vector:[{nx:+.2f},{ny:+.2f},{nz:+.2f}]", (x + 212, y + 50), FONT, 0.30, (0, 220, 255), 1)
    else:
        cv2.circle(canvas, (cx, cy), 3, (80, 80, 80), -1)
        _put(canvas, "HUONG LONG TAY 3D", (x + 212, y + 36), FONT, 0.32, COL_MUTED, 1)
        _put(canvas, "(chua phat hien tay)", (x + 212, y + 50), FONT, 0.30, COL_MUTED, 1)

    # 3. Chuẩn hóa tỷ lệ (Palm scale)
    _put(canvas, f"Ty le L_palm: {st.palm_len:.0f}px", (x + 212, y + 64), FONT, 0.28, (180, 180, 180), 1)


def _draw_kinematic_waveform(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                             st: SidePanelState) -> None:
    """Tầng 2: Đồ thị sóng vận tốc động học thời gian thực và trạng thái phân đoạn."""
    _draw_card(canvas, x, y, w, h, bg=(18, 18, 18))
    _put(canvas, "[2] DONG HOC & PHAN DOAN (KINEMATICS)", (x + 10, y + 16), FONT, 0.38, (0, 240, 200), 1)

    # Chip trạng thái phân đoạn góc phải
    seg_label = st.seg_text or "CHO TAY"
    (sw, _), _ = cv2.getTextSize(_ascii(seg_label), FONT, 0.32, 1)
    chip_x = x + w - sw - 18
    cv2.rectangle(canvas, (chip_x - 4, y + 5), (x + w - 8, y + 20), (35, 35, 35), -1)
    cv2.rectangle(canvas, (chip_x - 4, y + 5), (x + w - 8, y + 20), st.seg_color, 1)
    _put(canvas, seg_label, (chip_x, y + 16), FONT, 0.32, st.seg_color, 1)

    # Sóng vận tốc: nếu chạy webcam thật (fps > 0) thì tích lũy vào bộ đệm; nếu test tĩnh (fps <= 0) thì dùng mảng cố định để đảm bảo tính tất định
    if st.fps > 0.0:
        _VEL_HISTORY.append(max(0.0, min(1.0, float(st.v_norm) / 0.15)))
        history_vals = list(_VEL_HISTORY)
    else:
        cur_v = max(0.0, min(1.0, float(st.v_norm) / 0.15))
        history_vals = [cur_v] * 32

    # Vùng vẽ sóng
    gx0 = x + 12
    gx1 = x + w - 12
    gy0 = y + 25
    gy1 = y + h - 8
    gw = gx1 - gx0
    gh = gy1 - gy0

    # Lưới viền đồ thị
    cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (32, 32, 32), 1)

    # Vạch ngưỡng Trigger (ở mức 45% chiều cao)
    thresh_y = gy1 - int(0.45 * gh)
    for dot_x in range(gx0, gx1, 8):
        cv2.line(canvas, (dot_x, thresh_y), (min(gx1, dot_x + 4), thresh_y), (0, 165, 255), 1)
    _put(canvas, "Nguong Trigger", (gx0 + 4, thresh_y - 3), FONT, 0.26, (0, 165, 255), 1)

    # Vẽ đường sóng vận tốc
    n_pts = len(history_vals)
    if n_pts >= 2:
        pts = []
        max_len = max(32, len(history_vals))
        for i, val in enumerate(history_vals):
            px = gx0 + int(i * gw / (max_len - 1))
            py = gy1 - int(val * (gh - 4))
            pts.append((px, py))

        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            is_active = (history_vals[i] >= 0.45 or history_vals[i + 1] >= 0.45)
            color = (0, 255, 120) if is_active else (100, 130, 130)
            thickness = 2 if is_active else 1
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

        # Chấm sáng ở điểm hiện tại
        last_p = pts[-1]
        cv2.circle(canvas, last_p, 3, (0, 255, 255), -1, cv2.LINE_AA)


def _draw_matching_spectrum(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                            st: SidePanelState) -> None:
    """Tầng 3: Biểu đồ phổ so khớp Hybrid DTW + Deep Learning."""
    _draw_card(canvas, x, y, w, h)
    _put(canvas, "[3] SO KHOP HYBRID: DTW + DEEP LEARNING", (x + 10, y + 16), FONT, 0.38, (0, 240, 200), 1)

    mr = st.match_result
    if mr is None or not (mr.top_word or mr.top_display):
        _put(canvas, "Dang cho cu chi hoan tat de so khop...", (x + 18, y + 42), FONT, 0.35, COL_MUTED, 1)
        _put(canvas, "DTW so sanh khoang cach quy dao voi tu dien mau", (x + 18, y + 62), FONT, 0.30, (110, 110, 110), 1)
        return

    word = mr.top_display or mr.top_word
    s_col = status_color(mr.status)

    # 1. Thẻ kết quả Top 1 (Winner Badge)
    _fit_text(canvas, word, x + 12, y + 42, 170, 0.80, s_col, 2, min_scale=0.55)

    # Huy hiệu nguồn nhận diện [DTW] hoặc [DL]
    src_is_dl = (mr.result_source == "DL")
    src_bg = (20, 100, 220) if src_is_dl else (180, 110, 20)
    src_text = "GHI DE: DL" if src_is_dl else "NGUON: DTW"
    (bw, _), _ = cv2.getTextSize(src_text, FONT, 0.32, 1)
    cv2.rectangle(canvas, (x + 12, y + 48), (x + 16 + bw, y + 62), src_bg, -1)
    _put(canvas, src_text, (x + 14, y + 59), FONT, 0.32, (255, 255, 255), 1)

    # Thanh đo độ tin cậy to
    conf = max(0.0, min(1.0, float(mr.best_confidence)))
    bar_w = 110
    _bar(canvas, x + 24 + bw, y + 50, bar_w, 11, conf, s_col)
    _put(canvas, f"{conf * 100:.0f}%", (x + 28 + bw + bar_w, y + 60), FONT, 0.34, COL_TEXT, 1)

    # 2. Phổ so sánh Top-K Candidates (Visual Confidence Spectrum)
    top_y = y + 78
    _put(canvas, "TOP UNG VIEN:", (x + 12, top_y), FONT, 0.32, COL_MUTED, 1)

    if mr.candidates:
        cand_x = x + 105
        max_cands = min(3, len(mr.candidates))
        for idx in range(max_cands):
            cand = mr.candidates[idx]
            cy_line = top_y + idx * 16
            c_name = (cand.display_name or cand.word_key)[:10]
            _fit_text(canvas, c_name, cand_x, cy_line, 65, 0.32, (220, 220, 220), 1, min_scale=0.28)
            # Thanh phổ
            c_conf = max(0.0, min(1.0, float(cand.confidence)))
            c_bar_w = 120
            c_bar_col = s_col if idx == 0 else (120, 140, 140)
            _bar(canvas, cand_x + 70, cy_line - 9, c_bar_w, 9, c_conf, c_bar_col)
            # Khoảng cách DTW
            _put(canvas, f"d:{cand.distance:.2f}", (cand_x + 74 + c_bar_w, cy_line), FONT, 0.30, COL_MUTED, 1)
    else:
        _put(canvas, f"dist={mr.best_distance:.2f}  chat_luong={mr.match_quality:.0f}",
             (x + 105, top_y), FONT, 0.32, (200, 200, 200), 1)

    # Chú thích giải thích logic quyết định
    if mr.dtw_status and mr.dtw_status != mr.status:
        _put(canvas, f"DTW: {mr.dtw_word} ({mr.dtw_status})", (x + 12, y + h - 8), FONT, 0.28, COL_DTW, 1)
    elif mr.fusion_note:
        _fit_text(canvas, f"Fusion: {mr.fusion_note}", x + 12, y + h - 8, w - 24, 0.28, COL_DL, 1, min_scale=0.24)


def _draw_pipeline_log(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                       st: SidePanelState) -> None:
    """Tầng 4: Log hoạt động của luồng xử lý và phím tắt."""
    _draw_card(canvas, x, y, w, h)
    _put(canvas, "[4] LOG HOAT DONG (LIVE PIPELINE)", (x + 10, y + 16), FONT, 0.38, (0, 240, 200), 1)

    foot_y = y + h - 10
    if st.event_log is not None:
        items = st.event_log.items()
        max_lines = max(0, (foot_y - 18 - (y + 24)) // 15)
        curr_y = y + 24
        for t_evt, text, color in (items[-max_lines:] if max_lines else []):
            age = max(0.0, time.time() - t_evt)
            ts = time.strftime("%H:%M:%S", time.localtime(t_evt))
            line = f"[{ts}] {_ascii(text)}"
            fade = color if age < 6.0 else tuple(int(c * 0.55) for c in color)
            _fit_text(canvas, line, x + 12, curr_y + 10, w - 24, 0.30, fade, 1, min_scale=0.26)
            curr_y += 15

    _fit_text(canvas, "[u]:Giao dien [s]:Danh van [c]:Xoa cau [n]:Vu vo [x]:Sai tu",
              x + 12, foot_y, w - 24, 0.32, (120, 120, 120), 1, min_scale=0.28)


def _draw_right_debug(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), BG_COLOR, -1)
    pad = 14
    x = x0 + pad
    inner_w = w - 2 * pad
    y = y0 + 8

    # 0. Header (Hệ thống & Chế độ)
    _put(canvas, "VSL PIPELINE MONITOR", (x, y + 16), FONT, 0.50, (255, 255, 255), 2)
    fps_color = COL_OK if st.fps >= 24.0 else (COL_WARN if st.fps >= 20.0 else COL_BAD)
    meta = f"{st.fps:.0f} FPS  |  CAM {st.camera_index}"
    (mw, _), _ = cv2.getTextSize(meta, FONT, 0.38, 1)
    _put(canvas, meta, (x + inner_w - mw, y + 16), FONT, 0.38, fps_color, 1)
    y += 24

    # Sub-header: dòng thứ 2 (dưới y=60 để khớp test_che_do_khac_demo_dung_giao_dien_debug)
    sub = f"{st.ui_mode}/{st.ui_lang}  loa:{st.speech_status}"
    if st.profile_lines:
        sub = f"{st.profile_lines[0]}  {sub}"
    _put(canvas, sub, (x, y + 12), FONT, 0.34, COL_MUTED, 1)
    y += 18

    # Cảnh báo từ điển rỗng (nếu có)
    if st.vocabulary_empty:
        cv2.rectangle(canvas, (x, y), (x + inner_w, y + 24), (0, 0, 150), -1)
        _put(canvas, "CHUA CO TU DIEN! Can thu mau cu chi", (x + 6, y + 16), FONT, 0.32, (255, 255, 255), 1)
        y += 28

    # Tính toán chiều cao các khối theo h của panel (thường h >= 480)
    avail_h = (y0 + h) - y - 10
    card1_h = 76   # Tầng 1: Dáng tay & Gyro
    card2_h = 66   # Tầng 2: Động học & Sóng vận tốc
    card3_h = 132  # Tầng 3: So khớp Hybrid
    card4_h = max(65, avail_h - (card1_h + card2_h + card3_h + 24))  # Tầng 4: Log

    # 1. Tầng 1: Dáng tay 3D & La bàn hướng
    _draw_hand_visual(canvas, x, y, inner_w, card1_h, st)
    y += card1_h + 8

    # 2. Tầng 2: Động học vận tốc & Phân đoạn cử chỉ
    _draw_kinematic_waveform(canvas, x, y, inner_w, card2_h, st)
    y += card2_h + 8

    # 3. Tầng 3: So khớp Hybrid DTW + DL
    _draw_matching_spectrum(canvas, x, y, inner_w, card3_h, st)
    y += card3_h + 8

    # 4. Tầng 4: Log hoạt động & Phím tắt
    _draw_pipeline_log(canvas, x, y, inner_w, card4_h, st)


def _draw_bottom_debug(canvas: np.ndarray, x0: int, y0: int, w: int, h: int, st: SidePanelState) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (16, 16, 16), -1)
    border_col = COL_OK if st.sentence_finalized else COL_WARN
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), border_col, 2)

    pad = 16
    x = x0 + pad
    inner_w = w - 2 * pad

    # Tiêu đề tầng 4: Ngôn ngữ & Ghép câu
    tag_text = "CAU DA CHOT (FINALIZED)" if st.sentence_finalized else "CAU DANG GHEP (BUILDING)"
    _put(canvas, f"[4] NGU PHAP & GHAP CAU: {tag_text}", (x, y0 + 22), FONT, 0.42, border_col, 1)

    # 1. Chuỗi thẻ từ vựng (Word Chips)
    if st.sentence_words:
        cur_x = x
        chip_y = y0 + 32
        chip_h = 24
        for word in st.sentence_words:
            w_str = _ascii(word)
            (tw, _), _ = cv2.getTextSize(w_str, FONT, 0.42, 1)
            chip_w = tw + 14
            if cur_x + chip_w > x + inner_w - 60:
                _put(canvas, "...", (cur_x, chip_y + 17), FONT, 0.42, COL_MUTED, 1)
                break
            is_spell = w_str.startswith("SPELL:")
            chip_bg = (45, 25, 60) if is_spell else (36, 36, 36)
            chip_border = (220, 160, 40) if is_spell else (75, 75, 75)
            cv2.rectangle(canvas, (cur_x, chip_y), (cur_x + chip_w, chip_y + chip_h), chip_bg, -1)
            cv2.rectangle(canvas, (cur_x, chip_y), (cur_x + chip_w, chip_y + chip_h), chip_border, 1)
            _put(canvas, w_str, (cur_x + 7, chip_y + 17), FONT, 0.42, (255, 255, 255), 1)
            cur_x += chip_w + 8

        # Mũi tên chuyển ngữ
        _put(canvas, "->", (cur_x, chip_y + 17), FONT, 0.45, border_col, 1)
    else:
        _put(canvas, "(Dang cho cu chi dau tien de ghep cau...)", (x, y0 + 48), FONT, 0.45, COL_MUTED, 1)

    # 2. Câu tự nhiên hoàn chỉnh (Natural Composed Sentence)
    if st.sentence_finalized and st.composed_sentence:
        natural = _ascii(st.composed_sentence)
        _fit_text(canvas, natural, x, y0 + 88, inner_w, 0.75, COL_OK, 2, min_scale=0.45)
    elif st.sentence_words:
        # Nếu chưa chốt câu, hiện câu nối tạm thời
        temp_str = _ascii(" ".join(st.sentence_words))
        _fit_text(canvas, f"Gloss: \"{temp_str}\"", x, y0 + 84, inner_w, 0.58, (200, 200, 200), 1, min_scale=0.40)

    if st.help_line:
        _fit_text(canvas, st.help_line, x, y0 + h - 14, inner_w, 0.38, (160, 160, 160), 1, min_scale=0.30)



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
