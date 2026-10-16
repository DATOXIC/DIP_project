# -*- coding: utf-8 -*-
"""
DIP 5-STEP PIPELINE INSPECTOR (Cửa sổ trực quan hóa 5 bước xử lý ảnh số)
Dành cho buổi demo, kiểm tra và bảo vệ đồ án môn Xử lý ảnh số (DIP).

Kiến trúc 5 bước chuẩn hóa:
  [1] Lọc song phương (Bilateral Filter): Khử nhiễu ISO/JPEG bảo toàn cạnh biên sắc nét.
  [2] Phân đoạn màu da thích nghi: Không gian màu sắc độ kép CIELab (a*,b*) + YCrCb (Cr,Cb).
  [3] Chuẩn hóa hình học Image Moments: Xoay trục chính về 0° thẳng đứng, gọt cổ tay, hàn lỗ bóng đổ.
  [4] Tách biên đa tầng Auto-Canny: Ngưỡng tự động theo Median, bắt viền ngón tay mức 1 pixel.
  [5] Đạo hàm không gian Sobel Magnitude: Đo độ dốc |∇I| = sqrt(Gx² + Gy²), bắt nếp gấp lòng bàn tay.
  [*] Ghép Tensor CNN 3 kênh: Tensor (128, 128, 3) = [CLAHE, Auto-Canny, Sobel].
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from improved_dip_pipeline import (
    auto_canny,
    normalize_hand_orientation,
    segment_hand_crop_dip,
)

# Kích thước chuẩn của cửa sổ trực quan
INSPECTOR_W = 1080
INSPECTOR_H = 360

# Bảng màu giao diện tối hiện đại (BGR)
BG_CANVAS = (18, 20, 26)
BG_HEADER = (13, 16, 22)
BG_CARD = (24, 28, 38)
BORDER_CARD = (50, 62, 82)
BORDER_ACTIVE = (0, 220, 185)
TEXT_CYAN = (0, 225, 185)
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (150, 165, 185)
TEXT_WARN = (0, 180, 255)

# Cấu hình 5 thẻ bước DIP
STEP_TITLES = [
    "[1] LỌC SONG PHƯƠNG",
    "[2] PHÂN ĐOẠN MÀU DA",
    "[3] MOMENTS & HÌNH THÁI",
    "[4] TÁCH BIÊN CANNY",
    "[5] GRADIENT SOBEL",
]

STEP_SUBTITLES = [
    "Bilateral Filter (d=5)",
    "CIELab + YCrCb Gate",
    "Moments 0° + Gọt cổ tay",
    "Auto-Canny theo Median",
    "Độ dốc G=sqrt(Gx²+Gy²)",
]

STEP_TAGS = [
    "d=5 | σ=45 (Khử nhiễu)",
    "Lab(a*,b*) + YCrCb",
    "θ = 0.5*atan2(2μ11...)",
    "Ngưỡng thích nghi 1-px",
    "ksize=3 (Nếp gấp 3D)",
]

# Bộ nhớ đệm font chữ PIL
_FONT_TITLE = None
_FONT_CARD = None
_FONT_SUB = None


def _load_fonts():
    global _FONT_TITLE, _FONT_CARD, _FONT_SUB
    if _FONT_TITLE is not None:
        return _FONT_TITLE, _FONT_CARD, _FONT_SUB

    font_path = "C:/Windows/Fonts/segoeuib.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/arialbd.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/segoeui.ttf"

    try:
        _FONT_TITLE = ImageFont.truetype(font_path, 13)
        _FONT_CARD = ImageFont.truetype(font_path, 11)
        _FONT_SUB = ImageFont.truetype(font_path, 10)
    except Exception:
        _FONT_TITLE = ImageFont.load_default()
        _FONT_CARD = ImageFont.load_default()
        _FONT_SUB = ImageFont.load_default()

    return _FONT_TITLE, _FONT_CARD, _FONT_SUB


def _pad_to_target(img: np.ndarray, target_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """Căn giữa và padding ảnh vào canvas vuông giữ nguyên tỉ lệ khung hình."""
    if img is None or img.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Đảm bảo 3 kênh màu
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    scale = min(target_size[0] / w, target_size[1] / h)
    nw = max(1, min(target_size[0], int(w * scale)))
    nh = max(1, min(target_size[1], int(h * scale)))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    sy = (target_size[1] - nh) // 2
    sx = (target_size[0] - nw) // 2
    out[sy : sy + nh, sx : sx + nw] = resized
    return out


def compute_dip_5_steps(
    crop_bgr: Optional[np.ndarray], target_img_size: Tuple[int, int] = (150, 150)
) -> Optional[Dict[str, np.ndarray]]:
    """Thực thi tuần tự 5 bước xử lý ảnh số DIP trên vùng ảnh crop bàn tay."""
    if (
        crop_bgr is None
        or crop_bgr.size == 0
        or crop_bgr.shape[0] < 20
        or crop_bgr.shape[1] < 20
    ):
        return None

    # Bước 1: Bilateral Filter (Lọc song phương bảo toàn biên)
    s1_bilateral = cv2.bilateralFilter(crop_bgr, d=5, sigmaColor=45, sigmaSpace=45)

    # Bước 2: Phân đoạn màu da kép CIELab (a*,b*) + YCrCb (Cr,Cb)
    s2_mask = segment_hand_crop_dip(s1_bilateral)
    s2_vis = cv2.bitwise_and(s1_bilateral, s1_bilateral, mask=s2_mask)
    cnts, _ = cv2.findContours(s2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        # Vẽ viền màu vàng/cyan bao quanh contour bàn tay đã tách
        cv2.drawContours(s2_vis, cnts, -1, (0, 220, 255), 1)

    # Bước 3: Chuẩn hóa hình học bằng Image Moments (xoay trục chính 0° + gọt cổ tay)
    ch, cw = s1_bilateral.shape[:2]
    norm_hand, _, _ = normalize_hand_orientation(
        s1_bilateral,
        s2_mask,
        roi_coords=(0, ch, 0, cw),
        target_size=target_img_size,
        min_hand_area=max(100, int(ch * cw * 0.04)),
    )
    if norm_hand is None:
        s3_norm = _pad_to_target(s2_vis, target_img_size)
    else:
        s3_norm = norm_hand

    # Bước 4: Tách biên đa tầng Auto-Canny (ngưỡng tự động theo median)
    gray = (
        cv2.cvtColor(s3_norm, cv2.COLOR_BGR2GRAY)
        if len(s3_norm.shape) == 3
        else s3_norm.copy()
    )
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    s4_canny = auto_canny(blurred)

    # Tô màu viền Canny: Neon Cyan sắc nét trên nền đen
    s4_vis = np.zeros_like(s3_norm)
    s4_vis[s4_canny > 0] = (0, 255, 180)

    # Bước 5: Đạo hàm không gian Sobel Magnitude |∇I|
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    s5_mag = cv2.normalize(
        np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    s5_vis = cv2.cvtColor(s5_mag, cv2.COLOR_GRAY2BGR)

    # Kênh CLAHE bổ trợ cho Tensor 3 kênh
    blurred_clahe = cv2.GaussianBlur(gray, (3, 3), 0.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_res = clahe.apply(blurred_clahe)
    hand_mask = (gray > 0).astype(np.uint8)
    clahe_masked = cv2.bitwise_and(clahe_res, clahe_res, mask=hand_mask)

    # Tensor 3 kênh BGR: [B: CLAHE, G: Auto-Canny, R: Sobel]
    tensor_composite = np.stack([clahe_masked, s4_canny, s5_mag], axis=2)

    return {
        "step1": _pad_to_target(s1_bilateral, target_img_size),
        "step2": _pad_to_target(s2_vis, target_img_size),
        "step3": s3_norm,
        "step4": s4_vis,
        "step5": s5_vis,
        "tensor": cv2.resize(tensor_composite, (44, 44), interpolation=cv2.INTER_AREA),
    }


_BASE_CACHE: Dict[bool, np.ndarray] = {}

_CARD_W, _CARD_H, _Y_CARD, _GAP = 180, 245, 48, 20
_START_X = (INSPECTOR_W - (5 * _CARD_W + 4 * _GAP)) // 2
_Y_FOOT = INSPECTOR_H - 52


def _build_base(has_hand: bool) -> np.ndarray:
    """Dựng MỘT LẦN toàn bộ phần tĩnh (khung, mũi tên, toàn bộ chữ qua PIL); mỗi khung chỉ copy rồi dán ảnh."""
    canvas = np.full((INSPECTOR_H, INSPECTOR_W, 3), BG_CANVAS, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (INSPECTOR_W, 40), BG_HEADER, -1)
    cv2.line(canvas, (0, 40), (INSPECTOR_W, 40), (45, 55, 70), 1)
    border_col = BORDER_ACTIVE if has_hand else BORDER_CARD
    for i in range(5):
        cx = _START_X + i * (_CARD_W + _GAP)
        cv2.rectangle(canvas, (cx, _Y_CARD), (cx + _CARD_W, _Y_CARD + _CARD_H), BG_CARD, -1)
        cv2.rectangle(canvas, (cx, _Y_CARD), (cx + _CARD_W, _Y_CARD + _CARD_H), border_col, 1)
        cv2.rectangle(canvas, (cx, _Y_CARD), (cx + _CARD_W, _Y_CARD + 26), (30, 36, 48), -1)
        cv2.line(canvas, (cx, _Y_CARD + 26), (cx + _CARD_W, _Y_CARD + 26), (45, 55, 75), 1)
        img_x, img_y = cx + (_CARD_W - 150) // 2, _Y_CARD + 34
        cv2.rectangle(canvas, (img_x - 1, img_y - 1), (img_x + 151, img_y + 151), (45, 55, 75), 1)
        if not has_hand:
            cv2.rectangle(canvas, (img_x, img_y), (img_x + 150, img_y + 150), (12, 14, 18), -1)
            cv2.putText(canvas, "CHO TAY...", (img_x + 36, img_y + 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (90, 100, 115), 1, cv2.LINE_AA)
        if i < 4:
            ax, ay = cx + _CARD_W + 3, _Y_CARD + 109
            cv2.arrowedLine(canvas, (ax, ay), (ax + _GAP - 6, ay),
                            BORDER_ACTIVE if has_hand else (70, 80, 95), 2, tipLength=0.35)
    x_end = _START_X + 5 * _CARD_W + 4 * _GAP
    cv2.rectangle(canvas, (_START_X, _Y_FOOT), (x_end, INSPECTOR_H - 10), (22, 26, 36), -1)
    cv2.rectangle(canvas, (_START_X, _Y_FOOT), (x_end, INSPECTOR_H - 10), BORDER_CARD, 1)

    try:
        f_title, f_card, f_sub = _load_fonts()
        pil_img = Image.fromarray(canvas[:, :, ::-1])
        draw = ImageDraw.Draw(pil_img)
        draw.text((22, 12), "DIP 5-STEP PIPELINE INSPECTOR | HỆ THỐNG TRỰC QUAN XỬ LÝ ẢNH SỐ (VSL / ASL)",
                  font=f_title, fill=(0, 225, 185))
        draw.text((INSPECTOR_W - 250, 13), "● LIVE DIP ENGINE" if has_hand else "○ WAITING FOR HAND",
                  font=f_card, fill=(0, 255, 150) if has_hand else (150, 160, 175))
        draw.text((INSPECTOR_W - 95, 13), "[d]: Đóng", font=f_card, fill=(200, 200, 200))
        for i in range(5):
            cx = _START_X + i * (_CARD_W + _GAP)
            draw.text((cx + 10, _Y_CARD + 6), STEP_TITLES[i], font=f_card, fill=(0, 225, 185))
            draw.text((cx + 10, _Y_CARD + 192), STEP_SUBTITLES[i], font=f_sub, fill=(240, 240, 240))
            draw.text((cx + 10, _Y_CARD + 218), STEP_TAGS[i], font=f_sub, fill=(130, 150, 175))
        draw.text((_START_X + 14, _Y_FOOT + 6),
                  "TENSOR CNN 3 KÊNH (BGR) = [B: CLAHE Tương phản] + [G: Auto-Canny Biên] + [R: Sobel Gradient]",
                  font=f_card, fill=(0, 255, 200))
        draw.text((_START_X + 14, _Y_FOOT + 23),
                  "Quy trình DIP chạy thuần túy trên CPU (OpenCV/NumPy), chuẩn hóa hình học 0° và triệt tiêu biến thiên ánh sáng.",
                  font=f_sub, fill=(155, 168, 185))
        canvas = np.array(pil_img)[:, :, ::-1].copy()
    except Exception:
        cv2.putText(canvas, "DIP 5-STEP PIPELINE INSPECTOR (VSL / ASL)", (20, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, TEXT_CYAN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "[d]: DONG", (INSPECTOR_W - 100, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, TEXT_WHITE, 1, cv2.LINE_AA)
    return canvas


def render_dip_inspector(
    crop_bgr: Optional[np.ndarray] = None, fps: float = 0.0
) -> np.ndarray:
    """
    Dựng khung hình cho cửa sổ DIP 5-Step Pipeline Inspector (1080 x 360 px, BGR).
    Phần chữ/khung là lớp tĩnh được cache (hai biến thể: có tay / chưa có tay); mỗi lần gọi chỉ chạy 5 bước DIP
    trên crop rồi dán ảnh vào bản copy của lớp tĩnh.
    """
    steps_data = compute_dip_5_steps(crop_bgr, target_img_size=(150, 150))
    has_hand = steps_data is not None
    base = _BASE_CACHE.get(has_hand)
    if base is None:
        base = _BASE_CACHE[has_hand] = _build_base(has_hand)
    canvas = base.copy()
    if has_hand:
        for i, key in enumerate(("step1", "step2", "step3", "step4", "step5")):
            img_x = _START_X + i * (_CARD_W + _GAP) + (_CARD_W - 150) // 2
            canvas[_Y_CARD + 34:_Y_CARD + 184, img_x:img_x + 150] = steps_data[key]
        tx = _START_X + 5 * _CARD_W + 4 * _GAP - 52
        ty = _Y_FOOT + 4
        canvas[ty:ty + 44, tx:tx + 44] = steps_data["tensor"]
        cv2.rectangle(canvas, (tx - 1, ty - 1), (tx + 45, ty + 45), BORDER_ACTIVE, 1)
    return canvas
