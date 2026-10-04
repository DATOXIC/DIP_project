# -*- coding: utf-8 -*-
"""
ĐẶC TRƯNG TƯ THẾ BÀN TAY 3D CHUẨN HÓA — nhánh nhận diện chữ cái dựa trên landmark.

Vì sao có file này
------------------
Nhánh ảnh cũ (CLAHE + Canny + Sobel, xoay bằng moment) chỉ khử được xoay TRONG MẶT PHẲNG ẢNH.
Khi bàn tay nghiêng / quay khỏi camera, ảnh 2D mất thông tin 3D và CNN thất bại. Ở đây ta làm
chuẩn hóa trong KHÔNG GIAN 3D: dựng hệ trục gắn với chính bàn tay (từ cổ tay + gốc ngón trỏ/giữa/út)
rồi biểu diễn mọi khớp trong hệ trục đó. Kết quả bất biến với: dịch chuyển, tỉ lệ (xa/gần camera),
xoay 3D bất kỳ của cả bàn tay (nếu landmark 3D chính xác), và chiều tay trái/phải (lật gương).

GIỚI HẠN TRUNG THỰC: landmark 3D của MediaPipe là ƯỚC LƯỢNG từ ảnh 2D; khi tay nghiêng nhiều, độ sâu z
nhiễu hơn. Bất biến là chính xác về mặt toán học, còn độ tin cậy dữ liệu đầu vào thì giảm theo góc nghiêng.
Vì vậy `simulate_view` mô phỏng đúng loại sai số đó để tăng cường dữ liệu, và `view_angle_deg` cho phép
đo độ chính xác THEO GÓC NGHIÊNG.

Chỉ phụ thuộc numpy (+ dynamic_features chỉ để lấy độ xòe ngón — nguồn sự thật duy nhất).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dynamic_features import compute_finger_abduction

NUM_LANDMARKS = 21
TIP_IDS = (4, 8, 12, 16, 20)
_FLEX_A = (1, 5, 9, 13, 17)
_FLEX_B = (2, 6, 10, 14, 18)
_FLEX_C = (3, 7, 11, 15, 19)
_TIP_PAIRS = tuple((TIP_IDS[i], TIP_IDS[j]) for i in range(5) for j in range(i + 1, 5))  # 10 cặp

# 60 (toạ độ chuẩn hóa 20 điểm) + 5 (góc gập) + 4 (độ xòe) + 10 (khoảng cách đầu ngón)
FEATURE_DIM = 60 + 5 + 4 + len(_TIP_PAIRS)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def palm_scale(pts: np.ndarray) -> float:
    """Kích thước lòng bàn tay 3D: trung bình |cổ tay -> gốc 4 ngón|. Các gốc ngón là khớp CỨNG nên
    không đổi khi ngón gập (khác với chiều dài ngón)."""
    p = np.asarray(pts, dtype=np.float64)
    return float(np.mean([np.linalg.norm(p[i] - p[0]) for i in (5, 9, 13, 17)]))


def canonicalize(pts: np.ndarray, handedness: Optional[str] = None) -> np.ndarray:
    """
    (21, 3) landmark (hệ camera, đơn vị bất kỳ) -> (21, 3) toạ độ trong hệ trục bàn tay, chia cho palm_scale.

    Hệ trục:  gốc = cổ tay (lm0);  y = hướng cổ tay -> gốc ngón giữa (lm9);
              x = hướng gốc trỏ (lm5) -> gốc út (lm17), trực giao hóa với y;  z = x × y.
    Tay trái: đổi dấu z (phản chiếu gương). Hệ trục định nghĩa x theo (trỏ -> út) nên hai bên ngang
    đã khớp nhau; chỉ phía lòng bàn tay (dấu z) là đảo ngược giữa tay trái và tay phải.
    Suy biến (các điểm trùng nhau) -> trả về toàn 0.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
    s = palm_scale(p)
    if s < 1e-9:
        return np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    y = _unit(p[9] - p[0])
    xr = p[17] - p[5]
    x = xr - float(np.dot(xr, y)) * y
    if float(np.linalg.norm(x)) < 1e-9:
        return np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    x = _unit(x)
    z = np.cross(x, y)
    rot = np.stack([x, y, z], axis=0)          # hàng = trục mới
    canon = ((p - p[0]) @ rot.T) / s
    if handedness is not None and str(handedness).lower().startswith("l"):
        canon[:, 2] *= -1.0
    return canon.astype(np.float32)


def flexion_angles(pts: np.ndarray) -> np.ndarray:
    """5 góc gập ngón ∈ [0,1] (0 = duỗi thẳng) — cùng công thức với compute_joint_flexion_angles của pipeline động."""
    p = np.asarray(pts, dtype=np.float64)
    u = p[list(_FLEX_A)] - p[list(_FLEX_B)]
    v = p[list(_FLEX_C)] - p[list(_FLEX_B)]
    denom = np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-6)
    cosang = np.clip(np.sum(u * v, axis=1) / denom, -1.0, 1.0)
    return (1.0 - np.arccos(cosang) / np.pi).astype(np.float32)


def landmarks_to_features(pts: np.ndarray, handedness: Optional[str] = None) -> np.ndarray:
    """(21,3) -> vector FEATURE_DIM chiều, bất biến dịch chuyển / tỉ lệ / xoay 3D / chiều tay."""
    canon = canonicalize(pts, handedness)
    if not np.any(canon):
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    flex = flexion_angles(canon)
    abd = compute_finger_abduction(canon)
    tips = np.array([np.linalg.norm(canon[a] - canon[b]) for a, b in _TIP_PAIRS], dtype=np.float32)
    return np.concatenate([canon[1:].reshape(-1), flex, abd, tips]).astype(np.float32)


def view_angle_deg(pts: np.ndarray) -> float:
    """
    Góc giữa pháp tuyến lòng bàn tay và trục quang học camera (trục z), tính bằng độ ∈ [0, 90].
    0 = lòng/mu bàn tay hướng thẳng camera (thuận lợi nhất); 90 = bàn tay nhìn ngang như một cạnh (khó nhất).
    Dùng để chia nhóm đo độ chính xác theo mức nghiêng. Chỉ nhận landmark ở hệ camera (trước canonicalize).
    """
    p = np.asarray(pts, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
    n = np.cross(p[5] - p[0], p[17] - p[0])
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return 0.0
    return float(math.degrees(math.acos(min(1.0, abs(float(n[2])) / nn))))


def palm_faces_camera(pts: np.ndarray, handedness: Optional[str]) -> bool:
    """
    True nếu LÒNG bàn tay hướng về camera, False nếu MU bàn tay hướng về camera.
    Dấu của z trong (lm5 - lm0) x (lm17 - lm0) đảo theo mặt bàn tay; tay trái/phải đối xứng gương nên ngược dấu
    nhau. Quy ước TOÀN CỤC được suy ra từ dữ liệu thật (không thể suy từ lý thuyết vì phụ thuộc việc driver camera
    có tự lật gương hay không): với nhãn 'Left' thì z > 0 là lòng bàn tay hướng camera — kiểm chứng bằng hướng cuộn
    ngón (ngón gập cuộn về phía lòng bàn tay nên đầu ngón tiến gần camera). Nếu đổi camera/driver, hãy kiểm tra
    lại bằng mắt qua nhãn LONG/MU trong collect_letters_landmarks.py.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
    nz = float(np.cross(p[5] - p[0], p[17] - p[0])[2])
    is_left = handedness is not None and str(handedness).lower().startswith("l")
    return nz > 0.0 if is_left else nz < 0.0


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def simulate_view(pts: np.ndarray, rng: np.random.Generator,
                  max_yaw: float = 70.0, max_pitch: float = 50.0,
                  z_sigma_range: Tuple[float, float] = (0.0, 0.30),
                  xy_sigma: float = 0.02) -> np.ndarray:
    """
    Mô phỏng một góc nhìn khác của CÙNG dáng tay, kèm sai số ước lượng độ sâu giống cái MediaPipe mắc khi tay nghiêng.

    Chú ý quan trọng: xoay "hoàn hảo" rồi canonicalize cho kết quả giống hệt (đó là tính bất biến), nên xoay
    đơn thuần KHÔNG là tăng cường dữ liệu có ích. Phần có ích là nhiễu độ sâu: sau khi xoay, z bị cộng nhiễu
    Gauss (độ lệch ∝ palm_scale, ngẫu nhiên trong z_sigma_range) và x,y bị rung nhẹ. Cường độ nhiễu là GIẢ ĐỊNH
    cần kiểm chứng bằng dữ liệu thu thật nhiều góc (collect_letters_landmarks.py).
    """
    p = np.asarray(pts, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
    s = max(palm_scale(p), 1e-9)
    rot = _rotation_matrix(math.radians(rng.uniform(-max_yaw, max_yaw)),
                           math.radians(rng.uniform(-max_pitch, max_pitch)),
                           math.radians(rng.uniform(-180.0, 180.0)))
    q = (p - p[0]) @ rot.T
    zs = rng.uniform(*z_sigma_range) * s
    q[:, 2] += rng.normal(0.0, zs, NUM_LANDMARKS)
    q[:, :2] += rng.normal(0.0, xy_sigma * s, (NUM_LANDMARKS, 2))
    return (q + p[0]).astype(np.float32)


def extract_hand(det_result: Any, min_score: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Rút MỘT bàn tay từ kết quả mp.tasks HandLandmarker.detect (duck-typing, không import mediapipe).
    Trả về dict(world (21,3) mét, image (21,3) chuẩn hóa [0,1], handed 'Left'/'Right', score) hoặc None.
    world có thể là None nếu phiên bản MediaPipe không trả về — khi đó phía gọi dùng image với z*bề rộng.
    """
    if det_result is None or not getattr(det_result, "hand_landmarks", None):
        return None
    lm = det_result.hand_landmarks[0]
    image = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
    wl = getattr(det_result, "hand_world_landmarks", None)
    world = np.array([[l.x, l.y, l.z] for l in wl[0]], dtype=np.float32) if wl else None
    handed, score = "Right", 1.0
    hd = getattr(det_result, "handedness", None)
    if hd and hd[0]:
        handed, score = str(hd[0][0].category_name), float(hd[0][0].score)
    if score < min_score:
        return None
    return {"world": world, "image": image, "handed": handed, "score": score}
