"""
CHƯƠNG TRÌNH NHẬN DIỆN CỬ CHỈ ĐỘNG & GHÉP CÂU THỜI GIAN THỰC (ASL/VSL SENTENCE BUILDER)
===================================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Kiến trúc 4 Tầng Tinh Gọn (Lean 4-Layer Real-time Pipeline):
  1. TẦNG 1: Trích xuất Đặc trưng Tức thời (BlazeHand + DIP Engine v4 + Two-Stream CNN).
     - Bộ lọc EMA (Exponential Moving Average) làm mượt tọa độ cổ tay (Chống rung cảm biến).
     - Xen kẽ suy luận DIP + CNN mỗi 2 khung hình để tối ưu hóa CPU duy trì FPS >= 25.
  2. TẦNG 2: Bộ Dò Điểm Dừng Phân Đoạn theo Palm-Scale (KineticPauseSegmenter).
     - Dò chuyển động / điểm dừng tự nhiên qua vận tốc chuẩn hóa v_norm.
     - Lọc cổng thời lượng cử chỉ [0.4s, 2.5s] loại bỏ kích hoạt sai.
     - Lệnh ưu tiên cấp tốc Swipe-Left (Gạt tay sang trái) = Xóa từ (Backspace).
  3. TẦNG 3: Bộ So Khớp Chuỗi Lai (HybridDTWEngine) với dải băng Sakoe-Chiba.
     - Chuẩn hóa K=32 khung hình với gốc tọa độ cử chỉ (Stroke-origin anchor).
     - Hàm khoảng cách kết hợp quỹ đạo không gian, hướng tiếp tuyến và dáng tay.
     - Phân tầng 3 mức: ACCEPTED, UNCERTAIN, REJECTED.
  4. TẦNG 4: Bộ Điều Phối Ghép Câu Hoàn Chỉnh & Giao Diện Thoại (Sentence Builder & HUD):
     - Tự động ghép từ thành câu hoàn chỉnh bằng tiếng Anh (ASL).
     - Phát âm thanh giọng nói (TTS) không chặn (Non-blocking background thread) qua pyttsx3.
     - Banner phản hồi trực quan cỡ lớn hiển thị từ nhận diện và điểm tin cậy % giữ trong 2.0s.
     - Vệt sáng chuyển động neon (Fading Neon Motion Trail) phai mờ dần theo thời gian.
     - Hạ tay khỏi màn hình > 1.8s = Chốt câu hoàn chỉnh (Commit/Enter).

Phím điều khiển:
  [c] : Xóa trắng câu thoại làm lại từ đầu (Clear)
  [f] : Lật gương camera (Flip)
  [p] : Mở bảng cài đặt Camera (Auto Exposure / White Balance)
  [q] : Thoát chương trình
"""

import os
import sys
import time
import math
import threading
from typing import List, Tuple, Optional, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp

from train import TwoStreamHandCNN
from improved_dip_pipeline import (
    extract_smart_hand_crop,
    process_lean_dip_tensor
)
import json
from dynamic_dtw_engine import (
    KineticState,
    GestureFrame,
    KineticPauseSegmenter,
    ResampledGesture,
    resample_gesture_sequence,
    HybridDTWEngine,
    HybridDTWConfig,
    MatchResult,
    MatchCandidate
)

# =============================================================================
# 0. BẢNG CẤU HÌNH TRUNG TÂM (CENTRAL CONFIGURATION)
# Người dùng có thể tùy chỉnh mọi thông số tại đây mà không cần tìm trong code!
# =============================================================================
CONFIG = {
    # --- Chọn nguồn Camera: 0 = Webcam máy tính, 1 hoặc 2 = iPhone / Cam ngoài ---
    "camera_source": 1,          # Iriun Webcam (iPhone) = 0 trên máy này; webcam laptop = 1
                                 # (đã xác nhận qua snapshot: index 0 → iPhone, index 1 → laptop webcam)

    # --- Trọng số thuật toán DTW (Golden Triad 12D) ---
    "w_wrist_pos": 0.4,          # Trọng số tọa độ vị trí cổ tay (d_pos)
    "w_finger_angles": 0.8,      # Trọng số góc gập 5 ngón tay (d_angle)
    "w_wrist_dir": 0.5,          # Trọng số hướng vận tốc tiếp tuyến (d_dir)
    "w_palm_normal": 0.5,        # Trọng số vector pháp tuyến lòng bàn tay 3D (d_normal - Cosine)
    "lambda_shape": 0.5,         # Điểm phạt khi dáng tay CNN không khớp
    
    # --- Ngưỡng quyết định nhận diện ---
    "default_threshold": 1.25,   # Ngưỡng chấp nhận từ chuẩn (<= threshold là khớp)
    "uncertainty_ratio": 1.40,   # Vùng nghi vấn: từ threshold đến 1.4 * threshold
    "margin_threshold": 0.12,    # Độ chênh lệch an toàn giữa Top 1 và Top 2 (12%)
    
    # --- Phân đoạn động học (Kinetic Pause Segmenter) ---
    "v_start_ratio": 0.08,       # Vận tốc bắt đầu cử chỉ (8% palm_len/frame)
    "v_stop_ratio": 0.040,       # Vận tốc khựng tay dừng từ (4% palm_len/frame)
    "pause_frames": 9,           # Số frame khựng liên tục để chốt cử chỉ (~0.3s)
}

# =============================================================================
# 1. BẢNG TỪ ĐIỂN TỪ KHÓA & ÁNH XẠ TIẾNG ANH (ASL VOCABULARY MAPPING)
# =============================================================================

ASL_VOCAB_MAP: Dict[str, Dict[str, str]] = {
    "xin_chao": {"en": "Hello",     "vi": "Xin chào"},
    "cam_on":   {"en": "Thank You", "vi": "Cảm ơn"},
    "toi":      {"en": "Me",        "vi": "Tôi"},
    "ban":      {"en": "You",       "vi": "Bạn"},
    "tam_biet": {"en": "Goodbye",   "vi": "Tạm biệt"},
}


def get_asl_phrase(word_key: Optional[str], lang: str = "en", default: str = "Unknown") -> str:
    """Tra cứu cụm từ tiếng Anh / tiếng Việt tương ứng với khóa từ vựng mẫu."""
    if not word_key:
        return default
    info = ASL_VOCAB_MAP.get(word_key)
    if info and lang in info:
        return info[lang]
    return default


# =============================================================================
# 2. BỘ PHÁT ÂM THANH KHÔNG CHẶN (NON-BLOCKING TTS WORKER)
# =============================================================================

import queue

class NonBlockingTTS:
    """
    Bộ phát âm thanh Text-To-Speech chạy nền trong daemon thread chuyên trách, đảm bảo:
    - Không làm chậm hoặc chặn vòng lặp webcam (Latency < 5ms khi phát lệnh).
    - Khởi tạo pyttsx3 và COM STA trong worker thread độc lập.
    - Sử dụng threading.Lock() không chặn: nếu đang bận nói thì yêu cầu mới được bỏ qua an toàn.
    - Thu dọn COM và engine cẩn thận trong khối finally để tránh lỗi engine stuck.
    """
    def __init__(self, speech_rate: int = 175):
        self.speech_rate = speech_rate
        self._lock = threading.Lock()

    @property
    def _is_speaking(self) -> bool:
        return self._lock.locked()

    def _speech_worker(self, text_to_speak: str):
        engine = None
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.speech_rate)
            engine.say(text_to_speak)
            engine.runAndWait()
        except Exception as err:
            print(f"[TTS Error] Speech synthesis error: {err}")
        finally:
            if engine is not None:
                try:
                    if hasattr(engine, '_inLoop') and engine._inLoop:
                        engine._inLoop = False
                    engine.stop()
                except Exception:
                    pass
                try:
                    import pyttsx3
                    if hasattr(pyttsx3, '_activeEngines'):
                        pyttsx3._activeEngines.clear()
                except Exception:
                    pass
                del engine
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
            self._lock.release()

    def speak(self, text: str) -> bool:
        """
        Gửi yêu cầu phát âm thanh không chặn.
        Trả về True nếu lệnh được đưa vào xử lý thành công, False nếu đang bận hoặc văn bản không hợp lệ.
        """
        if not text or not isinstance(text, str):
            return False
        text = text.strip()
        if not text:
            return False
        import re
        clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        if not clean_text:
            return False

        if not self._lock.acquire(blocking=False):
            return False

        _speech_worker = self._speech_worker
        try:
            worker_thread = threading.Thread(target=_speech_worker, args=(clean_text,), daemon=True)
            worker_thread.start()
            return True
        except Exception as err:
            self._lock.release()
            return False


_global_tts_instance: Optional[NonBlockingTTS] = None


def speak_async(text: str) -> bool:
    """Hàm toàn cục phát âm thanh bất đồng bộ không chặn luồng chính."""
    global _global_tts_instance
    if _global_tts_instance is None:
        _global_tts_instance = NonBlockingTTS()
    return _global_tts_instance.speak(text)


# =============================================================================
# 3. BỘ LỌC LÀM MƯỢT TỌA ĐỘ CỔ TAY (1€ FILTER) & HÌNH THÁI HỌC DIP V2.0
# =============================================================================

class WristCoordinateSmoother:
    """
    Bộ lọc tọa độ cổ tay thích ứng đa tần số 1€ Filter (One-Euro Filter, Casiez et al., CHI 2012).
    - Tự động điều chỉnh tần số cắt fc dựa trên vận tốc tức thời:
        fc = fc_min + beta * speed
    - Khi đứng yên / chuyển động chậm: fc tiến về fc_min (1.0 Hz) giúp lọc triệt để rung nhiễu cảm biến.
    - Khi bứt tốc / ngoặt nhanh: fc tăng nhanh theo vận tốc giúp loại bỏ trễ pha (zero-lag).
    - Tương thích ngược 100% với chữ ký update(raw_x, raw_y) và reset().
    """
    def __init__(
        self,
        alpha: float = 0.70,
        f_c_min: Optional[float] = None,
        beta: float = 0.001,
        d_cutoff: float = 1.0
    ):
        self.alpha = float(np.clip(alpha, 0.01, 0.99))
        # Nếu f_c_min không chỉ định rõ, suy dẫn tự nhiên từ alpha theo chu kỳ 30 FPS:
        # fc = alpha / (2 * pi * dt * (1 - alpha))
        if f_c_min is not None:
            self.f_c_min = float(f_c_min)
        else:
            self.f_c_min = float(self.alpha / (2.0 * math.pi * (1.0 / 30.0) * (1.0 - self.alpha)))
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        self._dx: float = 0.0
        self._dy: float = 0.0
        self._prev_ts: Optional[float] = None

    @staticmethod
    def _alpha_from_cutoff(f_c: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(f_c, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def update(
        self,
        raw_x: float,
        raw_y: float,
        timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = float(raw_x)
            self.prev_y = float(raw_y)
            self._prev_ts = timestamp
            return self.prev_x, self.prev_y

        # Nếu không truyền timestamp (ví dụ trong các test harness giả lập): dùng EMA chuẩn
        if timestamp is None or self._prev_ts is None:
            self.prev_x = self.alpha * float(raw_x) + (1.0 - self.alpha) * self.prev_x
            self.prev_y = self.alpha * float(raw_y) + (1.0 - self.alpha) * self.prev_y
            return self.prev_x, self.prev_y

        # Khi có nhãn thời gian thực: áp dụng 1€ Filter thích ứng zero-lag
        dt = max(timestamp - self._prev_ts, 1e-6)

        # 1. Ước lượng đạo hàm vận tốc qua EMA
        alpha_d = self._alpha_from_cutoff(self.d_cutoff, dt)
        raw_dx = (float(raw_x) - self.prev_x) / dt
        raw_dy = (float(raw_y) - self.prev_y) / dt
        self._dx = alpha_d * raw_dx + (1.0 - alpha_d) * self._dx
        self._dy = alpha_d * raw_dy + (1.0 - alpha_d) * self._dy

        # 2. Tần số cắt thích ứng theo vận tốc chuyển động
        speed = math.hypot(self._dx, self._dy)
        f_c = self.f_c_min + self.beta * speed

        # 3. Lọc vị trí tọa độ
        alpha_val = self._alpha_from_cutoff(f_c, dt)
        smooth_x = alpha_val * float(raw_x) + (1.0 - alpha_val) * self.prev_x
        smooth_y = alpha_val * float(raw_y) + (1.0 - alpha_val) * self.prev_y

        self.prev_x = smooth_x
        self.prev_y = smooth_y
        self._prev_ts = timestamp
        return smooth_x, smooth_y

    def reset(self):
        self.prev_x = None
        self.prev_y = None
        self._dx = 0.0
        self._dy = 0.0
        self._prev_ts = None


class LKInertialFallback:
    """
    Phao cứu sinh quán tính Lucas-Kanade (Inertial Fallback Tracker) cho tọa độ cổ tay:
    - Khi MediaPipe BlazeHand tạm thời mất dấu 1-2 frames (do che khuất, bóng đổ, hoặc vung nhanh),
      sử dụng luồng quang thông thưa (Sparse Optical Flow) để nội suy vị trí cổ tay.
    - Giữ nhịp liên tục cho FSM KineticPauseSegmenter, tránh reset stroke giữa chừng.
    - Chỉ chạy fallback tối đa 2 frames liên tiếp; nếu mất dấu >= 4 frames thì chuyển về IDLE.
    """
    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def update_reference(self, gray_frame: np.ndarray, landmarks) -> None:
        """Lưu frame xám và 6 điểm mốc (LM0 cổ tay + 5 đầu ngón tay LM4, 8, 12, 16, 20)."""
        if gray_frame is None or landmarks is None or len(landmarks) < 21:
            return
        self.prev_gray = gray_frame.copy()
        h, w = gray_frame.shape[:2]
        pts = [(landmarks[idx].x * w, landmarks[idx].y * h) for idx in [0, 4, 8, 12, 16, 20]]
        self.prev_points = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def predict_wrist(self, curr_gray: np.ndarray) -> Optional[Tuple[float, float]]:
        """Dự đoán tọa độ cổ tay qua Optical Flow khi MediaPipe mất dấu."""
        if self.prev_gray is None or self.prev_points is None or curr_gray is None:
            return None
        try:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, self.prev_points, None, **self.lk_params
            )
            if status is None or status[0][0] != 1:
                return None
            wrist_pred = next_pts[0][0]
            self.prev_gray = curr_gray.copy()
            self.prev_points = next_pts.copy()
            return float(wrist_pred[0]), float(wrist_pred[1])
        except Exception:
            return None

    def reset(self):
        self.prev_gray = None
        self.prev_points = None


def compute_finger_count_from_defects(
    hand_crop: np.ndarray,
    palm_len: float
) -> Tuple[int, Optional[np.ndarray], Optional[List]]:
    """
    Đếm số ngón tay duỗi (0..5) từ Convex Hull & Convexity Defects trên vùng hand_crop.
    - Áp dụng segment_hand_crop_dip (CIELab + YCrCb + Morphological Closing/Opening) để tách da sạch nền.
    - Lọc defects bằng 2 tiêu chuẩn hình học kép:
        1. depth > 0.15 * palm_len (loại bỏ hoàn toàn răng cưa biên contour).
        2. 20.0 <= theta <= 90.0 (góc nhọn kẽ ngón tay; loại bỏ góc tù > 90° của hõm cổ tay/cẳng tay).
    - Khắc phục Off-by-one: nếu có kẽ ngón hợp lệ thì n_fingers = min(len(defects) + 1, 5);
      nếu không có kẽ nào thì n_fingers = 0 (nắm đấm / chữ A).

    Args:
        hand_crop: ảnh BGR crop bàn tay
        palm_len: độ dài lòng bàn tay chuẩn hóa

    Returns:
        (n_fingers, hull_pts, valid_defects)
    """
    if hand_crop is None or hand_crop.size == 0 or palm_len < 10.0:
        return 0, None, None

    # 1. Phân đoạn da bằng DIP thuần túy
    try:
        from improved_dip_pipeline import segment_hand_crop_dip
        mask = segment_hand_crop_dip(hand_crop)
    except Exception:
        gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if mask is None or mask.size == 0:
        return 0, None, None

    # 2. Tìm contour lớn nhất
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None, None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 300:
        return 0, None, None

    # 3. Tính Convex Hull và Convexity Defects
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        hull_pts = cv2.convexHull(contour, returnPoints=True)
        return 0, hull_pts, None

    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        hull_pts = cv2.convexHull(contour, returnPoints=True)
        return 0, hull_pts, None

    # 4. Lọc defects bằng ngưỡng kép (depth + angle)
    valid_defects = []
    min_depth = 0.15 * palm_len

    for start_idx, end_idx, far_idx, depth_raw in defects.reshape(-1, 4):
        depth = depth_raw / 256.0
        if depth < min_depth:
            continue

        s = contour[start_idx][0].astype(np.float64)
        e = contour[end_idx][0].astype(np.float64)
        f = contour[far_idx][0].astype(np.float64)

        a = np.linalg.norm(e - f)
        b = np.linalg.norm(s - f)
        c = np.linalg.norm(e - s)
        denom = 2.0 * a * b
        if denom < 1e-4:
            continue
        cos_theta = (a**2 + b**2 - c**2) / denom
        theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        if 20.0 <= theta_deg <= 90.0:
            valid_defects.append((contour[start_idx][0], contour[end_idx][0], contour[far_idx][0]))

    if len(valid_defects) > 0:
        n_fingers = min(len(valid_defects) + 1, 5)
    else:
        n_fingers = 0

    hull_pts = cv2.convexHull(contour, returnPoints=True)
    return n_fingers, hull_pts, valid_defects


def apply_soft_morphological_prior(
    probs: np.ndarray,
    n_fingers: int,
    classes: List[str]
) -> np.ndarray:
    """
    Cổng phân xử mềm Bayes (Soft Multiplicative Bayesian Prior Gate) với Adaptive Penalty:
    - Điều chỉnh phân phối xác suất CNN: P_final(S) = P_CNN(S) * gamma(n_fingers, S)
    - Thay thế Hard Veto bằng Adaptive Penalty:
        gamma_penalty = penalty_base + (1.0 - penalty_base) * conf_winner
      Khi CNN tự tin cao (conf_winner -> 1.0), gamma_penalty tiến về 1.0 (không triệt tiêu nhầm chữ B khi nghiêng 45°).
      Khi CNN phân vân (conf_winner -> 0.0), hình thái học DIP đóng vai trò quyết định để bẻ lái về lớp đúng.
    - Chuẩn hóa lại tổng xác suất bằng 1.0.
    """
    GAMMA_TABLE = {
        "A": {"match": [0],    "boost": 1.2, "penalty": 0.5},
        "B": {"match": [3, 4], "boost": 1.2, "penalty": 0.4},
        "C": {"match": [0, 1], "boost": 1.1, "penalty": 0.6},
        "L": {"match": [1],    "boost": 1.2, "penalty": 0.5},
        "V": {"match": [2],    "boost": 1.2, "penalty": 0.4},
    }
    if probs is None or len(probs) == 0:
        return probs

    conf_winner = float(np.max(probs))
    adjusted = np.array(probs, dtype=np.float64)

    for i, cls in enumerate(classes):
        if cls not in GAMMA_TABLE:
            continue
        entry = GAMMA_TABLE[cls]
        if n_fingers in entry["match"]:
            gamma = entry["boost"]
        else:
            base = entry["penalty"]
            gamma = base + (1.0 - base) * conf_winner
        adjusted[i] *= gamma

    total = adjusted.sum()
    if total > 1e-8:
        adjusted /= total
    return adjusted.astype(np.float32)


# =============================================================================
# 4. CÁC HÀM VẼ GIAO DIỆN PHẢN HỒI TRỰC QUAN (HUD & VISUAL FEEDBACK)
# =============================================================================

def draw_fading_neon_trail(
    frame: np.ndarray,
    trail_points: List[Tuple[int, int, float]],
    curr_time: float,
    max_age_sec: float = 1.2
) -> None:
    """
    Vẽ vệt sáng chuyển động cổ tay dạng đa tuyến neon phai mờ dần theo thời gian (Fading Neon Trail).
    Điểm càng cũ thì màu càng tối dần và độ dày nét vẽ càng giảm.
    """
    n_pts = len(trail_points)
    if n_pts < 2:
        return

    for i in range(1, n_pts):
        pt1 = (trail_points[i - 1][0], trail_points[i - 1][1])
        pt2 = (trail_points[i][0], trail_points[i][1])
        pt_time = trail_points[i][2]

        age = max(0.0, curr_time - pt_time)
        if age > max_age_sec:
            continue

        fade_factor = max(0.1, min(1.0, 1.0 - (age / max_age_sec)))
        # Màu neon xanh ngọc / lam (Cyan Neon) phai dần
        color = (
            int(255 * fade_factor),
            int(255 * fade_factor),
            int(40 * fade_factor)
        )
        thickness = max(1, int(4 * fade_factor))
        cv2.line(frame, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


def draw_prominent_hud_banner(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    color: Tuple[int, int, int],
    h_f: int,
    w_f: int
) -> None:
    """
    Vẽ banner thông báo phản hồi nhận diện cỡ lớn, nổi bật ở phần trên khung hình (Held for 2.0s).
    """
    if not title:
        return

    banner_x1 = 30
    banner_x2 = w_f - 30
    banner_y1 = 52
    banner_y2 = 118

    # Lớp phủ bán trong suốt cho banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (banner_x1, banner_y1), (banner_x2, banner_y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Viền sáng màu chủ đạo của banner
    cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2), color, 2)

    # Tiêu đề chính cỡ lớn (Ví dụ: "HELLO (96%)")
    font_scale = 0.88
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
    tx = max(banner_x1 + 15, (w_f - tw) // 2)
    cv2.putText(frame, title, (tx, banner_y1 + 34),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2, lineType=cv2.LINE_AA)

    # Phụ đề giải thích nhỏ hơn bên dưới
    if subtitle:
        sub_scale = 0.45
        (stw, sth), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
        stx = max(banner_x1 + 15, (w_f - stw) // 2)
        cv2.putText(frame, subtitle, (stx, banner_y1 + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 220, 220), 1, lineType=cv2.LINE_AA)


def draw_feature_metrics_hud(
    frame: np.ndarray,
    angles: Optional[np.ndarray],
    palm_len: float,
    v_norm: float,
    wrist_pt: Optional[Tuple[int, int]] = None,
    vel_vector: Optional[Tuple[float, float]] = None,
    palm_normal: Optional[np.ndarray] = None,
    x_offset: int = 15,
    y_offset: int = 55
) -> None:
    """
    Biểu diễn trực quan bộ đặc trưng 12D [P + A + JM + Normal] thời gian thực (Feature Visualizer HUD):
    1. 5 thanh đo góc uốn đốt ngón tay (Joint Flexion Gauges: Thumb, Index, Middle, Ring, Pinky).
    2. Chỉ số kích thước giải phẫu lòng bàn tay (PalmScale px).
    3. Vector pháp tuyến lòng bàn tay 3D (Palm Normal Vector: nx, ny, nz).
    4. Mũi tên vector vận tốc chuyển động (Motion Vector Arrow).
    """
    if angles is None or len(angles) < 5:
        return

    h_panel, w_panel = 180, 205
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + w_panel, y_offset + h_panel), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + w_panel, y_offset + h_panel), (0, 255, 200), 1)

    # Tiêu đề bảng
    cv2.putText(frame, "12D FEATURES [P+A+JM+N]", (x_offset + 10, y_offset + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, lineType=cv2.LINE_AA)

    # 1. 5 thanh đo góc ngón tay (Flexion Angles: 0.0 = Duỗi, 1.0 = Gập)
    finger_names = ["Thm", "Idx", "Mid", "Rng", "Pnk"]
    bar_x = x_offset + 48
    bar_max_w = 95

    for idx, (name, val) in enumerate(zip(finger_names, angles)):
        y_pos = y_offset + 36 + idx * 19
        cv2.putText(frame, name, (x_offset + 10, y_pos + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, lineType=cv2.LINE_AA)

        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_max_w, y_pos + 11), (50, 50, 50), -1)
        val_clamped = max(0.0, min(1.0, float(val)))
        bar_w = int(val_clamped * bar_max_w)
        
        r_c = int(255 * min(1.0, val_clamped * 1.5))
        g_c = int(255 * max(0.0, 1.0 - val_clamped * 1.2))
        bar_col = (40, g_c, r_c)

        if bar_w > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + 11), bar_col, -1)
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_max_w, y_pos + 11), (100, 100, 100), 1)

        pct_text = f"{int(val_clamped * 100)}%"
        cv2.putText(frame, pct_text, (bar_x + bar_max_w + 6, y_pos + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, lineType=cv2.LINE_AA)

    # 2. Thông số Palm Normal Vector [nx, ny, nz]
    norm_y = y_offset + 142
    if palm_normal is not None and len(palm_normal) >= 3:
        n_str = f"N:[{palm_normal[0]:+.2f},{palm_normal[1]:+.2f},{palm_normal[2]:+.2f}]"
        cv2.putText(frame, n_str, (x_offset + 10, norm_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, lineType=cv2.LINE_AA)

    # 3. Thông số Palm Scale & Vận tốc
    info_y = y_offset + h_panel - 10
    cv2.putText(frame, f"Palm: {palm_len:.1f}px | v: {v_norm*100:.1f}%", (x_offset + 10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    # 4. Vẽ mũi tên vector vận tốc tại tâm cổ tay
    if wrist_pt is not None and vel_vector is not None:
        vx, vy = vel_vector
        speed = math.hypot(vx, vy)
        if speed > 1.5:  # Chỉ vẽ khi tay thực sự có chuyển động
            arrow_scale = 1.8
            end_x = int(wrist_pt[0] + np.clip(vx * arrow_scale, -60, 60))
            end_y = int(wrist_pt[1] + np.clip(vy * arrow_scale, -60, 60))
            cv2.arrowedLine(frame, wrist_pt, (end_x, end_y), (0, 255, 255), 2, tipLength=0.25)


HAND_CONNECTIONS = [
    # Ngón cái (Thumb)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Ngón trỏ (Index)
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Ngón giữa (Middle)
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ngón áp út (Ring)
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Ngón út (Pinky)
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Nối ngang gốc bàn tay (Palmar Arch)
    (5, 9), (9, 13), (13, 17)
]


def draw_hand_skeleton_features(
    frame: np.ndarray,
    landmarks: Any,
    angles: Optional[np.ndarray],
    w_f: int,
    h_f: int,
    handedness: str = "Right"
) -> None:
    """
    Biểu diễn trực quan các đặc trưng trích xuất từ 21 điểm mốc trực tiếp lên bàn tay:
    1. Bộ khung xương 21 khớp ngón tay kết nối thành các đoạn xương.
    2. Mỗi ngón tay đổi màu sinh động theo góc uốn thực tế (Xanh lá = Thẳng, Cam/Đỏ = Gập).
    3. Vẽ chấm tròn tại các khớp và hiển thị nhãn góc uốn tại đầu mỗi ngón tay.
    """
    if landmarks is None or len(landmarks) < 21:
        return

    # Tọa độ pixel (x, y) của 21 điểm landmarks
    pts = [(int(lm.x * w_f), int(lm.y * h_f)) for lm in landmarks]

    # Màu sắc ngón tay dựa trên góc uốn (angles: Thm, Idx, Mid, Rng, Pnk)
    # Mặc định xanh lục
    finger_colors = [(0, 255, 100)] * 5
    if angles is not None and len(angles) >= 5:
        for f_idx in range(5):
            val = max(0.0, min(1.0, float(angles[f_idx])))
            # Chuyển màu: Duỗi (Xanh lá) -> Co (Vàng) -> Gập (Đỏ/Cam)
            r_c = int(255 * min(1.0, val * 1.5))
            g_c = int(255 * max(0.0, 1.0 - val * 1.2))
            finger_colors[f_idx] = (40, g_c, r_c)

    # Bản đồ ngón tay ứng với các cạnh nối
    def get_bone_color(p1_idx: int, p2_idx: int) -> Tuple[int, int, int]:
        if p2_idx in [1, 2, 3, 4] or (p1_idx, p2_idx) == (0, 1):
            return finger_colors[0]  # Cái
        elif p2_idx in [5, 6, 7, 8] or (p1_idx, p2_idx) == (0, 5):
            return finger_colors[1]  # Trỏ
        elif p2_idx in [9, 10, 11, 12] or (p1_idx, p2_idx) == (0, 9):
            return finger_colors[2]  # Giữa
        elif p2_idx in [13, 14, 15, 16] or (p1_idx, p2_idx) == (0, 13):
            return finger_colors[3]  # Áp út
        elif p2_idx in [17, 18, 19, 20] or (p1_idx, p2_idx) == (0, 17):
            return finger_colors[4]  # Út
        return (100, 200, 255)       # Vòm lòng bàn tay

    # 1. Vẽ các đường nối khung xương bàn tay
    for p1_i, p2_i in HAND_CONNECTIONS:
        col = get_bone_color(p1_i, p2_i)
        cv2.line(frame, pts[p1_i], pts[p2_i], col, 2, lineType=cv2.LINE_AA)

    # 2. Vẽ các điểm mốc (Joint landmarks)
    for i, pt in enumerate(pts):
        if i in [4, 8, 12, 16, 20]:  # Đầu ngón tay (Fingertip)
            cv2.circle(frame, pt, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (0, 180, 255), 1, lineType=cv2.LINE_AA)
        elif i == 0:  # Cổ tay (Wrist Origin)
            cv2.circle(frame, pt, 6, (255, 0, 128), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pt, 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        else:  # Khớp giữa
            cv2.circle(frame, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # 3. Hiển thị nhãn góc uốn ngay cạnh đầu 5 ngón tay
    if angles is not None and len(angles) >= 5:
        tip_indices = [4, 8, 12, 16, 20]
        finger_labels = ["Thm", "Idx", "Mid", "Rng", "Pnk"]
        for f_i, (tip_i, name) in enumerate(zip(tip_indices, finger_labels)):
            t_pt = pts[tip_i]
            val_pct = int(angles[f_i] * 100)
            tag_text = f"{name}:{val_pct}%"
            # Nhãn góc uốn nhỏ gọn ngay cạnh đầu ngón
            cv2.putText(frame, tag_text, (t_pt[0] + 8, t_pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, finger_colors[f_i], 1, lineType=cv2.LINE_AA)

    # 4. Trực quan hóa Vector Pháp Tuyến 3D của Lòng Bàn Tay (3D Palm Normal Visualization)
    try:
        p0 = np.array([landmarks[0].x * w_f, landmarks[0].y * h_f, landmarks[0].z * w_f], dtype=np.float32)
        p5 = np.array([landmarks[5].x * w_f, landmarks[5].y * h_f, landmarks[5].z * w_f], dtype=np.float32)
        p9 = np.array([landmarks[9].x * w_f, landmarks[9].y * h_f, landmarks[9].z * w_f], dtype=np.float32)
        p17 = np.array([landmarks[17].x * w_f, landmarks[17].y * h_f, landmarks[17].z * w_f], dtype=np.float32)

        # Trọng tâm lòng bàn tay (Palm Center) trong không gian 3D
        palm_center = (p0 + p5 + p9 + p17) / 4.0
        pc_x, pc_y = int(palm_center[0]), int(palm_center[1])
        
        # Tích có hướng tính vector pháp tuyến chuẩn hóa
        u = p5 - p0
        v = p17 - p0
        w = np.cross(u, v)
        norm_w = np.linalg.norm(w)
        if norm_w > 1e-5:
            n_vec = w / norm_w
            # Nếu truyền handedness left -> đảo chiều
            if handedness.lower() == "left":
                n_vec = -n_vec

            # Chiều dài vector 3D tỷ lệ với kích thước bàn tay (~65px)
            arrow_len = 65.0
            
            # Chiếu 3D lên 2D: đầu mũi tên
            target_x = int(pc_x + n_vec[0] * arrow_len)
            target_y = int(pc_y + n_vec[1] * arrow_len)
            
            # Mã màu theo độ sâu z: 
            # n_z > 0 (đâm về phía camera): Xanh Cyan/Neon sáng
            # n_z < 0 (quay vào trong người): Vàng/Cam
            nz_val = float(n_vec[2])
            if nz_val >= 0:
                arrow_col = (255, 255, 0)   # Cyan rực rỡ
                depth_tag = "FORWARD"
            else:
                arrow_col = (0, 165, 255)   # Cam neon
                depth_tag = "BACKWARD"

            # 1. Vẽ đĩa tròn gốc tại tâm lòng bàn tay (Palm Origin Disc)
            cv2.circle(frame, (pc_x, pc_y), 7, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (pc_x, pc_y), 5, arrow_col, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (pc_x, pc_y), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # 2. Vẽ thân mũi tên dày với bóng viền tương phản cao
            cv2.line(frame, (pc_x, pc_y), (target_x, target_y), (0, 0, 0), 5, lineType=cv2.LINE_AA)
            cv2.arrowedLine(frame, (pc_x, pc_y), (target_x, target_y),
                            arrow_col, 3, tipLength=0.28, lineType=cv2.LINE_AA)

            # 3. Vẽ vòng tròn kích thước thay đổi theo trục Z ở đầu mũi tên (Depth Visualizer)
            # Nếu chĩa thẳng vào camera (nz lớn) -> vòng tròn to hơn
            head_radius = int(np.clip(6 + nz_val * 4, 3, 12))
            cv2.circle(frame, (target_x, target_y), head_radius, arrow_col, 2, lineType=cv2.LINE_AA)

            # 4. Nhãn văn bản hiển thị vector [nx, ny, nz] nổi bật cạnh mũi tên
            badge_text = f"NORMAL [{n_vec[0]:+.2f}, {n_vec[1]:+.2f}, {n_vec[2]:+.2f}]"
            bx, by = target_x + 10, target_y - 6
            # Nền badge mờ chống lẫn hình
            (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(frame, (bx - 2, by - bh - 2), (bx + bw + 4, by + 4), (10, 10, 10), -1)
            cv2.rectangle(frame, (bx - 2, by - bh - 2), (bx + bw + 4, by + 4), arrow_col, 1)
            cv2.putText(frame, badge_text, (bx + 1, by - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    except Exception:
        pass


# =============================================================================
# 5. CÔNG NGHỆ NHẬN DIỆN THỦ NGỮ V5.1: GOLDEN TRIAD [P + A + JM] ENGINE
# =============================================================================

def compute_joint_flexion_angles(landmarks_3d: np.ndarray) -> np.ndarray:
    """
    Tính 5 góc uốn đốt ngón tay (Flexion Scores) chuẩn hóa ∈ [0, 1].
    - 0.0 = Duỗi thẳng hoàn toàn (Straight - 0% gập).
    - 1.0 = Gập sát hoàn toàn (Fully flexed - 100% gập).
    
    Landmarks sử dụng:
      Thumb:  LM1 -> LM2 (đỉnh) -> LM3  (CMC -> MCP -> IP)
      Index:  LM5 -> LM6 (đỉnh) -> LM7  (MCP -> PIP -> DIP)
      Middle: LM9 -> LM10 (đỉnh) -> LM11 (MCP -> PIP -> DIP)
      Ring:   LM13 -> LM14 (đỉnh) -> LM15 (MCP -> PIP -> DIP)
      Pinky:  LM17 -> LM18 (đỉnh) -> LM19 (MCP -> PIP -> DIP)
    """
    A_idx = [1, 5, 9, 13, 17]
    B_idx = [2, 6, 10, 14, 18]
    C_idx = [3, 7, 11, 15, 19]

    u = landmarks_3d[A_idx] - landmarks_3d[B_idx]
    v = landmarks_3d[C_idx] - landmarks_3d[B_idx]

    dot_vals = np.sum(u * v, axis=1)
    norm_u = np.linalg.norm(u, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    norm_vals = np.maximum(norm_u * norm_v, 1e-6)

    cos_angles = np.clip(dot_vals / norm_vals, -1.0, 1.0)
    raw_angles = np.arccos(cos_angles)
    flexion = 1.0 - (raw_angles / np.pi)
    return flexion.astype(np.float32)


def compute_palm_normal_vector(landmarks_3d: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """
    Tính vector pháp tuyến 3D của mặt phẳng lòng bàn tay (Palm Normal Vector):
    - Sử dụng 3 mốc cơ sinh học:
        LM 0: Cổ tay (Wrist)
        LM 5: Gốc ngón trỏ (Index MCP)
        LM 17: Gốc ngón út (Pinky MCP)
    - Vector u = LM5 - LM0 (hướng lên dọc lòng bàn tay ngón trỏ)
    - Vector v = LM17 - LM0 (hướng sang ngang vòm lòng bàn tay ngón út)
    - Tích có hướng w = u x v
    - Chuẩn hóa: ||n|| = 1.0
    - Nếu Handedness == "Left": n = -n (để vector luôn đâm vuông góc ra khỏi lòng bàn tay)
    """
    p0 = landmarks_3d[0]
    p5 = landmarks_3d[5]
    p17 = landmarks_3d[17]

    u = p5 - p0
    v = p17 - p0

    w = np.cross(u, v)
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-6:
        # Nếu bàn tay suy biến phẳng thành 1 đường thẳng
        n = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        n = (w / norm_w).astype(np.float32)

    # Đồng nhất chiều đâm ra khỏi mặt lòng bàn tay
    if handedness.lower() == "left":
        n = -n

    return n


def resample_9d_gesture_sequence(
    raw_frames_9d: np.ndarray,
    shapes: List[str],
    k_target: int = 32,
    min_frames: int = 8
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Nội suy chuỗi đặc trưng đa chiều (4D, 9D hoặc 12D) về K=32 frames cố định (Temporal Normalization).
    Tương thích hoàn toàn mảng raw_frames_9d có D kênh (D=4, D=9, D=12).
    """
    n_orig = len(raw_frames_9d)
    if n_orig < min_frames:
        return None, None

    orig_t = np.linspace(0.0, 1.0, n_orig)
    target_t = np.linspace(0.0, 1.0, k_target)

    # Nội suy tất cả các kênh liên tục
    res_channels = []
    for col in range(raw_frames_9d.shape[1]):
        interp_col = np.interp(target_t, orig_t, raw_frames_9d[:, col]).astype(np.float32)
        res_channels.append(interp_col)
    res_traj = np.column_stack(res_channels).astype(np.float32)

    # Nếu có 3 kênh vector pháp tuyến ở cuối (kênh 9, 10, 11), chuẩn hóa lại độ dài vector = 1 sau nội suy
    if res_traj.shape[1] >= 12:
        normals = res_traj[:, 9:12]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        res_traj[:, 9:12] = (normals / norms).astype(np.float32)

    # Nội suy Nearest-Neighbor cho nhãn dáng tay rời rạc
    res_shapes = []
    for t_val in target_t:
        orig_idx = int(np.clip(np.round(t_val * (n_orig - 1)), 0, n_orig - 1))
        res_shapes.append(shapes[orig_idx])

    return res_traj, res_shapes


def compute_golden_triad_dtw_distance(
    q_traj: np.ndarray,
    q_shapes: List[str],
    t_traj: np.ndarray,
    t_shapes: List[str],
    w_p: float = 0.6,
    w_a: float = 1.0,
    w_m: float = 0.4,
    w_n: float = 0.8,
    lambda_shape: float = 0.5,
    sakoe_chiba_radius: int = 4
) -> float:
    """
    Tính khoảng cách DTW 12D với dải băng Sakoe-Chiba và chuẩn hóa theo căn bậc hai số chiều:
      Cost(i, j) = w_p * ||P_q - P_t|| / sqrt(2)
                 + w_m * ||JM_q - JM_t|| / sqrt(2)
                 + w_a * ||A_q - A_t|| / sqrt(5)
                 + w_n * (1.0 - dot(n_q, n_t)) / 2.0  (Cosine distance)
                 + lambda_shape * I(shape_q != shape_t)
    """
    k = len(q_traj)
    r = sakoe_chiba_radius

    d_mat = np.full((k + 1, k + 1), np.inf, dtype=np.float32)
    d_mat[0, 0] = 0.0

    has_template_angles = (t_traj.shape[1] >= 9)
    has_template_normal = (t_traj.shape[1] >= 12 and q_traj.shape[1] >= 12)

    for i in range(1, k + 1):
        j_start = max(1, i - r)
        j_end = min(k, i + r)

        for j in range(j_start, j_end + 1):
            # 1. Quỹ đạo vị trí (x, y) - Chuẩn hóa căn bậc hai của 2
            d_pos = math.hypot(
                float(q_traj[i - 1, 0] - t_traj[j - 1, 0]),
                float(q_traj[i - 1, 1] - t_traj[j - 1, 1])
            ) / 1.41421356

            # 2. Độ dời tức thời (dx, dy) - Chuẩn hóa căn bậc hai của 2
            d_dir = math.hypot(
                float(q_traj[i - 1, 2] - t_traj[j - 1, 2]),
                float(q_traj[i - 1, 3] - t_traj[j - 1, 3])
            ) / 1.41421356

            # 3. Góc uốn 5 ngón tay - Chuẩn hóa căn bậc hai của 5
            if has_template_angles and q_traj.shape[1] >= 9:
                diff_angles = q_traj[i - 1, 4:9] - t_traj[j - 1, 4:9]
                d_angle = float(np.linalg.norm(diff_angles)) / 2.23606798
            else:
                d_angle = 0.0

            # 4. Vector pháp tuyến lòng bàn tay 3D - Khoảng cách góc Cosine [0, 1]
            if has_template_normal:
                nq = q_traj[i - 1, 9:12]
                nt = t_traj[j - 1, 9:12]
                cos_sim = float(np.dot(nq, nt))
                cos_sim = max(-1.0, min(1.0, cos_sim))
                d_norm = (1.0 - cos_sim) / 2.0
            else:
                d_norm = 0.0

            # 5. Hình phạt nhãn dáng tay rời rạc
            d_shape = lambda_shape if (q_shapes[i - 1] != t_shapes[j - 1]) else 0.0

            step_cost = w_p * d_pos + w_m * d_dir + w_a * d_angle + w_n * d_norm + d_shape
            d_mat[i, j] = step_cost + min(
                d_mat[i - 1, j],
                d_mat[i, j - 1],
                d_mat[i - 1, j - 1]
            )

    total_cost = d_mat[k, k]
    if math.isinf(total_cost):
        return 999.0
    return float(total_cost / float(k))


class GoldenTriadDTWMatcher:
    """
    Bộ não so khớp cử chỉ v5.1 tích hợp Golden Triad [P + A + JM].
    Tương thích ngược 100% với cả template 4D cũ và template 9D mới.
    """
    def __init__(
        self,
        templates_path: str = "dynamic_templates.json",
        w_p: float = 0.6,
        w_a: float = 1.0,
        w_m: float = 0.4,
        w_n: float = 0.8,
        lambda_shape: float = 0.5,
        margin_threshold: float = 0.12,
        uncertainty_ratio: float = 1.40,
        threshold_override: Optional[float] = None
    ):
        self.templates_path = templates_path
        self.w_p = w_p
        self.w_a = w_a
        self.w_m = w_m
        self.w_n = w_n
        self.lambda_shape = lambda_shape
        self.margin_threshold = margin_threshold
        self.uncertainty_ratio = uncertainty_ratio
        self.threshold_override = threshold_override
        self.vocabulary: Dict[str, Any] = {}
        self.feature_dim: int = 4
        self.is_legacy_4d: bool = True
        self.load_templates()

    def load_templates(self) -> None:
        if not os.path.exists(self.templates_path):
            return
        try:
            with open(self.templates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vocabulary = data.get("vocabulary", {})
            meta = data.get("metadata", {})
            self.feature_dim = meta.get("feature_dim", 4)
            self.is_legacy_4d = (self.feature_dim < 9)
            if self.is_legacy_4d:
                print(f"[!] LƯU Ý: {self.templates_path} đang chứa mẫu 4D cũ (chưa có 5D góc ngón tay 3D).")
                print("    -> Tự động kích hoạt Adaptive Margin Guard (8%) để chống nghẽn UNCERTAIN.")
                print("    -> Để có độ chính xác cao nhất, hãy chạy 'collect_dynamic_templates.py' để ghi mẫu 9D.")
        except Exception as e:
            print(f"[!] Lỗi đọc {self.templates_path}: {e}")

    @staticmethod
    def compute_margin_ratio(d_top1: float, d_top2: float, eps: float = 1e-4) -> float:
        """Relative Contrast Margin: (D2 - D1) / max(D1, eps)."""
        return float((d_top2 - d_top1) / max(d_top1, eps))

    def match(self, query_traj_9d: np.ndarray, query_shapes: List[str]) -> MatchResult:
        if not self.vocabulary:
            return MatchResult(status="REJECTED", top_word=None, top_display=None, best_distance=999.0, best_confidence=0.0, candidates=[])

        all_candidates: List[MatchCandidate] = []

        for word_key, word_data in self.vocabulary.items():
            display_name = word_data.get("display_name", word_key)
            threshold = float(word_data.get("threshold", 1.25))
            samples = word_data.get("samples", [])
            if not samples:
                continue

            sample_distances = []
            for s in samples:
                t_traj_raw = s.get("traj_features_9d")
                if t_traj_raw is None:
                    t_traj_raw = s.get("traj_features", [])

                t_traj = np.array(t_traj_raw, dtype=np.float32)
                t_shapes = s.get("hand_shapes", ["Unknown"] * 32)

                if len(t_traj) != 32:
                    continue

                dist = compute_golden_triad_dtw_distance(
                    q_traj=query_traj_9d,
                    q_shapes=query_shapes,
                    t_traj=t_traj,
                    t_shapes=t_shapes,
                    w_p=self.w_p,
                    w_a=self.w_a,
                    w_m=self.w_m,
                    w_n=self.w_n,
                    lambda_shape=self.lambda_shape
                )
                sample_distances.append(dist)

            if sample_distances:
                min_dist = min(sample_distances)
                conf = max(0.0, min(1.0, 1.0 - (min_dist / (threshold + 1e-6))))
                all_candidates.append(MatchCandidate(word_key=word_key, display_name=display_name, distance=min_dist, confidence=conf))

        if not all_candidates:
            return MatchResult(status="REJECTED", top_word=None, top_display=None, best_distance=999.0, best_confidence=0.0, candidates=[])

        all_candidates.sort(key=lambda c: c.distance)
        top1 = all_candidates[0]
        base_thresh = float(self.vocabulary.get(top1.word_key, {}).get("threshold", 1.25))
        word_thresh = self.threshold_override if self.threshold_override is not None else base_thresh

        # 1. Kiểm tra ngưỡng tuyệt đối (vượt ngưỡng nghi vấn -> REJECTED)
        if top1.distance > (word_thresh * self.uncertainty_ratio):
            return MatchResult(status="REJECTED", top_word=top1.word_key, top_display=top1.display_name, best_distance=top1.distance, best_confidence=top1.confidence, candidates=all_candidates)

        # 2. Kiểm tra Adaptive Margin Guard nếu có ít nhất 2 ứng viên
        effective_margin = 0.08 if (self.is_legacy_4d and self.margin_threshold >= 0.12) else self.margin_threshold

        if len(all_candidates) >= 2:
            top2 = all_candidates[1]
            margin_ratio = self.compute_margin_ratio(top1.distance, top2.distance)
            
            # Cổng Margin thích ứng: Nếu cử chỉ cực sát mẫu (D1 <= 0.70 * Thresh),
            # chỉ nghi vấn nếu 2 từ gần như hòa điểm tuyệt đối (< 4% chênh lệch)
            req_margin = 0.04 if top1.distance <= (0.70 * word_thresh) else effective_margin
            
            if margin_ratio < req_margin:
                # Quá sít sao -> Nghi vấn (UNCERTAIN)
                return MatchResult(status="UNCERTAIN", top_word=top1.word_key, top_display=top1.display_name, best_distance=top1.distance, best_confidence=top1.confidence, candidates=all_candidates)

        # 3. Chấp nhận (ACCEPTED) hoặc UNCERTAIN theo ngưỡng từ
        if top1.distance <= word_thresh:
            return MatchResult(status="ACCEPTED", top_word=top1.word_key, top_display=top1.display_name, best_distance=top1.distance, best_confidence=top1.confidence, candidates=all_candidates)
        else:
            return MatchResult(status="UNCERTAIN", top_word=top1.word_key, top_display=top1.display_name, best_distance=top1.distance, best_confidence=top1.confidence, candidates=all_candidates)


# =============================================================================
# 6. CHƯƠNG TRÌNH ĐIỀU PHỐI GHÉP CÂU HOÀN CHỈNH (MAIN PIPELINE RUNNER)
# =============================================================================

def run_sentence_builder(
    camera_index: int = 0,
    headless: bool = False,
    max_frames: Optional[int] = None,
    margin_threshold: float = 0.12,
    threshold_override: Optional[float] = None
):
    """
    Vòng lặp chính xử lý nhận diện cử chỉ động & ghép câu thời gian thực (ASL/VSL).
    """
    print("=" * 75)
    print("  HỆ THỐNG NHẬN DIỆN CỬ CHỈ ĐỘNG & GHÉP CÂU ASL (REAL-TIME SENTENCE BUILDER)")
    print("=" * 75)

    # 1. Khởi tạo MediaPipe Hand Landmarker
    model_asset_path = 'hand_landmarker.task'
    if not os.path.exists(model_asset_path):
        print(f"[-] Đang tải model {model_asset_path}...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_asset_path)
        print("✓ Tải xong bộ dò bàn tay!")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )
    detector = HandLandmarker.create_from_options(options)
    print("✓ Đã nạp BlazeHand Detector siêu nhẹ.")

    # 2. Khởi tạo Two-Stream CNN nhận diện Dáng tay tĩnh (Tầng 1)
    device = torch.device('cpu')
    model_path = 'vsl_dataset/models/best_model_v2.pth'
    cnn_model = None
    classes = ['A', 'B', 'C', 'L', 'V']
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        classes = [checkpoint['id_to_label'][i] for i in range(checkpoint['num_classes'])]
        cnn_model = TwoStreamHandCNN(num_classes=len(classes))
        cnn_model.load_state_dict(checkpoint['model_state'])
        cnn_model.eval()
        print(f"✓ Đã nạp Two-Stream CNN ({model_path}): {classes}")
    else:
        print("[!] Không tìm thấy best_model_v2.pth, dùng chế độ giả lập dáng tay.")

    # 3. Khởi tạo DTW Engine (Tầng 3) & Kinetic Segmenter (Tầng 2)
    templates_file = "dynamic_templates.json"
    dtw_engine = GoldenTriadDTWMatcher(
        templates_path=templates_file,
        w_p=CONFIG["w_wrist_pos"],
        w_a=CONFIG["w_finger_angles"],
        w_m=CONFIG["w_wrist_dir"],
        w_n=CONFIG["w_palm_normal"],
        lambda_shape=CONFIG["lambda_shape"],
        margin_threshold=margin_threshold if margin_threshold is not None else CONFIG["margin_threshold"],
        uncertainty_ratio=CONFIG["uncertainty_ratio"],
        threshold_override=threshold_override if threshold_override is not None else CONFIG["default_threshold"]
    )
    if not dtw_engine.vocabulary:
        print("[!] CẢNH BÁO: dynamic_templates.json chưa có mẫu nào hoặc không tồn tại!")
        print("[!] Vui lòng chạy 'collect_dynamic_templates.py' trước để thu mẫu cử chỉ.")
    else:
        print(f"✓ Đã nạp {len(dtw_engine.vocabulary)} từ vựng cử chỉ mẫu từ {templates_file} (Golden Triad Engine).")

    segmenter = KineticPauseSegmenter(
        v_start_ratio=CONFIG["v_start_ratio"],
        v_stop_ratio=CONFIG["v_stop_ratio"],
        pause_frames=CONFIG["pause_frames"],
        min_stroke_frames=8,
        max_stroke_frames=60,
        swipe_threshold=-0.80,
        swipe_ratio=2.5,
        swipe_cooldown_sec=1.0
    )

    # Khởi tạo bộ làm mượt tọa độ cổ tay (R3 Guard 1) - 1€ Filter thích ứng đa tần số
    wrist_smoother = WristCoordinateSmoother(alpha=0.70, f_c_min=1.0, beta=0.007, d_cutoff=1.0)
    lk_fallback = LKInertialFallback()
    cached_palm_len: float = 80.0
    cached_n_fingers: int = 0
    cached_hull_pts: Optional[np.ndarray] = None
    cached_valid_defects: Optional[List] = None
    stroke_angles_history: List[np.ndarray] = []
    stroke_normals_history: List[np.ndarray] = []

    # 4. Quản lý Bộ đệm Câu thoại (Tầng 4)
    sentence_words: List[str] = []
    last_commit_time = 0.0
    last_committed_word: Optional[str] = None
    last_hand_seen_time = time.time()
    sentence_finalized = False

    # Quản lý Banner phản hồi trực quan (R4 HUD)
    banner_title = ""
    banner_subtitle = ""
    banner_color = (0, 255, 0)
    banner_expire = 0.0

    # Quản lý Vệt sáng chuyển động neon (R4 Motion Trail)
    # Lưu các điểm dạng tuple: (x, y, timestamp)
    trail_points: List[Tuple[int, int, float]] = []
    MAX_TRAIL_AGE_SEC = 1.2
    hand_lost_frames = 0

    # Biến phục vụ tối ưu hóa xen kẽ suy luận DIP + CNN trên CPU
    frame_counter = 0
    cached_hand_shape = "B"
    cached_box = (0, 0, 0, 0)

    # Mở Webcam — thử MSMF trước (hoạt động tốt với Iriun Webcam iPhone),
    # rồi fallback sang DSHOW nếu MSMF không đọc được frame thực sự.
    cap = None
    if not headless:
        def _try_open_camera(index: int, backend) -> "cv2.VideoCapture | None":
            """Mở camera với backend chỉ định, trả về cap nếu đọc được frame thực, None nếu thất bại."""
            _cap = cv2.VideoCapture(index, backend)
            if not _cap.isOpened():
                _cap.release()
                return None
            _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, _ = _cap.read()   # kiểm tra đọc frame thực sự
            if not ret:
                _cap.release()
                return None
            return _cap

        backend_order = [
            (cv2.CAP_MSMF, "MSMF (Microsoft Media Foundation)"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Default"),
        ]

        print(f"[Camera] Đang khởi tạo Camera Index [{camera_index}]...")
        for backend_id, backend_name in backend_order:
            cap = _try_open_camera(camera_index, backend_id)
            if cap is not None:
                print(f"  → Backend: {backend_name}")
                break
            print(f"  [Skip] {backend_name} không đọc được frame tại index {camera_index}")

        if cap is not None and cap.isOpened():
            # Warm-up buffer: xả thêm vài frame đệm ban đầu
            for _ in range(4):
                cap.read()
            cam_type_name = "iPhone (Iriun Webcam)" if camera_index == 1 else ("Webcam Laptop" if camera_index == 0 else f"Camera Phu {camera_index}")
            print(f"[Camera] KET NOI THANH CONG: {cam_type_name} (Index {camera_index})")
        else:
            print(f"[!] KHONG THE mo Camera [{camera_index}] voi bat ky backend nao!")

    flip_cam = False
    show_features = True
    prev_time = time.time()
    fps = 0.0

    print("\n[HƯỚNG DẪN SỬ DỤNG]")
    print("  - Múa cử chỉ rõ ràng, khựng tay dừng dứt khoát 0.35s để chốt từ.")
    print("  - Gạt nhanh tay sang trái (Swipe-Left) để XÓA TỪ gần nhất (Backspace).")
    print("  - Hạ tay xuống khỏi màn hình > 1.8s để HOÀN THÀNH CÂU (Commit/Enter).")
    print("  - Hệ thống tự động phát âm tiếng Anh (ASL) bằng giọng nói pyttsx3.")
    print("  [d]: Bật/Tắt bảng đặc trưng 3D (Feature HUD) | [c]: Xóa câu | [f]: Lật gương | [q]: Thoát\n")

    processed_frames = 0

    try:
        while True:
            if headless or cap is None or not cap.isOpened():
                # Chế độ headless kiểm thử: tạo khung hình giả lập nếu không có camera thật
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                time.sleep(0.033)
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            if flip_cam:
                frame = cv2.flip(frame, 1)

            h_f, w_f = frame.shape[:2]
            curr_time = time.time()
            frame_counter += 1
            processed_frames += 1

            # Tính toán FPS trung bình trượt
            loop_duration = max(curr_time - prev_time, 1e-4)
            fps = 0.9 * fps + 0.1 * (1.0 / loop_duration)
            prev_time = curr_time

            display_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            det_result = detector.detect(mp_image)

            hand_found = False
            raw_wrist_x, raw_wrist_y = 0.0, 0.0
            wrist_x, wrist_y = 0.0, 0.0
            palm_len = cached_palm_len
            hand_shape = cached_hand_shape
            n_fingers = cached_n_fingers
            hull_pts = cached_hull_pts
            valid_defects = cached_valid_defects
            seg_state = KineticState.IDLE
            completed_stroke: Optional[List[GestureFrame]] = None
            is_swipe = False
            v_norm = 0.0
            x1, y1, x2, y2 = cached_box

            if det_result.hand_landmarks and len(det_result.hand_landmarks) > 0:
                hand_lost_frames = 0
                hand_found = True
                last_hand_seen_time = curr_time
                landmarks = det_result.hand_landmarks[0]
                handedness_label = det_result.handedness[0][0].category_name if det_result.handedness else "Right"

                # 1. Điểm mốc cổ tay & Kích thước Palm Scale (Anatomical 3D Foreshortening Invariant)
                lm0 = landmarks[0]
                lm5 = landmarks[5]
                lm9 = landmarks[9]
                lm17 = landmarks[17]
                raw_wrist_x = lm0.x * w_f
                raw_wrist_y = lm0.y * h_f
                
                palm_2d = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)
                palm_3d = math.sqrt(((lm9.x - lm0.x) * w_f)**2 + ((lm9.y - lm0.y) * h_f)**2 + ((lm9.z - lm0.z) * w_f)**2)
                knuckle_span = math.hypot((lm17.x - lm5.x) * w_f, (lm17.y - lm5.y) * h_f)
                palm_len = max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)
                cached_palm_len = palm_len

                # Áp dụng R3 Layer 1 Guard: 1€ Filter smoothing trên tọa độ cổ tay
                wrist_x, wrist_y = wrist_smoother.update(raw_wrist_x, raw_wrist_y, timestamp=curr_time)

                # Cập nhật mốc tham chiếu cho LK Inertial Fallback
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lk_fallback.update_reference(gray_frame, landmarks)

                # Trích xuất 5 góc uốn ngón tay 3D & Vector pháp tuyến lòng bàn tay 3D (12D Features)
                lm_3d = np.array([[lm.x * w_f, lm.y * h_f, lm.z * w_f] for lm in landmarks], dtype=np.float32)
                current_angles = compute_joint_flexion_angles(lm_3d)
                current_normal = compute_palm_normal_vector(lm_3d, handedness=handedness_label)
                stroke_angles_history.append(current_angles)
                stroke_normals_history.append(current_normal)
                if len(stroke_angles_history) > 120:
                    stroke_angles_history.pop(0)
                if len(stroke_normals_history) > 120:
                    stroke_normals_history.pop(0)

                # 2. Nhận diện Dáng tay tức thời qua DIP Engine v4 + Two-Stream CNN + Soft Bayesian Prior
                # Kỹ thuật xen kẽ suy luận CPU: thực hiện suy luận mỗi 2 khung hình để duy trì FPS >= 25
                run_dip_cnn = (frame_counter % 2 == 0) or (cached_hand_shape == "Unknown")
                if run_dip_cnn:
                    hand_crop, (x1, y1, x2, y2), touching_border, _ = extract_smart_hand_crop(
                        frame, landmarks, handedness_label=handedness_label, auto_flip_left=False
                    )
                    cached_box = (x1, y1, x2, y2)
                    if hand_crop.shape[0] >= 30 and hand_crop.shape[1] >= 30 and not touching_border:
                        # Trích xuất hình thái học DIP: Convex Hull & Convexity Defects
                        n_fingers, hull_pts, valid_defects = compute_finger_count_from_defects(hand_crop, palm_len)
                        cached_n_fingers = n_fingers
                        cached_hull_pts = hull_pts
                        cached_valid_defects = valid_defects

                        if cnn_model is not None:
                            tensor_np, _, _ = process_lean_dip_tensor(hand_crop, target_size=(128, 128))
                            t_pt = torch.from_numpy(tensor_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                            with torch.no_grad():
                                logits = cnn_model(t_pt)
                                raw_probs = F.softmax(logits, dim=1).numpy()[0]
                                probs = apply_soft_morphological_prior(raw_probs, n_fingers, classes)
                                hand_shape = classes[int(np.argmax(probs))]
                                cached_hand_shape = hand_shape
                    else:
                        cached_hull_pts = None
                        cached_valid_defects = None
                else:
                    hand_shape = cached_hand_shape
                    x1, y1, x2, y2 = cached_box
                    n_fingers = cached_n_fingers
                    hull_pts = cached_hull_pts
                    valid_defects = cached_valid_defects

            else:
                # Khi MediaPipe không thấy tay: Thử kích hoạt LK Inertial Fallback cho 1-2 frames
                hand_lost_frames += 1
                if 1 <= hand_lost_frames <= 2:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    predicted = lk_fallback.predict_wrist(gray_frame)
                    if predicted is not None:
                        wrist_x, wrist_y = wrist_smoother.update(*predicted, timestamp=curr_time)
                        palm_len = cached_palm_len
                        hand_shape = cached_hand_shape
                        n_fingers = cached_n_fingers
                        hand_found = True
                        last_hand_seen_time = curr_time

                if hand_lost_frames >= 4:
                    segmenter.reset()
                    wrist_smoother.reset()
                    lk_fallback.reset()
                    trail_points.clear()
                    stroke_angles_history.clear()

                # Tự động chốt câu hoàn chỉnh khi hạ tay > 1.8s (Sentence Commit Timeout)
                if len(sentence_words) > 0 and (curr_time - last_hand_seen_time) > 1.8 and not sentence_finalized:
                    sentence_finalized = True
                    final_sentence = " ".join(sentence_words) + " ."
                    banner_title = "SENTENCE COMMITTED (ENTER)"
                    banner_subtitle = f"\"{final_sentence}\""
                    banner_color = (0, 255, 0)
                    banner_expire = curr_time + 2.0
                    print(f"[SentenceBuilder] ✓ ĐÃ CHỐT CÂU HOÀN CHỈNH: {final_sentence}")

            if hand_found:
                # 3. Cập nhật Bộ phân đoạn động học (Tầng 2)
                seg_state, completed_stroke, is_swipe, v_norm = segmenter.update(
                    wrist_x=wrist_x,
                    wrist_y=wrist_y,
                    palm_len=palm_len,
                    hand_shape=hand_shape,
                    timestamp=curr_time
                )

                # 4. Quản lý vệt sáng chuyển động neon (Lưu tọa độ + nhãn thời gian)
                trail_points.append((int(wrist_x), int(wrist_y), curr_time))

                # 5. Xử lý LỆNH ƯU TIÊN CẤP TỐC: SWIPE-LEFT (BACKSPACE)
                if is_swipe:
                    if len(sentence_words) > 0:
                        removed = sentence_words.pop()
                        sentence_finalized = False
                        banner_title = f"<- BACKSPACE: -{removed}"
                        banner_subtitle = "Action: Deleted last word"
                        banner_color = (0, 140, 255)
                        banner_expire = curr_time + 1.6
                        print(f"[SentenceBuilder] Swipe-Left: Đã xóa '{removed}' | Còn lại: {sentence_words}")
                    else:
                        banner_title = "<- SENTENCE IS EMPTY"
                        banner_subtitle = "Nothing to delete"
                        banner_color = (0, 200, 255)
                        banner_expire = curr_time + 1.2
                    trail_points.clear()
                    stroke_angles_history.clear()

                # 6. Xử lý CỬ CHỈ HOÀN THÀNH -> SO KHỚP GOLDEN TRIAD DTW (Tầng 3)
                if completed_stroke is not None:
                    stroke_duration = completed_stroke[-1].timestamp - completed_stroke[0].timestamp

                    # Áp dụng R3 Layer 2 Guard: Cổng thời lượng cử chỉ [0.4s, 2.5s]
                    if 0.40 <= stroke_duration <= 2.50:
                        n_stroke = len(completed_stroke)
                        if len(stroke_angles_history) >= n_stroke:
                            angles_for_stroke = stroke_angles_history[-n_stroke:]
                        else:
                            pad_count = n_stroke - len(stroke_angles_history)
                            first_ang = stroke_angles_history[0] if stroke_angles_history else np.zeros(5, dtype=np.float32)
                            angles_for_stroke = [first_ang] * pad_count + stroke_angles_history

                        if len(stroke_normals_history) >= n_stroke:
                            normals_for_stroke = stroke_normals_history[-n_stroke:]
                        else:
                            pad_norm = n_stroke - len(stroke_normals_history)
                            first_norm = stroke_normals_history[0] if stroke_normals_history else np.array([0, 0, -1], dtype=np.float32)
                            normals_for_stroke = [first_norm] * pad_norm + stroke_normals_history

                        stroke_angles_history.clear()
                        stroke_normals_history.clear()

                        # Neo gốc chuyển động tại frame đầu tiên của stroke
                        x0, y0 = completed_stroke[0].x, completed_stroke[0].y
                        ref_palm = max(completed_stroke[0].palm_len, 15.0)

                        raw_xs = np.array([(f.x - x0) / ref_palm for f in completed_stroke], dtype=np.float32)
                        raw_ys = np.array([(f.y - y0) / ref_palm for f in completed_stroke], dtype=np.float32)

                        raw_dxs = np.zeros(n_stroke, dtype=np.float32)
                        raw_dys = np.zeros(n_stroke, dtype=np.float32)
                        raw_dxs[1:] = raw_xs[1:] - raw_xs[:-1]
                        raw_dys[1:] = raw_ys[1:] - raw_ys[:-1]
                        raw_dxs[0] = raw_dxs[1] if n_stroke > 1 else 0.0
                        raw_dys[0] = raw_dys[1] if n_stroke > 1 else 0.0

                        raw_angles_mat = np.array(angles_for_stroke, dtype=np.float32)
                        raw_normals_mat = np.array(normals_for_stroke, dtype=np.float32)
                        raw_12d = np.column_stack([raw_xs, raw_ys, raw_dxs, raw_dys, raw_angles_mat, raw_normals_mat]).astype(np.float32)
                        stroke_shapes = [f.hand_shape for f in completed_stroke]

                        res_traj_12d, res_shapes = resample_9d_gesture_sequence(raw_12d, stroke_shapes, k_target=32)
                        if res_traj_12d is not None:
                            match_result = dtw_engine.match(res_traj_12d, res_shapes)

                            # a) Chấp nhận từ vựng (ACCEPTED)
                            if match_result.status == "ACCEPTED" and match_result.top_word:
                                word_key = match_result.top_word
                                en_phrase = get_asl_phrase(word_key, lang="en", default=match_result.top_display or word_key)
                                vi_phrase = get_asl_phrase(word_key, lang="vi", default="")

                                # R3 Layer 3 / Debouncing: Chống bắn lặp từ trong 0.8s
                                is_repeat = (en_phrase == last_committed_word and (curr_time - last_commit_time) < 0.8)
                                if not is_repeat:
                                    sentence_words.append(en_phrase)
                                    last_commit_time = curr_time
                                    last_committed_word = en_phrase
                                    sentence_finalized = False

                                    # Điểm tin cậy quy đổi theo công thức yêu cầu
                                    target_thresh = dtw_engine.vocabulary[word_key]["threshold"]
                                    conf_pct = max(0.0, min(100.0, (1.0 - (match_result.best_distance / (target_thresh + 1e-6))) * 100.0))

                                    # Banner phản hồi trực quan cỡ lớn giữ trong 2.0s (R4)
                                    banner_title = f"{en_phrase} ({conf_pct:.0f}%)"
                                    banner_subtitle = f"ASL Match: {en_phrase} [{vi_phrase}] | Dist: {match_result.best_distance:.3f}"
                                    banner_color = (0, 255, 100)
                                    banner_expire = curr_time + 2.0

                                    # Phát âm tiếng Anh bất đồng bộ không chặn (R4 Non-blocking TTS)
                                    speak_async(en_phrase)
                                    print(f"[SentenceBuilder] Nhận từ: '{en_phrase}' ({conf_pct:.0f}%) | Câu: {' '.join(sentence_words)}")

                            # b) Nghi vấn Top-2 (UNCERTAIN) - R3 Layer 3 Guard: Không commit từ
                            elif match_result.status == "UNCERTAIN" and len(match_result.candidates) >= 2:
                                c1, c2 = match_result.candidates[0], match_result.candidates[1]
                                en1 = get_asl_phrase(c1.word_key, lang="en", default=c1.display_name)
                                en2 = get_asl_phrase(c2.word_key, lang="en", default=c2.display_name)
                                margin_pct = ((c2.distance - c1.distance) / max(c1.distance, 1e-4)) * 100.0
                                banner_title = f"? UNCERTAIN: {en1} or {en2}"
                                banner_subtitle = f"Dist: {match_result.best_distance:.2f} | Margin: {margin_pct:.1f}%"
                                banner_color = (0, 200, 255)
                                banner_expire = curr_time + 1.4
                                word_th = float(dtw_engine.vocabulary.get(c1.word_key, {}).get("threshold", 1.25))
                                reason = "Margin Guard (2 từ quá sát nhau)" if c1.distance <= word_th else "Vượt ngưỡng chuẩn"
                                print(f"[SentenceBuilder] ❓ Nghi vấn ({reason}): {en1} (d1={c1.distance:.2f}) vs {en2} (d2={c2.distance:.2f}) | Margin: {margin_pct:.1f}%")

                            # c) Từ chối (REJECTED) - R3 Layer 3 Guard: Khoảng cách quá xa
                            else:
                                banner_title = "X UNKNOWN GESTURE"
                                banner_subtitle = f"DTW Distance: {match_result.best_distance:.2f} (Rejected)"
                                banner_color = (0, 0, 240)
                                banner_expire = curr_time + 1.0
                                print(f"[SentenceBuilder] Từ chối cử chỉ không khớp (Dist: {match_result.best_distance:.2f})")
                    else:
                        # R3 Layer 2: Cử chỉ nằm ngoài [0.4s, 2.5s] bị bỏ qua trong im lặng
                        print(f"[SentenceBuilder] Cử chỉ bị loại bởi Duration Gate ({stroke_duration:.2f}s).")

                    # LƯU Ý R4: Không xóa trail_points tại đây để vệt sáng phai tự nhiên

                # Vẽ Bounding Box & Trực quan hóa DIP (Convex Hull & Defects)
                if x2 > x1 and y2 > y1:
                    box_col = (0, 255, 0) if seg_state == KineticState.RECORDING else (255, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_col, 2)
                    cv2.putText(display_frame, f"Shape: {hand_shape} (Fingers: {n_fingers})", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1, lineType=cv2.LINE_AA)

                    if hull_pts is not None and len(hull_pts) > 0:
                        hull_display = hull_pts + np.array([x1, y1])
                        cv2.polylines(display_frame, [hull_display], True, (0, 255, 128), 1, lineType=cv2.LINE_AA)
                    if valid_defects:
                        for s_p, e_p, f_p in valid_defects:
                            cv2.circle(display_frame, (int(f_p[0] + x1), int(f_p[1] + y1)), 4, (0, 0, 255), -1)

                # Vẽ 3D Skeleton & Góc gập (Flexion) & Vector Pháp Tuyến trực tiếp lên bàn tay
                if show_features:
                    draw_hand_skeleton_features(display_frame, landmarks, current_angles, w_f, h_f, handedness=handedness_label)

            # Lọc bớt các điểm vệt sáng quá cũ (> MAX_TRAIL_AGE_SEC)
            trail_points = [pt for pt in trail_points if (curr_time - pt[2]) <= MAX_TRAIL_AGE_SEC]

            # =================================================================
            # HIỂN THỊ GIAO DIỆN ĐỒ HỌA HUD THÔNG MINH (TẦNG 4)
            # =================================================================
            # 1. Vẽ vệt sáng chuyển động neon (Fading Neon Motion Trail)
            draw_fading_neon_trail(display_frame, trail_points, curr_time, max_age_sec=MAX_TRAIL_AGE_SEC)

            # 2. Thanh trạng thái phía trên (Top Status Bar)
            cv2.rectangle(display_frame, (0, 0), (w_f, 44), (20, 20, 20), -1)
            fps_color = (0, 255, 0) if fps >= 24.0 else ((0, 200, 255) if fps >= 20.0 else (0, 0, 255))
            cv2.putText(display_frame, f"FPS: {fps:.1f} [CAM {camera_index}]", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, fps_color, 2, lineType=cv2.LINE_AA)

            # Trạng thái phân đoạn động học
            if seg_state == KineticState.RECORDING:
                k_text = "DANG CHUYEN DONG"
                k_color = (0, 255, 0)
            else:
                k_text = "KHUNG TAY" if hand_found else "CHO TAY"
                k_color = (180, 180, 180)
            cv2.putText(display_frame, f"| {k_text}", (280, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, k_color, 2, lineType=cv2.LINE_AA)

            # Thanh hiển thị vận tốc chuẩn hóa
            v_bar = int(min(1.0, v_norm / 0.15) * 80)
            cv2.putText(display_frame, f"v: {v_norm*100:.1f}%", (w_f - 160, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.rectangle(display_frame, (w_f - 95, 15), (w_f - 95 + v_bar, 28), (0, 255, 255), -1)
            cv2.rectangle(display_frame, (w_f - 95, 15), (w_f - 15, 28), (100, 100, 100), 1)

            # 3. Banner nhận diện nổi bật cỡ lớn (Prominent Banner for 2.0s)
            if len(dtw_engine.vocabulary) == 0:
                cv2.rectangle(display_frame, (10, 52), (w_f - 10, 92), (0, 0, 180), -1)
                cv2.putText(display_frame, "CHUA CO TU DIEN! HAY CHAY collect_dynamic_templates.py",
                            (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            elif curr_time < banner_expire:
                draw_prominent_hud_banner(display_frame, banner_title, banner_subtitle, banner_color, h_f, w_f)

            # 4. Bảng biểu diễn trực quan đặc trưng 3D (Golden Triad Features HUD) khi bấm 'd'
            if show_features and hand_found and current_angles is not None:
                vel_vec = (raw_wrist_x - wrist_smoother.prev_x, raw_wrist_y - wrist_smoother.prev_y) if wrist_smoother.prev_x is not None else (0.0, 0.0)
                draw_feature_metrics_hud(
                    display_frame,
                    angles=current_angles,
                    palm_len=palm_len,
                    v_norm=v_norm,
                    wrist_pt=(int(wrist_x), int(wrist_y)),
                    vel_vector=vel_vec,
                    palm_normal=current_normal,
                    x_offset=15,
                    y_offset=h_f - 265
                )

            # 5. Hộp phụ đề câu thoại hoàn chỉnh ở đáy màn hình (Bottom Subtitle Box)
            cv2.rectangle(display_frame, (0, h_f - 75), (w_f, h_f - 24), (12, 12, 12), -1)
            border_box_col = (0, 255, 0) if sentence_finalized else (0, 200, 255)
            cv2.rectangle(display_frame, (0, h_f - 75), (w_f, h_f - 24), border_box_col, 2)

            if len(sentence_words) > 0:
                sentence_str = " ".join(sentence_words)
                if sentence_finalized:
                    sentence_str += " ."
                sentence_display = f"ASL SENTENCE: \"{sentence_str}\""
            else:
                sentence_display = "ASL SENTENCE: (Awaiting first gesture...)"

            text_col = (0, 255, 0) if sentence_finalized else (255, 255, 255)
            font_scale = 0.70
            (t_w, _), _ = cv2.getTextSize(sentence_display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            max_allowed_w = w_f - 30
            if t_w > max_allowed_w and t_w > 0:
                font_scale = max(0.40, font_scale * (max_allowed_w / t_w))
            cv2.putText(display_frame, sentence_display, (15, h_f - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, 2, lineType=cv2.LINE_AA)

            # 6. Hướng dẫn phím tắt chân trang
            cv2.putText(display_frame,
                        "[d]: Toggle Features HUD | [Swipe-Left]: Backspace | [Hand-Down]: Commit | [c]: Clear | [q]: Quit",
                        (10, h_f - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, lineType=cv2.LINE_AA)

            if not headless:
                cv2.imshow("ASL Dynamic Gesture Recognition & Sentence Builder", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in [ord('d'), ord('D')]:
                    show_features = not show_features
                    print(f"[HUD] {'BẬT' if show_features else 'TẮT'} bảng hiển thị đặc trưng 3D.")
                elif key in [ord('c'), ord('C')]:
                    sentence_words.clear()
                    sentence_finalized = False
                    trail_points.clear()
                    banner_title = "SENTENCE CLEARED"
                    banner_subtitle = "Word buffer emptied"
                    banner_color = (0, 200, 255)
                    banner_expire = curr_time + 1.5
                    print("[SentenceBuilder] Đã xóa toàn bộ câu thoại.")
                elif key == ord('f'):
                    flip_cam = not flip_cam
                elif key in [ord('p'), ord('P')]:
                    if cap is not None:
                        cap.set(cv2.CAP_PROP_SETTINGS, 1)

            if max_frames is not None and processed_frames >= max_frames:
                break

    finally:
        if cap is not None:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()
        detector.close()
        print("-> Đã đóng chương trình nhận diện ghép câu an toàn.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition")
    parser.add_argument("--cam", type=int, default=None, help="Camera index (0=built-in webcam, 1, 2=external/iPhone). Mặc định dùng camera_source từ CONFIG.")
    parser.add_argument("--margin", type=float, default=0.12, help="Relative contrast margin threshold (default: 0.12 = 12%)")
    parser.add_argument("--thresh", type=float, default=None, help="Override DTW distance threshold (default: None, uses template threshold 1.25)")
    args, _ = parser.parse_known_args()
    
    selected_cam = args.cam if args.cam is not None else CONFIG.get("camera_source", 0)
    run_sentence_builder(camera_index=selected_cam, margin_threshold=args.margin, threshold_override=args.thresh)
