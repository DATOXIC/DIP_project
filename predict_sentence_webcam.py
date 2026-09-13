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
from dynamic_dtw_engine import (
    KineticState,
    GestureFrame,
    KineticPauseSegmenter,
    ResampledGesture,
    resample_gesture_sequence,
    HybridDTWEngine,
    HybridDTWConfig,
    MatchResult
)

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
        now = timestamp if timestamp is not None else (
            (self._prev_ts + 1.0 / 30.0) if self._prev_ts is not None else 0.0
        )
        dt = (now - self._prev_ts) if self._prev_ts is not None else (1.0 / 30.0)
        dt = max(dt, 1e-6)

        if self.prev_x is None or self.prev_y is None:
            self.prev_x = float(raw_x)
            self.prev_y = float(raw_y)
            self._prev_ts = now
            return self.prev_x, self.prev_y

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
        self._prev_ts = now
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

    for i in range(defects.shape[0]):
        start_idx, end_idx, far_idx, depth_raw = defects[i, 0]
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


# =============================================================================
# 5. CHƯƠNG TRÌNH ĐIỀU PHỐI GHÉP CÂU HOÀN CHỈNH (MAIN PIPELINE RUNNER)
# =============================================================================

def run_sentence_builder(
    camera_index: int = 0,
    headless: bool = False,
    max_frames: Optional[int] = None
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
    cfg = HybridDTWConfig(
        alpha=1.0,
        beta=0.5,
        lambda_shape=0.6,
        k_resample=32,
        sakoe_chiba_radius=4,
        default_threshold=1.25,
        uncertainty_ratio=1.40
    )
    dtw_engine = HybridDTWEngine(templates_path=templates_file, config=cfg)
    if not dtw_engine.vocabulary:
        print("[!] CẢNH BÁO: dynamic_templates.json chưa có mẫu nào hoặc không tồn tại!")
        print("[!] Vui lòng chạy 'collect_dynamic_templates.py' trước để thu mẫu cử chỉ.")
    else:
        print(f"✓ Đã nạp {len(dtw_engine.vocabulary)} từ vựng cử chỉ mẫu từ {templates_file}.")

    segmenter = KineticPauseSegmenter(
        v_start_ratio=0.08,
        v_stop_ratio=0.035,
        pause_frames=11,
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

    # Mở Webcam DirectShow
    cap = None
    if not headless:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    flip_cam = False
    prev_time = time.time()
    fps = 0.0

    print("\n[HƯỚNG DẪN SỬ DỤNG]")
    print("  - Múa cử chỉ rõ ràng, khựng tay dừng dứt khoát 0.35s để chốt từ.")
    print("  - Gạt nhanh tay sang trái (Swipe-Left) để XÓA TỪ gần nhất (Backspace).")
    print("  - Hạ tay xuống khỏi màn hình > 1.8s để HOÀN THÀNH CÂU (Commit/Enter).")
    print("  - Hệ thống tự động phát âm tiếng Anh (ASL) bằng giọng nói pyttsx3.")
    print("  [c]: Xóa toàn bộ câu | [f]: Lật gương camera | [p]: Cài đặt Cam | [q]: Thoát\n")

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

                # 1. Điểm mốc cổ tay & Kích thước Palm Scale (LM 0 - LM 9)
                lm0 = landmarks[0]
                lm9 = landmarks[9]
                raw_wrist_x = lm0.x * w_f
                raw_wrist_y = lm0.y * h_f
                palm_len = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)
                cached_palm_len = palm_len

                # Áp dụng R3 Layer 1 Guard: 1€ Filter smoothing trên tọa độ cổ tay
                wrist_x, wrist_y = wrist_smoother.update(raw_wrist_x, raw_wrist_y, timestamp=curr_time)

                # Cập nhật mốc tham chiếu cho LK Inertial Fallback
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lk_fallback.update_reference(gray_frame, landmarks)

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

                # 6. Xử lý CỬ CHỈ HOÀN THÀNH -> SO KHỚP HYBRID DTW (Tầng 3)
                if completed_stroke is not None:
                    stroke_duration = completed_stroke[-1].timestamp - completed_stroke[0].timestamp

                    # Áp dụng R3 Layer 2 Guard: Cổng thời lượng cử chỉ [0.4s, 2.5s]
                    if 0.40 <= stroke_duration <= 2.50:
                        resampled = resample_gesture_sequence(completed_stroke, k_target=32)
                        if resampled is not None:
                            match_result = dtw_engine.match(resampled)

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
                                banner_title = f"? UNCERTAIN: {en1} or {en2}"
                                banner_subtitle = f"Dist: {match_result.best_distance:.2f} (Threshold exceeded, not committed)"
                                banner_color = (0, 200, 255)
                                banner_expire = curr_time + 1.4
                                print(f"[SentenceBuilder] Nghi vấn: {en1} ({c1.distance:.2f}) vs {en2} ({c2.distance:.2f})")

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
            cv2.putText(display_frame, f"FPS: {fps:.1f} (ASL HYBRID DTW)", (10, 28),
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

            # 4. Hộp phụ đề câu thoại hoàn chỉnh ở đáy màn hình (Bottom Subtitle Box)
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

            # 5. Hướng dẫn phím tắt chân trang
            cv2.putText(display_frame,
                        "[Swipe-Left]: Backspace | [Hand-Down]: Commit | [c]: Clear | [f]: Flip | [q]: Quit",
                        (10, h_f - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, lineType=cv2.LINE_AA)

            if not headless:
                cv2.imshow("ASL Dynamic Gesture Recognition & Sentence Builder", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
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
    run_sentence_builder()
