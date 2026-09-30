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
  3. TẦNG 3: Bộ So Khớp GoldenTriadDTWMatcher (Golden Triad 12D) với dải băng Sakoe-Chiba.
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
  [l] : Đổi ngôn ngữ giao diện + giọng nói Anh <-> Việt
  [u] : Chuyển giao diện demo <-> gỡ lỗi (debug)
  [t] : Bật/tắt loa
  [h] : Bật/tắt bảng trợ giúp
  [q] : Thoát chương trình
"""

import os
import sys
import time
import math
import threading
from collections import deque
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

import dl_fusion
from train import TwoStreamHandCNN
from improved_dip_pipeline import (
    extract_smart_hand_crop,
    process_lean_dip_tensor
)
import json
# SPEC 03: các hàm trích xuất 4 nhóm đặc trưng mới (z_norm, mốc cơ thể, độ xòe ngón) dùng CHUNG với
# collect_dynamic_templates.py / inspect_and_trim_asl_citizen.py / rebuild_templates_from_raw.py.
from dynamic_features import (
    CachedFaceDetector,
    compute_extended_columns,
    append_extended_columns,
    compute_palm_len,
    build_segment_features,
    extend_still_tail,
    records_to_gesture_frames,
)
from dynamic_dtw_engine import (
    KineticState,
    GestureFrame,
    KineticPauseSegmenter,
    ResampledGesture,
    resample_gesture_sequence,
    HybridDTWEngine,
    HybridDTWConfig,
    MatchResult,
    MatchCandidate,
    resample_9d_gesture_sequence,
    compute_golden_triad_dtw_distance,
    GoldenTriadDTWMatcher,
    canonicalize_chirality,
    STROKE_DURATION_MIN_SEC,
    STROKE_DURATION_MAX_SEC,
    is_stroke_duration_valid,
    is_movement_epenthesis
)
# Giao diện demo song ngữ (vẽ chữ Unicode) & dịch vụ giọng nói theo ngôn ngữ
import speech_service
from ui_render import DemoHUD, HudState, HudToast, draw_soft_trail
from camera_health import CameraHealthMonitor
from lead_trim import select_best_lead_trim
from session_enroll import SessionEnroller

# =============================================================================
# 0. BẢNG CẤU HÌNH TRUNG TÂM (CENTRAL CONFIGURATION)
# Người dùng có thể tùy chỉnh mọi thông số tại đây mà không cần tìm trong code!
# =============================================================================
CONFIG = {
    # --- Chọn nguồn Camera: 0 = Webcam máy tính, 1 hoặc 2 = iPhone / Cam ngoài ---
    "camera_source": 1,          # Iriun Webcam (iPhone) = 0 trên máy này; webcam laptop = 1
                                 # (đã xác nhận qua snapshot: index 0 → iPhone, index 1 → laptop webcam)

    # --- Trọng số thuật toán DTW (Golden Triad 24D - Calibrated SPEC 04A.1) ---
    "w_wrist_pos": 0.6,          # Trọng số tọa độ vị trí cổ tay (d_pos)
    "w_finger_angles": 1.0,      # Trọng số góc gập 5 ngón tay (d_angle)
    "w_wrist_dir": 0.4,          # Trọng số hướng vận tốc tiếp tuyến (d_dir)
    "w_palm_normal": 0.8,        # Trọng số vector pháp tuyến lòng bàn tay 3D (d_normal - Cosine)
    "lambda_shape": 0.0,         # Tắt: audit c_shape cho thấy 4/5 từ có độ nhất quán nhãn CNN quá thấp (30-49%) trong lúc chuyển động, tín hiệu nhiễu nhiều hơn hữu ích
    "asllex_blend_alpha": 1.0,        # 1.0 = giữ hành vi hiện tại (dùng thẳng ASL-LEX). Đổi xuống 0.3-0.7 để bật Hybrid Blend thật sự.
    "asllex_max_deviation": None,     # None = không giới hạn (giữ hành vi hiện tại). Đặt vd 0.5 để giới hạn ASL-LEX lệch tối đa ±50% so với mặc định.
    
    # --- Ngưỡng quyết định nhận diện & Chế độ độ khớp (SPEC 04B) ---
    "threshold_mode": "scale_k",   # "scale_k" (ngưỡng theo độ rộng nội lớp) hoặc "legacy" (cố định)
    "threshold_k": 1.9,            # Hệ số k cho ngưỡng chấp nhận (k * độ rộng). Chốt 2026-09-28 cùng width_blend=1.0 (cấu hình G1p90): người lạ Mai nhận 61%->73% (p=0.016), từ lạ nhận nhầm 3/175->0/175. (Trước đó: scale_k k=1.75, width_blend=0.0 — cấu hình C11_base.)
    "uncertainty_k": 2.5,          # Hệ số k cho ranh giới từ chối (= k + 0.6)
    "width_blend": 1.0,            # 1.0 = thang độ rộng chung (trung vị các từ) — mặc định từ 2026-09-28. 0.0 = mỗi từ dùng độ rộng riêng (hành vi cũ, phải đặt lại k=1.75/uncertainty_k=2.35). Xem docs/BAO_CAO_TANG_CUONG_VA_TU_CHOI.md
    "default_threshold": 0.40,   # Ngưỡng chấp nhận chuẩn hóa SPEC 04A.1 (cho chế độ legacy)
    "uncertainty_ratio": 1.40,   # Vùng nghi vấn: từ threshold đến 1.4 * threshold (legacy)
    "margin_threshold": 0.12,    # Độ chênh lệch an toàn giữa Top 1 và Top 2 (12%)
    
    # --- Phân đoạn động học (Kinetic Pause Segmenter) ---
    "v_start_ratio": 0.08,       # Vận tốc bắt đầu cử chỉ (8% palm_len/frame)
    "v_stop_ratio": 0.040,       # Vận tốc khựng tay dừng từ (4% palm_len/frame)
    "pause_frames": 9,           # Số frame khựng liên tục để chốt cử chỉ (~0.3s)
    # Chuẩn hóa vận tốc + Swipe-Left theo THỜI GIAN (mặc định BẬT từ 2026-09-29). Giúp khi camera có fps thật lệch 30
    # (thiếu sáng, 60 fps) -- quan trọng cho người lạ vì ta không kiểm soát được fps camera của họ.
    # Số đo ngoại tuyến: docs/BAO_CAO_TANG_CUONG_VA_TU_CHOI.md (eval_segmenter_fps.py) -- ở fps gốc (30) hành vi
    # giống hệt lúc tắt; ở 60fps/15fps/45fps/khung-trùng đều cải thiện rõ (nhớ giảm, swipe-nhầm giảm về 0).
    "time_norm": {"enabled": True, "ref_fps": 30.0, "max_stroke_sec": 2.0, "swipe_window_sec": 0.17, "v_scale_min": 1.0},
    "dedupe_frames": False,      # True = bỏ khung TRÙNG y hệt (camera thiếu sáng lặp bộ đệm) trước khi dò tay; cần camera thật (video/headless bỏ qua). Chưa kiểm chứng trên webcam thật.

    # --- Đường dẫn template và trọng số DTW theo từng ký hiệu (ASL-LEX 2.0) ---
    "templates_path": "webcam_templates.json",
    "sign_weights_path": "",     # Tắt ASL-LEX (chuỗi rỗng kích hoạt Global Weights theo SPEC 04A.1)

    # --- Tối ưu hóa hiệu năng (SPEC 02 & SPEC 05C - Chống lag) ---
    "enable_static_shape_cnn": False,   # Tắt DIP + CNN trong vòng lặp động (lambda_shape=0.0 nên không dùng)
    "detect_scale": 1.0,                 # SPEC 05C: Trả về 1.0 vì MediaPipe không giảm ms khi resize
    "detector_workers": None,           # SPEC 05C: Số worker song song (None = tự chọn theo CPU)

    # --- SPEC 03: 4 nhóm đặc trưng MỚI (17D -> 24D + số nhịp lặp). 0.0 = TẮT ---
    "w_z": 0.0,                          # z_norm (độ sâu ảo từ palm_len)
    "w_b": 0.8,                          # mốc cơ thể (cổ tay so với hộp mặt) — BẬT để chống cử chỉ rác
    "w_s": 0.0,                          # độ xòe ngón (4 góc giữa các ngón liền kề)
    "w_r": 0.0,                          # số nhịp lặp (cộng 1 lần sau DTW)
    "face_detect_every_n_frames": 8,     # bộ dò mặt có cache: chỉ chạy mạng mỗi N frame

    # --- Giao diện & ngôn ngữ (chỉ ảnh hưởng hiển thị / giọng nói, KHÔNG ảnh hưởng nhận dạng) ---
    "ui_mode": "demo",                   # "demo" = giao diện sạch cho người dùng cuối | "debug" = bảng đặc trưng cũ
    "ui_lang": "en",                     # "en" | "vi" (mặc định "en" vì máy chưa có giọng đọc tiếng Việt)

    # --- SPEC 08 "KÝ NHANH" + SPEC 08B "CẮT CHUYỂN TIẾP": bỏ thời gian chờ thừa giữa hai từ và loại phần tay di chuyển sang từ kế tiếp.
    #     MẶC ĐỊNH BẬT (bản chuẩn từ 2026-09-21, sau khi thử trực tiếp: chuyển tiếp không còn bị ghi thành cử chỉ lạ).
    #     Quay về hành vi cũ (phải dừng ~0.55 s giữa hai từ): --no-fast-chain, hoặc phím [j] lúc chạy. Tắt riêng cắt dẫn đầu: --no-lead-trim / phím [k].
    #     Giá trị chain_window_sec=0.0 và ready_stable_sec=0.1 là cấu hình ĐÃ ĐƯỢC THỬ TRỰC TIẾP; đừng đổi khi chưa thử lại.
    #     Mọi thời gian tính bằng GIÂY (quy đổi theo dấu thời gian thật). ---
    "fast_chain": {
        "enabled": True,
        "chain_window_sec": 0.0,         # sau khi chốt một từ, trong ngần này giây tay coi như đã sẵn sàng: cử động là ghi ngay
        "pause_sec": 0.12,               # tay khựng ngần này giây thì vào PENDING_COMMIT (cũ: 9 khung ~0.30 s). Chọn từ lưới mô phỏng SPEC 08
        "grace_window_sec": 0.05,        # cửa sổ chờ gộp pha (cũ: 0.15 s)
        "min_stroke_sec": 0.10,          # stroke tối thiểu (cũ: 8 khung); đuôi ngắn hơn nên không thể đòi 8 khung
        "early_commit": True,            # chốt sớm ngay khi dự đoán tạm đủ chắc (không đợi hết grace)
        "early_min_quality": 70,         # match_quality tối thiểu để chốt sớm (ngưỡng nhận diện KHÔNG đổi: vẫn phải ACCEPTED)
        "early_min_margin": 0.25,        # chênh lệch tương đối top-1/top-2 tối thiểu để chốt sớm
        "multi_phase_words": None,       # None = tự suy từ bank lúc khởi động; hoặc danh sách từ (không chốt sớm các từ này)
        "tail_pad_sec": 0.45,            # kéo dài đuôi đứng yên của stroke về ~0.45 s (như bank) trước khi dựng đặc trưng; None = tắt
        "ready_stable_sec": 0.1,
        "transition_lockout_sec": 0.0,   # SPEC 08B: khoảng cấm sau chốt từ (0 = tắt)
        "lead_trim": {                   # SPEC 08B: cắt phần dẫn đầu thích nghi (mặc định BẬT cùng chế độ ký nhanh)
            "enabled": True,
            "max_sec": 0.45,
            "max_variants": 6,
            "good_r": 0.80,
            "trim_penalty_per_sec": 0.30,
            "min_gain": 0.10,
            "min_gain_other_word": 0.25,
            "min_after_trim_sec": 0.30,
            "max_trim_fraction": 0.40,
        },
    },

    

    # --- Sức khỏe camera (camera_health.py): đo khung MỚI THẬT/giây + độ sáng, cảnh báo trên HUD. CHỈ đo/hiển thị. ---
    "camera_health": {
        "enabled": True,
        "window_sec": 2.0,               # cửa sổ trượt tính khung mới thật/giây
        "unique_fps_warn": 12.0,         # khung mới/giây (lúc tay chuyển động) dưới mức này -> "Camera đang chậm"
        "unique_fps_bad": 8.0,           # dưới mức này -> cảnh báo đậm
        "luma_low_warn": 95.0,           # độ sáng trung bình dưới mức này -> "Thiếu sáng"
        "luma_low_bad": 70.0,            # dưới mức này -> cảnh báo đậm
        "luma_high": 215.0,              # độ sáng trung bình trên mức này -> "Quá chói"
        "sat_frac_high": 0.25,           # tỉ lệ điểm ảnh >= 250 trên mức này -> "Quá chói"
        "persist_sec": 2.0,              # điều kiện phải liên tục ngần này giây mới bật cảnh báo
        "show_min_sec": 3.0,             # cảnh báo đã bật hiện tối thiểu ngần này giây
        "clear_sec": 2.0,                # phải bình thường liên tục ngần này giây mới tắt
        "hysteresis": 0.10,              # ngưỡng thoát nới ~10% so với ngưỡng vào
        "hand_moving_v": 0.03,           # v_norm >= mức này coi là tay đang chuyển động
        "warn_on_video": False,          # chạy từ file video: tắt cảnh báo (khung trùng của video là bình thường)
    },
}

# =============================================================================
# 1. BẢNG TỪ ĐIỂN TỪ KHÓA & ÁNH XẠ TIẾNG ANH (ASL VOCABULARY MAPPING)
# =============================================================================

ASL_VOCAB_MAP: Dict[str, Dict[str, str]] = {
    # 15 Canonical Sign Keys chuẩn hóa từ ASL-LEX 2.0
    "HELLO":     {"en": "Hello",     "vi": "Xin chào"},
    "BYE":       {"en": "Bye",       "vi": "Tạm biệt"},
    "ME":        {"en": "Me",        "vi": "Tôi"},
    "YOU":       {"en": "You",       "vi": "Bạn"},
    "THANK_YOU": {"en": "Thank You", "vi": "Cảm ơn"},
    "GO":        {"en": "Go",        "vi": "Đi"},
    "COME":      {"en": "Come",      "vi": "Đến"},
    "WANT":      {"en": "Want",      "vi": "Muốn"},
    "MORE":      {"en": "More",      "vi": "Thêm"},
    "WATER":     {"en": "Water",     "vi": "Nước"},
    "HELP":      {"en": "Help",      "vi": "Giúp"},
    "GOOD":      {"en": "Good",      "vi": "Tốt"},
    "NAME":      {"en": "Name",      "vi": "Tên"},
    "DRINK":     {"en": "Drink",     "vi": "Uống"},
    "STOP":      {"en": "Stop",      "vi": "Dừng"},
    # Các từ 1 tay mở rộng
    "BAD":       {"en": "Bad",       "vi": "Tệ"},
    "EAT":       {"en": "Eat",       "vi": "Ăn"},
    "FATHER":    {"en": "Father",    "vi": "Bố"},
    "LIKE":      {"en": "Like",      "vi": "Thích"},
    "MOTHER":    {"en": "Mother",    "vi": "Mẹ"},
    "PLEASE":    {"en": "Please",    "vi": "Làm ơn"},
    "SEE":       {"en": "See",       "vi": "Thấy"},
    "SORRY":     {"en": "Sorry",     "vi": "Xin lỗi"},
    "TOMORROW":  {"en": "Tomorrow",  "vi": "Ngày mai"},
    "WHERE":     {"en": "Where",     "vi": "Ở đâu"},
    "YES":       {"en": "Yes",       "vi": "Vâng"},
    "HUNGRY":    {"en": "Hungry",    "vi": "Đói"},
    # Alias tương thích ngược cho template tiếng Việt cũ nếu có
    "xin_chao":  {"en": "Hello",     "vi": "Xin chào"},
    "cam_on":    {"en": "Thank You", "vi": "Cảm ơn"},
    "toi":       {"en": "Me",        "vi": "Tôi"},
    "ban":       {"en": "You",       "vi": "Bạn"},
    "tam_biet":  {"en": "Bye",       "vi": "Tạm biệt"},
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

# Tái xuất từ pipeline_common (nguồn sự thật duy nhất) để giữ tương thích ngược 100%
from pipeline_common import (
    WristCoordinateSmoother,
    LKInertialFallback,
    make_wrist_smoother,
    make_segmenter,
    find_multi_phase_words,
    early_commit_ok,
    open_camera,
    MIN_COLLECT_FPS,
    SlidingFPSMeter,
    ParallelHandDetector,
    get_default_detector_workers,
)


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


def _make_toast(kind: str, keys: List[str], quality: Optional[float], now: float, duration: float) -> HudToast:
    """Tạo sự kiện phản hồi có cấu trúc cho giao diện demo (kind, khóa từ, độ khớp, thời gian bắt đầu/hết hạn)."""
    return HudToast(kind=kind, keys=list(keys), quality=quality, t_start=now, t_expire=now + duration)


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


def draw_face_anchor_box(
    frame: np.ndarray,
    face_box: Optional[Tuple[float, float, float, float]],
    wrist_pt: Optional[Tuple[int, int]] = None,
    draw_vector: bool = False
) -> None:
    """
    Vẽ bounding box và tâm mốc cơ thể của khuôn mặt lên frame OpenCV.
    - face_box: (cx, cy, w, h) tính bằng pixel khung gốc.
    - wrist_pt: (wx, wy) tọa độ cổ tay (tùy chọn).
    - draw_vector: nếu True, vẽ đường nối từ tâm mặt đến cổ tay (vector mốc cơ thể).
    """
    if face_box is None:
        return
    h_f, w_f = frame.shape[:2]
    cx, cy, w, h = face_box
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_f - 1, x2), min(h_f - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    # Khung chữ nhật bo góc nhấn màu cyan mềm (255, 200, 0)
    box_col = (255, 200, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 1)

    corner_len = max(8, int(min(w, h) * 0.15))
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_col, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_col, 2)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_col, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_col, 2)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_col, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_col, 2)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_col, 2)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_col, 2)

    # Tâm khuôn mặt (Body Anchor Origin)
    icx, icy = int(round(cx)), int(round(cy))
    cv2.circle(frame, (icx, icy), 4, (0, 255, 255), -1)

    # Nhãn text tinh tế phía trên hộp
    cv2.putText(frame, "MAT (BODY ANCHOR)", (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_col, 1, lineType=cv2.LINE_AA)

    # Đường vector nối từ tâm mặt đến cổ tay khi bật chế độ xem đặc trưng
    if draw_vector and wrist_pt is not None:
        wx, wy = wrist_pt
        cv2.line(frame, (icx, icy), (wx, wy), (0, 180, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (wx, wy), 3, (0, 255, 0), -1)


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


# resample_9d_gesture_sequence, compute_golden_triad_dtw_distance và GoldenTriadDTWMatcher
# đã được dời sang dynamic_dtw_engine.py (Mục 6: "BỘ NÃO GOLDEN TRIAD 12D") và được
# import lại ở đầu file này — xem khối `from dynamic_dtw_engine import (...)` phía trên.


# =============================================================================
# 6. CHƯƠNG TRÌNH ĐIỀU PHỐI GHÉP CÂU HOÀN CHỈNH (MAIN PIPELINE RUNNER)
# =============================================================================

def _history_tail(history: List[Any], n: int, default_first: Any) -> List[Any]:
    """
    Lấy n phần tử CUỐI của một lịch sử theo khung; nếu lịch sử ngắn hơn n thì đệm ĐẦU bằng phần tử đầu tiên
    của lịch sử (hoặc `default_first` khi lịch sử rỗng). Đúng hành vi đoạn code inline cũ (bit-giống hệt).
    """
    if len(history) >= n:
        return history[-n:]
    pad = n - len(history)
    first = history[0] if history else default_first
    return [first] * pad + history


def _segment_records(stroke_frames: List[GestureFrame], angle_hist: List[Any], normal_hist: List[Any],
                     ext_hist: List[Any]) -> List[Dict[str, Any]]:
    """
    Ghép các khung của một đoạn (GestureFrame) với lịch sử góc / pháp tuyến / (landmark, hộp mặt) cùng độ dài
    thành danh sách bản ghi cho `dynamic_features.build_segment_features`. Dùng chung cho nhánh chốt thật
    và nhánh dự đoán tạm.
    """
    n = len(stroke_frames)
    angs = _history_tail(angle_hist, n, np.zeros(5, dtype=np.float32))
    norms = _history_tail(normal_hist, n, np.array([0, 0, -1], dtype=np.float32))
    exts = _history_tail(ext_hist, n, (None, None))
    return [
        {
            "x": f.x, "y": f.y, "palm_len": f.palm_len, "hand_shape": f.hand_shape, "t": f.timestamp,
            "angles": angs[i], "normal": norms[i], "landmarks": exts[i][0], "face_box": exts[i][1],
        }
        for i, f in enumerate(stroke_frames)
    ]


def run_sentence_builder(
    camera_index: int = 0,
    headless: bool = False,
    max_frames: Optional[int] = None,
    margin_threshold: float = 0.12,
    threshold_override: Optional[float] = None,
    templates_path: Optional[str] = None,
    sign_weights_path: Any = "KEEP_DEFAULT",
    video_path: Optional[str] = None,
    profile: bool = False,
    enable_static_shape_cnn: Optional[bool] = None,
    detect_scale: Optional[float] = None,
    show_features: Optional[bool] = None,
    log_session_path: Optional[str] = None,
    detector_workers: Optional[int] = None,
    threshold_mode: Optional[str] = None,
    threshold_k: Optional[float] = None,
    uncertainty_k: Optional[float] = None,
    ui_mode: Optional[str] = None,
    ui_lang: Optional[str] = None,
    fast_chain: Optional[bool] = None,
    transition_lockout: Optional[float] = None,
    lead_trim: Optional[bool] = None,
    width_blend: Optional[float] = None,
    dedupe_frames: Optional[bool] = None,
    confirm_enroll: bool = True,
    dl_model_path: Optional[str] = None,
    dl_conf_threshold: Optional[float] = None,
):
    """
    Vòng lặp chính xử lý nhận diện cử chỉ động & ghép câu thời gian thực (ASL/VSL).
    ui_mode: "demo" | "debug" (mặc định CONFIG["ui_mode"]); ui_lang: "en" | "vi" (mặc định CONFIG["ui_lang"]).
    fast_chain: SPEC 08 chế độ "ký nhanh" (None = lấy CONFIG["fast_chain"]["enabled"], mặc định BẬT); phím [j] bật/tắt lúc chạy.
    transition_lockout: SPEC 08B khoảng cấm sau chốt (giây). None = lấy từ cấu hình.
    lead_trim: SPEC 08B cắt phần dẫn đầu thích nghi (None = lấy từ cấu hình); phím [k] bật/tắt lúc chạy.
    confirm_enroll: chế độ HỎI XÁC NHẬN (session_enroll.py): cử chỉ UNCERTAIN/REJECTED được hỏi lại; phím [y]/[2]/[n]/[z].
    dl_model_path: đường dẫn checkpoint .pth của tầng DL dự phòng (xem dl_fusion.py). None (mặc định) =
                   TẮT, hành vi giữ nguyên như trước khi có tầng này. CHỈ dùng checkpoint train với
                   --test_signer none (train_dynamic_dl.py) - các checkpoint LOSO (best_gesture_model_
                   <Ten>.pth) CỐ Ý giấu 1 người, không phải model để chạy thật.
    dl_conf_threshold: ngưỡng tin cậy softmax tối thiểu để tầng DL được nhận (None = dl_fusion.DEFAULT_CONF_THRESHOLD).
    """
    print("=" * 75)
    print("  HỆ THỐNG NHẬN DIỆN CỬ CHỈ ĐỘNG & GHÉP CÂU ASL (REAL-TIME SENTENCE BUILDER)")
    print("=" * 75)

    if detector_workers is None:
        detector_workers = CONFIG.get("detector_workers", None)

    th_mode = threshold_mode if threshold_mode is not None else CONFIG.get("threshold_mode", "scale_k")
    th_k = threshold_k if threshold_k is not None else float(CONFIG.get("threshold_k", 1.6))
    unc_k = uncertainty_k if uncertainty_k is not None else float(CONFIG.get("uncertainty_k", 2.2))
    w_blend = width_blend if width_blend is not None else float(CONFIG.get("width_blend", 0.0))

    if threshold_override is not None and th_mode == "scale_k":
        raise ValueError(
            "Không thể dùng đồng thời --thresh (khoảng cách tuyệt đối) và --threshold-mode scale_k. "
            "Hãy dùng --threshold-k hoặc đổi sang --threshold-mode legacy."
        )

    # Khởi tạo CSV session logger nếu có chỉ định log_session_path
    session_file = None
    session_writer = None
    if log_session_path is not None:
        import csv
        log_dir = os.path.dirname(os.path.abspath(log_session_path))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_is_new = (not os.path.exists(log_session_path)) or (os.path.getsize(log_session_path) == 0)
        session_file = open(log_session_path, mode="a", newline="", encoding="utf-8")
        session_writer = csv.writer(session_file)
        if file_is_new:
            session_writer.writerow([
                "timestamp", "event", "predicted_word", "best_distance",
                "second_word", "second_distance", "margin_pct",
                "duration_sec", "num_frames", "body_anchor_valid", "fps",
                "best_confidence", "match_quality", "r_ratio", "t_unix", "commit_mode", "t_start",
                "lead_trim_sec", "lead_trim_tried"
            ])
            session_file.flush()
        print(f"✓ Đang ghi log phiên cử chỉ vào: '{log_session_path}'")

    # 1. Khởi tạo MediaPipe Hand Landmarker song song đa luồng (SPEC 05C)
    model_asset_path = 'hand_landmarker.task'
    if not os.path.exists(model_asset_path):
        print(f"[-] Đang tải model {model_asset_path}...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_asset_path)
        print("✓ Tải xong bộ dò bàn tay!")

    detector_pool = ParallelHandDetector(
        model_path=model_asset_path,
        workers=detector_workers,
        num_hands=1
    )
    print(f"✓ Đã nạp ParallelHandDetector ({detector_pool.num_workers} workers, max_in_flight={detector_pool.max_in_flight}).")

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
    templates_file = templates_path if templates_path is not None else CONFIG.get("templates_path", "dynamic_templates.json")
    sw_file = CONFIG.get("sign_weights_path") if sign_weights_path == "KEEP_DEFAULT" else sign_weights_path
    if sw_file:
        print(f"ℹ️ [DTW] Chế độ trọng số: ASL-LEX theo từ (Per-Sign Weights từ '{sw_file}').")
    else:
        print("ℹ️ [DTW] Chế độ trọng số: Toàn cục (Global Weights từ CONFIG).")

    eff_thresh_override = None if th_mode == "scale_k" else (threshold_override if threshold_override is not None else CONFIG.get("default_threshold", 0.40))
    dtw_engine = GoldenTriadDTWMatcher(
        templates_path=templates_file,
        sign_weights_path=sw_file,
        w_p=CONFIG["w_wrist_pos"],
        w_a=CONFIG["w_finger_angles"],
        w_m=CONFIG["w_wrist_dir"],
        w_n=CONFIG["w_palm_normal"],
        lambda_shape=CONFIG["lambda_shape"],
        margin_threshold=margin_threshold if margin_threshold is not None else CONFIG["margin_threshold"],
        uncertainty_ratio=CONFIG["uncertainty_ratio"],
        threshold_override=eff_thresh_override,
        asllex_blend_alpha=CONFIG.get("asllex_blend_alpha", 1.0),
        asllex_max_deviation=CONFIG.get("asllex_max_deviation", None),
        w_z=CONFIG.get("w_z", 0.0),
        w_b=CONFIG.get("w_b", 0.0),
        w_s=CONFIG.get("w_s", 0.0),
        w_r=CONFIG.get("w_r", 0.0),
        threshold_mode=th_mode,
        threshold_k=th_k,
        uncertainty_k=unc_k,
        width_blend=w_blend,
        dl_model_path=dl_model_path,
        dl_conf_threshold=dl_conf_threshold if dl_conf_threshold is not None else dl_fusion.DEFAULT_CONF_THRESHOLD,
    )
    if dl_model_path:
        print(f"ℹ️ [DL] Tầng dự phòng BẬT: {dl_model_path} (ngưỡng={dtw_engine.dl_conf_threshold:.2f}).")

    # SPEC 03: bộ dò mặt có cache — CHỈ tạo khi nhóm mốc cơ thể đang bật (w_b > 0), để không tốn CPU vô ích.
    # Nếu không tạo được (thiếu model / không có mạng để tải), query sẽ không có mốc cơ thể và
    # match() tự vô hiệu w_b cho lần so khớp đó (query_has_body=False) — không làm sập vòng lặp.
    face_detector: Optional[CachedFaceDetector] = None
    if CONFIG.get("w_b", 0.0) > 0.0:
        try:
            face_detector = CachedFaceDetector(detect_every_n_frames=CONFIG.get("face_detect_every_n_frames", 8))
            print("✓ Đã nạp bộ dò mặt (mốc cơ thể) có cache.")
        except Exception as e:
            print(f"[!] Không nạp được bộ dò mặt ({e}) -> tắt nhóm mốc cơ thể (w_b) cho phiên này.")
            face_detector = None
    if not dtw_engine.vocabulary:
        print("[!] CẢNH BÁO: dynamic_templates.json chưa có mẫu nào hoặc không tồn tại!")
        print("[!] Vui lòng chạy 'collect_dynamic_templates.py' trước để thu mẫu cử chỉ.")
    else:
        print(f"✓ Đã nạp {len(dtw_engine.vocabulary)} từ vựng cử chỉ mẫu từ {templates_file} (Golden Triad Engine).")

    enroller: Optional[SessionEnroller] = SessionEnroller(dtw_engine) if confirm_enroll else None
    if enroller is not None:
        print("✓ Chế độ HỎI XÁC NHẬN: cử chỉ không chắc sẽ được hỏi lại. Phím [y]=đúng (gợi ý 1)  [2]=đúng (gợi ý 2)  [n]=bỏ  [z]=hoàn tác.")

    # SPEC 08 / 08B: chế độ "ký nhanh" và cắt chuyển tiếp. Mặc định BẬT (CONFIG["fast_chain"]); --no-fast-chain => make_segmenter(CONFIG) như cũ.
    fc_cfg: Dict[str, Any] = dict(CONFIG.get("fast_chain", {}))
    if transition_lockout is not None:
        fc_cfg["transition_lockout_sec"] = float(transition_lockout)
    lt_cfg: Dict[str, Any] = dict(fc_cfg.get("lead_trim", {}))
    if lead_trim is not None:
        lt_cfg["enabled"] = bool(lead_trim)
    fc_cfg["lead_trim"] = lt_cfg

    fast_on = bool(fast_chain) if fast_chain is not None else bool(fc_cfg.get("enabled", False))
    lead_trim_on = bool(lt_cfg.get("enabled", False))

    if fc_cfg.get("multi_phase_words") is not None:
        multi_phase_words: List[str] = list(fc_cfg["multi_phase_words"])
    else:
        multi_phase_words = sorted(find_multi_phase_words(dtw_engine.vocabulary))
    if fast_on:
        print(f"⚡ [FastChain] BẬT: pause={fc_cfg.get('pause_sec')}s, grace={fc_cfg.get('grace_window_sec')}s, "
              f"cửa sổ nối={fc_cfg.get('chain_window_sec')}s, lockout={fc_cfg.get('transition_lockout_sec', 0.0)}s, "
              f"lead_trim={'BẬT' if lead_trim_on else 'TẮT'}, chốt sớm={fc_cfg.get('early_commit')} "
              f"(chất lượng>={fc_cfg.get('early_min_quality')}, margin>={fc_cfg.get('early_min_margin')}), "
              f"kéo đuôi={fc_cfg.get('tail_pad_sec')}s. Từ nhiều pha (không chốt sớm): {multi_phase_words or 'không có'}")
    else:
        print("ℹ️ [FastChain] TẮT (bật bằng --fast-chain hoặc phím [j]).")
    segmenter = make_segmenter(CONFIG, fc_cfg if fast_on else None)
    early_pre: Optional[Tuple[Any, Any, bool, Any, float, int]] = None   # (traj, shapes, body_ok, match, tau, n_tried) của dự đoán tạm khi chốt sớm
    lead_trim_history: List[float] = []   # Lưu lại tau của các từ đã commit trong phiên để in thống kê lúc thoát
    _commit_mode = "normal"

    # Khởi tạo bộ làm mượt tọa độ cổ tay (R3 Guard 1) - 1€ Filter thích ứng đa tần số
    wrist_smoother = make_wrist_smoother()
    lk_fallback = LKInertialFallback()
    cached_palm_len: float = 80.0
    cached_n_fingers: int = 0
    cached_hull_pts: Optional[np.ndarray] = None
    cached_valid_defects: Optional[List] = None
    stroke_angles_history: List[np.ndarray] = []
    stroke_normals_history: List[np.ndarray] = []
    # SPEC 03: lịch sử (21 landmark pixel-z*w, hộp mặt) từng frame, đồng bộ với stroke_angles_history.
    stroke_ext_history: List[Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]] = []
    current_face_box: Optional[Tuple[float, float, float, float]] = None

    # 4. Quản lý Bộ đệm Câu thoại (Tầng 4)
    sentence_words: List[str] = []
    # Danh sách khóa từ SONG SONG với sentence_words (luôn cùng độ dài) để hiển thị lại câu theo ngôn ngữ đang chọn.
    sentence_keys: List[str] = []
    last_commit_time = 0.0
    last_committed_word: Optional[str] = None
    last_hand_seen_time = time.time()
    sentence_finalized = False

    # Quản lý Banner phản hồi trực quan (R4 HUD)
    banner_title = ""
    banner_subtitle = ""
    banner_color = (0, 255, 0)
    banner_expire = 0.0

    # Giao diện demo song ngữ: toast có cấu trúc, ngôn ngữ, loa, bảng trợ giúp
    ui_mode = (ui_mode if ui_mode is not None else CONFIG.get("ui_mode", "demo"))
    ui_mode = ui_mode if ui_mode in ("demo", "debug") else "demo"
    ui_lang = (ui_lang if ui_lang is not None else CONFIG.get("ui_lang", "en"))
    ui_lang = ui_lang if ui_lang in ("en", "vi") else "en"
    hud_toast: Optional[HudToast] = None
    show_help = False
    demo_hud = DemoHUD(phrase_lookup=lambda k, lg: get_asl_phrase(k, lang=lg, default=""))
    hud_warned = False
    speech_service.init_speech()  # dò giọng tiếng Việt ở luồng nền (không chặn)

    # Quản lý Vệt sáng chuyển động neon (R4 Motion Trail)
    # Lưu các điểm dạng tuple: (x, y, timestamp)
    trail_points: List[Tuple[int, int, float]] = []
    MAX_TRAIL_AGE_SEC = 1.2
    hand_lost_frames = 0

    # Cấu hình tối ưu hóa hiệu năng (SPEC 02)
    if enable_static_shape_cnn is None:
        enable_static_shape_cnn = bool(CONFIG.get("enable_static_shape_cnn", False))
    if detect_scale is None:
        detect_scale = float(CONFIG.get("detect_scale", 1.0))
    detect_scale = max(0.1, min(1.0, float(detect_scale)))

    # Biến phục vụ tối ưu hóa xen kẽ suy luận DIP + CNN trên CPU
    frame_counter = 0
    cached_hand_shape = "Unknown" if not enable_static_shape_cnn else "B"
    cached_box = (0, 0, 0, 0)
    recording_streak = 0   # Số khung liên tiếp đang ở trạng thái RECORDING (dùng để ép CNN chạy đủ ngay lúc bắt đầu cử chỉ)

    # Bộ đệm đo thời gian theo tầng (SPEC 02 Nhiệm vụ 1)
    if profile:
        prof_mediapipe = deque(maxlen=60)
        prof_dip_cnn = deque(maxlen=60)
        prof_features = deque(maxlen=60)
        prof_dtw = deque(maxlen=60)
        prof_draw = deque(maxlen=60)
        prof_total = deque(maxlen=60)
    else:
        prof_total = []

    # Mở nguồn video hoặc Webcam — thử MSMF trước, rồi fallback sang DSHOW
    cap = None
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] KHÔNG THỂ mở video: {video_path}")
            cap = None
        else:
            print(f"[Video] KẾT NỐI THÀNH CÔNG video file: {video_path}")
    elif not headless:
        cap = open_camera(camera_index, prefer_mjpg=True)

    flip_cam = False
    if show_features is None:
        show_features = False
    prev_time = time.time()
    fps = 0.0

    # Giám sát sức khỏe camera: CHỈ đo (khung mới thật/giây, độ sáng) và cảnh báo trên HUD, không đổi nhận diện.
    # Khung giả lập headless (cap None) không được đo; video file mặc định không cảnh báo (khung trùng là bình thường).
    cam_cfg = dict(CONFIG.get("camera_health", {}))
    cam_monitor: Optional[CameraHealthMonitor] = None
    if cam_cfg.get("enabled", True) and cap is not None:
        cam_monitor = CameraHealthMonitor(
            cam_cfg, warnings_enabled=(video_path is None or bool(cam_cfg.get("warn_on_video", False))))
    cam_hand_moving_v = float(cam_cfg.get("hand_moving_v", 0.03))
    dedupe_frames = bool(dedupe_frames if dedupe_frames is not None else CONFIG.get("dedupe_frames", False))
    cam_last_v_norm = 0.0   # v_norm của khung xử lý gần nhất (kết quả dò tay chạy nền nên trễ vài khung)

    print("\n[HƯỚNG DẪN SỬ DỤNG]")
    print("  - Múa cử chỉ rõ ràng, nghỉ nhẹ ~0.2s giữa hai từ (chế độ KÝ NHANH mặc định). Nếu tắt bằng [j], phải dừng ~0.55s.")
    print("  - Gạt nhanh tay sang trái (Swipe-Left) để XÓA TỪ gần nhất (Backspace).")
    print("  - Hạ tay xuống khỏi màn hình > 1.8s để HOÀN THÀNH CÂU (Commit/Enter).")
    print("  - Hệ thống tự động phát âm tiếng Anh (ASL) bằng giọng nói pyttsx3.")
    print("  [d]: Bật/Tắt bảng đặc trưng 3D (Feature HUD, chỉ ở giao diện debug) | [c]: Xóa câu | [f]: Lật gương | [q]: Thoát")
    print("  [l]: Đổi ngôn ngữ Anh/Việt | [u]: Đổi giao diện demo/debug | [t]: Bật/tắt loa | [h]: Trợ giúp")
    print("  [j]: Bật/tắt chế độ KÝ NHANH (nối từ, nghỉ ngắn ~0.2s là đủ) | [k]: Bật/tắt CẮT DẪN ĐẦU (lead-trim)\n")
    print(f"  Giao diện: {ui_mode} | Ngôn ngữ: {ui_lang}\n")

    processed_frames = 0
    has_predicted_pending = False
    ready_queue: deque = deque()
    cap_exhausted = False

    try:
        while True:
            if profile:
                _t_frame_start = time.perf_counter()
                _t_mp = 0.0
                _t_dip = 0.0
                _t_feat = 0.0
                _t_dtw = 0.0
                _t_draw = 0.0

            if not cap_exhausted:
                if cap is None or not cap.isOpened():
                    # Chế độ headless kiểm thử: tạo khung hình giả lập nếu không có camera thật
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    time.sleep(0.033)
                    ret = True
                else:
                    ret, frame = cap.read()
                    if not ret:
                        if video_path is not None and max_frames is not None and processed_frames < max_frames:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        if not ret:
                            cap_exhausted = True
                            ready_queue.extend(detector_pool.flush())

                if ret and frame is not None:
                    capture_ts = time.time()
                    if cam_monitor is not None:
                        # Gọi MỖI lần đọc được khung (kể cả khung trùng), trước lật gương; tay chuyển động = v_norm khung trước
                        cam_monitor.update(frame, capture_ts, cam_last_v_norm >= cam_hand_moving_v)
                    # Bỏ khung TRÙNG (driver lặp đúng bộ đệm cũ, is_new=False): không dò tay, không đưa vào bộ phân đoạn, để
                    # dt và vận tốc phản ánh khung MỚI thật. Mặc định TẮT (CONFIG["dedupe_frames"]); vẫn poll kết quả dò như thường.
                    is_dup_frame = bool(dedupe_frames and cam_monitor is not None and not cam_monitor.is_new)
                    frame_counter += 1
                    if not is_dup_frame:
                        if flip_cam:
                            frame = cv2.flip(frame, 1)

                        if detect_scale < 0.999:
                            det_frame = cv2.resize(frame, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
                        else:
                            det_frame = frame
                        rgb_frame = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)

                        meta = {
                            "capture_ts": capture_ts,
                            "frame": frame,
                            "rgb_frame": rgb_frame,
                            "submit_perf_ts": time.perf_counter()
                        }
                        detector_pool.submit(frame_counter, rgb_frame, meta)

                    if profile:
                        _t0_mp = time.perf_counter()
                    new_ready = detector_pool.poll(block_oldest=(detector_pool.in_flight >= detector_pool.max_in_flight))
                    if profile:
                        _t_mp = time.perf_counter() - _t0_mp
                    ready_queue.extend(new_ready)

            if not ready_queue:
                if cap_exhausted:
                    break
                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                continue

            seq, meta_item, det_result = ready_queue.popleft()
            processed_frames += 1
            frame = meta_item["frame"]
            curr_time = meta_item["capture_ts"]
            rgb_frame = meta_item.get("rgb_frame")
            if rgb_frame is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_f, w_f = frame.shape[:2]

            # Tính toán FPS trung bình trượt từ capture_ts
            loop_duration = max(curr_time - prev_time, 1e-4)
            fps = 0.9 * fps + 0.1 * (1.0 / loop_duration)
            prev_time = curr_time

            display_frame = frame.copy()
            cam_last_v_norm = 0.0   # cập nhật lại sau khi phân đoạn tính v_norm; mất tay -> 0 (không chuyển động)

            # SPEC 03: hộp mặt (cx, cy, w, h) theo pixel KHUNG GỐC; ảnh dò có thể đã thu nhỏ (detect_scale).
            if face_detector is not None:
                current_face_box = face_detector.detect(rgb_frame, out_size=(w_f, h_f))
            else:
                current_face_box = None

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
            early_pre = None    # SPEC 08: kết quả dự đoán tạm dùng lại khi chốt sớm (chỉ đặt trong nhánh 3.5)
            is_swipe = False
            v_norm = 0.0
            x1, y1, x2, y2 = cached_box
            current_angles: Optional[np.ndarray] = None
            current_normal: Optional[np.ndarray] = None
            hand_center_px: Optional[Tuple[float, float]] = None   # tâm lòng bàn tay (px) cho vòng trạng thái ở giao diện demo

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
                hand_center_px = ((lm0.x + lm9.x) * 0.5 * w_f, (lm0.y + lm9.y) * 0.5 * h_f)

                # Nguồn sự thật duy nhất của palm_len (dynamic_features.compute_palm_len), dùng chung với collector.
                palm_len = compute_palm_len(lm0, lm5, lm9, lm17, w_f, h_f)
                cached_palm_len = palm_len

                # Áp dụng R3 Layer 1 Guard: 1€ Filter smoothing trên tọa độ cổ tay
                wrist_x, wrist_y = wrist_smoother.update(raw_wrist_x, raw_wrist_y, timestamp=curr_time)

                # Cập nhật mốc tham chiếu cho LK Inertial Fallback
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lk_fallback.update_reference(gray_frame, landmarks)

                # Trích xuất 5 góc uốn ngón tay 3D & Vector pháp tuyến lòng bàn tay 3D (12D Features)
                lm_3d = np.array([[lm.x * w_f, lm.y * h_f, lm.z * w_f] for lm in landmarks], dtype=np.float32)
                if profile:
                    _t0_feat = time.perf_counter()
                current_angles = compute_joint_flexion_angles(lm_3d)
                current_normal = compute_palm_normal_vector(lm_3d, handedness=handedness_label)
                if profile:
                    _t_feat = time.perf_counter() - _t0_feat
                stroke_angles_history.append(current_angles)
                stroke_normals_history.append(current_normal)
                stroke_ext_history.append((lm_3d, current_face_box))
                if len(stroke_angles_history) > 120:
                    stroke_angles_history.pop(0)
                if len(stroke_normals_history) > 120:
                    stroke_normals_history.pop(0)
                if len(stroke_ext_history) > 120:
                    stroke_ext_history.pop(0)

                # 2. Nhận diện Dáng tay tức thời qua DIP Engine v4 + Two-Stream CNN + Soft Bayesian Prior
                # Kỹ thuật xen kẽ suy luận CPU: thực hiện suy luận mỗi 2 khung hình để duy trì FPS >= 25
                if enable_static_shape_cnn:
                    if profile:
                        _t0_dip = time.perf_counter()
                    run_dip_cnn = (frame_counter % 2 == 0) or (cached_hand_shape == "Unknown") or (recording_streak < 4)
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
                    if profile:
                        _t_dip = time.perf_counter() - _t0_dip
                else:
                    hand_shape = "Unknown"
                    cached_hand_shape = "Unknown"

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
                    stroke_ext_history.clear()

                # Tự động chốt câu hoàn chỉnh khi hạ tay > 1.8s (Sentence Commit Timeout)
                if len(sentence_words) > 0 and (curr_time - last_hand_seen_time) > 1.8 and not sentence_finalized:
                    sentence_finalized = True
                    final_sentence = " ".join(sentence_words) + " ."
                    banner_title = "SENTENCE COMMITTED (ENTER)"
                    banner_subtitle = f"\"{final_sentence}\""
                    banner_color = (0, 255, 0)
                    banner_expire = curr_time + 2.0
                    hud_toast = _make_toast("committed", [], None, curr_time, 2.0)
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
                cam_last_v_norm = v_norm
                if seg_state == KineticState.RECORDING:
                    recording_streak += 1
                    has_predicted_pending = False
                else:
                    recording_streak = 0
                if seg_state in (KineticState.IDLE, KineticState.READY):
                    has_predicted_pending = False

                # 3.5. Dự đoán tạm thời khi PENDING_COMMIT (Nhiệm vụ E)
                if seg_state == KineticState.PENDING_COMMIT and not has_predicted_pending:
                    has_predicted_pending = True
                    temp_stroke = segmenter.buffer
                    if temp_stroke and len(temp_stroke) >= segmenter.effective_min_stroke_frames():
                        temp_recs = _segment_records(temp_stroke, stroke_angles_history,
                                                     stroke_normals_history, stroke_ext_history)
                        if fast_on and lead_trim_on:
                            # SPEC 08B: Cắt phần dẫn đầu thích nghi khi PENDING_COMMIT
                            _lt_res = select_best_lead_trim(
                                temp_recs, dtw_engine, cfg=lt_cfg, handedness=handedness_label,
                                query_has_body=True, tail_pad_sec=fc_cfg.get("tail_pad_sec"),
                                v_stop=segmenter.v_stop_ratio
                            )
                            t_match = _lt_res.match_result
                            _tau_sel = float(_lt_res.trim_sec)
                            _n_tried = int(_lt_res.n_tried)
                            if fast_on and early_commit_ok(t_match, fc_cfg, multi_phase_words):
                                _forced = segmenter.force_commit(curr_time)
                                if _forced is not None:
                                    completed_stroke = _forced
                                    early_pre = (None, None, True, t_match, _tau_sel, _n_tried)
                                    seg_state = segmenter.state
                            if early_pre is None and t_match.status == "ACCEPTED" and t_match.top_word:
                                t_phrase = get_asl_phrase(t_match.top_word, lang="en", default=t_match.top_display or t_match.top_word)
                                banner_title = f"? {t_phrase} (pending)"
                                banner_subtitle = f"Temporary prediction (tau={_tau_sel:.2f}s)"
                                banner_color = (180, 180, 180) # màu xám nhạt
                                banner_expire = curr_time + 1.0
                                hud_toast = _make_toast("pending", [t_match.top_word], None, curr_time, 1.0)
                        else:
                            if fast_on and fc_cfg.get("tail_pad_sec"):
                                # Chế độ ký nhanh: đuôi yên ngắn hơn bank => kéo về ~0.45 s (nhân bản khung cuối) để nhận dạng
                                # không phụ thuộc thời gian chờ; thời lượng cho cổng [0.4, 2.5]s cũng tính theo bản đã kéo.
                                temp_recs, _ = extend_still_tail(temp_recs, float(fc_cfg["tail_pad_sec"]), segmenter.v_stop_ratio,
                                                                replay_jitter=True)
                                temp_duration = temp_recs[-1]["t"] - temp_recs[0]["t"]
                                temp_epi_frames = records_to_gesture_frames(temp_recs)   # độ thẳng tính trên bản đã kéo đuôi (có nhiễu)
                            else:
                                temp_duration = temp_stroke[-1].timestamp - temp_stroke[0].timestamp
                                temp_epi_frames = temp_stroke
                            if is_stroke_duration_valid(temp_duration) and not is_movement_epenthesis(temp_epi_frames):
                                # Dựng đặc trưng bằng hàm dùng chung (dynamic_features.build_segment_features), bit-giống hệt
                                # đoạn inline cũ; đệm lịch sử khi ngắn hơn stroke do _segment_records lo.
                                rt_17d, rt_shapes, t_body_ok, _t_dom = build_segment_features(temp_recs)
                                if rt_17d is not None:
                                    if profile:
                                        _t0_dtw = time.perf_counter()
                                    t_match = dtw_engine.match(rt_17d, rt_shapes, handedness=handedness_label, query_has_body=t_body_ok)
                                    if profile:
                                        _t_dtw += time.perf_counter() - _t0_dtw
                                    # SPEC 08 - CHỐT SỚM: dự đoán tạm đủ chắc (ACCEPTED, chất lượng, margin, không phải từ nhiều pha)
                                    # thì chốt ngay, không đợi hết grace. Dùng lại kết quả match vừa tính (không gọi match() lần hai);
                                    # force_commit() đưa FSM về IDLE nên stroke này KHÔNG thể được chốt lần thứ hai.
                                    if fast_on and early_commit_ok(t_match, fc_cfg, multi_phase_words):
                                        _forced = segmenter.force_commit(curr_time)
                                        if _forced is not None:
                                            completed_stroke = _forced
                                            early_pre = (rt_17d, rt_shapes, t_body_ok, t_match, 0.0, 1)
                                            seg_state = segmenter.state
                                    if early_pre is None and t_match.status == "ACCEPTED" and t_match.top_word:
                                        t_phrase = get_asl_phrase(t_match.top_word, lang="en", default=t_match.top_display or t_match.top_word)
                                        banner_title = f"? {t_phrase} (pending)"
                                        banner_subtitle = f"Temporary prediction"
                                        banner_color = (180, 180, 180) # màu xám nhạt
                                        banner_expire = curr_time + 1.0
                                        hud_toast = _make_toast("pending", [t_match.top_word], None, curr_time, 1.0)

                # 4. Quản lý vệt sáng chuyển động neon (Lưu tọa độ + nhãn thời gian)
                trail_points.append((int(wrist_x), int(wrist_y), curr_time))

                # 5. Xử lý LỆNH ƯU TIÊN CẤP TỐC: SWIPE-LEFT (BACKSPACE)
                if is_swipe:
                    if len(sentence_words) > 0:
                        removed = sentence_words.pop()
                        removed_key = sentence_keys.pop() if sentence_keys else ""
                        hud_toast = _make_toast("deleted", [removed_key] if removed_key else [], None, curr_time, 1.6)
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
                        hud_toast = _make_toast("empty", [], None, curr_time, 1.2)
                    trail_points.clear()
                    stroke_angles_history.clear()
                    stroke_ext_history.clear()

                # 6. Xử lý CỬ CHỈ HOÀN THÀNH -> SO KHỚP GOLDEN TRIAD DTW (Tầng 3)
                if completed_stroke is not None:
                    stroke_duration = completed_stroke[-1].timestamp - completed_stroke[0].timestamp
                    _commit_mode = "early" if early_pre is not None else "normal"
                    cur_lead_trim_sec = 0.0
                    cur_lead_trim_tried = 1
                    seg_records = None
                    epi_frames = completed_stroke
                    if fast_on and fc_cfg.get("tail_pad_sec"):
                        # Chế độ ký nhanh: kéo đuôi yên về ~0.45 s như bank (xem nhánh 3.5); thời lượng cổng tính theo bản đã kéo.
                        seg_records, _ = extend_still_tail(
                            _segment_records(completed_stroke, stroke_angles_history, stroke_normals_history, stroke_ext_history),
                            float(fc_cfg["tail_pad_sec"]), segmenter.v_stop_ratio, replay_jitter=True)
                        stroke_duration = seg_records[-1]["t"] - seg_records[0]["t"]
                        epi_frames = records_to_gesture_frames(seg_records)   # độ thẳng tính trên bản đã kéo đuôi (có nhiễu)

                    # Áp dụng R3 Layer 2 Guard: Cổng thời lượng cử chỉ [0.4s, 2.5s].
                    # Hai con số này lấy từ hằng số DÙNG CHUNG trong dynamic_dtw_engine.py
                    # (STROKE_DURATION_MIN_SEC / STROKE_DURATION_MAX_SEC) để cổng ở đây và
                    # cổng ở collect_dynamic_templates.py không bao giờ lệch nhau nữa.
                    if is_stroke_duration_valid(stroke_duration):
                        if is_movement_epenthesis(epi_frames):
                            print("[BO QUA: chuyen dong chuyen tiep]")
                        else:
                            n_stroke = len(completed_stroke)
                            # Dựng đặc trưng bằng hàm dùng chung (dynamic_features.build_segment_features), bit-giống hệt
                            # đoạn inline cũ (neo tại khung đầu, 17D + 7 cột mở rộng, resample K=32).
                            raw_records = _segment_records(completed_stroke, stroke_angles_history,
                                                           stroke_normals_history, stroke_ext_history)
                            stroke_angles_history.clear()
                            stroke_normals_history.clear()
                            stroke_ext_history.clear()
                            match_result = None
                            if early_pre is not None:
                                # Chốt sớm: dùng lại đặc trưng + kết quả match của dự đoán tạm (không tính lại).
                                _ep_traj, _ep_shp, _ep_body, match_result, cur_lead_trim_sec, cur_lead_trim_tried = early_pre
                                body_ok = _ep_body
                            elif fast_on and lead_trim_on:
                                # SPEC 08B: Cắt phần dẫn đầu thích nghi ở nhánh chốt thường
                                _lt_res = select_best_lead_trim(
                                    raw_records, dtw_engine, cfg=lt_cfg, handedness=handedness_label,
                                    query_has_body=True, tail_pad_sec=fc_cfg.get("tail_pad_sec"),
                                    v_stop=segmenter.v_stop_ratio
                                )
                                match_result = _lt_res.match_result
                                cur_lead_trim_sec = float(_lt_res.trim_sec)
                                cur_lead_trim_tried = int(_lt_res.n_tried)
                                body_ok = True
                            else:
                                if seg_records is None:
                                    seg_records = raw_records
                                res_traj_17d, res_shapes, body_ok, _seg_dom = build_segment_features(seg_records)
                                if res_traj_17d is not None:
                                    if profile:
                                        _t0_dtw = time.perf_counter()
                                    match_result = dtw_engine.match(res_traj_17d, res_shapes, handedness=handedness_label, query_has_body=body_ok)
                                    if profile:
                                        _t_dtw += time.perf_counter() - _t0_dtw

                            if match_result is not None:
                                if match_result.fusion_note:
                                    print(f"[Fusion] {match_result.fusion_note}")
                                # a) Chấp nhận từ vựng (ACCEPTED)
                                if match_result.status == "ACCEPTED" and match_result.top_word:
                                    word_key = match_result.top_word
                                    en_phrase = get_asl_phrase(word_key, lang="en", default=match_result.top_display or word_key)
                                    vi_phrase = get_asl_phrase(word_key, lang="vi", default="")

                                    # R3 Layer 3 / Debouncing: Chống bắn lặp từ trong 0.8s
                                    is_repeat = (en_phrase == last_committed_word and (curr_time - last_commit_time) < 0.8)
                                    if not is_repeat:
                                        sentence_words.append(en_phrase)
                                        sentence_keys.append(word_key)
                                        last_commit_time = curr_time
                                        last_committed_word = en_phrase
                                        sentence_finalized = False
                                        lead_trim_history.append(cur_lead_trim_sec)

                                        # Điểm độ khớp (SPEC 04B: Do khop thay vì xác suất)
                                        do_khop = match_result.match_quality
                                        r_val = match_result.r_ratio

                                        # Banner phản hồi trực quan cỡ lớn giữ trong 2.0s (R4 & SPEC 04B)
                                        banner_title = f"{en_phrase} (Do khop: {do_khop:.0f})"
                                        banner_subtitle = f"ASL Match: {en_phrase} [{vi_phrase}] | Dist: {match_result.best_distance:.3f} (r={r_val:.2f})"
                                        if cur_lead_trim_sec > 0.0:
                                            banner_subtitle += f" [trim {cur_lead_trim_sec:.2f}s]"
                                        banner_color = (0, 255, 100)
                                        banner_expire = curr_time + 2.0
                                        hud_toast = _make_toast("accepted", [word_key], float(do_khop), curr_time, 2.0)

                                        # Phát âm theo ngôn ngữ giao diện, bất đồng bộ không chặn (R4 Non-blocking TTS).
                                        # Tôn trọng phím [t] (tắt loa); lỗi giọng nói không được làm sập vòng lặp.
                                        try:
                                            speech_service.speak_word(
                                                word_key, ui_lang,
                                                get_asl_phrase(word_key, lang=ui_lang, default=en_phrase)
                                            )
                                        except Exception as speech_err:
                                            print(f"[Speech] Lỗi giọng nói ({speech_err}) — bỏ qua, tiếp tục chạy.")
                                        trim_str = f" [trim: {cur_lead_trim_sec:.2f}s, {cur_lead_trim_tried} tried]" if cur_lead_trim_sec > 0.0 else ""
                                        print(f"[SentenceBuilder] Nhận từ: '{en_phrase}' (Do khop: {do_khop:.0f}){trim_str} | Câu: {' '.join(sentence_words)}")

                                    # Ghi CSV Session Log
                                    if session_writer is not None:
                                        c2_k = match_result.candidates[1].word_key if len(match_result.candidates) >= 2 else ""
                                        c2_d = round(match_result.candidates[1].distance, 4) if len(match_result.candidates) >= 2 else ""
                                        m_val = round(((match_result.candidates[1].distance - match_result.best_distance) / max(match_result.best_distance, 1e-4)) * 100.0, 2) if len(match_result.candidates) >= 2 else 100.0
                                        session_writer.writerow([
                                            time.strftime("%Y-%m-%d %H:%M:%S"), "ACCEPTED", word_key, round(match_result.best_distance, 4),
                                            c2_k, c2_d, m_val, round(stroke_duration, 3), n_stroke, body_ok, round(fps, 1),
                                            round(match_result.best_confidence, 4), round(match_result.match_quality, 2), round(match_result.r_ratio, 4), round(curr_time, 3), _commit_mode, round(completed_stroke[0].timestamp, 3),
                                            round(cur_lead_trim_sec, 3), cur_lead_trim_tried
                                        ])
                                        session_file.flush()

                                # b) Nghi vấn Top-2 (UNCERTAIN) - R3 Layer 3 Guard: Không commit từ
                                elif match_result.status == "UNCERTAIN" and len(match_result.candidates) >= 2:
                                    c1, c2 = match_result.candidates[0], match_result.candidates[1]
                                    en1 = get_asl_phrase(c1.word_key, lang="en", default=c1.display_name)
                                    en2 = get_asl_phrase(c2.word_key, lang="en", default=c2.display_name)
                                    margin_pct = ((c2.distance - c1.distance) / max(c1.distance, 1e-4)) * 100.0
                                    banner_title = f"? UNCERTAIN: {en1} or {en2}"
                                    banner_subtitle = f"Dist: {match_result.best_distance:.2f} | Do khop: {match_result.match_quality:.0f} | Margin: {margin_pct:.1f}%"
                                    banner_color = (0, 200, 255)
                                    banner_expire = curr_time + 1.4
                                    hud_toast = _make_toast("uncertain", [c1.word_key, c2.word_key],
                                                            float(match_result.match_quality), curr_time, 1.4)
                                    reason = "Margin Guard (2 từ quá sát nhau)" if c1.distance <= (c1.r_ratio * 1.6) else "Vượt ngưỡng chuẩn"
                                    print(f"[SentenceBuilder] ❓ Nghi vấn ({reason}): {en1} (d1={c1.distance:.2f}, Do khop={c1.match_quality:.0f}) vs {en2} (d2={c2.distance:.2f}) | Margin: {margin_pct:.1f}%")

                                    # Ghi CSV Session Log
                                    if session_writer is not None:
                                        session_writer.writerow([
                                            time.strftime("%Y-%m-%d %H:%M:%S"), "UNCERTAIN", c1.word_key, round(c1.distance, 4),
                                            c2.word_key, round(c2.distance, 4), round(margin_pct, 2), round(stroke_duration, 3), n_stroke, body_ok, round(fps, 1),
                                            round(match_result.best_confidence, 4), round(match_result.match_quality, 2), round(match_result.r_ratio, 4), round(curr_time, 3), _commit_mode, round(completed_stroke[0].timestamp, 3),
                                            round(cur_lead_trim_sec, 3), cur_lead_trim_tried
                                        ])
                                        session_file.flush()

                                # c) Từ chối (REJECTED) - R3 Layer 3 Guard: Khoảng cách quá xa
                                else:
                                    banner_title = "X UNKNOWN GESTURE"
                                    banner_subtitle = f"DTW Distance: {match_result.best_distance:.2f} | Do khop: {match_result.match_quality:.0f} (Rejected)"
                                    banner_color = (140, 140, 140)  # Xám per SPEC 04B
                                    banner_expire = curr_time + 1.0
                                    hud_toast = _make_toast("rejected", [], float(match_result.match_quality), curr_time, 1.0)
                                    print(f"[SentenceBuilder] Từ chối cử chỉ không khớp (Dist: {match_result.best_distance:.2f}, Do khop: {match_result.match_quality:.0f})")

                                    # Ghi CSV Session Log
                                    if session_writer is not None:
                                        top_k = match_result.candidates[0].word_key if match_result.candidates else ""
                                        session_writer.writerow([
                                            time.strftime("%Y-%m-%d %H:%M:%S"), "REJECTED", top_k, round(match_result.best_distance, 4),
                                            "", "", 0.0, round(stroke_duration, 3), n_stroke, body_ok, round(fps, 1),
                                            round(match_result.best_confidence, 4), round(match_result.match_quality, 2), round(match_result.r_ratio, 4), round(curr_time, 3), _commit_mode, round(completed_stroke[0].timestamp, 3),
                                            round(cur_lead_trim_sec, 3), cur_lead_trim_tried
                                        ])
                                        session_file.flush()
                            if enroller is not None:
                                # Hỏi xác nhận: chỉ khi không chắc / từ chối và có ứng viên; ACCEPTED thì xóa câu hỏi cũ.
                                enroller.clear()
                                if match_result is not None and match_result.status != "ACCEPTED" and match_result.candidates:
                                    _enr_traj, _enr_shp, _enr_body, _ = build_segment_features(seg_records if seg_records is not None else raw_records)
                                    _enr_keys = [c.word_key for c in match_result.candidates[:2]]
                                    if enroller.offer(_enr_traj, _enr_shp, handedness_label, _enr_body, stroke_duration, _enr_keys, curr_time):
                                        _enr_names = [get_asl_phrase(k, lang="en", default=k) for k in _enr_keys]
                                        banner_title = f"Is it {_enr_names[0]}?"
                                        banner_subtitle = "[Y] yes   " + (f"[2] {_enr_names[1]}   " if len(_enr_names) > 1 else "") + "[N] skip"
                                        banner_color = (0, 200, 255)
                                        banner_expire = curr_time + enroller.ttl
                                        print(f"[XacNhan] Hỏi: có phải '{_enr_names[0]}'? (y) / '{_enr_names[1] if len(_enr_names) > 1 else '-'}'? (2) / bỏ (n)")

                    else:
                        # R3 Layer 2: Cử chỉ nằm ngoài [0.4s, 2.5s] bị bỏ qua trong im lặng
                        print(f"[SentenceBuilder] Cử chỉ bị loại bởi Duration Gate ({stroke_duration:.2f}s).")
                        if session_writer is not None:
                            session_writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"), "OUT_OF_DURATION", "", 0.0,
                                "", "", 0.0, round(stroke_duration, 3), len(completed_stroke), False, round(fps, 1),
                                0.0, 0.0, 0.0, round(curr_time, 3), _commit_mode, round(completed_stroke[0].timestamp, 3),
                                0.0, 1
                            ])
                            session_file.flush()

                    # LƯU Ý R4: Không xóa trail_points tại đây để vệt sáng phai tự nhiên

                if profile:
                    _t0_draw_box = time.perf_counter()
                # Vẽ Bounding Box & Trực quan hóa DIP (Convex Hull & Defects)
                if ui_mode == "debug" and cached_box is not None and len(cached_box) == 4 and cached_box[2] > cached_box[0] and cached_box[3] > cached_box[1]:
                    x1, y1, x2, y2 = cached_box
                    box_col = (0, 255, 0) if seg_state in (KineticState.RECORDING, KineticState.PENDING_COMMIT) else (255, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_col, 2)
                    cv2.putText(display_frame, f"Shape: {hand_shape} (Fingers: {n_fingers})", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1, lineType=cv2.LINE_AA)

                    if hull_pts is not None and len(hull_pts) > 0:
                        hull_display = hull_pts + np.array([x1, y1])
                        cv2.polylines(display_frame, [hull_display], True, (0, 255, 128), 1, lineType=cv2.LINE_AA)
                    if valid_defects is not None and len(valid_defects) > 0:
                        for s_p, e_p, f_p in valid_defects:
                            cv2.circle(display_frame, (int(f_p[0] + x1), int(f_p[1] + y1)), 4, (0, 0, 255), -1)

                # Vẽ 3D Skeleton & Góc gập (Flexion) & Vector Pháp Tuyến trực tiếp lên bàn tay
                if ui_mode == "debug" and show_features:
                    draw_hand_skeleton_features(display_frame, landmarks, current_angles, w_f, h_f, handedness=handedness_label)
                if profile:
                    _t_draw += time.perf_counter() - _t0_draw_box

            if profile:
                _t0_draw_hud = time.perf_counter()
            # Lọc bớt các điểm vệt sáng quá cũ (> MAX_TRAIL_AGE_SEC)
            trail_points = [pt for pt in trail_points if (curr_time - pt[2]) <= MAX_TRAIL_AGE_SEC]

            # =================================================================
            # HIỂN THỊ GIAO DIỆN ĐỒ HỌA HUD THÔNG MINH (TẦNG 4)
            # =================================================================
            # 0. Vẽ Bounding Box & Mốc cơ thể Khuôn mặt trực quan (Body Anchor)
            if ui_mode == "debug":
                draw_face_anchor_box(
                    display_frame,
                    face_box=current_face_box,
                    wrist_pt=(int(wrist_x), int(wrist_y)) if hand_found else None,
                    draw_vector=show_features
                )

            # 1. Vẽ vệt sáng chuyển động neon (Fading Neon Motion Trail)
            if ui_mode == "debug":
                draw_fading_neon_trail(display_frame, trail_points, curr_time, max_age_sec=MAX_TRAIL_AGE_SEC)
            else:
                draw_soft_trail(display_frame, trail_points, curr_time, max_age_sec=MAX_TRAIL_AGE_SEC)

            if ui_mode == "debug":
                # 2. Thanh trạng thái phía trên (Top Status Bar)
                cv2.rectangle(display_frame, (0, 0), (w_f, 44), (20, 20, 20), -1)
                fps_color = (0, 255, 0) if fps >= 24.0 else ((0, 200, 255) if fps >= 20.0 else (0, 0, 255))
                cv2.putText(display_frame, f"FPS: {fps:.1f} [CAM {camera_index}]", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 2, lineType=cv2.LINE_AA)
                # Chẩn đoán camera: khung MỚI THẬT/s, trùng (giống hệt từng điểm ảnh), gần-trùng (ngưỡng cũ 0.5), độ sáng
                if cam_monitor is not None:
                    cv2.putText(display_frame,
                                f"that:{cam_monitor.unique_fps:.0f}/s trung:{cam_monitor.dup_ratio*100:.0f}% "
                                f"(gan-trung {cam_monitor.near_dup_ratio*100:.0f}%) sang:{cam_monitor.luma_mean:.0f}",
                                (10, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 120), 1, lineType=cv2.LINE_AA)

                # Trạng thái mốc cơ thể (khuôn mặt)
                if face_detector is None:
                    face_badge = "MAT: TAT"
                    face_col = (130, 130, 130)
                elif current_face_box is not None:
                    face_badge = "MAT: OK"
                    face_col = (0, 255, 0)
                else:
                    face_badge = "MAT: TIM..."
                    face_col = (0, 165, 255)
                cv2.putText(display_frame, f"| {face_badge}", (195, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, face_col, 2, lineType=cv2.LINE_AA)

                # Trạng thái phân đoạn động học
                if seg_state == KineticState.RECORDING:
                    k_text = "DANG GHI"
                    k_color = (0, 255, 0)
                elif seg_state == KineticState.READY:
                    k_text = "SAN SANG"
                    k_color = (0, 255, 0)
                else:
                    k_text = "KHUNG TAY" if hand_found else "CHO TAY"
                    k_color = (180, 180, 180)
                cv2.putText(display_frame, f"| {k_text}", (310, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, k_color, 2, lineType=cv2.LINE_AA)

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

            else:
                # ---------------- GIAO DIỆN DEMO SONG NGỮ (ui_render.DemoHUD) ----------------
                # Không vẽ hộp bao / hull / defects / khung mặt / skeleton / thanh vận tốc / bảng 12D / khoảng cách DTW.
                _seg_val = getattr(seg_state, "value", str(seg_state))
                _settle: Optional[float] = None
                if getattr(segmenter, "require_ready_pose", False):
                    if seg_state == KineticState.READY:
                        _settle = 1.0
                    elif seg_state == KineticState.IDLE:
                        _settle = min(1.0, segmenter.ready_counter / max(1, segmenter.effective_ready_stable_frames()))
                _rec = 0.0
                if seg_state in (KineticState.RECORDING, KineticState.PENDING_COMMIT):
                    _rec = min(1.0, len(segmenter.buffer) / max(1, segmenter.effective_max_stroke_frames()))
                # SPEC 08: cửa sổ nối đang mở => vòng quanh tay giữ trạng thái "sẵn sàng" (đầy, xanh), chip "Ký tiếp được rồi"
                _chain_ready = bool(fast_on and hand_found and segmenter.chain_window_open(curr_time))
                if not hand_found:
                    _center = None
                elif hand_center_px is not None:
                    _center = hand_center_px
                else:
                    _center = (wrist_x, wrist_y)
                if face_detector is None:
                    _face_state = "off"
                elif current_face_box is not None:
                    _face_state = "ok"
                else:
                    _face_state = "searching"
                _cam_hud_fields = {}
                if cam_monitor is not None:
                    _cam_status = cam_monitor.status()
                    _cam_hud_fields = dict(
                        unique_fps=cam_monitor.unique_fps, loop_fps=cam_monitor.loop_fps,
                        camera_warning=_cam_status[0] if _cam_status else None,
                        camera_warning_level=_cam_status[1] if _cam_status else "warn",
                        dup_ratio=cam_monitor.dup_ratio, near_dup_ratio=cam_monitor.near_dup_ratio,
                        luma_mean=cam_monitor.luma_mean,
                    )
                hud_state = HudState(
                    lang=ui_lang,
                    seg_state=_seg_val,
                    hand_found=hand_found,
                    hand_center=_center,
                    palm_len=palm_len,
                    v_norm=v_norm,
                    settle_progress=_settle,
                    record_progress=_rec,
                    fps=fps,
                    sentence_keys=list(sentence_keys),
                    sentence_finalized=sentence_finalized,
                    toast=hud_toast if (hud_toast is not None and curr_time <= hud_toast.t_expire) else None,
                    vocabulary_empty=(len(dtw_engine.vocabulary) == 0),
                    face_state=_face_state,
                    speech_status=speech_service.get_speech_status(ui_lang),
                    show_help=show_help,
                    camera_index=camera_index,
                    now=curr_time,
                    chain_ready=_chain_ready,
                    **_cam_hud_fields,
                )
                try:
                    demo_hud.render(display_frame, hud_state)
                except Exception as hud_err:
                    if not hud_warned:
                        hud_warned = True
                        print(f"[HUD] Lỗi vẽ giao diện demo ({hud_err}) — bỏ qua khung hình này, vòng lặp vẫn chạy.")

            if profile:
                _t_draw += time.perf_counter() - _t0_draw_hud

            # Bảng nhỏ profile góc màn hình (SPEC 02 Nhiệm vụ 1)
            if profile and len(prof_total) > 0:
                avg_mp = sum(prof_mediapipe) / len(prof_mediapipe) * 1000.0
                avg_dip = sum(prof_dip_cnn) / len(prof_dip_cnn) * 1000.0
                avg_feat = sum(prof_features) / len(prof_features) * 1000.0
                avg_dtw = sum(prof_dtw) / len(prof_dtw) * 1000.0
                avg_draw = sum(prof_draw) / len(prof_draw) * 1000.0
                avg_tot = sum(prof_total) / len(prof_total) * 1000.0
                fps_prof = 1000.0 / avg_tot if avg_tot > 0 else 0.0

                px0 = max(10, w_f - 230)
                py0 = 50
                cv2.rectangle(display_frame, (px0, py0), (w_f - 10, py0 + 155), (15, 15, 15), -1)
                cv2.rectangle(display_frame, (px0, py0), (w_f - 10, py0 + 155), (80, 80, 80), 1)
                lines_prof = [
                    "PROFILE (60f avg)",
                    f"FPS:       {fps_prof:5.1f}",
                    f"mediapipe: {avg_mp:5.2f} ms",
                    f"dip_cnn:   {avg_dip:5.2f} ms",
                    f"features:  {avg_feat:5.2f} ms",
                    f"dtw:       {avg_dtw:5.2f} ms",
                    f"draw:      {avg_draw:5.2f} ms",
                    f"total:     {avg_tot:5.2f} ms",
                ]
                for li, ltext in enumerate(lines_prof):
                    col = (0, 255, 255) if li == 0 else ((0, 255, 0) if li == 1 else (220, 220, 220))
                    cv2.putText(display_frame, ltext, (px0 + 8, py0 + 16 + li * 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, lineType=cv2.LINE_AA)

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
                    sentence_keys.clear()
                    hud_toast = _make_toast("cleared", [], None, curr_time, 1.5)
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
                elif key in [ord('l'), ord('L')]:
                    ui_lang = "vi" if ui_lang == "en" else "en"
                    print(f"[UI] Ngôn ngữ giao diện + giọng nói: {ui_lang.upper()}")
                elif key in [ord('u'), ord('U')]:
                    ui_mode = "debug" if ui_mode == "demo" else "demo"
                    print(f"[UI] Chuyển sang giao diện: {ui_mode}")
                elif key in [ord('t'), ord('T')]:
                    _muted_now = speech_service.toggle_muted()
                    print(f"[UI] Loa: {'TẮT' if _muted_now else 'BẬT'}")
                elif key in [ord('h'), ord('H')]:
                    show_help = not show_help
                elif key in [ord('j'), ord('J')]:
                    # SPEC 08: bật/tắt chế độ KÝ NHANH lúc chạy. Dựng lại bộ phân đoạn (tham số khác) và bỏ dở stroke đang ghi.
                    fast_on = not fast_on
                    segmenter = make_segmenter(CONFIG, fc_cfg if fast_on else None)
                    has_predicted_pending = False
                    stroke_angles_history.clear()
                    stroke_normals_history.clear()
                    stroke_ext_history.clear()
                    print(f"[UI] Chế độ KÝ NHANH: {'BẬT' if fast_on else 'TẮT'}"
                          + (f" | từ nhiều pha (không chốt sớm): {multi_phase_words or 'không có'}" if fast_on else ""))
                elif enroller is not None and key in [ord('y'), ord('Y'), ord('2'), ord('n'), ord('N'), ord('z'), ord('Z')]:
                    if key in [ord('n'), ord('N')]:
                        enroller.clear()
                        banner_expire = 0.0
                    elif key in [ord('z'), ord('Z')]:
                        _undone = enroller.undo()
                        banner_title = f"UNDO: {_undone}" if _undone else "NOTHING TO UNDO"
                        banner_subtitle = f"Session samples: {enroller.count()}"
                        banner_color = (150, 150, 150)
                        banner_expire = curr_time + 1.5
                        print(f"[XacNhan] Hoàn tác mẫu '{_undone}'." if _undone else "[XacNhan] Không có mẫu nào để hoàn tác.")
                    else:
                        _picked = enroller.confirm(1 if key == ord('2') else 0, curr_time)
                        if _picked:
                            _pen = get_asl_phrase(_picked, lang="en", default=_picked)
                            sentence_words.append(_pen)
                            sentence_keys.append(_picked)
                            sentence_finalized = False
                            banner_title = f"SAVED: {_pen}"
                            banner_subtitle = f"Session samples: {enroller.count()} ({enroller.summary()})"
                            banner_color = (0, 255, 100)
                            banner_expire = curr_time + 2.0
                            print(f"[XacNhan] Đã thêm mẫu '{_picked}' vào bank phiên (tổng {enroller.count()}: {enroller.summary()}).")
                        else:
                            print("[XacNhan] Không có câu hỏi đang chờ, đã hết hạn, hoặc từ này đã đủ số mẫu tự thêm.")
                elif key in [ord('k'), ord('K')]:
                    # SPEC 08B: bật/tắt CẮT DẪN ĐẦU (lead-trim) lúc chạy.
                    lead_trim_on = not lead_trim_on
                    lt_cfg["enabled"] = lead_trim_on
                    banner_title = f"LEAD TRIM: {'ON' if lead_trim_on else 'OFF'}"
                    banner_subtitle = f"Adaptive lead trimming {'enabled' if lead_trim_on else 'disabled'}"
                    banner_color = (0, 255, 200) if lead_trim_on else (150, 150, 150)
                    banner_expire = curr_time + 1.5
                    hud_toast = _make_toast("lead_trim", [], None, curr_time, 1.5)
                    print(f"[UI] Cắt phần dẫn đầu (lead-trim): {'BẬT' if lead_trim_on else 'TẮT'}")

            if max_frames is not None and processed_frames >= max_frames:
                if profile:
                    _t_frame_end = time.perf_counter()
                    _t_total = _t_frame_end - _t_frame_start
                    prof_mediapipe.append(_t_mp)
                    prof_dip_cnn.append(_t_dip)
                    prof_features.append(_t_feat)
                    prof_dtw.append(_t_dtw)
                    prof_draw.append(_t_draw)
                    prof_total.append(_t_total)
                break

            if profile:
                _t_frame_end = time.perf_counter()
                _t_total = _t_frame_end - _t_frame_start
                prof_mediapipe.append(_t_mp)
                prof_dip_cnn.append(_t_dip)
                prof_features.append(_t_feat)
                prof_dtw.append(_t_dtw)
                prof_draw.append(_t_draw)
                prof_total.append(_t_total)

                if frame_counter % 60 == 0 and len(prof_total) > 0:
                    avg_mp = sum(prof_mediapipe) / len(prof_mediapipe) * 1000.0
                    avg_dip = sum(prof_dip_cnn) / len(prof_dip_cnn) * 1000.0
                    avg_feat = sum(prof_features) / len(prof_features) * 1000.0
                    avg_dtw = sum(prof_dtw) / len(prof_dtw) * 1000.0
                    avg_draw = sum(prof_draw) / len(prof_draw) * 1000.0
                    avg_tot = sum(prof_total) / len(prof_total) * 1000.0
                    fps_prof = 1000.0 / avg_tot if avg_tot > 0 else 0.0
                    print(f"[PROFILE] Frame {frame_counter:4d} (60f avg) | FPS: {fps_prof:5.1f}")
                    print(f"  mediapipe: {avg_mp:6.2f} ms")
                    print(f"  dip_cnn:   {avg_dip:6.2f} ms")
                    print(f"  features:  {avg_feat:6.2f} ms")
                    print(f"  dtw:       {avg_dtw:6.2f} ms")
                    print(f"  draw:      {avg_draw:6.2f} ms")
                    print(f"  total:     {avg_tot:6.2f} ms")

    finally:
        speech_service.shutdown()
        if cam_monitor is not None:
            _cam_sum = cam_monitor.session_summary()
            if _cam_sum:
                print(f"[Camera] Cả phiên: {_cam_sum}")
                print("[Camera] Gợi ý: nếu 'trùng y hệt' ≈ 0% và 'khung mới thật' ≈ 30/s thì --dedupe-frames/--time-norm gần như không đổi gì.")
        if enroller is not None and enroller.count() > 0:
            try:
                _is_production_bank = os.path.abspath(templates_file) == os.path.abspath(
                    CONFIG.get("templates_path", "webcam_templates.json"))
                if _is_production_bank:
                    # AN TOAN: khong bao gio tu dong ghi/gop vao bank san xuat chinh.
                    # Chi khi chay voi --templates profiles/<ten>.json (ho so ca nhan) moi tich luy duoc.
                    _enr_path = os.path.join("profiles", f"xacnhan_{time.strftime('%Y%m%d_%H%M%S')}.json")
                    os.makedirs("profiles", exist_ok=True)
                    _n = enroller.save(_enr_path)
                    print(f"[XacNhan] Cả phiên: thêm {_n} mẫu ({enroller.summary()}) -> {_enr_path} "
                          f"(KHÔNG gộp vào bank sản xuất; muốn tích lũy qua nhiều lần dùng: chạy lại với "
                          f"--templates profiles/<tên hồ sơ>.json)")
                else:
                    _n = enroller.merge_into_file(templates_file)
                    print(f"[XacNhan] Cả phiên: đã cộng dồn {_n} mẫu ({enroller.summary()}) vào hồ sơ "
                          f"'{templates_file}' — lần dùng sau sẽ tự nhận tốt hơn, không cần lặp lại.")
            except Exception as _enr_err:
                print(f"[XacNhan] Không lưu được mẫu phiên: {_enr_err}")
        if session_file is not None:
            try:
                session_file.close()
                print(f"[SessionLogger] ✓ Đã lưu và đóng file log phiên: '{log_session_path}'")
            except Exception:
                pass
        if cap is not None:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()
        if lead_trim_history:
            taus = np.array(lead_trim_history)
            tau_med = float(np.median(taus))
            tau_p90 = float(np.percentile(taus, 90))
            tau_pos_rate = float(np.mean(taus > 0.0)) * 100.0
            print(f"\n[LeadTrim] Thống kê cắt dẫn đầu phiên ({len(taus)} từ đã chốt):")
            print(f"  - Trung vị tau: {tau_med:.3f}s | p90 tau: {tau_p90:.3f}s | Tỉ lệ tau > 0: {tau_pos_rate:.1f}%")
        print("-> Đã đóng chương trình nhận diện ghép câu an toàn.")

    if profile and len(prof_total) > 0:
        avg_mp = sum(prof_mediapipe) / len(prof_mediapipe) * 1000.0
        avg_dip = sum(prof_dip_cnn) / len(prof_dip_cnn) * 1000.0
        avg_feat = sum(prof_features) / len(prof_features) * 1000.0
        avg_dtw = sum(prof_dtw) / len(prof_dtw) * 1000.0
        avg_draw = sum(prof_draw) / len(prof_draw) * 1000.0
        avg_tot = sum(prof_total) / len(prof_total) * 1000.0
        fps_prof = 1000.0 / avg_tot if avg_tot > 0 else 0.0
        return {
            "mediapipe_ms": avg_mp,
            "dip_cnn_ms": avg_dip,
            "features_ms": avg_feat,
            "dtw_ms": avg_dtw,
            "draw_ms": avg_draw,
            "total_ms": avg_tot,
            "fps": fps_prof
        }
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition")
    parser.add_argument("--cam", type=int, default=None, help="Camera index (0=built-in webcam, 1, 2=external/iPhone). Mặc định dùng camera_source từ CONFIG.")
    parser.add_argument("--templates", type=str, default=None, help="Đường dẫn file template JSON (ví dụ: webcam_templates.json). Mặc định dùng templates_path từ CONFIG.")
    parser.add_argument("--margin", type=float, default=0.12, help="Relative contrast margin threshold (default: 0.12 = 12%%)")
    parser.add_argument("--thresh", type=float, default=None, help="Override DTW distance threshold (default: None, uses template threshold 1.25)")
    parser.add_argument("--global-weights", action="store_true", help="Bật chế độ dùng chung 1 công thức CONFIG cho mọi từ (tắt Per-Sign weights ASL-LEX)")
    parser.add_argument("--profile", action="store_true", help="Bật đo thời gian chi tiết theo tầng (ms) và hiển thị HUD / console mỗi 60 frame")
    parser.add_argument("--with-cnn", action="store_true", help="Bật lại suy luận DIP + CNN dáng tay tĩnh (mặc định tắt theo CONFIG)")
    parser.add_argument("--detect-scale", type=float, default=None, help="Tỉ lệ scale frame đưa vào MediaPipe detector (mặc định 1.0 từ CONFIG)")
    parser.add_argument("--detector-workers", type=int, default=None, help="Số luồng làm việc song song cho MediaPipe (mặc định None = tự chọn min(3, max(1, CPU//4))).")
    parser.add_argument("--video", type=str, default=None, help="Đường dẫn file video đầu vào thay vì webcam")
    parser.add_argument("--log-session", "-l", type=str, default=None, help="Đường dẫn file CSV lưu log kết quả từng cử chỉ live (phục vụ test Phiên 3)")
    parser.add_argument("--headless", action="store_true", default=False, help="Chạy không mở cửa sổ đồ họa OpenCV (dùng cho đo benchmark/test).")
    parser.add_argument("--max-frames", type=int, default=None, help="Dừng sau số khung hình nhất định (dùng cho benchmark/test).")
    parser.add_argument("--threshold-mode", type=str, choices=["legacy", "scale_k"], default=None, help="Chế độ ngưỡng: 'scale_k' (ngưỡng theo độ rộng nội lớp) hoặc 'legacy' (cố định).")
    parser.add_argument("--threshold-k", type=float, default=None, help="Hệ số k cho ngưỡng chấp nhận (k * độ_rộng, mặc định 1.6)")
    parser.add_argument("--uncertainty-k", type=float, default=None, help="Hệ số k cho ranh giới từ chối (uncertainty_k * độ_rộng, mặc định 2.2)")
    parser.add_argument("--time-norm", action=argparse.BooleanOptionalAction, default=None,
                        help="Bật/tắt chuẩn hóa vận tốc + Swipe-Left theo thời gian (mặc định lấy CONFIG['time_norm']['enabled'] = TẮT).")
    parser.add_argument("--dedupe-frames", action=argparse.BooleanOptionalAction, default=None,
                        help="Bật/tắt bỏ khung trùng trước khi dò tay (mặc định lấy CONFIG['dedupe_frames'] = TẮT).")
    parser.add_argument("--confirm-enroll", action=argparse.BooleanOptionalAction, default=True,
                        help="Chế độ hỏi xác nhận: cử chỉ không chắc/bị từ chối được hỏi lại, mẫu được xác nhận "
                             "thêm vào bank (phím y/2/n/z). MẶC ĐỊNH BẬT (đã đo: đây là đòn bẩy lớn nhất cho "
                             "người lạ, xem eval_enroll.py) — dùng --no-confirm-enroll để tắt.")
    parser.add_argument("--width-blend", type=float, default=None, help="Pha trộn độ rộng chuẩn hóa ngưỡng 0..1 (0 = độ rộng riêng từng từ, 1 = thang chung). Mặc định lấy CONFIG['width_blend'] (0.0).")
    parser.add_argument("--ui", type=str, choices=["demo", "debug"], default=None, help="Giao diện: 'demo' (sạch, cho người dùng cuối) hoặc 'debug' (bảng đặc trưng cũ). Mặc định lấy CONFIG['ui_mode'].")
    parser.add_argument("--lang", type=str, choices=["en", "vi"], default=None, help="Ngôn ngữ giao diện + giọng nói: 'en' hoặc 'vi'. Mặc định lấy CONFIG['ui_lang'].")
    parser.add_argument("--fast-chain", action=argparse.BooleanOptionalAction, default=None,
                        help="SPEC 08: bật/tắt chế độ KÝ NHANH (nghỉ ngắn ~0.2s là đủ, chốt sớm khi đủ chắc). "
                             "Mặc định lấy CONFIG['fast_chain']['enabled'] (BẬT). Dùng --no-fast-chain để về chế độ cũ. Phím [j] đổi lúc chạy.")
    parser.add_argument("--transition-lockout", type=float, default=None,
                        help="SPEC 08B: Thời gian khóa chuyển tiếp (giây) sau khi chốt cử chỉ trong fast_chain. "
                             "Mặc định None lấy từ CONFIG['fast_chain']['transition_lockout_sec'] (0.0).")
    parser.add_argument("--lead-trim", action=argparse.BooleanOptionalAction, default=None,
                        help="SPEC 08B: Bật/tắt cắt thích ứng phần dẫn đầu (adaptive lead trimming). "
                             "Mặc định None lấy từ CONFIG['fast_chain']['lead_trim']['enabled'] (BẬT). Dùng --no-lead-trim để tắt. Phím [k] đổi lúc chạy.")
    parser.add_argument("--dl", action=argparse.BooleanOptionalAction, default=None,
                        help="Bật/tắt tầng DL dự phòng (xem dl_fusion.py, docs/BAO_CAO_DEEP_LEARNING_HYBRID.md). "
                             "Mặc định None = TỰ ĐỘNG bật nếu tìm thấy 'best_gesture_model_none.pth' (hoặc file chỉ "
                             "định qua --dl-model), tắt lặng lẽ nếu không có file. Dùng --no-dl để tắt hẳn dù có file. "
                             "Đã đo: gate an toàn (Hybrid không tệ hơn max(DTW,DL) ở đa số từ, xem results_hybrid_*.json), "
                             "và nested CV cho 96.6% trung bình 4 người lạ thật, không rò rỉ ngưỡng.")
    parser.add_argument("--dl-model", type=str, default=None,
                        help="Đường dẫn checkpoint .pth tùy chỉnh cho tầng DL (mặc định 'best_gesture_model_none.pth').")
    parser.add_argument("--dl-conf-threshold", type=float, default=None,
                        help="Ngưỡng tin cậy softmax tối thiểu để tầng DL được nhận khi DTW không ACCEPTED. "
                             "Mặc định 0.38 (chọn giữa miền ổn định qua nested CV 4 người lạ, xem dl_fusion.py).")
    args, _ = parser.parse_known_args()

    eff_dl_model_path = None
    if args.dl is not False:
        candidate = args.dl_model if args.dl_model else "best_gesture_model_none.pth"
        if os.path.exists(candidate):
            eff_dl_model_path = candidate
        elif args.dl is True:
            print(f"[!] --dl được bật nhưng không tìm thấy '{candidate}' -> chạy KHÔNG có tầng DL.")

    selected_cam = args.cam if args.cam is not None else CONFIG.get("camera_source", 0)
    selected_templates = args.templates if args.templates is not None else CONFIG.get("templates_path", "dynamic_templates.json")
    sw_override = "" if args.global_weights else CONFIG.get("sign_weights_path", "")
    enable_cnn = True if args.with_cnn else None
    if args.time_norm is not None:
        CONFIG["time_norm"]["enabled"] = bool(args.time_norm)
    run_sentence_builder(
        camera_index=selected_cam,
        headless=args.headless,
        max_frames=args.max_frames,
        margin_threshold=args.margin,
        threshold_override=args.thresh,
        templates_path=selected_templates,
        sign_weights_path=sw_override,
        video_path=args.video,
        profile=args.profile,
        enable_static_shape_cnn=enable_cnn,
        detect_scale=args.detect_scale,
        log_session_path=args.log_session,
        detector_workers=args.detector_workers,
        threshold_mode=args.threshold_mode,
        threshold_k=args.threshold_k,
        uncertainty_k=args.uncertainty_k,
        ui_mode=args.ui,
        ui_lang=args.lang,
        fast_chain=args.fast_chain,
        transition_lockout=args.transition_lockout,
        lead_trim=args.lead_trim,
        width_blend=args.width_blend,
        dedupe_frames=args.dedupe_frames,
        confirm_enroll=args.confirm_enroll,
        dl_model_path=eff_dl_model_path,
        dl_conf_threshold=args.dl_conf_threshold,
    )
