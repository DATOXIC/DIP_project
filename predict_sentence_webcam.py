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
from ui_render import DemoHUD, HudState, HudToast, draw_soft_trail, strip_diacritics
import ui_side_panel
from sentence_composer import compose as compose_sentence
from camera_health import CameraHealthMonitor
from lead_trim import select_best_lead_trim
from session_enroll import SessionEnroller, TeachPlan

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
    # --- Bộ lọc ứng viên theo DÁNG TAY (handshape_gate.py, nhánh landmark 3D chuẩn hóa) ---
    # Loại ứng viên DTW mâu thuẫn rõ với dáng tay đo ở đầu/cuối cử chỉ; chỉ can thiệp khi DTW KHÔNG ACCEPTED.
    # Mặc định TẮT. Số đo: results/handshape_gate_eval.json (eval_handshape_gate.py). Bật: --handshape-gate.
    "handshape_gate": {"enabled": False, "model": "letter_landmark_model_image.pt", "table": "handshape_gate_table.json",
                       "conf": 0.6, "compat": 0.15},
    "max_sentence_words": 8,            # Chốt câu tự động khi đủ ngần này từ (0 = tắt); từ kế tiếp mở câu mới
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

    # --- SPEC 14: sinh câu tự nhiên từ chuỗi từ (sentence_composer.py, luật offline, KHÔNG ảnh hưởng nhận dạng).
    #     Khi chốt câu (hạ tay > 1.8s): hiện thêm dòng câu tự nhiên dưới dòng gloss gốc, và đọc câu đó nếu nó KHÁC cách nối thô.
    #     Tắt: --no-composer (hành vi y hệt cũ). ---
    "composer": {
        "enabled": True,          # sinh câu tự nhiên khi chốt câu (chỉ hiển thị/giọng nói)
        "speak_sentence": True,   # chốt câu xong thì đọc câu tự nhiên nếu nó KHÁC cách nối thô
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

from spotter_gate import OneShotLock, Spotter, make_spotter   # KHÓA MỘT LẦN (--one-shot), import nhẹ

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
    teach_words: Optional[List[str]] = None,
    teach_per_word: int = 3,
    enroll_save_path: Optional[str] = None,
    dl_model_path: Optional[str] = None,
    dl_conf_threshold: Optional[float] = None,
    one_shot: bool = False,
    lock_min_gap: float = 1.2,
    lock_soft_prob: float = 0.25,
    spotter_kind: str = "rule",
    spotter_model_path: str = "spotter_model.pth",
    video_clock: bool = False,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    events_path: Optional[str] = None,
    mute_speech: bool = False,
    ready_pose: Optional[bool] = None,
    hold_on_hand_loss: bool = False,
    force_stroke: bool = False,
    trace_path: Optional[str] = None,
    seg_tune: Optional[Dict[str, Any]] = None,
    motion_mode: str = "wrist",
    composer: Optional[bool] = None,
):
    """
    Vòng lặp chính xử lý nhận diện cử chỉ động & ghép câu thời gian thực (ASL/VSL).
    start_sec / end_sec: chỉ áp dụng khi có video_path; nhảy tới start_sec và dừng khi quá end_sec (giây trong video).
                         Ở chế độ video, MỌI mốc thời gian của bộ phân đoạn dùng đồng hồ của VIDEO (chỉ số khung / fps),
                         không dùng đồng hồ máy, nên kết quả không phụ thuộc tốc độ xử lý.
    events_path: ghi mỗi cử chỉ (ACCEPTED/UNCERTAIN/REJECTED/OUT_OF_DURATION) thành một dòng JSON (JSONL), kèm nguồn DTW/DL.
    mute_speech: tắt hẳn giọng nói (dùng khi chạy nền, ví dụ bản web).
    ready_pose: False = KHÔNG đòi tay đứng yên trước khi bắt đầu ghi (người ký nhanh, đưa tay lên là ký ngay). None/True = như cũ.
    hold_on_hand_loss: True = khi tay biến mất giữa lúc đang ghi (ra khỏi khung, nhòe do quá nhanh), giữ nguyên tư thế cuối thêm
                       ~0.3 s để bộ phân đoạn kịp chốt cử chỉ thay vì xóa cả bộ đệm. Mặc định tắt.
    force_stroke: True (chỉ với --video) = KHÔNG dùng bộ phân đoạn để tìm cử chỉ: toàn bộ đoạn [start_sec, end_sec] (các khung có tay) được coi là
                  MỘT cử chỉ, bỏ qua cổng thời lượng/chuyển tiếp/lead-trim. Dùng để lấy mẫu từ đoạn người dùng tự xác định.
    seg_tune: ghi đè riêng các ngưỡng chốt cử chỉ (None = giữ cấu hình): pause_sec (tay khựng bao lâu thì vào chờ chốt), grace_sec (chờ thêm
              trước khi chốt hẳn), v_stop / v_start (ngưỡng vận tốc coi là đứng yên / bắt đầu ký, đơn vị lòng bàn tay mỗi khung),
              early_commit (False = tắt chốt sớm khi dự đoán tạm đã chắc). Áp dụng cho cả chế độ chuỗi nhanh lẫn cơ bản.
    motion_mode: "wrist" (mặc định) = vận tốc phân đoạn chỉ từ cổ tay; "hand" = lấy max(cổ tay, chuyển động của các khớp so với cổ tay), để cử chỉ
                 nhẹ (ngón cử động, cổ tay gần đứng yên) vẫn được nhận là đang ký.
    trace_path: ghi mỗi khung đã xử lý một dòng JSON (thời gian video, có tay không, vận tốc, trạng thái FSM, 21 mốc bàn tay) để bản web
                duyệt từng khung và vẽ khung xương.
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
    one_shot: KHÓA MỘT LẦN (spotter_gate.py): mỗi khoảng "đang ký" chỉ commit tối đa MỘT từ. Mặc định False = hành vi cũ.
    spotter_kind: loại spotter cấp tín hiệu "đang ký" cho one_shot: "rule" (luật vận tốc) hoặc "dl" (CNN nhỏ, spotter_dl.py;
                  chế độ NGHIÊM: stroke ngoài khoảng "đang ký" bị chặn => loại từ vu vơ). Bỏ qua khi one_shot=False.
    spotter_model_path: checkpoint cho spotter "dl" (train_spotter.py --save).
    video_clock: khi chạy từ --video, dùng ĐỒNG HỒ CỦA VIDEO (khung n tại n/fps) thay cho giờ máy làm mốc thời gian khung, để các ngưỡng
                 theo giây và đặc trưng vận tốc đúng như lúc thu dù xử lý nhanh hơn thời gian thật. Mặc định False = như cũ.
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

    # Nhật ký sự kiện có cấu trúc (JSONL): mỗi cử chỉ hoàn thành một dòng, kèm nguồn quyết định DTW/DL.
    events_file = None
    eof_clip = [float("inf")]   # giây video thật cuối cùng; cắt t_end của cử chỉ chốt nhờ đoạn đệm cuối video
    if events_path is not None:
        _ev_dir = os.path.dirname(os.path.abspath(events_path))
        if _ev_dir and not os.path.exists(_ev_dir):
            os.makedirs(_ev_dir, exist_ok=True)
        events_file = open(events_path, mode="w", encoding="utf-8")

    def _emit_event(event: str, mr: Optional[MatchResult], stroke_frames, duration_sec: float, commit_mode: str,
                    records: Optional[List[Dict[str, Any]]] = None) -> None:
        if events_file is None:
            return
        widths = getattr(dtw_engine, "intra_widths", {}) or {}
        cands = []
        if mr is not None:
            for c in mr.candidates:
                cands.append({"word": c.word_key, "distance": round(float(c.distance), 4),
                              "quality": round(float(c.match_quality), 1), "r_ratio": round(float(c.r_ratio), 3),
                              "width": (round(float(widths[c.word_key]), 4) if c.word_key in widths else None)})
        rec = {
            "event": event,
            "word": (mr.top_word if (mr is not None and event == "ACCEPTED") else (cands[0]["word"] if cands else "")),
            "t_start": round(float(stroke_frames[0].timestamp), 3),
            "t_end": round(min(float(stroke_frames[-1].timestamp), eof_clip[0]), 3),
            "duration": round(float(duration_sec), 3),
            "commit_mode": commit_mode,
            "status": mr.status if mr is not None else "",
            "distance": round(float(mr.best_distance), 4) if mr is not None else None,
            "quality": round(float(mr.match_quality), 1) if mr is not None else None,
            "r_ratio": round(float(mr.r_ratio), 4) if mr is not None else None,
            "source": mr.result_source if mr is not None else "",
            "dtw_status": mr.dtw_status if mr is not None else "",
            "dtw_word": mr.dtw_word if mr is not None else None,
            "dl_word": mr.dl_word if mr is not None else None,
            "dl_confidence": (round(float(mr.dl_confidence), 3) if (mr is not None and mr.dl_confidence >= 0.0) else None),
            "fusion_note": mr.fusion_note if mr is not None else "",
            "candidates": cands,
            "forced": bool(force_stroke),
        }
        # Mẫu thô để lưu vào hồ sơ (đúng đặc trưng 24D mà bộ so khớp nhìn thấy) + xác suất DL của MỌI từ (hỏi DL cả khi DTW đã ACCEPTED).
        if records is not None:
            try:
                s_traj, s_shapes, s_body, _s_dom = build_segment_features(records)
            except Exception:
                s_traj = None
            if s_traj is not None:
                s_arr = np.nan_to_num(np.asarray(s_traj, dtype=np.float32))
                rec["sample"] = {"traj_features": np.round(s_arr, 6).tolist(), "hand_shapes": list(s_shapes),
                                 "handedness": handedness_label, "body_anchor_valid": bool(s_body),
                                 "duration_sec": round(float(duration_sec), 3)}
                try:
                    _bundle = dtw_engine._get_dl_bundle()
                    if _bundle is not None:
                        _pred = dl_fusion.predict(_bundle, s_arr)
                        if _pred is not None:
                            _probs = {w: round(float(pp), 4) for w, pp in zip(_bundle["words"], _pred[2])}
                            rec["dl_probs"] = dict(sorted(_probs.items(), key=lambda kv: -kv[1])[:6])
                            rec["dl_top"] = _pred[0]
                except Exception:
                    pass
        events_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        events_file.flush()

    trace_file = None
    if trace_path is not None:
        _tr_dir = os.path.dirname(os.path.abspath(trace_path))
        if _tr_dir and not os.path.exists(_tr_dir):
            os.makedirs(_tr_dir, exist_ok=True)
        trace_file = open(trace_path, mode="w", encoding="utf-8")

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

    enroller: Optional[SessionEnroller] = SessionEnroller(dtw_engine) if (confirm_enroll or teach_words) else None
    # DẠY TỪ (hiệu chỉnh nhanh, --day-tu): mỗi cử chỉ hoàn chỉnh được lưu làm mẫu của từ đang dạy thay vì đem đi nhận diện;
    # dạy xong thì tự chuyển sang nhận diện bình thường, bank đã có từ mới ngay (không train).
    teach_plan: Optional[TeachPlan] = TeachPlan(teach_words, teach_per_word) if teach_words else None
    if teach_plan is not None:
        print(f"✓ Chế độ DẠY TỪ: {', '.join(teach_plan.words)} — mỗi từ {teach_plan.per_word} mẫu. Ký từ đang hiển thị, "
              f"nghỉ giữa các lần ký; [z]=hoàn tác mẫu vừa lưu.")
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
    if seg_tune and seg_tune.get("early_commit") is False:
        fc_cfg["early_commit"] = False   # early_commit_ok() đọc khóa này mỗi lần quyết định chốt sớm
        print("ℹ️ [Tinh chỉnh] Đã TẮT chốt sớm: cử chỉ chỉ chốt khi hết thời gian chờ.")
    if seg_tune:
        print(f"ℹ️ [Tinh chỉnh] Ghi đè bộ phân đoạn: {seg_tune}")

    def _new_segmenter():
        seg = make_segmenter(CONFIG, fc_cfg if fast_on else None)
        if ready_pose is False:
            seg.require_ready_pose = False
        if seg_tune:
            if seg_tune.get("pause_sec") is not None:
                seg.pause_sec = float(seg_tune["pause_sec"])
            if seg_tune.get("grace_sec") is not None:
                seg.grace_window_sec = float(seg_tune["grace_sec"])
            if seg_tune.get("v_stop") is not None:
                seg.v_stop_ratio = float(seg_tune["v_stop"])
            if seg_tune.get("v_start") is not None:
                seg.v_start_ratio = float(seg_tune["v_start"])
        if force_stroke:
            # Ghi liên tục từ khung có tay đầu tiên, không bao giờ "khựng", không Swipe: cử chỉ chỉ chốt khi hết đoạn người dùng chọn.
            seg.require_ready_pose = False
            seg.v_start_ratio = 0.0
            seg.v_stop_ratio = -1.0
            seg.max_stroke_frames = 10 ** 6
            seg.max_stroke_sec = None
            seg.swipe_cooldown_sec = 1e9
        return seg

    segmenter = _new_segmenter()
    early_pre: Optional[Tuple[Any, Any, bool, Any, float, int]] = None   # (traj, shapes, body_ok, match, tau, n_tried) của dự đoán tạm khi chốt sớm
    lead_trim_history: List[float] = []   # Lưu lại tau của các từ đã commit trong phiên để in thống kê lúc thoát
    _commit_mode = "normal"

    # KHÓA MỘT LẦN (--one-shot): mặc định TẮT (one_shot_lock = None) => mọi nhánh bên dưới bị bỏ qua, hành vi như cũ.
    one_shot_lock: Optional[OneShotLock] = None
    one_shot_spotter: Optional[Spotter] = None
    if one_shot:
        one_shot_spotter = make_spotter(spotter_kind, v_stop_ratio=float(segmenter.v_stop_ratio), model_path=spotter_model_path)
        one_shot_lock = OneShotLock(strict=(spotter_kind == "dl"), min_gap_sec=lock_min_gap, soft_prob=lock_soft_prob)
        print(f"[OneShot] BẬT khóa một lần (spotter={spotter_kind}"
              f"{', NGHIÊM: chặn stroke ngoài khoảng đang ký' if spotter_kind == 'dl' else ''}): mỗi khoảng ký chỉ chốt tối đa 1 từ.")

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
    # SPEC 14: bộ sinh câu tự nhiên. Kết quả được tính lại khi (chuỗi từ, ngôn ngữ) đổi, nên không cần reset ở từng chỗ sửa câu.
    composer_cfg: Dict[str, Any] = dict(CONFIG.get("composer", {}))
    composer_on = bool(composer_cfg.get("enabled", True)) if composer is None else bool(composer)
    composer_cache_key: Optional[Tuple[Tuple[str, ...], str]] = None
    composed_result: Optional[Any] = None

    def _phrase_for_composer(key: str, lang: str) -> str:
        return get_asl_phrase(key, lang=lang, default="")

    # Quản lý Banner phản hồi trực quan (R4 HUD)
    banner_title = ""
    banner_subtitle = ""
    banner_color = (0, 255, 0)
    banner_expire = 0.0
    if teach_plan is not None and enroller is not None and teach_plan.current(enroller) is not None:
        _tw0, _tk0, _tn0 = teach_plan.current(enroller)
        banner_title = f"TEACH: sign '{_tw0}' ({_tk0}/{_tn0})"
        banner_subtitle = "Sign it, then pause. [Z] undo"
        banner_color = (0, 200, 255)
        banner_expire = 1e12

    # Panel bên/dưới thay thế HUD đè lên camera + log sự kiện thay terminal (xem ui_side_panel.py)
    event_log = ui_side_panel.EventLog(maxlen=30)
    _last_banner_log_key: Optional[Tuple[str, float]] = None
    last_match_result: Optional[MatchResult] = None

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
    if mute_speech:
        speech_service.set_muted(True)

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

    # Bộ lọc dáng tay (mặc định tắt). Lỗi nạp => tắt lặng lẽ, KHÔNG làm hỏng pipeline.
    shape_gate = None
    _sg_cfg = CONFIG.get("handshape_gate", {})
    if _sg_cfg.get("enabled"):
        try:
            from handshape_gate import HandshapeGateRuntime
            shape_gate = HandshapeGateRuntime(_sg_cfg["model"], _sg_cfg["table"],
                                              conf_thresh=float(_sg_cfg.get("conf", 0.6)),
                                              compat_thresh=float(_sg_cfg.get("compat", 0.15)))
            print(f"[HandshapeGate] BẬT (model={_sg_cfg['model']}, bảng={_sg_cfg['table']})")
        except Exception as _sg_err:
            print(f"[HandshapeGate] Không nạp được, tắt: {_sg_err}")

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
    _vclk_n = 0
    _vclk_t0 = time.time()
    _vclk_fps = 30.0
    if video_path is not None:
        if str(video_path).startswith(("http://", "https://")):
            print(f"[Video] Nhận diện nguồn là URL trực tuyến: {video_path}")
            import requests, tempfile
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                print("[Video] Đang tải video về bộ đệm tạm...")
                resp = requests.get(video_path, headers=headers, stream=True, timeout=25)
                resp.raise_for_status()
                tmp_suffix = os.path.splitext(video_path.split("?")[0])[1] or ".mp4"
                tmp_file = os.path.join(tempfile.gettempdir(), f"vsl_input_video{tmp_suffix}")
                with open(tmp_file, "wb") as f_out:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f_out.write(chunk)
                print(f"[Video] Tải thành công! Đã nạp vào pipeline: {tmp_file}")
                video_path = tmp_file
            except Exception as e_dl:
                print(f"[!] Không thể tải video từ URL ({e_dl}). Thử mở trực tiếp qua OpenCV...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] KHÔNG THỂ mở video: {video_path}")
            cap = None
        else:
            print(f"[Video] KẾT NỐI THÀNH CÔNG video file: {video_path}")
            if video_clock:
                _vclk_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                print(f"[Video] Dùng ĐỒNG HỒ VIDEO ({_vclk_fps:.1f} fps) làm mốc thời gian khung (--video-clock). VIDEO_CLOCK_T0={_vclk_t0:.3f}")
    elif not headless:
        cap = open_camera(camera_index, prefer_mjpg=True)

    # Đồng hồ VIDEO: chỉ số khung / fps thay cho time.time(), để bộ phân đoạn (nghỉ, cổng thời lượng...) không phụ thuộc
    # tốc độ xử lý. Khung được thưa bớt về <= ~30 fps (video 60 fps lấy cách một khung) cho giống điều kiện webcam.
    vid_fps = 30.0
    vid_stride = 1
    vid_idx = 0
    vid_ts = 0.0
    vid_seek_sec = 0.0
    vid_total_sec = 0.0
    vid_progress_every = 15
    if video_path is not None and cap is not None:
        _fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        vid_fps = _fps_raw if (_fps_raw == _fps_raw and _fps_raw > 1.0) else 30.0
        vid_stride = max(1, int(round(vid_fps / 30.0)))
        _nframes = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        _dur = (_nframes / vid_fps) if _nframes > 0 else 0.0
        if start_sec is not None and start_sec > 0.0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * vid_fps))
        vid_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        vid_seek_sec = vid_idx / vid_fps
        vid_total_sec = float(end_sec) if end_sec is not None else _dur
        print(f"[Video] fps={vid_fps:.2f} stride={vid_stride} bắt đầu={vid_seek_sec:.2f}s "
              f"kết thúc={'hết video' if end_sec is None else f'{end_sec:.2f}s'} (tổng {_dur:.2f}s)")
        # Mọi đồng hồ nội bộ khởi tạo theo đồng hồ video (ban đầu là time.time() của đồng hồ máy).
        last_hand_seen_time = vid_seek_sec

    flip_cam = False
    if show_features is None:
        show_features = False
    prev_time = vid_seek_sec if (video_path is not None and cap is not None) else time.time()
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
    last_real_item = None
    last_hand_det = None
    loss_hold = 0
    eof_padded = False
    force_flush = False
    force_done = False
    prev_rel = None
    art_ema = 0.0
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
                    vid_ts = 0.0
                    if video_path is not None:
                        vid_ts = vid_idx / vid_fps
                        vid_idx += 1
                        for _skip in range(vid_stride - 1):
                            if cap.grab():
                                vid_idx += 1
                        if ret and end_sec is not None and vid_ts > end_sec:
                            ret = False
                    if not ret:
                        if video_path is not None and max_frames is not None and processed_frames < max_frames:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        if not ret:
                            cap_exhausted = True
                            ready_queue.extend(detector_pool.flush())

                if ret and frame is not None:
                    capture_ts = vid_ts if video_path is not None else time.time()
                    if video_clock and video_path is not None:
                        _vclk_n += 1
                        capture_ts = _vclk_t0 + _vclk_n / _vclk_fps
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
                    # Cuối video mà tay vẫn đang ở tư thế ký: đệm ~1 s "giữ nguyên tư thế cuối" để bộ phân đoạn chốt nốt cử chỉ
                    # dở dang (ngoài đời sau cử chỉ luôn có thời gian nghỉ; clip bị cắt sát thì không có).
                    if (video_path is not None and cap is not None and not eof_padded and last_real_item is not None
                            and (last_real_item[2].hand_landmarks or (force_stroke and last_hand_det is not None))):
                        eof_padded = True
                        _eseq, _emeta, _edet = last_real_item
                        if force_stroke and not _edet.hand_landmarks:
                            _edet = last_hand_det
                        eof_clip[0] = float(_emeta["capture_ts"])
                        _edt = 1.0 / max(vid_fps / vid_stride, 1.0)
                        if force_stroke:
                            force_flush = True
                        for _ek in range(1, (1 if force_stroke else int(round(1.0 / _edt))) + 1):
                            _em = dict(_emeta)
                            _em["capture_ts"] = float(_emeta["capture_ts"]) + _ek * _edt
                            ready_queue.append((_eseq, _em, _edet))
                        continue
                    break
                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                continue

            seq, meta_item, det_result = ready_queue.popleft()
            if det_result.hand_landmarks:
                last_hand_det = det_result
                loss_hold = 0
            elif (hold_on_hand_loss and last_hand_det is not None and loss_hold < 9
                  and segmenter.state in (KineticState.RECORDING, KineticState.PENDING_COMMIT)):
                # Tay biến mất giữa lúc đang ghi: coi như tay giữ nguyên tư thế cuối (giống đoạn đệm cuối video và
                # extend_still_tail) để bộ phân đoạn chốt bằng đường thường, đủ lịch sử đặc trưng, thay vì reset mất cử chỉ.
                det_result = last_hand_det
                loss_hold += 1
            last_real_item = (seq, meta_item, det_result)
            processed_frames += 1
            if events_path is not None and video_path is not None and processed_frames % vid_progress_every == 0:
                _span = max(vid_total_sec - vid_seek_sec, 1e-3)
                print(f"@@PROGRESS {min(1.0, max(0.0, (meta_item['capture_ts'] - vid_seek_sec) / _span)):.4f}", flush=True)
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
            art_v = None
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
                if motion_mode == "hand":
                    _rel = lm_3d[:, :2] - lm_3d[0:1, :2]
                    if prev_rel is not None:
                        _raw_art = float(np.mean(np.linalg.norm(_rel - prev_rel, axis=1))) / max(palm_len, 15.0)
                        art_ema = 0.5 * art_ema + 0.5 * _raw_art   # làm trơn nhẹ: mốc ngón chưa qua bộ lọc nên nhiễu hơn cổ tay
                        art_v = art_ema
                    prev_rel = _rel
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

                if hand_lost_frames >= 4 and not force_stroke:
                    prev_rel = None
                    art_ema = 0.0
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
                    event_log.push(f"Chốt câu: \"{final_sentence}\"", ui_side_panel.COL_OK)
                    if composer_on:
                        # SPEC 14: câu tự nhiên song hành với gloss gốc (gloss không bị đổi/che).
                        try:
                            composed_result = compose_sentence(sentence_keys, ui_lang, phrase_fn=_phrase_for_composer)
                            composer_cache_key = (tuple(sentence_keys), ui_lang)
                            if composed_result.changed:
                                print(f"[Composer] ({composed_result.rule}) {composed_result.text}")
                                event_log.push(f"Cau tu nhien: {strip_diacritics(composed_result.text)}", ui_side_panel.COL_OK)
                                if composer_cfg.get("speak_sentence", True):
                                    speech_service.speak_word("__SENTENCE__", ui_lang, composed_result.text)
                        except Exception as composer_err:
                            composed_result = None
                            print(f"[Composer] Lỗi sinh câu ({composer_err}) — bỏ qua, vẫn hiển thị chuỗi từ như cũ.")

            if trace_file is not None and not hand_found:
                trace_file.write(json.dumps({"t": round(float(curr_time), 3), "h": 0, "s": segmenter.state.name[:3]}) + "\n")
            if hand_found:
                # 3. Cập nhật Bộ phân đoạn động học (Tầng 2)
                seg_state, completed_stroke, is_swipe, v_norm = segmenter.update(
                    wrist_x=wrist_x,
                    wrist_y=wrist_y,
                    palm_len=palm_len,
                    hand_shape=hand_shape,
                    timestamp=curr_time,
                    extra_motion=art_v
                )
                cam_last_v_norm = v_norm
                if one_shot_lock is not None:
                    # KHÓA MỘT LẦN: spotter nhận đúng v_norm của segmenter, lock ghi span theo thời gian.
                    # landmarks chỉ hợp lệ ở khung MediaPipe thấy tay thật (hand_lost_frames == 0); khung LK-fallback => None.
                    _os_gate = one_shot_spotter.update(
                        curr_time, True, float(v_norm), landmarks if hand_lost_frames == 0 else None, (w_f, h_f))
                    one_shot_lock.observe(curr_time, _os_gate, getattr(one_shot_spotter, "last_prob", None))
                if force_stroke and force_flush and not force_done and seg_state == KineticState.RECORDING:
                    # Hết đoạn người dùng chọn: chốt toàn bộ bộ đệm thành MỘT cử chỉ.
                    force_done = True
                    if len(segmenter.buffer) >= 3:
                        completed_stroke = list(segmenter.buffer)
                        segmenter.buffer.clear()
                        segmenter.state = KineticState.IDLE
                        seg_state = segmenter.state
                if trace_file is not None:
                    _lm_xy = None
                    if det_result.hand_landmarks:
                        _lm_xy = [[round(float(pt.x), 3), round(float(pt.y), 3)] for pt in det_result.hand_landmarks[0]]
                    trace_file.write(json.dumps({"t": round(float(curr_time), 3), "h": 1, "v": round(float(v_norm), 3),
                                                 "s": seg_state.name[:3], "lm": _lm_xy}) + "\n")
                if seg_state == KineticState.RECORDING:
                    recording_streak += 1
                    has_predicted_pending = False
                else:
                    recording_streak = 0
                if seg_state in (KineticState.IDLE, KineticState.READY):
                    has_predicted_pending = False

                # 3.5. Dự đoán tạm thời khi PENDING_COMMIT (Nhiệm vụ E)
                if (seg_state == KineticState.PENDING_COMMIT and not has_predicted_pending
                        and not (teach_plan is not None and not teach_plan.finished(enroller))):
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
                                    last_match_result = t_match
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
                        event_log.push(f"Quét tay: Đã xóa từ '{removed}'", ui_side_panel.COL_WARN)
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
                    if force_stroke or is_stroke_duration_valid(stroke_duration):
                        if (not force_stroke) and is_movement_epenthesis(epi_frames):
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
                            if teach_plan is not None and not teach_plan.finished(enroller):
                                # DẠY TỪ: lưu cử chỉ này làm mẫu của từ đang dạy (không nhận diện, không vào câu).
                                _tw, _tk, _tn = teach_plan.current(enroller)
                                _t_traj, _t_shp, _t_body, _ = build_segment_features(seg_records if seg_records is not None else raw_records)
                                if _t_traj is not None:
                                    _tk = enroller.add_taught(_tw, _t_traj, _t_shp, handedness_label, _t_body, stroke_duration)
                                    _cur = teach_plan.current(enroller)
                                    if _cur is not None:
                                        banner_title = f"SAVED {_tw} ({_tk}/{_tn})" if _cur[0] == _tw else f"{_tw} done! Next: '{_cur[0]}'"
                                        banner_subtitle = f"Sign '{_cur[0]}' ({_cur[1]}/{_cur[2]}). [Z] undo"
                                    else:
                                        banner_title = "TEACHING DONE"
                                        banner_subtitle = "Now recognising with the new words"
                                    banner_color = (0, 255, 100)
                                    banner_expire = curr_time + (1e12 if _cur is not None else 4.0)
                                    print(f"[DayTu] Lưu mẫu {_tk}/{_tn} cho '{_tw}' (stroke {stroke_duration:.2f}s)."
                                          + ("" if _cur is not None else " Hoàn tất — chuyển sang nhận diện."))
                                else:
                                    print("[DayTu] Cử chỉ không dựng được đặc trưng, bỏ qua — hãy ký lại.")
                            elif early_pre is not None:
                                # Chốt sớm: dùng lại đặc trưng + kết quả match của dự đoán tạm (không tính lại).
                                _ep_traj, _ep_shp, _ep_body, match_result, cur_lead_trim_sec, cur_lead_trim_tried = early_pre
                                body_ok = _ep_body
                            elif fast_on and lead_trim_on and not force_stroke:
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

                            if match_result is not None and shape_gate is not None:
                                match_result = shape_gate.apply(
                                    match_result, seg_records if seg_records is not None else raw_records, handedness_label)
                            if match_result is not None:
                                last_match_result = match_result
                                if match_result.fusion_note:
                                    print(f"[Fusion] {match_result.fusion_note}")
                                    event_log.push(f"Fusion: {match_result.fusion_note}", ui_side_panel.COL_DL)
                                # a) Chấp nhận từ vựng (ACCEPTED)
                                if match_result.status == "ACCEPTED" and match_result.top_word:
                                    word_key = match_result.top_word
                                    en_phrase = get_asl_phrase(word_key, lang="en", default=match_result.top_display or word_key)
                                    vi_phrase = get_asl_phrase(word_key, lang="vi", default="")

                                    # R3 Layer 3 / Debouncing: Chống bắn lặp từ trong 0.8s
                                    is_repeat = (en_phrase == last_committed_word and (curr_time - last_commit_time) < 0.8)
                                    # KHÓA MỘT LẦN: stroke này đã được FSM chốt (kể cả early commit qua force_commit) nên
                                    # không thể chốt lần hai ở đường normal; ở đây chỉ quyết định có COMMIT TỪ hay không.
                                    # Chỉ hỏi khóa khi không phải is_repeat (is_repeat vẫn do cơ chế cũ xử lý, không tốn lượt của span).
                                    _os_blocked = False
                                    if one_shot_lock is not None and not is_repeat:
                                        _os_ok, _os_reason = one_shot_lock.allow_commit(float(completed_stroke[0].timestamp), word_key)
                                        _os_blocked = not _os_ok
                                    if _os_blocked:
                                        hud_toast = _make_toast("suppressed", [word_key], None, curr_time, 1.0)
                                        _os_why = ("ngoài khoảng đang ký (spotter báo không ký)" if _os_reason == "no_span"
                                                   else "đã có từ được chốt trong khoảng ký này")
                                        print(f"[OneShot] Chặn '{en_phrase}' ({_os_reason}): {_os_why}. (p tối đa trong stroke = {one_shot_lock.last_pmax:.2f})")
                                        if session_writer is not None:
                                            session_writer.writerow([
                                                time.strftime("%Y-%m-%d %H:%M:%S"), "SUPPRESSED_DUP", word_key, round(match_result.best_distance, 4),
                                                "", "", 0.0, round(stroke_duration, 3), n_stroke, body_ok, round(fps, 1),
                                                round(match_result.best_confidence, 4), round(match_result.match_quality, 2), round(match_result.r_ratio, 4), round(curr_time, 3), _commit_mode, round(completed_stroke[0].timestamp, 3),
                                                round(cur_lead_trim_sec, 3), cur_lead_trim_tried
                                            ])
                                            session_file.flush()
                                    elif not is_repeat:
                                        if sentence_words and sentence_finalized:
                                            # Câu trước đã chốt (hạ tay > 1,8 s hoặc đủ từ): từ mới mở CÂU MỚI, không nối vào câu cũ.
                                            sentence_words.clear()
                                            sentence_keys.clear()
                                        sentence_words.append(en_phrase)
                                        sentence_keys.append(word_key)
                                        last_commit_time = curr_time
                                        last_committed_word = en_phrase
                                        sentence_finalized = False
                                        _max_sw = int(CONFIG.get("max_sentence_words", 8))
                                        if _max_sw > 0 and len(sentence_words) >= _max_sw:
                                            # Ký liên tục không hạ tay: chốt câu khi đủ số từ để câu không dài vô hạn.
                                            sentence_finalized = True
                                            hud_toast = _make_toast("committed", [], None, curr_time, 2.0)
                                            print(f"[SentenceBuilder] ✓ ĐỦ {_max_sw} TỪ, CHỐT CÂU: {' '.join(sentence_words)} .")
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
                                        event_log.push(f"Nhận từ: {en_phrase} (Khớp {do_khop:.0f}%, nguồn {match_result.result_source})", ui_side_panel.COL_OK)

                                    if not _os_blocked:
                                        _emit_event("ACCEPTED", match_result, completed_stroke, stroke_duration, _commit_mode,
                                                records=(seg_records if seg_records is not None else raw_records))
                                    # Ghi CSV Session Log (bị khóa một lần thì đã ghi dòng SUPPRESSED_DUP ở trên)
                                    if session_writer is not None and not _os_blocked:
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

                                    _emit_event("UNCERTAIN", match_result, completed_stroke, stroke_duration, _commit_mode,
                                            records=(seg_records if seg_records is not None else raw_records))
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

                                    _emit_event("REJECTED", match_result, completed_stroke, stroke_duration, _commit_mode,
                                            records=(seg_records if seg_records is not None else raw_records))
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
                        event_log.push(f"Loai bo (Duration Gate): {stroke_duration:.2f}s", ui_side_panel.COL_MUTED)
                        _emit_event("OUT_OF_DURATION", None, completed_stroke, stroke_duration, _commit_mode)
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
            elif one_shot_lock is not None:
                # KHÓA MỘT LẦN: mất tay -> spotter đóng span (v_norm = 0).
                one_shot_lock.observe(curr_time, one_shot_spotter.update(curr_time, False, 0.0))

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

            # Từ đây, TOÀN BỘ trạng thái/kết quả/log/demo quá trình nhận diện được vẽ vào panel
            # bên phải + panel dưới (xem ui_side_panel.py) — KHÔNG còn overlay nào đè lên khung
            # camera nữa (chỉ giữ face-anchor-box/trail vẽ trực tiếp lên tay ở trên, vốn là một
            # phần của việc demo chính DIP đang "nhìn" tay, không phải HUD số liệu che khung).
            fps_color = (0, 255, 0) if fps >= 24.0 else ((0, 200, 255) if fps >= 20.0 else (0, 0, 255))
            if face_detector is None:
                face_badge = "MAT: TAT"
                face_col = (130, 130, 130)
            elif current_face_box is not None:
                face_badge = "MAT: OK"
                face_col = (0, 255, 0)
            else:
                face_badge = "MAT: TIM..."
                face_col = (0, 165, 255)

            if seg_state == KineticState.RECORDING:
                k_text = "DANG GHI"
                k_color = (0, 255, 0)
            elif seg_state == KineticState.READY:
                k_text = "SAN SANG"
                k_color = (0, 255, 0)
            else:
                k_text = "KHUNG TAY" if hand_found else "CHO TAY"
                k_color = (180, 180, 180)

            # Mọi sự kiện phản hồi (banner_title/subtitle: commit, uncertain, unknown, backspace,
            # save, undo...) trước đây CHỈ hiện 1-2s trên camera; giờ đẩy vào log sự kiện của panel
            # để không phụ thuộc terminal/kịp nhìn. Khoá theo (title, banner_expire) để mỗi lần
            # banner được set MỚI chỉ log đúng 1 lần, không log lặp lại mỗi khung hình đang hiển thị.
            if banner_title and curr_time < banner_expire:
                _banner_log_key = (banner_title, banner_expire)
                if _banner_log_key != _last_banner_log_key:
                    _last_banner_log_key = _banner_log_key
                    _log_text = f"{banner_title} | {banner_subtitle}" if banner_subtitle else banner_title
                    event_log.push(_log_text, banner_color)

            if ui_mode == "legacy_demo":
                # ---------------- GIAO DIỆN DEMO SONG NGỮ CŨ (ui_render.DemoHUD) ----------------
                # Chỉ dùng khi cố tình chạy legacy_demo không có side panel
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
                        event_log.push(f"Loi HUD demo: {hud_err}", (0, 0, 255))

            if profile:
                _t_draw += time.perf_counter() - _t0_draw_hud

            # Số liệu profile (SPEC 02 Nhiệm vụ 1) — vẽ vào panel bên phải thay vì đè lên camera.
            profile_lines: List[str] = []
            if profile and len(prof_total) > 0:
                avg_mp = sum(prof_mediapipe) / len(prof_mediapipe) * 1000.0
                avg_dip = sum(prof_dip_cnn) / len(prof_dip_cnn) * 1000.0
                avg_feat = sum(prof_features) / len(prof_features) * 1000.0
                avg_dtw = sum(prof_dtw) / len(prof_dtw) * 1000.0
                avg_draw = sum(prof_draw) / len(prof_draw) * 1000.0
                avg_tot = sum(prof_total) / len(prof_total) * 1000.0
                fps_prof = 1000.0 / avg_tot if avg_tot > 0 else 0.0
                profile_lines = [
                    f"PROFILE(60f) fps={fps_prof:.1f}",
                    f"mp={avg_mp:.1f}ms dip={avg_dip:.1f}ms feat={avg_feat:.1f}ms",
                    f"dtw={avg_dtw:.1f}ms draw={avg_draw:.1f}ms tot={avg_tot:.1f}ms",
                ]

            cam_diag = None
            if cam_monitor is not None:
                cam_diag = (f"that:{cam_monitor.unique_fps:.0f}/s trung:{cam_monitor.dup_ratio*100:.0f}% "
                            f"(gan-trung {cam_monitor.near_dup_ratio*100:.0f}%) sang:{cam_monitor.luma_mean:.0f}")

            # SPEC 14: câu tự nhiên chỉ hiện khi câu đã chốt và nó KHÁC cách nối thô; tiếng Việt bỏ dấu vì panel vẽ bằng cv2.putText.
            composed_display: Optional[str] = None
            if composer_on and sentence_finalized and sentence_keys:
                _ckey = (tuple(sentence_keys), ui_lang)
                if _ckey != composer_cache_key:
                    try:
                        composed_result = compose_sentence(sentence_keys, ui_lang, phrase_fn=_phrase_for_composer)
                    except Exception:
                        composed_result = None
                    composer_cache_key = _ckey
                if composed_result is not None and composed_result.changed:
                    composed_display = strip_diacritics(composed_result.text)

            panel_state = ui_side_panel.SidePanelState(
                fps=fps,
                camera_index=camera_index,
                cam_diag=cam_diag,
                face_badge=face_badge,
                face_color=face_col,
                seg_text=k_text,
                seg_color=k_color,
                v_norm=v_norm,
                vocabulary_empty=(len(dtw_engine.vocabulary) == 0),
                hand_found=hand_found,
                match_result=last_match_result,
                dl_enabled=bool(dl_model_path),
                dl_conf_threshold=dtw_engine.dl_conf_threshold,
                sentence_words=list(sentence_words),
                sentence_finalized=sentence_finalized,
                show_features=show_features,
                angles=current_angles,
                palm_len=palm_len,
                palm_normal=current_normal,
                event_log=event_log,
                help_line="[d]:Dac trung [Swipe-Left]:Xoa tu [Ha tay]:Chot cau [c]:Xoa cau [u]:Doi giao dien [q]:Thoat",
                ui_mode=ui_mode,
                ui_lang=ui_lang,
                profile_lines=profile_lines,
                speech_status=speech_service.get_speech_status(ui_lang),
                composed_sentence=composed_display,
            )
            canvas, _cw, _ch = ui_side_panel.compose_canvas(display_frame)
            ui_side_panel.draw_right_panel(canvas, w_f, 0, ui_side_panel.PANEL_W, h_f + ui_side_panel.BOTTOM_H, panel_state)
            ui_side_panel.draw_bottom_panel(canvas, 0, h_f, w_f, ui_side_panel.BOTTOM_H, panel_state)
            display_frame = canvas

            if one_shot_lock is not None and not headless:
                # Chỉ báo spotter (gỡ lỗi): xanh = đang ký (gate mở), xám = không ký; kèm xác suất nếu là spotter DL.
                _g_open = bool(one_shot_spotter.gate.is_open if hasattr(one_shot_spotter, "gate") else one_shot_spotter.is_open)
                _g_p = getattr(one_shot_spotter, "last_prob", None)
                _g_txt = f"SPOTTER {'KY' if _g_open else '--'}" + (f" p={_g_p:.2f}" if _g_p is not None else "")
                cv2.putText(display_frame, _g_txt, (max(10, display_frame.shape[1] - 230), 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0) if _g_open else (150, 150, 150), 2, lineType=cv2.LINE_AA)

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
                    event_log.push("Đã xóa toàn bộ câu thoại", ui_side_panel.COL_WARN)
                elif key == ord('f'):
                    flip_cam = not flip_cam
                    event_log.push(f"Lật camera: {'BẬT (Gương)' if flip_cam else 'TẮT'}", ui_side_panel.COL_MUTED)
                elif key in [ord('p'), ord('P')]:
                    if cap is not None:
                        cap.set(cv2.CAP_PROP_SETTINGS, 1)
                elif key in [ord('l'), ord('L')]:
                    ui_lang = "vi" if ui_lang == "en" else "en"
                    print(f"[UI] Ngôn ngữ giao diện + giọng nói: {ui_lang.upper()}")
                    event_log.push(f"Đổi ngôn ngữ: {ui_lang.upper()}", ui_side_panel.COL_DL)
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
                    segmenter = _new_segmenter()
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
        if one_shot_lock is not None:
            print(f"[OneShot] Cả phiên: cho phép {one_shot_lock.stats['allowed']} lần commit, "
                  f"chặn {one_shot_lock.stats['suppressed']} lần trùng trong cùng span "
                  f"(tổng {len(one_shot_lock.spans)} span gần nhất đang giữ).")
        if cam_monitor is not None:
            _cam_sum = cam_monitor.session_summary()
            if _cam_sum:
                print(f"[Camera] Cả phiên: {_cam_sum}")
                print("[Camera] Gợi ý: nếu 'trùng y hệt' ≈ 0% và 'khung mới thật' ≈ 30/s thì --dedupe-frames/--time-norm gần như không đổi gì.")
        if enroller is not None and enroller.count() > 0:
            try:
                _is_production_bank = os.path.abspath(templates_file) == os.path.abspath(
                    CONFIG.get("templates_path", "webcam_templates.json"))
                _is_temp_merged = os.path.basename(templates_file).startswith("_gop_")
                if enroll_save_path:
                    # Chạy bằng bank GỘP tạm (Đạt + hồ sơ): mẫu mới ghi vào HỒ SƠ CÁ NHÂN thật, không bao giờ vào file gộp tạm
                    # (file đó bị dựng lại mỗi lần chạy nên mẫu sẽ mất) và không vào bank Đạt.
                    _n = enroller.merge_into_file(enroll_save_path, enrolled_cap=6)
                    print(f"[XacNhan] Cả phiên: đã lưu {_n} mẫu ({enroller.summary()}) vào hồ sơ '{enroll_save_path}' — "
                          f"lần sau tự dùng lại (bank gộp Đạt + hồ sơ).")
                elif _is_temp_merged:
                    _enr_path = os.path.join("profiles", f"xacnhan_{time.strftime('%Y%m%d_%H%M%S')}.json")
                    _n = enroller.save(_enr_path)
                    print(f"[XacNhan] Cả phiên: thêm {_n} mẫu ({enroller.summary()}) -> {_enr_path} "
                          f"(bank gộp tạm không lưu được; hãy chạy với --luu-ho-so profiles/<tên>.json để tích lũy).")
                elif _is_production_bank:
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
        if events_file is not None:
            try:
                events_file.close()
            except Exception:
                pass
        if trace_file is not None:
            try:
                trace_file.close()
            except Exception:
                pass
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
    parser.add_argument("--video-clock", action="store_true", default=False,
                        help="Với --video: dùng đồng hồ của video (khung n tại n/fps) làm mốc thời gian, để phát lại nhanh hơn thời gian thật "
                             "mà các ngưỡng theo giây vẫn đúng như lúc thu.")
    parser.add_argument("--headless", action="store_true", default=False, help="Chạy không mở cửa sổ đồ họa OpenCV (dùng cho đo benchmark/test).")
    parser.add_argument("--max-frames", type=int, default=None, help="Dừng sau số khung hình nhất định (dùng cho benchmark/test).")
    parser.add_argument("--threshold-mode", type=str, choices=["legacy", "scale_k"], default=None, help="Chế độ ngưỡng: 'scale_k' (ngưỡng theo độ rộng nội lớp) hoặc 'legacy' (cố định).")
    parser.add_argument("--threshold-k", type=float, default=None, help="Hệ số k cho ngưỡng chấp nhận (k * độ_rộng, mặc định 1.6)")
    parser.add_argument("--uncertainty-k", type=float, default=None, help="Hệ số k cho ranh giới từ chối (uncertainty_k * độ_rộng, mặc định 2.2)")
    parser.add_argument("--time-norm", action=argparse.BooleanOptionalAction, default=None,
                        help="Bật/tắt chuẩn hóa vận tốc + Swipe-Left theo thời gian (mặc định lấy CONFIG['time_norm']['enabled'] = TẮT).")
    parser.add_argument("--dedupe-frames", action=argparse.BooleanOptionalAction, default=None,
                        help="Bật/tắt bỏ khung trùng trước khi dò tay (mặc định lấy CONFIG['dedupe_frames'] = TẮT).")
    parser.add_argument("--composer", action=argparse.BooleanOptionalAction, default=None,
                        help="SPEC 14: bật/tắt sinh câu tự nhiên khi chốt câu (dòng thứ hai dưới chuỗi từ + đọc câu). "
                             "Mặc định lấy CONFIG['composer']['enabled'] (BẬT); dùng --no-composer để về hành vi cũ.")
    parser.add_argument("--confirm-enroll", action=argparse.BooleanOptionalAction, default=True,
                        help="Chế độ hỏi xác nhận: cử chỉ không chắc/bị từ chối được hỏi lại, mẫu được xác nhận "
                             "thêm vào bank (phím y/2/n/z). MẶC ĐỊNH BẬT (đã đo: đây là đòn bẩy lớn nhất cho "
                             "người lạ, xem eval_enroll.py) — dùng --no-confirm-enroll để tắt.")
    parser.add_argument("--day-tu", nargs="+", default=None, metavar="TU",
                        help="HIỆU CHỈNH NHANH: dạy các từ này (từ có sẵn hoặc HOÀN TOÀN MỚI), mỗi từ --day-so-mau mẫu, rồi tự chuyển "
                             "sang nhận diện — dùng được ngay, không train. Ví dụ: --day-tu HUNGRY SORRY")
    parser.add_argument("--day-so-mau", type=int, default=3, help="Số mẫu mỗi từ khi dạy (mặc định 3; 1-2 vẫn dùng được nhưng yếu hơn).")
    parser.add_argument("--luu-ho-so", type=str, default=None,
                        help="Lưu mẫu xác nhận/dạy của phiên vào hồ sơ cá nhân này (dip.py tự truyền khi chạy bank gộp Đạt + hồ sơ).")
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
    parser.add_argument("--handshape-gate", action=argparse.BooleanOptionalAction, default=None,
                        help="Bật/tắt bộ lọc ứng viên theo dáng tay (handshape_gate.py). Mặc định tắt. Cần letter_landmark_model_image.pt "
                             "và handshape_gate_table.json (py build_handshape_table.py).")
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
    parser.add_argument("--lock-min-gap", type=float, default=1.2,
                        help="(nâng cao) Khóa một lần: khoảng tối thiểu (s) giữa hai từ KHÁC nhau cùng span; số rất lớn = tắt cho phép ký nhanh.")
    parser.add_argument("--lock-soft-prob", type=float, default=0.25,
                        help="(nâng cao) Khóa nghiêm (spotter dl): stroke ngoài span vẫn được chốt nếu p >= ngưỡng này; >1 = tắt.")
    parser.add_argument("--one-shot", action="store_true", default=False,
                        help="KHÓA MỘT LẦN: mỗi khoảng 'đang ký' (span do spotter báo) chỉ commit tối đa MỘT từ; "
                             "các stroke còn lại trong cùng span bị chặn (SUPPRESSED_DUP). Mặc định TẮT = hành vi cũ.")
    parser.add_argument("--spotter", type=str, choices=["rule", "dl"], default="rule",
                        help="Bộ spotter cấp tín hiệu 'đang ký' cho --one-shot: 'rule' = luật vận tốc (chỉ chặn từ trùng), "
                             "'dl' = CNN nhỏ (chặn cả từ vu vơ ngoài khoảng đang ký). Chỉ có tác dụng khi bật --one-shot.")
    parser.add_argument("--spotter-model", type=str, default="spotter_model.pth",
                        help="Checkpoint spotter DL (train_spotter.py --save). Chỉ dùng khi --spotter dl.")
    parser.add_argument("--start", type=float, default=None, help="Chỉ với --video: nhảy tới giây này của video trước khi nhận diện.")
    parser.add_argument("--end", type=float, default=None, help="Chỉ với --video: dừng khi quá giây này của video.")
    parser.add_argument("--events", type=str, default=None, help="Ghi mỗi cử chỉ thành một dòng JSON (JSONL) kèm nguồn DTW/DL (dùng cho bản web).")
    parser.add_argument("--mute", action="store_true", default=False, help="Tắt hẳn giọng nói.")
    parser.add_argument("--force-stroke", action="store_true", default=False,
                        help="Chỉ với --video --start --end: coi TOÀN BỘ đoạn đã chọn là một cử chỉ (không dùng bộ phân đoạn) để lấy mẫu.")
    parser.add_argument("--motion", type=str, choices=["wrist", "hand"], default="wrist",
                        help="Nguồn vận tốc cho bộ phân đoạn: 'wrist' (cổ tay, mặc định) hoặc 'hand' (thêm chuyển động ngón tay so với cổ tay, cho cử chỉ nhẹ).")
    parser.add_argument("--pause-sec", type=float, default=None, help="Tay khựng ngần này giây thì vào chờ chốt (mặc định 0.12 ở chuỗi nhanh).")
    parser.add_argument("--grace-sec", type=float, default=None, help="Chờ thêm ngần này giây trước khi chốt hẳn (mặc định 0.05 ở chuỗi nhanh).")
    parser.add_argument("--v-stop", type=float, default=None, help="Ngưỡng vận tốc coi là đứng yên (mặc định 0.04 lòng bàn tay/khung).")
    parser.add_argument("--v-start", type=float, default=None, help="Ngưỡng vận tốc bắt đầu ký (mặc định 0.08 lòng bàn tay/khung).")
    parser.add_argument("--early-commit", action=argparse.BooleanOptionalAction, default=None, help="Bật/tắt chốt sớm khi dự đoán tạm đã chắc.")
    parser.add_argument("--trace", type=str, default=None, help="Ghi vết từng khung (JSONL) cho bản web.")
    parser.add_argument("--no-ready-pose", action="store_true", default=False,
                        help="Không đòi tay đứng yên trước khi bắt đầu ghi (cho người ký nhanh, video không có nhịp dừng).")
    parser.add_argument("--hold-on-hand-loss", action="store_true", default=False,
                        help="Khi tay biến mất giữa lúc đang ghi, giữ tư thế cuối ~0.3 s để chốt cử chỉ thay vì xóa.")
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
    if args.handshape_gate is not None:
        CONFIG["handshape_gate"]["enabled"] = bool(args.handshape_gate)
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
        teach_words=args.day_tu,
        teach_per_word=args.day_so_mau,
        enroll_save_path=args.luu_ho_so,
        dl_model_path=eff_dl_model_path,
        dl_conf_threshold=args.dl_conf_threshold,
        one_shot=args.one_shot,
        lock_min_gap=args.lock_min_gap,
        lock_soft_prob=args.lock_soft_prob,
        spotter_kind=args.spotter,
        spotter_model_path=args.spotter_model,
        video_clock=args.video_clock,
        start_sec=args.start,
        end_sec=args.end,
        events_path=args.events,
        mute_speech=args.mute,
        ready_pose=False if args.no_ready_pose else None,
        hold_on_hand_loss=args.hold_on_hand_loss,
        force_stroke=args.force_stroke,
        trace_path=args.trace,
        motion_mode=args.motion,
        composer=args.composer,
        seg_tune={k: v for k, v in {"pause_sec": args.pause_sec, "grace_sec": args.grace_sec, "v_stop": args.v_stop,
                                    "v_start": args.v_start, "early_commit": args.early_commit}.items() if v is not None} or None,
    )
