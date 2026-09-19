"""
LÕI GIẢI THUẬT NHẬN DIỆN CỬ CHỈ ĐỘNG & SO KHỚP CHUỖI LAI (HYBRID DTW ENGINE)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Kiến trúc Lõi (Core Engine):
  1. KineticPauseSegmenter:
     - Phân đoạn động học dựa trên vận tốc chuẩn hóa lòng bàn tay (Palm-Scale Invariance).
     - Máy trạng thái FSM: [IDLE] -> [RECORDING] -> [PAUSE_DETECTED].
     - Bắt lệnh ưu tiên cấp tốc Swipe-Left (Backspace) theo tiêu chí hình học tức thời.
  2. ResampledGesture & Nội suy 2 nhánh:
     - Tọa độ (x, y): Nội suy tuyến tính đều về K = 32 frames.
     - Vector tiếp tuyến (dx, dy): Tính sai phân trực tiếp trên chuỗi 32 điểm.
     - Dáng bàn tay S_t (Categorical): Nội suy Nearest-Neighbor bảo toàn nhãn rời rạc.
  3. HybridDTWEngine:
     - Hàm chi phí lai: Sai số Quỹ đạo (α=1.0) + Sai số Hướng (β=0.5) + Phạt Dáng tay (λ=0.6).
     - Dải băng Sakoe-Chiba (r = 4) giới hạn uốn cong bệnh lý, tốc độ < 0.05ms trên CPU.
     - Chiến lược quyết định 3 tầng: CHẤP NHẬN / NGHI VẤN TOP-2 / TỪ CHỐI.
     - Schema lưu trữ JSON phiên bản hóa (v1.0).
"""

import os
import sys
import time
import json
import math
from enum import Enum
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# SPEC 03: hàm đếm nhịp lặp dùng chung (dynamic_features.py chỉ phụ thuộc numpy, không import ngược file này ở mức module).
from dynamic_features import compute_repetition_count, BODY_X_COLUMN

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# 1. CẤU HÌNH & CẤU TRÚC DỮ LIỆU CỐT LÕI
# =============================================================================

class KineticState(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    RECORDING = "RECORDING"
    PENDING_COMMIT = "PENDING_COMMIT"   # Đang chờ xác nhận grace-window trước khi chốt cử chỉ
    PAUSE_DETECTED = "PAUSE_DETECTED"   # LEGACY: giữ lại để tương thích tên gọi cũ, FSM mới không còn
                                         # dừng lại ở trạng thái này (self.state không bao giờ nghỉ ở đây nữa)


# Đường dẫn mặc định file trọng số Per-Sign ASL-LEX 2.0, dùng làm fallback cho
# GoldenTriadDTWMatcher khi không có sign_weights_path nào được truyền vào.
# Giá trị này KHÔNG phụ thuộc predict_sentence_webcam.CONFIG (tránh import vòng lặp),
# nhưng giữ nguyên literal string giống hệt CONFIG["sign_weights_path"] ở đó.
DEFAULT_SIGN_WEIGHTS_PATH = "ASL_ALEX_2.0/sign_weights_asllex.json"


# =============================================================================
# CỔNG THỜI LƯỢNG CỬ CHỈ (STROKE DURATION GATE) — NGUỒN SỰ THẬT DUY NHẤT
# =============================================================================
# Trước đây hai con số [0.40, 2.50] chỉ được hardcode ở predict_sentence_webcam.py
# (R3 Layer 2 Guard) mà collect_dynamic_templates.py KHÔNG kiểm tra. Hậu quả đo được:
# 4/34 mẫu webcam đã thu (toàn bộ thuộc từ YOU, thời lượng 3.04s - 3.59s) nằm NGOÀI
# cổng thời lượng lúc suy luận => những template đó vĩnh viễn không thể được sinh ra
# ở thời điểm inference, chúng chỉ là "xác" làm phình phương sai nội lớp (intra-class
# variance) và kéo ngưỡng của từ lên một cách vô ích.
#
# Vì vậy hai hằng số này được khai báo MỘT LẦN duy nhất tại đây và cả hai file
# (bộ thu mẫu + bộ suy luận) đều import lại, bảo đảm chúng không bao giờ lệch nhau nữa.
STROKE_DURATION_MIN_SEC = 0.40
STROKE_DURATION_MAX_SEC = 2.50


def is_stroke_duration_valid(duration_sec: float) -> bool:
    """Kiểm tra thời lượng một stroke có nằm trong cổng [0.40s, 2.50s] dùng chung hay không."""
    return STROKE_DURATION_MIN_SEC <= float(duration_sec) <= STROKE_DURATION_MAX_SEC


# =============================================================================
# CHUẨN HÓA TAY THUẬN / CHIRALITY CANONICALIZATION
# =============================================================================

# max_quan_sat_duoc = 0.9806. Đo ngày 19/09/2026. Do đó kẹp vào [0.85, 0.97]
MAX_TRAJECTORY_STRAIGHTNESS = 0.97

def compute_trajectory_straightness(xs: np.ndarray, ys: np.ndarray) -> float:
    path_length = np.sum(np.hypot(np.diff(xs), np.diff(ys)))
    net_displacement = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    if path_length < 1e-6:
        return 0.0
    return net_displacement / path_length

def is_movement_epenthesis(stroke_frames, straightness_max: float = MAX_TRAJECTORY_STRAIGHTNESS,
                           min_path_palm_ratio: float = 0.8) -> bool:
    """
    Kiểm tra chuyển động chêm giữa các ký hiệu (movement epenthesis).
    Nếu hành động quá thẳng, nó không phải là cử chỉ thật.
    """
    if len(stroke_frames) < 2:
        return False
    ref_palm = max(stroke_frames[0].palm_len, 15.0)
    xs = np.array([(f.x - stroke_frames[0].x) / ref_palm for f in stroke_frames], dtype=np.float32)
    ys = np.array([(f.y - stroke_frames[0].y) / ref_palm for f in stroke_frames], dtype=np.float32)
    
    path_length_da_chuan_hoa = np.sum(np.hypot(np.diff(xs), np.diff(ys)))
    if path_length_da_chuan_hoa < min_path_palm_ratio:
        return False
        
    straightness = compute_trajectory_straightness(xs, ys)
    return straightness > straightness_max


# Ngưỡng dải "mập mờ" của heuristic nx: max(|nx|) trên toàn chuỗi nhỏ hơn giá trị này
# thì dấu của nx không mang thông tin đáng tin cậy (lòng bàn tay gần như không thể hiện rõ
# phương ngang trong suốt cử chỉ), nên KHÔNG lật để tránh phá dữ liệu bằng một quyết định ngẫu nhiên.
CHIRALITY_AMBIGUITY_EPS = 0.05

# Đúng 3 cột bị đảo dấu khi lật gương: x (cột 0), dx (cột 2), nx (cột 9).
# y/dy (dọc), 5 góc uốn ngón tay (vô hướng), ny/nz và 5 vận tốc góc đều KHÔNG đổi dấu.
CHIRALITY_FLIP_COLUMNS = (0, 2, 9)


def canonicalize_chirality(
    traj: np.ndarray,
    handedness: Optional[str] = None,
    ambiguity_eps: float = CHIRALITY_AMBIGUITY_EPS,
    flip_body_x: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Đưa một quỹ đạo đặc trưng 12D/17D về KHÔNG GIAN CHUẨN TẮC "tay phải" (canonical space),
    bằng cách lật gương theo trục ngang đúng 3 kênh mang dấu chiều ngang: x, dx, nx.

    ---------------------------------------------------------------------------
    VÌ SAO PHẢI CÓ HÀM NÀY? (nguyên nhân gốc rễ, đã đo bằng số thật)
    ---------------------------------------------------------------------------
    MediaPipe gán nhãn handedness ("Left"/"Right") với GIẢ ĐỊNH ảnh đầu vào đã được
    LẬT GƯƠNG (selfie view). Nhưng cả collect_dynamic_templates.py lẫn
    predict_sentence_webcam.py đều mặc định flip_cam = False, tức là đưa vào khung hình
    CHƯA lật => nhãn handedness bị đảo so với thực tế vật lý.

    Hàm compute_palm_normal_vector(landmarks_3d, handedness) trong predict_sentence_webcam.py
    lại đảo dấu TOÀN BỘ vector pháp tuyến khi handedness == "Left". Kết quả: quy ước dấu của
    vector pháp tuyến lòng bàn tay KHÁC NHAU giữa dữ liệu thu bằng webcam và dữ liệu trích
    từ video ASL Citizen.

    Hậu quả định lượng: trọng số w_n = 0.8 trong khi ngưỡng chấp nhận mỗi từ chỉ khoảng 0.6.
    Một vector pháp tuyến bị lật dấu tự nó đã đóng góp ~0.8 vào khoảng cách DTW —
    LỚN HƠN cả ngưỡng — nên chắc chắn bị TỪ CHỐI dù cử chỉ hoàn toàn đúng. Đo được:
      - webcam -> webcam (leave-one-out): 94.1%  (engine vốn đã tốt)
      - webcam -> template ASL Citizen:   35.3%, margin ÂM (d_correct 0.846 > d_wrong 0.690)
      - sau khi chuẩn hóa chirality:      64.7%, margin DƯƠNG

    ---------------------------------------------------------------------------
    HAI NHÁNH QUYẾT ĐỊNH — VÀ TẠI SAO CHÚNG KHÔNG NGANG HÀNG NHAU
    ---------------------------------------------------------------------------
    1) NHÁNH CHÍNH THỐNG (handedness != None):
       Lật khi và chỉ khi handedness.strip().lower() == "left". Đây là cách ĐÚNG:
       một quy ước tường minh, nhất quán, dựa trên nhãn tay thuận có thật.

    2) NHÁNH CỨU HỘ DỮ LIỆU CŨ (handedness == None) — CHỈ LÀ HEURISTIC:
       Thay vì tính mean(nx) (vốn bị triệt tiêu khi lòng bàn tay quay qua phương thẳng đứng
       giữa cử chỉ, khiến trung bình gần 0 dù chirality rất rõ), ta tìm frame có |nx| lớn nhất.
       Đó là thời điểm chirality bộc lộ rõ rệt nhất trong toàn bộ stroke.
       Lật nếu giá trị nx tại frame cực đại đó < 0.
       ĐÂY KHÔNG PHẢI cách sửa đúng, chỉ là phao cứu sinh cho các mẫu đã lưu từ trước
       mà không hề ghi lại nhãn tay thuận. Vì vậy:
         - Nếu max(|nx|) trên toàn chuỗi < ambiguity_eps: thực sự không quan sát được
           chirality, coi là mập mờ và KHÔNG lật (thà giữ nguyên còn hơn đoán).
         - Mọi mẫu MỚI đều phải ghi trường "handedness" để nhánh này không bao giờ cần dùng.

    ---------------------------------------------------------------------------
    THAM SỐ & GIÁ TRỊ TRẢ VỀ
    ---------------------------------------------------------------------------
    traj          : np.ndarray shape (K, D). D < 12 (dữ liệu 4D/9D cũ không có pháp tuyến)
                    sẽ được trả về nguyên vẹn — không có gì để chuẩn hóa, và hành vi
                    4D legacy tuyệt đối không được phép xê dịch.
    handedness    : "Left" / "Right" / None (None = mẫu cũ, dùng nhánh heuristic).
    ambiguity_eps : Nửa độ rộng dải mập mờ của nhánh heuristic.

    Trả về (bản_sao_có_thể_đã_lật, đã_lật_hay_chưa). KHÔNG BAO GIỜ sửa traj tại chỗ.
    """
    arr = np.asarray(traj, dtype=np.float32)

    # Dữ liệu 4D/9D legacy hoặc chuỗi rỗng: không có kênh pháp tuyến (9, 10, 11) => không chuẩn hóa.
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 12:
        return arr, False

    if handedness is not None:
        should_flip = (str(handedness).strip().lower() == "left")
    else:
        # Nhánh cứu hộ heuristic cho mẫu cũ không có nhãn tay thuận:
        # Dùng giá trị nx tại frame có |nx| lớn nhất thay vì mean(nx) (tránh triệt tiêu dấu khi xoay tay).
        abs_nx = np.abs(arr[:, 9])
        if len(abs_nx) == 0 or not np.all(np.isfinite(abs_nx)):
            return arr, False
        max_abs = float(np.max(abs_nx))
        if max_abs < float(ambiguity_eps):
            return arr, False
        peak_idx = int(np.argmax(abs_nx))
        peak_nx = float(arr[peak_idx, 9])
        should_flip = (peak_nx < 0.0)

    if not should_flip:
        return arr, False

    flipped = arr.copy()
    for col in CHIRALITY_FLIP_COLUMNS:
        flipped[:, col] = -flipped[:, col]
    # SPEC 03: với vector 24D, cột 18 (x của mốc cơ thể = (wrist_x - tâm mặt) / bề rộng mặt) cũng là một
    # đại lượng ngang trong ảnh nên PHẢI đổi dấu theo khi lật gương, nếu không mẫu đã lật và chưa lật
    # của cùng một từ sẽ có mốc cơ thể trái dấu nhau. Vector <= 17D KHÔNG bị ảnh hưởng (tương thích ngược).
    if flip_body_x and flipped.shape[1] >= BODY_X_COLUMN + 2:
        flipped[:, BODY_X_COLUMN] = -flipped[:, BODY_X_COLUMN]
    return flipped, True


# LEGACY (Phase 2 sẽ dời): HybridDTWConfig thuộc bộ engine 4D cũ — không còn được
# predict_sentence_webcam.py sử dụng trong production, chỉ còn phục vụ tests/ cũ.
@dataclass
class HybridDTWConfig:
    """Tham số cấu hình thuật toán DTW Lai và ngưỡng phân loại."""
    alpha: float = 1.0           # Trọng số sai số tọa độ quỹ đạo Euclidean
    beta: float = 0.5            # Trọng số sai số vector hướng tiếp tuyến (dx, dy)
    lambda_shape: float = 0.6    # Hình phạt khác biệt dáng ngón tay rời rạc
    k_resample: int = 32         # Số frame chuẩn hóa cố định
    sakoe_chiba_radius: int = 4  # Bán kính dải băng Sakoe-Chiba (|i - j| <= r)
    default_threshold: float = 1.25  # Ngưỡng chấp nhận từ chuẩn
    uncertainty_ratio: float = 1.40  # Tỷ lệ vùng nghi vấn (Threshold -> 1.4 * Threshold)


# DÙNG CHUNG (KHÔNG PHẢI LEGACY THUẦN TÚY): GestureFrame là cấu trúc khung hình thô
# do KineticPauseSegmenter (TẦNG 2 - vẫn chạy production) sinh ra và trả về trong
# completed_stroke; đồng thời cũng là input của resample_gesture_sequence() (4D cũ)
# bên dưới. Vì vậy KHÔNG được xóa/đổi tên dù resample_gesture_sequence là legacy.
@dataclass
class GestureFrame:
    """Dữ liệu tức thời tại một khung hình camera."""
    x: float                     # Tọa độ x khớp cổ tay (pixel)
    y: float                     # Tọa độ y khớp cổ tay (pixel)
    palm_len: float              # Độ dài lòng bàn tay chuẩn hóa (LM 0 - LM 9)
    hand_shape: str              # Nhãn dáng tay tĩnh ('A', 'B', 'C', 'L', 'V', 'Unknown'...)
    timestamp: float             # Thời gian ghi nhận (time.time())


# LEGACY (Phase 2 sẽ dời): ResampledGesture chỉ phục vụ schema 4D cũ
# (resample_gesture_sequence / HybridDTWEngine) — engine 12D Golden Triad thật đang
# chạy production dùng np.ndarray traj_9d thô + List[str] shapes, không dùng class này.
@dataclass
class ResampledGesture:
    """Chuỗi cử chỉ đã được nội suy chuẩn hóa về K=32 frames."""
    # traj có shape (32, 4): các cột lần lượt là [x_norm, y_norm, dx, dy]
    traj: np.ndarray
    shapes: List[str]            # Danh sách 32 nhãn dáng tay tương ứng
    duration_sec: float          # Thời gian thực tế của cử chỉ gốc (giây)
    # --- Hai trường BỔ SUNG (mặc định None/False nên mọi lời gọi cũ vẫn chạy y nguyên) ---
    handedness: Optional[str] = None       # Nhãn tay thuận MediaPipe chiếm đa số của stroke
    chirality_canonical: bool = False      # True = traj đã nằm trong không gian chuẩn tắc
    session_id: Optional[str] = None       # SPEC 05A: Mã phiên thu (session_id)
    # Giữ lại nguyên vẹn mọi khóa lạ đọc được từ JSON (vd "source_video") để chu trình
    # load -> save của HybridDTWEngine không âm thầm làm mất dữ liệu gốc khó thu lại.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.extra)
        out.update({
            "traj_features": np.round(self.traj, 4).tolist(),
            "hand_shapes": self.shapes,
            "duration_sec": round(self.duration_sec, 3)
        })
        if self.handedness is not None:
            out["handedness"] = self.handedness
        if self.chirality_canonical:
            out["chirality_canonical"] = True
        if self.session_id is not None:
            out["session_id"] = self.session_id
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResampledGesture":
        traj_arr = np.array(data["traj_features"], dtype=np.float32)
        known = {"traj_features", "hand_shapes", "duration_sec", "handedness", "chirality_canonical", "session_id"}
        return cls(
            traj=traj_arr,
            shapes=data["hand_shapes"],
            duration_sec=data.get("duration_sec", 1.0),
            handedness=data.get("handedness"),
            chirality_canonical=bool(data.get("chirality_canonical", False)),
            session_id=data.get("session_id"),
            extra={k: v for k, v in data.items() if k not in known}
        )


@dataclass
class MatchCandidate:
    """Ứng viên từ vựng phù hợp trong quá trình so khớp DTW."""
    word_key: str
    display_name: str
    distance: float
    confidence: float            # Điểm tin cậy quy đổi [0.0, 1.0]


@dataclass
class MatchResult:
    """Kết quả phân loại chuỗi cử chỉ — dùng chung cho cả HybridDTWEngine (4D legacy)
    và GoldenTriadDTWMatcher (12D, engine thật đang chạy production)."""
    status: str                  # 'ACCEPTED' | 'UNCERTAIN' | 'REJECTED'
    top_word: Optional[str]
    top_display: Optional[str]
    best_distance: float
    best_confidence: float
    candidates: List[MatchCandidate] = field(default_factory=list)


# =============================================================================
# 2. BỘ PHÂN ĐOẠN ĐỘNG HỌC & ĐIỂM DỪNG (KINETIC PAUSE SEGMENTER)
# =============================================================================

class KineticPauseSegmenter:
    """
    TẦNG 2: Bộ Dò Điểm Dừng & Phân Đoạn theo Tỷ Lệ Lòng Bàn Tay (Palm Scale).
    - Đo vận tốc chuẩn hóa không thứ nguyên: v_norm = Δw / palm_len.
    - Quản lý FSM 4 trạng thái:
        [IDLE] -> [RECORDING] -> [PENDING_COMMIT] -> (RECORDING gộp lại | IDLE chốt cử chỉ)
      Khi tay khựng đủ lâu (pause_frames), FSM KHÔNG chốt cử chỉ ngay mà chuyển sang
      PENDING_COMMIT — một cửa sổ chờ xác nhận (grace_window_sec, mặc định 0.20s).
      Trong lúc chờ, nếu tay cử động lại rõ rệt (v_norm >= v_start_ratio) thì GỘP pha
      mới vào cùng 1 cử chỉ liên tục (quay lại RECORDING, không mất buffer) — phục vụ
      các từ vựng đa pha (VD: "Thank You" chạm cằm rồi dừng nhẹ trước khi đẩy tay ra).
      Nếu hết grace window mà tay vẫn đứng yên, cử chỉ mới thực sự được CHỐT và FSM về IDLE.
    - Bắt lệnh ưu tiên Swipe-Left (Backspace) tức thời mà không cần điểm dừng, nhưng chỉ
      kích hoạt khi buffer cử chỉ hiện tại còn NGẮN (<= swipe_max_buffer_frames, mặc định
      10 khung ~0.33s) — tránh cắt ngang từ vựng thật đã RECORDING lâu (VD: Goodbye vẫy tay
      có thành phần chuyển động ngang-trái mạnh).
    """
    def __init__(
        self,
        v_start_ratio: float = 0.08,      # Bắt đầu cử chỉ khi v >= 8% lòng bàn tay/frame
        v_stop_ratio: float = 0.035,      # Khựng tay khi v <= 3.5% lòng bàn tay/frame
        pause_frames: int = 11,           # Khựng liên tục 11 frames (~0.35s ở 30fps)
        min_stroke_frames: int = 8,       # Cử chỉ tối thiểu 8 frames (~0.27s)
        max_stroke_frames: int = 60,      # Cử chỉ tối đa 60 frames (~2.0s)
        swipe_threshold: float = -0.80,   # Quét sang trái >= 80% lòng bàn tay trong 6 frames
        swipe_ratio: float = 2.5,         # |Δx| / (|Δy| + ε) >= 2.5
        swipe_cooldown_sec: float = 1.0,  # Khóa lệnh xóa 1.0s sau mỗi lần kích hoạt
        grace_window_sec: float = 0.20,     # Cửa sổ chờ gộp cử chỉ đa pha (VD: "Thank You" có 2 pha)
        swipe_max_buffer_frames: int = 10,  # Chỉ cho Swipe-Left kích hoạt khi buffer còn ngắn (≤ ~0.33s)
        require_ready_pose: bool = False,     # Mac dinh False = giu nguyen FSM cu cho tests
        ready_stable_frames: int = 8,         # Tay phai dung yen 8 frame (~0.25s @30fps) moi coi la san sang
        max_phase_merges: int = 999,   # Mac dinh 999 = gan nhu vo han = giu hanh vi cu
    ):
        self.v_start_ratio = v_start_ratio
        self.v_stop_ratio = v_stop_ratio
        self.pause_frames = pause_frames
        self.min_stroke_frames = min_stroke_frames
        self.max_stroke_frames = max_stroke_frames
        self.swipe_threshold = swipe_threshold
        self.swipe_ratio = swipe_ratio
        self.swipe_cooldown_sec = swipe_cooldown_sec
        self.grace_window_sec = grace_window_sec
        self.swipe_max_buffer_frames = swipe_max_buffer_frames
        self.require_ready_pose = require_ready_pose
        self.ready_stable_frames = ready_stable_frames
        self.max_phase_merges = max_phase_merges

        self.state = KineticState.IDLE
        self.buffer: List[GestureFrame] = []
        self.pause_counter = 0
        self.prev_wrist: Optional[Tuple[float, float]] = None
        self.last_swipe_time = -999.0
        self.recent_wrists = deque(maxlen=6)   # Cửa sổ trượt 6 frames phát hiện Swipe
        self._pending_deadline: float = 0.0
        self.ready_counter = 0
        self.merge_count = 0

    def update(
        self,
        wrist_x: float,
        wrist_y: float,
        palm_len: float,
        hand_shape: str = "Unknown",
        timestamp: Optional[float] = None
    ) -> Tuple[KineticState, Optional[List[GestureFrame]], bool, float]:
        """
        Cập nhật từng khung hình camera.

        Returns:
            current_state: KineticState (IDLE, RECORDING, PENDING_COMMIT, PAUSE_DETECTED)
            completed_stroke: List[GestureFrame] nếu cử chỉ vừa hoàn tất, ngược lại None
            is_swipe_left: bool (True nếu phát hiện lệnh Backspace tức thời)
            v_norm: float (Vận tốc chuẩn hóa hiện tại)
        """
        if timestamp is None:
            timestamp = time.time()

        safe_palm = max(palm_len, 15.0)
        curr_frame = GestureFrame(
            x=wrist_x, y=wrist_y,
            palm_len=safe_palm,
            hand_shape=hand_shape,
            timestamp=timestamp
        )

        # 1. Tính vận tốc chuẩn hóa theo cơ thể
        if self.prev_wrist is not None:
            dx = wrist_x - self.prev_wrist[0]
            dy = wrist_y - self.prev_wrist[1]
            dist_px = math.hypot(dx, dy)
            v_norm = dist_px / safe_palm
        else:
            v_norm = 0.0
            dx, dy = 0.0, 0.0

        self.prev_wrist = (wrist_x, wrist_y)
        self.recent_wrists.append((wrist_x, wrist_y, dx, dy, safe_palm, timestamp))

        # 2. KIỂM TRA LỆNH ƯU TIÊN: SWIPE-LEFT (BACKSPACE) TỨC THỜI
        #    R6 FIX: Chỉ cho phép kích hoạt khi buffer cử chỉ hiện tại còn NGẮN
        #    (<= swipe_max_buffer_frames, mặc định 10 khung ~0.33s). Một cú quẹt
        #    thật luôn hoàn tất đủ tiêu chí hình học trong 5-6 khung đầu tiên (vì
        #    peak_vx phải >= 2.5x v_start_ratio), nên điều kiện này KHÔNG làm mất
        #    khả năng bắt Swipe thật. Nó chỉ tắt Swipe sau khi một từ vựng thật đã
        #    được RECORDING lâu (buffer dài), để tránh cắt ngang từ có thành phần
        #    chuyển động ngang-trái mạnh (VD: Goodbye vẫy tay).
        #    Lưu ý: chặn cứng theo state==IDLE (thử nghiệm ban đầu) đã bị loại bỏ vì
        #    nó vô hiệu hóa Swipe hoàn toàn — state luôn rời IDLE ngay khung đầu của
        #    chính cú quẹt do v_start_ratio (0.08) thấp hơn nhiều peak_vx của Swipe (0.20).
        is_swipe_left = False
        swipe_buffer_ok = len(self.buffer) <= self.swipe_max_buffer_frames
        if swipe_buffer_ok and len(self.recent_wrists) >= 5 and (timestamp - self.last_swipe_time) >= self.swipe_cooldown_sec:
            old_x, old_y, _, _, ref_palm, _ = self.recent_wrists[0]
            delta_x_total = (wrist_x - old_x) / ref_palm
            delta_y_total = (wrist_y - old_y) / ref_palm

            # Kiểm tra 3 tiêu chí: Quãng đường âm mạnh + Áp đảo ngang + Tốc độ bứt phá
            peak_vx = min(item[2] / item[4] for item in self.recent_wrists)
            if (
                delta_x_total <= self.swipe_threshold
                and (abs(delta_x_total) / (abs(delta_y_total) + 1e-4)) >= self.swipe_ratio
                and peak_vx <= -0.20
            ):
                is_swipe_left = True
                self.last_swipe_time = timestamp
                self.reset()  # Reset bộ đệm để không lưu cú quẹt vào từ vựng
                return KineticState.IDLE, None, True, v_norm

        # 3. MÁY TRẠNG THÁI HỮU HẠN (FSM) CHO CỬ CHỈ TỪ VỰNG
        completed_stroke: Optional[List[GestureFrame]] = None

        if self.state == KineticState.IDLE:
            if not self.require_ready_pose:
                if v_norm >= self.v_start_ratio:
                    self.state = KineticState.RECORDING
                    self.buffer = [curr_frame]
                    self.pause_counter = 0
            else:
                if v_norm <= self.v_stop_ratio:
                    self.ready_counter += 1
                    if self.ready_counter >= self.ready_stable_frames:
                        self.state = KineticState.READY
                else:
                    self.ready_counter = 0

        elif self.state == KineticState.READY:
            if v_norm >= self.v_start_ratio:
                self.state = KineticState.RECORDING
                self.buffer = [curr_frame]
                self.pause_counter = 0

        elif self.state == KineticState.RECORDING:
            self.buffer.append(curr_frame)

            if v_norm <= self.v_stop_ratio:
                self.pause_counter += 1
            else:
                self.pause_counter = 0

            if self.pause_counter >= self.pause_frames:
                # R7 FIX: Vừa khựng đủ lâu — KHÔNG chốt ngay. Chuyển sang chờ
                # xác nhận (grace window) để tránh cắt đôi cử chỉ đa pha (VD:
                # "Thank You" chạm cằm rồi dừng nhẹ trước khi đẩy tay ra).
                self.state = KineticState.PENDING_COMMIT
                self._pending_deadline = timestamp + self.grace_window_sec
                self.pause_counter = 0
            elif len(self.buffer) > self.max_stroke_frames:
                self.reset()

        elif self.state == KineticState.PENDING_COMMIT:
            # Vẫn tiếp tục ghi khung hình trong lúc chờ, để không mất dữ liệu
            # nếu tay cử động lại (gộp pha 2 vào cùng 1 cử chỉ).
            self.buffer.append(curr_frame)

            if v_norm >= self.v_start_ratio:
                if self.merge_count < self.max_phase_merges and (timestamp - self.buffer[0].timestamp) <= STROKE_DURATION_MAX_SEC:
                    # Tay cử động lại rõ rệt trong lúc chờ -> GỘP làm 1 cử chỉ liên tục
                    self.state = KineticState.RECORDING
                    self.pause_counter = 0
                    self.merge_count += 1
                else:
                    total_n = len(self.buffer)
                    if self.min_stroke_frames <= total_n <= self.max_stroke_frames:
                        completed_stroke = list(self.buffer)
                    self.buffer.clear()
                    self.pause_counter = 0
                    self.state = KineticState.IDLE
                    self.merge_count = 0
            elif timestamp >= self._pending_deadline:
                # Hết thời gian chờ, tay vẫn đứng yên -> CHỐT CỬ CHỈ THẬT SỰ
                total_n = len(self.buffer)
                if self.min_stroke_frames <= total_n <= self.max_stroke_frames:
                    completed_stroke = list(self.buffer)
                self.buffer.clear()
                self.pause_counter = 0
                self.state = KineticState.IDLE
                self.merge_count = 0
            elif len(self.buffer) > self.max_stroke_frames:
                self.reset()

        if completed_stroke is not None:
            self.merge_count = 0

        return self.state, completed_stroke, is_swipe_left, v_norm

    def reset(self):
        """Khởi tạo lại toàn bộ trạng thái bộ đệm."""
        self.state = KineticState.IDLE
        self.buffer.clear()
        self.pause_counter = 0
        self.recent_wrists.clear()
        self._pending_deadline = 0.0
        self.prev_wrist = None
        self.ready_counter = 0
        self.merge_count = 0


# =============================================================================
# 3. NỘI SUY 2 NHÁNH & CHUẨN HÓA KHÔNG GIAN - THỜI GIAN
# =============================================================================
# LEGACY (Phase 2 sẽ dời): hàm nội suy 4D cũ, chỉ phục vụ HybridDTWEngine
# (không dùng trong production — xem resample_9d_gesture_sequence ở Mục 6).

def resample_gesture_sequence(
    raw_frames: List[GestureFrame],
    k_target: int = 32,
    min_frames: int = 8
) -> Optional[ResampledGesture]:
    """
    TẦNG 3.1: Chuẩn hóa Hình học & Nội suy 2 Nhánh về K=32 frames cố định.
    - Neo gốc chuyển động (Stroke-Origin Anchor): Trừ (x0, y0) và chia palm_len0.
    - Tọa độ (x, y): Nội suy tuyến tính 1D (Linear Interpolation).
    - Hướng tiếp tuyến (dx, dy): Đạo hàm sai phân trên chuỗi 32 điểm đã nội suy.
    - Dáng bàn tay S_t (Categorical): Nội suy Nearest-Neighbor bảo toàn nhãn rời rạc.
    """
    n_orig = len(raw_frames)
    if n_orig < min_frames:
        return None

    duration = raw_frames[-1].timestamp - raw_frames[0].timestamp
    if duration <= 0:
        duration = n_orig / 30.0

    # 1. Neo gốc chuyển động tại frame đầu tiên
    x0, y0 = raw_frames[0].x, raw_frames[0].y
    ref_palm = max(raw_frames[0].palm_len, 15.0)

    orig_t = np.linspace(0.0, 1.0, n_orig)
    target_t = np.linspace(0.0, 1.0, k_target)

    # Tọa độ chuẩn hóa gốc (N,)
    raw_xs = np.array([(f.x - x0) / ref_palm for f in raw_frames], dtype=np.float32)
    raw_ys = np.array([(f.y - y0) / ref_palm for f in raw_frames], dtype=np.float32)

    # 2. Nội suy tuyến tính cho Tọa độ (x, y) -> (32,)
    res_x = np.interp(target_t, orig_t, raw_xs).astype(np.float32)
    res_y = np.interp(target_t, orig_t, raw_ys).astype(np.float32)

    # 3. Tính Vector Hướng tiếp tuyến (dx, dy) bằng đạo hàm sai phân trên chuỗi 32 điểm
    res_dx = np.zeros(k_target, dtype=np.float32)
    res_dy = np.zeros(k_target, dtype=np.float32)
    res_dx[1:] = res_x[1:] - res_x[:-1]
    res_dy[1:] = res_y[1:] - res_y[:-1]
    res_dx[0] = res_dx[1]
    res_dy[0] = res_dy[1]

    # 4. Nội suy Nearest-Neighbor cho Nhãn Dáng Tay rời rạc
    res_shapes: List[str] = []
    for t_val in target_t:
        orig_idx = int(np.clip(np.round(t_val * (n_orig - 1)), 0, n_orig - 1))
        res_shapes.append(raw_frames[orig_idx].hand_shape)

    # Ghép ma trận đặc trưng (32, 4): [x_norm, y_norm, dx, dy]
    traj_matrix = np.column_stack([res_x, res_y, res_dx, res_dy]).astype(np.float32)

    return ResampledGesture(
        traj=traj_matrix,
        shapes=res_shapes,
        duration_sec=float(duration)
    )


# =============================================================================
# 4. THUẬT TOÁN HYBRID DTW VỚI DẢI BĂNG SAKOE-CHIBA
# =============================================================================
# LEGACY (Phase 2 sẽ dời): hàm khoảng cách DTW 4D cũ, chỉ phục vụ HybridDTWEngine
# (không dùng trong production — xem compute_golden_triad_dtw_distance ở Mục 6).

def compute_hybrid_dtw_distance(
    seq_q: ResampledGesture,
    seq_t: ResampledGesture,
    config: HybridDTWConfig
) -> float:
    """
    TẦNG 3.2: Tính khoảng cách Dynamic Time Warping Lai có dải băng Sakoe-Chiba.
    
    Hàm chi phí:
      Cost(i, j) = alpha * ||pos_q - pos_t|| + beta * ||dir_q - dir_t|| + lambda * I(shape_q != shape_t)
    
    Ràng buộc Sakoe-Chiba: |i - j| <= r (r = 4).
    Thời gian thực thi: < 0.05 ms trên CPU.
    """
    k = config.k_resample
    r = config.sakoe_chiba_radius

    # Khởi tạo ma trận tích lũy D với vô cực
    d_mat = np.full((k + 1, k + 1), np.inf, dtype=np.float32)
    d_mat[0, 0] = 0.0

    q_traj = seq_q.traj
    t_traj = seq_t.traj
    q_shapes = seq_q.shapes
    t_shapes = seq_t.shapes

    alpha = config.alpha
    beta = config.beta
    lambda_shape = config.lambda_shape

    for i in range(1, k + 1):
        j_start = max(1, i - r)
        j_end = min(k, i + r)

        for j in range(j_start, j_end + 1):
            # 1. Sai số tọa độ vị trí (x, y)
            d_pos = math.hypot(
                q_traj[i - 1, 0] - t_traj[j - 1, 0],
                q_traj[i - 1, 1] - t_traj[j - 1, 1]
            )

            # 2. Sai số vector hướng tiếp tuyến (dx, dy)
            d_dir = math.hypot(
                q_traj[i - 1, 2] - t_traj[j - 1, 2],
                q_traj[i - 1, 3] - t_traj[j - 1, 3]
            )

            # 3. Hình phạt khác biệt dáng bàn tay
            d_shape = lambda_shape if (q_shapes[i - 1] != t_shapes[j - 1]) else 0.0

            step_cost = alpha * d_pos + beta * d_dir + d_shape
            d_mat[i, j] = step_cost + min(
                d_mat[i - 1, j],       # Nén
                d_mat[i, j - 1],       # Giãn
                d_mat[i - 1, j - 1]    # Khớp chéo
            )

    # Chuẩn hóa khoảng cách tích lũy theo chiều dài chuỗi
    total_cost = d_mat[k, k]
    if math.isinf(total_cost):
        return 999.0

    return float(total_cost / float(k))


# =============================================================================
# 5. BỘ NÃO NHẬN DIỆN CHUỖI TỔNG THỂ (HYBRID DTW ENGINE)
# =============================================================================
# LEGACY (Phase 2 sẽ dời): HybridDTWEngine là engine 4D cũ. PROJECT.md từng ghi chú
# đây là core "STABLE/IMMUTABLE", nhưng thực tế KHÔNG phải engine đang chạy production
# — predict_sentence_webcam.py và collect_dynamic_templates.py dùng GoldenTriadDTWMatcher
# (12D, Mục 6 bên dưới). Lớp này được GIỮ NGUYÊN VẸN, KHÔNG xóa/đổi tên, vì toàn bộ
# tests/ (conftest.py + test_tier1-4*.py + test_verification_steps.py +
# test_harness_utils.py) vẫn import và kiểm thử trực tiếp nó — việc di dời sang engine
# thật là công việc của Phase 2 (deferred), ngoài phạm vi đợt refactor này.

class HybridDTWEngine:
    """
    TẦNG 3: Quản lý Cơ sở dữ liệu mẫu chuẩn và Phân loại Chuỗi Cử chỉ Động.
    - Quản lý file lưu trữ dynamic_templates.json có phiên bản hóa (v1.0).
    - So khớp khoảng cách DTW với tất cả mẫu trong cơ sở dữ liệu.
    - Quyết định phân tầng: CHẤP NHẬN / NGHI VẤN TOP-2 / TỪ CHỐI.
    """
    def __init__(
        self,
        templates_path: str = "dynamic_templates.json",
        config: Optional[HybridDTWConfig] = None
    ):
        self.templates_path = templates_path
        self.config = config or HybridDTWConfig()
        self.vocabulary: Dict[str, Dict[str, Any]] = {}
        self.load_templates()

    def load_templates(self) -> bool:
        """Đọc cơ sở dữ liệu mẫu chuẩn từ file JSON."""
        if not os.path.exists(self.templates_path):
            self.vocabulary = {}
            return False

        try:
            with open(self.templates_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_vocab = data.get("vocabulary", {})
            self.vocabulary = {}

            for w_key, w_info in raw_vocab.items():
                parsed_samples = []
                for s_dict in w_info.get("samples", []):
                    parsed_samples.append(ResampledGesture.from_dict(s_dict))

                self.vocabulary[w_key] = {
                    "display_name": w_info.get("display_name", w_key),
                    "threshold": float(w_info.get("threshold", self.config.default_threshold)),
                    "samples": parsed_samples
                }

            print(f"[DTWEngine] Đã nạp {len(self.vocabulary)} từ vựng từ {self.templates_path}")
            return True
        except Exception as e:
            print(f"[DTWEngine] Lỗi đọc file template ({e}), khởi tạo từ điển rỗng.")
            self.vocabulary = {}
            return False

    def save_templates(self) -> bool:
        """Ghi lưu toàn bộ cơ sở dữ liệu mẫu chuẩn ra file JSON có cấu trúc versioned."""
        vocab_out = {}
        for w_key, w_info in self.vocabulary.items():
            samples_out = [s.to_dict() for s in w_info["samples"]]
            vocab_out[w_key] = {
                "display_name": w_info["display_name"],
                "threshold": round(w_info["threshold"], 3),
                "num_samples": len(samples_out),
                "samples": samples_out
            }

        existing_created_at = None
        existing_feature_dim = None
        if os.path.exists(self.templates_path):
            try:
                with open(self.templates_path, "r", encoding="utf-8") as f_old:
                    old_data = json.load(f_old)
                    existing_created_at = old_data.get("metadata", {}).get("created_at")
                    existing_feature_dim = old_data.get("metadata", {}).get("feature_dim")
            except Exception:
                pass

        # Suy ra feature_dim từ dữ liệu thật thay vì hardcode 4.
        # Lấy độ dài của traj_features[0] của mẫu đầu tiên tìm được trong toàn bộ vocabulary.
        # Nếu không có mẫu nào (từ vựng rỗng), ưu tiên giữ existing_feature_dim nếu có, hoặc 4.
        inferred_feature_dim = 4
        found_dim = False
        for w_info in vocab_out.values():
            for s in w_info.get("samples", []):
                tf = s.get("traj_features") or s.get("traj_features_9d")
                if tf and len(tf) > 0 and hasattr(tf[0], "__len__") and len(tf[0]) > 0:
                    inferred_feature_dim = len(tf[0])
                    found_dim = True
                    break
            if found_dim:
                break
        if not found_dim and existing_feature_dim is not None:
            inferred_feature_dim = existing_feature_dim

        payload = {
            "schema_version": "1.0",
            "metadata": {
                "k_resample_frames": self.config.k_resample,
                "feature_dim": inferred_feature_dim,
                "created_at": existing_created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": "Dynamic VSL Templates for Hybrid DTW Engine"
            },
            "hyperparameters": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "lambda_shape": self.config.lambda_shape,
                "default_threshold": self.config.default_threshold
            },
            "vocabulary": vocab_out
        }

        try:
            with open(self.templates_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[DTWEngine] Đã lưu thành công vào {self.templates_path}")
            return True
        except Exception as e:
            print(f"[DTWEngine] Lỗi lưu template ({e})")
            return False

    def add_sample(
        self,
        word_key: str,
        display_name: str,
        sample: ResampledGesture,
        threshold: Optional[float] = None
    ):
        """Thêm một mẫu chuẩn mới vào từ điển."""
        if word_key not in self.vocabulary:
            self.vocabulary[word_key] = {
                "display_name": display_name,
                "threshold": threshold or self.config.default_threshold,
                "samples": []
            }

        self.vocabulary[word_key]["samples"].append(sample)
        if threshold is not None:
            self.vocabulary[word_key]["threshold"] = threshold

    def match(self, input_gesture: ResampledGesture) -> MatchResult:
        """
        So khớp cử chỉ đầu vào với toàn bộ cơ sở dữ liệu từ vựng.
        
        Quyết định phân tầng:
          1. best_dist <= threshold               -> ACCEPTED (Chấp nhận)
          2. threshold < best_dist <= 1.4*thresh   -> UNCERTAIN (Nghi vấn Top-2)
          3. best_dist > 1.4*thresh               -> REJECTED (Từ chối)
        """
        if not self.vocabulary:
            return MatchResult(
                status="REJECTED",
                top_word=None,
                top_display=None,
                best_distance=999.0,
                best_confidence=0.0
            )

        candidates: List[MatchCandidate] = []

        # So khớp với từng từ vựng trong cơ sở dữ liệu
        for w_key, w_info in self.vocabulary.items():
            samples = w_info["samples"]
            if not samples:
                continue

            # Khoảng cách của từ vựng là giá trị nhỏ nhất trong số các mẫu của từ đó
            min_word_dist = 999.0
            for tmpl in samples:
                dist = compute_hybrid_dtw_distance(input_gesture, tmpl, self.config)
                if dist < min_word_dist:
                    min_word_dist = dist

            thresh = w_info["threshold"]
            # Điểm tin cậy quy đổi [0.0, 1.0]
            conf = max(0.0, min(1.0, 1.0 - (min_word_dist / (thresh * 1.5 + 1e-6))))

            candidates.append(MatchCandidate(
                word_key=w_key,
                display_name=w_info["display_name"],
                distance=round(min_word_dist, 3),
                confidence=round(conf, 3)
            ))

        if not candidates:
            return MatchResult(
                status="REJECTED",
                top_word=None,
                top_display=None,
                best_distance=999.0,
                best_confidence=0.0
            )

        # Sắp xếp ứng viên theo khoảng cách nhỏ nhất
        candidates.sort(key=lambda c: c.distance)
        top_cand = candidates[0]
        top_thresh = self.vocabulary[top_cand.word_key]["threshold"]

        # Chiến lược phân tầng 3 mức
        if top_cand.distance <= top_thresh:
            status = "ACCEPTED"
        elif top_cand.distance <= top_thresh * self.config.uncertainty_ratio:
            status = "UNCERTAIN"
        else:
            status = "REJECTED"

        return MatchResult(
            status=status,
            top_word=top_cand.word_key if status != "REJECTED" else None,
            top_display=top_cand.display_name if status != "REJECTED" else None,
            best_distance=top_cand.distance,
            best_confidence=top_cand.confidence,
            candidates=candidates[:3]  # Trả về Top-3 ứng viên gần nhất
        )


# =============================================================================
# 6. BỘ NÃO GOLDEN TRIAD 12D (GoldenTriadDTWMatcher) — ENGINE THẬT ĐANG CHẠY PRODUCTION
# =============================================================================
# Được dời từ predict_sentence_webcam.py (v5.1) sang đây để dynamic_dtw_engine.py trở
# thành nơi tập trung duy nhất cho mọi engine so khớp DTW (cả 4D legacy lẫn 12D thật).
# predict_sentence_webcam.py giờ chỉ import lại 3 tên dưới đây (re-export tự nhiên qua
# Python import), không định nghĩa cục bộ nữa — collect_dynamic_templates.py không cần
# sửa gì vì nó cũng import các tên này từ predict_sentence_webcam.py.

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


def _compute_dominant_shape(shapes: List[str]) -> str:
    """Nhãn dáng tay chiếm đa số trong toàn bộ chuỗi mẫu (majority vote), bỏ qua 'Unknown'."""
    known = [s for s in shapes if s != "Unknown"]
    if not known:
        return "Unknown"
    counts: Dict[str, int] = {}
    for s in known:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


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
    lambda_shape_global: float = 0.0,
    sakoe_chiba_radius: int = 4,
    w_z: float = 0.0,
    w_b: float = 0.0,
    w_s: float = 0.0,
    w_r: float = 0.0,
    q_rep_count: Optional[int] = None,
    t_rep_count: Optional[int] = None
) -> float:
    """
    Tính khoảng cách DTW 12D/17D/24D với dải băng Sakoe-Chiba và chuẩn hóa theo căn bậc hai số chiều:
      Cost(i, j) = w_p * ||P_q - P_t|| / sqrt(2)
                 + w_m * ||JM_q - JM_t|| / sqrt(2)
                 + w_a * d_angle
                 + w_n * (1.0 - dot(n_q, n_t)) / 2.0  (Cosine distance)
                 + lambda_shape * I(shape_q != shape_t)

    Trong đó d_angle = ||A_q - A_t|| / sqrt(5) (với dữ liệu 12D cũ) hoặc là trung bình cộng
    giữa góc tĩnh và vận tốc góc: (||A_q - A_t||/sqrt(5) + ||dA_q - dA_t||/sqrt(5)) / 2.0
    khi dữ liệu là 17D (tương thích ngược hoàn toàn).

    SPEC 03 — 4 nhóm MỚI (CỘNG THÊM, mặc định trọng số 0.0 => kết quả GIỐNG HỆT từng bit như cũ):
      + w_z * |z_q - z_t|                          (cột 17, độ sâu ảo, chia 1.0)
      + w_b * ||body_q - body_t|| / sqrt(2)        (cột 18:20, mốc cơ thể)
      + w_s * ||abd_q - abd_t|| / 2.0              (cột 20:24, độ xòe ngón, sqrt(4) = 2)
      Nhóm nào thiếu cột ở một trong hai phía (has_* == False) thì bị bỏ qua (coi như 0).
      + w_r * |q_rep_count - t_rep_count|          (số nhịp lặp: thuộc tính TOÀN CỤC của chuỗi nên chỉ
        cộng MỘT LẦN sau đường DTW, cạnh lambda_shape_global — nhồi thành cột hằng số thì DTW chỉ cộng
        một offset không đổi, vô dụng).

    TỐI ƯU HÓA v2 (NumPy vectorized): phần chi phí cục bộ (d_pos, d_dir, d_angle,
    d_norm, d_shape) được tính hàng loạt bằng NumPy cho toàn bộ dải băng của mỗi
    hàng i cùng lúc, thay vì gọi math.hypot/np.dot từng cặp (i, j) một trong vòng
    lặp Python thuần. Chỉ phần tích lũy min() của công thức đệ quy DTW (có phụ
    thuộc dữ liệu tuần tự theo j) vẫn giữ vòng lặp scalar. Kết quả số học giống
    hệt bản gốc, chỉ giảm overhead Python khi vốn từ vựng mở rộng.
    """
    k = len(q_traj)
    r = sakoe_chiba_radius

    d_mat = np.full((k + 1, k + 1), np.inf, dtype=np.float32)
    d_mat[0, 0] = 0.0

    has_template_angles = (t_traj.shape[1] >= 9 and q_traj.shape[1] >= 9)
    has_template_normal = (t_traj.shape[1] >= 12 and q_traj.shape[1] >= 12)
    has_template_delta_angle = (t_traj.shape[1] >= 17 and q_traj.shape[1] >= 17)
    # SPEC 03: phát hiện khả dụng của 3 nhóm cột mới theo đúng khuôn has_template_delta_angle.
    has_z = (t_traj.shape[1] >= 18 and q_traj.shape[1] >= 18)
    has_body = (t_traj.shape[1] >= 20 and q_traj.shape[1] >= 20)
    has_abd = (t_traj.shape[1] >= 24 and q_traj.shape[1] >= 24)
    # Trọng số bằng 0 => bỏ hẳn phép tính (cộng 0.0 vào float là chính xác, nên vẫn bit-identical, mà nhanh hơn).
    use_z = has_z and w_z != 0.0
    use_body = has_body and w_b != 0.0
    use_abd = has_abd and w_s != 0.0

    t_shapes_arr = np.asarray(t_shapes, dtype=object)
    q_shapes_arr = np.asarray(q_shapes, dtype=object)

    SQRT2 = 1.41421356
    SQRT5 = 2.23606798

    for i in range(1, k + 1):
        j_start = max(1, i - r)
        j_end = min(k, i + r)
        j_idx = np.arange(j_start, j_end + 1)
        t_slice = t_traj[j_idx - 1]
        q_row = q_traj[i - 1]

        d_pos = np.hypot(q_row[0] - t_slice[:, 0], q_row[1] - t_slice[:, 1]) / SQRT2
        d_dir = np.hypot(q_row[2] - t_slice[:, 2], q_row[3] - t_slice[:, 3]) / SQRT2

        if has_template_angles:
            diff_angles = q_row[4:9] - t_slice[:, 4:9]
            d_angle = np.linalg.norm(diff_angles, axis=1) / SQRT5
        else:
            d_angle = np.zeros(len(j_idx), dtype=np.float32)

        if has_template_delta_angle:
            diff_dangles = q_row[12:17] - t_slice[:, 12:17]
            d_delta_angle = np.linalg.norm(diff_dangles, axis=1) / SQRT5
            # Lấy trung bình cộng giữa góc tĩnh và vận tốc góc để giữ đúng thang đo chi phí
            # của nhóm góc (w_a) — không cộng thẳng vào d_angle vì nếu cộng thẳng thì
            # ngân sách chi phí của nhóm A sẽ tăng gấp đôi một cách âm thầm.
            d_angle = (d_angle + d_delta_angle) / 2.0

        if has_template_normal:
            nq = q_row[9:12]
            nt = t_slice[:, 9:12]
            cos_sim = np.clip(nt @ nq, -1.0, 1.0)
            d_norm = (1.0 - cos_sim) / 2.0
        else:
            d_norm = np.zeros(len(j_idx), dtype=np.float32)

        # "Unknown" nghĩa là CNN không phân loại được dáng tay ở frame đó (thiếu
        # dữ liệu do crop quá nhỏ, model thiếu, khung hình mờ...) — KHÔNG có nghĩa
        # là dáng tay thực sự khác nhau. Trung lập hóa (không phạt) khi 1 trong 2
        # phía là "Unknown", tránh phạt kép các frame vốn đã kém chất lượng.
        t_slice_shapes = t_shapes_arr[j_idx - 1]
        q_shape_val = q_shapes_arr[i - 1]
        is_unknown = (t_slice_shapes == "Unknown") | (q_shape_val == "Unknown")
        raw_mismatch = np.where(t_slice_shapes == q_shape_val, 0.0, lambda_shape)
        d_shape = np.where(is_unknown, 0.0, raw_mismatch)

        step_cost = w_p * d_pos + w_m * d_dir + w_a * d_angle + w_n * d_norm + d_shape

        # SPEC 03: cộng thêm 3 nhóm mới (vectorized, chuẩn hóa chia sqrt(số chiều) như các nhóm cũ).
        if use_z:
            d_z = np.abs(q_row[17] - t_slice[:, 17])
            step_cost = step_cost + w_z * d_z
        if use_body:
            d_body = np.hypot(q_row[18] - t_slice[:, 18], q_row[19] - t_slice[:, 19]) / SQRT2
            step_cost = step_cost + w_b * d_body
        if use_abd:
            d_abd = np.linalg.norm(q_row[20:24] - t_slice[:, 20:24], axis=1) / 2.0
            step_cost = step_cost + w_s * d_abd

        for local_idx, j in enumerate(j_idx):
            d_mat[i, j] = step_cost[local_idx] + min(
                d_mat[i - 1, j],
                d_mat[i, j - 1],
                d_mat[i - 1, j - 1]
            )

    total_cost = d_mat[k, k]
    if math.isinf(total_cost):
        return 999.0
    base_distance = float(total_cost / float(k))

    # Hình phạt dáng tay CHIẾM ĐA SỐ (majority vote) tính 1 LẦN cho cả chuỗi —
    # khác với lambda_shape (per-frame, đã tắt do nhiễu CNN quá cao theo audit).
    # Đây là tín hiệu ổn định hơn nhiều: dù từng frame CNN có nhiễu, nhãn chiếm
    # đa số của cả chuỗi vẫn phản ánh đúng xu hướng dáng tay tổng thể.
    if lambda_shape_global > 0.0:
        q_dominant = _compute_dominant_shape(q_shapes)
        t_dominant = _compute_dominant_shape(t_shapes)
        if q_dominant != "Unknown" and t_dominant != "Unknown" and q_dominant != t_dominant:
            base_distance += lambda_shape_global

    # SPEC 03: số nhịp lặp là thuộc tính TOÀN CỤC của chuỗi => cộng MỘT LẦN, sau khi đã có đường DTW.
    if w_r > 0.0 and q_rep_count is not None and t_rep_count is not None:
        base_distance += w_r * abs(int(q_rep_count) - int(t_rep_count))

    return base_distance


class GoldenTriadDTWMatcher:
    """
    Bộ não so khớp cử chỉ v5.1 tích hợp Golden Triad [P + A + JM].
    Tương thích ngược 100% với cả template 4D cũ và template 9D mới.
    """
    def __init__(
        self,
        templates_path: str = "dynamic_templates.json",
        sign_weights_path: Optional[str] = None,
        w_p: float = 0.6,
        w_a: float = 1.0,
        w_m: float = 0.4,
        w_n: float = 0.8,
        lambda_shape: float = 0.0,
        lambda_shape_global: float = 0.0,  # Tắt: thực nghiệm cho thấy nhãn CNN dominant-shape KHÔNG ổn định giữa các mẫu cùng từ (vd MOTHER: 4 mẫu ra 4 nhãn khác nhau C/V/B/A) — bật lên làm tổng mẫu nghi vấn toàn hệ thống tăng 46->74 dù cục bộ giúp BYE. Giữ code cơ chế lại (không xóa) để có thể bật lại thử nghiệm sau nếu cần.
        margin_threshold: float = 0.12,
        uncertainty_ratio: float = 1.40,
        threshold_override: Optional[float] = None,
        strict_weights: bool = True,
        asllex_blend_alpha: float = 1.0,
        asllex_max_deviation: Optional[float] = None,
        pair_weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        w_z: float = 0.0,
        w_b: float = 0.0,
        w_s: float = 0.0,
        w_r: float = 0.0,
        templates_dict: Optional[Dict[str, Any]] = None,
        asllex_mode: str = "replace",
        asllex_lambda: float = 0.0,
        asllex_mapping: str = "legacy",
        flip_body_x: bool = True,
        threshold_dict: Optional[Dict[str, float]] = None
    ):
        self.templates_path = templates_path
        self.sign_weights_path = sign_weights_path if sign_weights_path is not None else DEFAULT_SIGN_WEIGHTS_PATH
        self.w_p = w_p
        self.w_a = w_a
        self.w_m = w_m
        self.w_n = w_n
        # SPEC 03: 4 trọng số nhóm MỚI. KHÔNG có trong file ASL-LEX nên KHÔNG đi qua bước blend
        # (_get_weights_for_sign chỉ trả 4 kênh cũ) — chúng luôn dùng đúng giá trị toàn cục này.
        self.w_z = w_z
        self.w_b = w_b
        self.w_s = w_s
        self.w_r = w_r
        self.lambda_shape = lambda_shape
        self.lambda_shape_global = lambda_shape_global
        self.margin_threshold = margin_threshold
        self.uncertainty_ratio = uncertainty_ratio
        self.threshold_override = threshold_override
        self.strict_weights = strict_weights
        self.blend_alpha = float(asllex_blend_alpha)
        self.max_deviation = asllex_max_deviation
        self.pair_weight_overrides = pair_weight_overrides if pair_weight_overrides is not None else {
            "FATHER": {"w_p": 1.3},
            "MOTHER": {"w_p": 1.3},
            "THANK_YOU": {"w_p": 1.3},
            "GOOD": {"w_p": 1.3},
        }
        self.vocabulary: Dict[str, Any] = {}
        self.sign_weights: Dict[str, Dict[str, float]] = {}
        self.feature_dim: int = 4
        self.is_legacy_4d: bool = True

        self.templates_dict = templates_dict
        self.asllex_mode = str(asllex_mode).lower()
        if self.asllex_mode == "replace" and asllex_lambda == 0.0:
            self.asllex_lambda = float(asllex_blend_alpha)
        else:
            self.asllex_lambda = float(asllex_lambda)
        self.asllex_mapping = str(asllex_mapping).lower()
        self.flip_body_x = bool(flip_body_x)
        self.threshold_dict = threshold_dict

        self.load_sign_weights()
        if self.templates_dict is not None:
            self.set_vocabulary(
                self.templates_dict.get("vocabulary", self.templates_dict),
                self.templates_dict.get("metadata")
            )
        else:
            self.load_templates()
        self.validate_weights_against_vocabulary()

    def load_sign_weights(self) -> None:
        """
        Nạp trọng số Per-Sign DTW từ file JSON tạo bởi ASL-LEX 2.0.
        Nếu sign_weights_path là None, tự động kích hoạt chế độ Global Weights (1 công thức chung từ CONFIG).
        """
        if not self.sign_weights_path:
            self.sign_weights = {}
            print(f"ℹ️ [DTW] Chế độ Global Weights: Dùng chung 1 công thức CONFIG cho tất cả ký hiệu "
                  f"(w_p={self.w_p}, w_a={self.w_a}, w_m={self.w_m}, w_n={self.w_n}).")
            return

        if not os.path.exists(self.sign_weights_path):
            if self.strict_weights:
                raise FileNotFoundError(f"[!] Không tìm thấy file trọng số ASL-LEX: {self.sign_weights_path}")
            print(f"[!] CẢNH BÁO: Không tìm thấy {self.sign_weights_path}. Sử dụng trọng số toàn cục CONFIG.")
            return

        try:
            with open(self.sign_weights_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            parsed_weights = {}
            for raw_key, w_entry in raw_data.items():
                canonical_k = str(raw_key).strip().upper()
                # Hỗ trợ cả schema trực tiếp {"P":..., "A":...} hoặc lồng {"weights": {"P":...}}
                w_dict = w_entry.get("weights", w_entry) if isinstance(w_entry, dict) else {}

                for required_field in ["P", "A", "JM", "N"]:
                    if required_field not in w_dict:
                        raise ValueError(f"[!] Ký hiệu '{canonical_k}' thiếu trường trọng số '{required_field}'!")

                p = float(w_dict["P"])
                a = float(w_dict["A"])
                jm = float(w_dict["JM"])
                n = float(w_dict["N"])

                for val, name in [(p, "P"), (a, "A"), (jm, "JM"), (n, "N")]:
                    if val < 0.0 or math.isnan(val) or math.isinf(val):
                        raise ValueError(f"[!] Ký hiệu '{canonical_k}' có trọng số '{name}' không hợp lệ: {val}")

                weight_sum = p + a + jm + n
                if abs(weight_sum - 1.0) > 1e-4:
                    raise ValueError(f"[!] Ký hiệu '{canonical_k}' có tổng trọng số != 1.0 (Tổng={weight_sum:.6f})!")

                parsed_weights[canonical_k] = {
                    "P": p, "A": a, "JM": jm, "N": n
                }

            self.sign_weights = parsed_weights
            print(f"✓ [DTW] Đã nạp thành công {len(self.sign_weights)} bộ trọng số Per-Sign từ '{self.sign_weights_path}'.")
        except Exception as e:
            if self.strict_weights:
                raise
            print(f"[!] Lỗi đọc {self.sign_weights_path}: {e}")

    # Ánh xạ bí danh tương thích ngược giữa các phiên bản template (VSL tiếng Việt <-> ASL tiếng Anh)
    SIGN_KEY_ALIASES: Dict[str, str] = {
        "XIN_CHAO": "HELLO",
        "CAM_ON": "THANK_YOU",
        "TOI": "ME",
        "BAN": "YOU",
        "TAM_BIET": "BYE",
        "GOOD-BYE": "BYE",
        "GOOD_BYE": "BYE",
    }

    def validate_weights_against_vocabulary(self) -> None:
        """
        Kiểm tra tính nhất quán 2 chiều giữa từ điển templates và bảng trọng số ASL-LEX.
        Triệt tiêu hoàn toàn lỗi âm thầm (Silent Failure / Key Mismatch).
        Tự động hỗ trợ ánh xạ bí danh cho các phiên bản template cũ.
        """
        if getattr(self, "asllex_mode", "replace") == "none":
            return
        if not self.sign_weights_path or not self.vocabulary or not self.sign_weights:
            return

        missing_weights = []
        for word_key in self.vocabulary.keys():
            raw_k = str(word_key).strip().upper()
            canonical_k = self.SIGN_KEY_ALIASES.get(raw_k, raw_k)
            if canonical_k not in self.sign_weights:
                missing_weights.append(word_key)

        if missing_weights:
            msg = (
                f"[CRITICAL] Phát hiện {len(missing_weights)} từ vựng trong '{self.templates_path}' "
                f"không có trọng số trong '{self.sign_weights_path}': {missing_weights}\n"
                f"Vui lòng đồng bộ Canonical Key hoặc bổ sung trọng số ASL-LEX!"
            )
            if self.strict_weights:
                raise KeyError(msg)
            else:
                print(f"[!] CẢNH BÁO: {msg}")

    def _get_weights_for_sign_replace(self, word_key: str) -> Tuple[float, float, float, float]:
        """Hành vi replace hiện tại của GoldenTriadDTWMatcher (giữ nguyên 100% logic cũ)."""
        base = np.array([self.w_p, self.w_a, self.w_m, self.w_n], dtype=np.float64)
        base_sum = float(base.sum())

        if not self.sign_weights_path or not self.sign_weights:
            return self.w_p, self.w_a, self.w_m, self.w_n

        raw_k = str(word_key).strip().upper()
        canonical_k = self.SIGN_KEY_ALIASES.get(raw_k, raw_k)
        if canonical_k not in self.sign_weights:
            print(f"[!] CẢNH BÁO FALLBACK: Ký hiệu '{word_key}' (Canonical: '{canonical_k}') không có Per-Sign Weight! "
                  f"Tạm thời dùng trọng số CONFIG (w_p={self.w_p}, w_a={self.w_a}, w_m={self.w_m}, w_n={self.w_n}).")
            return self.w_p, self.w_a, self.w_m, self.w_n

        sw = self.sign_weights[canonical_k]
        asllex = np.array([sw["P"], sw["A"], sw["JM"], sw["N"]], dtype=np.float64)  # tổng đã = 1.0 (đảm bảo bởi load_sign_weights)

        # Bước 1: chuẩn hóa base về tổng = 1.0 (tỷ lệ tương đối giữa 4 kênh)
        base_norm = base / base_sum if base_sum > 1e-9 else base

        # Bước 2: trộn tuyến tính theo alpha (cả 2 vế đều có tổng = 1.0 nên phép trộn hợp lệ)
        blend_norm = (1.0 - self.blend_alpha) * base_norm + self.blend_alpha * asllex

        # Bước 3: nội suy CẢ độ lớn tổng theo alpha (không cố định về base_sum),
        # đảm bảo alpha=1.0 trả về ĐÚNG HỆT giá trị ASL-LEX thô (tương thích ngược tuyệt đối),
        # và alpha=0.0 trả về ĐÚNG HỆT giá trị mặc định gốc.
        target_sum = (1.0 - self.blend_alpha) * base_sum + self.blend_alpha * 1.0
        w_final = blend_norm * target_sum

        # Bước 4: giới hạn lệch tối đa so với trọng số mặc định (nếu bật)
        if self.max_deviation is not None:
            lower = base * (1.0 - self.max_deviation)
            upper = base * (1.0 + self.max_deviation)
            w_final = np.clip(w_final, lower, upper)

        override = self.pair_weight_overrides.get(canonical_k)
        if override:
            idx_map = {"w_p": 0, "w_a": 1, "w_m": 2, "w_n": 3}
            for ch_name, mult in override.items():
                if ch_name in idx_map:
                    w_final[idx_map[ch_name]] *= mult

        return float(w_final[0]), float(w_final[1]), float(w_final[2]), float(w_final[3])

    def _get_effective_weights(self, word_key: str) -> Dict[str, float]:
        """
        Truy xuất bộ 8 trọng số hiệu dụng [w_p, w_a, w_m, w_n, w_z, w_b, w_s, w_r] theo candidate sign.
        Hỗ trợ 3 chế độ:
          - "none": chỉ dùng weights toàn cục (không đọc ASL-LEX).
          - "replace": hành vi production hiện tại (blend tuyến tính giữa base và ASL-LEX, cộng overrides).
          - "modulate": chế độ MỚI từ SPEC 04, điều biến w_p, w_a, w_m, w_n, w_b, w_s theo ASL-LEX,
                        rồi chuẩn hóa để tổng trọng số nhóm bằng tổng toàn cục. w_z và w_r không bị điều biến.
                        lambda = 0 cho kết quả giống hệt bit chế độ "none".
        """
        w_global = {
            "w_p": float(self.w_p),
            "w_a": float(self.w_a),
            "w_m": float(self.w_m),
            "w_n": float(self.w_n),
            "w_z": float(self.w_z),
            "w_b": float(self.w_b),
            "w_s": float(self.w_s),
            "w_r": float(self.w_r),
        }

        if getattr(self, "asllex_mode", "replace") == "none":
            return w_global

        raw_k = str(word_key).strip().upper()
        canonical_k = self.SIGN_KEY_ALIASES.get(raw_k, raw_k)

        if self.asllex_mode == "replace":
            wp, wa, wm, wn = self._get_weights_for_sign_replace(word_key)
            res = w_global.copy()
            res["w_p"] = wp
            res["w_a"] = wa
            res["w_m"] = wm
            res["w_n"] = wn
            return res

        if self.asllex_mode == "modulate":
            res = w_global.copy()
            mod_keys = ["w_p", "w_a", "w_m", "w_n", "w_b", "w_s"]
            if not self.sign_weights or canonical_k not in self.sign_weights or self.asllex_lambda == 0.0:
                return res

            sw = self.sign_weights[canonical_k]
            if self.asllex_mapping == "semantic":
                share_g = {
                    "w_b": sw["P"],
                    "w_a": sw["A"],
                    "w_s": sw["A"],
                    "w_p": sw["JM"],
                    "w_m": sw["JM"],
                    "w_n": sw["N"]
                }
            else:  # legacy
                share_g = {
                    "w_p": sw["P"],
                    "w_a": sw["A"],
                    "w_m": sw["JM"],
                    "w_n": sw["N"],
                    "w_b": 0.25,
                    "w_s": 0.25
                }

            sum_global = sum(w_global[k] for k in mod_keys)
            for k in mod_keys:
                share = share_g.get(k, 0.25)
                factor_k = float(np.clip(1.0 + self.asllex_lambda * (share / 0.25 - 1.0), 0.5, 1.5))
                res[k] = w_global[k] * factor_k

            sum_eff = sum(res[k] for k in mod_keys)
            if sum_eff > 1e-9 and sum_global > 1e-9:
                scale = sum_global / sum_eff
                for k in mod_keys:
                    res[k] *= scale
            return res

        return w_global

    def _get_weights_for_sign(self, word_key: str) -> Tuple[float, float, float, float]:
        eff = self._get_effective_weights(word_key)
        return float(eff["w_p"]), float(eff["w_a"]), float(eff["w_m"]), float(eff["w_n"])

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

            self._canonicalize_loaded_templates()
        except Exception as e:
            print(f"[!] Lỗi đọc {self.templates_path}: {e}")

    def set_vocabulary(self, vocabulary: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Gán trực tiếp từ điển mẫu in-memory (dùng cho cross-validation / LOSO / kiểm định).
        Không cần ghi và đọc lại file JSON. Tự động chuẩn hóa chirality một lần cho toàn bộ mẫu.
        """
        import copy
        self.vocabulary = copy.deepcopy(vocabulary)
        meta = metadata or {}
        self.feature_dim = meta.get("feature_dim", 24 if self.vocabulary else 4)
        self.is_legacy_4d = (self.feature_dim < 9)
        self._canonicalize_loaded_templates()
        self.validate_weights_against_vocabulary()

    def _canonicalize_loaded_templates(self) -> None:
        """
        Chuẩn hóa chirality MỘT LẦN cho toàn bộ template vừa nạp, để template và query
        luôn được so khớp trong CÙNG một không gian dấu (xem canonicalize_chirality).

        - Mẫu đã có cờ "chirality_canonical": true  -> bỏ qua (đã chuẩn hóa từ lúc ghi).
        - Mẫu có trường "handedness"                -> dùng nhánh chính thống.
        - Mẫu cũ không có gì                        -> dùng nhánh heuristic max|nx|.
        """
        n_total = 0
        n_flipped = 0
        n_skipped = 0
        n_by_label = 0

        flip_body = getattr(self, "flip_body_x", True)
        for word_data in self.vocabulary.values():
            for s in word_data.get("samples", []):
                if not isinstance(s, dict):
                    continue
                key = "traj_features_9d" if s.get("traj_features_9d") is not None else "traj_features"
                raw = s.get(key)
                if raw is None:
                    continue
                n_total += 1

                if s.get("chirality_canonical") is True:
                    n_skipped += 1
                    continue

                traj = np.array(raw, dtype=np.float32)
                handed = s.get("handedness")
                if handed is not None:
                    n_by_label += 1
                canon, was_flipped = canonicalize_chirality(traj, handedness=handed, flip_body_x=flip_body)
                if was_flipped:
                    s[key] = canon.tolist()
                    n_flipped += 1
                s["chirality_canonical"] = True

        if n_total > 0:
            print(f"[Chirality] Đã chuẩn hóa {n_total} template: lật {n_flipped}, "
                  f"bỏ qua {n_skipped} (đã chuẩn tắc), {n_by_label} dùng nhãn tay thuận "
                  f"(còn lại dùng heuristic max|nx|).")

    @staticmethod
    def compute_margin_ratio(d_top1: float, d_top2: float, eps: float = 1e-4) -> float:
        """Relative Contrast Margin: (D2 - D1) / max(D1, eps)."""
        return float((d_top2 - d_top1) / max(d_top1, eps))

    def match(
        self,
        query_traj_9d: np.ndarray,
        query_shapes: List[str],
        handedness: Optional[str] = None,
        query_has_body: bool = True,
        query_rep_count: Optional[int] = None
    ) -> MatchResult:
        """
        So khớp cử chỉ đầu vào với toàn bộ từ vựng.

        Tham số handedness (MỚI, mặc định None nên mọi lời gọi cũ vẫn chạy y nguyên):
        nhãn tay thuận MediaPipe của cử chỉ đang truy vấn. Query được chuẩn hóa chirality
        bằng CÙNG hàm canonicalize_chirality mà template đã đi qua ở load_templates(),
        nếu không thì hai bên nằm ở hai không gian dấu khác nhau và w_n = 0.8 sẽ tự động
        loại đúng từ cần nhận (xem docstring canonicalize_chirality).

        SPEC 03 (2 tham số mới, mặc định giữ nguyên hành vi cũ):
          query_has_body : False khi stroke của query KHÔNG dò được mặt => nhóm mốc cơ thể bị
                           vô hiệu (w_b = 0) cho toàn bộ lần so khớp này. Template có cờ
                           "body_anchor_valid": False cũng bị vô hiệu w_b riêng cho template đó.
          query_rep_count: số nhịp lặp của query; None = tự tính từ cột 0, 1 của chuỗi K=32.
        """
        if query_traj_9d is not None:
            query_traj_9d, _ = canonicalize_chirality(query_traj_9d, handedness=handedness, flip_body_x=getattr(self, "flip_body_x", True))

        # Số nhịp lặp của query chỉ cần khi w_r > 0 (tránh tốn phép tính khi nhóm này đang tắt).
        q_rep = query_rep_count
        if q_rep is None and self.w_r > 0.0 and query_traj_9d is not None and len(query_traj_9d) > 0:
            q_rep = compute_repetition_count(query_traj_9d[:, 0], query_traj_9d[:, 1])

        if not self.vocabulary:
            return MatchResult(status="REJECTED", top_word=None, top_display=None, best_distance=999.0, best_confidence=0.0, candidates=[])

        all_candidates: List[MatchCandidate] = []

        for word_key, word_data in self.vocabulary.items():
            display_name = word_data.get("display_name", word_key)
            if self.threshold_override is not None:
                threshold = self.threshold_override
            elif self.threshold_dict is not None and word_key in self.threshold_dict:
                threshold = float(self.threshold_dict[word_key])
            else:
                threshold = float(word_data.get("threshold", 1.25))

            samples = word_data.get("samples", [])
            if not samples:
                continue

            # Lấy vector trọng số hiệu dụng cho candidate sign này
            eff = self._get_effective_weights(word_key)
            wp = eff["w_p"]
            wa = eff["w_a"]
            wm = eff["w_m"]
            wn = eff["w_n"]
            wb = eff["w_b"]
            ws = eff["w_s"]
            wz = eff["w_z"]
            wr = eff["w_r"]

            sample_distances = []
            for s in samples:
                t_traj_raw = s.get("traj_features_9d")
                if t_traj_raw is None:
                    t_traj_raw = s.get("traj_features", [])

                t_traj = np.array(t_traj_raw, dtype=np.float32)
                t_shapes = s.get("hand_shapes", ["Unknown"] * 32)

                if len(t_traj) != 32:
                    continue

                # SPEC 03: w_b bị vô hiệu nếu query HOẶC template không có mốc cơ thể hợp lệ.
                wb_eff = wb if (query_has_body and s.get("body_anchor_valid", True)) else 0.0

                dist = compute_golden_triad_dtw_distance(
                    q_traj=query_traj_9d,
                    q_shapes=query_shapes,
                    t_traj=t_traj,
                    t_shapes=t_shapes,
                    w_p=wp,
                    w_a=wa,
                    w_m=wm,
                    w_n=wn,
                    lambda_shape=self.lambda_shape,
                    lambda_shape_global=self.lambda_shape_global,
                    w_z=wz,
                    w_b=wb_eff,
                    w_s=ws,
                    w_r=wr,
                    q_rep_count=q_rep,
                    t_rep_count=s.get("rep_count")
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
        if self.threshold_override is not None:
            word_thresh = self.threshold_override
        elif self.threshold_dict is not None and top1.word_key in self.threshold_dict:
            word_thresh = float(self.threshold_dict[top1.word_key])
        else:
            word_thresh = base_thresh

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
# 7. TỰ ĐỘNG KIỂM THỬ ĐỘC LẬP (SELF-CONTAINED UNIT TEST)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  KIỂM THỬ ĐỘC LẬP: DYNAMIC DTW ENGINE & KINETIC SEGMENTER")
    print("=" * 70)

    # 1. Kiểm tra Bất biến Tốc độ (Fast vs Slow Gesture)
    print("\n[TEST 1] Kiểm tra tính Bất biến Tốc độ (Time-scale Invariance):")
    # Tạo cử chỉ mẫu: Cung tròn theo chiều kim đồng hồ (30 frames)
    t_slow = np.linspace(0, np.pi, 30)
    slow_frames = [
        GestureFrame(x=100 + 50 * np.cos(t), y=200 + 50 * np.sin(t), palm_len=60.0, hand_shape="B", timestamp=i * 0.033)
        for i, t in enumerate(t_slow)
    ]
    # Tạo cùng cử chỉ đó nhưng người làm lướt rất nhanh (12 frames)
    t_fast = np.linspace(0, np.pi, 12)
    fast_frames = [
        GestureFrame(x=350 + 50 * np.cos(t), y=150 + 50 * np.sin(t), palm_len=60.0, hand_shape="B", timestamp=i * 0.033)
        for i, t in enumerate(t_fast)
    ]

    res_slow = resample_gesture_sequence(slow_frames, k_target=32)
    res_fast = resample_gesture_sequence(fast_frames, k_target=32)
    assert res_slow is not None and res_fast is not None

    cfg = HybridDTWConfig()
    dist_speed = compute_hybrid_dtw_distance(res_slow, res_fast, cfg)
    print(f"  -> Khoảng cách giữa cử chỉ CHẬM (30 frames) và NHANH (12 frames): {dist_speed:.4f}")
    assert dist_speed < 0.20, f"Thất bại: Khoảng cách quá lớn ({dist_speed})"
    print("  ✓ PASS: Bất biến tốc độ hoạt động xuất sắc!")

    # 2. Kiểm tra Phân biệt Chiều Di chuyển (Direction Sensitivity via dx, dy)
    print("\n[TEST 2] Kiểm tra Phân biệt Chiều Di chuyển (Cùng hình dạng nhưng ngược chiều):")
    # Cử chỉ đi từ Trái sang Phải
    t_l2r = np.linspace(0, 100, 20)
    l2r_frames = [
        GestureFrame(x=100 + t, y=200.0, palm_len=60.0, hand_shape="B", timestamp=i * 0.033)
        for i, t in enumerate(t_l2r)
    ]
    # Cử chỉ đi từ Phải sang Trái
    t_r2l = np.linspace(100, 0, 20)
    r2l_frames = [
        GestureFrame(x=100 + t, y=200.0, palm_len=60.0, hand_shape="B", timestamp=i * 0.033)
        for i, t in enumerate(t_r2l)
    ]

    res_l2r = resample_gesture_sequence(l2r_frames, 32)
    res_r2l = resample_gesture_sequence(r2l_frames, 32)
    dist_dir = compute_hybrid_dtw_distance(res_l2r, res_r2l, cfg)
    print(f"  -> Khoảng cách giữa chiều Trái->Phải và Phải->Trái: {dist_dir:.4f}")
    assert dist_dir > 0.50, f"Thất bại: Không phân biệt được chiều di chuyển ({dist_dir})"
    print("  ✓ PASS: Vector hướng tiếp tuyến (dx, dy) đã ngăn chặn nhầm lẫn ngược chiều!")

    # 3. Kiểm tra Hình phạt Khác biệt Dáng tay (Hand Shape Penalty)
    print("\n[TEST 3] Kiểm tra Hình phạt Dáng tay (Cùng quỹ đạo nhưng khác dáng ngón tay):")
    fist_frames = [
        GestureFrame(x=100 + t, y=200.0, palm_len=60.0, hand_shape="A", timestamp=i * 0.033)
        for i, t in enumerate(t_l2r)
    ]
    res_fist = resample_gesture_sequence(fist_frames, 32)
    dist_shape = compute_hybrid_dtw_distance(res_l2r, res_fist, cfg)
    print(f"  -> Khoảng cách khi đổi dáng tay từ Xòe B sang Nắm A: {dist_shape:.4f}")
    assert dist_shape > dist_speed + 0.40
    print("  ✓ PASS: Hình phạt dáng ngón tay lambda_shape đã hoạt động chuẩn xác!")

    # 4. Kiểm tra Lệnh Kích hoạt Nhanh Swipe-Left (Backspace)
    print("\n[TEST 4] Kiểm tra Bộ phát hiện Swipe-Left (Backspace):")
    seg = KineticPauseSegmenter()
    # Giả lập tay quẹt nhanh sang trái
    swipe_triggered = False
    for i in range(10):
        # x giật mạnh từ 400 về 150 (độ dời > 200px trên palm_len=60)
        curr_x = 400 - i * 35
        curr_y = 200.0
        _, _, is_swipe, _ = seg.update(curr_x, curr_y, palm_len=60.0, timestamp=i * 0.033)
        if is_swipe:
            swipe_triggered = True
            break
    print(f"  -> Trạng thái phát hiện Swipe-Left: {swipe_triggered}")
    assert swipe_triggered, "Thất bại: Không bắt được lệnh Swipe-Left"
    print("  ✓ PASS: Lệnh Swipe-Left kích hoạt thành công!")

    # 5. Kiểm tra Lưu / Tải JSON Schema Phiên bản hóa
    print("\n[TEST 5] Kiểm tra Lưu / Tải JSON Schema (dynamic_templates.json):")
    test_json = "test_templates.json"
    engine = HybridDTWEngine(templates_path=test_json, config=cfg)
    engine.add_sample("xin_chao", "Xin chào", res_slow, threshold=1.20)
    engine.save_templates()

    engine2 = HybridDTWEngine(templates_path=test_json, config=cfg)
    assert "xin_chao" in engine2.vocabulary
    match_res = engine2.match(res_fast)
    print(f"  -> Kết quả so khớp mẫu nhanh với từ điển: {match_res.status} | Từ: {match_res.top_display} (Khoảng cách: {match_res.best_distance:.4f})")
    assert match_res.status == "ACCEPTED"
    assert match_res.top_word == "xin_chao"
    print("  ✓ PASS: JSON Schema phiên bản hóa và So khớp hoàn chỉnh!")

    if os.path.exists(test_json):
        os.remove(test_json)

    print("\n" + "=" * 70)
    print("  HOÀN THÀNH 100% TẤT CẢ CÁC BÀI UNIT TEST CHO HYBRID DTW ENGINE!")
    print("=" * 70)
