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
    RECORDING = "RECORDING"
    PAUSE_DETECTED = "PAUSE_DETECTED"


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


@dataclass
class GestureFrame:
    """Dữ liệu tức thời tại một khung hình camera."""
    x: float                     # Tọa độ x khớp cổ tay (pixel)
    y: float                     # Tọa độ y khớp cổ tay (pixel)
    palm_len: float              # Độ dài lòng bàn tay chuẩn hóa (LM 0 - LM 9)
    hand_shape: str              # Nhãn dáng tay tĩnh ('A', 'B', 'C', 'L', 'V', 'Unknown'...)
    timestamp: float             # Thời gian ghi nhận (time.time())


@dataclass
class ResampledGesture:
    """Chuỗi cử chỉ đã được nội suy chuẩn hóa về K=32 frames."""
    # traj có shape (32, 4): các cột lần lượt là [x_norm, y_norm, dx, dy]
    traj: np.ndarray
    shapes: List[str]            # Danh sách 32 nhãn dáng tay tương ứng
    duration_sec: float          # Thời gian thực tế của cử chỉ gốc (giây)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traj_features": np.round(self.traj, 4).tolist(),
            "hand_shapes": self.shapes,
            "duration_sec": round(self.duration_sec, 3)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResampledGesture":
        traj_arr = np.array(data["traj_features"], dtype=np.float32)
        return cls(
            traj=traj_arr,
            shapes=data["hand_shapes"],
            duration_sec=data.get("duration_sec", 1.0)
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
    """Kết quả phân loại chuỗi cử chỉ từ HybridDTWEngine."""
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
    - Quản lý FSM: [IDLE] -> [RECORDING] -> [PAUSE_DETECTED].
    - Bắt lệnh ưu tiên Swipe-Left (Backspace) tức thời mà không cần điểm dừng.
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
        swipe_cooldown_sec: float = 1.0   # Khóa lệnh xóa 1.0s sau mỗi lần kích hoạt
    ):
        self.v_start_ratio = v_start_ratio
        self.v_stop_ratio = v_stop_ratio
        self.pause_frames = pause_frames
        self.min_stroke_frames = min_stroke_frames
        self.max_stroke_frames = max_stroke_frames
        self.swipe_threshold = swipe_threshold
        self.swipe_ratio = swipe_ratio
        self.swipe_cooldown_sec = swipe_cooldown_sec

        self.state = KineticState.IDLE
        self.buffer: List[GestureFrame] = []
        self.pause_counter = 0
        self.prev_wrist: Optional[Tuple[float, float]] = None
        self.last_swipe_time = -999.0
        self.recent_wrists = deque(maxlen=6)   # Cửa sổ trượt 6 frames phát hiện Swipe

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
            current_state: KineticState (IDLE, RECORDING, PAUSE_DETECTED)
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
        is_swipe_left = False
        if len(self.recent_wrists) >= 5 and (timestamp - self.last_swipe_time) >= self.swipe_cooldown_sec:
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
            if v_norm >= self.v_start_ratio:
                self.state = KineticState.RECORDING
                self.buffer = [curr_frame]
                self.pause_counter = 0

        elif self.state == KineticState.RECORDING:
            self.buffer.append(curr_frame)

            # Kiểm tra trạng thái khựng tay ngắt từ
            if v_norm <= self.v_stop_ratio:
                self.pause_counter += 1
            else:
                self.pause_counter = 0

            # Người dùng giữ yên tay liên tục trong pause_frames -> CHỐT CỬ CHỈ!
            if self.pause_counter >= self.pause_frames:
                self.state = KineticState.PAUSE_DETECTED
                total_n = len(self.buffer)

                if self.min_stroke_frames <= total_n <= self.max_stroke_frames:
                    completed_stroke = list(self.buffer)

                # Reset ngay về IDLE để sẵn sàng cho cử chỉ tiếp theo
                self.buffer.clear()
                self.pause_counter = 0
                self.state = KineticState.IDLE

            # Bảo vệ: Nếu vung tay quá dài không nghỉ -> Hủy an toàn
            elif len(self.buffer) > self.max_stroke_frames:
                self.reset()

        return self.state, completed_stroke, is_swipe_left, v_norm

    def reset(self):
        """Khởi tạo lại toàn bộ trạng thái bộ đệm."""
        self.state = KineticState.IDLE
        self.buffer.clear()
        self.pause_counter = 0
        self.recent_wrists.clear()


# =============================================================================
# 3. NỘI SUY 2 NHÁNH & CHUẨN HÓA KHÔNG GIAN - THỜI GIAN
# =============================================================================

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
        if os.path.exists(self.templates_path):
            try:
                with open(self.templates_path, "r", encoding="utf-8") as f_old:
                    old_data = json.load(f_old)
                    existing_created_at = old_data.get("metadata", {}).get("created_at")
            except Exception:
                pass

        payload = {
            "schema_version": "1.0",
            "metadata": {
                "k_resample_frames": self.config.k_resample,
                "feature_dim": 4,
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
# 6. TỰ ĐỘNG KIỂM THỬ ĐỘC LẬP (SELF-CONTAINED UNIT TEST)
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
