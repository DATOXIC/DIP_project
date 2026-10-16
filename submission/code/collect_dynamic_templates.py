"""
CÔNG CỤ THU THẬP MẪU CỬ CHỈ ĐỘNG (DYNAMIC TEMPLATE COLLECTOR)
============================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Chức năng:
  - Thu thập 3 mẫu vector chuẩn cho từng từ vựng thủ ngữ động (VSL).
  - Tự động bắt đầu và chốt điểm dừng (KineticPauseSegmenter):
    Người dùng chỉ cần múa cử chỉ rồi dừng tay dứt khoát 0.35s.
  - Tự động chuẩn hóa tọa độ neo gốc (Stroke-Origin) và nội suy K=32 frames.
  - Lưu trực tiếp vào file dynamic_templates.json có phiên bản hóa (v1.0).

Danh mục từ vựng mặc định:
  [1] xin_chao  : "Xin chào"
  [2] cam_on    : "Cảm ơn"
  [3] toi       : "Tôi"
  [4] ban       : "Bạn"
  [5] tam_biet  : "Tạm biệt"
"""

import os
import sys
import time
import math
import json
from typing import Dict, Tuple, Optional, List, Any

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
# SPEC 03: 4 nhóm đặc trưng mới dùng CHUNG với các đường trích xuất khác (nguồn sự thật duy nhất).
from dynamic_features import (
    CachedFaceDetector,
    compute_extended_columns,
    append_extended_columns,
    compute_repetition_count,
    compute_palm_len,
)
from dynamic_dtw_engine import (
    KineticState,
    GestureFrame,
    KineticPauseSegmenter,
    ResampledGesture,
    HybridDTWEngine,
    HybridDTWConfig,
    canonicalize_chirality,
    STROKE_DURATION_MIN_SEC,
    STROKE_DURATION_MAX_SEC,
    is_stroke_duration_valid,
    is_movement_epenthesis
)
from collections import deque
import queue
import threading


from predict_sentence_webcam import (
    CONFIG,
    GoldenTriadDTWMatcher,
    compute_golden_triad_dtw_distance,
    compute_joint_flexion_angles,
    compute_palm_normal_vector,
    resample_9d_gesture_sequence,
    WristCoordinateSmoother,
    draw_feature_metrics_hud,
    draw_hand_skeleton_features
)
from pipeline_common import (
    MIN_COLLECT_FPS,
    SHARED_CONFIG,
    ParallelHandDetector,
    get_default_detector_workers,
    decode_fourcc,
    open_camera,
    make_wrist_smoother,
    make_segmenter,
    LKInertialFallback,
    SlidingFPSMeter,
)


# Thư mục lưu trữ STROKE THÔ (raw stroke archive) — xem write_raw_stroke_archive().
RAW_STROKES_DIR = "raw_strokes"

# Phiên bản schema của file lưu trữ thô. TĂNG số này mỗi khi thay đổi cấu trúc,
# để rebuild_templates_from_raw.py biết cách đọc đúng từng thế hệ dữ liệu.
#   raw_stroke_v1.0 : không có hộp mặt => KHÔNG tái tạo được mốc cơ thể (SPEC 03)
#   raw_stroke_v1.1 : mỗi frame có thêm "face_box": [cx, cy, w, h] (pixel khung gốc) hoặc null
#   raw_stroke_v1.2 : bổ sung session_id vào top-level metadata và dict mẫu (SPEC 05A)
#   raw_stroke_v1.3 : bổ sung capture_fourcc và capture_fps_reported vào top-level metadata (SPEC 05B)
RAW_STROKE_SCHEMA_VERSION = "raw_stroke_v1.3"

# Ngưỡng lọc khung ổn định cho nhãn dáng tay CNN (SPEC: chống nhiễu khung chuyển động).
# v_norm <= STABLE_V_NORM_MAX: tay gần như đứng yên tại khung đó (không phải đang chuyển tiếp).
# hand_shape_conf >= STABLE_SHAPE_CONF_MIN: CNN đủ tin để dùng nhãn của khung đó.
STABLE_V_NORM_MAX = 0.04
STABLE_SHAPE_CONF_MIN = 0.55


def stabilize_hand_shapes(frames: List[GestureFrame]) -> List[str]:
    """
    Thay nhãn dáng tay của các khung KHÔNG ổn định (tay đang di chuyển hoặc CNN không đủ tin)
    bằng nhãn của khung ỔN ĐỊNH gần nhất trong cùng stroke (forward-fill, lan ngược cho các
    khung trước khung ổn định đầu tiên).

    LÝ DO: trước đây mỗi khung hình trong lúc ký hiệu ĐỘNG đều được CNN phân loại độc lập,
    kể cả các khung tay đang chuyển động (mờ, góc xấu) -- khác hẳn lúc test CNN tĩnh (giữ yên
    tay). Kết quả: nhãn dáng tay nhiễu ngay cả khi CNN bản thân hoạt động tốt (VD dáng C của
    DRINK không được nhận ra khi thu động dù test tĩnh riêng lẻ vẫn ổn -- 2026-09-30).
    Nếu KHÔNG có khung nào đủ ổn định/tin cậy trong cả stroke, trả về nhãn gốc không đổi
    (an toàn khi CNN tắt -- hand_shape_conf luôn None và hand_shape luôn "Unknown").
    """
    shapes = [f.hand_shape for f in frames]
    stable_mask = [
        (f.v_norm <= STABLE_V_NORM_MAX)
        and (f.hand_shape_conf is None or f.hand_shape_conf >= STABLE_SHAPE_CONF_MIN)
        and f.hand_shape != "Unknown"
        for f in frames
    ]
    if not any(stable_mask):
        return shapes
    out = list(shapes)
    last_stable = None
    for i, is_stable in enumerate(stable_mask):
        if is_stable:
            last_stable = shapes[i]
        elif last_stable is not None:
            out[i] = last_stable
    first_stable_idx = stable_mask.index(True)
    for i in range(first_stable_idx):
        out[i] = shapes[first_stable_idx]
    return out


def write_raw_stroke_archive(
    word_key: str,
    display_name: str,
    frame_records: List[Dict[str, Any]],
    meta: Dict[str, Any],
    session_id: Optional[str] = None,
    raw_dir: Optional[str] = None
) -> Optional[str]:
    """
    Ghi TOÀN BỘ dữ liệu THÔ của một stroke ra raw_strokes/<WORD>/<timestamp>.json (hoặc thư mục chỉ định qua raw_dir).

    VÌ SAO CẦN LỚP LƯU TRỮ NÀY?
    ---------------------------
    Mỗi lần bộ đặc trưng thay đổi (4D -> 9D -> 12D -> 17D), toàn bộ mẫu đã thu trước đó
    lập tức trở thành vô giá trị vì không thể tính lại vector mới từ vector cũ — phải ĐI
    THU LẠI TỪ ĐẦU. Đó là công sức đã mất một lần rồi. File thô này chứa đầy đủ 21 mốc
    bàn tay + palm_len + góc + pháp tuyến + nhãn dáng tay + timestamp cho TỪNG frame,
    tức là mọi thứ cần thiết để TÍNH LẠI BẤT KỲ vector đặc trưng nào trong tương lai
    (xem rebuild_templates_from_raw.py). Từ nay mọi thay đổi đặc trưng chỉ tốn 1 lệnh
    rebuild offline, không bao giờ phải thu lại mẫu nữa.

    Trả về đường dẫn file đã ghi, hoặc None nếu lỗi (lỗi lưu trữ KHÔNG được phép làm
    sập phiên thu mẫu đang chạy).
    """
    try:
        safe_word = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(word_key)).upper()
        target_raw_dir = raw_dir or RAW_STROKES_DIR
        out_dir = os.path.join(target_raw_dir, safe_word)
        os.makedirs(out_dir, exist_ok=True)

        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int((now % 1.0) * 1000):03d}"
        out_path = os.path.join(out_dir, f"{stamp}.json")

        actual_session = session_id or meta.get("session_id", "unknown")

        payload = {
            "schema_version": RAW_STROKE_SCHEMA_VERSION,
            "session_id": actual_session,
            "word_key": word_key,
            "display_name": display_name,
            "num_frames": len(frame_records),
            "capture_fourcc": meta.get("capture_fourcc", "UNKNOWN"),
            "capture_fps_reported": meta.get("capture_fps_reported", 30.0),
            **meta,
            "frames": frame_records
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        return out_path
    except Exception as e:
        print(f"[RawArchive] Khong ghi duoc stroke tho: {e}")
        return None


class BackgroundRawArchiveWriter:
    """
    Luồng ghi nền bất đồng bộ cho kho dữ liệu thô (SPEC 05B Nhiệm vụ 3b.4).
    Đảm bảo I/O ghi đĩa JSON không làm chậm vòng lặp nhận diện/thu thập.
    """
    def __init__(self, raw_dir: Optional[str] = None):
        self.raw_dir = raw_dir
        self.q = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            try:
                word_key, display_name, frame_records, meta, session_id = item
                write_raw_stroke_archive(word_key, display_name, frame_records, meta, session_id, raw_dir=self.raw_dir)
            except Exception as e:
                print(f"[BackgroundRawArchiveWriter] Loi luu file tho: {e}")
            finally:
                self.q.task_done()

    def enqueue(
        self,
        word_key: str,
        display_name: str,
        frame_records: List[Dict[str, Any]],
        meta: Dict[str, Any],
        session_id: Optional[str] = None
    ):
        self.q.put((word_key, display_name, list(frame_records), dict(meta), session_id))

    def flush_and_stop(self):
        self.q.put(None)
        self.worker.join(timeout=3.0)


# Hằng số cảnh báo mẫu lệch (Outlier Warning)
OUTLIER_WARN_RATIO = 0.50       # Cảnh báo khi max_dist > 50% threshold của từ (ví dụ 1.25 * 0.5 = 0.62)
OUTLIER_WARN_FALLBACK = 0.60    # Ngưỡng dự phòng nếu từ chưa có threshold


def draw_overlay_trajectories(
    frame: np.ndarray,
    samples: List[ResampledGesture],
    anchor_x: int,
    anchor_y: int,
    palm_len_now: float,
    max_samples: int = 5
) -> None:
    """
    Vẽ tối đa max_samples quỹ đạo mẫu cũ gần nhất, neo tại (anchor_x, anchor_y),
    scale theo palm_len_now hiện tại để đối chiếu trực quan với vệt đang múa.
    Hiệu ứng phân tầng độ mới:
      - Mẫu gần nhất: Màu xanh cyan nhạt (220, 220, 100), nét 2px.
      - Các mẫu cũ hơn: Mờ dần về màu xám trung tính (70..120), nét 1px.
    """
    if not samples:
        return

    subset = samples[-max_samples:]
    n_sub = len(subset)

    for idx, s in enumerate(subset):
        traj = np.asarray(s.traj, dtype=np.float32)
        if len(traj) < 2:
            continue

        recency = idx / max(n_sub - 1, 1)

        if idx == n_sub - 1 and n_sub > 1:
            color = (220, 220, 100)   # Cyan nhạt nổi bật cho mẫu gần nhất
            thickness = 2
        else:
            gray_val = int(70 + 50 * recency)
            color = (gray_val, gray_val, gray_val)
            thickness = 1

        pts = [
            (int(anchor_x + float(row[0]) * palm_len_now),
             int(anchor_y + float(row[1]) * palm_len_now))
            for row in traj
        ]

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)


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

    # 4 góc viền nhấn tạo cảm giác giao diện HUD hiện đại
    corner_len = max(8, int(min(w, h) * 0.15))
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_col, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_col, 2)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_col, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_col, 2)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_col, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_col, 2)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_col, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_col, 2)

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



def load_negative_strokes(file_path: str = "negative_strokes.json") -> List[Dict[str, Any]]:
    """Đọc danh sách mẫu âm từ negative_strokes.json nếu đã tồn tại."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("samples", [])
    except Exception as e:
        print(f"[Negative] Loi doc file {file_path}: {e}")
        return []


def save_negative_strokes(file_path: str, samples: List[Dict[str, Any]]) -> bool:
    """Lưu danh sách mẫu âm vào negative_strokes.json với schema 24D chuẩn."""
    try:
        parent_dir = os.path.dirname(os.path.abspath(file_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f_old:
                    old_meta = json.load(f_old).get("metadata", {})
                    created_at = old_meta.get("created_at", created_at)
            except Exception:
                pass

        payload = {
            "schema_version": "1.0",
            "metadata": {
                "feature_dim": 24,
                "k_resample_frames": 32,
                "chirality_canonical": True,
                "created_at": created_at,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": len(samples),
                "description": "Kho mau cu chi am ngoai tu vung (negative strokes) phuc vu do false accept rate"
            },
            "samples": samples
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[Negative] Loi ghi file {file_path}: {e}")
        return False



# =============================================================================
# DEMO VIDEO PLAYER — phát video ASL Citizen loop trong sidebar
# =============================================================================

class DemoVideoPlayer:
    """
    Phát video demo ASL Citizen trên thread riêng, loop liên tục.
    Cung cấp frame hiện tại (BGR, đã resize) qua get_frame() cho vòng lặp chính.

    Manifest file: asl_demo_videos.json (tạo bởi scratch/select_demo_videos.py)
    Nếu không tìm thấy file / từ không có video → trả về None.
    """

    # Kích thước thumbnail hiển thị trong sidebar
    THUMB_W = 310
    THUMB_H = 174  # ~16:9

    def __init__(self, manifest_path: str = "asl_demo_videos.json") -> None:
        self._manifest: Dict[str, Optional[Dict]] = {}
        self._load_manifest(manifest_path)

        self._lock = threading.Lock()
        self._current_word: Optional[str] = None
        self._current_frame: Optional[np.ndarray] = None  # BGR thumbnail
        self._cap: Optional[cv2.VideoCapture] = None
        self._stop_event = threading.Event()
        self._switch_event = threading.Event()

        self._thread = threading.Thread(target=self._loop, daemon=True, name="DemoVideoPlayer")
        self._thread.start()

    # ------------------------------------------------------------------
    def _load_manifest(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"[DemoVideo] Không tìm thấy manifest: {path} — demo video bị tắt.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Resolve absolute paths (manifest có thể lưu abs_path sẵn)
            base = os.path.dirname(os.path.abspath(path))
            for word, entry in raw.items():
                if entry is None:
                    self._manifest[word] = None
                    continue
                # Ưu tiên abs_path nếu tồn tại, fallback path tương đối
                abs_p = entry.get("abs_path") or os.path.join(base, entry["path"])
                if os.path.exists(abs_p):
                    self._manifest[word] = {**entry, "abs_path": abs_p}
                else:
                    self._manifest[word] = None
            n_ok = sum(1 for v in self._manifest.values() if v is not None)
            print(f"[DemoVideo] Đã nạp manifest: {n_ok}/{len(self._manifest)} từ có video demo.")
        except Exception as e:
            print(f"[DemoVideo] Lỗi đọc manifest: {e}")

    # ------------------------------------------------------------------
    def switch(self, word_id: str) -> None:
        """Chuyển sang video demo cho word_id (dùng khi người dùng đổi từ)."""
        with self._lock:
            if self._current_word == word_id:
                return
            self._current_word = word_id
        self._switch_event.set()

    def get_frame(self) -> Optional[np.ndarray]:
        """Trả về frame demo hiện tại (BGR numpy, THUMB_W×THUMB_H) hoặc None."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def stop(self) -> None:
        """Dừng thread player."""
        self._stop_event.set()
        self._switch_event.set()  # unblock nếu đang chờ

    # ------------------------------------------------------------------
    def _open_video(self, word_id: Optional[str]) -> Optional[cv2.VideoCapture]:
        """Mở VideoCapture cho word_id. Trả về None nếu không có video."""
        if word_id is None:
            return None
        entry = self._manifest.get(word_id)
        if not entry:
            return None
        path = entry["abs_path"]
        if not os.path.exists(path):
            return None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _loop(self) -> None:
        """Thread loop: đọc frame từ VideoCapture, resize, loop lại khi hết."""
        cap: Optional[cv2.VideoCapture] = None
        last_word: Optional[str] = None
        frame_delay = 1.0 / 30.0  # fallback

        while not self._stop_event.is_set():
            # Kiểm tra có yêu cầu switch không
            if self._switch_event.is_set():
                self._switch_event.clear()
                with self._lock:
                    word_now = self._current_word
                if word_now != last_word:
                    if cap is not None:
                        cap.release()
                    cap = self._open_video(word_now)
                    last_word = word_now
                    if cap is not None:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        frame_delay = 1.0 / fps
                    else:
                        with self._lock:
                            self._current_frame = None

            if cap is None:
                # Không có video: chờ 50ms rồi thử lại
                self._stop_event.wait(timeout=0.05)
                continue

            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                # Hết video → loop lại từ đầu
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret and frame is not None:
                thumb = cv2.resize(frame, (self.THUMB_W, self.THUMB_H), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._current_frame = thumb

            # Ngủ đúng frame delay, tránh burn CPU, wake up sớm nếu switch
            elapsed = time.monotonic() - t0
            wait_sec = max(0.0, frame_delay - elapsed)
            self._switch_event.wait(timeout=wait_sec)

        # Dọn dẹp
        if cap is not None:
            cap.release()


DEFAULT_VOCAB = {
    '1': ('HELLO',     'Hello / Xin chào',    'Vẫy tay mở B từ trán ra ngoài hoặc vẫy ngang'),
    '2': ('BYE',       'Bye / Tạm biệt',      'Nâng tay B lên và vẫy ngón tay'),
    '3': ('ME',        'Me / Tôi',            'Chỉ ngón trỏ L/1 vào ngực'),
    '4': ('YOU',       'You / Bạn',           'Chi ngon tro ve camera + gat co tay 1 nhip (len/xuong >= 0.5 ban tay ~32px, < 0.3s)'),
    '5': ('THANK_YOU', 'Thank You / Cảm ơn',  'Chạm tay B/Flat dưới cằm rồi đưa ra trước'),
    '6': ('EAT',       'Eat / Ăn',            'Bàn tay chụm O/O-flat đưa chạm miệng/môi'),
    '7': ('DRINK',     'Drink / Uống',        'Tay C đưa lên miệng như cầm cốc uống'),
    '8': ('WATER',     'Water / Nước',        'Chữ W (3 ngón giữa) chạm cằm 2 lần'),
    '9': ('SORRY',     'Sorry / Xin lỗi',     'Bàn tay A/nắm đấm xoa tròn trước ngực'),
    '0': ('PLEASE',    'Please / Làm ơn',     'Lòng bàn tay B mở phẳng xoa tròn trước ngực'),
    'a': ('YES',       'Yes / Vâng, Có',      'Nắm tay chữ S gật lên xuống như đầu gật'),
    'b': ('BAD',       'Bad / Tệ, Xấu',       'Chạm các ngón tay vào cằm rồi lật úp hướng xuống'),
    'c': ('HUNGRY',    'Hungry / Đói',        'Bàn tay khum chữ C kéo từ cổ họng xuống ngực'),
    # 5 từ mới bổ sung
    'e': ('SLEEP',     'Sleep / Ngủ',         'Nghiêng đầu, áp lòng bàn tay vào má'),
    'g': ('CRY',       'Cry / Khóc',          'Hai ngón trỏ vuốt từ dưới mắt xuống má'),
    'h': ('THINK',     'Think / Nghĩ',        'Ngón trỏ chạm vào thái dương hoặc trán'),
    'k': ('KNOW',      'Know / Biết',         'Đầu 4 ngón tay chạm nhẹ vào trán'),
    'p': ('PHONE',     'Phone / Điện thoại',  'Tay chữ Y áp ngón cái vào tai, ngón út vào miệng'),
}

TARGET_SAMPLES_PER_WORD = 8
TARGET_NEGATIVE_SAMPLES = 20


def run_dynamic_collector(
    camera_index: int = 0,
    templates_path: Optional[str] = None,
    session_id: Optional[str] = None,
    negative_path: Optional[str] = None,
    profile: bool = False,
    video_path: Optional[str] = None,
    dry_run: bool = False,
    enable_static_shape_cnn: Optional[bool] = None,
    detect_scale: Optional[float] = None,
    max_frames: Optional[int] = None,
    headless: bool = False,
    detector_workers: Optional[int] = None,
    raw_dir: str = "raw_strokes"
):
    global RAW_STROKES_DIR
    RAW_STROKES_DIR = raw_dir
    os.makedirs(raw_dir, exist_ok=True)

    if not session_id:
        session_id = time.strftime("%Y%m%d_%H%M%S")
    negative_file = negative_path or "negative_strokes.json"
    neg_parent = os.path.dirname(os.path.abspath(negative_file))
    if neg_parent:
        os.makedirs(neg_parent, exist_ok=True)
    negative_samples = load_negative_strokes(negative_file)
    is_collecting_negative = False
    session_sample_count = 0
    last_x_press_time = 0.0
    frame_counter = 0
    processed_frames = 0

    if enable_static_shape_cnn is None:
        enable_static_shape_cnn = CONFIG.get("enable_static_shape_cnn", False)
    if detect_scale is None:
        detect_scale = float(CONFIG.get("detect_scale", 1.0))
    if detector_workers is None:
        detector_workers = CONFIG.get("detector_workers", None)

    print("=" * 75)
    print("  CONG CU THU MAU CU CHI DONG THU NGU VSL (DYNAMIC COLLECTOR v1.3)")
    print(f"  Session ID    : {session_id}")
    print(f"  Raw Dir       : {raw_dir}")
    print(f"  Profile       : {profile}")
    print(f"  Dry Run       : {dry_run}")
    print(f"  Detect Scale  : {detect_scale}")
    print(f"  Workers       : {detector_workers if detector_workers is not None else 'Auto'}")
    print(f"  Static CNN    : {enable_static_shape_cnn}")
    if video_path:
        print(f"  Video Source  : {video_path}")
    print("=" * 75)

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

    # 2. Khởi tạo Two-Stream CNN nhận diện Dáng tay (nếu bật)
    device = torch.device('cpu')
    model_path = 'vsl_dataset/models/best_model_v3.pth'
    cnn_model = None
    classes = ['A', 'B', 'C', 'L', 'V']
    if enable_static_shape_cnn:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            classes = [checkpoint['id_to_label'][i] for i in range(checkpoint['num_classes'])]
            cnn_model = TwoStreamHandCNN(num_classes=len(classes))
            cnn_model.load_state_dict(checkpoint['model_state'])
            cnn_model.eval()
            print(f"✓ Đã nạp Two-Stream CNN ({model_path}): {classes}")
        else:
            print("[!] Không tìm thấy best_model_v2.pth, dùng nhãn giả lập.")
    else:
        print("✓ Tắt DIP + CNN dáng tay tĩnh theo SPEC 05B (tiết kiệm CPU, lambda_shape=0.0).")

    # 3. Khởi tạo DTW Engine, Segmenter, Smoother, LK Fallback dùng chung
    templates_file = templates_path if templates_path is not None else CONFIG.get("templates_path", "dynamic_templates.json")
    tpl_parent = os.path.dirname(os.path.abspath(templates_file))
    if tpl_parent:
        os.makedirs(tpl_parent, exist_ok=True)
    if not os.path.exists(templates_file) and not dry_run:
        empty_payload = {
            "schema_version": "1.0",
            "metadata": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "feature_dim": 24,
                "description": "Dynamic Sign Language Templates (Golden Triad 24D)",
                "k_resample_frames": 32,
                "chirality_canonical": True,
            },
            "vocabulary": {}
        }
        with open(templates_file, "w", encoding="utf-8") as f:
            json.dump(empty_payload, f, ensure_ascii=False, indent=2)
        print(f"[Collector] Da khoi tao bank template moi rong: {templates_file}")

    dtw_engine = HybridDTWEngine(templates_path=templates_file)
    dtw_matcher = GoldenTriadDTWMatcher(templates_path=templates_file, strict_weights=False)
    segmenter = make_segmenter(CONFIG)
    wrist_smoother = make_wrist_smoother()
    lk_fallback = LKInertialFallback()
    try:
        bg_archive_writer = BackgroundRawArchiveWriter(raw_dir=raw_dir)
    except TypeError:
        bg_archive_writer = BackgroundRawArchiveWriter()
    sliding_fps_meter = SlidingFPSMeter(window_sec=1.0)

    # SPEC 03: bộ dò mặt có cache (mỗi 8 frame mới chạy mạng)
    try:
        face_detector: Optional[CachedFaceDetector] = CachedFaceDetector(detect_every_n_frames=8)
        print("✓ Đã nạp bộ dò mặt (mốc cơ thể) có cache.")
    except Exception as e:
        print(f"[!] KHÔNG nạp được bộ dò mặt ({e}) -> mẫu thu sẽ KHÔNG có mốc cơ thể (body_anchor_valid=False).")
        face_detector = None
    current_face_box = None

    # 4. Mở nguồn Video hoặc Webcam qua pipeline_common.open_camera
    cap = None
    actual_fourcc = "UNKNOWN"
    actual_fps = 30.0
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] KHÔNG THỂ mở video: {video_path}")
            cap = None
        else:
            actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            print(f"[Video] KẾT NỐI THÀNH CÔNG video: {video_path} (FPS: {actual_fps:.1f}, FourCC: {actual_fourcc})")
    else:
        cap = open_camera(camera_index, prefer_mjpg=True)
        if cap is not None and cap.isOpened():
            actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            print(f"[!] KHONG THE mo Camera [{camera_index}]!")
            return

    # Bộ đệm dữ liệu đo kiểm hiệu năng (Profile)
    if profile:
        prof_mediapipe = deque(maxlen=60)
        prof_face = deque(maxlen=60)
        prof_dip_cnn = deque(maxlen=60)
        prof_features = deque(maxlen=60)
        prof_smoother_seg = deque(maxlen=60)
        prof_draw = deque(maxlen=60)
        prof_archive = deque(maxlen=60)
        prof_total = deque(maxlen=60)
    else:
        prof_total = []

    # Trạng thái phiên làm việc
    current_key = '1'
    curr_word_id, curr_display_name, curr_desc = DEFAULT_VOCAB[current_key]
    flip_cam = False
    trail_points: List[Tuple[int, int]] = []
    stroke_angles_history: List[np.ndarray] = []
    stroke_normals_history: List[np.ndarray] = []
    stroke_handed_history: List[str] = []
    stroke_raw_history: List[Dict[str, Any]] = []
    stroke_ext_history: List[Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]] = []
    fps_estimate = 30.0
    prev_loop_time = time.time()
    flash_message = ""
    flash_color = (0, 255, 0)
    flash_time = 0.0
    hand_lost_frames = 0
    show_features = False
    show_overlay = True

    cached_palm_len: float = 50.0
    cached_hand_shape: str = "Unknown"
    cached_handedness: str = "Right"

    # Khởi tạo player video demo ASL Citizen (chạy nền, loop tự động)
    demo_player = DemoVideoPlayer(manifest_path="asl_demo_videos.json")
    demo_player.switch(curr_word_id)


    print("\n" + "=" * 75)
    print("  [HUONG DAN THU MAU - 10 TU VUNG ASL + MAU AM (SPEC 05B)]")
    print(f"  Phien thu (session_id) : {session_id}")
    print(f"  File bank templates    : {templates_file}")
    print(f"  File mau am            : {negative_file} (Hien co: {len(negative_samples)}/{TARGET_NEGATIVE_SAMPLES})")
    print("  [1-9, 0]: Chon tu 1..10 (HELLO .. PLEASE)")
    print("  [n]     : Bat/Tat che do thu MAU AM (Negative gestures)")
    print("  [d]     : Bat/Tat bang dac trung 12D (Feature HUD)")
    print("  [o]     : Bat/Tat Overlay quy dao mau cu")
    print("  [r]     : Undo mau vua thu gan nhat")
    print("  [x]     : XOA SACH toan bo mau (bam 2 lan trong 2s de xac nhan)")
    print("  [f]     : Lat guong camera")
    print("  [s]     : Luu file templates & negative_strokes.json")
    print("  [q]     : Thoat chuong trinh")
    print("=" * 75 + "\n")

    ready_queue: deque = deque()
    cap_exhausted = False

    while True:
        if profile:
            _t_frame_start = time.perf_counter()
            _t_mp = 0.0
            _t_face = 0.0
            _t_dip = 0.0
            _t_feat = 0.0
            _t_seg = 0.0
            _t_draw = 0.0
            _t_arch = 0.0

        # Đọc và submit frame mới nếu nguồn hình chưa cạn
        if not cap_exhausted:
            ret, frame = cap.read()
            if not ret:
                if video_path is not None and max_frames is not None and processed_frames < max_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    cap_exhausted = True
                    # Xả toàn bộ frame còn lại trong hàng đợi của ParallelHandDetector
                    ready_queue.extend(detector_pool.flush())

            if ret and frame is not None:
                capture_ts = time.time()
                frame_counter += 1
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

        # Kiểm tra nếu chưa có kết quả nào sẵn sàng trong hàng đợi
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

        # Ước lượng FPS trung bình trượt từ capture_ts
        _dt = curr_time - prev_loop_time
        if 1e-4 < _dt < 1.0:
            fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / _dt)
        prev_loop_time = curr_time
        fps_sliding = sliding_fps_meter.update(curr_time)

        # SPEC 03: hộp mặt (cx, cy, w, h) pixel khung gốc
        if face_detector is not None:
            if profile:
                _t0_face = time.perf_counter()
            current_face_box = face_detector.detect(rgb_frame, out_size=(w_f, h_f))
            if profile:
                _t_face = time.perf_counter() - _t0_face
        else:
            current_face_box = None

        hand_found = False
        wrist_x, wrist_y = 0.0, 0.0
        palm_len = cached_palm_len
        hand_shape = cached_hand_shape
        hand_shape_conf: Optional[float] = None
        seg_state = KineticState.IDLE
        completed_stroke = None
        is_swipe = False
        v_norm = 0.0
        current_angles: Optional[np.ndarray] = None
        current_normal: Optional[np.ndarray] = None

        if det_result.hand_landmarks and len(det_result.hand_landmarks) > 0:
            hand_lost_frames = 0
            hand_found = True
            landmarks = det_result.hand_landmarks[0]
            handedness_label = det_result.handedness[0][0].category_name if det_result.handedness else "Right"

            # 1. Điểm mốc cổ tay & Kích thước lòng bàn tay (Anatomical 3D Palm Scale)
            lm0 = landmarks[0]
            lm5 = landmarks[5]
            lm9 = landmarks[9]
            lm17 = landmarks[17]
            raw_wrist_x = lm0.x * w_f
            raw_wrist_y = lm0.y * h_f
            wrist_x, wrist_y = wrist_smoother.update(raw_wrist_x, raw_wrist_y, timestamp=curr_time)

            # Tính Palm Scale chống co ngắn phối cảnh khi chĩa ngón tay vào camera (Foreshortening Invariant)
            # Nguồn sự thật duy nhất của palm_len (dynamic_features.compute_palm_len), dùng chung với predict.
            # (Lỗi cũ: thành phần y của kích thước 3D nhân bề rộng thay vì bề cao, làm palm_len lúc thu lớn hơn lúc chạy tới 1.33 lần.)
            palm_len = compute_palm_len(lm0, lm5, lm9, lm17, w_f, h_f)
            cached_palm_len = palm_len

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
            stroke_handed_history.append(handedness_label)
            stroke_ext_history.append((lm_3d, current_face_box))
            if len(stroke_ext_history) > 120:
                stroke_ext_history.pop(0)
            if len(stroke_angles_history) > 120:
                stroke_angles_history.pop(0)
            if len(stroke_normals_history) > 120:
                stroke_normals_history.pop(0)
            if len(stroke_handed_history) > 120:
                stroke_handed_history.pop(0)

            # 2. Nhận diện Dáng tay tức thời bằng Two-Stream CNN (nếu bật)
            if enable_static_shape_cnn:
                if profile:
                    _t0_dip = time.perf_counter()
                hand_crop, (x1, y1, x2, y2), _, _ = extract_smart_hand_crop(
                    frame, landmarks, handedness_label=handedness_label, auto_flip_left=False
                )
                if hand_crop.shape[0] >= 30 and hand_crop.shape[1] >= 30 and cnn_model is not None:
                    tensor_np, _, _ = process_lean_dip_tensor(hand_crop, target_size=(128, 128))
                    t_pt = torch.from_numpy(tensor_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        logits = cnn_model(t_pt)
                        probs = F.softmax(logits, dim=1).numpy()[0]
                        hand_shape = classes[int(np.argmax(probs))]
                        hand_shape_conf = float(np.max(probs))
                if profile:
                    _t_dip = time.perf_counter() - _t0_dip
            else:
                hand_shape = "Unknown"
                hand_shape_conf = None

            cached_hand_shape = hand_shape
            cached_handedness = handedness_label

            # 2b. LƯU TRỮ THÔ: ghi lại MỌI thứ của frame này
            stroke_raw_history.append({
                "timestamp": round(float(curr_time), 6),
                "landmarks_norm": [[round(float(lm.x), 6), round(float(lm.y), 6), round(float(lm.z), 6)]
                                   for lm in landmarks],
                "landmarks_px": [[round(float(lm.x) * w_f, 6), round(float(lm.y) * h_f, 6), round(float(lm.z) * w_f, 6)]
                                 for lm in landmarks],
                "palm_len": round(float(palm_len), 6),
                "wrist_raw": [round(float(raw_wrist_x), 6), round(float(raw_wrist_y), 6)],
                "wrist_smoothed": [round(float(wrist_x), 6), round(float(wrist_y), 6)],
                "flexion_angles": [round(float(a), 6) for a in current_angles],
                "palm_normal": [round(float(n), 6) for n in current_normal],
                "hand_shape": hand_shape,
                "hand_shape_conf": (round(hand_shape_conf, 6) if hand_shape_conf is not None else None),
                "handedness": handedness_label,
                "face_box": ([round(float(v), 3) for v in current_face_box] if current_face_box is not None else None),
            })
            if len(stroke_raw_history) > 120:
                stroke_raw_history.pop(0)

        else:
            # MediaPipe không phát hiện bàn tay: Kích hoạt LK Inertial Fallback cho 1-2 frames
            hand_lost_frames += 1
            if 1 <= hand_lost_frames <= 2:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                predicted = lk_fallback.predict_wrist(gray_frame)
                if predicted is not None:
                    wrist_x, wrist_y = wrist_smoother.update(*predicted, timestamp=curr_time)
                    palm_len = cached_palm_len
                    hand_shape = cached_hand_shape
                    handedness_label = cached_handedness
                    hand_found = True
                    if stroke_angles_history:
                        stroke_angles_history.append(stroke_angles_history[-1])
                    if stroke_normals_history:
                        stroke_normals_history.append(stroke_normals_history[-1])
                    if stroke_handed_history:
                        stroke_handed_history.append(stroke_handed_history[-1])
                    if stroke_ext_history:
                        stroke_ext_history.append(stroke_ext_history[-1])

            if hand_lost_frames >= 4:
                segmenter.reset()
                wrist_smoother.reset()
                lk_fallback.reset()
                trail_points.clear()
                stroke_angles_history.clear()
                stroke_normals_history.clear()
                stroke_handed_history.clear()
                stroke_ext_history.clear()
                stroke_raw_history.clear()

        if hand_found:
            # 3. Cập nhật Bộ phân đoạn động học
            if profile:
                _t0_seg = time.perf_counter()
            seg_state, completed_stroke, is_swipe, v_norm = segmenter.update(
                wrist_x=wrist_x,
                wrist_y=wrist_y,
                palm_len=palm_len,
                hand_shape=hand_shape,
                timestamp=curr_time,
                hand_shape_conf=hand_shape_conf
            )
            if profile:
                _t_seg = time.perf_counter() - _t0_seg

            # Vẽ vệt đuôi chuyển động
            if seg_state == KineticState.RECORDING:
                trail_points.append((int(wrist_x), int(wrist_y)))
            else:
                if len(trail_points) > 0:
                    trail_points.pop(0)

            # 4. Khi cử chỉ hoàn thành (Khựng tay ngắt từ) -> LƯU MẪU TỰ ĐỘNG
            if completed_stroke is not None:
                n_stroke = len(completed_stroke)
                stroke_duration = completed_stroke[-1].timestamp - completed_stroke[0].timestamp
                if stroke_duration <= 0:
                    stroke_duration = n_stroke / max(fps_estimate, 1.0)

                # ---------------------------------------------------------------
                # CỔNG FPS THU MẪU (SPEC 05B Nhiệm vụ 3c)
                # ---------------------------------------------------------------
                if fps_sliding < MIN_COLLECT_FPS and video_path is None:
                    flash_color = (0, 0, 255)
                    flash_message = (
                        f"(!) FPS THAP ({fps_sliding:.1f} < {MIN_COLLECT_FPS:.0f}) - KHONG LUU MAU! "
                        f"KIEM TRA ANH SANG / CAMERA!"
                    )
                    flash_time = curr_time + 3.0
                    print(f"[Collector] TU CHOI stroke: FPS trượt ({fps_sliding:.1f} < {MIN_COLLECT_FPS:.0f}) -> KHONG LUU.")
                    stroke_angles_history.clear()
                    stroke_normals_history.clear()
                    stroke_handed_history.clear()
                    stroke_ext_history.clear()
                    stroke_raw_history.clear()
                    trail_points.clear()
                    completed_stroke = None
                elif not is_stroke_duration_valid(stroke_duration):
                    reason = "QUA CHAM" if stroke_duration > STROKE_DURATION_MAX_SEC else "QUA NHANH"
                    flash_color = (0, 0, 255)
                    flash_message = (
                        f"(!) CU CHI {reason} ({stroke_duration:.2f}s) - NGOAI CONG "
                        f"[{STROKE_DURATION_MIN_SEC:.2f}s, {STROKE_DURATION_MAX_SEC:.2f}s]! "
                        f"KHONG LUU - LAM LAI CHO DUNG NHIP!"
                    )
                    flash_time = curr_time + 3.5
                    print(f"[Collector] TU CHOI stroke: thoi luong {stroke_duration:.2f}s ngoai cong "
                          f"[{STROKE_DURATION_MIN_SEC}, {STROKE_DURATION_MAX_SEC}] -> khong luu mau.")
                    stroke_angles_history.clear()
                    stroke_normals_history.clear()
                    stroke_handed_history.clear()
                    stroke_ext_history.clear()
                    stroke_raw_history.clear()
                    trail_points.clear()
                    completed_stroke = None
                elif is_movement_epenthesis(completed_stroke):
                    flash_color = (0, 0, 255)
                    flash_message = "(!) BO QUA: chuyen dong chuyen tiep"
                    flash_time = curr_time + 3.5
                    print("[Collector] TU CHOI stroke: chuyen dong chuyen tiep (epenthesis).")
                    stroke_angles_history.clear()
                    stroke_normals_history.clear()
                    stroke_handed_history.clear()
                    stroke_ext_history.clear()
                    stroke_raw_history.clear()
                    trail_points.clear()
                    completed_stroke = None
                else:
                    # Trích xuất dữ liệu stroke
                    if len(stroke_angles_history) >= n_stroke:
                        angles_for_stroke = stroke_angles_history[-n_stroke:]
                    else:
                        pad_len = n_stroke - len(stroke_angles_history)
                        first_val = stroke_angles_history[0] if stroke_angles_history else np.zeros(5, dtype=np.float32)
                        angles_for_stroke = [first_val] * pad_len + stroke_angles_history

                    if len(stroke_normals_history) >= n_stroke:
                        normals_for_stroke = stroke_normals_history[-n_stroke:]
                    else:
                        pad_len = n_stroke - len(stroke_normals_history)
                        first_val = stroke_normals_history[0] if stroke_normals_history else np.array([0.0, 0.0, -1.0], dtype=np.float32)
                        normals_for_stroke = [first_val] * pad_len + stroke_normals_history

                    if len(stroke_handed_history) >= n_stroke:
                        handed_for_stroke = stroke_handed_history[-n_stroke:]
                    else:
                        handed_for_stroke = list(stroke_handed_history)

                    if len(stroke_ext_history) >= n_stroke:
                        ext_for_stroke = stroke_ext_history[-n_stroke:]
                    else:
                        ext_for_stroke = [stroke_ext_history[0] if stroke_ext_history else (None, None)] * (n_stroke - len(stroke_ext_history)) + stroke_ext_history

                    if len(stroke_raw_history) >= n_stroke:
                        raw_for_stroke = stroke_raw_history[-n_stroke:]
                    else:
                        raw_for_stroke = list(stroke_raw_history)

                    dominant_handed = max(set(handed_for_stroke), key=handed_for_stroke.count) if handed_for_stroke else "Right"

                    stroke_angles_history.clear()
                    stroke_normals_history.clear()
                    stroke_handed_history.clear()
                    stroke_ext_history.clear()

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
                    raw_dangles_mat = np.zeros_like(raw_angles_mat)
                    raw_dangles_mat[1:] = raw_angles_mat[1:] - raw_angles_mat[:-1]
                    raw_dangles_mat[0] = raw_dangles_mat[1] if n_stroke > 1 else 0.0

                    raw_normals_mat = np.array(normals_for_stroke, dtype=np.float32)
                    raw_17d = np.column_stack([raw_xs, raw_ys, raw_dxs, raw_dys, raw_angles_mat, raw_normals_mat, raw_dangles_mat]).astype(np.float32)
                    raw_ext, body_ok = compute_extended_columns(
                        [f.palm_len for f in completed_stroke],
                        [(f.x, f.y) for f in completed_stroke],
                        [e[0] for e in ext_for_stroke],
                        [e[1] for e in ext_for_stroke]
                    )
                    raw_17d = append_extended_columns(raw_17d, raw_ext)
                    if not body_ok:
                        print("[Collector] CẢNH BÁO: không dò được mặt trong stroke này -> mẫu không có mốc cơ thể (body_anchor_valid=False).")
                    stroke_shapes = stabilize_hand_shapes(completed_stroke)

                    res_traj_17d, res_shapes = resample_9d_gesture_sequence(raw_17d, stroke_shapes, k_target=32)
                    if res_traj_17d is not None:
                        duration = stroke_duration
                        res_traj_17d, was_flipped = canonicalize_chirality(res_traj_17d, handedness=dominant_handed)
                        rep_count = compute_repetition_count(res_traj_17d[:, 0], res_traj_17d[:, 1])

                        if profile:
                            _t0_arch = time.perf_counter()

                        if dry_run:
                            flash_color = (0, 255, 255)
                            flash_message = f"[DRY-RUN] HOP LE ({duration:.2f}s) - KHONG GHI DIA"
                            flash_time = curr_time + 2.5
                            print(f"[Dry-Run] Cử chỉ '{curr_display_name}' ({duration:.2f}s) hợp lệ - BỎ QUA GHI ĐĨA.")
                            stroke_raw_history.clear()
                            trail_points.clear()
                        elif is_collecting_negative:
                            neg_dict = {
                                "traj_features": np.round(res_traj_17d, 4).tolist(),
                                "hand_shapes": res_shapes,
                                "duration_sec": round(float(duration), 3),
                                "handedness": dominant_handed,
                                "chirality_canonical": True,
                                "chirality_flipped_on_save": bool(was_flipped),
                                "session_id": session_id,
                                "rep_count": int(rep_count),
                                "body_anchor_valid": bool(body_ok),
                                "is_negative": True
                            }
                            negative_samples.append(neg_dict)
                            save_negative_strokes(negative_file, negative_samples)
                            session_sample_count += 1
                            sample_index = len(negative_samples) - 1

                            bg_archive_writer.enqueue(
                                word_key="NEGATIVE",
                                display_name="Mau Am",
                                frame_records=raw_for_stroke,
                                session_id=session_id,
                                meta={
                                    "camera_index": int(camera_index),
                                    "frame_width": int(w_f),
                                    "frame_height": int(h_f),
                                    "flip_cam": bool(flip_cam),
                                    "fps_estimate": round(float(fps_estimate), 3),
                                    "capture_fourcc": actual_fourcc,
                                    "capture_fps_reported": round(float(actual_fps), 2),
                                    "stroke_duration_sec": round(float(duration), 6),
                                    "dominant_handedness": dominant_handed,
                                    "chirality_flipped_on_save": bool(was_flipped),
                                    "k_resample": 32,
                                    "templates_path": negative_file,
                                    "template_sample_index": int(sample_index),
                                    "is_negative": True,
                                    "session_id": session_id,
                                    "segmenter_config": {
                                        "v_start_ratio": segmenter.v_start_ratio,
                                        "v_stop_ratio": segmenter.v_stop_ratio,
                                        "pause_frames": segmenter.pause_frames,
                                        "min_stroke_frames": segmenter.min_stroke_frames,
                                        "max_stroke_frames": segmenter.max_stroke_frames,
                                        "grace_window_sec": segmenter.grace_window_sec,
                                        "swipe_threshold": segmenter.swipe_threshold,
                                        "swipe_ratio": segmenter.swipe_ratio,
                                        "swipe_cooldown_sec": segmenter.swipe_cooldown_sec,
                                        "swipe_max_buffer_frames": segmenter.swipe_max_buffer_frames,
                                    },
                                    "duration_gate": [STROKE_DURATION_MIN_SEC, STROKE_DURATION_MAX_SEC],
                                    "wrist_smoother_config": "f_c_min=1.0,beta=0.007",
                                }
                            )
                            stroke_raw_history.clear()
                            flash_color = (0, 255, 0)
                            flash_message = f"DA LUU MAU AM [{len(negative_samples)}/{TARGET_NEGATIVE_SAMPLES}]!"
                            flash_time = curr_time + 2.5
                            trail_points.clear()
                        else:
                            resampled = ResampledGesture(
                                traj=res_traj_17d,
                                shapes=res_shapes,
                                duration_sec=float(duration),
                                handedness=dominant_handed,
                                chirality_canonical=True,
                                session_id=session_id,
                                extra={
                                    "session_id": session_id,
                                    "rep_count": int(rep_count),
                                    "body_anchor_valid": bool(body_ok)
                                }
                            )

                            dtw_engine.add_sample(
                                word_key=curr_word_id,
                                display_name=curr_display_name,
                                sample=resampled
                            )
                            dtw_engine.save_templates()
                            session_sample_count += 1

                            sample_index = len(dtw_engine.vocabulary.get(curr_word_id, {}).get("samples", [])) - 1
                            bg_archive_writer.enqueue(
                                word_key=curr_word_id,
                                display_name=curr_display_name,
                                frame_records=raw_for_stroke,
                                session_id=session_id,
                                meta={
                                    "camera_index": int(camera_index),
                                    "frame_width": int(w_f),
                                    "frame_height": int(h_f),
                                    "flip_cam": bool(flip_cam),
                                    "fps_estimate": round(float(fps_estimate), 3),
                                    "capture_fourcc": actual_fourcc,
                                    "capture_fps_reported": round(float(actual_fps), 2),
                                    "stroke_duration_sec": round(float(duration), 6),
                                    "dominant_handedness": dominant_handed,
                                    "chirality_flipped_on_save": bool(was_flipped),
                                    "k_resample": 32,
                                    "templates_path": templates_file,
                                    "template_sample_index": int(sample_index),
                                    "session_id": session_id,
                                    "segmenter_config": {
                                        "v_start_ratio": segmenter.v_start_ratio,
                                        "v_stop_ratio": segmenter.v_stop_ratio,
                                        "pause_frames": segmenter.pause_frames,
                                        "min_stroke_frames": segmenter.min_stroke_frames,
                                        "max_stroke_frames": segmenter.max_stroke_frames,
                                        "grace_window_sec": segmenter.grace_window_sec,
                                        "swipe_threshold": segmenter.swipe_threshold,
                                        "swipe_ratio": segmenter.swipe_ratio,
                                        "swipe_cooldown_sec": segmenter.swipe_cooldown_sec,
                                        "swipe_max_buffer_frames": segmenter.swipe_max_buffer_frames,
                                    },
                                    "duration_gate": [STROKE_DURATION_MIN_SEC, STROKE_DURATION_MAX_SEC],
                                    "wrist_smoother_config": "f_c_min=1.0,beta=0.007",
                                }
                            )
                            stroke_raw_history.clear()

                            # DTW preview
                            preview_text = ""
                            curr_samples = dtw_engine.vocabulary.get(curr_word_id, {}).get("samples", [])
                            word_threshold = float(dtw_engine.vocabulary.get(curr_word_id, {}).get("threshold", 1.25))
                            warn_thresh = round(word_threshold * OUTLIER_WARN_RATIO, 2)
                            is_outlier = False
                            max_dist = 0.0
                            min_dist = 0.0

                            if len(curr_samples) >= 2:
                                old_samples = curr_samples[:-1]
                                dists = []
                                wp, wa, wm, wn = dtw_matcher._get_weights_for_sign(curr_word_id)
                                for old_s in old_samples:
                                    old_traj = np.asarray(old_s.traj, dtype=np.float32)
                                    old_shapes = old_s.shapes
                                    if len(old_traj) != 32:
                                        continue
                                    if not old_s.chirality_canonical:
                                        old_s_traj, _ = canonicalize_chirality(old_traj, handedness=old_s.handedness)
                                    else:
                                        old_s_traj = old_traj
                                    d = compute_golden_triad_dtw_distance(
                                        q_traj=res_traj_17d,
                                        q_shapes=res_shapes,
                                        t_traj=old_s_traj,
                                        t_shapes=old_shapes,
                                        w_p=wp, w_a=wa, w_m=wm, w_n=wn,
                                        lambda_shape=dtw_matcher.lambda_shape,
                                        lambda_shape_global=dtw_matcher.lambda_shape_global
                                    )
                                    dists.append(d)

                                if dists:
                                    min_dist = min(dists)
                                    max_dist = max(dists)
                                    preview_text = f" | DTW: min={min_dist:.2f}, max={max_dist:.2f}"
                                    if max_dist > warn_thresh:
                                        is_outlier = True

                            samples_count = len(curr_samples)
                            if is_outlier:
                                flash_color = (0, 0, 255)
                                flash_message = f"(!) MAU LECH BAT THUONG! DTW max={max_dist:.2f} (nguong {warn_thresh}) | Nhan [r] de XOA!"
                                flash_time = curr_time + 4.0
                            else:
                                flash_color = (0, 255, 0)
                                flash_message = f"DA LUU MAU [{samples_count}/{TARGET_SAMPLES_PER_WORD}] CHO '{curr_display_name}'!{preview_text}"
                                flash_time = curr_time + 2.5
                            trail_points.clear()

                        if profile:
                            _t_arch = time.perf_counter() - _t0_arch

        # =====================================================================
        # =====================================================================
        # HIỂN THỊ GIAO DIỆN HUD THU MẪU CHUYÊN NGHIỆP (SIDEBAR + WEBCAM)
        # =====================================================================
        # 0. Vẽ Bounding Box & Mốc cơ thể Khuôn mặt trực quan (Body Anchor) lên frame webcam
        if profile:
            _t0_draw = time.perf_counter()

        draw_face_anchor_box(
            frame,
            face_box=current_face_box,
            wrist_pt=(int(wrist_x), int(wrist_y)) if hand_found else None,
            draw_vector=show_features
        )

        # 1. Vẽ Overlay các quỹ đạo mẫu cũ (tham khảo trực quan thời gian thực)
        if show_overlay and hand_found and not is_collecting_negative:
            existing_samples = dtw_engine.vocabulary.get(curr_word_id, {}).get("samples", [])
            if existing_samples:
                anc_x = trail_points[0][0] if trail_points else int(wrist_x)
                anc_y = trail_points[0][1] if trail_points else int(wrist_y)
                draw_overlay_trajectories(frame, existing_samples, anc_x, anc_y, palm_len)

        # 2. Vẽ vệt sáng quỹ đạo bàn tay thật (vệt vàng đè lên trên overlay)
        if len(trail_points) > 1:
            for i in range(1, len(trail_points)):
                pt1 = trail_points[i - 1]
                pt2 = trail_points[i]
                cv2.line(frame, pt1, pt2, (0, 255, 255), 3)

        # 3. Thanh tiêu đề trên cùng của khung Webcam (Gọn gàng, không che mặt)
        cv2.rectangle(frame, (0, 0), (w_f, 44), (20, 20, 20), -1)
        if seg_state == KineticState.RECORDING:
            state_color = (0, 255, 0)
            state_text = "DANG GHI..."
        elif seg_state == KineticState.READY:
            state_color = (0, 255, 255)
            state_text = "SAN SANG"
        elif seg_state == KineticState.PENDING_COMMIT:
            state_color = (0, 200, 255)
            state_text = "CHOT MAU..."
        else:
            state_color = (180, 180, 180)
            state_text = "CHO CU CHI"

        # Trạng thái mốc cơ thể (khuôn mặt)
        if face_detector is None:
            face_badge = "MAT: TAT"
            face_col = (130, 130, 130)
        elif current_face_box is not None:
            face_badge = "MAT: OK"
            face_col = (0, 255, 0)
        else:
            face_badge = "MAT: CHUA THAY"
            face_col = (0, 165, 255)

        fps_is_low = (fps_sliding < MIN_COLLECT_FPS and video_path is None)
        fps_str = f"FPS:{fps_sliding:4.1f}"
        if fps_is_low:
            fps_str += " [FPS THAP - KHONG LUU]"

        status_line = f"[CAM {camera_index}] {state_text} | {face_badge} | {fps_str}"
        cv2.putText(frame, status_line, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255) if fps_is_low else state_color, 2 if fps_is_low else 1, lineType=cv2.LINE_AA)

        # Đo vận tốc chuẩn hóa
        v_bar_w = int(min(1.0, v_norm / 0.15) * 60)
        cv2.putText(frame, f"v:{v_norm*100:.0f}%", (w_f - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (w_f - 75, 15), (w_f - 75 + v_bar_w, 28), (0, 255, 255), -1)
        cv2.rectangle(frame, (w_f - 75, 15), (w_f - 15, 28), (100, 100, 100), 1)

        # 4. Thông báo Toast nổi bật khi vừa lưu mẫu (Xanh lá nếu chuẩn, Đỏ nếu Outlier)
        if curr_time < flash_time:
            cv2.rectangle(frame, (20, h_f // 2 - 32), (w_f - 20, h_f // 2 + 32), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, h_f // 2 - 32), (w_f - 20, h_f // 2 + 32), flash_color, 2)
            cv2.putText(frame, flash_message, (30, h_f // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, flash_color, 2, lineType=cv2.LINE_AA)

        # 5. Bảng hiển thị đặc trưng 3D & Khung xương tay trực quan (12D Features HUD) khi bật 'd'
        if show_features and hand_found and current_angles is not None:
            draw_hand_skeleton_features(frame, landmarks, current_angles, w_f, h_f, handedness=handedness_label)
            vel_vec = (raw_wrist_x - wrist_smoother.prev_x, raw_wrist_y - wrist_smoother.prev_y) if wrist_smoother.prev_x is not None else (0.0, 0.0)
            draw_feature_metrics_hud(
                frame,
                angles=current_angles,
                palm_len=palm_len,
                v_norm=v_norm,
                wrist_pt=(int(wrist_x), int(wrist_y)),
                vel_vector=vel_vec,
                palm_normal=current_normal,
                x_offset=w_f - 215,
                y_offset=55
            )

        # =====================================================================
        # GHÉP SIDEBAR BÊN TRÁI ĐỘC LẬP (KHÔNG CHE HÌNH ẢNH WEBCAM)
        # =====================================================================
        SIDEBAR_W = 340
        canvas = np.zeros((h_f, SIDEBAR_W + w_f, 3), dtype=np.uint8)

        # Ghép frame webcam vào bên phải nguyên vẹn 100%
        canvas[:, SIDEBAR_W:] = frame

        # Vẽ Sidebar bên trái canvas
        cv2.rectangle(canvas, (0, 0), (SIDEBAR_W, h_f), (20, 20, 20), -1)
        cv2.line(canvas, (SIDEBAR_W - 1, 0), (SIDEBAR_W - 1, h_f), (45, 45, 45), 1)

        # A. Header Sidebar
        cv2.rectangle(canvas, (0, 0), (SIDEBAR_W, 44), (28, 28, 28), -1)
        cv2.line(canvas, (0, 44), (SIDEBAR_W, 44), (45, 45, 45), 1)
        cv2.putText(canvas, "BANG DIEU KHIEN THU MAU", (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, lineType=cv2.LINE_AA)
        sess_short = session_id.replace("session_", "")
        cv2.putText(canvas, f"Phien: {sess_short} ({session_sample_count} mau)", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, lineType=cv2.LINE_AA)

        # B. Card Từ Đang Thu (Chiếm y: 52 -> 156)
        card_b_y1, card_b_y2 = 52, 156
        if is_collecting_negative:
            neg_count = len(negative_samples)
            box_col = (0, 255, 0) if neg_count >= TARGET_NEGATIVE_SAMPLES else (0, 165, 255)
            cv2.rectangle(canvas, (8, card_b_y1), (SIDEBAR_W - 8, card_b_y2), (12, 12, 12), -1)
            cv2.rectangle(canvas, (8, card_b_y1), (SIDEBAR_W - 8, card_b_y2), box_col, 2)
            cv2.putText(canvas, f"MAU AM (NEGATIVE) [{neg_count}/{TARGET_NEGATIVE_SAMPLES}]",
                        (14, card_b_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_col, 2, lineType=cv2.LINE_AA)
            cv2.putText(canvas, "Cu chi ngoai tu vung (gai mat, vuot toc, voi chuot...)",
                        (14, card_b_y1 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 190, 190), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, "Cu chi tu nhien -> DUNG TAY 0.35s de chot",
                        (14, card_b_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, "Muc tieu do false accept rate cua nguong",
                        (14, card_b_y1 + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, lineType=cv2.LINE_AA)
        else:
            samples_now = len(dtw_engine.vocabulary.get(curr_word_id, {}).get("samples", []))
            box_col = (0, 255, 0) if samples_now >= TARGET_SAMPLES_PER_WORD else (0, 255, 255)
            cv2.rectangle(canvas, (8, card_b_y1), (SIDEBAR_W - 8, card_b_y2), (12, 12, 12), -1)
            cv2.rectangle(canvas, (8, card_b_y1), (SIDEBAR_W - 8, card_b_y2), box_col, 2)
            cv2.putText(canvas, f"TU: {curr_display_name} [{samples_now}/{TARGET_SAMPLES_PER_WORD}]",
                        (14, card_b_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_col, 2, lineType=cv2.LINE_AA)
            cv2.putText(canvas, f"Dang tay: {hand_shape}",
                        (14, card_b_y1 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            desc_short = curr_desc if len(curr_desc) <= 44 else (curr_desc[:42] + "...")
            cv2.putText(canvas, f"HD: {desc_short}",
                        (14, card_b_y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 190, 190), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, "Dung yen 0.3s (SAN SANG) -> Ky -> DUNG 0.35s",
                        (14, card_b_y1 + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 220, 255), 1, lineType=cv2.LINE_AA)

        # C. Card Danh sách Từ Vựng & Mẫu Âm (Chiếm y: 164 -> 382)
        card_c_y1, card_c_y2 = 164, 382
        cv2.rectangle(canvas, (8, card_c_y1), (SIDEBAR_W - 8, card_c_y2), (14, 14, 14), -1)
        cv2.rectangle(canvas, (8, card_c_y1), (SIDEBAR_W - 8, card_c_y2), (38, 38, 38), 1)
        cv2.putText(canvas, f"TIEN DO {len(DEFAULT_VOCAB)} TU VUNG (MUC TIEU {TARGET_SAMPLES_PER_WORD} MAU):",
                    (14, card_c_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1, lineType=cv2.LINE_AA)

        vocab_items = list(DEFAULT_VOCAB.items())
        half = (len(vocab_items) + 1) // 2
        col1 = vocab_items[:half]
        col2 = vocab_items[half:]

        y_base = card_c_y1 + 35
        row_h = 16 if len(vocab_items) > 13 else (18 if len(vocab_items) > 10 else 22)
        font_sc = 0.30 if len(vocab_items) > 13 else (0.33 if len(vocab_items) > 10 else 0.35)

        for i, (k_opt, (w_id, d_name, _)) in enumerate(col1):
            n_samp = len(dtw_engine.vocabulary.get(w_id, {}).get("samples", []))
            is_active = (not is_collecting_negative) and (k_opt == current_key)
            col = (0, 255, 0) if is_active else ((0, 220, 0) if n_samp >= TARGET_SAMPLES_PER_WORD else (160, 160, 160))
            prefix = ">" if is_active else " "
            short_name = d_name.split('/')[0].strip()
            cv2.putText(canvas, f"{prefix}[{k_opt}] {short_name}:{n_samp}/{TARGET_SAMPLES_PER_WORD}",
                        (14, y_base + i * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sc, col, (2 if is_active else 1), lineType=cv2.LINE_AA)

        for i, (k_opt, (w_id, d_name, _)) in enumerate(col2):
            n_samp = len(dtw_engine.vocabulary.get(w_id, {}).get("samples", []))
            is_active = (not is_collecting_negative) and (k_opt == current_key)
            col = (0, 255, 0) if is_active else ((0, 220, 0) if n_samp >= TARGET_SAMPLES_PER_WORD else (160, 160, 160))
            prefix = ">" if is_active else " "
            short_name = d_name.split('/')[0].strip()
            cv2.putText(canvas, f"{prefix}[{k_opt}] {short_name}:{n_samp}/{TARGET_SAMPLES_PER_WORD}",
                        (170, y_base + i * row_h), cv2.FONT_HERSHEY_SIMPLEX, font_sc, col, (2 if is_active else 1), lineType=cv2.LINE_AA)

        # Dòng mẫu âm & tổng số
        y_div = y_base + max(len(col1), len(col2)) * row_h + 3
        cv2.line(canvas, (14, y_div), (SIDEBAR_W - 14, y_div), (35, 35, 35), 1)
        neg_col = (0, 255, 0) if is_collecting_negative else ((0, 220, 0) if len(negative_samples) >= TARGET_NEGATIVE_SAMPLES else (160, 160, 160))
        prefix_neg = ">" if is_collecting_negative else " "
        cv2.putText(canvas, f"{prefix_neg}[n] Mau am (Negative): {len(negative_samples)}/{TARGET_NEGATIVE_SAMPLES}",
                    (14, y_div + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, neg_col, (2 if is_collecting_negative else 1), lineType=cv2.LINE_AA)

        total_word_samples = sum(len(dtw_engine.vocabulary.get(wid, {}).get("samples", [])) for wid in [v[0] for v in DEFAULT_VOCAB.values()])
        max_total = len(DEFAULT_VOCAB) * TARGET_SAMPLES_PER_WORD
        cv2.putText(canvas, f"Tong: {total_word_samples}/{max_total} mau tu | {len(negative_samples)}/{TARGET_NEGATIVE_SAMPLES} mau am",
                    (14, y_div + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (140, 140, 140), 1, lineType=cv2.LINE_AA)

        # D. Card Hướng dẫn phím tắt chân Sidebar (Chiếm y: 390 -> 452)
        card_d_y1, card_d_y2 = 390, 452
        cv2.rectangle(canvas, (8, card_d_y1), (SIDEBAR_W - 8, card_d_y2), (12, 12, 12), -1)
        cv2.rectangle(canvas, (8, card_d_y1), (SIDEBAR_W - 8, card_d_y2), (32, 32, 32), 1)
        cv2.putText(canvas, "[Phim tat]: Chon tu | [n]: Mau am | [f]: Flip",
                    (14, card_d_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "[o]: Overlay | [d]: HUD 3D | [r]: Undo | [x]: Xoa",
                    (14, card_d_y1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "[q]: Thoat va luu du lieu an toan",
                    (14, card_d_y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 220, 255), 1, lineType=cv2.LINE_AA)

        # E. Video Demo ASL Citizen — cửa sổ phụ riêng, chiếu lặp liên tục
        if not headless:
            demo_frame = demo_player.get_frame()
            if demo_frame is not None:
                # Thêm label tên từ lên thumbnail trước khi hiển thị
                demo_display = demo_frame.copy()
                label_word = curr_word_id if not is_collecting_negative else "---"
                (tw, th), _ = cv2.getTextSize(label_word, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                lx = (DemoVideoPlayer.THUMB_W - tw) // 2
                # Nền mờ phía dưới để chữ dễ đọc
                cv2.rectangle(demo_display,
                              (0, DemoVideoPlayer.THUMB_H - th - 14),
                              (DemoVideoPlayer.THUMB_W, DemoVideoPlayer.THUMB_H),
                              (0, 0, 0), -1)
                cv2.putText(demo_display, label_word,
                            (lx, DemoVideoPlayer.THUMB_H - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2, lineType=cv2.LINE_AA)
                cv2.imshow("Demo ASL Citizen (Chieu lap)", demo_display)
            else:
                # Hiển thị placeholder khi không có video
                placeholder = np.zeros((DemoVideoPlayer.THUMB_H, DemoVideoPlayer.THUMB_W, 3), dtype=np.uint8)
                no_vid_msg = f"Khong co demo: {curr_word_id if not is_collecting_negative else 'MAU AM'}"
                cv2.putText(placeholder, no_vid_msg,
                            (10, DemoVideoPlayer.THUMB_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, lineType=cv2.LINE_AA)
                cv2.imshow("Demo ASL Citizen (Chieu lap)", placeholder)


        # Bảng nhỏ profile góc màn hình (SPEC 05B Nhiệm vụ 3a)
        if profile and len(prof_total) > 0:
            avg_mp = sum(prof_mediapipe) / len(prof_mediapipe) * 1000.0
            avg_face = sum(prof_face) / len(prof_face) * 1000.0
            avg_dip = sum(prof_dip_cnn) / len(prof_dip_cnn) * 1000.0
            avg_feat = sum(prof_features) / len(prof_features) * 1000.0
            avg_seg = sum(prof_smoother_seg) / len(prof_smoother_seg) * 1000.0
            avg_draw = sum(prof_draw) / len(prof_draw) * 1000.0
            avg_arch = sum(prof_archive) / len(prof_archive) * 1000.0
            avg_tot = sum(prof_total) / len(prof_total) * 1000.0
            fps_prof = 1000.0 / avg_tot if avg_tot > 0 else 0.0

            px0 = SIDEBAR_W + max(10, w_f - 230)
            py0 = 50
            cv2.rectangle(canvas, (px0, py0), (px0 + 220, py0 + 175), (15, 15, 15), -1)
            cv2.rectangle(canvas, (px0, py0), (px0 + 220, py0 + 175), (80, 80, 80), 1)
            lines_prof = [
                "PROFILE (60f avg)",
                f"FPS:       {fps_prof:5.1f}",
                f"mediapipe: {avg_mp:5.2f} ms",
                f"face:      {avg_face:5.2f} ms",
                f"dip_cnn:   {avg_dip:5.2f} ms",
                f"features:  {avg_feat:5.2f} ms",
                f"smooth_seg:{avg_seg:5.2f} ms",
                f"draw:      {avg_draw:5.2f} ms",
                f"arch_write:{avg_arch:5.2f} ms",
                f"total:     {avg_tot:5.2f} ms",
            ]
            for li, ltext in enumerate(lines_prof):
                col = (0, 255, 255) if li == 0 else ((0, 255, 0) if li == 1 else (220, 220, 220))
                cv2.putText(canvas, ltext, (px0 + 8, py0 + 16 + li * 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, lineType=cv2.LINE_AA)

        if profile:
            _t_draw = time.perf_counter() - _t0_draw

        if not headless:
            cv2.imshow("Thu Thap Mau Cu Chi Dong (VSL Dynamic Collector)", canvas)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 255
        if key == ord('q'):
            break
        elif key in [ord('d'), ord('D')]:
            show_features = not show_features
            print(f"[HUD] {'BẬT' if show_features else 'TẮT'} bảng hiển thị đặc trưng 3D.")
        elif key in [ord('o'), ord('O')]:
            show_overlay = not show_overlay
            print(f"[Overlay] {'BẬT' if show_overlay else 'TẮT'} hiển thị quỹ đạo mẫu cũ.")
        elif key in [ord('n'), ord('N')]:
            is_collecting_negative = not is_collecting_negative
            segmenter.reset()
            wrist_smoother.reset()
            trail_points.clear()
            stroke_angles_history.clear()
            stroke_normals_history.clear()
            stroke_handed_history.clear()
            stroke_ext_history.clear()
            stroke_raw_history.clear()
            if is_collecting_negative:
                flash_message = f"CHE DO: THU MAU AM ({len(negative_samples)}/{TARGET_NEGATIVE_SAMPLES})"
                flash_color = (0, 165, 255)
                flash_time = curr_time + 2.0
                print(f"-> Chuyển sang CHE DO THU MAU AM (Negative Gestures). Hiện có: {len(negative_samples)}/{TARGET_NEGATIVE_SAMPLES} mẫu.")
            else:
                flash_message = f"CHE DO: THU TU VUNG '{curr_display_name}'"
                flash_color = (0, 255, 0)
                flash_time = curr_time + 2.0
                print(f"-> Quay lại thu từ: {curr_display_name}")
        elif chr(key).lower() in DEFAULT_VOCAB:
            is_collecting_negative = False
            current_key = chr(key).lower()
            curr_word_id, curr_display_name, curr_desc = DEFAULT_VOCAB[current_key]
            segmenter.reset()
            wrist_smoother.reset()
            trail_points.clear()
            stroke_angles_history.clear()
            stroke_normals_history.clear()
            stroke_handed_history.clear()
            stroke_ext_history.clear()
            stroke_raw_history.clear()
            demo_player.switch(curr_word_id)   # Cập nhật video demo khi chuyển từ
            print(f"-> Chuyển sang thu từ: {curr_display_name}")
        elif key == ord('r'):
            if is_collecting_negative:
                if len(negative_samples) > 0:
                    negative_samples.pop()
                    save_negative_strokes(negative_file, negative_samples)
                    if session_sample_count > 0:
                        session_sample_count -= 1
                    now_neg = len(negative_samples)
                    flash_message = f"DA UNDO MAU AM! Con lai {now_neg}/{TARGET_NEGATIVE_SAMPLES} mau"
                    flash_color = (0, 255, 255)
                    flash_time = curr_time + 2.0
                    print(f"-> Đã xóa mẫu âm gần nhất (còn lại {now_neg} mẫu)")
            else:
                # Xóa mẫu gần nhất của từ hiện tại (Undo)
                if curr_word_id in dtw_engine.vocabulary and len(dtw_engine.vocabulary[curr_word_id]["samples"]) > 0:
                    dtw_engine.vocabulary[curr_word_id]["samples"].pop()
                    dtw_engine.save_templates()
                    if session_sample_count > 0:
                        session_sample_count -= 1
                    now_s = len(dtw_engine.vocabulary[curr_word_id]["samples"])
                    flash_message = f"DA UNDO MAU! Con lai {now_s}/{TARGET_SAMPLES_PER_WORD} mau cho '{curr_display_name}'"
                    flash_color = (0, 255, 255)
                    flash_time = curr_time + 1.5
                    print(f"-> Đã xóa mẫu gần nhất của {curr_display_name} (còn lại {now_s} mẫu)")
        elif key in [ord('x'), ord('X')]:
            # Phím x: Xác nhận 2 lần trong vòng 2 giây
            curr_t = time.time()
            if curr_t - last_x_press_time <= 2.0:
                last_x_press_time = 0.0
                if is_collecting_negative:
                    negative_samples.clear()
                    save_negative_strokes(negative_file, negative_samples)
                    flash_message = "DA XOA SACH TOAN BO MAU AM (NEGATIVE)!"
                    flash_color = (0, 0, 255)
                    flash_time = curr_t + 2.5
                    print("-> Đã xóa sạch toàn bộ mẫu âm (negative)")
                else:
                    if curr_word_id in dtw_engine.vocabulary:
                        dtw_engine.vocabulary[curr_word_id]["samples"].clear()
                        dtw_engine.save_templates()
                        flash_message = f"DA XOA SACH TOAN BO MAU CUA '{curr_display_name}'!"
                        flash_color = (0, 0, 255)
                        flash_time = curr_t + 2.5
                        print(f"-> Đã xóa sạch toàn bộ mẫu của {curr_display_name}")
            else:
                last_x_press_time = curr_t
                flash_message = "CANH BAO: BAM 'x' THEM 1 LAN TRONG 2S DE XAC NHAN XOA!"
                flash_color = (0, 0, 255)
                flash_time = curr_t + 2.0
                print("[Collector] CẢNH BÁO: Bấm 'x' thêm một lần nữa trong 2 giây để xác nhận xóa sạch.")
        elif key == ord('f'):
            flip_cam = not flip_cam
        elif key == ord('s'):
            if not dry_run:
                dtw_engine.save_templates()
                save_negative_strokes(negative_file, negative_samples)
                flash_message = "DA LUU TEMPLATES & NEGATIVE_STROKES!"
            else:
                flash_message = "[DRY-RUN] BO QUA GHI DIA!"
            flash_time = curr_time + 1.5

        if profile:
            _t_frame_end = time.perf_counter()
            _t_total = _t_frame_end - _t_frame_start
            prof_mediapipe.append(_t_mp)
            prof_face.append(_t_face)
            prof_dip_cnn.append(_t_dip)
            prof_features.append(_t_feat)
            prof_smoother_seg.append(_t_seg)
            prof_draw.append(_t_draw)
            prof_archive.append(_t_arch)
            prof_total.append(_t_total)

            if frame_counter % 60 == 0 and len(prof_total) > 0:
                avg_mp = sum(prof_mediapipe) / len(prof_mediapipe) * 1000.0
                avg_face = sum(prof_face) / len(prof_face) * 1000.0
                avg_dip = sum(prof_dip_cnn) / len(prof_dip_cnn) * 1000.0
                avg_feat = sum(prof_features) / len(prof_features) * 1000.0
                avg_seg = sum(prof_smoother_seg) / len(prof_smoother_seg) * 1000.0
                avg_draw = sum(prof_draw) / len(prof_draw) * 1000.0
                avg_arch = sum(prof_archive) / len(prof_archive) * 1000.0
                avg_tot = sum(prof_total) / len(prof_total) * 1000.0
                fps_prof = 1000.0 / avg_tot if avg_tot > 0 else 0.0
                print(f"[PROFILE] Frame {frame_counter:4d} (60f avg) | FPS: {fps_prof:5.1f}")
                print(f"  mediapipe:     {avg_mp:6.2f} ms")
                print(f"  face:          {avg_face:6.2f} ms")
                print(f"  dip_cnn:       {avg_dip:6.2f} ms")
                print(f"  features:      {avg_feat:6.2f} ms")
                print(f"  smoother_seg:  {avg_seg:6.2f} ms")
                print(f"  draw:          {avg_draw:6.2f} ms")
                print(f"  archive_write: {avg_arch:6.2f} ms")
                print(f"  total:         {avg_tot:6.2f} ms")

        if max_frames is not None and processed_frames >= max_frames:
            break

    try:
        bg_archive_writer.flush_and_stop()
    except Exception as e:
        print(f"[Collector] Loi flush archive: {e}")

    # Dừng thread demo video player
    demo_player.stop()

    if cap is not None:
        cap.release()
    if not headless:
        cv2.destroyAllWindows()
    detector_pool.close()
    print("-> Đã đóng công cụ thu thập mẫu.")

    if profile and len(prof_total) > 0:
        avg_mp = sum(prof_mediapipe) / len(prof_mediapipe) * 1000.0
        avg_face = sum(prof_face) / len(prof_face) * 1000.0
        avg_dip = sum(prof_dip_cnn) / len(prof_dip_cnn) * 1000.0
        avg_feat = sum(prof_features) / len(prof_features) * 1000.0
        avg_seg = sum(prof_smoother_seg) / len(prof_smoother_seg) * 1000.0
        avg_draw = sum(prof_draw) / len(prof_draw) * 1000.0
        avg_arch = sum(prof_archive) / len(prof_archive) * 1000.0
        avg_tot = sum(prof_total) / len(prof_total) * 1000.0
        fps_prof = 1000.0 / avg_tot if avg_tot > 0 else 0.0
        return {
            "mediapipe_ms": avg_mp,
            "face_ms": avg_face,
            "dip_cnn_ms": avg_dip,
            "features_ms": avg_feat,
            "smoother_seg_ms": avg_seg,
            "draw_ms": avg_draw,
            "archive_write_ms": avg_arch,
            "total_ms": avg_tot,
            "fps": fps_prof
        }
    return {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Dynamic Sign Language Templates (SPEC 05B 30 FPS)")
    parser.add_argument("--cam", type=int, default=None, help="Camera index (0=built-in webcam, 1, 2=external/iPhone). Mặc định dùng camera_source từ CONFIG.")
    parser.add_argument("--templates", type=str, default=None, help="Đường dẫn file template JSON (ví dụ: webcam_templates.json). Mặc định dùng templates_path từ CONFIG.")
    parser.add_argument("--session", type=str, default=None, help="Mã phiên thu (session_id, vd: sang_den_yeu). Mặc định tự sinh YYYYMMDD_HHMMSS.")
    parser.add_argument("--negative-file", type=str, default="negative_strokes.json", help="Đường dẫn file lưu mẫu âm (mặc định: negative_strokes.json).")
    parser.add_argument("--profile", action="store_true", default=False, help="Bật đo thời gian chi tiết từng tầng xử lý (Profile).")
    parser.add_argument("--video", type=str, default=None, help="Chạy trên file video thay vì webcam.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Chạy toàn bộ pipeline nhưng KHÔNG ghi bất kỳ file nào (không lưu mẫu, không ghi raw_strokes, không ghi negative).")
    parser.add_argument("--with-cnn", action="store_true", default=False, help="Bật nhận diện dáng tay DIP + CNN (mặc định tắt để đạt 30 FPS).")
    parser.add_argument("--detect-scale", type=float, default=None, help="Hệ số thu nhỏ frame trước khi đưa vào MediaPipe (0.5 hoặc 1.0). Mặc định đọc từ CONFIG (1.0).")
    parser.add_argument("--detector-workers", type=int, default=None, help="Số luồng làm việc song song cho MediaPipe (mặc định None = tự chọn min(3, max(1, CPU//4))).")
    parser.add_argument("--max-frames", type=int, default=None, help="Dừng sau số khung hình nhất định (dùng cho benchmark/test).")
    parser.add_argument("--raw-dir", type=str, default="raw_strokes", help="Thư mục lưu trữ nét thô (mặc định: raw_strokes).")
    parser.add_argument("--headless", action="store_true", default=False, help="Chạy không mở cửa sổ đồ họa OpenCV (dùng cho đo benchmark).")
    args, _ = parser.parse_known_args()
    
    selected_cam = args.cam if args.cam is not None else CONFIG.get("camera_source", 0)
    selected_templates = args.templates if args.templates is not None else CONFIG.get("templates_path", "dynamic_templates.json")
    run_dynamic_collector(
        camera_index=selected_cam,
        templates_path=selected_templates,
        session_id=args.session,
        negative_path=args.negative_file,
        profile=args.profile,
        video_path=args.video,
        dry_run=args.dry_run,
        enable_static_shape_cnn=args.with_cnn,
        detect_scale=args.detect_scale,
        max_frames=args.max_frames,
        headless=args.headless,
        detector_workers=args.detector_workers,
        raw_dir=args.raw_dir,
    )
