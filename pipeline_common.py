"""
MODULE DÙNG CHUNG CHO PIPELINE THU MẪU & DỰ ĐOÁN (PIPELINE COMMON)
=================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Mô tả:
  Thu gom toàn bộ các thành phần đường ống xử lý (pipeline) phải giống hệt nhau
  giữa collect_dynamic_templates.py và predict_sentence_webcam.py:
  1. open_camera: chuẩn hóa mở webcam với FourCC MJPG, 30 FPS, buffer=1, đọc xả 5 frame.
  2. make_wrist_smoother: 1€ Filter thích ứng (f_c_min=1.0, beta=0.007, d_cutoff=1.0).
  3. make_segmenter: KineticPauseSegmenter với bộ tham số chuẩn hóa.
  4. LKInertialFallback: phao cứu sinh quang thông Lucas-Kanade cho 1-2 frame mất tay.
  5. Cổng kiểm soát FPS thu mẫu MIN_COLLECT_FPS = 24.
"""

import os
import sys
import time
import math
import queue
import threading
from typing import Dict, Tuple, Optional, List, Any

import cv2
import numpy as np
import mediapipe as mp

from dynamic_dtw_engine import KineticPauseSegmenter

# Ngưỡng FPS tối thiểu chấp nhận khi thu mẫu (Dưới 24 FPS từ chối lưu để chống méo dữ liệu)
MIN_COLLECT_FPS: float = 24.0

# Các khóa cấu hình dùng chung giữa Collector và Predict
SHARED_CONFIG: Dict[str, Any] = {
    "camera_source": 1,
    "detect_scale": 1.0,                 # SPEC 05C: Trả về 1.0 vì MediaPipe không giảm ms khi resize
    "detector_workers": None,           # None = tự động min(3, max(1, os.cpu_count() // 4))
    "enable_static_shape_cnn": False,
    "v_start_ratio": 0.08,
    "v_stop_ratio": 0.040,
    "pause_frames": 9,
}


def get_default_detector_workers() -> int:
    """
    Số luồng làm việc mặc định cho ParallelHandDetector (SPEC 05C):
    min(3, max(1, os.cpu_count() // 4)).
    Trên máy 16 luồng -> 3 workers.
    """
    cpus = os.cpu_count() or 4
    return min(3, max(1, cpus // 4))


class ParallelHandDetector:
    """
    Bộ dò bàn tay MediaPipe song song hóa đa luồng (SPEC 05C).
    Giải quyết hiện tượng tụt FPS khi có bàn tay trong khung hình (MediaPipe IMAGE mode tốn ~49ms/frame trên CPU).

    Đặc tính kỹ thuật:
    - N worker threads độc lập, mỗi worker sở hữu 1 instance HandLandmarker (IMAGE mode) riêng biệt.
    - Phân phối frame qua task_queue.
    - Thu nhận kết quả qua result_queue và reorder buffer (self.buffered).
    - Bảo đảm 100% trả kết quả đúng theo thứ tự seq tăng dần liên tiếp (không đảo frame, không nhảy cóc).
    - Khống chế max_in_flight (mặc định: 1 nếu workers==1; workers + 1 nếu workers > 1) để tránh phình hàng đợi.
    - workers=1 đảm bảo hành vi đồng bộ tương đương code cũ.
    - Quản lý đóng sạch tài nguyên giải phóng detector.
    """
    def __init__(
        self,
        model_path: str = "hand_landmarker.task",
        workers: Optional[int] = None,
        num_hands: int = 1,
        max_in_flight: Optional[int] = None
    ):
        if workers is None:
            workers = get_default_detector_workers()
        self.num_workers = max(1, int(workers))
        self.model_path = model_path
        self.num_hands = int(num_hands)

        if max_in_flight is not None:
            self.max_in_flight = max(1, int(max_in_flight))
        else:
            self.max_in_flight = 1 if self.num_workers == 1 else (self.num_workers + 1)

        self.task_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()
        self.buffered: Dict[int, Tuple[dict, Any]] = {}

        self.next_seq_out: Optional[int] = None
        self.in_flight_count = 0
        self.stopped = False
        self.error: Optional[Exception] = None

        self.ready_event = threading.Event()
        self.ready_count = 0
        self.lock = threading.Lock()

        self.worker_threads: List[threading.Thread] = []
        for w_idx in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(w_idx,),
                daemon=True,
                name=f"ParallelHandDetectorWorker-{w_idx}"
            )
            t.start()
            self.worker_threads.append(t)

        # Chờ toàn bộ worker khởi tạo xong và hoàn thành warm-up
        self.ready_event.wait(timeout=15.0)
        if self.error:
            raise self.error

    @property
    def in_flight(self) -> int:
        return self.in_flight_count

    def _worker_loop(self, w_idx: int):
        try:
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=self.num_hands
            )
            detector = HandLandmarker.create_from_options(options)

            # Warm-up 1 frame giả lập để nạp delegate và tensor engine
            dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
            mp_dummy = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_img)
            detector.detect(mp_dummy)

            with self.lock:
                self.ready_count += 1
                if self.ready_count >= self.num_workers:
                    self.ready_event.set()
        except Exception as e:
            self.error = e
            self.ready_event.set()
            return

        while True:
            item = self.task_queue.get()
            if item is None:
                self.task_queue.task_done()
                break
            seq, image_rgb, meta = item
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                det_result = detector.detect(mp_image)
                self.result_queue.put((seq, meta, det_result))
            except Exception as e:
                self.error = e
                self.result_queue.put((seq, meta, None))
            finally:
                self.task_queue.task_done()

        try:
            detector.close()
        except Exception:
            pass

    def submit(self, seq: int, image_rgb: np.ndarray, meta: dict) -> None:
        if self.error:
            raise self.error
        if self.stopped:
            raise RuntimeError("ParallelHandDetector da dong, khong the submit frame moi.")
        if self.next_seq_out is None:
            self.next_seq_out = seq
        self.in_flight_count += 1
        self.task_queue.put((seq, image_rgb.copy(), meta))

    def _drain_result_queue(self):
        while True:
            try:
                seq, meta, det_result = self.result_queue.get_nowait()
                self.buffered[seq] = (meta, det_result)
            except queue.Empty:
                break

    def poll(self, block_oldest: bool = False) -> List[Tuple[int, dict, Any]]:
        if self.error:
            raise self.error

        self._drain_result_queue()

        if self.next_seq_out is None:
            return []

        if block_oldest and self.in_flight > 0:
            while self.next_seq_out not in self.buffered and not self.stopped:
                if self.error:
                    raise self.error
                try:
                    seq, meta, det_result = self.result_queue.get(timeout=0.02)
                    self.buffered[seq] = (meta, det_result)
                except queue.Empty:
                    pass

        results = []
        while self.next_seq_out in self.buffered:
            meta, det_result = self.buffered.pop(self.next_seq_out)
            results.append((self.next_seq_out, meta, det_result))
            self.next_seq_out += 1
            self.in_flight_count -= 1

        return results

    def flush(self) -> List[Tuple[int, dict, Any]]:
        all_res = []
        while self.in_flight > 0 and not self.stopped:
            items = self.poll(block_oldest=True)
            if not items and self.stopped:
                break
            all_res.extend(items)
        return all_res

    def close(self) -> None:
        self.stopped = True
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        for t in self.worker_threads:
            t.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def decode_fourcc(val: float) -> str:
    """Chuyển đổi mã FourCC số thực/nguyên sang chuỗi 4 ký tự (ví dụ: 'MJPG', 'YUY2')."""
    try:
        int_val = int(val)
        chars = [chr((int_val >> (8 * i)) & 0xFF) for i in range(4)]
        decoded = "".join(chars).strip()
        return decoded if decoded else f"0x{int_val:08X}"
    except Exception:
        return str(val)


def open_camera(index: int, prefer_mjpg: bool = True) -> Optional[cv2.VideoCapture]:
    """
    Mở camera với các thiết lập chuẩn hóa cho cả Collector và Predict:
    - Thử theo thứ tự backend: MSMF -> DSHOW -> DEFAULT.
    - Kích thước: 640x480.
    - Ép FourCC MJPG (nếu prefer_mjpg=True) để vượt qua giới hạn FPS của định dạng YUY2.
    - Đặt FPS = 30, Buffer Size = 1 để giảm độ trễ hàng đợi khung hình.
    - Đọc lại giá trị thực từ driver và in ra console.
    - Đọc xả 5 frame đầu tiên để làm nóng cảm biến và auto-exposure.
    """
    def _try_open(idx: int, backend_id: int) -> Optional[cv2.VideoCapture]:
        _cap = cv2.VideoCapture(idx, backend_id)
        if not _cap.isOpened():
            _cap.release()
            return None
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if prefer_mjpg:
            _cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        _cap.set(cv2.CAP_PROP_FPS, 30)
        _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, _ = _cap.read()
        if not ret:
            _cap.release()
            return None
        return _cap

    backend_order = [
        (cv2.CAP_MSMF, "MSMF (Microsoft Media Foundation)"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_ANY, "Default Backend"),
    ]

    print(f"[Camera] Đang khởi tạo Camera Index [{index}]...")
    cap: Optional[cv2.VideoCapture] = None
    chosen_backend_name = "None"
    for backend_id, backend_name in backend_order:
        cap = _try_open(index, backend_id)
        if cap is not None:
            chosen_backend_name = backend_name
            print(f"  → Backend: {backend_name}")
            break
        print(f"  [Skip] {backend_name} không đọc được frame tại index {index}")

    if cap is None or not cap.isOpened():
        print(f"[!] KHÔNG THỂ mở Camera [{index}] với bất kỳ backend nào!")
        return None

    # Đọc lại các giá trị thực tế từ driver sau khi đặt
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    actual_buf = cap.get(cv2.CAP_PROP_BUFFERSIZE)

    cam_type_name = "iPhone (Iriun Webcam)" if index == 1 else ("Webcam Laptop" if index == 0 else f"Camera Phụ {index}")
    print(f"[Camera] KẾT NỐI THÀNH CÔNG: {cam_type_name} (Index {index}, Backend: {chosen_backend_name})")
    print(f"[Camera] THÔNG SỐ THỰC TỪ DRIVER: FOURCC={actual_fourcc} | FPS={actual_fps:.1f} | Kích thước={actual_w}x{actual_h} | Buffer={actual_buf}")

    # Đọc bỏ 5 frame đầu để làm nóng camera / cân bằng sáng
    for _ in range(5):
        cap.read()

    return cap


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

        if timestamp is None or self._prev_ts is None:
            self.prev_x = self.alpha * float(raw_x) + (1.0 - self.alpha) * self.prev_x
            self.prev_y = self.alpha * float(raw_y) + (1.0 - self.alpha) * self.prev_y
            return self.prev_x, self.prev_y

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


def make_wrist_smoother(
    alpha: float = 0.70,
    f_c_min: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0
) -> WristCoordinateSmoother:
    """
    Hàm tạo bộ làm mượt cổ tay duy nhất dùng chung cho cả Collector và Predict:
    alpha=0.70, f_c_min=1.0, beta=0.007, d_cutoff=1.0 (cấu hình 1€ Filter của Predict).
    """
    return WristCoordinateSmoother(
        alpha=alpha,
        f_c_min=f_c_min,
        beta=beta,
        d_cutoff=d_cutoff
    )


def make_segmenter(config: Optional[Dict[str, Any]] = None) -> KineticPauseSegmenter:
    """
    Hàm tạo KineticPauseSegmenter duy nhất dùng chung cho cả Collector và Predict.
    Các ngưỡng lấy từ config (hoặc mặc định chuẩn của CONFIG).
    """
    cfg = config or SHARED_CONFIG
    return KineticPauseSegmenter(
        v_start_ratio=cfg.get("v_start_ratio", 0.08),
        v_stop_ratio=cfg.get("v_stop_ratio", 0.040),
        pause_frames=cfg.get("pause_frames", 9),
        min_stroke_frames=8,
        max_stroke_frames=60,
        swipe_threshold=-0.80,
        swipe_ratio=2.5,
        swipe_cooldown_sec=1.0,
        require_ready_pose=True,
        max_phase_merges=1,
        grace_window_sec=0.15
    )


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


class SlidingFPSMeter:
    """Bộ đo FPS trượt trong cửa sổ 1.0 giây chuẩn xác."""
    def __init__(self, window_sec: float = 1.0):
        self.window_sec = float(window_sec)
        self.timestamps: List[float] = []

    def update(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        self.timestamps.append(now)
        cutoff = now - self.window_sec
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
        n = len(self.timestamps)
        if n < 2:
            return 30.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 1e-4:
            return 30.0
        return float((n - 1) / elapsed)
