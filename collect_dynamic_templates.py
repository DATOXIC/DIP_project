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


# Thư mục lưu trữ STROKE THÔ (raw stroke archive) — xem write_raw_stroke_archive().
RAW_STROKES_DIR = "raw_strokes"

# Phiên bản schema của file lưu trữ thô. TĂNG số này mỗi khi thay đổi cấu trúc,
# để rebuild_templates_from_raw.py biết cách đọc đúng từng thế hệ dữ liệu.
#   raw_stroke_v1.0 : không có hộp mặt => KHÔNG tái tạo được mốc cơ thể (SPEC 03)
#   raw_stroke_v1.1 : mỗi frame có thêm "face_box": [cx, cy, w, h] (pixel khung gốc) hoặc null
RAW_STROKE_SCHEMA_VERSION = "raw_stroke_v1.1"


def write_raw_stroke_archive(
    word_key: str,
    display_name: str,
    frame_records: List[Dict[str, Any]],
    meta: Dict[str, Any]
) -> Optional[str]:
    """
    Ghi TOÀN BỘ dữ liệu THÔ của một stroke ra raw_strokes/<WORD>/<timestamp>.json.

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
        out_dir = os.path.join(RAW_STROKES_DIR, safe_word)
        os.makedirs(out_dir, exist_ok=True)

        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int((now % 1.0) * 1000):03d}"
        out_path = os.path.join(out_dir, f"{stamp}.json")

        payload = {
            "schema_version": RAW_STROKE_SCHEMA_VERSION,
            "word_key": word_key,
            "display_name": display_name,
            "num_frames": len(frame_records),
            **meta,
            "frames": frame_records
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        return out_path
    except Exception as e:
        print(f"[RawArchive] Khong ghi duoc stroke tho: {e}")
        return None


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


DEFAULT_VOCAB = {
    '1': ('HELLO',     'Hello / Xin chào',    'Vẫy tay mở B qua lại 1 nhịp'),
    '2': ('BYE',       'Bye / Tạm biệt',      'Nâng tay B lên và vẫy ngón tay'),
    '3': ('ME',        'Me / Tôi',            'Chỉ ngón trỏ L/1 vào ngực'),
    '4': ('YOU',       'You / Bạn',           'Chỉ ngón trỏ L/1 về phía camera'),
    '5': ('THANK_YOU', 'Thank You / Cảm ơn',  'Chạm tay B/Flat dưới cằm rồi đưa ra trước'),
    '6': ('GO',        'Go / Đi',             'Đưa hai ngón trỏ hướng ra xa'),
    '7': ('COME',      'Come / Đến',          'Hai ngón trỏ ngoắc về phía người'),
    '8': ('WANT',      'Want / Muốn',         'Hai tay mở ngửa kéo lùi về người và co nhẹ'),
    '9': ('MORE',      'More / Thêm',         'Hai bàn tay chụm ngón chạm vào nhau lặp lại'),
    '0': ('WATER',     'Water / Nước',        'Chữ W (3 ngón giữa) chạm cằm 2 lần'),
    'a': ('HELP',      'Help / Giúp',         'Tay A đặt lên lòng bàn tay kia nâng lên'),
    'b': ('GOOD',      'Good / Tốt',          'Chạm cằm đưa xuống lòng bàn tay kia'),
    'c': ('NAME',      'Name / Tên',          'Hai ngón H (trỏ+giữa) gõ chéo lên nhau'),
    'e': ('DRINK',     'Drink / Uống',        'Tay C đưa lên miệng như cầm cốc uống'),
    'g': ('STOP',      'Stop / Dừng',         'Cạnh bàn tay B chém xuống lòng bàn tay kia'),
}

TARGET_SAMPLES_PER_WORD = 3


def run_dynamic_collector(camera_index: int = 0, templates_path: Optional[str] = None):
    print("=" * 70)
    print("  CÔNG CỤ THU MẪU CỬ CHỈ ĐỘNG THỦ NGỮ VSL (DYNAMIC COLLECTOR v1.0)")
    print("=" * 70)

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

    # 2. Khởi tạo Two-Stream CNN nhận diện Dáng tay
    device = torch.device('cpu')
    model_path = 'vsl_dataset/models/best_model_v2.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        classes = [checkpoint['id_to_label'][i] for i in range(checkpoint['num_classes'])]
        cnn_model = TwoStreamHandCNN(num_classes=len(classes))
        cnn_model.load_state_dict(checkpoint['model_state'])
        cnn_model.eval()
        print(f"✓ Đã nạp Two-Stream CNN ({model_path}): {classes}")
    else:
        print("[!] Không tìm thấy best_model_v2.pth, dùng nhãn giả lập.")
        cnn_model = None
        classes = ['A', 'B', 'C', 'L', 'V']

    # 3. Khởi tạo DTW Engine & Segmenter
    templates_file = templates_path if templates_path is not None else CONFIG.get("templates_path", "dynamic_templates.json")
    dtw_engine = HybridDTWEngine(templates_path=templates_file)
    dtw_matcher = GoldenTriadDTWMatcher(templates_path=templates_file, strict_weights=False)
    segmenter = KineticPauseSegmenter(
        v_start_ratio=0.08,
        v_stop_ratio=0.040,
        pause_frames=9,
        require_ready_pose=True,
        max_phase_merges=1,
        grace_window_sec=0.15
    )
    wrist_smoother = WristCoordinateSmoother(alpha=0.70)

    # SPEC 03: bộ dò mặt có cache (mỗi 8 frame mới chạy mạng). Bộ thu mẫu LUÔN bật để archive thô
    # lưu được hộp mặt cho từng frame — nhờ đó mốc cơ thể tái tạo được offline mà không phải thu lại.
    try:
        face_detector: Optional[CachedFaceDetector] = CachedFaceDetector(detect_every_n_frames=8)
        print("✓ Đã nạp bộ dò mặt (mốc cơ thể) có cache.")
    except Exception as e:
        print(f"[!] KHÔNG nạp được bộ dò mặt ({e}) -> mẫu thu sẽ KHÔNG có mốc cơ thể (body_anchor_valid=False).")
        face_detector = None
    current_face_box = None

    # 4. Mở Webcam — thử MSMF trước (tốt hơn với Iriun Webcam iPhone), rồi fallback DSHOW / Default
    def _try_open_camera(index: int, backend) -> "cv2.VideoCapture | None":
        _cap = cv2.VideoCapture(index, backend)
        if not _cap.isOpened():
            _cap.release()
            return None
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, _ = _cap.read()
        if not ret:
            _cap.release()
            return None
        return _cap

    print(f"[Camera] Dang khoi tao Camera Index [{camera_index}]...")
    cap = None
    for backend_id, backend_name in [
        (cv2.CAP_MSMF,  "MSMF (Microsoft Media Foundation)"),
        (cv2.CAP_DSHOW, "DSHOW (DirectShow)"),
        (cv2.CAP_ANY,   "DEFAULT Backend"),
    ]:
        cap = _try_open_camera(camera_index, backend_id)
        if cap is not None:
            print(f"  -> Backend: {backend_name}")
            break
        print(f"  [Skip] {backend_name} khong doc duoc frame tai index {camera_index}")

    if cap is not None and cap.isOpened():
        for _ in range(4):
            cap.read()
        cam_type_name = "iPhone (Iriun Webcam)" if camera_index == 1 else ("Webcam Laptop" if camera_index == 0 else f"Camera Phu {camera_index}")
        print(f"[Camera] KET NOI THANH CONG: {cam_type_name} (Index {camera_index})")
    else:
        print(f"[!] KHONG THE mo Camera [{camera_index}] voi bat ky backend nao!")
        print(f"    Goi y: thu 'py collect_dynamic_templates.py --cam 0' hoac '--cam 1'")
        return

    # Trạng thái phiên làm việc
    current_key = '1'
    curr_word_id, curr_display_name, curr_desc = DEFAULT_VOCAB[current_key]
    flip_cam = False
    trail_points: List[Tuple[int, int]] = []
    stroke_angles_history: List[np.ndarray] = []
    stroke_normals_history: List[np.ndarray] = []
    # Lịch sử nhãn tay thuận và bản ghi THÔ từng frame — cùng nhịp append với 2 list trên,
    # nên cùng dùng được phép cắt [-n_stroke:] khi stroke được chốt.
    stroke_handed_history: List[str] = []
    stroke_raw_history: List[Dict[str, Any]] = []
    # SPEC 03: (21 landmark pixel-z*w, hộp mặt) từng frame, cùng nhịp append với stroke_angles_history.
    stroke_ext_history: List[Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]] = []
    fps_estimate = 30.0
    prev_loop_time = time.time()
    flash_message = ""
    flash_color = (0, 255, 0)
    flash_time = 0.0
    hand_lost_frames = 0
    show_features = False
    show_overlay = True

    print("\n[HƯỚNG DẪN THU MẪU - 15 TỪ VỰNG CANONICAL ASL-LEX]")
    print("  [1-9, 0]: Chọn từ 1..10 (HELLO .. WATER)")
    print("  [a,b,c,e,g]: Chọn từ 11..15 (HELP, GOOD, NAME, DRINK, STOP)")
    print("  [d]     : Bật/Tắt bảng đặc trưng 12D (Feature HUD)")
    print("  [o]     : Bật/Tắt Overlay quỹ đạo mẫu cũ")
    print("  [r]     : Undo mẫu vừa thu gần nhất của từ hiện tại")
    print("  [x]     : XÓA SẠCH toàn bộ mẫu của từ hiện tại")
    print("  [f]     : Lật gương camera")
    print("  [s]     : Lưu file dynamic_templates.json")
    print("  [q]     : Thoát chương trình\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip_cam:
            frame = cv2.flip(frame, 1)

        h_f, w_f = frame.shape[:2]
        curr_time = time.time()
        # Ước lượng FPS trung bình trượt (EMA) để ghi vào metadata file thô
        _dt = curr_time - prev_loop_time
        if 1e-4 < _dt < 1.0:
            fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / _dt)
        prev_loop_time = curr_time
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        det_result = detector.detect(mp_image)
        # SPEC 03: hộp mặt (cx, cy, w, h) pixel khung gốc; gọi MỖI frame để cache đếm đúng độ cũ.
        current_face_box = face_detector.detect(rgb_frame, out_size=(w_f, h_f)) if face_detector is not None else None

        hand_found = False
        wrist_x, wrist_y = 0.0, 0.0
        palm_len = 50.0
        hand_shape = "B"
        hand_shape_conf: Optional[float] = None
        seg_state = KineticState.IDLE
        completed_stroke = None
        is_swipe = False
        v_norm = 0.0

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
            palm_2d = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)
            palm_3d = math.sqrt(((lm9.x - lm0.x) * w_f)**2 + ((lm9.y - lm0.y) * h_f)**2 + ((lm9.z - lm0.z) * w_f)**2)
            knuckle_span = math.hypot((lm17.x - lm5.x) * w_f, (lm17.y - lm5.y) * h_f)
            palm_len = max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)

            # Trích xuất 5 góc uốn ngón tay 3D & Vector pháp tuyến lòng bàn tay 3D (12D Features)
            lm_3d = np.array([[lm.x * w_f, lm.y * h_f, lm.z * w_f] for lm in landmarks], dtype=np.float32)
            current_angles = compute_joint_flexion_angles(lm_3d)
            current_normal = compute_palm_normal_vector(lm_3d, handedness=handedness_label)
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

            # 2. Nhận diện Dáng tay tức thời bằng Two-Stream CNN
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

            # 2b. LƯU TRỮ THÔ: ghi lại MỌI thứ của frame này để sau có thể tính lại
            # bất kỳ bộ đặc trưng nào (xem write_raw_stroke_archive).
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
                # SPEC 03 (schema raw_stroke_v1.1): hộp mặt [cx, cy, w, h] pixel khung gốc, hoặc null
                "face_box": ([round(float(v), 3) for v in current_face_box] if current_face_box is not None else None),
            })
            if len(stroke_raw_history) > 120:
                stroke_raw_history.pop(0)

            # 3. Cập nhật Bộ phân đoạn động học
            seg_state, completed_stroke, is_swipe, v_norm = segmenter.update(
                wrist_x=wrist_x,
                wrist_y=wrist_y,
                palm_len=palm_len,
                hand_shape=hand_shape,
                timestamp=curr_time
            )

            # Vẽ vệt đuôi chuyển động
            if seg_state == KineticState.RECORDING:
                trail_points.append((int(wrist_x), int(wrist_y)))
            else:
                if len(trail_points) > 0:
                    trail_points.pop(0)

            # 4. Khi cử chỉ hoàn thành (Khựng tay ngắt từ) -> LƯU MẪU TỰ ĐỘNG (Golden Triad 9D)
            if completed_stroke is not None:
                n_stroke = len(completed_stroke)

                # ---------------------------------------------------------------
                # CỔNG THỜI LƯỢNG (TASK 3): dùng ĐÚNG hằng số mà bộ suy luận dùng
                # ---------------------------------------------------------------
                # Trước đây bộ thu mẫu KHÔNG kiểm tra thời lượng, nên đã lọt 4 mẫu từ YOU
                # dài 3.04s - 3.59s. Những mẫu đó nằm ngoài cổng [0.40s, 2.50s] của
                # predict_sentence_webcam.py => KHÔNG BAO GIỜ có thể được sinh ra lúc suy
                # luận, chỉ làm phình phương sai nội lớp. Từ đây, thu mẫu và suy luận dùng
                # chung MỘT cặp hằng số (STROKE_DURATION_MIN_SEC / MAX) trong dynamic_dtw_engine.
                stroke_duration = completed_stroke[-1].timestamp - completed_stroke[0].timestamp
                if stroke_duration <= 0:
                    stroke_duration = n_stroke / max(fps_estimate, 1.0)

                if not is_stroke_duration_valid(stroke_duration):
                    if stroke_duration > STROKE_DURATION_MAX_SEC:
                        reason = "QUA CHAM"
                    else:
                        reason = "QUA NHANH"
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

            if completed_stroke is not None:
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

                if len(stroke_handed_history) >= n_stroke:
                    handed_for_stroke = stroke_handed_history[-n_stroke:]
                else:
                    pad_h = n_stroke - len(stroke_handed_history)
                    first_h = stroke_handed_history[0] if stroke_handed_history else "Right"
                    handed_for_stroke = [first_h] * pad_h + stroke_handed_history

                if len(stroke_ext_history) >= n_stroke:
                    ext_for_stroke = stroke_ext_history[-n_stroke:]
                else:
                    ext_for_stroke = [stroke_ext_history[0] if stroke_ext_history else (None, None)] * (n_stroke - len(stroke_ext_history)) + stroke_ext_history

                if len(stroke_raw_history) >= n_stroke:
                    raw_for_stroke = stroke_raw_history[-n_stroke:]
                else:
                    raw_for_stroke = list(stroke_raw_history)

                # Nhãn tay thuận CHIẾM ĐA SỐ trên toàn bộ stroke — bền hơn nhãn của
                # một frame lẻ (MediaPipe thỉnh thoảng nhảy nhãn ở frame mờ).
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
                # SPEC 03: nối 7 cột mới [z_norm, mốc cơ thể(2), độ xòe ngón(4)] => 24D (cùng công thức với 3 đường còn lại)
                raw_ext, body_ok = compute_extended_columns(
                    [f.palm_len for f in completed_stroke],
                    [(f.x, f.y) for f in completed_stroke],
                    [e[0] for e in ext_for_stroke],
                    [e[1] for e in ext_for_stroke]
                )
                raw_17d = append_extended_columns(raw_17d, raw_ext)
                if not body_ok:
                    print("[Collector] CẢNH BÁO: không dò được mặt trong stroke này -> mẫu không có mốc cơ thể (body_anchor_valid=False).")
                stroke_shapes = [f.hand_shape for f in completed_stroke]

                res_traj_17d, res_shapes = resample_9d_gesture_sequence(raw_17d, stroke_shapes, k_target=32)
                if res_traj_17d is not None:
                    duration = stroke_duration

                    # CHUẨN HÓA CHIRALITY NGAY TRƯỚC KHI LƯU: mẫu ghi xuống đĩa luôn nằm
                    # trong không gian chuẩn tắc, và ta ghi kèm nhãn tay thuận thật nên
                    # nhánh heuristic mean(nx) không bao giờ phải dùng cho dữ liệu mới.
                    res_traj_17d, was_flipped = canonicalize_chirality(res_traj_17d, handedness=dominant_handed)

                    # SPEC 03: số nhịp lặp tính trên chuỗi K=32 (cùng chỗ với match()) và cờ mốc cơ thể hợp lệ
                    # được lưu vào dict mẫu (qua ResampledGesture.extra), KHÔNG nhồi vào traj_features.
                    rep_count = compute_repetition_count(res_traj_17d[:, 0], res_traj_17d[:, 1])

                    resampled = ResampledGesture(
                        traj=res_traj_17d,
                        shapes=res_shapes,
                        duration_sec=float(duration),
                        handedness=dominant_handed,
                        chirality_canonical=True,
                        extra={"rep_count": int(rep_count), "body_anchor_valid": bool(body_ok)}
                    )

                    dtw_engine.add_sample(
                        word_key=curr_word_id,
                        display_name=curr_display_name,
                        sample=resampled
                    )
                    dtw_engine.save_templates()

                    # ---------------------------------------------------------------
                    # LƯU TRỮ STROKE THÔ (TASK 2) — bảo hiểm khỏi phải thu lại mẫu
                    # ---------------------------------------------------------------
                    sample_index = len(dtw_engine.vocabulary.get(curr_word_id, {}).get("samples", [])) - 1
                    raw_path = write_raw_stroke_archive(
                        word_key=curr_word_id,
                        display_name=curr_display_name,
                        frame_records=raw_for_stroke,
                        meta={
                            "camera_index": int(camera_index),
                            "frame_width": int(w_f),
                            "frame_height": int(h_f),
                            "flip_cam": bool(flip_cam),
                            "fps_estimate": round(float(fps_estimate), 3),
                            "stroke_duration_sec": round(float(duration), 6),
                            "dominant_handedness": dominant_handed,
                            "chirality_flipped_on_save": bool(was_flipped),
                            "k_resample": 32,
                            "templates_path": templates_file,
                            "template_sample_index": int(sample_index),
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
                            "wrist_smoother_alpha": 0.70,
                        }
                    )
                    stroke_raw_history.clear()
                    if raw_path:
                        print(f"[RawArchive] Da luu stroke tho: {raw_path}")

                    # Tính DTW distance preview với TOÀN BỘ các mẫu cũ trong cùng từ bằng đúng per-sign weights
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
                            # HybridDTWEngine (tầng lưu trữ 4D legacy) KHÔNG tự chuẩn hóa
                            # chirality — không được phép sửa vì tests/ cũ phụ thuộc hành vi
                            # của nó. Nên ta chuẩn hóa TẠI ĐÂY cho mẫu cũ, để bản xem trước
                            # DTW min/max nằm cùng không gian dấu với mẫu vừa thu.
                            if not old_s.chirality_canonical:
                                old_traj, _ = canonicalize_chirality(old_traj, handedness=old_s.handedness)
                            d = compute_golden_triad_dtw_distance(
                                q_traj=res_traj_17d,
                                q_shapes=res_shapes,
                                t_traj=old_traj,
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
                        flash_color = (0, 0, 255)  # Màu đỏ cảnh báo
                        flash_message = f"(!) MAU LECH BAT THUONG! DTW max={max_dist:.2f} (nguong {warn_thresh}) | Nhan [r] de XOA!"
                        flash_time = curr_time + 4.0
                    else:
                        flash_color = (0, 255, 0)  # Màu xanh lá chuẩn
                        flash_message = f"DA LUU MAU [{samples_count}] CHO '{curr_display_name}'!{preview_text}"
                        flash_time = curr_time + 2.5
                    trail_points.clear()
        else:
            hand_lost_frames += 1
            if hand_lost_frames >= 4:
                segmenter.reset()
                wrist_smoother.reset()
                trail_points.clear()
                stroke_angles_history.clear()
                stroke_normals_history.clear()
                stroke_handed_history.clear()
                stroke_ext_history.clear()
                stroke_raw_history.clear()

        # =====================================================================
        # HIỂN THỊ GIAO DIỆN HUD THU MẪU CHUYÊN NGHIỆP
        # =====================================================================
        # 1. Vẽ Overlay các quỹ đạo mẫu cũ (tham khảo trực quan thời gian thực)
        if show_overlay and hand_found:
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

        # 2. Thanh tiêu đề trên cùng
        cv2.rectangle(frame, (0, 0), (w_f, 44), (20, 20, 20), -1)
        state_color = (0, 255, 0) if seg_state == KineticState.RECORDING else (180, 180, 180)
        state_text = "DANG GHI CU CHI..." if seg_state == KineticState.RECORDING else "CHO CHUYEN DONG (IDLE)"
        cv2.putText(frame, f"[CAM {camera_index}] TRANG THAI: {state_text}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, state_color, 2)

        # Đo vận tốc chuẩn hóa
        v_bar_w = int(min(1.0, v_norm / 0.15) * 100)
        cv2.putText(frame, f"v: {v_norm*100:.1f}%", (w_f - 180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.rectangle(frame, (w_f - 110, 15), (w_f - 110 + v_bar_w, 28), (0, 255, 255), -1)
        cv2.rectangle(frame, (w_f - 110, 15), (w_f - 10, 28), (100, 100, 100), 1)

        # 3. Hộp thông tin từ đang thu
        cv2.rectangle(frame, (10, 55), (380, 135), (10, 10, 10), -1)
        cv2.rectangle(frame, (10, 55), (380, 135), (0, 255, 0), 2)
        samples_now = len(dtw_engine.vocabulary.get(curr_word_id, {}).get("samples", []))
        cv2.putText(frame, f"TU DANG THU: {curr_display_name} [{samples_now} mau]",
                    (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Dang tay: {hand_shape} | Huong dan: {curr_desc}",
                    (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(frame, "Mua cu chi -> DUNG TAY 0.35s de chot mau",
                    (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # Danh sách 15 từ vựng (2 Cột gọn gàng)
        y_list = 150
        cv2.rectangle(frame, (10, y_list - 8), (280, y_list + 165), (15, 15, 15), -1)
        cv2.rectangle(frame, (10, y_list - 8), (280, y_list + 165), (60, 60, 60), 1)
        
        vocab_items = list(DEFAULT_VOCAB.items())
        col1 = vocab_items[:8]
        col2 = vocab_items[8:]

        for i, (k_opt, (w_id, d_name, _)) in enumerate(col1):
            n_samp = len(dtw_engine.vocabulary.get(w_id, {}).get("samples", []))
            is_active = (k_opt == current_key)
            col = (0, 255, 0) if is_active else ((0, 200, 255) if n_samp >= 3 else (150, 150, 150))
            short_name = d_name.split('/')[0].strip()
            cv2.putText(frame, f"[{k_opt}] {short_name}:{n_samp}", (15, y_list + 12 + i * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        for i, (k_opt, (w_id, d_name, _)) in enumerate(col2):
            n_samp = len(dtw_engine.vocabulary.get(w_id, {}).get("samples", []))
            is_active = (k_opt == current_key)
            col = (0, 255, 0) if is_active else ((0, 200, 255) if n_samp >= 3 else (150, 150, 150))
            short_name = d_name.split('/')[0].strip()
            cv2.putText(frame, f"[{k_opt}] {short_name}:{n_samp}", (145, y_list + 12 + i * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        # 5. Thông báo Toast nổi bật khi vừa lưu mẫu (Xanh lá nếu chuẩn, Đỏ nếu Outlier)
        if curr_time < flash_time:
            cv2.rectangle(frame, (20, h_f // 2 - 32), (w_f - 20, h_f // 2 + 32), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, h_f // 2 - 32), (w_f - 20, h_f // 2 + 32), flash_color, 2)
            cv2.putText(frame, flash_message, (30, h_f // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, flash_color, 2)

        # 6. Bảng hiển thị đặc trưng 3D & Khung xương tay trực quan (12D Features HUD) khi bật 'd'
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

        # 7. Hướng dẫn chân trang
        cv2.putText(frame, "[1-9,0,a,b,c,e,g]: Chon tu | [d]: HUD | [o]: Overlay | [r]: Undo | [x]: Xoa | [q]: Thoat",
                    (10, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)

        cv2.imshow("Thu Thap Mau Cu Chi Dong (VSL Dynamic Collector)", frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('d'), ord('D')]:
            show_features = not show_features
            print(f"[HUD] {'BẬT' if show_features else 'TẮT'} bảng hiển thị đặc trưng 3D.")
        elif key in [ord('o'), ord('O')]:
            show_overlay = not show_overlay
            print(f"[Overlay] {'BẬT' if show_overlay else 'TẮT'} hiển thị quỹ đạo mẫu cũ.")
        elif chr(key).lower() in DEFAULT_VOCAB:
            current_key = chr(key).lower()
            curr_word_id, curr_display_name, curr_desc = DEFAULT_VOCAB[current_key]
            segmenter.reset()
            wrist_smoother.reset()
            trail_points.clear()
            stroke_angles_history.clear()
            stroke_normals_history.clear()
            stroke_ext_history.clear()
            print(f"-> Chuyển sang thu từ: {curr_display_name}")
        elif key == ord('r'):
            # Xóa mẫu gần nhất của từ hiện tại (Undo)
            if curr_word_id in dtw_engine.vocabulary and len(dtw_engine.vocabulary[curr_word_id]["samples"]) > 0:
                dtw_engine.vocabulary[curr_word_id]["samples"].pop()
                dtw_engine.save_templates()
                now_s = len(dtw_engine.vocabulary[curr_word_id]["samples"])
                flash_message = f"DA UNDO MAU! Con lai {now_s} mau cho '{curr_display_name}'"
                flash_time = curr_time + 1.5
                print(f"-> Đã xóa mẫu gần nhất của {curr_display_name} (còn lại {now_s} mẫu)")
        elif key in [ord('x'), ord('X')]:
            # Phím x: Xóa toàn bộ mẫu của từ hiện tại
            if curr_word_id in dtw_engine.vocabulary:
                dtw_engine.vocabulary[curr_word_id]["samples"].clear()
                dtw_engine.save_templates()
                flash_message = f"DA XOA SACH TOAN BO MAU CUA '{curr_display_name}'!"
                flash_time = curr_time + 1.5
                print(f"-> Đã xóa sạch toàn bộ mẫu của {curr_display_name}")
        elif key == ord('f'):
            flip_cam = not flip_cam
        elif key == ord('s'):
            dtw_engine.save_templates()
            flash_message = "DA LUU DYNAMIC_TEMPLATES.JSON!"
            flash_time = curr_time + 1.5

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("-> Đã đóng công cụ thu thập mẫu.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Dynamic Sign Language Templates")
    parser.add_argument("--cam", type=int, default=None, help="Camera index (0=built-in webcam, 1, 2=external/iPhone). Mặc định dùng camera_source từ CONFIG.")
    parser.add_argument("--templates", type=str, default=None, help="Đường dẫn file template JSON (ví dụ: dynamic_templates.backup_v51.json). Mặc định dùng templates_path từ CONFIG.")
    args, _ = parser.parse_known_args()
    
    selected_cam = args.cam if args.cam is not None else CONFIG.get("camera_source", 0)
    selected_templates = args.templates if args.templates is not None else CONFIG.get("templates_path", "dynamic_templates.json")
    run_dynamic_collector(camera_index=selected_cam, templates_path=selected_templates)
