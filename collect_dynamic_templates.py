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
from typing import Dict, Tuple, Optional, List

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
    HybridDTWConfig
)


DEFAULT_VOCAB = {
    '1': ('xin_chao', 'Xin chào', 'Vẫy tay mở B qua lại 1 nhịp'),
    '2': ('cam_on',   'Cảm ơn',   'Chạm tay B/Flat dưới cằm rồi đưa ra trước'),
    '3': ('toi',      'Tôi',      'Chỉ ngón trỏ L/1 vào ngực'),
    '4': ('ban',      'Bạn',      'Chỉ ngón trỏ L/1 về phía camera'),
    '5': ('tam_biet', 'Tạm biệt', 'Nâng tay B lên và lắc cổ tay'),
}

TARGET_SAMPLES_PER_WORD = 3


def run_dynamic_collector():
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
    templates_file = "dynamic_templates.json"
    dtw_engine = HybridDTWEngine(templates_path=templates_file)
    segmenter = KineticPauseSegmenter(
        v_start_ratio=0.08,
        v_stop_ratio=0.035,
        pause_frames=11
    )

    # 4. Mở Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Trạng thái phiên làm việc
    current_key = '1'
    curr_word_id, curr_display_name, curr_desc = DEFAULT_VOCAB[current_key]
    flip_cam = False
    trail_points: List[Tuple[int, int]] = []
    flash_message = ""
    flash_time = 0.0
    hand_lost_frames = 0

    print("\n[HƯỚNG DẪN THU MẪU]")
    print("  [1] - [5]: Chọn từ muốn thu mẫu (Xin chào, Cảm ơn, Tôi, Bạn, Tạm biệt)")
    print("  [r]      : Xóa mẫu vừa thu gần nhất của từ hiện tại")
    print("  [f]      : Lật gương camera")
    print("  [s]      : Lưu ngay từ điển ra file dynamic_templates.json")
    print("  [q]      : Thoát chương trình\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip_cam:
            frame = cv2.flip(frame, 1)

        h_f, w_f = frame.shape[:2]
        curr_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        det_result = detector.detect(mp_image)

        hand_found = False
        wrist_x, wrist_y = 0.0, 0.0
        palm_len = 50.0
        hand_shape = "B"
        seg_state = KineticState.IDLE
        completed_stroke = None
        is_swipe = False
        v_norm = 0.0

        if det_result.hand_landmarks and len(det_result.hand_landmarks) > 0:
            hand_lost_frames = 0
            hand_found = True
            landmarks = det_result.hand_landmarks[0]
            handedness_label = det_result.handedness[0][0].category_name if det_result.handedness else "Right"

            # 1. Điểm mốc cổ tay & Kích thước lòng bàn tay (Palm Scale LM 0 - LM 9)
            lm0 = landmarks[0]
            lm9 = landmarks[9]
            wrist_x = lm0.x * w_f
            wrist_y = lm0.y * h_f
            palm_len = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)

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

            # 4. Khi cử chỉ hoàn thành (Khựng tay ngắt từ) -> LƯU MẪU TỰ ĐỘNG
            if completed_stroke is not None:
                resampled = resample_gesture_sequence(completed_stroke, k_target=32)
                if resampled is not None:
                    curr_samples = len(dtw_engine.vocabulary.get(curr_word_id, {}).get("samples", []))
                    if curr_samples >= TARGET_SAMPLES_PER_WORD:
                        flash_message = f"DA DU {TARGET_SAMPLES_PER_WORD} MAU! Nhan [r] de xoa mau neu muon thu lai."
                        flash_time = curr_time + 2.5
                        trail_points.clear()
                    else:
                        dtw_engine.add_sample(
                            word_key=curr_word_id,
                            display_name=curr_display_name,
                            sample=resampled
                        )
                        # Tự động lưu ra đĩa
                        dtw_engine.save_templates()

                        samples_count = len(dtw_engine.vocabulary.get(curr_word_id, {}).get("samples", []))
                        flash_message = f"DA LUU MAU {samples_count}/{TARGET_SAMPLES_PER_WORD} CHO '{curr_display_name}'!"
                        flash_time = curr_time + 2.0
                        trail_points.clear()
        else:
            hand_lost_frames += 1
            if hand_lost_frames >= 4:
                segmenter.reset()
                trail_points.clear()

        # =====================================================================
        # HIỂN THỊ GIAO DIỆN HUD THU MẪU CHUYÊN NGHIỆP
        # =====================================================================
        # 1. Vẽ vệt sáng quỹ đạo bàn tay
        if len(trail_points) > 1:
            for i in range(1, len(trail_points)):
                pt1 = trail_points[i - 1]
                pt2 = trail_points[i]
                cv2.line(frame, pt1, pt2, (0, 255, 255), 3)

        # 2. Thanh tiêu đề trên cùng
        cv2.rectangle(frame, (0, 0), (w_f, 44), (20, 20, 20), -1)
        state_color = (0, 255, 0) if seg_state == KineticState.RECORDING else (180, 180, 180)
        state_text = "DANG GHI CU CHI..." if seg_state == KineticState.RECORDING else "CHO CHUYEN DONG (IDLE)"
        cv2.putText(frame, f"TRANG THAI: {state_text}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

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
        cv2.putText(frame, f"TU DANG THU: {curr_display_name} [{samples_now}/{TARGET_SAMPLES_PER_WORD}]",
                    (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Dang tay: {hand_shape} | Huong dan: {curr_desc}",
                    (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(frame, "Mua cu chi -> DUNG TAY 0.35s de chot mau",
                    (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # 4. Danh sách 5 từ vựng ở góc trái dưới
        y_list = 160
        cv2.rectangle(frame, (10, y_list - 15), (220, y_list + 115), (15, 15, 15), -1)
        cv2.rectangle(frame, (10, y_list - 15), (220, y_list + 115), (60, 60, 60), 1)
        for k_opt, (w_id, d_name, _) in DEFAULT_VOCAB.items():
            n_samp = len(dtw_engine.vocabulary.get(w_id, {}).get("samples", []))
            is_active = (k_opt == current_key)
            col = (0, 255, 0) if is_active else ((0, 200, 255) if n_samp >= TARGET_SAMPLES_PER_WORD else (150, 150, 150))
            prefix = "-> " if is_active else "   "
            cv2.putText(frame, f"{prefix}[{k_opt}] {d_name}: {n_samp}/3", (15, y_list + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            y_list += 22

        # 5. Thông báo Toast nổi bật khi vừa lưu mẫu
        if curr_time < flash_time:
            cv2.rectangle(frame, (80, h_f // 2 - 30), (w_f - 80, h_f // 2 + 30), (0, 0, 0), -1)
            cv2.rectangle(frame, (80, h_f // 2 - 30), (w_f - 80, h_f // 2 + 30), (0, 255, 0), 2)
            cv2.putText(frame, flash_message, (95, h_f // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

        # 6. Hướng dẫn chân trang
        cv2.putText(frame, "[1-5]: Chon tu | [r]: Xoa mau cuoi | [f]: Lat guong | [q]: Thoat",
                    (10, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        cv2.imshow("Thu Thap Mau Cu Chi Dong (VSL Dynamic Collector)", frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            current_key = chr(key)
            curr_word_id, curr_display_name, curr_desc = DEFAULT_VOCAB[current_key]
            segmenter.reset()
            trail_points.clear()
            print(f"-> Chuyển sang thu từ: {curr_display_name}")
        elif key == ord('r'):
            # Xóa mẫu gần nhất của từ hiện tại
            if curr_word_id in dtw_engine.vocabulary and len(dtw_engine.vocabulary[curr_word_id]["samples"]) > 0:
                dtw_engine.vocabulary[curr_word_id]["samples"].pop()
                dtw_engine.save_templates()
                now_s = len(dtw_engine.vocabulary[curr_word_id]["samples"])
                flash_message = f"DA XOA MAU! Con lai {now_s}/{TARGET_SAMPLES_PER_WORD}"
                flash_time = curr_time + 1.5
                print(f"-> Đã xóa mẫu gần nhất của {curr_display_name}")
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
    run_dynamic_collector()
