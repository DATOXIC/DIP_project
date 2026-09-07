"""
CHƯƠNG TRÌNH NHẬN DIỆN CỬ CHỈ ĐỘNG & GHÉP CÂU THỜI GIAN THỰC (VSL SENTENCE BUILDER)
===================================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Kiến trúc 4 Tầng Tinh Gọn (Lean 4-Layer Real-time Pipeline):
  1. TẦNG 1: Trích xuất Đặc trưng Tức thời (BlazeHand + DIP Engine v4 + Two-Stream CNN).
  2. TẦNG 2: Bộ Dò Điểm Dừng Phân Đoạn theo Palm-Scale (KineticPauseSegmenter).
  3. TẦNG 3: Bộ So Khớp Chuỗi Lai (HybridDTWEngine) với dải băng Sakoe-Chiba.
  4. TẦNG 4: Bộ Điều Phối Ghép Câu Hoàn Chỉnh & Giao Diện Thoại (Sentence Builder & HUD):
     - Tự động ghép từ thành câu có nghĩa.
     - Lệnh ưu tiên cấp tốc Swipe-Left (Gạt tay sang trái) = Xóa từ (Backspace).
     - Hạ tay khỏi màn hình > 1.8s = Chốt câu hoàn chỉnh (Commit).

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
from typing import List, Tuple, Optional

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


def run_sentence_builder():
    print("=" * 70)
    print("  HỆ THỐNG NHẬN DIỆN CỬ CHỈ ĐỘNG & GHÉP CÂU VSL (REAL-TIME SENTENCE BUILDER)")
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

    # 2. Khởi tạo Two-Stream CNN nhận diện Dáng tay tĩnh (Tầng 1)
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
        print("[!] Không tìm thấy best_model_v2.pth, dùng chế độ giả lập dáng tay.")
        cnn_model = None
        classes = ['A', 'B', 'C', 'L', 'V']

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
        print("[!] CANH BAO: dynamic_templates.json chua co mau nao hoac khong ton tai!")
        print("[!] Vui long chay 'collect_dynamic_templates.py' truoc de thu mau cu chi.")

    segmenter = KineticPauseSegmenter(
        v_start_ratio=0.08,
        v_stop_ratio=0.035,
        pause_frames=11,
        swipe_threshold=-0.80,
        swipe_ratio=2.5,
        swipe_cooldown_sec=1.0
    )

    # 4. Quản lý Bộ đệm Câu thoại (Tầng 4)
    sentence_words: List[str] = []
    last_commit_time = 0.0
    last_committed_word: Optional[str] = None
    last_hand_seen_time = time.time()
    sentence_finalized = False

    # Thông báo Toast phản hồi trực quan
    toast_message = ""
    toast_color = (0, 255, 0)
    toast_expire = 0.0

    # Vệt sáng quỹ đạo bàn tay
    trail_points: List[Tuple[int, int]] = []
    hand_lost_frames = 0

    # Mở Webcam DirectShow
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    flip_cam = False
    prev_time = time.time()
    fps = 0.0

    print("\n[HƯỚNG DẪN SỬ DỤNG]")
    print("  - Múa cử chỉ rõ ràng, khựng tay dừng dứt khoát 0.35s để chốt từ.")
    print("  - Gạt nhanh tay sang trái (Swipe-Left) để XÓA TỪ gần nhất (Backspace).")
    print("  - Hạ tay xuống khỏi màn hình > 1.8s để HOÀN THÀNH CÂU (Enter).")
    print("  [c]: Xóa toàn bộ câu | [f]: Lật gương camera | [q]: Thoát\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip_cam:
            frame = cv2.flip(frame, 1)

        h_f, w_f = frame.shape[:2]
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-4))
        prev_time = curr_time

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        det_result = detector.detect(mp_image)

        hand_found = False
        wrist_x, wrist_y = 0.0, 0.0
        x1, y1, x2, y2 = 0, 0, 0, 0
        palm_len = 50.0
        hand_shape = "B"
        seg_state = KineticState.IDLE
        completed_stroke = None
        is_swipe = False
        v_norm = 0.0

        if det_result.hand_landmarks and len(det_result.hand_landmarks) > 0:
            hand_lost_frames = 0
            hand_found = True
            last_hand_seen_time = curr_time
            sentence_finalized = False
            landmarks = det_result.hand_landmarks[0]
            handedness_label = det_result.handedness[0][0].category_name if det_result.handedness else "Right"

            # 1. Điểm mốc cổ tay & Kích thước Palm Scale (LM 0 - LM 9)
            lm0 = landmarks[0]
            lm9 = landmarks[9]
            wrist_x = lm0.x * w_f
            wrist_y = lm0.y * h_f
            palm_len = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)

            # 2. Nhận diện Dáng tay tức thời qua DIP Engine v4 + Two-Stream CNN
            hand_crop, (x1, y1, x2, y2), touching_border, _ = extract_smart_hand_crop(
                frame, landmarks, handedness_label=handedness_label, auto_flip_left=False
            )
            if hand_crop.shape[0] >= 30 and hand_crop.shape[1] >= 30 and cnn_model is not None and not touching_border:
                tensor_np, _, _ = process_lean_dip_tensor(hand_crop, target_size=(128, 128))
                t_pt = torch.from_numpy(tensor_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    logits = cnn_model(t_pt)
                    probs = F.softmax(logits, dim=1).numpy()[0]
                    hand_shape = classes[int(np.argmax(probs))]

            # 3. Cập nhật Bộ phân đoạn động học (Tầng 2)
            seg_state, completed_stroke, is_swipe, v_norm = segmenter.update(
                wrist_x=wrist_x,
                wrist_y=wrist_y,
                palm_len=palm_len,
                hand_shape=hand_shape,
                timestamp=curr_time
            )

            # 4. Xử lý LỆNH ƯU TIÊN: SWIPE-LEFT (BACKSPACE)
            if is_swipe:
                if len(sentence_words) > 0:
                    removed = sentence_words.pop()
                    toast_message = f"<- DA XOA TU: '{removed}'"
                    toast_color = (0, 140, 255)
                    toast_expire = curr_time + 1.6
                    print(f"[SentenceBuilder] Swipe-Left: Đã xóa '{removed}'")
                else:
                    toast_message = "<- CAU DANG TRONG, KHONG CO GI DE XOA"
                    toast_color = (0, 200, 255)
                    toast_expire = curr_time + 1.2
                trail_points.clear()

            # Quản lý vệt sáng quỹ đạo
            if seg_state == KineticState.RECORDING:
                trail_points.append((int(wrist_x), int(wrist_y)))
            else:
                if len(trail_points) > 0:
                    trail_points.pop(0)

            # 5. Xử lý CỬ CHỈ HOÀN THÀNH -> SO KHỚP HYBRID DTW (Tầng 3)
            if completed_stroke is not None:
                resampled = resample_gesture_sequence(completed_stroke, k_target=32)
                if resampled is not None:
                    match_result = dtw_engine.match(resampled)

                    # a) Chấp nhận từ vựng (ACCEPTED)
                    if match_result.status == "ACCEPTED" and match_result.top_display:
                        word_to_add = match_result.top_display

                        # Chống bắn lặp từ trong vòng 0.8s
                        is_repeat = (word_to_add == last_committed_word and (curr_time - last_commit_time) < 0.8)
                        if not is_repeat:
                            sentence_words.append(word_to_add)
                            last_commit_time = curr_time
                            last_committed_word = word_to_add
                            toast_message = f"+ '{word_to_add}' (Dist: {match_result.best_distance:.2f})"
                            toast_color = (0, 255, 0)
                            toast_expire = curr_time + 1.8
                            print(f"[SentenceBuilder] Nhận từ: '{word_to_add}' | Câu hiện tại: {' '.join(sentence_words)}")

                    # b) Nghi vấn Top-2 (UNCERTAIN)
                    elif match_result.status == "UNCERTAIN" and len(match_result.candidates) >= 2:
                        c1, c2 = match_result.candidates[0], match_result.candidates[1]
                        toast_message = f"? Nghi van: {c1.display_name} ({c1.confidence*100:.0f}%) | {c2.display_name} ({c2.confidence*100:.0f}%)"
                        toast_color = (0, 255, 255)
                        toast_expire = curr_time + 2.0

                    # c) Từ chối (REJECTED)
                    else:
                        toast_message = "X Cu chi khong khop (Tu choi)"
                        toast_color = (0, 0, 255)
                        toast_expire = curr_time + 1.0

                    trail_points.clear()

            # Vẽ Bounding Box quanh bàn tay
            if x2 > x1 and y2 > y1:
                box_col = (0, 255, 0) if seg_state == KineticState.RECORDING else (255, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_col, 2)
                cv2.putText(display_frame, f"Dang: {hand_shape}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1)

        else:
            # Khi không thấy tay: Cho phép mất dấu tối đa 4 frames (grace period) trước khi hủy nét vẽ cử chỉ
            hand_lost_frames += 1
            if hand_lost_frames >= 4:
                segmenter.reset()
                trail_points.clear()

            # Kiểm tra Timeout chốt câu hoàn chỉnh (1.8s)
            if len(sentence_words) > 0 and (curr_time - last_hand_seen_time) > 1.8 and not sentence_finalized:
                sentence_finalized = True
                toast_message = "✓ DA HOAN TAT CAU (ENTER)!"
                toast_color = (0, 255, 0)
                toast_expire = curr_time + 2.0
                print(f"[SentenceBuilder] Câu hoàn chỉnh: {' '.join(sentence_words)}.")

        # =====================================================================
        # HIỂN THỊ GIAO DIỆN HUD THÔNG MINH (TẦNG 4)
        # =====================================================================
        # 1. Vẽ vệt sáng quỹ đạo bàn tay
        if len(trail_points) > 1:
            for i in range(1, len(trail_points)):
                pt1 = trail_points[i - 1]
                pt2 = trail_points[i]
                cv2.line(display_frame, pt1, pt2, (0, 255, 255), 4)

        # 2. Thanh trạng thái trên cùng
        cv2.rectangle(display_frame, (0, 0), (w_f, 44), (20, 20, 20), -1)
        fps_color = (0, 255, 0) if fps >= 24 else (0, 165, 255)
        cv2.putText(display_frame, f"FPS: {fps:.1f} (HYBRID DTW)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

        # Trạng thái vận động
        k_color = (0, 255, 0) if seg_state == KineticState.RECORDING else (180, 180, 180)
        k_text = "DANG CHUYEN DONG" if seg_state == KineticState.RECORDING else ("KHUNG TAY" if hand_found else "CHO TAY")
        cv2.putText(display_frame, f"| {k_text}", (245, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, k_color, 2)

        # Thanh đo vận tốc chuẩn hóa
        v_bar = int(min(1.0, v_norm / 0.15) * 80)
        cv2.putText(display_frame, f"v: {v_norm*100:.1f}%", (w_f - 160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        cv2.rectangle(display_frame, (w_f - 95, 15), (w_f - 95 + v_bar, 28), (0, 255, 255), -1)
        cv2.rectangle(display_frame, (w_f - 95, 15), (w_f - 15, 28), (100, 100, 100), 1)

        # 3. Toast phản hồi nổi bật hoặc Cảnh báo từ điển rỗng
        if len(dtw_engine.vocabulary) == 0:
            cv2.rectangle(display_frame, (10, 52), (w_f - 10, 88), (0, 0, 180), -1)
            cv2.putText(display_frame, "CHUA CO TU DIEN! HAY CHAY collect_dynamic_templates.py",
                        (18, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        elif curr_time < toast_expire:
            cv2.rectangle(display_frame, (10, 52), (480, 95), (10, 10, 10), -1)
            cv2.rectangle(display_frame, (10, 52), (480, 95), toast_color, 2)
            cv2.putText(display_frame, toast_message, (20, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, toast_color, 2)

        # 4. HỘP PHỤ ĐỀ CÂU THOẠI HOÀN CHỈNH Ở ĐÁY MÀN HÌNH
        cv2.rectangle(display_frame, (0, h_f - 75), (w_f, h_f - 24), (10, 10, 10), -1)
        border_box_col = (0, 255, 0) if sentence_finalized else (0, 200, 255)
        cv2.rectangle(display_frame, (0, h_f - 75), (w_f, h_f - 24), border_box_col, 2)

        if len(sentence_words) > 0:
            sentence_str = " ".join(sentence_words)
            if sentence_finalized:
                sentence_str += " ."
            sentence_display = f"CAU: \"{sentence_str}\""
        else:
            sentence_display = "CAU: (Dang cho cu chi dau tien...)"

        text_col = (0, 255, 0) if sentence_finalized else (255, 255, 255)
        font_scale = 0.72
        (t_w, _), _ = cv2.getTextSize(sentence_display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        max_allowed_w = w_f - 30
        if t_w > max_allowed_w and t_w > 0:
            font_scale = max(0.40, font_scale * (max_allowed_w / t_w))
        cv2.putText(display_frame, sentence_display, (15, h_f - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, 2)

        # 5. Thanh hướng dẫn chân trang
        cv2.putText(display_frame, "[Gat trai]: Xoa tu (Backspace) | [Ha tay]: Chot cau | [c]: Xoa trang | [q]: Thoat",
                    (10, h_f - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow("Nhan Dien Cu Chi Dong & Ghep Cau (VSL Sentence Builder)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('c'), ord('C')]:
            sentence_words.clear()
            sentence_finalized = False
            toast_message = "DA XOA TOAN BO CAU (CLEAR)"
            toast_color = (0, 200, 255)
            toast_expire = curr_time + 1.5
            print("[SentenceBuilder] Đã xóa toàn bộ câu thoại.")
        elif key == ord('f'):
            flip_cam = not flip_cam
        elif key in [ord('p'), ord('P')]:
            cap.set(cv2.CAP_PROP_SETTINGS, 1)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("-> Đã đóng chương trình nhận diện ghép câu.")


if __name__ == "__main__":
    run_sentence_builder()
