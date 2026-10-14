"""
CHƯƠNG TRÌNH NHẬN DIỆN THỦ NGỮ THỜI GIAN THỰC (LEAN CONSOLIDATED PIPELINE v4)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Kiến trúc Hợp nhất Tinh gọn (Lean Coarse-to-Fine Pipeline):
  1. Giai đoạn 1 (Định vị & Crop thông minh):
     - BlazeHand Detector siêu nhẹ (~7.8MB) bám theo bàn tay ở bất kỳ đâu trên màn hình.
     - Cắt đúng Landmark 0 (khớp cổ tay): triệt tiêu 100% cẳng tay (Case 4).
     - Tự động nhận diện tay Trái/Phải: tự lật gương tay trái về tay phải chuẩn (Case 14).
     - Kiểm tra chạm mép camera (Case 12).
  2. Giai đoạn 2 (DIP Tinh gọn - Không dùng mặt nạ màu da):
     - Bilateral Filter: Khử sạch nhiễu hạt ISO (Case 1) và Block JPEG (Case 18) nhưng GIỮ SẮC NÉT VIỀN.
     - Cân bằng trắng Gray-World: Khử ám sắc AWB drift (Case 2).
     - Image Moments: Xoay trục chính về 0° thẳng đứng và scale về 128x128 (Case 6).
     - Trích xuất Tensor 3 kênh DIP trực tiếp:
         * Kênh 0: CLAHE (hiện rõ ngón cái đè lòng bàn tay - Case 16).
         * Kênh 1: AutoCanny (viền ngón tay nguyên vẹn, chấp cả son móng tay - Case 13).
         * Kênh 2: Sobel Gradient Magnitude (bắt cấu trúc khi ngược sáng - Case 10).
  3. Giai đoạn 3 (Phân loại & Lọc nhảy):
     - Two-Stream CNN (tự động ưu tiên best_model_v3.pth, chạy an toàn với best_model_v2.pth).
     - Bộ lọc đa số SlidingWindowVoter (N=7 frames, triệt tiêu 1s trễ sáng - Case 19).

Phím điều khiển:
  - 'f' : Bật/Tắt lật gương camera (Mirror Flip)
  - 'p' : Mở bảng cài đặt phần cứng Camera (khóa Auto-Exposure / White Balance)
  - 'q' : Thoát chương trình
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import os
import sys
import time
from pathlib import Path
from collections import deque, Counter
from typing import Tuple, Optional, List, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp

from train import TwoStreamHandCNN
from improved_dip_pipeline import (
    MotionTracker,
    extract_smart_hand_crop,
    process_lean_dip_tensor
)


# =============================================================================
# 1. BỘ LỌC ĐA SỐ & LÀM MƯỢT XÁC SUẤT (SLIDING WINDOW VOTER)
# =============================================================================

class SlidingWindowVoter:
    """
    Bộ lọc thời gian Bầu cử Đa số (Sliding Window Majority Voting):
    - Lưu lịch sử dự đoán của N khung hình gần nhất (N=7).
    - Tính trung bình xác suất (Soft Probability Smoothing) để thanh đo êm ái.
    - Đếm phiếu bầu đa số (Hard Majority Voting >= 4/7 phiếu).
    - Triệt tiêu 100% hiện tượng nhảy chữ (Anti-flicker).
    """
    def __init__(self, window_size: int = 7, min_votes_ratio: float = 0.55):
        self.window_size = window_size
        self.min_votes = int(np.ceil(window_size * min_votes_ratio))
        self.label_history = deque(maxlen=window_size)
        self.prob_history = deque(maxlen=window_size)
        self.current_stable_label = "..."
        self.current_stable_conf = 0.0
        self.vote_count = 0

    def update(self, pred_label: str, probs: np.ndarray, confidence: float) -> Tuple[str, float, int, int, np.ndarray]:
        self.label_history.append(pred_label)
        self.prob_history.append(probs)

        avg_probs = np.mean(list(self.prob_history), axis=0)
        valid_votes = [lbl for lbl in self.label_history if lbl not in ["...", "Khong ro", "CHAM MEP!"]]
        if not valid_votes:
            self.current_stable_label = pred_label
            self.current_stable_conf = confidence
            self.vote_count = 0
            return self.current_stable_label, self.current_stable_conf, 0, len(self.label_history), avg_probs

        counter = Counter(valid_votes)
        top_label, votes = counter.most_common(1)[0]
        self.vote_count = votes

        if votes >= self.min_votes:
            self.current_stable_label = top_label
            self.current_stable_conf = float(np.max(avg_probs))
        else:
            if self.current_stable_label == "...":
                self.current_stable_label = "Dang nhan..."
            self.current_stable_conf = float(np.max(avg_probs))

        return self.current_stable_label, self.current_stable_conf, votes, len(self.label_history), avg_probs

    def reset(self):
        self.label_history.clear()
        self.prob_history.clear()
        self.current_stable_label = "..."
        self.current_stable_conf = 0.0
        self.vote_count = 0


# =============================================================================
# 2. NẠP MÔ HÌNH HỌC SÂU (LOAD TRAINED CNN)
# =============================================================================

def load_trained_model(model_path: str, device: torch.device):
    """Nạp trọng số mô hình CNN và metadata nhãn."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file model tại: {model_path}\n"
            f"Hãy đảm bảo file mô hình đã được tải vào thư mục!"
        )

    checkpoint = torch.load(p, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 5)
    id_to_label = checkpoint.get("id_to_label", {0: 'A', 1: 'B', 2: 'C', 3: 'L', 4: 'V'})
    id_to_label = {int(k): str(v) for k, v in id_to_label.items()}

    model = TwoStreamHandCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"✓ Đã nạp Two-Stream CNN thành công (Val Acc: {val_acc:.2f}%)")
    print(f"  Các nhãn nhận diện: {list(id_to_label.values())}")

    return model, id_to_label


# =============================================================================
# 3. CHƯƠNG TRÌNH NHẬN DIỆN THỜI GIAN THỰC (MAIN INFERENCE LOOP)
# =============================================================================

def run_realtime_prediction(
    model_path: str = "./vsl_dataset/models/best_model_v2.pth",
    confidence_threshold: float = 0.60
):
    print("=" * 72)
    print("  HỆ THỐNG NHẬN DIỆN THỦ NGỮ VSL HỢP NHẤT (LEAN PIPELINE v4)")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị tính toán: {device}")

    # Tự động nạp: nếu có best_model_v3.pth thì dùng, còn chưa có thì chạy an toàn với best_model_v2.pth
    v3_path = "./vsl_dataset/models/best_model_v3.pth"
    v2_path = "./vsl_dataset/models/best_model_v2.pth"
    if os.path.exists(v3_path):
        active_model_path = v3_path
        print("-> Phát hiện mô hình mới nhất: [best_model_v3.pth]")
    elif os.path.exists(v2_path):
        active_model_path = v2_path
        print("-> Đang sử dụng mô hình ổn định: [best_model_v2.pth]")
    else:
        active_model_path = model_path

    try:
        model, id_to_label = load_trained_model(active_model_path, device)
    except FileNotFoundError as e:
        print(f"\n[LỖI] {e}")
        return

    # 1. Tải mô hình Hand Detector (BlazeHand ~7.8MB)
    model_asset_path = 'hand_landmarker.task'
    if not os.path.exists(model_asset_path):
        print(f"[-] Đang tải model {model_asset_path}...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_asset_path)
        print("✓ Tải xong Hand Detector!")

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

    # 2. Khởi tạo Voter và Tracker
    voter = SlidingWindowVoter(window_size=7, min_votes_ratio=0.55)
    tracker = MotionTracker(history_len=16)
    num_classes = len(id_to_label)

    # 3. Mở Webcam với DirectShow
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n👉 Phím điều khiển:")
    print("  - [f] : Bật/Tắt lật gương camera (Mirror Flip)")
    print("  - [p] : Mở bảng cài đặt Camera (khóa Auto-Exposure / White Balance)")
    print("  - [q] : Thoát chương trình\n")

    flip_cam = False
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip_cam:
            frame = cv2.flip(frame, 1)

        display_frame = frame.copy()
        h_f, w_f = frame.shape[:2]

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
        prev_time = curr_time

        # Giai đoạn 1: Dò tìm vị trí bàn tay (BlazeHand)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        det_result = detector.detect(mp_image)

        hand_found = False
        tensor_vis = np.zeros((128, 384, 3), dtype=np.uint8)
        canvas_vis = np.zeros((128, 128, 3), dtype=np.uint8)
        stable_label = "..."
        stable_conf = 0.0
        votes = 0
        window_len = 0
        smoothed_probs = np.zeros(num_classes)
        touching_border = False
        is_flipped = False

        if det_result.hand_landmarks and len(det_result.hand_landmarks) > 0:
            hand_found = True
            landmarks = det_result.hand_landmarks[0]
            handedness_label = det_result.handedness[0][0].category_name if det_result.handedness else "Right"

            # Giai đoạn 1: Smart Crop cắt cổ tay (Case 4) & Border Check (Case 12)
            hand_crop, (x1, y1, x2, y2), touching_border, is_flipped = extract_smart_hand_crop(
                frame, landmarks, handedness_label=handedness_label, auto_flip_left=False
            )
            ch, cw = hand_crop.shape[:2]

            if ch >= 35 and cw >= 35:
                # Giai đoạn 2: Lean DIP Engine (Bilateral Filter + Moments + CLAHE + AutoCanny + Sobel)
                tensor_np, canvas_128, rot_angle = process_lean_dip_tensor(hand_crop, target_size=(128, 128))
                canvas_vis = canvas_128

                # Trực quan hóa 3 kênh Tensor
                vis_ch0 = cv2.cvtColor(tensor_np[:, :, 0], cv2.COLOR_GRAY2BGR)
                vis_ch1 = cv2.cvtColor(tensor_np[:, :, 1], cv2.COLOR_GRAY2BGR)
                vis_ch2 = cv2.cvtColor(tensor_np[:, :, 2], cv2.COLOR_GRAY2BGR)
                tensor_vis = np.hstack([vis_ch0, vis_ch1, vis_ch2])

                # Cập nhật Motion Tracker
                wrist_cx = int(landmarks[0].x * w_f)
                wrist_cy = int(landmarks[0].y * h_f)
                tracker.update(wrist_cx, wrist_cy)
                tracker.draw_trail(display_frame)

                # Giai đoạn 3: Phân loại bằng Two-Stream CNN
                if not touching_border:
                    tensor_pt = torch.from_numpy(tensor_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(tensor_pt)
                        raw_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                        raw_pred_idx = int(np.argmax(raw_probs))
                        raw_conf = float(raw_probs[raw_pred_idx])

                    if raw_conf >= confidence_threshold:
                        raw_label = id_to_label.get(raw_pred_idx, f"Lop {raw_pred_idx}")
                    else:
                        raw_label = "Khong ro"

                    # Bầu cử đa số 7 frames
                    stable_label, stable_conf, votes, window_len, smoothed_probs = voter.update(
                        raw_label, raw_probs, raw_conf
                    )
                else:
                    voter.reset()
                    stable_label = "CHAM MEP!"

            # Vẽ Dynamic Bounding Box ôm sát bàn tay
            box_color = (0, 0, 255) if touching_border else (0, 255, 0) if stable_label not in ["...", "Khong ro", "Dang nhan...", "CHAM MEP!"] else (0, 255, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            hand_str = f"Tay: {handedness_label}"
            cv2.putText(display_frame, f"{hand_str} [{x2-x1}x{y2-y1}]", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 1)

        else:
            voter.reset()
            tracker.reset()

        # =====================================================================
        # HIỂN THỊ GIAO DIỆN HUD THÔNG MINH
        # =====================================================================
        # 1. Thanh trạng thái trên cùng
        cv2.rectangle(display_frame, (0, 0), (w_f, 42), (20, 20, 20), -1)
        fps_color = (0, 255, 0) if fps >= 24 else (0, 165, 255)
        cv2.putText(display_frame, f"FPS: {fps:.1f} (LEAN v4)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

        motion_state = "MOVING" if tracker.is_moving() else "STABLE"
        status_txt = f"TAY: {motion_state}" if hand_found else "DUA TAY VAO WEBCAM"
        status_col = (0, 255, 0) if hand_found else (0, 0, 255)
        cv2.putText(display_frame, f"| {status_txt}", (235, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        if touching_border:
            cv2.putText(display_frame, "! DUA TAY VAO GIUA WEBCAM !", (w_f - 290, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

        # 2. Khung kết quả nhận diện nổi bật
        if stable_label not in ["...", "Khong ro", "Dang nhan...", "CHAM MEP!"]:
            res_text = f"CHU CAI: {stable_label} ({stable_conf*100:.0f}%) [{votes}/{window_len}]"
            cv2.rectangle(display_frame, (10, 52), (390, 102), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, 52), (390, 102), (0, 255, 0), 2)
            cv2.putText(display_frame, res_text, (20, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        elif stable_label == "CHAM MEP!":
            cv2.rectangle(display_frame, (10, 52), (390, 102), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, 52), (390, 102), (0, 0, 255), 2)
            cv2.putText(display_frame, "TAY DANG CHAM MEP CAM", (20, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 3. Bảng xác suất các chữ cái (làm mượt bằng smoothed_probs)
        if hand_found and np.sum(smoothed_probs) > 0 and not touching_border:
            y_offset = 125
            for i, prob in enumerate(smoothed_probs):
                lbl_name = id_to_label.get(i, str(i))
                bar_len = int(prob * 110)
                is_top = (lbl_name == stable_label and stable_conf >= confidence_threshold)
                col = (0, 255, 0) if is_top else (180, 180, 180)
                cv2.putText(display_frame, f"{lbl_name}: {prob*100:4.1f}%", (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
                cv2.rectangle(display_frame, (90, y_offset - 10), (90 + bar_len, y_offset), col, -1)
                y_offset += 22

        # 4. Hướng dẫn chân trang
        cv2.putText(display_frame, f"[f]: Lat guong ({'BAT' if flip_cam else 'TAT'}) | [p]: Cam Settings | [q]: Thoat",
                    (10, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Hiển thị các cửa sổ
        cv2.imshow("VSL Recognition - Webcam", display_frame)
        cv2.imshow("1. Ban Tay Chuan Hoa (Moments 128x128)", canvas_vis)

        # Đánh dấu tiêu đề cho 3 kênh Tensor
        cv2.putText(tensor_vis, "CLAHE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(tensor_vis, "CANNY", (138, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(tensor_vis, "SOBEL", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("2. Tensor 3 Kenh DIP (128x128x3)", tensor_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('p'), ord('P')]:
            print("\n-> Đang mở bảng Cài đặt Phần cứng Camera (Tắt Auto Exposure/White Balance)...")
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        elif key == ord('f'):
            flip_cam = not flip_cam
            voter.reset()
            print(f"-> Lật gương camera: {'BẬT' if flip_cam else 'TẮT'}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("-> Đã đóng chương trình nhận diện.")


if __name__ == "__main__":
    run_realtime_prediction()
