"""
CÔNG CỤ THU THẬP DỮ LIỆU TỪ WEBCAM (LEAN CONSOLIDATED v4)
==========================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Nâng cấp theo Kiến trúc Hợp nhất Tinh gọn:
  1. Giai đoạn 1 (Định vị & Crop thông minh):
     - BlazeHand Detector siêu nhẹ (~7.8MB) bám theo bàn tay ở bất kỳ đâu trên màn hình.
     - Cắt đúng Landmark 0 (khớp cổ tay): triệt tiêu 100% cẳng tay (Case 4).
     - Tự động nhận diện tay Trái/Phải: tự lật gương tay trái về tay phải chuẩn (Case 14).
     - Kiểm tra chạm mép camera (Case 12).
  2. Giai đoạn 2 (DIP Tinh gọn - Đồng bộ 100% với predict_webcam.py):
     - Bilateral Filter khử nhiễu ISO giữ viền (Case 1 & 18).
     - Cân bằng trắng Gray-World (Case 2).
     - Image Moments xoay trục chính về 0° và scale 128x128 (Case 6).
     - Trích xuất Tensor 3 kênh trực tiếp: [CLAHE | AutoCanny | Sobel].
  3. Thước đo chất lượng mẫu thông minh:
     - Đánh giá độ nét cạnh, độ ổn định chuyển động và an toàn mép viền.
     - Chống lưu ảnh rung nhòe, mờ nét.
  4. Quản lý lưu trữ an toàn:
     - Chống ghi đè file, tự động cập nhật số lượng mẫu hiện có trên đĩa.

Tập 5 ký hiệu chuẩn ASL:
  [1] Chữ 'A' - Nắm tay, ngón cái áp sát
  [2] Chữ 'B' - 4 ngón thẳng, ngón cái gập
  [3] Chữ 'C' - Bàn tay khum hình chữ C
  [4] Chữ 'L' - Ngón cái và ngón trỏ vuông góc 90 độ
  [5] Chữ 'V' - Xòe 2 ngón tay hình chữ V
"""

import os
import sys
import time
import math
from typing import Dict, Tuple, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import mediapipe as mp

from improved_dip_pipeline import (
    MotionTracker,
    extract_smart_hand_crop,
    process_lean_dip_tensor
)
from dataset_builder import DIPDatasetBuilder


# Cấu hình 5 ký hiệu chuẩn
GESTURE_CONFIG: Dict[str, Tuple[str, str]] = {
    '1': ('A', 'Chu A - Nam tay, ngon cai ap sat'),
    '2': ('B', 'Chu B - 4 ngon thang, ngon cai gap'),
    '3': ('C', 'Chu C - Ban tay khum hinh chu C'),
    '4': ('L', 'Chu L - Ngon cai va ngon tro vuong goc 90 do'),
    '5': ('V', 'Chu V - Xoe 2 ngon tay hinh chu V'),
}

TARGET_SAMPLES = 35  # Mục tiêu mỗi chữ cái thu thập khoảng 35-50 mẫu từ webcam

# 3 Chế độ bảo vệ thu thập dữ liệu
GUARD_MODES = [
    ("STANDARD", 65.0, "Thu chuan (Sang dep)"),
    ("HARD_ENV", 45.0, "Moi truong xau (Thieu sang/Den vang)"),
    ("OFF",       0.0, "Tat bao ve (Luu tu do 100%)")
]


# =============================================================================
# 1. THƯỚC ĐO CHẤT LƯỢNG MẪU DIP (SAMPLE QUALITY ASSESSOR)
# =============================================================================

class LeanQualityAssessor:
    """
    Đánh giá định lượng chất lượng mẫu thu thập theo Kiến trúc Tinh gọn:
    - Độ nét cạnh Canny & Sobel: 45%
    - Ổn định chuyển động (MotionTracker): 35%
    - An toàn mép viền camera (Margin Safety): 20%
    """
    @classmethod
    def evaluate(
        cls,
        tensor: Optional[np.ndarray],
        touching_border: bool,
        motion_tracker: MotionTracker,
        threshold: float = 65.0
    ) -> Tuple[float, bool, List[str], Tuple[int, int, int]]:
        if tensor is None:
            return 0.0, False, ["KHONG CO BAN TAY"], (0, 0, 255)

        warnings: List[str] = []

        # 1. Độ nét cạnh Canny & Sobel (45%)
        canny_ch = tensor[:, :, 1]
        sobel_ch = tensor[:, :, 2]
        canny_pixels = int(np.count_nonzero(canny_ch > 128))
        sobel_mean = float(np.mean(sobel_ch))

        if 350 <= canny_pixels <= 2200:
            s_edge = 100.0
        elif canny_pixels < 350:
            s_edge = max(0.0, (canny_pixels / 350.0) * 100.0)
            if canny_pixels < 180:
                warnings.append("BIEN MO / CHOI SANG")
        else:
            s_edge = max(0.0, 100.0 - ((canny_pixels - 2200) / 1000.0) * 100.0)
            if canny_pixels > 2600:
                warnings.append("NHIEU HAT QUA MUC")

        if sobel_mean < 7.0:
            s_edge *= 0.6
            if "BIEN MO / CHOI SANG" not in warnings:
                warnings.append("ANH PHANG / THIEU TUONG PHAN")

        # 2. Ổn định chuyển động (35%)
        traj = list(motion_tracker.history)
        if len(traj) >= 2:
            dx = traj[-1][0] - traj[-2][0]
            dy = traj[-1][1] - traj[-2][1]
            velocity = math.hypot(dx, dy)
        else:
            velocity = 0.0

        if velocity <= 5.0:
            s_motion = 100.0
        elif velocity <= 16.0:
            s_motion = max(0.0, 100.0 - ((velocity - 5.0) / 11.0) * 60.0)
        else:
            s_motion = max(0.0, 40.0 - ((velocity - 16.0) / 10.0) * 40.0)
            warnings.append("TAY DANG CHUYEN DONG")

        # 3. An toàn mép viền camera (20%)
        if touching_border:
            s_margin = 0.0
            warnings.append("TAY CHAM MEP WEBCAM")
        else:
            s_margin = 100.0

        total_score = (
            0.45 * s_edge +
            0.35 * s_motion +
            0.20 * s_margin
        )
        total_score = round(max(0.0, min(100.0, total_score)), 1)
        is_acceptable = (total_score >= threshold)

        if total_score >= 70.0:
            color = (0, 255, 0)
        elif is_acceptable:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        return total_score, is_acceptable, warnings, color


# =============================================================================
# 2. CHƯƠNG TRÌNH THU THẬP MẪU CHÍNH (MAIN COLLECTOR RUNNER)
# =============================================================================

def run_webcam_collector(camera_idx: int = 1):
    print("=" * 72)
    print("  BỘ THU THẬP MẪU WEBCAM (LEAN CONSOLIDATED PIPELINE v4)")
    print(f"  Camera Index: {camera_idx}")
    print("=" * 72)

    # 1. Tải Hand Detector
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

    # 2. Khởi tạo Builder & Tracker
    builder = DIPDatasetBuilder(output_dir="./vsl_dataset")
    motion_tracker = MotionTracker(history_len=16)

    # 3. Mở Webcam
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở webcam (Camera index: {camera_idx})!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    guard_idx = 0
    flip_cam = False
    prev_time = time.time()
    fps = 0.0

    last_save_time = 0.0
    save_cooldown = 0.15  # 150ms giữa 2 lần lưu khi giữ phím

    flash_save_text = ""
    flash_save_color = (0, 255, 0)
    flash_expire_time = 0.0

    print("\n👉 Hướng dẫn thu mẫu:")
    print("  - Đưa tay vào khung hình: Hộp thông minh sẽ tự động cắt chuẩn khớp cổ tay.")
    print("  - Giữ phím [1-5] để thu thập mẫu tương ứng:")
    print("      [1]: Chu A | [2]: Chu B | [3]: Chu C | [4]: Chu L | [5]: Chu V")
    print("  - Phím [g]: Đổi chế độ bảo vệ (STANDARD -> HARD_ENV -> OFF)")
    print("  - Phím [f]: Lật gương camera")
    print("  - Phím [p]: Cài đặt Camera")
    print("  - Phím [q]: Hoàn tất và thoát\n")

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

        # Giai đoạn 1: Dò tìm bàn tay
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        det_result = detector.detect(mp_image)

        hand_found = False
        current_tensor = None
        canvas_vis = np.zeros((128, 128, 3), dtype=np.uint8)
        score = 0.0
        is_acceptable = False
        warnings = []
        quality_color = (0, 0, 255)
        touching_border = False
        is_flipped = False
        x1, y1, x2, y2 = 0, 0, 0, 0

        guard_name, guard_thresh, guard_desc = GUARD_MODES[guard_idx]

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
                current_tensor, canvas_128, rot_angle = process_lean_dip_tensor(hand_crop, target_size=(128, 128))
                canvas_vis = canvas_128

                # Cập nhật Motion Tracker
                wrist_cx = int(landmarks[0].x * w_f)
                wrist_cy = int(landmarks[0].y * h_f)
                motion_tracker.update(wrist_cx, wrist_cy)
                motion_tracker.draw_trail(display_frame)

                # Đánh giá chất lượng mẫu
                score, is_acceptable, warnings, quality_color = LeanQualityAssessor.evaluate(
                    tensor=current_tensor,
                    touching_border=touching_border,
                    motion_tracker=motion_tracker,
                    threshold=guard_thresh
                )

                # Hiển thị 3 kênh Tensor
                ch0 = cv2.cvtColor(current_tensor[:, :, 0], cv2.COLOR_GRAY2BGR)
                ch1 = cv2.cvtColor(current_tensor[:, :, 1], cv2.COLOR_GRAY2BGR)
                ch2 = cv2.cvtColor(current_tensor[:, :, 2], cv2.COLOR_GRAY2BGR)
                vis_3ch = np.hstack([ch0, ch1, ch2])
                cv2.putText(vis_3ch, "CLAHE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(vis_3ch, "AUTOCANNY", (138, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(vis_3ch, "SOBEL", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("2. Tensor 3 Kenh DIP (128x128x3)", vis_3ch)
                cv2.imshow("3. Ban Tay Chuan Hoa (Moments 128x128)", canvas_vis)

            # Vẽ Dynamic BBox
            thickness = 3 if is_acceptable else 2
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), quality_color, thickness)
            hand_str = f"Tay: {handedness_label}"
            cv2.putText(display_frame, f"{hand_str}: {score:.0f}%", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)

        else:
            motion_tracker.reset()

        # =====================================================================
        # HIỂN THỊ GIAO DIỆN BẢNG TIẾN ĐỘ THU THẬP
        # =====================================================================
        # 1. Bảng tiến độ từng chữ cái ở góc trái
        cv2.rectangle(display_frame, (10, 10), (280, 175), (20, 20, 20), -1)
        cv2.rectangle(display_frame, (10, 10), (280, 175), (80, 80, 80), 1)
        cv2.putText(display_frame, "TIEN DO THU THAP (v4)", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        y_p = 58
        for k, (c_name, _) in GESTURE_CONFIG.items():
            count = builder.sample_counts.get(c_name, 0)
            col = (0, 255, 0) if count >= TARGET_SAMPLES else (0, 255, 255) if count > 0 else (160, 160, 160)
            bar_w = int(min(1.0, count / float(TARGET_SAMPLES)) * 90)
            cv2.putText(display_frame, f"[{k}] {c_name}: {count:2d}/{TARGET_SAMPLES}", (20, y_p),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
            cv2.rectangle(display_frame, (135, y_p - 11), (135 + bar_w, y_p - 2), col, -1)
            cv2.rectangle(display_frame, (135, y_p - 11), (135 + 90, y_p - 2), (100, 100, 100), 1)
            y_p += 22

        # 2. Thanh đo chất lượng mẫu
        cv2.rectangle(display_frame, (300, 10), (w_f - 10, 50), (20, 20, 20), -1)
        cv2.putText(display_frame, f"Chat luong: {score:.0f}% [{guard_name}]", (310, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, quality_color, 1)
        q_bar = int((score / 100.0) * 110)
        cv2.rectangle(display_frame, (w_f - 130, 20), (w_f - 130 + q_bar, 34), quality_color, -1)
        cv2.rectangle(display_frame, (w_f - 130, 20), (w_f - 20, 34), (100, 100, 100), 1)

        # 3. Thông báo lưu flash
        curr_time = time.time()
        if curr_time < flash_expire_time:
            cv2.putText(display_frame, flash_save_text, (20, h_f - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, flash_save_color, 2)

        # 4. Cảnh báo chất lượng nếu có
        if warnings and hand_found:
            cv2.putText(display_frame, f"! {warnings[0]}", (310, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # 5. Chân trang hướng dẫn
        cv2.putText(display_frame, f"Giu [1-5]: Luu | [g]: Guard ({guard_name}) | [f]: Flip | [p]: Cam | [q]: Xong",
                    (10, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow("1. Webcam Thu Thap Du Lieu (LEAN v4)", display_frame)

        # -------------------------------------------------------------
        # BẮT PHÍM ĐỂ LƯU MẪU
        # -------------------------------------------------------------
        key_code = cv2.waitKey(1) & 0xFF
        key_char = chr(key_code) if key_code < 128 else ""

        if key_char in GESTURE_CONFIG:
            target_class = GESTURE_CONFIG[key_char][0]
            if current_tensor is None:
                flash_save_text = "[CHUA THAY BAN TAY HOAC CHUA CO TENSOR]"
                flash_save_color = (0, 0, 255)
                flash_expire_time = curr_time + 0.8
            elif not is_acceptable:
                warn_msg = warnings[0] if warnings else "CHAT LUONG CHUA DAT"
                flash_save_text = f"[CHAN LUU ({guard_name})]: {warn_msg} ({score:.0f}%)"
                flash_save_color = (0, 0, 255)
                flash_expire_time = curr_time + 0.6
            else:
                if curr_time - last_save_time >= save_cooldown:
                    rel_path = builder.save_sample(current_tensor, target_class, prefix="webcam")
                    last_save_time = curr_time
                    cnt = builder.sample_counts[target_class]
                    flash_save_text = f"✓ DA GHI MAU #{cnt:02d} CHU [{target_class}] ({score:.0f}%)"
                    flash_save_color = (0, 255, 0)
                    flash_expire_time = curr_time + 0.5
                    print(f"✓ Ghi mau #{cnt:02d} cho chu [{target_class}] ({score:.0f}%) -> {rel_path}")

        elif key_char in ['g', 'G']:
            guard_idx = (guard_idx + 1) % len(GUARD_MODES)
            g_name, g_thresh, g_desc = GUARD_MODES[guard_idx]
            flash_save_text = f"GUARD: [{g_name}] ({g_thresh:.0f}%) - {g_desc}"
            flash_save_color = (0, 255, 255)
            flash_expire_time = curr_time + 1.2
            print(f"\n-> CHUYỂN CHẾ ĐỘ BẢO VỆ: [{g_name}] (Ngưỡng: {g_thresh}%) — {g_desc}")

        elif key_char in ['f', 'F']:
            flip_cam = not flip_cam
            motion_tracker.reset()
            print(f"-> Flip camera: {'BẬT (Gương)' if flip_cam else 'TẮT'}")

        elif key_char in ['p', 'P']:
            print("\n-> Đang mở bảng Cài đặt Phần cứng Camera...")
            cap.set(cv2.CAP_PROP_SETTINGS, 1)

        elif key_char in ['s', 'S']:
            print("\n[LƯU] Đang lưu cấu hình dữ liệu vào metadata.json...")
            builder.save_metadata()
            print("✓ Đã lưu thành công!")

        elif key_char in ['q', 'Q']:
            print("\n[HOÀN TẤT] Đang hoàn thiện dữ liệu webcam...")
            if len(builder.samples) > 0:
                builder.save_metadata()
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\n✓ Đã kết thúc phiên thu thập dữ liệu webcam.")


if __name__ == "__main__":
    cam_id = 1
    if len(sys.argv) > 1:
        try:
            cam_id = int(sys.argv[1])
        except ValueError:
            pass
    run_webcam_collector(camera_idx=cam_id)
