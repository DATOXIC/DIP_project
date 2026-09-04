"""
CÔNG CỤ THU THẬP DỮ LIỆU TỪ WEBCAM (NGUỒN 1) - PHIÊN BẢN v3 (CIELab + MotionTracker)
=====================================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Nâng cấp v3:
  1. Phân đoạn CIELab (LabAdaptiveHandSegmenter): Kháng biến thiên ánh sáng vượt trội.
  2. Tích hợp MotionTracker (deque maxlen=16): Vẽ vệt sao băng chuyển động và 
     giám sát trạng thái ổn định/chuyển động của bàn tay khi thu mẫu.
  3. Đồng bộ chuẩn trả về 3 giá trị của normalize_hand_orientation (nhận centroid).

Tập 5 ký hiệu ASL:
  [1] Chữ 'A' - Nắm tay, ngón cái áp sát bên cạnh
  [2] Chữ 'B' - Bàn tay phẳng 4 ngón, ngón cái gập
  [3] Chữ 'C' - Bàn tay khum hình chữ C
  [4] Chữ 'L' - Ngón cái và ngón trỏ vuông góc 90 độ
  [5] Chữ 'V' - Xòe 2 ngón tay hình chữ V
"""

import os
import sys
import time
from typing import Dict, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

from improved_dip_pipeline import (
    LabAdaptiveHandSegmenter,   # Phân đoạn CIELab v3 cao cấp
    AdaptiveHandSegmenter,      # Giữ lại dự phòng nếu cần HSV
    MotionTracker,              # Bộ đệm trượt 16 khung hình theo dõi quỹ đạo
    normalize_hand_orientation,
    extract_multi_channel_features
)
from dataset_builder import DIPDatasetBuilder


# Cấu hình 5 ký hiệu chuẩn ASL
GESTURE_CONFIG: Dict[str, Tuple[str, str]] = {
    '1': ('A', 'Chu A - Nam tay, ngon cai ap sat'),
    '2': ('B', 'Chu B - 4 ngon thang, ngon cai gap'),
    '3': ('C', 'Chu C - Ban tay khum hinh chu C'),
    '4': ('L', 'Chu L - Ngon cai va ngon tro vuong goc 90 do'),
    '5': ('V', 'Chu V - Xoe 2 ngon tay hinh chu V'),
}

TARGET_SAMPLES = 35  # Mục tiêu mỗi chữ cái thu thập khoảng 35 mẫu từ webcam


def run_webcam_collector(output_dir: str = "./vsl_dataset", use_cielab: bool = True) -> None:
    engine_name = "CIELab (a*, b*)" if use_cielab else "HSV"
    print("=" * 72)
    print(f"KHỞI ĐỘNG CÔNG CỤ THU THẬP DỮ LIỆU WEBCAM (DIP ENGINE v3: {engine_name})")
    print("=" * 72)

    builder = DIPDatasetBuilder(output_dir=output_dir)

    for key, (class_name, desc) in GESTURE_CONFIG.items():
        builder.register_class(class_name, description=desc)

    # Khởi tạo bộ phân đoạn: Mặc định dùng LabAdaptiveHandSegmenter (v3)
    if use_cielab:
        segmenter = LabAdaptiveHandSegmenter(
            calibration_roi=(80, 420, 320, 620),
            max_calibration_frames=30
        )
    else:
        segmenter = AdaptiveHandSegmenter(
            calibration_roi=(80, 420, 320, 620),
            max_calibration_frames=30
        )

    # Khởi tạo bộ theo dõi quỹ đạo động deque(maxlen=16)
    motion_tracker = MotionTracker(history_len=16, movement_threshold=8)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể kết nối với Webcam. Hãy kiểm tra lại camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_save_time = 0.0
    save_cooldown = 0.09    # Cách quãng ~90ms mỗi ảnh để người dùng lắc nhẹ tạo độ đa dạng
    is_calibrating = False  # Cờ điều khiển chỉ học màu khi người dùng nhấn SPACE

    top, bottom, left, right = segmenter.top, segmenter.bottom, segmenter.left, segmenter.right

    print("\n[HƯỚNG DẪN BẮT ĐẦU]:")
    print("1. Đặt lòng bàn tay vào khung, nhấn [SPACE] để bắt đầu hiệu chuẩn màu da.")
    print("2. Khi hiện 'CALIBRATED - READY', tạo dáng bàn tay và NHẤN GIỮ phím [1-5] để ghi mẫu.")
    print("3. Vệt sáng sao băng trên tay thể hiện quỹ đạo từ MotionTracker deque(16).")
    print("4. Nhấn 's' để lưu metadata | 'r': Calib lại | 'q': Hoàn tất và thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Lật ảnh gương
        h_frame, w_frame = frame.shape[:2]

        # -------------------------------------------------------------
        # 1. CALIBRATION HỌC MÀU DA (CHỜ NGƯỜI DÙNG NHẤN SPACE)
        # -------------------------------------------------------------
        if not segmenter.is_calibrated:
            frame_color = (0, 165, 255) if use_cielab else (0, 255, 255)
            if not is_calibrating:
                # Trạng thái chờ: Người dùng căn chỉnh tay vào khung
                cv2.rectangle(frame, (left, top), (right, bottom), frame_color, 2)
                cv2.putText(
                    frame, f"DAT TAY VAO KHUNG [{engine_name}] ROI NHAN [SPACE]", 
                    (left - 130, top - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, frame_color, 2
                )
                cv2.putText(
                    frame, "Nhan [SPACE] de bat dau hoc mau da | [q]: Thoat", (10, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                )
                cv2.imshow("1. Webcam Thu Thap Du Lieu (DIP Engine v3)", frame)
                key_code = cv2.waitKey(1) & 0xFF
                if key_code in [ord(' '), ord('c'), ord('C')]:
                    print(f"\n-> Bắt đầu hiệu chuẩn màu da ({engine_name})...")
                    is_calibrating = True
                elif key_code in [ord('q'), ord('Q')]:
                    break
                continue
            else:
                # Tiến trình 30 khung hình
                segmenter.draw_calibration_progress(frame)
                cv2.imshow("1. Webcam Thu Thap Du Lieu (DIP Engine v3)", frame)
                key_code = cv2.waitKey(1) & 0xFF
                if key_code in [ord('q'), ord('Q')]:
                    break
                if segmenter.is_calibrated:
                    is_calibrating = False
                    print(f"✓ Hiệu chuẩn màu da {engine_name} hoàn tất! Sẵn sàng thu thập dữ liệu.")
                continue

        # -------------------------------------------------------------
        # 2. XỬ LÝ DIP PIPELINE TRÍCH XUẤT TENSOR & QUỸ ĐẠO
        # -------------------------------------------------------------
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{engine_name} READY", 
            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        # Phân đoạn thích nghi (CIELab hoặc HSV)
        mask = segmenter.segment(frame)

        # Chuẩn hóa hình học: nhận 3 giá trị (hand, mask, centroid)
        normalized_hand, _, centroid = normalize_hand_orientation(
            frame, mask, 
            roi_coords=(top, bottom, left, right), 
            target_size=(128, 128),
            min_hand_area=segmenter.min_hand_area
        )

        current_tensor = None
        if normalized_hand is not None:
            current_tensor = extract_multi_channel_features(normalized_hand)

            # Cập nhật MotionTracker với tọa độ centroid
            if centroid is not None:
                motion_tracker.update(centroid[0], centroid[1])

            # Vẽ vệt sao băng chuyển động lên khung hình chính
            motion_tracker.draw_trail(frame, offset_x=left, offset_y=top)

            # Cửa sổ hiển thị trực quan 3 kênh DIP cạnh nhau
            ch0 = current_tensor[:, :, 0]
            ch1 = current_tensor[:, :, 1]
            ch2 = current_tensor[:, :, 2]
            vis_3ch = np.hstack([ch0, ch1, ch2])
            vis_3ch = cv2.cvtColor(vis_3ch, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis_3ch, "CLAHE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(vis_3ch, "AUTOCANNY", (138, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(vis_3ch, "SOBEL", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("2. Tensor 3 Kenh DIP (128x128x3)", vis_3ch)
        else:
            motion_tracker.reset()

        # Hiển thị trạng thái cử động trên màn hình
        motion_dir = motion_tracker.get_gesture_direction()
        is_moving = motion_tracker.is_moving()
        motion_color = (0, 165, 255) if is_moving else (0, 255, 0)
        motion_status = f"MOTION: {motion_dir} ({'MOVING' if is_moving else 'STABLE'})"
        cv2.putText(frame, motion_status, (left, bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)

        # -------------------------------------------------------------
        # 3. BẢNG THEO DÕI SỐ LƯỢNG MẪU ĐÃ THU
        # -------------------------------------------------------------
        overlay_y = 30
        cv2.putText(frame, "--- TIEN DO THU THAP WEBCAM (v3) ---", (10, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        for k, (c_name, _) in GESTURE_CONFIG.items():
            overlay_y += 25
            count = builder.sample_counts.get(c_name, 0)
            color = (0, 255, 0) if count >= TARGET_SAMPLES else (0, 255, 255)
            status_text = f"Phim [{k}] Chu {c_name:<3}: {count}/{TARGET_SAMPLES}"
            cv2.putText(frame, status_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, "Giu phim [1-5] de ghi | 's': Luu data | 'r': Calib lai | 'q': Xong",
                    (10, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("1. Webcam Thu Thap Du Lieu (DIP Engine v3)", frame)
        cv2.imshow("3. Mat Na Nhi Phan Skin Mask", mask)

        # -------------------------------------------------------------
        # 4. BẮT SỰ KIỆN PHÍM BẤM ĐỂ GHI MẪU
        # -------------------------------------------------------------
        key_code = cv2.waitKey(1) & 0xFF
        key_char = chr(key_code) if key_code < 128 else ""
        current_time = time.time()

        if key_char in GESTURE_CONFIG:
            target_class = GESTURE_CONFIG[key_char][0]
            if current_tensor is None:
                cv2.putText(frame, "CHUA THAY BAN TAY!", (left, bottom + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if current_time - last_save_time >= save_cooldown:
                    rel_path = builder.save_sample(current_tensor, target_class)
                    last_save_time = current_time
                    cnt = builder.sample_counts[target_class]
                    print(f"-> Ghi mau #{cnt:02d} cho chu [{target_class}] -> {rel_path}")

        elif key_char in ['s', 'S']:
            print("\n[LƯU] Đang lưu cấu hình dữ liệu vào metadata.json...")
            builder.save_metadata()
            print("✓ Đã lưu thành công!")

        elif key_char in ['r', 'R']:
            print("\n[CALIB] Đang hiệu chuẩn lại màu da...")
            segmenter.reset_calibration()
            motion_tracker.reset()
            is_calibrating = False

        elif key_char in ['q', 'Q']:
            print("\n[HOÀN TẤT] Đang hoàn thiện dữ liệu webcam...")
            if len(builder.samples) > 0:
                builder.save_metadata()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Đã kết thúc phiên thu thập dữ liệu webcam.")


if __name__ == "__main__":
    # Mặc định sử dụng CIELab (use_cielab=True). Đặt False nếu muốn dùng HSV.
    run_webcam_collector(use_cielab=True)
