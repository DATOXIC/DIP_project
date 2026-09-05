"""
CHƯƠNG TRÌNH NHẬN DIỆN THỦ NGỮ THỜI GIAN THỰC (REAL-TIME VSL INFERENCE)
=======================================================================
Chức năng:
  - Bật Webcam và nạp mô hình Two-Stream CNN đã huấn luyện (best_model.pth).
  - Áp dụng đầy đủ Pipeline DIP v3 (CIELab + Moments + 3-channel Tensor).
  - Phân loại cử chỉ thời gian thực, hiển thị:
      + Tên chữ cái dự đoán (A, B, C, L, V) với kích thước lớn.
      + Độ tự tin % (Confidence Score) và thanh xác suất.
      + Cửa sổ trực quan hóa 3 kênh DIP: [CLAHE | AutoCanny | Sobel].
      + Cửa sổ phân đoạn màu da CIELab.
      + Trạng thái chuyển động từ MotionTracker (deque maxlen=16).

Phím điều khiển:
  - 'r' : Hiệu chuẩn lại màu da (Recalibrate)
  - 'q' : Thoát chương trình
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Import kiến trúc Two-Stream CNN từ train.py
from train import TwoStreamHandCNN

# Import toàn bộ Pipeline DIP v3
from improved_dip_pipeline import (
    LabAdaptiveHandSegmenter,
    MotionTracker,
    normalize_hand_orientation,
    extract_multi_channel_features
)


def load_trained_model(model_path: str, device: torch.device):
    """Nạp trọng số mô hình best_model.pth và metadata nhãn."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file model tại: {model_path}\n"
            f"Hãy đảm bảo bạn đã train xong và đặt file best_model.pth vào đúng thư mục!"
        )

    checkpoint = torch.load(p, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 5)
    id_to_label = checkpoint.get("id_to_label", {0: 'A', 1: 'B', 2: 'C', 3: 'L', 4: 'V'})

    # Chuyển đổi key của id_to_label về kiểu int nếu bị lưu dạng chuỗi
    id_to_label = {int(k): str(v) for k, v in id_to_label.items()}

    model = TwoStreamHandCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"-> Đã nạp model thành công (Độ chính xác lúc train: {val_acc:.1f}%)")
    print(f"   Các lớp nhận diện: {list(id_to_label.values())}")

    return model, id_to_label


def run_realtime_prediction(
    model_path: str = "./vsl_dataset/models/best_model.pth",
    calibration_roi: tuple = (80, 420, 320, 620),
    confidence_threshold: float = 0.60
):
    print("=" * 72)
    print("  HỆ THỐNG NHẬN DIỆN THỦ NGỮ VSL THỜI GIAN THỰC (TWO-STREAM CNN + DIP)")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị chạy: {device}")

    try:
        model, id_to_label = load_trained_model(model_path, device)
    except FileNotFoundError as e:
        print(f"\n[LỖI] {e}")
        return

    # Khởi tạo Pipeline DIP v3
    segmenter = LabAdaptiveHandSegmenter(calibration_roi=calibration_roi, max_calibration_frames=30)
    tracker = MotionTracker(history_len=16)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam!")
        return

    # Cài đặt độ phân giải webcam chuẩn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    top, bottom, left, right = calibration_roi
    prev_time = time.time()
    fps = 0.0
    is_calibrating = False  # Trạng thái chờ bấm SPACE hay đang đếm 30 frame
    flip_cam = False        # Mặc định KHÔNG lật gương để khớp 100% với hướng của tập Kaggle!

    print("\n👉 Hướng dẫn:")
    print("  1. Đặt bàn tay vào ô màu cam, giữ yên rồi NHẤN PHÍM [SPACE] để bắt đầu hiệu chuẩn.")
    print("  2. Chờ thanh tiến trình chạy 100% -> Hiện chữ 'READY' -> Bắt đầu nhận diện.")
    print("  3. Nhấn 'f' để BẬT/TẮT lật gương (Mirror Flip) camera.")
    print("  4. Nhấn 'r' để hiệu chuẩn lại nếu đổi góc ngồi/đổi đèn.")
    print("  5. Nhấn 'q' để kết thúc.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip_cam:
            frame = cv2.flip(frame, 1)  # Lật gương

        display_frame = frame.copy()

        # Tính FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
        prev_time = curr_time

        # -------------------------------------------------------------
        # GIAI ĐOẠN 1: HIỆU CHUẨN MÀU DA (CALIBRATION - KÍCH HOẠT BẰNG SPACE)
        # -------------------------------------------------------------
        if not segmenter.is_calibrated:
            h_r, w_r = bottom - top, right - left
            # Vẽ khung ROI màu cam
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 165, 255), 2)
            cv2.rectangle(display_frame,
                          (left + int(w_r * 0.15), top  + int(h_r * 0.15)),
                          (right - int(w_r * 0.15), bottom - int(h_r * 0.15)),
                          (0, 140, 255), 1)

            if not is_calibrating:
                # Đang chờ người dùng đưa tay vào ổn định và bấm SPACE
                cv2.putText(display_frame, "Dat tay vao o roi NHAN [SPACE]",
                            (left - 40, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(display_frame, "Giu yen tay trong o cam",
                            (left + 25, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                # Đang kích hoạt chạy 30 frame
                segmenter.draw_calibration_progress(display_frame)
                if segmenter.is_calibrated:
                    is_calibrating = False
                    print("-> Hiệu chuẩn thành công! Bắt đầu nhận diện.")

            cv2.putText(display_frame, f"[SPACE]: Bat dau Calib | [f]: Lat guong ({'BAT' if flip_cam else 'TAT'}) | [q]: Thoat", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("VSL Recognition - Webcam", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                flip_cam = not flip_cam
                print(f"-> Lật gương camera: {flip_cam}")
            elif key == 32:  # Phím SPACE
                if not is_calibrating:
                    print("-> Bắt đầu hiệu chuẩn màu da qua 30 khung hình...")
                    is_calibrating = True
            continue

        # -------------------------------------------------------------
        # GIAI ĐOẠN 2: THỰC THI PIPELINE DIP v3
        # -------------------------------------------------------------
        # 1. Vẽ khung ROI
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 2. Phân đoạn bàn tay bằng CIELab
        mask = segmenter.segment(frame)
        cv2.imshow("1. Mat Na Nhi Phan (Skin Mask)", mask)

        # 3. Chuẩn hóa hình học bằng Image Moments (xoay về 0 độ, pad 128x128)
        norm_hand, _, centroid = normalize_hand_orientation(
            frame, mask,
            roi_coords=calibration_roi,
            target_size=(128, 128)
        )

        pred_label = "..."
        confidence = 0.0

        if norm_hand is not None:
            # 4. Trích xuất Tensor 3 kênh (CLAHE, AutoCanny, Sobel)
            tensor_np = extract_multi_channel_features(norm_hand)

            # Hiển thị trực quan 3 kênh DIP
            vis_ch0 = tensor_np[:, :, 0]
            vis_ch1 = tensor_np[:, :, 1]
            vis_ch2 = tensor_np[:, :, 2]
            vis_tensor = np.hstack([vis_ch0, vis_ch1, vis_ch2])
            vis_tensor = cv2.cvtColor(vis_tensor, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis_tensor, "CLAHE | CANNY | SOBEL", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow("2. Tensor 3 Kenh DIP (128x128x3)", vis_tensor)

            # Cập nhật Motion Tracker
            if centroid is not None:
                tracker.update(centroid[0], centroid[1])
                tracker.draw_trail(display_frame, offset_x=left, offset_y=top)

            # -------------------------------------------------------------
            # GIAI ĐOẠN 3: DỰ ĐOÁN BẰNG TWO-STREAM CNN
            # -------------------------------------------------------------
            # Chuẩn bị Tensor cho PyTorch: (128,128,3) -> (1, 3, 128, 128) float32 [0..1]
            tensor_pt = torch.from_numpy(tensor_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_pt)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

            if confidence >= confidence_threshold:
                pred_label = id_to_label.get(pred_idx, f"Lop {pred_idx}")
            else:
                pred_label = "Khong ro"

            # Vẽ bảng xác suất các chữ cái ở góc trái
            y_offset = 120
            for i, prob in enumerate(probs):
                lbl_name = id_to_label.get(i, str(i))
                bar_len = int(prob * 120)
                color = (0, 255, 0) if i == pred_idx and confidence >= confidence_threshold else (180, 180, 180)
                cv2.putText(display_frame, f"{lbl_name}: {prob*100:4.1f}%", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                cv2.rectangle(display_frame, (110, y_offset - 12), (110 + bar_len, y_offset - 2), color, -1)
                y_offset += 24

        # -------------------------------------------------------------
        # HIỂN THỊ KẾT QUẢ DỰ ĐOÁN LÊN MÀN HÌNH CHÍNH
        # -------------------------------------------------------------
        # Khung kết quả nổi bật ở góc trên bên phải
        cv2.rectangle(display_frame, (left, top - 65), (right, top - 10), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (left, top - 65), (right, top - 10), (0, 255, 0), 2)

        res_text = f"CHU CAI: {pred_label} ({confidence*100:.0f}%)"
        text_color = (0, 255, 0) if pred_label != "..." and pred_label != "Khong ro" else (0, 165, 255)
        cv2.putText(display_frame, res_text, (left + 10, top - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

        # Thông tin FPS và trạng thái chuyển động
        motion_state = "MOVING" if tracker.is_moving() else "STABLE"
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Tay: {motion_state}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(display_frame, f"[r]: Calib lai | [f]: Lat guong ({'BAT' if flip_cam else 'TAT'}) | [q]: Thoat", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("VSL Recognition - Webcam", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            flip_cam = not flip_cam
            print(f"-> Lật gương camera: {flip_cam}")
        elif key == ord('r'):
            print("-> Tiến hành hiệu chuẩn lại...")
            segmenter.reset_calibration()
            is_calibrating = False

    cap.release()
    cv2.destroyAllWindows()
    print("-> Đã đóng chương trình.")


if __name__ == "__main__":
    run_realtime_prediction()
