"""
THỬ NGHIỆM HƯỚNG 2.3: MÔ HÌNH LAI THÔNG MINH (HYBRID DETECTOR + DIP ENGINE v4)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Kiến trúc Coarse-to-Fine (Từ thô đến tinh):
  1. Giai đoạn 1 (Coarse - Trợ lý định vị):
     - Mạng nơ-ron siêu nhẹ (BlazeHand ~7MB) tìm Bounding Box động ôm khít bàn tay.
     - Loại bỏ 90% phông nền phức tạp (tủ gỗ, tường, quần áo) ngay từ đầu.
  2. Giai đoạn 2 (Fine - Thuật toán Xử Lý Ảnh Số thuần túy):
     - Áp dụng trên vùng Crop nhỏ (tiết kiệm 80% tài nguyên CPU).
     - Phân đoạn màu da CIELab + YCrCb + FillPoly lấp bóng đổ.
     - Chuẩn hóa hình học Image Moments xoay trục chính về 0 độ.
     - Trích xuất Tensor 3 kênh DIP: [CLAHE | AutoCanny | Sobel].
  3. Giai đoạn 3 (Phân loại):
     - TwoStreamHandCNN (best_model_v2.pth) + Bộ lọc đa số SlidingWindowVoter.

Phím điều khiển:
  [f] : Lật gương (Flip camera)
  [p] : Bảng cài đặt phần cứng Camera (khóa Auto-Exposure / White Balance)
  [q] : Thoát chương trình
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import os
import sys
import time
from collections import deque, Counter
from typing import Tuple, Optional, List

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
    auto_canny,
    apply_gray_world_white_balance,
    normalize_hand_orientation,
    extract_multi_channel_features
)


# =============================================================================
# 1. BỘ LỌC ĐA SỐ (SLIDING WINDOW VOTER)
# =============================================================================

class SlidingWindowVoter:
    def __init__(self, window_size: int = 7, min_votes_ratio: float = 0.55):
        self.window_size = window_size
        self.min_votes = int(np.ceil(window_size * min_votes_ratio))
        self.history = deque(maxlen=window_size)
        self.prob_history = deque(maxlen=window_size)

    def update(self, pred_label: str, probs: np.ndarray) -> Tuple[Optional[str], float, int]:
        self.history.append(pred_label)
        self.prob_history.append(probs)

        if len(self.history) < 3:
            return None, 0.0, 0

        counts = Counter(self.history)
        top_label, top_votes = counts.most_common(1)[0]

        if top_votes >= self.min_votes:
            avg_probs = np.mean(self.prob_history, axis=0)
            return top_label, float(np.max(avg_probs)), top_votes
        return None, 0.0, top_votes

    def reset(self):
        self.history.clear()
        self.prob_history.clear()


# =============================================================================
# 2. PHÂN ĐOẠN DIP TRÊN VÙNG CROP BÀN TAY (CROP DIP SEGMENTER)
# =============================================================================

def segment_hand_crop_dip(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Thuật toán DIP phân đoạn bàn tay trên vùng ảnh cắt nhỏ (Tight Crop):
    - Cân bằng trắng Gray-World.
    - Không gian màu kép: CIELab (a*, b*) kết hợp YCrCb (Cr, Cb).
    - Phép đóng/mở hình thái học (Morphological Closing & Opening).
    - Tự động lấp đầy lỗ thủng bóng đổ (cv2.fillPoly).
    """
    wb = apply_gray_world_white_balance(crop_bgr)
    
    # 1. Không gian CIELab
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2Lab)
    mask_lab = cv2.inRange(lab, (25, 128, 125), (245, 185, 185))

    # 2. Không gian YCrCb
    ycrcb = cv2.cvtColor(wb, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 175, 127))

    # Giao thoa 2 không gian màu để triệt tiêu nhiễu ngoài da
    skin_raw = cv2.bitwise_and(mask_lab, mask_ycrcb)

    # 3. Phép biến đổi hình thái học DIP
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(skin_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Lấy contour lớn nhất (loại bỏ đốm nhiễu li ti)
    cnts, hier = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask_out = np.zeros_like(clean)
    if cnts and hier is not None:
        max_idx = -1
        max_area = 0
        for i, c in enumerate(cnts):
            if hier[0][i][3] == -1:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_idx = i

        if max_idx >= 0:
            cv2.drawContours(mask_out, cnts, max_idx, 255, -1)
            for i in range(len(cnts)):
                if hier[0][i][3] == max_idx:
                    cv2.fillPoly(mask_out, [cnts[i]], 255)

    return mask_out


# =============================================================================
# 3. CHƯƠNG TRÌNH THỬ NGHIỆM CHÍNH (MAIN HYBRID RUNNER)
# =============================================================================

def run_hybrid_tester():
    print("=" * 70)
    print("  HỆ THỐNG LAI THÔNG MINH: DETECTOR + DIP ENGINE v4 (HYBRID PIPELINE)")
    print("=" * 70)

    # 1. Tải mô hình Hand Detector (BlazeHand TFLite via MediaPipe Tasks)
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
    print("✓ Khởi tạo Hand Detector siêu nhẹ thành công!")

    # 2. Tải mô hình Phân loại Two-Stream CNN
    cnn_model_path = 'vsl_dataset/models/best_model_v2.pth'
    device = torch.device('cpu')
    if os.path.exists(cnn_model_path):
        checkpoint = torch.load(cnn_model_path, map_location=device)
        classes = [checkpoint['id_to_label'][i] for i in range(checkpoint['num_classes'])]
        model = TwoStreamHandCNN(num_classes=len(classes))
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        val_acc = checkpoint.get('val_acc', 0.0)
        print(f"✓ Đã nạp Two-Stream CNN: {cnn_model_path} (Val Acc: {val_acc:.2f}%)")
        print(f"  Các lớp nhận diện: {classes}")
    else:
        print(f"[!] Không tìm thấy {cnn_model_path}, chỉ chạy thử nghiệm DIP.")
        model = None
        classes = ['A', 'B', 'C', 'L', 'V']

    voter = SlidingWindowVoter(window_size=7, min_votes_ratio=0.55)

    # 3. Mở Webcam với backend DirectShow
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[HƯỚNG DẪN]")
    print("  [f]: Lật gương camera (Flip)")
    print("  [p]: Cài đặt phơi sáng Camera (Auto Exposure)")
    print("  [q]: Thoát\n")

    flip_cam = False
    prev_time = time.time()
    fps = 0.0

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

        # Chuẩn bị ảnh cho Detector (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Giai đoạn 1: Dò tìm Bounding Box bàn tay
        det_result = detector.detect(mp_image)

        hand_detected = False
        tensor_vis = np.zeros((128, 384, 3), dtype=np.uint8)
        mask_vis = np.zeros((160, 160), dtype=np.uint8)
        pred_label = None
        pred_conf = 0.0
        votes = 0
        probs_all = None

        if det_result.hand_landmarks and len(det_result.hand_landmarks) > 0:
            hand_detected = True
            landmarks = det_result.hand_landmarks[0]
            xs = [lm.x * w_f for lm in landmarks]
            ys = [lm.y * h_f for lm in landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Mở rộng hộp Bounding Box thêm 22% lề (margin) để không cắt cụt ngón tay / cổ tay
            bw = x_max - x_min
            bh = y_max - y_min
            margin = 0.22
            x1 = max(0, int(x_min - margin * bw))
            y1 = max(0, int(y_min - margin * bh))
            x2 = min(w_f, int(x_max + margin * bw))
            y2 = min(h_f, int(y_max + margin * bh))

            hand_crop = frame[y1:y2, x1:x2]
            ch, cw = hand_crop.shape[:2]

            if ch >= 30 and cw >= 30:
                # Giai đoạn 2: DIP trên vùng Crop (Loại sạch nền tủ gỗ)
                crop_mask = segment_hand_crop_dip(hand_crop)
                mask_vis = cv2.resize(crop_mask, (160, 160), interpolation=cv2.INTER_NEAREST)

                # Chuẩn hóa góc xoay bằng Image Moments (DIP)
                norm_roi, norm_mask, _ = normalize_hand_orientation(
                    hand_crop, crop_mask,
                    roi_coords=(0, ch, 0, cw),
                    target_size=(128, 128),
                    min_hand_area=max(300, int(ch * cw * 0.05))
                )

                if norm_roi is not None:
                    # Trích xuất 3 kênh DIP: [CLAHE | AutoCanny | Sobel]
                    tensor = extract_multi_channel_features(norm_roi)

                    # Hiển thị trực quan 3 kênh DIP
                    ch_clahe = cv2.cvtColor(tensor[:, :, 0], cv2.COLOR_GRAY2BGR)
                    ch_canny = cv2.cvtColor(tensor[:, :, 1], cv2.COLOR_GRAY2BGR)
                    ch_sobel = cv2.cvtColor(tensor[:, :, 2], cv2.COLOR_GRAY2BGR)
                    tensor_vis = np.hstack([ch_clahe, ch_canny, ch_sobel])

                    # Giai đoạn 3: Phân loại CNN
                    if model is not None:
                        t_norm = tensor.astype(np.float32) / 255.0
                        t_torch = torch.from_numpy(t_norm).permute(2, 0, 1).unsqueeze(0).float()
                        with torch.no_grad():
                            outputs = model(t_torch)
                            probs_all = F.softmax(outputs, dim=1).numpy()[0]

                        raw_pred_idx = np.argmax(probs_all)
                        raw_pred_label = classes[raw_pred_idx]

                        pred_label, pred_conf, votes = voter.update(raw_pred_label, probs_all)

            # Vẽ Dynamic Bounding Box trên khung hình chính
            box_color = (0, 255, 0) if pred_label else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label_top = f"Hand ROI: {x2-x1}x{y2-y1}"
            cv2.putText(frame, label_top, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        else:
            voter.reset()

        # =====================================================================
        # VẼ GIAO DIỆN HUD THÔNG MINH
        # =====================================================================
        # 1. Thanh trạng thái trên cùng
        cv2.rectangle(frame, (0, 0), (w_f, 40), (20, 20, 20), -1)
        fps_color = (0, 255, 0) if fps >= 24 else (0, 165, 255)
        cv2.putText(frame, f"FPS: {fps:.1f} (HYBRID 2.3)", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

        status_str = "TAY: DA TIM THAY" if hand_detected else "DUA TAY VAO WEBCAM"
        status_col = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, f"| {status_str}", (220, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        # 2. Bảng kết quả dự đoán
        if pred_label:
            banner_text = f"CHU CAI: {pred_label} ({pred_conf*100:.0f}%) [{votes}/7]"
            cv2.rectangle(frame, (10, 48), (380, 95), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 48), (380, 95), (0, 255, 0), 2)
            cv2.putText(frame, banner_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 3. Biểu đồ xác suất nếu có
        if probs_all is not None:
            bar_start_y = 115
            for i, (cls_name, p) in enumerate(zip(classes, probs_all)):
                y_pos = bar_start_y + i * 22
                bar_len = int(p * 120)
                is_top = (cls_name == pred_label)
                col = (0, 255, 0) if is_top else (180, 180, 180)
                cv2.putText(frame, f"{cls_name}: {p*100:4.1f}%", (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                cv2.rectangle(frame, (85, y_pos - 10), (85 + bar_len, y_pos), col, -1)

        # 4. Hướng dẫn chân trang
        cv2.putText(frame, "[f]: Flip | [p]: Cam Settings | [q]: Thoat",
                    (10, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Hiển thị 3 cửa sổ
        cv2.imshow("1. VSL Hybrid Realtime (Dynamic Hand BBox)", frame)
        cv2.imshow("2. Mat Na Nhi Phan (Dynamic DIP Mask)", mask_vis)

        # Đánh dấu tiêu đề các kênh Tensor
        cv2.putText(tensor_vis, "CLAHE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(tensor_vis, "CANNY", (138, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(tensor_vis, "SOBEL", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("3. Tensor 3 Kenh DIP (CLAHE | Canny | Sobel)", tensor_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            flip_cam = not flip_cam
        elif key == ord('p'):
            cap.set(cv2.CAP_PROP_SETTINGS, 1)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\n✓ Đã kết thúc chương trình thử nghiệm Hướng 2.3.")


if __name__ == "__main__":
    run_hybrid_tester()
