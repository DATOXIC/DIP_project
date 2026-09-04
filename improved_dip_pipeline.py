"""
HỆ THỐNG PIPELINE DIP NÂNG CẤP CHO PHÂN TÍCH THỦ NGỮ (VSL)
============================================================
Sinh viên: Tô Phương Đạt
Môn học: Xử lý ảnh số (Digital Image Processing - DIP)

Kiến trúc chuẩn theo hướng dẫn:
1. Phân đoạn thích nghi: Histogram Backprojection (Hue - Saturation 2D) + Otsu
2. Chuẩn hóa hình học: Central Image Moments (Tính góc xoay trục chính, dựng thẳng bàn tay)
3. Trích xuất đặc trưng đa kênh: Tensor 3 kênh (128, 128, 3) gồm:
   - Kênh 0: Grayscale + CLAHE (Cấu trúc & tương phản)
   - Kênh 1: Canny Edge Detection (Đường biên & hình dạng ngón tay)
   - Kênh 2: Sobel Gradient Magnitude (Độ chi tiết cục bộ & nếp gấp)
"""

import cv2
import numpy as np


class AdaptiveHandSegmenter:
    """
    Phân đoạn bàn tay thích nghi dựa trên mô hình xác suất màu da:
    - Giai đoạn hiệu chuẩn (Calibration): Tự động học Histogram 2D (H, S) của người dùng.
    - Giai đoạn phân đoạn: cv2.calcBackProject -> Otsu Thresholding tự động.
    """

    def __init__(self, calibration_roi=(80, 420, 320, 620), max_calibration_frames=30):
        # Tọa độ vùng hiệu chuẩn (top, bottom, left, right)
        self.top, self.bottom, self.left, self.right = calibration_roi
        self.max_calibration_frames = max_calibration_frames
        self.calibration_count = 0
        self.is_calibrated = False
        self.hist = None

    def reset_calibration(self):
        """Khởi động lại quá trình hiệu chuẩn."""
        self.calibration_count = 0
        self.is_calibrated = False
        self.hist = None

    def calibrate(self, frame):
        """
        Thu thập mẫu màu da trong vùng ROI để tính Histogram 2D trên không gian màu HSV.
        Sử dụng trung bình trượt qua các khung hình để tạo biểu đồ phân bố chuẩn.
        """
        roi = frame[self.top:self.bottom, self.left:self.right]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Tính Histogram 2D trên kênh Hue [0..180] và Saturation [0..256]
        hist_current = cv2.calcHist(
            [hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256]
        )
        cv2.normalize(hist_current, hist_current, 0, 255, cv2.NORM_MINMAX)

        if self.hist is None:
            self.hist = hist_current
        else:
            # Trung bình trượt tích lũy (Càng về sau càng ổn định)
            self.hist = 0.85 * self.hist + 0.15 * hist_current

        self.calibration_count += 1
        if self.calibration_count >= self.max_calibration_frames:
            # Chuẩn hóa lần cuối sau khi hoàn tất calibration
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
            self.is_calibrated = True

    def segment(self, frame):
        """
        Chiếu ngược Histogram lên toàn khung hình -> Lọc xác suất -> Phân ngưỡng tự động bằng Otsu.
        """
        if not self.is_calibrated or self.hist is None:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Chỉ xử lý trong vùng ROI quan tâm để tối ưu tốc độ và tránh bắt nhầm mặt
        roi = frame[self.top:self.bottom, self.left:self.right]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 1. Chiếu ngược Histogram: Mỗi pixel mang giá trị xác suất thuộc màu da
        backproj = cv2.calcBackProject([hsv_roi], [0, 1], self.hist, [0, 180, 0, 256], scale=1)

        # 2. Lọc tích chập làm mịn bản đồ xác suất với phần tử elip
        disc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(backproj, -1, disc_kernel, backproj)

        # 3. Phân ngưỡng nhị phân tự động thích nghi bằng thuật toán Otsu
        _, mask = cv2.threshold(backproj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Phép toán hình thái học tẩy sạch nhiễu và hàn liền khối bàn tay
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask


def normalize_hand_orientation(frame, mask, roi_coords=(80, 420, 320, 620), target_size=(128, 128)):
    """
    Chuẩn hóa hình học bàn tay bằng Image Moments (Central Moments):
    - Tìm tâm trọng khối Centroid (cx, cy)
    - Tính góc trục chính theta = 0.5 * atan2(2*mu11, mu20 - mu02)
    - Xoay bàn tay dựng thẳng đứng 0 độ
    - Cắt và resize giữ nguyên tỉ lệ về kích thước target_size (128, 128)
    """
    top, bottom, left, right = roi_coords
    roi = frame[top:bottom, left:right]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 2000:
        return None, None

    # Tính Image Moments
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return None, None

    # Tọa độ tâm trọng khối (Centroid)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Central moments (Phương sai & hiệp phương sai bậc 2)
    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]

    # Góc trục chính của bàn tay (Principal Axis)
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = np.degrees(angle_rad)

    # Đưa trục chính về phương thẳng đứng (xoay quanh Centroid)
    # Nếu bàn tay nghiêng góc angle_deg, xoay góc (angle_deg - 90) hoặc tương đương
    rot_angle = angle_deg - 90 if angle_deg > 0 else angle_deg + 90

    # Ma trận xoay 2D quanh tâm bàn tay
    rot_matrix = cv2.getRotationMatrix2D((cx, cy), rot_angle, scale=1.0)
    h_roi, w_roi = roi.shape[:2]

    # Xoay cả ảnh BGR và Mặt nạ nhị phân
    rotated_roi = cv2.warpAffine(roi, rot_matrix, (w_roi, h_roi), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(mask, rot_matrix, (w_roi, h_roi), flags=cv2.INTER_NEAREST)

    # Tìm lại Bounding Box trên mặt nạ đã xoay thẳng
    rot_contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(rot_contours) == 0:
        return None, None

    rot_max_c = max(rot_contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(rot_max_c)

    # Thêm lề (padding) 10 pixel an toàn
    pad = 10
    y1 = max(0, ry - pad)
    y2 = min(h_roi, ry + rh + pad)
    x1 = max(0, rx - pad)
    x2 = min(w_roi, rx + rw + pad)

    cropped_hand = rotated_roi[y1:y2, x1:x2]
    if cropped_hand.size == 0 or cropped_hand.shape[0] < 20 or cropped_hand.shape[1] < 20:
        return None, None

    # Kiểm tra tính xuôi ngược: Đầu ngón tay phải hướng lên trên (y nhỏ hơn)
    # Nếu phần nửa trên đặc hơn nửa dưới (ngược đầu), xoay 180 độ
    cropped_mask = rotated_mask[y1:y2, x1:x2]
    mid_y = cropped_mask.shape[0] // 2
    top_area = cv2.countNonZero(cropped_mask[:mid_y, :])
    bot_area = cv2.countNonZero(cropped_mask[mid_y:, :])

    if top_area > bot_area * 1.5:  # Bị lộn ngược đầu cổ tay lên trên
        cropped_hand = cv2.rotate(cropped_hand, cv2.ROTATE_180)
        cropped_mask = cv2.rotate(cropped_mask, cv2.ROTATE_180)

    # Resize giữ nguyên tỉ lệ (Aspect Ratio Padding) về kích thước target_size (128, 128)
    ch_h, ch_w = cropped_hand.shape[:2]
    scale = min(target_size[1] / ch_w, target_size[0] / ch_h)
    new_w = int(ch_w * scale)
    new_h = int(ch_h * scale)

    resized_hand = cv2.resize(cropped_hand, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Tạo khung nền đen vuông target_size và dán bàn tay vào giữa
    normalized_hand = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    start_y = (target_size[0] - new_h) // 2
    start_x = (target_size[1] - new_w) // 2
    normalized_hand[start_y:start_y + new_h, start_x:start_x + new_w] = resized_hand

    return normalized_hand, rotated_mask


def extract_multi_channel_features(normalized_hand):
    """
    Trích xuất đặc trưng đa kênh từ bàn tay đã chuẩn hóa:
    Xuất ra Tensor (128, 128, 3) gồm:
    - Kênh 0: Grayscale + CLAHE (Độ tương phản và cấu trúc bề mặt)
    - Kênh 1: Canny Edges (Đường viền biên sắc nét của các ngón tay)
    - Kênh 2: Sobel Gradient Magnitude (Chi tiết vi mô và các nếp gấp ngón tay)
    """
    if len(normalized_hand.shape) == 3:
        gray = cv2.cvtColor(normalized_hand, cv2.COLOR_BGR2GRAY)
    else:
        gray = normalized_hand.copy()

    # Kênh 0: Grayscale + CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channel0 = clahe.apply(gray)

    # Làm mờ nhẹ để khử nhiễu đạo hàm cho Canny và Sobel
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Kênh 1: Canny Edges
    channel1 = cv2.Canny(blurred, 30, 100)

    # Kênh 2: Sobel Gradient Magnitude
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    channel2 = magnitude_norm.astype(np.uint8)

    # Xếp chồng 3 kênh thành Tensor (128, 128, 3)
    multi_channel_tensor = np.stack([channel0, channel1, channel2], axis=2)
    return multi_channel_tensor


def full_preprocessing_pipeline(frame, segmenter, target_size=(128, 128)):
    """
    Hàm wrapper tích hợp toàn bộ pipeline DIP:
    Nhận: Frame thô từ webcam + Bộ phân đoạn đã hiệu chuẩn
    Trả về: Tensor 3 kênh (128, 128, 3) sẵn sàng nạp thẳng vào Deep Learning (CNN)
    """
    if not segmenter.is_calibrated:
        return None

    # 1. Phân đoạn thích nghi bằng Histogram Backprojection
    mask = segmenter.segment(frame)

    # 2. Chuẩn hóa hình học: Xoay thẳng bằng Image Moments
    normalized_hand, _ = normalize_hand_orientation(
        frame, mask, roi_coords=(segmenter.top, segmenter.bottom, segmenter.left, segmenter.right), target_size=target_size
    )

    if normalized_hand is None:
        return None

    # 3. Trích xuất đặc trưng 3 kênh
    multi_channel_tensor = extract_multi_channel_features(normalized_hand)
    return multi_channel_tensor


# =========================================================================
# VÒNG LẶP CHẠY THỰC NGHIỆM TRỰC QUAN (REAL-TIME PIPELINE DEMO)
# =========================================================================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở Webcam!")
        exit()

    segmenter = AdaptiveHandSegmenter(calibration_roi=(80, 420, 320, 620), max_calibration_frames=30)

    print("=" * 60)
    print("PIPELINE DIP NÂNG CẤP CHO PHÂN TÍCH THỦ NGỮ ĐÃ KHỞI ĐỘNG")
    print("- Đặt bàn tay vào khung vàng trong 30 khung hình để Calibration.")
    print("- Nhấn 'c' hoặc 'r' để Calibrate lại bất kỳ lúc nào.")
    print("- Nhấn 'q' để thoát chương trình.")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lật ảnh gương
        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]

        # Khung ROI hiệu chuẩn
        top, bottom, left, right = segmenter.top, segmenter.bottom, segmenter.left, segmenter.right

        # =====================================================================
        # GIAI ĐOẠN 1: CALIBRATION (TỰ ĐỘNG HỌC MÀU DA)
        # =====================================================================
        if not segmenter.is_calibrated:
            segmenter.calibrate(frame)

            # Vẽ hướng dẫn trên màn hình
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            progress = int((segmenter.calibration_count / segmenter.max_calibration_frames) * 100)

            # Thanh tiến trình
            cv2.putText(
                frame, f"DANG CALIBRATION MAU DA: {progress}%", (left - 50, top - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.rectangle(frame, (left, bottom + 10), (left + int((right - left) * progress / 100), bottom + 25), (0, 255, 0), -1)

            cv2.imshow("1. Webcam Input & Pipeline Status", frame)

        # =====================================================================
        # GIAI ĐOẠN 2: XỬ LÝ DIP TOÀN PHẦN (SEGMENTATION + MOMENTS + 3-CHANNELS)
        # =====================================================================
        else:
            # 1. Phân đoạn
            mask = segmenter.segment(frame)

            # 2. Chuẩn hóa hình học
            normalized_hand, rotated_mask = normalize_hand_orientation(
                frame, mask, roi_coords=(top, bottom, left, right), target_size=(128, 128)
            )

            # Vẽ khung ROI giám sát màu xanh lá (đã calibrate)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "CALIBRATED - READY FOR DL", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 3. Trích xuất đặc trưng đa kênh nếu có bàn tay
            if normalized_hand is not None:
                tensor_3ch = extract_multi_channel_features(normalized_hand)

                # Hiển thị 3 kênh đặc trưng xếp cạnh nhau để kiểm tra trực quan
                ch0 = tensor_3ch[:, :, 0]  # CLAHE
                ch1 = tensor_3ch[:, :, 1]  # Canny
                ch2 = tensor_3ch[:, :, 2]  # Sobel

                # Ghép 3 ảnh thành một dải băng ngang (128, 128*3)
                three_channels_vis = np.hstack([ch0, ch1, ch2])
                three_channels_vis = cv2.cvtColor(three_channels_vis, cv2.COLOR_GRAY2BGR)

                # Thêm chú thích cho từng kênh
                cv2.putText(three_channels_vis, "CH0: CLAHE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.putText(three_channels_vis, "CH1: CANNY", (138, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.putText(three_channels_vis, "CH2: SOBEL", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                cv2.imshow("3. Normalized Hand (Moments 0 deg)", normalized_hand)
                cv2.imshow("4. Multi-Channel DIP Tensor (128x128x3)", three_channels_vis)

            # Hiển thị cửa sổ chính và mặt nạ
            cv2.imshow("1. Webcam Input & Pipeline Status", frame)
            cv2.imshow("2. Adaptive Skin Mask (Backprojection)", mask)

        # Phím điều khiển
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') or key == ord('r'):
            print("Đang Reset Calibration...")
            segmenter.reset_calibration()

    cap.release()
    cv2.destroyAllWindows()
