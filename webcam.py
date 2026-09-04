import cv2
import numpy as np
from collections import deque, Counter

# 1. Mở Webcam (0 là camera mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở Webcam!")
    exit()

# Tọa độ khung ROI (Region of Interest) - vị trí đặt bàn tay
ROI_TOP, ROI_BOTTOM = 80, 420
ROI_LEFT, ROI_RIGHT = 320, 620

# =========================================================================
# GIẢI QUYẾT THỦ PHẠM 5: BỘ ĐỆM LÀM MƯỢT THỜI GIAN (TEMPORAL VOTING BUFFER)
# =========================================================================
# Lưu kết quả 7 khung hình gần nhất (khoảng 0.23 giây) để lọc nhiễu nháy số
voting_buffer = deque(maxlen=7)
stable_fingers = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được luồng hình ảnh từ Webcam.")
        break

    # Lật ngược ảnh theo chiều ngang (như soi gương)
    frame = cv2.flip(frame, 1)

    # 1. Cắt vùng quan tâm (ROI) - chỉ xử lý bàn tay trong ô này
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

    # =========================================================================
    # GIẢI QUYẾT THỦ PHẠM 1: CÂN BẰNG SÁNG & TÁCH SẮC ĐỘ ĐỘNG
    # =========================================================================
    # 2.1. Cân bằng trắng Gray-World để triệt tiêu ám sắc đèn vàng/xanh
    b, g, r = cv2.split(roi)
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    
    # Tính hệ số bù màu (tránh chia cho 0)
    kb = mean_gray / (mean_b + 1e-6)
    kg = mean_gray / (mean_g + 1e-6)
    kr = mean_gray / (mean_r + 1e-6)
    
    balanced_roi = cv2.merge([
        np.clip(b * kb, 0, 255).astype(np.uint8),
        np.clip(g * kg, 0, 255).astype(np.uint8),
        np.clip(r * kr, 0, 255).astype(np.uint8)
    ])

    # 2.2. Làm mờ nhẹ vùng ROI đã cân bằng màu để giảm nhiễu hạt sensor
    blurred = cv2.GaussianBlur(balanced_roi, (5, 5), 0)

    # 2.3. Chuyển đổi sang YCrCb và cân bằng độ tương phản kênh Y bằng CLAHE
    ycrcb_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_channel = clahe.apply(y_channel)
    ycrcb_clahe = cv2.merge([y_channel, cr_channel, cb_channel])

    # 2.4. Ngưỡng lọc da tay trên YCrCb (đã ổn định sau khi cân bằng Y)
    # Đặt ngưỡng sàn Y >= 50 để loại bỏ rãnh bóng tối kẽ ngón tay sau khi bị CLAHE dìm xuống
    lower_ycrcb = np.array([50, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb_clahe, lower_ycrcb, upper_ycrcb)

    # 2.5. Bộ lọc phụ HSV (tách sắc tướng Hue của da người, chống bắt nhầm nền vàng)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([25, 180, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    # 2.6. Tạo mặt nạ nhị phân kết hợp (Chỉ nhận pixel thỏa mãn cả YCrCb và HSV)
    skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # 6. Khử nhiễu hình thái học (Morphological Operations)
    kernel = np.ones((5, 5), np.uint8)
    # Morph OPEN: Xóa các hạt nhiễu trắng li ti
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    # Morph CLOSE: Lấp các lỗ hổng đen nhỏ trên bàn tay
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # 7. Trích xuất đường biên (Find Contours) từ Mask trong ROI
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_fingers = 0

    # 8. Phân tích contour bàn tay bên trong ROI
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(max_contour) > 2000:
            # =================================================================
            # GIẢI QUYẾT THỦ PHẠM 2: CHUẨN HÓA BẤT BIẾN TỈ LỆ (SCALE INVARIANCE)
            # =================================================================
            # Lấy kích thước bao của bàn tay để tạo thước đo co dãn động
            x_box, y_box, w_box, h_box = cv2.boundingRect(max_contour)
            hand_size = max(w_box, h_box)
            min_depth = hand_size * 0.08  # Ngưỡng sàn: Lọc răng cưa / nếp nhăn nông
            max_depth = hand_size * 0.60  # Ngưỡng trần: Lọc điểm lõm cổ tay quá sâu cực đoan

            # Vẽ đường biên bàn tay lên ROI (màu xanh lá)
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            # Lấy chỉ số Vỏ lồi (returnPoints=False để tìm defects)
            hull_indices = cv2.convexHull(max_contour, returnPoints=False)
            
            # Vẽ đa giác bao lồi (màu xanh dương)
            hull_points = cv2.convexHull(max_contour, returnPoints=True)
            cv2.drawContours(roi, [hull_points], -1, (255, 0, 0), 2)

            finger_count = 0

            # Tìm các điểm lõm (Convexity Defects)
            if hull_indices is not None and len(hull_indices) > 3:
                defects = cv2.convexityDefects(max_contour, hull_indices)

                if defects is not None:
                    for s_idx, e_idx, f_idx, d in defects.reshape(-1, 4):
                        depth = d / 256.0

                        start = tuple(map(int, max_contour[s_idx].flatten()))
                        end = tuple(map(int, max_contour[e_idx].flatten()))
                        far = tuple(map(int, max_contour[f_idx].flatten()))

                        # Tính độ dài 3 cạnh tam giác (start, end, far)
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(end) - np.array(far))

                        # Định lý cosin để tính góc kẽ tay
                        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c + 1e-6)
                        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

                        # =============================================================
                        # GIẢI QUYẾT THỦ PHẠM 3 & 4: LỌC KẼ TAY & LOẠI BỎ HÕM CỔ TAY
                        # =============================================================
                        # Giải pháp A (Thủ phạm 4): Kẽ tay chỉ được nằm ở 75% phía trên của bàn tay
                        # (Triệt tiêu 100% các điểm lõm ma do cổ tay/cẳng tay tạo ra ở mép dưới)
                        is_in_hand_region = far[1] < (y_box + h_box * 0.75)

                        # Nhánh 1: Kẽ 4 ngón tay song song thông thường (Góc nhọn <= 90 độ)
                        is_normal_gap = (angle <= 90) and (min_depth < depth < max_depth) and is_in_hand_region

                        # Nhánh 2: Kẽ ngón cái & ngón trỏ (Mở rộng góc tới 115 độ, yêu cầu độ sâu sâu hơn)
                        is_thumb_gap = (90 < angle <= 115) and (min_depth * 1.4 < depth < max_depth) and is_in_hand_region

                        if is_normal_gap or is_thumb_gap:
                            finger_count += 1
                            # Vẽ chấm đỏ ở kẽ tay
                            cv2.circle(roi, far, 6, (0, 0, 255), -1)
                            # Vẽ đường nối tạo khung ngón tay
                            cv2.line(roi, start, far, (0, 255, 255), 2)
                            cv2.line(roi, far, end, (0, 255, 255), 2)

            # Số ngón tay = số kẽ tay + 1
            if finger_count > 0:
                total_fingers = finger_count + 1
            else:
                # =============================================================
                # GIẢI QUYẾT THỦ PHẠM 4 (Giải pháp B): PHÂN BIỆT 0 NGÓN VS 1 NGÓN
                # =============================================================
                # Tính độ đặc (Solidity) và Tỉ lệ khung hình (Aspect Ratio = h / w)
                contour_area = cv2.contourArea(max_contour)
                hull_area = cv2.contourArea(hull_points)
                solidity = float(contour_area) / hull_area if hull_area > 0 else 0
                aspect_ratio = float(h_box) / (w_box + 1e-6)

                # Giơ 1 ngón: Ngón trỏ vươn cao làm dáng thon dài (aspect_ratio cao) hoặc rỗng bên cạnh
                # Nắm tay (0 ngón): Dáng vuông vức tròn xoe (aspect_ratio thấp) và đặc ruột
                if aspect_ratio > 1.35 or solidity < 0.78:
                    total_fingers = 1  # Giơ 1 ngón
                else:
                    total_fingers = 0  # Nắm tay

    # =========================================================================
    # LÀM MƯỢT KẾT QUẢ BẰNG BỘ ĐỆM BỎ PHIẾU ĐA SỐ (MAJORITY VOTING)
    # =========================================================================
    voting_buffer.append(total_fingers)
    if len(voting_buffer) > 0:
        stable_fingers = Counter(voting_buffer).most_common(1)[0][0]

    # 9. Vẽ khung hộp ROI màu vàng để hướng dẫn người dùng đặt tay
    cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 255), 2)
    cv2.putText(frame, "Dat tay vao day", (ROI_LEFT, ROI_TOP - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Hiển thị số lượng ngón tay đã được làm mượt ổn định lên màn hình chính
    cv2.putText(frame, f"Ngon tay: {stable_fingers}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 10. Hiển thị kết quả ra màn hình
    cv2.imshow("1. Webcam Hand Tracking (ROI)", frame)
    cv2.imshow("2. Mat Na Nhi Phan (Skin Mask ROI)", skin_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()