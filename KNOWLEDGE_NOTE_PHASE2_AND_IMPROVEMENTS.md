# TỔNG HỢP KIẾN TRÚC GIAI ĐOẠN 2 & KẾ HOẠCH CẢI TIẾN TỐI ƯU HÓA HỆ THỐNG
**Đồ án:** Nhận diện Thủ ngữ Thời gian thực (VSL/ASL Real-Time Pipeline)  
**Sinh viên:** Tô Phương Đạt | MSSV: 24110086  
**Môn học:** Xử Lý Ảnh Số (DIP)

---

## PHẦN 1: BẢN ĐỒ KIẾN TRÚC GIAI ĐOẠN 2 (ĐÃ HOÀN THÀNH)

### 1. Workflow 3 Nhịp Cốt Lõi:
1. **Khử rung giật & Giữ chuyển động mượt mà:**
   - **Điểm neo giải phẫu:** Khớp cổ tay (Landmark 0).
   - **1€ Filter (Bộ lọc thông thấp thích ứng vận tốc 2 tầng):**
     - Bộ lọc 1 (f_c_rate = 1.0 Hz): Làm phẳng vận tốc để tìm tần số cắt động f_c = f_c_min + β * |v|.
     - Bộ lọc 2: Làm mượt tọa độ vị trí. Khi đứng yên: v ≈ 0 -> lọc cực mạnh (triệt tiêu rung giật). Khi vung nhanh: v lớn -> nhả lọc (độ trễ bằng 0).
   - **Lucas-Kanade Optical Flow (Phao cứu sinh quán tính):**
     - Khi MediaPipe mất dấu 1–2 frame (do vung nhanh nhòe hình), dùng cửa sổ 15x15 trên ảnh xám để bám theo 6 điểm mốc neo theo quán tính quang học, bù điểm bị mất để không làm đứt chuỗi cử chỉ.
   - **Kinetic Pause Segmenter FSM:**
     - Đo vận tốc chuẩn hóa theo kích thước bàn tay: v_norm = Δw / palm_len.
     - Chuyển trạng thái: IDLE -> RECORDING -> PAUSE_DETECTED.
     - 3 Chốt chặn biên: Thời lượng 0.4s <= T <= 2.5s; Quãng đường > 0.5 * palm_len; Swipe-Left (Vuốt trái nhanh = Xóa từ Backspace).

2. **Chuẩn hóa Bất biến & So khớp Chuỗi (DTW):**
   - **Biến đổi Hệ quy chiếu:** Trừ gốc ban đầu (x_0, y_0) (bất biến vị trí đứng) và chia palm_len (bất biến khoảng cách xa/gần).
   - **Nội suy K=32 frames:**
     - Nhánh số [x, y, dx, dy]: Nội suy tuyến tính (Linear Interpolation).
     - Nhánh chữ c_shape: Nội suy láng giềng gần nhất (Nearest-Neighbor).
   - **Hybrid DTW + Dải băng Sakoe-Chiba (r=4):**
     - Hàm chi phí kết hợp: Cost(i, j) = 1.0 * ||pos|| + 0.5 * ||dir|| + 0.6 * I(c_shape_i != c_shape_j).
     - Dải kẹp Sakoe-Chiba khống chế lệch không quá 4 frame, chống hiện tượng chờ quá lâu (bệnh lý) và tăng tốc CPU.

3. **Phán đoán Quyết định & Giao tiếp (HCI):**
   - **3 Tầng Quyết định:** ACCEPTED (<= 1.25), UNCERTAIN (1.25 - 1.75), REJECTED (> 1.75).
   - **Sentence Builder & Non-blocking TTS:** Tự động chốt câu khi hạ tay > 1.8s, phát âm thanh qua luồng ngầm daemon không gây tụt FPS.

---

## PHẦN 2: THỐNG NHẤT THIẾT KẾ 3 CẢI TIẾN TRỌNG TÂM (/grill-me CONSENSUS)

### Nguyên tắc thiết kế cốt lõi:
- **Tương thích ngược 100%:** Chỉ chỉnh sửa trong `predict_sentence_webcam.py` (và các lớp bổ trợ), TUYỆT ĐỐI KHÔNG sửa phá vỡ các file lõi đã khóa.
- **Chi phí tính toán gần bằng 0:** Không gọi mô hình nặng, giữ nguyên FPS >= 30 trên CPU.
- **Tập trung dứt điểm 3 cải tiến Giai đoạn 2 (đã bỏ qua chỉnh sửa Soft Gate GĐ1 để giữ ổn định):**

### 1. CẢI TIẾN 1: SOFT CONFUSION MATRIX CHO DÁNG TAY TRONG DTW (Phương án Hybrid 1AB)
- **Vấn đề cũ:** Khác dáng tay phạt cứng 0.6 bất kể lệch nhẹ hay lệch nặng.
- **Giải pháp mới:** 
  - Tính toán khoảng cách phạt dáng tay d_shape dựa trên bảng ma trận tương đồng hình thái DIP:
    + Cùng lớp: phạt = 0.0
    + Tương đồng cao (ví dụ: A vs C lệch khum nhẹ): phạt nhẹ = 0.20
    + Tương đồng trung bình (A vs L, C vs V): phạt vừa = 0.50
    + Hoàn toàn đối lập (Nắm đấm A vs Xòe tay B): phạt nặng = 0.85
  - Nhúng độ tin cậy xác suất CNN: Nếu CNN tự tin thấp (< 0.6), giảm bớt tiền phạt để tránh phạt oan khi camera bị mờ.

### 2. CẢI TIẾN 2: CHIỀU SÂU ẢO Z-DEPTH (Phương án 2A)
- **Vấn đề cũ:** Chỉ có [x, y, dx, dy] trên mặt phẳng 2D, khó phân biệt rõ từ đẩy tới ("Bạn") và kéo về ("Tôi").
- **Giải pháp mới:** 
  - Trích xuất độ co giãn giải phẫu theo chiều sâu:
    $$z_{norm}(t) = \frac{palm\_len(t) - palm\_len(0)}{palm\_len(0)}$$
  - Khi tay đẩy về phía camera: bàn tay to lên -> z_norm > 0.
  - Khi tay kéo lùi về ngực: bàn tay nhỏ đi -> z_norm < 0.
  - Bổ sung vào khoảng cách hình học: d_pos = sqrt((Δx)^2 + (Δy)^2 + (w_z * Δz)^2) với w_z = 0.8.

### 3. CẢI TIẾN 3: NGƯỠNG ĐỘNG THÍCH NGHI & TOP-2 MARGIN GAP (Chiến lược 3A)
- **Vấn đề cũ:** Dùng chung default_threshold = 1.25 dễ gây đoán bừa khi múa cử chỉ lạ hoặc mấp mé giữa 2 từ tương đồng.
- **Giải pháp mới:** 
  - Áp dụng cấu hình ngưỡng riêng (per-word threshold) cho từng từ:
    + Từ đơn giản ("Tôi", "Bạn"): threshold = 1.10 (chặt chẽ).
    + Từ phức tạp ("Xin chào", "Tạm biệt"): threshold = 1.35 (thoáng hơn).
  - Thêm điều kiện an toàn **Margin Ratio**:
    $$\text{Margin Ratio} = \frac{D_{\text{Top-2}}}{D_{\text{Top-1}}} \ge 1.20$$
    Nếu tỷ lệ < 1.20 (nghĩa là Top-1 và Top-2 quá sát nhau, hệ thống đang phân vân không chắc chắn) -> Lập tức phân loại thành `UNCERTAIN` để yêu cầu múa lại, triệt tiêu hoàn toàn lỗi đoán bừa!
