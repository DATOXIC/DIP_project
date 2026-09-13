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

## PHẦN 2: LỘ TRÌNH 4 CẢI TIẾN TĂNG ĐỘ CHÍNH XÁC (/grill-me ROADMAP)

### Đánh giá tác động hệ thống:
- **Tính khả thi:** Cả 4 cải tiến nằm ở các khâu độc lập trên luồng dữ liệu, **không triệt tiêu lẫn nhau** mà hỗ trợ tương hỗ.
- **Tác động hiệu năng:** Hoàn toàn dựa trên toán học ma trận và logic O(1), **tổng thời gian thêm vào ≈ 0.01 ms**, duy trì FPS >= 30 - 45 FPS trên CPU.

### 4 Trụ cột Cải tiến:
1. **Cải tiến 1 - Soft Confusion Matrix cho Dáng tay trong DTW:** Thay vì phạt cứng 0.6 cho mọi cặp chữ cái khác nhau, áp dụng ma trận phạt mềm dựa trên độ tương đồng hình thái học (ví dụ: nhầm 'A' với 'C' phạt nhẹ 0.2, nhầm 'A' với 'B' phạt nặng 0.8).
2. **Cải tiến 2 - Bổ sung Chiều sâu ảo Z-Depth vào Vector Quỹ đạo:** Tận dụng độ co giãn tức thời của palm_len hoặc tọa độ Z của MediaPipe thành vector 5 chiều [x, y, z, dx, dy], giúp phân biệt hoàn hảo cử chỉ đẩy tới trước ("Bạn") và kéo về ngực ("Tôi").
3. **Cải tiến 3 - Ngưỡng Động Thích nghi & Margin Gap Top-2:** Thiết lập ngưỡng riêng cho từng từ trong từ điển và yêu cầu khoảng cách giữa Top-1 và Top-2 phải đủ lớn trước khi chấp nhận.
4. **Cải tiến 4 - Tinh chỉnh Cổng phân xử mềm Bayes (Soft Gate Giai đoạn 1):** Tối ưu bảng trọng số γ theo số ngón tay n_fingers để cung cấp nhãn c_shape sạch hơn vào DTW.
