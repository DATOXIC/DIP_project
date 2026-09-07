# BẢN THIẾT KẾ KIẾN TRÚC: NHẬN DIỆN CỬ CHỈ ĐỘNG & GHÉP CÂU HOÀN CHỈNH
## Vietnamese Sign Language (VSL) Dynamic Sentence Builder
**Sinh viên:** Tô Phương Đạt | **MSSV:** 24110086 | **Môn học:** Xử Lý Ảnh Số (DIP)

---

## 1. TỔNG QUAN & RÀNG BUỘC KỸ THUẬT

### 1.1. Mục tiêu
Nâng cấp hệ thống từ nhận diện ký hiệu tĩnh (chữ cái rời rạc A, B, C, L, V) lên **Nhận diện cử chỉ động theo chuỗi thời gian để ghép thành một câu thoại hoàn chỉnh** (Ví dụ: *"Tôi"* + *"Cảm ơn"* + *"Bạn"* -> `"Tôi Cảm ơn Bạn."`).

### 1.2. Ràng buộc & Tiền đề Thiết kế (Design Assumptions)
1. **Quy ước Người dùng:** Cử chỉ thực hiện **rõ ràng, ngắt quãng, tách biệt**:
   * Bắt đầu cử chỉ -> Chuyển động -> Khựng tay dừng dứt khoát 0.35s ở cuối cử chỉ để máy chốt từ.
2. **Tài nguyên Hệ thống:**
   * Hoạt động thời gian thực (>= 25 FPS) **thuần túy trên CPU** của laptop phổ thông.
   * Webcam độ phân giải khiêm tốn (640x480), không yêu cầu GPU rời.
3. **Kế thừa Tinh hoa:**
   * Tái sử dụng 100% bộ lọc **BlazeHand Detector + DIP Engine v4 + Two-Stream CNN** đã chứng minh khả năng kháng 19 bad cases môi trường.

---

## 2. ĐÁP ÁN 4 ĐIỂM KỸ THUẬT THEN CHỐT

### 2.1. Trọng số Hàm Chi Phí Lai (Hybrid DTW Cost Function)
$$\text{Cost}(i,j) = \alpha \cdot \|(x_i,y_i) - (x_j,y_j)\|_2 + \beta \cdot \|(\Delta x_i, \Delta y_i) - (\Delta x_j, \Delta y_j)\|_2 + \lambda \cdot \mathbb{I}(S_i \neq S_j)$$

* **$\alpha = 1.0$:** Trọng số Quỹ đạo hình học nền tảng (khoảng cách tọa độ).
* **$\beta = 0.5$:** Trọng số Hướng di chuyển tiếp tuyến (loại trừ cử chỉ cùng hình dạng nhưng ngược chiều).
* **$\lambda = 0.6$:** Hình phạt khác biệt dáng bàn tay (đủ mạnh để loại trừ khác dáng, nhưng bao dung nếu CNN lệch 1-2 frame chuyển tiếp).

### 2.2. Cơ chế Nội suy 2 Nhánh về K=32 frames
1. **Tọa độ liên tục $(x_t, y_t)$:** Nội suy tuyến tính 1D (Linear Interpolation) về 32 mốc đều nhau.
2. **Vector hướng $(\Delta x, \Delta y)$:** Tính trực tiếp bằng đạo hàm sai phân hữu hạn trên chuỗi 32 điểm đã nội suy: $\Delta x_k = x_k - x_{k-1}, \Delta y_k = y_k - y_{k-1}$.
3. **Dáng tay $S_t$ (Categorical):** Dùng **Nearest-Neighbor Interpolation** để bảo toàn nhãn rời rạc $A, B, C, L, V$.

### 2.3. Quy tắc Hình học Kích hoạt Lệnh Swipe-Left (Backspace)
Trong cửa sổ trượt ngắn $W = 6\text{ frames}$ ($\approx 0.2\text{s}$), lệnh xóa từ kích hoạt tức thời khi thỏa mãn 3 điều kiện:
1. $\Delta x_{\text{total}} < -0.80 \times \text{palm\_len}$ (quãng đường dời sang trái $\ge 80\%$ lòng bàn tay).
2. $\frac{|\Delta x|}{|\Delta y| + 10^{-4}} \ge 2.5$ (áp đảo phương ngang, loại cử chỉ chéo).
3. Tồn tại ít nhất 1 frame có $v_x^{\text{norm}} < -0.20$ (bứt tốc nhanh).
4. Khóa Cooldown $1.0\text{s}$ sau khi xóa.

### 2.4. Schema Lưu Trữ Chuẩn Hóa (`dynamic_templates.json`)
Tách biệt mảng số thực `traj_features` ($32 \times 4$) và mảng chuỗi nhãn `hand_shapes` (độ dài 32), có `schema_version = "1.0"` và metadata độc lập.

---

## 3. SƠ ĐỒ KIẾN TRÚC TOÀN CẢNH (LEAN 4-LAYER PIPELINE)

```
                            LUỒNG DỮ LIỆU THỜI GIAN THỰC
                            
   Webcam Input (640x480 @ 30 FPS)
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: TRÍCH XUẤT ĐẶC TRƯNG TỨC THỜI (Per-Frame Spatio-Temporal Extractor) │
│  • BlazeHand: 21 Landmarks, Tâm cổ tay w_t = (x_t, y_t), Kích thước palm_len │
│  • DIP Engine v4: Bilateral + CIELab/YCrCb + Moments + Tensor 3 kênh        │
│  • Two-Stream CNN: Nhãn dáng bàn tay tĩnh S_t in {A, B, C, L, V, Flat, ...} │
│  • Vector hướng tiếp tuyến: Δx_t = x_t - x_{t-1}, Δy_t = y_t - y_{t-1}      │
│  ==> Gói tin frame t: F_t = (x_norm, y_norm, Δx, Δy, S_t, v_norm)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: PHÂN ĐOẠN ĐỘNG HỌC THEO PALM-SCALE (Kinetic Pause Segmenter)        │
│  • Vận tốc chuẩn hóa theo cơ thể: v_norm = Δw / palm_len (% lòng bàn tay)   │
│  • Máy trạng thái hữu hạn (FSM):                                            │
│      [IDLE: Chờ v_norm < 0.04]                                              │
│         │ (v_norm >= 0.08)                                                  │
│         ▼                                                                   │
│      [RECORDING: Tích lũy chuỗi quỹ đạo vào buffer]                         │
│         │ (v_norm <= 0.03 liên tục trong 0.35s / 11 frames)                 │
│         ▼                                                                   │
│      [PAUSE_DETECTED: Khựng tay chốt từ] ──> Kiểm tra độ dài N in [8, 60]   │
│  ==> Chuỗi cử chỉ nguyên bản: Q_raw = [F_1, F_2, ..., F_N]                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: BỘ SO KHỚP CHUỖI NỘI SUY LAI (Resampled Hybrid DTW Engine)          │
│  • Nội suy 2 nhánh: Tọa độ Linear, Hướng Finite Diff, Dáng tay Nearest-N    │
│  • Neo gốc chuyển động (Stroke-Origin Anchor): Đưa điểm đầu về (0, 0)       │
│  • Dynamic Time Warping với dải băng Sakoe-Chiba (Bán kính r = 4):          │
│      Cost(i, j) = 1.0*||pos_i - pos_j|| + 0.5*||dir_i - dir_j|| + 0.6*I(S_i!=S_j)
│  • Chiến lược phân tầng ngưỡng (Decision Hierarchy):                         │
│      - Dist <= Thresh          : CHẤP NHẬN TỪ                               │
│      - Thresh < Dist <= 1.4*Th : NGHI VẤN (Hiển thị Top-2 debug trên HUD)   │
│      - Dist > 1.4*Th          : TỪ CHỐI (Cử chỉ lạ / Quơ tay rác)          │
│  ==> Kết quả: Từ vựng nhận diện (Ví dụ: 'Xin chào', 'Cảm ơn', 'Tôi')        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 4: BỘ ĐIỀU PHỐI GHÉP CÂU & GIAO DIỆN (Sentence Builder & Dialogue HUD) │
│  • Bộ điều khiển Lệnh Ưu tiên Cấp tốc (Priority Gesture Command):           │
│      - Swipe Left (Vận tốc ngang cực đại v_x < -0.25): XÓA TỪ (Backspace)  │
│  • Chống bắn từ lặp lại (Debouncing 0.8s)                                   │
│  • Bộ ghép câu: sentence = ['Tôi', 'Cảm ơn', 'Bạn'] ──> 'Tôi Cảm ơn Bạn.'   │
│  • Cử chỉ hoàn tất câu (Enter): Hạ tay khỏi webcam quá 1.5s                 │
│  • Giao diện HUD: Phụ đề câu thoại, vệt sáng quỹ đạo, thanh đo vận tốc      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. KẾ HOẠCH TRIỂN KHAI 4 GIAI ĐOẠN

| Giai đoạn | Tên module / Tác vụ | File tạo mới | Trọng tâm kiểm thử |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 1** | **Lõi Giải thuật Động học & So khớp** | `dynamic_dtw_engine.py` | Unit test: Kiểm tra tính bất biến tốc độ, dải Sakoe-Chiba, nội suy 2 nhánh, phân biệt hướng. |
| **Giai đoạn 2** | **Công cụ Thu thập Mẫu Chuẩn (1 Chạm)** | `collect_dynamic_templates.py` | Thu thập 3 mẫu chuẩn cho 5 từ VSL mẫu. |
| **Giai đoạn 3** | **Chương trình Nhận diện Ghép câu Realtime** | `predict_sentence_webcam.py` | Tích hợp trọn vẹn 4 tầng lên webcam: Múa ghép câu mượt mà. |
| **Giai đoạn 4** | **Kiểm thử Thực chiến & Viết Báo cáo** | `walkthrough.md` | Đánh giá FPS, khả năng chống nhiễu, kiểm tra Backspace. |
