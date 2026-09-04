# Đánh Giá Toàn Diện: `improved_dip_pipeline.py`

> Mục tiêu: Phân tích kỹ lưỡng pipeline DIP cho nhận diện thủ ngữ VSL — xác định điểm mạnh, điểm yếu, lỗ hổng khi triển khai thực tế, và nguồn tham khảo để học hỏi.

---

## 1. Điểm Mạnh — Đã Làm Tốt ✅

| Kỹ thuật | Đánh giá |
|----------|----------|
| **Histogram Backprojection 2D (H, S)** | Thay thế ngưỡng cứng bằng mô hình xác suất. Đây là cải tiến đúng hướng, thích nghi được với từng người. |
| **Skin Prior Gate khi calibration** | Chỉ học histogram từ pixel có `S ≥ 38`, loại trắng/xám — thiết kế tốt, ngăn học nhầm màu tủ/tường. |
| **Center Focus Mask** | Chỉ lấy mẫu ở 70% tâm ROI, tránh pixel biên lẫn phông nền. |
| **Running Average Histogram** (`0.85 old + 0.15 new`) | Ổn định dần qua 30 frame, không bị nhiễu từ frame đơn lẻ. |
| **Saturation Gate khi segment** (`S ≥ 32`) | Triệt tiêu vật phản quang, nền trắng. |
| **Hole Filling bằng `cv2.FILLED`** | Hàn kín lòng bàn tay, loại lỗ thủng do phản quang — kỹ thuật đúng đắn. |
| **Image Moments + Principal Axis** | Chuẩn hóa hướng bàn tay, giúp CNN chỉ cần học 1 hướng chuẩn. |
| **Flip Detection** (`top_area vs bot_area`) | Phát hiện và sửa trường hợp bàn tay bị lật ngược. |
| **Aspect Ratio Padding** | Giữ đúng tỉ lệ khi resize về 128×128, không méo hình. |
| **3-channel feature tensor** (CLAHE + Canny + Sobel) | Cung cấp đặc trưng phong phú cho CNN. |
| **CLAHE với hand mask** | Chỉ áp dụng CLAHE trên vùng tay, nền đen vẫn bằng 0 — đúng kỹ thuật. |

---

## 2. Điểm Yếu & Lỗ Hổng — Cần Cải Thiện ⚠️

### 🔴 Nghiêm Trọng — Ảnh Hưởng Lớn Đến Dữ Liệu Thực Tế

#### [A] Không có cân bằng màu trước khi calibrate/segment

Pipeline mới **bỏ đi** bước Gray-World White Balance mà `webcam.py` đã có. Đây là bước lùi nghiêm trọng:

```
Vấn đề:
  - Đèn vàng (tungsten): dịch Hue +10..+15 độ → histogram học sai sắc da
  - Đèn xanh (LED lạnh): dịch Hue -5..−10 → backprojection miss bàn tay
  - Ánh nắng chiều: Saturation tăng vọt → Otsu nhảy ngưỡng sai

Hậu quả:
  Histogram calibrate dưới ánh đèn phòng A, dùng ở phòng B (khác đèn)
  → Mask gần như trống hoặc bắt nhầm nền.
```

**Giải pháp:** Thêm lại Gray-World (hoặc von Kries) ngay đầu pipeline, trước khi convert sang HSV:

```python
def apply_white_balance(roi_bgr):
    b, g, r = cv2.split(roi_bgr)
    mean_gray = (np.mean(b) + np.mean(g) + np.mean(r)) / 3.0
    balanced = cv2.merge([
        np.clip(b * (mean_gray / (np.mean(b) + 1e-6)), 0, 255).astype(np.uint8),
        np.clip(g * (mean_gray / (np.mean(g) + 1e-6)), 0, 255).astype(np.uint8),
        np.clip(r * (mean_gray / (np.mean(r) + 1e-6)), 0, 255).astype(np.uint8),
    ])
    return balanced
```

---

#### [B] Histogram bị đóng băng sau calibration — không thích nghi khi ánh sáng thay đổi

Sau 30 frame, `self.hist` không bao giờ được cập nhật nữa. Nếu người dùng bật thêm đèn hoặc di chuyển sang vị trí khác:

```
Calibrate lúc 7h tối (đèn vàng) → Histogram học sắc vàng của đèn
Quay tiếp lúc 8h (bật đèn trắng) → Histogram cũ không còn đúng
→ Mask bị hỏng, dữ liệu thu thập sai
```

**Giải pháp:** Thêm **online histogram update** với EMA rất chậm trong `segment()`:

```python
# Trong segment(), sau khi backproject thành công
# Tính histogram mới từ vùng mask xác nhận là da
# Cập nhật rất chậm: alpha = 0.01
if high_confidence_pixels > threshold:
    self.hist = 0.99 * self.hist + 0.01 * hist_new
```

---

#### [C] ROI cố định — không theo dõi bàn tay di chuyển

Cả calibration lẫn segmentation đều bị khóa vào tọa độ `(80, 420, 320, 620)`. Nếu bàn tay ra khỏi khung:

```
→ mask trả về zeros
→ normalize_hand_orientation() trả None
→ Frame bị bỏ qua, dataset trống
```

Đây là giới hạn lớn khi thu thập dữ liệu thực tế — người dùng phải giữ tay trong khung chính xác.

---

#### [D] Ngưỡng diện tích contour cứng 2000 pixel

```python
if cv2.contourArea(max_contour) < 2000:
    return None, None
```

Ngưỡng 2000 px được thiết kế cho camera cụ thể. Nếu camera có độ phân giải khác hoặc tay đặt gần/xa hơn:
- Camera 1920×1080: 2000 px quá nhỏ → bắt nhầm noise
- Camera 320×240: 2000 px quá lớn → bỏ qua bàn tay thực

**Giải pháp:** Đặt ngưỡng tương đối theo diện tích ROI:

```python
roi_area = (bottom - top) * (right - left)
min_hand_area = roi_area * 0.05  # ít nhất 5% ROI là bàn tay
if cv2.contourArea(max_contour) < min_hand_area:
    return None, None
```

---

### 🟡 Trung Bình — Cần Chú Ý

#### [E] Bất nhất giữa ngưỡng Saturation calibration vs segment

```python
# calibrate(): Skin Prior Gate dùng S >= 38
lower_skin1 = np.array([0, 38, 40], ...)

# segment(): Saturation Gate dùng S >= 32
sat_gate = (hsv_roi[:, :, 1] >= 32).astype(np.uint8) * 255
```

Sự lệch 6 đơn vị Saturation này có thể khiến những pixel nằm trong khoảng `[32, 37]` **không được học** trong histogram nhưng **vẫn được phép qua** cổng khi segment → một số điểm nhiễu nhẹ lọt qua.

**Sửa:** Đồng bộ về cùng ngưỡng, ví dụ `S >= 35` cho cả hai.

---

#### [F] Canny thresholds cố định (30, 100)

```python
channel1 = cv2.Canny(blurred, 30, 100)
```

Ngưỡng Canny `(30, 100)` là một lựa chọn an toàn nhưng không thích nghi. Với ảnh thiếu sáng (nhiều noise), Canny sẽ tạo ra rất nhiều cạnh giả. Với ảnh quá sáng (overexposed), các cạnh ngón tay mờ đi và Canny bỏ sót.

**Giải pháp:** Dùng **Auto-Canny** dựa trên median của ảnh:

```python
def auto_canny(gray, sigma=0.33):
    v = np.median(gray[gray > 0])  # Median chỉ trên vùng tay (> 0)
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lo, hi)
```

---

#### [G] Góc Principal Axis không ổn định với một số tư thế tay

```python
rot_angle = angle_deg - 90 if angle_deg > 0 else angle_deg + 90
```

Công thức `atan2(2*mu11, mu20 - mu02)` trả về giá trị trong `[-90°, 90°]`, có **ambiguity 180°**. Bước flip detection bằng `top_area/bot_area` là heuristic hợp lý nhưng có thể fail với:
- Ký hiệu VSL có ngón tay xòe ngang (axis gần nằm ngang)
- Tư thế bàn tay ngón cái tách ra nhiều (phân phối moments bị lệch)

Khi `mu20 ≈ mu02` (hình tròn/vuông), `atan2` sẽ bất ổn.

**Sửa:** Thêm kiểm tra độ tin cậy của moments:

```python
mu_diff = abs(mu20 - mu02)
if mu_diff < 1e6 and abs(mu11) < 1e6:
    # Moments quá cân bằng → không xoay, giữ nguyên
    rot_angle = 0.0
```

---

#### [H] Không xử lý nhiều người trong frame / mặt người

Pipeline chỉ dùng largest contour và giới hạn trong ROI. Tuy nhiên nếu người dùng mặc áo màu da hoặc có thêm vật thể skin-tone trong ROI (cổ tay, cẳng tay), contour lớn nhất có thể **không phải bàn tay**.

Không có bộ lọc hình dạng bổ sung (circularity, convexity) để xác nhận đây là bàn tay thực.

---

### 🟢 Nhỏ — Cải Thiện Ưu Tiên Thấp

#### [I] Gaussian Blur dùng trước Canny/Sobel nhưng không dùng trước CLAHE

CLAHE đang áp dụng trực tiếp trên ảnh grayscale gốc (chưa blur). Ở ảnh có nhiều noise sensor, CLAHE có thể khuếch đại noise thành cấu trúc giả.

```python
# Thứ tự hiện tại:
gray → CLAHE → channel0   ← thiếu blur trước CLAHE
gray → blur → Canny       ← đúng

# Nên đổi:
blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)  # Blur nhẹ trước
channel0 = clahe.apply(blurred)
```

#### [J] Không có temporal smoothing cho mask

Otsu threshold thay đổi frame-by-frame → mask có thể nhảy ngưỡng → tensor đầu ra bất nhất giữa các frame liên tiếp. Với dữ liệu training tĩnh thì ổn, nhưng với inference realtime sẽ thấy mask "nhấp nháy".

---

## 3. Bảng Tổng Hợp Mức Độ Ưu Tiên

| # | Vấn đề | Mức độ | Khuyến nghị |
|---|--------|--------|-------------|
| A | Thiếu White Balance | 🔴 Cao | Thêm ngay trước convert HSV |
| B | Histogram đóng băng | 🔴 Cao | Thêm online EMA update |
| C | ROI cứng không tracking | 🔴 Cao | Chấp nhận giới hạn hoặc thêm tracking |
| D | Ngưỡng area 2000px cứng | 🟡 TB | Đổi sang % diện tích ROI |
| E | Bất nhất S threshold | 🟡 TB | Đồng bộ về S ≥ 35 |
| F | Canny threshold cứng | 🟡 TB | Dùng Auto-Canny |
| G | Principal Axis không ổn | 🟡 TB | Thêm kiểm tra độ tin cậy moments |
| H | Không lọc hình dạng | 🟡 TB | Thêm convexity check |
| I | Thiếu blur trước CLAHE | 🟢 Thấp | Blur nhẹ (3×3) trước CLAHE |
| J | Mask nhấp nháy | 🟢 Thấp | EMA mask hoặc temporal voting |

---

## 4. Trả Lời Câu Hỏi Gốc: Pipeline Đã "Chiến Thắng Ngoại Cảnh" Chưa?

**Trả lời thành thật: Chưa hoàn toàn.**

Pipeline hiện tại hoạt động **tốt trong điều kiện phòng lab ổn định** (ánh sáng trắng cố định, nền đồng nhất, người dùng giữ tay trong khung). Nhưng sẽ gặp khó khăn khi:

| Tình huống | Kết quả dự đoán |
|------------|-----------------|
| Đổi phòng, đèn khác màu nhiệt độ | ❌ Histogram hỏng |
| Ánh sáng thay đổi trong lúc quay | ❌ Mask dần hỏng |
| Người da tối/da sáng khác người calibrate | ✅ Vẫn OK (vì calibrate theo người) |
| Nền phức tạp nhưng khác màu da | ✅ Saturation Gate xử lý được |
| Nền có màu gần tông da (gỗ vàng, tường beige) | ⚠️ Có thể bắt nhầm |
| Tay ra ngoài ROI | ❌ Mất tracking |
| Ký hiệu VSL tay xòe ngang | ⚠️ Moments không ổn định |

**Cải thiện ưu tiên nhất:** Thêm lại Gray-World White Balance (vốn đã có trong `webcam.py`) là giải pháp đơn giản nhất, ảnh hưởng lớn nhất.

---

## 5. Nguồn Tham Khảo Để Đối Chiếu & Học Hỏi

### 5.1. Code Mẫu Trực Tiếp Để So Sánh

| Repository | Điểm nổi bật để học |
|-----------|---------------------|
| [**googlesamples/mediapipe**](https://github.com/google-ai-edge/mediapipe) | Cách dùng neural hand detector + landmark thay vì rule-based |
| [**TensorFlow/hand_tracking**](https://github.com/tensorflow/tfjs-models/tree/master/hand-pose-detection) | Multi-stage pipeline: palm detector → hand landmark |
| [**kinivi/hand-gesture-recognition-mediapipe**](https://github.com/kinivi/hand-gesture-recognition-mediapipe) | Pipeline hoàn chỉnh: MediaPipe → LSTM/CNN → label, dễ so sánh |
| [**Kazuhito00/hand-gesture-recognition-using-image-processing**](https://github.com/Kazuhito00/hand-gesture-recognition-using-image-processing) | Dùng landmark + keypoint classifier — benchmark nhanh |

### 5.2. Về Skin Segmentation Nâng Cao

| Nguồn | Nội dung |
|-------|----------|
| [OpenCV Tutorial: Histogram Back Projection](https://docs.opencv.org/4.x/dc/df6/tutorial_py_histogram_backprojection.html) | Cơ sở kỹ thuật đang dùng |
| **Paper:** *"Skin Detection: A Short Tutorial"* — Vezhnevets et al. (2003) | Survey kinh điển về skin color models, so sánh RGB/HSV/YCrCb |
| **Paper:** *"Survey of skin-color modeling and detection methods"* — Kakumanu et al. (2007) | So sánh Gaussian Mixture Model vs histogram vs neural |

### 5.3. Về Image Moments & Orientation Normalization

| Nguồn | Nội dung |
|-------|----------|
| [OpenCV Moments Tutorial](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html) | Chuẩn |
| **Paper:** *"Shape Orientation using Principal Component Analysis"* | Cách dùng PCA thay vì moments — robust hơn với hình dạng phức tạp |
| **Blog:** pyimagesearch — "Image Moments with OpenCV" | Giải thích trực quan nhất |

### 5.4. Về Multi-Channel Feature Engineering

| Nguồn | Nội dung |
|-------|----------|
| **Paper:** *"Hand Gesture Recognition Using Convolutional Neural Networks"* — Molchanov et al. (2015, CVPRW) | Dùng depth + IR + RGB multi-channel, cách khai thác đặc trưng |
| **Paper:** *"Sign Language Recognition with Multi-modal Data"* | Kết hợp kênh RGB + depth + optical flow |
| [Auto-Canny by pyimagesearch](https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/) | Auto thresholding cho Canny — áp dụng ngay được |

### 5.5. Về VSL Cụ Thể

| Nguồn | Nội dung |
|-------|----------|
| *"Vietnamese Sign Language Recognition using Dynamic Object Extraction"* (IEEE 2020) | Paper nền về VSL |
| *"Vietnamese Sign Language Alphabet Recognition Using Deep Learning and MediaPipe"* (2025) | So sánh pipeline tự xây vs MediaPipe |
| [Kaggle: ASL/Sign Language Datasets](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | Dataset tham chiếu để đo lường chéo |

### 5.6. White Balance — Để Giải Quyết Vấn đề A

| Nguồn | Nội dung |
|-------|----------|
| [OpenCV Color Correction](https://docs.opencv.org/4.x/d2/de8/group__core__array.html) | Hàm normalize, convertScaleAbs |
| **Paper:** *"Gray World Assumption"* (Buchsbaum, 1980) | Lý thuyết cơ sở — rất ngắn, 2 trang |
| **Code mẫu:** Đã có trong `webcam.py` của bạn! | Copy lại và tích hợp vào `improved_dip_pipeline.py` |

---

## 6. Kế Hoạch Cải Thiện Đề Xuất (Theo Thứ Tự Ưu Tiên)

```
Ưu tiên 1 (Làm ngay — 30 phút):
  ✏️ Copy Gray-World từ webcam.py vào calibrate() và segment()
  ✏️ Đồng bộ Saturation threshold về S >= 35

Ưu tiên 2 (1-2 giờ):
  ✏️ Đổi ngưỡng area về % diện tích ROI
  ✏️ Thêm Auto-Canny thay thế threshold cứng (30, 100)
  ✏️ Blur nhẹ trước CLAHE

Ưu tiên 3 (Nếu có thời gian):
  ✏️ Thêm kiểm tra độ tin cậy moments (mu_diff threshold)
  ✏️ Thêm online histogram EMA update rất chậm (alpha = 0.005)
```

---

*Kết luận: Pipeline đã đặt nền móng đúng và có cấu trúc tốt hơn hẳn `webcam.py`. Việc bổ sung White Balance + Auto-Canny sẽ đưa nó lên mức có thể thu dữ liệu thực tế ổn định trong nhiều điều kiện ánh sáng khác nhau.*
