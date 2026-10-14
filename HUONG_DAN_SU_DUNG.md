# Hướng Dẫn Sử Dụng: Pipeline DIP Nâng Cấp cho Phân Tích Thủ Ngữ

> **TÀI LIỆU LỊCH SỬ (đối chiếu 2026-10-14) — KHÔNG phản ánh hệ thống hiện tại.**
> File này mô tả giai đoạn đầu: phân đoạn tay bằng histogram backprojection, CNN trên ảnh DIP và `dataset_builder.py`
> (nay ở `legacy/dataset_builder.py`). Các file `dnn_model.py`, `vsl_model.pth`, `vsl_dataset/splits/*.json` nhắc bên dưới không còn trong repo.
> Hệ thống hiện tại nhận diện từ bằng **landmark MediaPipe + DTW 24D**, chạy bằng `py dip.py`.
> Xem: `PROJECT.md`, `docs/HUONG_DAN_DIP.md` (cách chạy), `docs/DE_CUONG_DO_AN_DIP_ASL.md` (kiến trúc), `KIEM_TRA_TRUC_TIEP.md` (kiểm tra trực tiếp).

**Dự Án:** Nhận Diện Thủ Ngữ Tiếng Việt (Vietnamese Sign Language Recognition)  
**Giai Đoạn:** DIP (Digital Image Processing) + Deep Learning  
**Sinh Viên:** Đạt (Tô Phương Đạt)  
**Nâng Cấp:** Histogram Backprojection + Image Moments + Multi-Channel Features

---

## Tóm Tắt Nâng Cấp

### Vấn Đề của Code Cũ

| Vấn Đề | Ảnh Hưởng |
|--------|----------|
| **Ngưỡng màu da cố định** ([50,135,85]→[255,180,135]) | Bị sai khi người dùng khác, ánh sáng khác |
| **Không chuẩn hóa hình học (xoay)** | CNN phải học mọi góc xoay, tốn tài nguyên |
| **Chỉ dùng 1 kênh (grayscale)** | Thông tin quá ít, CNN khó phân biệt ký hiệu tương tự |
| **Đếm ngón tay theo rule** | Cạnh tranh với DL model, không đủ tín hiệu cho VSL |

### Giải Pháp Nâng Cấp

1. **Histogram Backprojection** → Tự động thích nghi với mỗi người dùng ✅
2. **Image Moments** → Chuẩn hóa hướng bàn tay về thẳng đứng ✅
3. **Multi-Channel Features** → 3 kênh (CLAHE, Canny, Gradient) cho CNN ✅

---

## Các File Chính

```
project/
├── improved_dip_pipeline.py      # Pipeline DIP nâng cấp (chạy realtime)
├── dataset_builder.py             # Lưu trữ dữ liệu huấn luyện
├── dnn_model.py                   # (Chuẩn bị) CNN + LSTM cho VSL
└── HUONG_DAN_SU_DUNG.md           # File này
```

---

## Bước 1: Chạy Pipeline DIP

### Cài Đặt Dependencies

```bash
pip install opencv-python numpy
```

### Chạy Realtime từ Webcam

```bash
python improved_dip_pipeline.py
```

**Quy Trình:**

1. **Giai Đoạn Calibration (30 khung đầu tiên)**
   - Đặt tay vào khung vàng trên màn hình
   - Hệ thống sẽ học histogram màu da của bạn tự động
   - Không cần chọn ngưỡng cứng!

2. **Giai Đoạn Xử Lý (sau calibration)**
   - Pipeline tự động:
     - Phân đoạn tay (histogram backprojection)
     - Chuẩn hóa hướng (image moments)
     - Trích đặc trưng (3 kênh)
   - Hiển thị 3 kênh đầu vào cho CNN

3. **Thoát:** Nhấn `q`

---

## Bước 2: Tích Hợp Với Deep Learning

### Quy Trình Huấn Luyện Dataset

```python
from improved_dip_pipeline import AdaptiveHandSegmenter, full_preprocessing_pipeline
from dataset_builder import DIPDatasetBuilder
import cv2
import numpy as np

# ===== BƯỚC 1: Xử lý video thô thành dataset =====

builder = DIPDatasetBuilder("./vsl_dataset")

# Đăng ký các ký hiệu bạn muốn nhận diện
builder.register_class("A", "Ký hiệu chữ A")
builder.register_class("B", "Ký hiệu chữ B")
builder.register_class("hello", "Lời chào")
# ... thêm các ký hiệu khác

# Xử lý video/webcam
cap = cv2.VideoCapture("video_training.mp4")  # hoặc 0 cho webcam
segmenter = AdaptiveHandSegmenter()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Calibration lần đầu (30 khung)
    if not segmenter.is_calibrated and frame_count < 30:
        segmenter.calibrate(frame)
        continue
    
    # Xử lý DIP
    multi_channel = full_preprocessing_pipeline(frame, segmenter)
    
    if multi_channel is not None:
        # Bạn tự quyết định nhãn (ký hiệu) dựa vào frame number hoặc input tay
        # Ví dụ: nếu frame 0-100 là ký hiệu "A"
        if frame_count < 100:
            class_name = "A"
        elif frame_count < 200:
            class_name = "B"
        else:
            class_name = "hello"
        
        # Lưu sample
        builder.save_sample(multi_channel, class_name)
    
    frame_count += 1

cap.release()

# Lưu metadata và tạo train/val/test split
builder.save_metadata()
builder.create_train_val_test_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

print("✓ Dataset đã tạo xong!")
```

### Tạo CNN cho Ký Hiệu Tĩnh

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np

class VSLDataset(Dataset):
    """Dataset loader từ DIP-processed data."""
    
    def __init__(self, split_file, dataset_root="./vsl_dataset"):
        with open(split_file, 'r') as f:
            data = json.load(f)
        
        self.samples = data['samples']
        self.dataset_root = dataset_root
        
        # Ánh xạ class name -> id
        self.class_to_id = {}
        for i, sample in enumerate(self.samples):
            class_name = sample['class']
            if class_name not in self.class_to_id:
                self.class_to_id[class_name] = len(self.class_to_id)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tải ảnh 3 kênh (128, 128, 3)
        img_path = self.dataset_root + "/" + sample['path']
        multi_channel = np.load(img_path).astype(np.float32) / 255.0  # Normalize
        
        # Chuyển (H, W, C) → (C, H, W) cho PyTorch
        multi_channel = np.transpose(multi_channel, (2, 0, 1))
        
        class_id = self.class_to_id[sample['class']]
        
        return torch.from_numpy(multi_channel), class_id


class SimpleCNN(nn.Module):
    """CNN đơn giản cho phân loại ký hiệu tĩnh."""
    
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Input: (3, 128, 128) - 3 kênh DIP
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
        )
        
        # Fully connected
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Huấn luyện
def train_model():
    # Load dataset
    train_dataset = VSLDataset("./vsl_dataset/splits/train_split.json")
    val_dataset = VSLDataset("./vsl_dataset/splits/val_split.json")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    num_classes = len(train_dataset.class_to_id)
    model = SimpleCNN(num_classes)
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Huấn luyện
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_accuracy:.2f}%")
    
    return model

# Chạy
model = train_model()
torch.save(model.state_dict(), "vsl_model.pth")
```

---

## Bước 3: So Sánh Với Baseline (MediaPipe)

Để tăng điểm "sáng tạo" và minh chứng pipeline của bạn tốt, hãy so sánh với MediaPipe:

```python
import mediapipe as mp
import cv2
import numpy as np

# ===== BASELINE: MediaPipe Hands =====

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_mediapipe_landmarks(frame):
    """Trích xuất 21 landmark từ MediaPipe."""
    results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        # Chuyển 21 points (x, y, z) thành 1D array 63 chiều
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
        return coords
    
    return None


# ===== SO SÁNH: Pipeline DIP vs MediaPipe =====

print("="*60)
print("SO SÁNH: Pipeline DIP Tự Xây vs MediaPipe Baseline")
print("="*60)

cap = cv2.VideoCapture("test_video.mp4")
segmenter = AdaptiveHandSegmenter()

frame_count = 0
dip_time = 0
mediapipe_time = 0

import time

while True:
    ret, frame = cap.read()
    if not ret or frame_count > 100:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Skip calibration
    if frame_count < 30:
        segmenter.calibrate(frame)
        continue
    
    # ===== Pipeline DIP =====
    t0 = time.time()
    dip_output = full_preprocessing_pipeline(frame, segmenter)
    dip_time += time.time() - t0
    
    # ===== MediaPipe =====
    t0 = time.time()
    mp_output = extract_mediapipe_landmarks(frame)
    mediapipe_time += time.time() - t0
    
    frame_count += 1

print(f"\nDIP Pipeline: {dip_time/(frame_count-30)*1000:.2f} ms/frame")
print(f"MediaPipe:    {mediapipe_time/(frame_count-30)*1000:.2f} ms/frame")
print(f"\nDIP Output:  (128, 128, 3) - Ảnh DIP đa kênh")
print(f"MediaPipe:   (63,) - Vector landmark 21 điểm x 3 tọa độ")

cap.release()
```

**Kết Quả So Sánh Để Ghi Vào Báo Cáo:**

| Tiêu Chí | Pipeline DIP | MediaPipe |
|----------|-------------|-----------|
| **Phụ thuộc mô hình sẵn** | ❌ Không (thuần DIP) | ✅ Có (Google) |
| **Tốc độ** | ~XX ms/frame | ~YY ms/frame |
| **Độ chính xác** | XX% (CNN huấn luyện) | YY% (CNN từ landmark) |
| **Học thêm kiến thức DIP** | ✅ Có nhiều | ❌ Không |
| **Độ ổn định với người dùng khác** | ✅ Tốt | ✅ Tốt |

---

## Bước 4: Viết Báo Cáo / Slide Trình Bày

### Cấu Trúc Báo Cáo

```
1. Giới Thiệu
   - Tầm quan trọng VSL
   - Thách thức hiện tại

2. Phương Pháp Đề Xuất
   2.1 Giai Đoạn DIP
       - Histogram Backprojection (thay ngưỡng cứng)
       - Image Moments (chuẩn hóa xoay)
       - Multi-Channel Features (3 kênh)
   2.2 Giai Đoạn Deep Learning
       - CNN cho ký hiệu tĩnh
       - LSTM cho ký hiệu động (nếu có)

3. Kết Quả & Đánh Giá
   - Accuracy, Precision, Recall, F1-score
   - So sánh với baseline (MediaPipe)
   - Visualize: confusion matrix, sample predictions

4. Kết Luận & Hướng Phát Triển
   - Kỹ thuật DIP nào hiệu quả nhất?
   - Vấn đề còn lại?
   - Hướng cải thiện tương lai?
```

### Hình Vẽ Minh Họa

```
[INPUT: Webcam frame] 
          ↓
[Calibration: Learn histogram]
          ↓
[Segmentation: Histogram Backprojection] → Output: Binary mask
          ↓
[Normalization: Image Moments] → Output: Rotated hand
          ↓
[Feature Extraction: 3 Channels] → Output: (128, 128, 3)
          ↓
[CNN Classifier] → Output: Sign label
          ↓
[Voting Buffer] → Output: Stable prediction
```

### Hình Vẽ: So Sánh 3 Kênh DIP

```
Input hand → Channel 0 (Grayscale+CLAHE) 
          → Channel 1 (Canny Edges)
          → Channel 2 (Gradient Magnitude)
          → Stack thành (128, 128, 3) → CNN input
```

---

## Troubleshooting

### Calibration không hoạt động (toàn đen)

**Nguyên nhân:** ROI không có tay trong 30 khung đầu  
**Cách sửa:**
```python
# Tăng max_calibration_frames
segmenter.max_calibration_frames = 50

# Hoặc điều chỉnh ROI để rộng hơn
segmenter = AdaptiveHandSegmenter(
    calibration_roi_region=(50, 450, 250, 650)  # Rộng hơn
)
```

### Segmentation mờ nhạt

**Nguyên nhân:** Histogram quá mịn (trung bình động quá mạnh)  
**Cách sửa:**
```python
# Trong phương thức calibrate(), thay:
# self.hist = 0.7 * self.hist + 0.3 * hist_current
# Thành:
self.hist = 0.9 * self.hist + 0.1 * hist_current  # Giữ nguyên lâu hơn
```

### Ảnh xoay không đúng hướng

**Nguyên nhân:** Moments tính sai góc  
**Cách sửa:** Debug bằng in góc:
```python
angle_deg = np.degrees(angle_rad)
print(f"DEBUG: angle = {angle_deg}")
# Nếu âm, thử bỏ dấu trừ: xoay(frame, -angle_deg, ...)
```

---

## Tham Khảo & Nguồn Học Tập

### Về Histogram Backprojection
- OpenCV Doc: `cv2.calcBackProject()`
- Ý tưởng: Chiếu ngược histogram thay vì dùng ngưỡng cứng
- Ứng dụng: Skin detection, color-based tracking

### Về Image Moments
- OpenCV Doc: `cv2.moments()`
- Công thức: μ = ∫∫ (x-x̄)^p (y-ȳ)^q I(x,y) dxdy
- Ứng dụng: Tìm hướng object, tâm khối lượng, bất biến hình dạng

### Về Multi-Channel Features
- CLAHE: Adaptive Histogram Equalization
- Canny: Edge detection algorithm
- Sobel: Gradient-based edge detection
- Ý tưởng: Kết hợp nhiều loại đặc trưng để CNN học tốt hơn

### Bài Báo VSL Gần Đây (Tham Khảo)
1. "Vietnamese Sign Language Alphabet Recognition Using Deep Learning and MediaPipe Methods" (2025)
2. "A Unified Hand-Landmark-Based Deep Learning Framework for Static and Dynamic VSL Recognition" (2026)
3. "Vietnamese Sign Language Recognition using Dynamic Object Extraction and Deep Learning" (IEEE 2020)

---

## Tóm Tắt: Tại Sao Cách Này Tốt Hơn

| Giai Đoạn | Cải Tiến | Lợi Ích |
|-----------|----------|---------|
| **Phân Đoạn** | Histogram thay ngưỡng cứng | ✅ Thích nghi tự động, không cần calibrate lại |
| **Chuẩn Hóa** | Image moments xoay | ✅ CNN học 1 hướng thay vì N hướng, tiết kiệm |
| **Đặc Trưng** | 3 kênh thay 1 kênh | ✅ Thông tin phong phú → CNN học tốt hơn |
| **Kiến Trúc** | DL riêng, không rule | ✅ Tách rõ vai trò, dễ maintain, dễ nâng cấp |

**→ Kết quả:** Độ chính xác cao hơn, ít bug hơn, code sạch hơn, điểm cao hơn! 🎓

---

*Chúc bạn làm dự án thành công!*
