"""
CÔNG CỤ XỬ LÝ ẢNH NGOÀI HÀNG LOẠT (EXTERNAL DATASET PROCESSOR)
==============================================================
Chức năng:
  - Đọc hàng loạt ảnh tĩnh (.jpg, .png, .jpeg) từ thư mục dữ liệu bên ngoài
    (ví dụ: ảnh tải từ Kaggle, ảnh chụp từ điện thoại).
  - Áp dụng CHÍNH XÁC 100% CÁC BƯỚC CỦA PIPELINE DIP v3:
      1. Cân bằng trắng Gray-World (khử ám sắc ảnh chụp)
      2. Tự động tìm vùng da bằng CIELab Skin Prior + Chroma Gate
      3. TỰ HIỆU CHUẨN TỨC THÌ (Per-Image Self-Calibration):
         Tính 2D Histogram (a*, b*) 64x64 bins riêng cho bàn tay của bức ảnh đó
      4. Histogram Backprojection + Otsu Thresholding + Morphology (Open/Close)
      5. Chuẩn hóa hình học bằng Image Moments:
         - Centroid (cx, cy)
         - Trục chính theta = 0.5 * atan2(2*mu11, mu20 - mu02) -> xoay về 0 độ
         - Elongation Guard + Solidity Check [0.45, 0.97]
         - Aspect Ratio Padding về (128, 128)
      6. Trích xuất Tensor 3 kênh ngữ nghĩa:
         - Kênh 0: Blur(3x3) + CLAHE
         - Kênh 1: AutoCanny (median chỉ trên vùng tay)
         - Kênh 2: Sobel Gradient Magnitude
  - Xuất ra file .npy chuẩn Tensor (128, 128, 3) lưu vào vsl_dataset/samples/<class_name>/
    để nạp thẳng vào train.py cùng với dữ liệu webcam!

Cấu trúc thư mục đầu vào mong muốn:
  ./external_raw/
     ├── A/
     │   ├── img_001.jpg
     │   └── img_002.png
     ├── B/
     │   └── ...
     └── ...

Cách chạy:
  py process_external_images.py --input_dir ./external_raw --output_dir ./vsl_dataset
  py process_external_images.py --preview  # Bật xem trước từng ảnh đã xử lý
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

# Tận dụng trực tiếp các hàm chuẩn đã kiểm thử trong improved_dip_pipeline
from improved_dip_pipeline import (
    apply_gray_world_white_balance,
    auto_canny,
    normalize_hand_orientation,
    extract_multi_channel_features
)
from dataset_builder import DIPDatasetBuilder


class StaticImageDIPProcessor:
    """
    Bộ xử lý ảnh tĩnh áp dụng đầy đủ quy trình DIP v3 (CIELab + Moments + 3ch Tensor).
    Sử dụng cơ chế Per-Image Self-Calibration (Tự hiệu chuẩn màu da riêng cho từng ảnh).
    """

    # Ngưỡng màu da CIELab chuẩn 8-bit (a*, b*)
    _SKIN_A_LOW,  _SKIN_A_HIGH = 130, 175
    _SKIN_B_LOW,  _SKIN_B_HIGH = 130, 188
    _CHROMA_MIN                = 12.0

    def _make_skin_gate(self, lab_img: np.ndarray) -> np.ndarray:
        """Lọc vùng da sơ bộ dựa trên phân phối elip CIELab và khoảng cách Chroma."""
        lower = np.array([0, self._SKIN_A_LOW, self._SKIN_B_LOW], dtype=np.uint8)
        upper = np.array([255, self._SKIN_A_HIGH, self._SKIN_B_HIGH], dtype=np.uint8)
        rect_gate = cv2.inRange(lab_img, lower, upper)

        # Chroma Gate loại trừ màu xám, trắng, đen
        a_f = lab_img[:, :, 1].astype(np.float32) - 128.0
        b_f = lab_img[:, :, 2].astype(np.float32) - 128.0
        chroma_gate = (np.sqrt(a_f**2 + b_f**2) >= self._CHROMA_MIN).astype(np.uint8) * 255

        return cv2.bitwise_and(rect_gate, chroma_gate)

    def segment_static_image(self, bgr_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Phân đoạn bàn tay trên ảnh tĩnh bằng CIELab Backprojection + Otsu:
          1. Gray-World White Balance
          2. Tìm vùng da ứng viên lớn nhất làm mẫu hiệu chuẩn (Self-Calibration)
          3. Tính 2D Histogram (a, b) 64x64 bins
          4. Backprojection + Chroma Gate + Otsu + Morphology
        """
        h, w = bgr_img.shape[:2]
        if h < 50 or w < 50:
            return None

        # 1. Cân bằng trắng Gray-World
        wb_img = apply_gray_world_white_balance(bgr_img)
        lab_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2Lab)

        # 2. Tạo mặt nạ da sơ bộ để tìm vị trí bàn tay
        skin_gate = self._make_skin_gate(lab_img)

        contours, _ = cv2.findContours(skin_gate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < (h * w * 0.02):
            # Vùng da quá nhỏ (< 2% diện tích ảnh) -> Không có bàn tay rõ ràng
            return None

        # 3. TỰ HIỆU CHUẨN (Instant Self-Calibration):
        # Lấy mẫu màu da từ chính contour bàn tay lớn nhất vừa phát hiện
        calib_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(calib_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

        hist = cv2.calcHist([lab_img], [1, 2], calib_mask, [64, 64], [0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        # 4. Chiếu ngược xác suất Backprojection
        backproj = cv2.calcBackProject([lab_img], [1, 2], hist, [0, 256, 0, 256], scale=1)

        # Kết hợp Chroma Gate
        a_f = lab_img[:, :, 1].astype(np.float32) - 128.0
        b_f = lab_img[:, :, 2].astype(np.float32) - 128.0
        chroma_gate = (np.sqrt(a_f**2 + b_f**2) >= self._CHROMA_MIN).astype(np.uint8) * 255
        backproj = cv2.bitwise_and(backproj, chroma_gate)

        # 5. Phân ngưỡng Otsu + Hình thái học Morphology
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(backproj, -1, disc, backproj)
        _, mask = cv2.threshold(backproj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    def process_image(self, bgr_img: np.ndarray, target_size=(128, 128)) -> Optional[np.ndarray]:
        """
        Chuỗi xử lý hoàn chỉnh từ ảnh BGR thô -> Tensor (128, 128, 3).
        """
        h, w = bgr_img.shape[:2]

        # Bước 1: Phân đoạn bằng CIELab
        mask = self.segment_static_image(bgr_img)
        if mask is None:
            return None

        # Bước 2: Chuẩn hóa hình học bằng Image Moments (xoay trục chính về 0 độ)
        min_hand_area = h * w * 0.03  # Ít nhất 3% diện tích ảnh
        normalized_hand, _, _ = normalize_hand_orientation(
            bgr_img, mask,
            roi_coords=(0, h, 0, w),  # ROI là toàn bộ ảnh tĩnh
            target_size=target_size,
            min_hand_area=min_hand_area
        )

        if normalized_hand is None:
            return None

        # Bước 3: Trích xuất 3 kênh đặc trưng (CLAHE + AutoCanny + Sobel)
        tensor = extract_multi_channel_features(normalized_hand)
        return tensor


def run_batch_processing(input_dir: str, output_dir: str, preview: bool = False):
    print("=" * 72)
    print("XỬ LÝ ẢNH NGOÀI HÀNG LOẠT BẰNG PIPELINE DIP v3 (CIELab + MOMENTS)")
    print("=" * 72)

    in_path = Path(input_dir)
    if not in_path.exists():
        print(f"\n[LỖI] Không tìm thấy thư mục đầu vào: {input_dir}")
        print("      Hãy tạo thư mục và chia ảnh theo cấu trúc:")
        print(f"      {input_dir}/A/*.jpg")
        print(f"      {input_dir}/B/*.png ...")
        return

    # Khởi tạo Dataset Builder và Processor
    builder = DIPDatasetBuilder(output_dir=output_dir)
    processor = StaticImageDIPProcessor()

    # Tìm các thư mục con tương ứng với tên nhãn (A, B, C, ...)
    class_dirs = sorted([d for d in in_path.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"[CẢNH BÁO] Không có thư mục con nào trong {input_dir}.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total_processed = 0
    total_success = 0
    total_failed = 0

    for c_dir in class_dirs:
        class_name = c_dir.name
        builder.register_class(class_name, description=f"Ký hiệu {class_name} từ nguồn ảnh ngoài")

        # Quét tất cả file ảnh hợp lệ
        image_files = [f for f in c_dir.iterdir() if f.suffix.lower() in valid_extensions]
        print(f"\n-> Đang xử lý nhãn [{class_name}]: {len(image_files)} ảnh...")

        for idx, img_file in enumerate(image_files, 1):
            total_processed += 1
            # Đọc ảnh an toàn với đường dẫn tiếng Việt trên Windows
            img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"   [BỎ QUA] Lỗi đọc file: {img_file.name}")
                total_failed += 1
                continue

            # Đưa qua Pipeline DIP v3
            tensor = processor.process_image(img, target_size=(128, 128))

            if tensor is None:
                # Không tìm thấy bàn tay hoặc vi phạm kiểm tra hình học
                total_failed += 1
                continue

            # Lưu vào dataset dưới dạng file .npy
            builder.save_sample(tensor, class_name)
            total_success += 1

            # Xem trước nếu bật cờ --preview
            if preview:
                ch0, ch1, ch2 = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
                vis = np.hstack([ch0, ch1, ch2])
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                cv2.putText(vis, f"{class_name}: CLAHE | CANNY | SOBEL", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.imshow("Preview DIP Tensor (128x128x3)", vis)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("\n-> Đã dừng xem trước.")
                    preview = False
                    cv2.destroyAllWindows()

            if idx % 10 == 0 or idx == len(image_files):
                print(f"   Tiến độ [{class_name}]: {idx}/{len(image_files)} (Thành công: {total_success})", end="\r")

        print()

    if preview:
        cv2.destroyAllWindows()

    # Cập nhật metadata tổng thể của dataset
    builder.save_metadata()

    print("\n" + "=" * 72)
    print("HOÀN TẤT XỬ LÝ HÀNG LOẠT DỮ LIỆU NGOÀI!")
    print(f"  Tổng số ảnh quét     : {total_processed}")
    print(f"  Thành công (Đạt DIP) : {total_success} mẫu .npy")
    print(f"  Bị loại (Nhiễu/Hỏng) : {total_failed} ảnh")
    print(f"  Dữ liệu lưu tại      : {output_dir}/samples/")
    print("=" * 72)
    print("👉 Bây giờ bạn có thể chạy 'py train.py' để huấn luyện mô hình ngay!")


def main():
    parser = argparse.ArgumentParser(
        description="Chuyen doi anh ngoai thanh Tensor DIP v3 (128x128x3) luu vao dataset"
    )
    parser.add_argument("--input_dir", type=str, default="./external_raw",
                        help="Thu muc chua anh tho (chia theo subfolder A, B, C...)")
    parser.add_argument("--output_dir", type=str, default="./vsl_dataset",
                        help="Thu muc dataset chung voi webcam")
    parser.add_argument("--preview", action="store_true",
                        help="Bat cua so OpenCV xem truoc ket qua DIP tung anh")

    args = parser.parse_args()
    run_batch_processing(args.input_dir, args.output_dir, args.preview)


if __name__ == "__main__":
    main()
