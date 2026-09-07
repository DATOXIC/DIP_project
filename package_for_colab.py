"""
CÔNG CỤ ĐÓNG GÓI DỰ ÁN ĐỂ HUẤN LUYỆN TRÊN GOOGLE COLAB
======================================================
Tác giả: Tô Phương Đạt (24110086)
Mục đích:
  - Quét toàn bộ mã nguồn cần thiết và thư mục vsl_dataset/ (các file .npy).
  - Tự động nén thành file 'vsl_colab_package.zip' gọn nhẹ (bỏ qua __pycache__, .git, models).
  - Xuất ra mã lệnh sẵn sàng dán vào Google Colab để chạy train GPU siêu tốc!

Cách dùng:
  py package_for_colab.py
"""

import os
import sys
import zipfile
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def create_colab_package():
    project_dir = Path(__file__).resolve().parent
    zip_filename = project_dir / "vsl_colab_package.zip"

    # Danh sách các file code cốt lõi cần tải lên Colab
    code_files = [
        "train.py",
        "improved_dip_pipeline.py",
        "dataset_builder.py",
        "process_external_images.py",
        "VSL_Training_Colab.ipynb",
    ]

    print("=" * 70)
    print("  ĐÓNG GÓI DỮ LIỆU & MÃ NGUỒN CHO GOOGLE COLAB")
    print("=" * 70)

    dataset_dir = project_dir / "vsl_dataset"
    if not dataset_dir.exists():
        print(f"[CẢNH BÁO] Không tìm thấy thư mục {dataset_dir}")
        return

    # Thống kê mẫu dữ liệu trước khi nén
    samples_dir = dataset_dir / "samples"
    sample_stats = {}
    total_npy = 0
    if samples_dir.exists():
        for c_dir in sorted(samples_dir.iterdir()):
            if c_dir.is_dir():
                count = len(list(c_dir.glob("*.npy")))
                sample_stats[c_dir.name] = count
                total_npy += count

    print(f"-> Thống kê mẫu .npy hiện có ({total_npy} mẫu):")
    for c_name, c_cnt in sample_stats.items():
        print(f"   + Lớp [{c_name}]: {c_cnt} mẫu")

    if total_npy == 0:
        print("\n[LƯU Ý] Chưa có mẫu .npy nào trong vsl_dataset/samples/!")
        print("  Nếu bạn có ảnh ngoài (Kaggle), hãy chạy:")
        print("    py process_external_images.py --input_dir <duong_dan_anh> --flip_horizontal")
        print("  Hoặc nếu nạp ảnh Kaggle trực tiếp trên Colab thì có thể upload file này lên.")

    print(f"\n-> Đang tạo file nén: {zip_filename.name} ...")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Thêm các file Python cốt lõi
        for cf in code_files:
            p = project_dir / cf
            if p.exists():
                zf.write(p, arcname=cf)
                print(f"   [+] Đã thêm code : {cf}")
            else:
                print(f"   [-] Bỏ qua (không thấy): {cf}")

        # 2. Thêm thư mục vsl_dataset/ (loại bỏ models cũ để zip nhẹ hơn)
        for root, dirs, files in os.walk(dataset_dir):
            dirs[:] = [d for d in dirs if d not in ["models", "__pycache__"]]
            for f in files:
                if f.endswith(".pth") or f.endswith(".pt"):
                    continue
                full_p = Path(root) / f
                rel_p = full_p.relative_to(project_dir)
                zf.write(full_p, arcname=str(rel_p).replace("\\", "/"))

    size_mb = zip_filename.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f"  HOÀN THÀNH ĐÓNG GÓI! Kích thước: {size_mb:.2f} MB")
    print(f"  File nén lưu tại: {zip_filename}")
    print("=" * 70)

    print("\n[HƯỚNG DẪN CHẠY TRÊN GOOGLE COLAB]:")
    print("1. Mở Google Colab (chọn Runtime -> Change runtime type -> T4 GPU).")
    print("2. Tải file 'vsl_colab_package.zip' lên giao diện Files của Colab.")
    print("3. Tạo 1 ô code và dán lệnh sau để giải nén và train:")
    print("   ------------------------------------------------------------------")
    print("   !unzip -q vsl_colab_package.zip")
    print("   !python train.py --epochs 80 --batch_size 32 --lr 1e-3 --flip_kaggle")
    print("   ------------------------------------------------------------------")
    print("4. Sau khi train xong:")
    print("   - Tải file 'vsl_dataset/models/best_model.pth' về máy tính.")
    print("   - Copy vào thư mục: DIP_project/vsl_dataset/models/best_model.pth")
    print("   - Chạy 'py predict_webcam.py' để kiểm thử nhận diện thời gian thực!\n")


if __name__ == "__main__":
    create_colab_package()
