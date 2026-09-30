"""
CÔNG CỤ ĐÓNG GÓI DỮ LIỆU & MÃ NGUỒN CỬ CHỈ ĐỘNG CHO GOOGLE COLAB
================================================================
Tác giả: Tô Phương Đạt (24110086) - Dự án DIP Project
Mục đích:
  - Tự động gom toàn bộ dữ liệu Đa Nguồn:
      + ASL Citizen 24D (52 người Mỹ)
      + Template Webcam Đạt
      + Hồ sơ toàn bộ Tình nguyện viên (Mai, Đăng, Phi, Vương, Việt)
      + Mã nguồn mô hình Deep Learning (train_dynamic_dl.py)
      + Mã nguồn kiểm thử so sánh Hybrid (eval_dynamic_hybrid.py)
      + Bộ so khớp DTW chuẩn và cấu hình G1p90
  - Đóng gói thành file zip 'vsl_dynamic_colab.zip' siêu nhẹ (~3MB).
  - Tự động sinh tệp Jupyter Notebook 'Train_Dynamic_Gesture_Colab.ipynb' sẵn sàng mở trên Colab!
"""

import os
import sys
import json
import zipfile
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def create_colab_notebook(output_path: Path):
    """
    Sinh tệp Jupyter Notebook chuyên nghiệp để chạy trên Google Colab.
    """
    notebook_content = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🇻🇳 HUẤN LUYỆN MÔ HÌNH HỌC SÂU NHẬN DIỆN CỬ CHỈ ĐỘNG (VSL / ASL)\n",
                    "**Đề tài:** Nhận diện Cử chỉ Thủ ngữ Động Bền bỉ trên Thiết bị Phổ thông (DIP Project)  \n",
                    "**Sinh viên:** Tô Phương Đạt | **MSSV:** 24110086  \n",
                    "\n",
                    "---\n",
                    "### 🌟 Điểm đột phá kiến trúc:\n",
                    "1. **Khai thác dữ liệu Đa Nguồn:** Kết hợp dữ liệu ASL Citizen (52 người Mỹ khiếm thính) + Webcam Đạt + 5 Tình nguyện viên Việt Nam.\n",
                    "2. **Kiến trúc Siêu Nhẹ:** 1D-CNN + Bi-GRU + Temporal Attention Pooling (~85K params, suy luận < 1.5ms trên CPU).\n",
                    "3. **Kiểm thử Người Lạ (Leave-One-Signer-Out - LOSO):** Chứng minh khoa học khả năng giải quyết lỗi người lạ mà DTW đơn thuần gặp phải.\n",
                    "4. **Hybrid Late Fusion:** Kết hợp sức mạnh tổng quát của Deep Learning và độ tin cậy gác cổng của DTW."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Kiểm tra môi trường GPU / CPU\n",
                    "> Khuyến nghị chọn: **Runtime -> Change runtime type -> T4 GPU** (hoặc chạy CPU vẫn cực nhanh)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!nvidia-smi"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Giải nén gói dữ liệu & mã nguồn (`vsl_dynamic_colab.zip`)\n",
                    "> **Hướng dẫn:**\n",
                    "> 1. Kéo thả file `vsl_dynamic_colab.zip` từ máy tính vào thanh tab **Files (📁)** bên trái giao diện Colab.\n",
                    "> 2. Chạy ô code bên dưới để giải nén."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!unzip -q -o vsl_dynamic_colab.zip\n",
                    "!pip install -q onnx onnxscript\n",
                    "!ls -lh"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Thử nghiệm 1: Kiểm thử Người Lạ (LOSO) trên Tình nguyện viên ĐĂNG\n",
                    "> **Mục đích:** Giấu toàn bộ mẫu của Đăng làm Test Set, chỉ huấn luyện trên ASL Citizen + Đạt + các tình nguyện viên khác."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python train_dynamic_dl.py --test_signer Dang --epochs 100 --batch_size 16 --lr 1e-3 --export_onnx\n",
                    "!python eval_dynamic_hybrid.py --test_signer Dang"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Thử nghiệm 2: Kiểm thử Người Lạ (LOSO) trên Tình nguyện viên MAI\n",
                    "> **Mục đích:** Mai có tới 100 mẫu thu thập thực tế, là bài test thử thách nhất về vóc dáng và phong cách vung tay."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python train_dynamic_dl.py --test_signer Mai --epochs 100 --batch_size 16 --lr 1e-3 --export_onnx\n",
                    "!python eval_dynamic_hybrid.py --test_signer Mai"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Huấn luyện Mô hình Hoàn chỉnh (Production Model)\n",
                    "> Huấn luyện trên **100% dữ liệu** (ASL Citizen + Đạt + Toàn bộ Tình nguyện viên) để sẵn sàng đem về máy tính chạy thực tế qua Webcam!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python train_dynamic_dl.py --test_signer none --epochs 120 --batch_size 16 --lr 1e-3 --export_onnx"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Tải Mô hình Tốt nhất về máy tính\n",
                    "> Tải 2 file `best_gesture_model_none.pth` và `best_gesture_model_none.onnx` về máy để tích hợp vào ứng dụng Webcam."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "import os\n",
                    "\n",
                    "if os.path.exists('best_gesture_model_none.pth'):\n",
                    "    files.download('best_gesture_model_none.pth')\n",
                    "if os.path.exists('best_gesture_model_none.onnx'):\n",
                    "    files.download('best_gesture_model_none.onnx')\n",
                    "print('✓ Đang gửi lệnh tải file về trình duyệt!')"
                ]
            }
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, ensure_ascii=False, indent=2)
    print(f"✓ Đã tạo thành công Notebook Colab: {output_path.name}")


def package_dynamic_project():
    project_dir = Path(__file__).resolve().parent
    zip_path = project_dir / "vsl_dynamic_colab.zip"
    nb_path = project_dir / "Train_Dynamic_Gesture_Colab.ipynb"

    print("=" * 72)
    print("  ĐÓNG GÓI DỮ LIỆU & MÃ NGUỒN CỬ CHỈ ĐỘNG CHO GOOGLE COLAB")
    print("=" * 72)

    # 1. Tạo Notebook Colab
    create_colab_notebook(nb_path)

    # 2. Danh sách các file đơn lẻ cần thêm vào zip
    files_to_add = [
        "train_dynamic_dl.py",
        "eval_dynamic_hybrid.py",
        "dynamic_dtw_engine.py",
        "scoreboard.py",
        "pipeline_common.py",
        "asl_citizen_templates_24d.json",
        "webcam_templates.json",
        "configs/G1p90.json",
        "ASL_ALEX_2.0/sign_weights_asllex.json",
        "Train_Dynamic_Gesture_Colab.ipynb"
    ]

    print(f"\n-> Đang tạo file nén: {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Thêm các file đơn lẻ
        for f_rel in files_to_add:
            full_p = project_dir / f_rel
            if full_p.exists():
                zf.write(full_p, arcname=f_rel)
                print(f"   [+] Đã thêm: {f_rel}")
            else:
                print(f"   [-] Bỏ qua (không tìm thấy): {f_rel}")

        # Thêm toàn bộ thư mục profiles/
        profiles_dir = project_dir / "profiles"
        if profiles_dir.exists():
            for p_file in profiles_dir.glob("*.json"):
                rel_p = p_file.relative_to(project_dir)
                zf.write(p_file, arcname=str(rel_p).replace("\\", "/"))
                print(f"   [+] Đã thêm hồ sơ: {rel_p}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 72)
    print(f"  HOÀN THÀNH ĐÓNG GÓI! Kích thước: {size_mb:.2f} MB")
    print(f"  File nén: {zip_path}")
    print(f"  File Notebook: {nb_path}")
    print("=" * 72)
    print("\n[HƯỚNG DẪN 3 BƯỚC CHẠY TRÊN GOOGLE COLAB]:")
    print("  1. Truy cập https://colab.research.google.com -> Chọn thẻ 'Upload' -> Tải lên file 'Train_Dynamic_Gesture_Colab.ipynb'.")
    print("  2. Trong giao diện Colab, mở biểu tượng Thư mục (Files 📁) ở mép trái -> Kéo thả file 'vsl_dynamic_colab.zip' vào.")
    print("  3. Nhấn 'Runtime' -> 'Run all' (hoặc nhấn nút Play từng ô theo thứ tự) để huấn luyện siêu tốc!")


if __name__ == "__main__":
    package_dynamic_project()
