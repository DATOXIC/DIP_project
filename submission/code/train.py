"""
HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN THỦ NGỮ VSL — TWO-STREAM CNN
===========================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Kiến trúc mô hình: Two-Stream CNN
  Input:   Tensor (128, 128, 3) từ improved_dip_pipeline.py v3
           Kênh 0: CLAHE           → Luồng 1 (Texture Stream)
           Kênh 1: AutoCanny       → Luồng 2 (Shape Stream)
           Kênh 2: Sobel Gradient  → Luồng 2 (Shape Stream)

  Tại sao Two-Stream tốt hơn Single-Stream:
  - Kênh CLAHE (liên tục 0-255) và Canny (nhị phân 0/255) có phân phối dữ liệu
    hoàn toàn khác nhau → Nếu gom vào 1 mạng, Conv Layer đầu phải "cân bằng"
    giữa 2 loại đặc trưng quá khác biệt → khó tối ưu.
  - Two-Stream: mỗi nhánh chuyên biệt → kết hợp (Fusion) ở tầng cuối.

Online Augmentation — Source-Aware 2 lớp (Kaggle vs Webcam):
  Mọi mẫu (Kaggle + Webcam) — Biến đổi Hình học (VSL-Safe, KHÔNG flip ngang):
    - Random Rotation ±8°       → Mô phỏng tay nghiêng nhẹ
    - Random Translate ±4%      → Mô phỏng tay lệch tâm ROI
    - Random Scale 0.90–1.10    → Mô phỏng khoảng cách camera thay đổi

  Chỉ mẫu Kaggle/Studio — Physics-Based Degradation (RealisticWebcamAugmenter):
    (Ảnh webcam thật được bỏ qua lớp này để tránh suy thoái kép - Double Degradation)
    CẤP ĐỘ 1 (100%): Nhiễu Gaussian nhẹ | Dịch gain từng kênh | Banding Noise
    CẤP ĐỘ 2 ( 50%): Specular Highlight | Thiếu sáng+Nhiễu | Motion Blur |
                      Defocus Blur | JPEG Artifact | Bóng cứng một hướng
    CẤP ĐỘ 3 ( 20%): Ngược sáng Backlight | Nhấp nháy Fluorescent |
                      Méo thấu kính Barrel | Hắt sáng nền màu | Dịch màu da

Cách chạy:
  py train.py                              # Cài đặt mặc định
  py train.py --epochs 80 --batch_size 8   # Tuỳ chỉnh
  py train.py --data_dir ./vsl_dataset --lr 5e-4
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Kiểm tra PyTorch trước khi import để báo lỗi rõ ràng
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("[LOI] Chua cai PyTorch!")
    print("      Chay lenh: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

import numpy as np


# =============================================================================
# SECTION A: ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huan luyen Two-Stream CNN cho Thu Ngu VSL (DIP v3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",    type=str,   default="./vsl_dataset",
                        help="Thu muc chua dataset (chua thu muc samples/)")
    parser.add_argument("--model_dir",   type=str,   default="./vsl_dataset/models",
                        help="Thu muc luu model va ket qua")
    parser.add_argument("--epochs",      type=int,   default=60,
                        help="So epoch huan luyen toi da")
    parser.add_argument("--batch_size",  type=int,   default=16,
                        help="Kich thuoc batch")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate ban dau")
    parser.add_argument("--val_ratio",   type=float, default=0.15,
                        help="Ti le mau danh cho Validation (0.0 - 0.4)")
    parser.add_argument("--test_ratio",  type=float, default=0.15,
                        help="Ti le mau danh cho Test (0.0 - 0.4)")
    parser.add_argument("--patience",    type=int,   default=15,
                        help="Early Stopping: so epoch cho neu Val Acc khong tang")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed de tai lap ket qua")
    parser.add_argument("--flip_kaggle", dest="flip_kaggle", action="store_true", default=True,
                        help="Tu dong lat ngang (Horizontal Flip) cac mau Kaggle/anh ngoai de dong bo huong tay voi Webcam")
    parser.add_argument("--no_flip_kaggle", dest="flip_kaggle", action="store_false",
                        help="Khong lat ngang mau Kaggle (giu nguyen huong goc)")
    parser.add_argument("--num_workers", type=int,   default=2 if sys.platform != "win32" else 0,
                        help="So DataLoader workers (tu dong chon 2 tren Colab/Linux, 0 tren Windows)")
    parser.add_argument("--class_weight", dest="class_weight", action="store_true", default=True,
                        help="Can bang loss theo nghich dao tan suat lop (chong thien vi lop nhieu mau, VD B=155 vs C=64/V=54)")
    parser.add_argument("--no_class_weight", dest="class_weight", action="store_false",
                        help="Tat can bang loss, dung CrossEntropyLoss thuong (hanh vi cu)")
    return parser.parse_args()


# =============================================================================
# SECTION B: DATASET & ONLINE AUGMENTATION
# =============================================================================

class VSLDataset(Dataset):
    """
    Dataset doc file .npy tu vsl_dataset/samples/<class_name>/*.npy.
    Ap dung Online Augmentation VSL-Safe trong qua trinh huan luyen.

    Args:
        samples:  List[(filepath: str, class_id: int)]
        is_train: True → bat Augmentation | False → giu nguyen (cho Val/Test)
    """

    # Tham so Augmentation VSL-Safe (khong co flip ngang)
    _ROT_DEG      = 8       # ±8 do
    _TRANS_FRAC   = 0.04    # ±4% chieu rong/cao anh
    _SCALE_LOW    = 0.95
    _SCALE_HIGH   = 1.05
    _NOISE_STD    = 0.02    # Do lech chuan nhieu Gaussian
    _NOISE_PROB   = 0.5     # 50% xac suat them nhieu

    def __init__(self, samples: List[Tuple[str, int]], is_train: bool = True, flip_kaggle: bool = True) -> None:
        if not samples:
            raise ValueError("Danh sach samples khong duoc rong!")
        self.samples     = samples
        self.is_train    = is_train
        self.flip_kaggle = flip_kaggle

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]

        try:
            arr = np.load(filepath)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"Khong doc duoc file: {filepath} | Loi: {e}") from e

        # Normalize [0, 255] → [0.0, 1.0], chuyen sang (3, 128, 128)
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Dong bo toa do: Lat ngang cac mau tu Kaggle/anh ngoai de trung khop 100% huong tay voi Webcam
        if self.flip_kaggle and not self._is_webcam_sample(filepath) and not self._is_already_flipped(filepath):
            tensor = torch.flip(tensor, dims=[-1])

        if self.is_train:
            tensor = self._augment(tensor, filepath=filepath)

        return tensor, label

    def _augment(self, t: torch.Tensor, filepath: str = "") -> torch.Tensor:
        """
        Chuỗi biến đổi Source-Aware Augmentation an toàn cho Thủ Ngữ:
          - Mọi mẫu (Kaggle & Webcam): Đều áp dụng biến đổi hình học (Affine: Xoay ±8°, Dịch ±4%, Scale 0.90-1.10)
            để mô phỏng góc tay và cự ly khác nhau.
          - Mẫu Kaggle / Ảnh ngoài: Áp dụng Physics-Based Degradation (_WEBCAM_AUG) để giả lập môi trường webcam.
          - Mẫu Webcam thật: BỎ QUA _WEBCAM_AUG vì ảnh đã có sẵn nhiễu và tật xấu phần cứng thật,
            tránh hiện tượng suy thoái kép (Double Degradation) làm hỏng đặc trưng.
        """
        # --- 1. Affine Geometric Transforms (Áp dụng cho CẢ HAI nguồn, VSL-Safe, no flip) ---
        angle = random.uniform(-8.0, 8.0)
        t = _affine_transform(t,
            cos_a=math.cos(math.radians(angle)),
            sin_a=math.sin(math.radians(angle)),
            tx=0.0, ty=0.0, scale=1.0)

        tx = random.uniform(-0.04, 0.04)
        ty = random.uniform(-0.04, 0.04)
        t  = _affine_transform(t, cos_a=1.0, sin_a=0.0, tx=tx, ty=ty, scale=1.0)

        scale = random.uniform(0.90, 1.10)
        t = _affine_transform(t, cos_a=1.0, sin_a=0.0, tx=0.0, ty=0.0, scale=scale)

        # --- 2. Source-Aware Physics Augmentation ---
        is_webcam = self._is_webcam_sample(filepath)
        if not is_webcam:
            # Ảnh Kaggle/Studio: Giả lập môi trường webcam bằng Physics Degradation
            t = _WEBCAM_AUG(t)
        # Nếu là webcam thật: Không áp dụng _WEBCAM_AUG (bảo vệ chất lượng quang học thực tế)

        return t

    @staticmethod
    def _is_webcam_sample(filepath: str) -> bool:
        """
        Xác định mẫu có phải từ webcam thực tế hay không.
        Quy ước: tên file bắt đầu bằng 'webcam' hoặc đường dẫn chứa 'webcam'.
        """
        if not filepath:
            return False
        p_str = filepath.lower().replace("\\", "/")
        fname = Path(filepath).name.lower()
        return "webcam" in fname or "/webcam" in p_str or fname.startswith("cam_")

    @staticmethod
    def _is_already_flipped(filepath: str) -> bool:
        """
        Kiem tra xem mau da duoc lat ngang truoc do chua (tranh hien tuong lat 2 lan).
        Quy uoc: ten file hoac duong dan chua 'flipped' hoac '_flip'.
        """
        if not filepath:
            return False
        fname = Path(filepath).name.lower()
        return "flipped" in fname or "_flip" in fname


# =============================================================================
# PHYSICS-BASED WEBCAM AUGMENTER (Mô phỏng toàn bộ suy giảm quang học thực tế)
# =============================================================================

class RealisticWebcamAugmenter:
    """
    Mô phỏng 18 hiện tượng suy giảm quang học/cảm biến có thật của webcam laptop giá rẻ
    thành 3 cấp độ tăng cường theo xác suất, đưa vào Online Augmentation.

    CẤP ĐỘ 1 — Luôn áp dụng (100%): Nhiễu nhẹ, thay đổi màu nhẹ, scale nhẹ.
    CẤP ĐỘ 2 — Ngẫu nhiên 50%: Cháy sáng, nhòe chuyển động, nén JPEG, thiếu sáng.
    CẤP ĐỘ 3 — Ngẫu nhiên 20%: Ngược sáng, bóng cứng, phối cảnh, dịch màu da.

    Lưu ý quan trọng:
        - Input tensor: (3, H, W) float32 trong [0, 1].
          Kênh 0 = CLAHE (Texture), Kênh 1 = AutoCanny (Shape), Kênh 2 = Sobel (Shape).
        - Cấp 1 và 2 tác động cả 3 kênh để mô phỏng tác động toàn bộ DIP pipeline.
        - Augmentation KHÔNG flip ngang (bảo toàn tính đúng đắn tay trái/phải VSL).
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Áp dụng tuần tự 3 cấp độ augmentation."""
        t = self._level1_always(t)
        t = self._level2_half(t)
        t = self._level3_rare(t)
        return torch.clamp(t, 0.0, 1.0)

    # ------------------------------------------------------------------
    # CẤP ĐỘ 1: Luôn áp dụng — biến động nền tảng (100% xác suất)
    # ------------------------------------------------------------------
    def _level1_always(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 1:  Nhiễu Gaussian nhẹ (Thermal Read Noise) — luôn có trên webcam.
        Case 3:  Dịch nhiệt độ màu ấm/lạnh nhẹ (Color Temp Jitter ±400K).
        Case 14: Biến động khoảng cách nhỏ (Scale Jitter) đã xử lý ở _augment().
        Case 17: Nhiễu sọc ngang cực nhỏ (Banding Noise) — luôn tồn tại ở mức thấp.
        """
        # Case 1: Thermal Gaussian Noise nhẹ (luôn có)
        std = random.uniform(0.005, 0.018)
        noise = torch.randn_like(t) * std
        t = t + noise

        # Case 3 & 10: Color Temperature Shift ±400K
        # BGR channel scaling: warm shift (R↑ G≈ B↓) hoặc cool shift (R↓ G≈ B↑)
        # Vì kênh đầu ra là CLAHE/Canny/Sobel (xám), ta dịch độ sáng không đều:
        ch_scales = torch.ones(3)
        # Mô phỏng camera auto-gain phản ứng không đều giữa các kênh
        ch_scales[0] *= random.uniform(0.95, 1.05)   # CLAHE channel
        ch_scales[1] *= random.uniform(0.95, 1.05)   # Canny channel
        ch_scales[2] *= random.uniform(0.95, 1.05)   # Sobel channel
        t = t * ch_scales.view(3, 1, 1)

        # Case 17: Banding Noise — sọc ngang ở mức cực nhỏ (luôn có ở webcam USB)
        if random.random() < 0.3:
            H = t.shape[1]
            band_rows = random.randint(1, 3)
            for _ in range(band_rows):
                row = random.randint(0, H - 1)
                t[:, row, :] *= random.uniform(0.88, 1.0)

        return t

    # ------------------------------------------------------------------
    # CẤP ĐỘ 2: Ngẫu nhiên 50% — suy giảm vừa phải thường gặp
    # ------------------------------------------------------------------
    def _level2_half(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() >= 0.5:
            return t

        # Chọn ngẫu nhiên 1 trong 6 hiện tượng cấp 2 (KHÔNG kết hợp nhiều cùng lúc
        # để tránh tạo ra ảnh quá phi thực tế):
        choice = random.randint(0, 5)

        if choice == 0:
            # Case 1: Specular Highlight — đốm sáng đèn trần dội vào đầu ngón tay
            t = self._specular_highlight(t)
        elif choice == 1:
            # Case 2 & 2b: Global Underexposure + Gaussian Noise (thiếu sáng + muỗi hạt)
            t = self._low_light_noise(t)
        elif choice == 2:
            # Case 5: Motion Blur — nhòe chuyển động theo 1 hướng ngẫu nhiên
            t = self._motion_blur(t)
        elif choice == 3:
            # Case 7: Defocus Blur — mờ ngoài tiêu cự
            t = self._defocus_blur(t)
        elif choice == 4:
            # Case 8: JPEG Compression Artifact — nhiễu ô vuông nén
            t = self._jpeg_artifact(t)
        elif choice == 5:
            # Case 6: Directional Gradient Lighting — đèn đơn một hướng, tạo bóng
            t = self._directional_shadow(t)

        return t

    # ------------------------------------------------------------------
    # CẤP ĐỘ 3: Ngẫu nhiên 20% — tình huống khắc nghiệt bất thường
    # ------------------------------------------------------------------
    def _level3_rare(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() >= 0.20:
            return t

        choice = random.randint(0, 4)

        if choice == 0:
            # Case 4: Backlight Silhouette — ngược sáng cửa sổ
            t = self._backlight(t)
        elif choice == 1:
            # Case 9: Fluorescent Flicker — nhấp nháy đèn huỳnh quang
            t = self._flicker(t)
        elif choice == 2:
            # Case 11: Lens Barrel Distortion — méo thấu kính
            t = self._lens_distortion(t)
        elif choice == 3:
            # Case 12: Inter-reflection — hắt sáng từ nền màu
            t = self._inter_reflection(t)
        elif choice == 4:
            # Case 18: Skin Tone Shift — thay đổi màu da (sau tập thể thao, trời lạnh)
            t = self._skin_tone_shift(t)

        return t

    # ============================================================
    # Các hàm mô phỏng hiện tượng cụ thể
    # ============================================================

    def _specular_highlight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 1: Cháy sáng đèn trần (Specular Highlight).
        Tạo đốm sáng Gaussian 2D tâm nằm ở 1/3 trên của ảnh (vùng đầu ngón tay).
        Công thức: G(x,y) = A * exp(-(dx²+dy²) / (2σ²))
        """
        _, H, W = t.shape
        # Tâm đốm sáng nằm ngẫu nhiên ở phần trên 40% của ảnh
        cx = random.randint(int(W * 0.2), int(W * 0.8))
        cy = random.randint(0, int(H * 0.4))
        sigma = random.uniform(H * 0.05, H * 0.18)
        amplitude = random.uniform(0.25, 0.55)

        y_grid, x_grid = np.mgrid[0:H, 0:W]
        gaussian = amplitude * np.exp(
            -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2)
        )
        g_tensor = torch.from_numpy(gaussian.astype(np.float32)).unsqueeze(0)  # (1,H,W)
        t = t + g_tensor.expand_as(t)
        return t

    def _low_light_noise(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 2: Thiếu sáng toàn cục + Nhiễu Poisson/Gaussian (Low-Light + Sensor Noise).
        Công thức: I_noisy = I * alpha + N(0, sigma²) với alpha ∈ [0.4, 0.7].
        Muỗi hạt mạnh hơn ở vùng tối (sigma tăng khi alpha giảm).
        """
        alpha = random.uniform(0.40, 0.70)
        # Sigma tỷ lệ nghịch với độ sáng (thiếu sáng → gain cao → nhiễu nhiều)
        sigma = (1.0 - alpha) * random.uniform(0.08, 0.18)
        t = t * alpha
        noise = torch.randn_like(t) * sigma
        # Case 2b: Muối tiêu (Salt-and-Pepper) xác suất thấp
        if random.random() < 0.3:
            salt_mask = torch.rand_like(t) < 0.005
            pepper_mask = torch.rand_like(t) < 0.005
            t[salt_mask] = 1.0
            t[pepper_mask] = 0.0
        return t + noise

    def _motion_blur(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 5: Nhòe chuyển động (Motion Blur).
        Xây dựng kernel đường thẳng k×k theo góc theta ngẫu nhiên.
        Convolution 2D trên từng kênh: I_blur = I * h(theta, k).
        """
        k = random.choice([5, 7, 9])
        angle_rad = random.uniform(0, math.pi)
        kernel = np.zeros((k, k), dtype=np.float32)
        cx_k, cy_k = k // 2, k // 2
        for i in range(k):
            offset = i - cx_k
            x = cx_k + round(offset * math.cos(angle_rad))
            y = cy_k + round(offset * math.sin(angle_rad))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        s = kernel.sum()
        if s > 0:
            kernel /= s
        kernel_t = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

        blurred = torch.zeros_like(t)
        pad = k // 2
        for c in range(t.shape[0]):
            ch = t[c].unsqueeze(0).unsqueeze(0)
            import torch.nn.functional as F_func
            blurred[c] = F_func.conv2d(ch, kernel_t, padding=pad).squeeze()
        return blurred

    def _defocus_blur(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 7: Mờ ngoài tiêu cự (Defocus Blur).
        Kernel đĩa tròn (Circle of Confusion) bán kính r pixel.
        h_defocus(x,y) = 1/(π*r²) nếu x²+y² ≤ r², ngược lại = 0.
        """
        r = random.randint(1, 3)
        k = 2 * r + 1
        y_g, x_g = np.ogrid[-r:r+1, -r:r+1]
        disk = (x_g**2 + y_g**2 <= r**2).astype(np.float32)
        disk /= disk.sum()
        kernel_t = torch.from_numpy(disk).unsqueeze(0).unsqueeze(0)
        blurred = torch.zeros_like(t)
        pad = r
        for c in range(t.shape[0]):
            ch = t[c].unsqueeze(0).unsqueeze(0)
            import torch.nn.functional as F_func
            blurred[c] = F_func.conv2d(ch, kernel_t, padding=pad).squeeze()
        return blurred

    def _jpeg_artifact(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 8: JPEG Compression Artifact (Block boundary + Ringing).
        Mô phỏng bằng cách thêm nhiễu dạng ô vuông 8×8 (block artifact pattern).
        """
        _, H, W = t.shape
        block_noise = np.zeros((H, W), dtype=np.float32)
        block_size = 8
        for by in range(0, H, block_size):
            for bx in range(0, W, block_size):
                val = random.uniform(-0.04, 0.04)
                block_noise[by:by+block_size, bx:bx+block_size] = val
        # Gaussian smoothing nhẹ (Ringing artifact)
        noise_t = torch.from_numpy(block_noise).unsqueeze(0)
        return t + noise_t.expand_as(t)

    def _directional_shadow(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 6: Bóng đổ một hướng (Directional/Hard Shadow).
        Tạo gradient sáng tối (Linear Ramp) theo hướng ngẫu nhiên.
        """
        _, H, W = t.shape
        direction = random.choice(['left', 'right', 'top', 'bottom', 'diagonal'])
        ramp = np.ones((H, W), dtype=np.float32)

        dark_fac = random.uniform(0.45, 0.75)
        if direction == 'left':
            ramp = np.linspace(dark_fac, 1.0, W).reshape(1, W) * np.ones((H, 1))
        elif direction == 'right':
            ramp = np.linspace(1.0, dark_fac, W).reshape(1, W) * np.ones((H, 1))
        elif direction == 'top':
            ramp = np.linspace(dark_fac, 1.0, H).reshape(H, 1) * np.ones((1, W))
        elif direction == 'bottom':
            ramp = np.linspace(1.0, dark_fac, H).reshape(H, 1) * np.ones((1, W))
        else:  # diagonal
            ramp_h = np.linspace(dark_fac, 1.0, H).reshape(H, 1)
            ramp_w = np.linspace(dark_fac, 1.0, W).reshape(1, W)
            ramp = (ramp_h + ramp_w) / 2.0

        ramp_t = torch.from_numpy(ramp.astype(np.float32)).unsqueeze(0)
        return t * ramp_t.expand_as(t)

    def _backlight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 4: Ngược sáng (Backlight/Counter-Illumination).
        Camera overexpose nền → hạ thấp toàn bộ ảnh như bị silhouette.
        Vẫn giữ lại một chút đặc trưng biên để CNN không bị blind hoàn toàn.
        """
        alpha = random.uniform(0.20, 0.45)
        return t * alpha

    def _flicker(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 9: Nhấp nháy đèn huỳnh quang (50/60Hz Fluorescent Flicker).
        Mô phỏng frame bị tối đột ngột do không đồng bộ tốc độ màn trập & tần số AC.
        """
        dark_factor = random.uniform(0.50, 0.80)
        return t * dark_factor

    def _lens_distortion(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 11: Méo thấu kính Barrel (Barrel Distortion).
        Công thức méo hướng tâm: r_d = r_u * (1 + k1*r_u² + k2*r_u⁴)
        Mô phỏng bằng lưới lấy mẫu biến dạng (Distortion Grid + grid_sample).
        """
        _, H, W = t.shape
        k1 = random.uniform(-0.10, 0.10)

        y_norm, x_norm = np.mgrid[-1:1:complex(H), -1:1:complex(W)]
        r_sq = x_norm**2 + y_norm**2
        distort_factor = 1.0 + k1 * r_sq
        x_dist = (x_norm * distort_factor).astype(np.float32)
        y_dist = (y_norm * distort_factor).astype(np.float32)

        grid = np.stack([x_dist, y_dist], axis=-1)
        grid_t = torch.from_numpy(grid).unsqueeze(0)  # (1, H, W, 2)

        import torch.nn.functional as F_func
        out = F_func.grid_sample(
            t.unsqueeze(0), grid_t,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        return out.squeeze(0)

    def _inter_reflection(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 12: Hắt sáng từ nền màu (Inter-Reflection / Color Bleed).
        Thêm một vùng gradient màu mờ nhẹ từ một góc của ảnh.
        """
        _, H, W = t.shape
        corner = random.choice(['tl', 'tr', 'bl', 'br'])
        intensity = random.uniform(0.06, 0.15)
        # Chọn kênh bị ảnh hưởng (mô phỏng màu của nền: chỉ CLAHE bị ảnh hưởng)
        channel = 0  # CLAHE texture channel
        ramp_h = np.linspace(1.0, 0.0, H).astype(np.float32).reshape(H, 1)
        ramp_w = np.linspace(1.0, 0.0, W).astype(np.float32).reshape(1, W)
        if corner == 'tl':
            overlay = ramp_h * ramp_w
        elif corner == 'tr':
            overlay = ramp_h * ramp_w[:, ::-1]
        elif corner == 'bl':
            overlay = ramp_h[::-1] * ramp_w
        else:
            overlay = ramp_h[::-1] * ramp_w[:, ::-1]
        overlay = np.ascontiguousarray(overlay)
        overlay_t = torch.from_numpy(overlay) * intensity
        t[channel] = t[channel] + overlay_t
        return t

    def _skin_tone_shift(self, t: torch.Tensor) -> torch.Tensor:
        """
        Case 18: Thay đổi màu da (Skin Tone Shift — sau tập thể dục hoặc trời lạnh).
        Kênh CLAHE sẽ sáng hơn (da đỏ/tươi) hoặc tối hơn (da xanh tái) một chút.
        Ảnh hưởng chủ yếu đến kênh CLAHE (Texture), ít ảnh hưởng kênh Canny/Sobel.
        """
        shift = random.uniform(-0.12, 0.15)  # Âm = da xanh tái, Dương = da đỏ ấm
        t[0] = t[0] + shift  # Chỉ kênh CLAHE
        return t


# Singleton instance — tạo một lần, dùng cho toàn bộ quá trình train
_WEBCAM_AUG = RealisticWebcamAugmenter()


def _affine_transform(
    tensor: torch.Tensor,
    cos_a: float, sin_a: float,
    tx: float, ty: float,
    scale: float,
) -> torch.Tensor:
    """
    Ap dung bien doi affine len tensor (C, H, W) bang grid_sample.
    Khong dung torchvision de giam dependency.

    Ma tran affine 2x3 tuong duong:
      | scale*cos  -scale*sin   tx |
      | scale*sin   scale*cos   ty |
    """
    s = scale
    theta = torch.tensor(
        [[s * cos_a, -s * sin_a, tx],
         [s * sin_a,  s * cos_a, ty]],
        dtype=torch.float32,
    )
    grid = torch.nn.functional.affine_grid(
        theta.unsqueeze(0),
        tensor.unsqueeze(0).shape,
        align_corners=False,
    )
    out = torch.nn.functional.grid_sample(
        tensor.unsqueeze(0), grid,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return out.squeeze(0)


# =============================================================================
# SECTION C: TWO-STREAM CNN ARCHITECTURE
# =============================================================================

class ConvBlock(nn.Module):
    """
    Khoi tich chap co ban: Conv2d → BatchNorm → ReLU → (MaxPool tuy chon).
    BatchNorm giup on dinh gradient, day nhanh qua trinh hoi tu.
    """
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TextureStream(nn.Module):
    """
    Luong 1: Hoc dac trung Ket cau (Texture) tu kenh CLAHE.
    Input:  (B, 1, 128, 128) — 1 kenh CLAHE
    Output: Vector (B, 256)

    Chuyen mon hoa vao: Do min long ban tay, nep nhan, tuong phan cuc bo.
    """
    OUT_DIM = 64 * 2 * 2  # = 256

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1,  16, pool=True),    # (B,16,64,64)
            ConvBlock(16, 32, pool=True),    # (B,32,32,32)
            ConvBlock(32, 64, pool=True),    # (B,64,16,16)
            nn.AdaptiveAvgPool2d((2, 2)),    # (B,64,2,2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)        # (B, 256)


class ShapeStream(nn.Module):
    """
    Luong 2: Hoc dac trung Hinh hoc (Shape) tu kenh AutoCanny + Sobel.
    Input:  (B, 2, 128, 128) — 2 kenh [Canny, Sobel]
    Output: Vector (B, 256)

    Chuyen mon hoa vao: Duong vien ngon tay, goc gap dot ngon, ke ho giua ngon.
    """
    OUT_DIM = 64 * 2 * 2  # = 256

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(2,  16, pool=True),    # (B,16,64,64)
            ConvBlock(16, 32, pool=True),    # (B,32,32,32)
            ConvBlock(32, 64, pool=True),    # (B,64,16,16)
            nn.AdaptiveAvgPool2d((2, 2)),    # (B,64,2,2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)        # (B, 256)


class TwoStreamHandCNN(nn.Module):
    """
    Kien truc Two-Stream CNN cho Nhan dien Thu Ngu VSL.

    Dong chay du lieu:
      x (B,3,128,128)
        ├─ x[:,0:1]  → TextureStream → feat_tex (B,256)
        └─ x[:,1:3]  → ShapeStream  → feat_shp (B,256)
                          concat → (B,512) → FC(128) → Dropout(0.4) → FC(num_classes)

    Args:
        num_classes: So luong chu cai/ky hieu can phan loai.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes phai >= 2, nhan duoc: {num_classes}")

        self.texture_stream = TextureStream()
        self.shape_stream   = ShapeStream()
        fusion_dim = self.texture_stream.OUT_DIM + self.shape_stream.OUT_DIM  # 512

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128) — tensor tu DIP pipeline v3
        Returns:
            logits: (B, num_classes)
        """
        feat_tex = self.texture_stream(x[:, 0:1, :, :])   # (B, 256)
        feat_shp = self.shape_stream(x[:, 1:3, :, :])     # (B, 256)
        fused    = torch.cat([feat_tex, feat_shp], dim=1)  # (B, 512)
        return self.classifier(fused)                       # (B, num_classes)


# =============================================================================
# SECTION D: DATA LOADING UTILITIES
# =============================================================================

def scan_dataset(data_dir: str) -> Tuple[List[Tuple[str, int]], Dict[int, str]]:
    """
    Quet thu muc samples/ va tra ve danh sach (filepath, class_id) + bang nhan.

    Thu muc phai co cu truc:
        <data_dir>/samples/<ten_chu_cai>/*.npy

    Returns:
        all_samples: List[(filepath, class_id)]  — sap xep theo class
        id_to_label: Dict[class_id → ten_class]  — vi du {0:'A', 1:'B'}

    Raises:
        FileNotFoundError: Neu thu muc khong ton tai hoac khong co class nao.
        ValueError:        Neu co it hon 2 class.
    """
    samples_dir = Path(data_dir) / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc samples tai: {samples_dir}\n"
            f"Hay chay legacy/collect_webcam.py truoc de thu thap du lieu!"
        )

    class_dirs = sorted(d for d in samples_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(
            f"Khong co thu muc con nao trong {samples_dir}. "
            f"Hay thu thap du lieu bang legacy/collect_webcam.py!"
        )

    all_samples: List[Tuple[str, int]] = []
    id_to_label: Dict[int, str] = {}

    for class_id, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        npy_files  = sorted(class_dir.glob("*.npy"))

        if not npy_files:
            print(f"  [WARN] Thu muc '{class_name}' khong co file .npy nao → Bo qua.")
            continue

        id_to_label[class_id] = class_name
        for fp in npy_files:
            all_samples.append((str(fp), class_id))

        print(f"  [{class_id}] {class_name:<5}: {len(npy_files):>3} mau")

    # Re-index neu co class bi bo qua
    if len(id_to_label) < len(class_dirs):
        remap = {old: new for new, old in enumerate(sorted(id_to_label))}
        id_to_label   = {remap[old]: name for old, name in id_to_label.items()}
        all_samples   = [(fp, remap[lbl]) for fp, lbl in all_samples]

    if len(id_to_label) < 2:
        raise ValueError(
            f"Can it nhat 2 lop de phan loai. Hien co: {len(id_to_label)} lop.\n"
            f"Hay thu thap them it nhat 1 chu cai nua!"
        )

    n_webcam = sum(1 for fp, _ in all_samples if VSLDataset._is_webcam_sample(fp))
    n_kaggle = len(all_samples) - n_webcam
    print(f"\n  [PHÂN LOẠI NGUỒN DỮ LIỆU - SOURCE-AWARE AUGMENTATION]:")
    print(f"  - Kaggle / Ảnh ngoài : {n_kaggle:>4} mẫu -> Áp dụng Affine + Physics Augmentation")
    print(f"  - Webcam thật       : {n_webcam:>4} mẫu -> Chỉ áp dụng Affine (Tránh suy thoái kép)\n")

    return all_samples, id_to_label


def stratified_split(
    samples: List[Tuple[str, int]],
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Phan chia du lieu Stratified (can bang ti le moi class trong ca 3 tap).
    Dam bao tap Train/Val/Test deu co du mau cua moi chu cai.

    Returns:
        (train_list, val_list, test_list)
    """
    if not (0.0 < val_ratio < 1.0) or not (0.0 < test_ratio < 1.0):
        raise ValueError("val_ratio va test_ratio phai nam trong khoang (0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio phai < 1.0 de con du phan Train")

    rng = random.Random(seed)
    class_map: Dict[int, List] = defaultdict(list)
    for fp, label in samples:
        class_map[label].append((fp, label))

    train_list, val_list, test_list = [], [], []

    for label, s_list in sorted(class_map.items()):
        rng.shuffle(s_list)
        n = len(s_list)

        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Qua it mau: dung het cho Train, khong chia Val/Test cho class nay
            print(f"  [WARN] Lop {label} chi co {n} mau — dung het cho Train, bo qua Val/Test.")
            train_list.extend(s_list)
            continue

        train_list.extend(s_list[:n_train])
        val_list.extend(s_list[n_train: n_train + n_val])
        test_list.extend(s_list[n_train + n_val:])

    return train_list, val_list, test_list


# =============================================================================
# SECTION E: TRAINING & EVALUATION LOOPS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Huan luyen 1 epoch qua toan bo DataLoader.

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient Clipping: tranh gradient "bung no" khi dataset nho
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size  = inputs.size(0)
        total_loss += loss.item() * batch_size
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += batch_size

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Danh gia model tren tap Val hoac Test (khong cap nhat trong so).

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        loss   = criterion(logits, labels)

        batch_size  = inputs.size(0)
        total_loss += loss.item() * batch_size
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += batch_size

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """
    Tinh ma tran nham lan (Confusion Matrix) kich thuoc (num_classes, num_classes).
    cm[i][j] = so mau thuoc lop i nhung bi du doan la lop j.
    """
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for inputs, labels in loader:
        inputs = inputs.to(device)
        preds  = model(inputs).argmax(dim=1).cpu().numpy()
        truths = labels.cpu().numpy()
        for true_lbl, pred_lbl in zip(truths, preds):
            cm[int(true_lbl)][int(pred_lbl)] += 1

    return cm


# =============================================================================
# SECTION F: MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    args = parse_args()

    # --- Seed --- (dam bao ket qua tai lap)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  HUAN LUYEN TWO-STREAM CNN — THU NGU VSL (DIP v3)")
    print(f"{'='*65}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Patience : {args.patience}  |  Seed: {args.seed}")
    print(f"  Workers  : {args.num_workers}  |  Flip Kaggle: {args.flip_kaggle} (Dong bo huong tay voi Webcam)")
    print(f"{'='*65}\n")

    # --- Tao thu muc luu model ---
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # BUOC 1: Quet va kiem tra dataset
    # -------------------------------------------------------------------------
    print("[1/5] Quet du lieu...")
    try:
        all_samples, id_to_label = scan_dataset(args.data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[LOI] {e}")
        sys.exit(1)

    num_classes = len(id_to_label)
    label_names = [id_to_label[i] for i in range(num_classes)]
    print(f"\n  Tong: {len(all_samples)} mau | {num_classes} lop: {label_names}\n")

    # -------------------------------------------------------------------------
    # BUOC 2: Phan chia Stratified Train / Val / Test
    # -------------------------------------------------------------------------
    print("[2/5] Phan chia du lieu (Stratified Split)...")
    try:
        train_list, val_list, test_list = stratified_split(
            all_samples, args.val_ratio, args.test_ratio, args.seed
        )
    except ValueError as e:
        print(f"\n[LOI] {e}")
        sys.exit(1)

    print(f"  Train: {len(train_list)}  |  Val: {len(val_list)}  |  Test: {len(test_list)}\n")

    if len(train_list) == 0 or len(val_list) == 0:
        print("[LOI] Khong du mau sau khi chia. Thu thap them du lieu!")
        sys.exit(1)

    # --- DataLoaders ---
    train_ds = VSLDataset(train_list, is_train=True,  flip_kaggle=args.flip_kaggle)   # Bat Augmentation
    val_ds   = VSLDataset(val_list,   is_train=False, flip_kaggle=args.flip_kaggle)   # Tat Augmentation
    test_ds  = VSLDataset(test_list,  is_train=False, flip_kaggle=args.flip_kaggle)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------------------------------
    # BUOC 3: Khoi tao model, loss, optimizer, scheduler
    # -------------------------------------------------------------------------
    print("[3/5] Khoi tao Two-Stream CNN...")
    model = TwoStreamHandCNN(num_classes=num_classes).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tong tham so (Trainable): {trainable:,}\n")

    # CrossEntropyLoss + Label Smoothing nhe (giam overfit khi dataset nho)
    # + Class Weight (nghich dao tan suat) de chong thien vi lop nhieu mau -- phat hien 2026-09-30:
    # B=155 mau vs C=64/V=54 khien model hoc lech, V chi dat 63% ngay tren du lieu train (20/54 nham thanh L).
    class_weight_tensor = None
    if args.class_weight:
        train_counts = Counter(c for _, c in train_list)
        counts = np.array([max(1, train_counts.get(i, 0)) for i in range(num_classes)], dtype=np.float32)
        raw_weight = counts.sum() / (num_classes * counts)
        class_weight_tensor = torch.tensor(raw_weight / raw_weight.mean(), dtype=torch.float32).to(device)
        print(f"  Class weight ({'BAT' if args.class_weight else 'TAT'}): " +
              ", ".join(f"{label_names[i]}={class_weight_tensor[i]:.2f}" for i in range(num_classes)))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, weight=class_weight_tensor)

    # Adam + weight decay nhe
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ReduceLROnPlateau: Giam LR x0.5 neu Val Acc khong tang sau 7 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-5,
    )

    # -------------------------------------------------------------------------
    # BUOC 4: Training Loop
    # -------------------------------------------------------------------------
    print("[4/5] Bat dau huan luyen...\n")
    header = f"{'Epoch':>6} | {'TrLoss':>7} | {'TrAcc':>6} | {'VaLoss':>7} | {'VaAcc':>6} | {'LR':>8} | {'Time':>5} |"
    print(header)
    print("-" * len(header))

    best_val_acc    = -1.0
    best_epoch      = 0
    patience_count  = 0
    best_model_path = model_dir / "best_model.pth"

    history: Dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        # Luu lich su
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc,   2))
        history["val_loss"].append(round(val_loss,     4))
        history["val_acc"].append(round(val_acc,       2))

        # Kiem tra best model
        mark = ""
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "id_to_label": id_to_label,
                    "val_acc":     best_val_acc,
                    "args":        vars(args),
                },
                best_model_path,
            )
            mark = " *BEST*"
        else:
            patience_count += 1

        print(
            f"{epoch:>6} | {train_loss:>7.4f} | {train_acc:>5.1f}% |"
            f" {val_loss:>7.4f} | {val_acc:>5.1f}% | {current_lr:>8.2e} | {elapsed:>4.1f}s |{mark}"
        )

        if patience_count >= args.patience:
            print(f"\n[Early Stopping] Val Acc khong tang sau {args.patience} epoch.")
            break

    # -------------------------------------------------------------------------
    # BUOC 5: Danh gia tren tap TEST bang model tot nhat
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Danh gia model tot nhat (Epoch {best_epoch}) tren tap TEST...\n")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    cm = compute_confusion_matrix(model, test_loader, num_classes, device)

    # In ket qua
    print(f"{'='*65}")
    print(f"  KET QUA CUOI CUNG")
    print(f"{'='*65}")
    print(f"  Best Val Accuracy  : {best_val_acc:.1f}%  (Epoch {best_epoch})")
    print(f"  Test Loss          : {test_loss:.4f}")
    print(f"  Test Accuracy      : {test_acc:.1f}%")

    # Confusion Matrix
    print(f"\n  CONFUSION MATRIX (hang = that, cot = du doan):")
    col_header = "        " + "   ".join(f"{l:>3}" for l in label_names)
    print(f"  {col_header}")
    print(f"  {'  ' + '-'*len(col_header)}")
    for i, row in enumerate(cm):
        row_str = "   ".join(f"{v:>3}" for v in row)
        print(f"  {label_names[i]:>5} | {row_str}")

    # Do chinh xac tung lop
    print(f"\n  DO CHINH XAC TUNG LOP:")
    for i in range(num_classes):
        total_i = int(cm[i].sum())
        if total_i > 0:
            acc_i = 100.0 * cm[i][i] / total_i
            bar   = "█" * int(acc_i // 5) + "░" * (20 - int(acc_i // 5))
            print(f"    [{label_names[i]}] {bar} {acc_i:.1f}%  ({cm[i][i]}/{total_i})")
    print(f"{'='*65}")

    # Luu lich su huan luyen
    history_path = model_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history":       history,
                "id_to_label":   id_to_label,
                "num_classes":   num_classes,
                "best_epoch":    best_epoch,
                "best_val_acc":  best_val_acc,
                "test_acc":      test_acc,
                "confusion_matrix": cm.tolist(),
                "args":          vars(args),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n  Da luu:")
    print(f"    Model tot nhat : {best_model_path}")
    print(f"    Lich su train  : {history_path}")

    # Ve bieu do (neu co matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        epochs_x = list(range(1, len(history["train_loss"]) + 1))

        axes[0].plot(epochs_x, history["train_loss"], label="Train Loss", color="tomato",    linewidth=2)
        axes[0].plot(epochs_x, history["val_loss"],   label="Val Loss",   color="steelblue", linewidth=2)
        axes[0].axvline(best_epoch, color="gold", linestyle="--", label=f"Best (Epoch {best_epoch})")
        axes[0].set_title("Loss Curve")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs_x, history["train_acc"], label="Train Acc", color="tomato",    linewidth=2)
        axes[1].plot(epochs_x, history["val_acc"],   label="Val Acc",   color="steelblue", linewidth=2)
        axes[1].axvline(best_epoch, color="gold", linestyle="--", label=f"Best: {best_val_acc:.1f}%")
        axes[1].set_title("Accuracy Curve")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.suptitle(
            f"Two-Stream CNN — VSL  |  Test Acc: {test_acc:.1f}%  |  Classes: {label_names}",
            fontsize=12,
        )
        plt.tight_layout()

        chart_path = model_dir / "training_curves.png"
        plt.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    Bieu do Loss/Acc   : {chart_path}")

        # Ve bieu do ma tran nham lan (Confusion Matrix)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig_cm.colorbar(im, ax=ax_cm)
        ax_cm.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=label_names,
            yticklabels=label_names,
            title=f"Confusion Matrix (Test Acc: {test_acc:.1f}%)",
            ylabel="True Label (Thuc te)",
            xlabel="Predicted Label (Du doan)",
        )
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold" if i == j else "normal",
                )
        plt.tight_layout()
        cm_path = model_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close(fig_cm)
        print(f"    Ma tran nham lan   : {cm_path}")

    except ImportError:
        print("    [INFO] Cai matplotlib de xuat bieu do: pip install matplotlib")

    print(f"\n  Huan luyen hoan tat!\n")


if __name__ == "__main__":
    main()
