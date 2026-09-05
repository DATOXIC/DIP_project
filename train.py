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

Online Augmentation (VSL-Safe — KHÔNG flip ngang để tránh đổi tay trái/phải):
  - Random Rotation ±8°       → Mô phỏng tay nghiêng nhẹ
  - Random Translate ±4%      → Mô phỏng tay lệch tâm ROI
  - Random Scale 0.95–1.05    → Mô phỏng khoảng cách camera thay đổi
  - Gaussian Noise (50% xác suất) → Mô phỏng webcam chất lượng thấp

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
from collections import defaultdict
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
    parser.add_argument("--num_workers", type=int,   default=0,
                        help="So DataLoader workers (0=main process, an toan tren Windows)")
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

    def __init__(self, samples: List[Tuple[str, int]], is_train: bool = True) -> None:
        if not samples:
            raise ValueError("Danh sach samples khong duoc rong!")
        self.samples  = samples
        self.is_train = is_train

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

        if self.is_train:
            tensor = self._augment(tensor)

        return tensor, label

    def _augment(self, t: torch.Tensor) -> torch.Tensor:
        """Chuoi bien doi ngau nhien an toan cho Thu Ngu."""
        angle = random.uniform(-self._ROT_DEG, self._ROT_DEG)
        t = _affine_transform(t,
            cos_a=math.cos(math.radians(angle)),
            sin_a=math.sin(math.radians(angle)),
            tx=0.0, ty=0.0, scale=1.0)

        tx = random.uniform(-self._TRANS_FRAC, self._TRANS_FRAC)
        ty = random.uniform(-self._TRANS_FRAC, self._TRANS_FRAC)
        t  = _affine_transform(t, cos_a=1.0, sin_a=0.0, tx=tx, ty=ty, scale=1.0)

        scale = random.uniform(self._SCALE_LOW, self._SCALE_HIGH)
        t = _affine_transform(t, cos_a=1.0, sin_a=0.0, tx=0.0, ty=0.0, scale=scale)

        if random.random() < self._NOISE_PROB:
            noise = torch.randn_like(t) * self._NOISE_STD
            t = torch.clamp(t + noise, 0.0, 1.0)

        return t


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
            f"Hay chay collect_webcam.py truoc de thu thap du lieu!"
        )

    class_dirs = sorted(d for d in samples_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(
            f"Khong co thu muc con nao trong {samples_dir}. "
            f"Hay thu thap du lieu bang collect_webcam.py!"
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
        truths = labels.numpy()
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
    train_ds = VSLDataset(train_list, is_train=True)    # Bat Augmentation
    val_ds   = VSLDataset(val_list,   is_train=False)   # Tat Augmentation
    test_ds  = VSLDataset(test_list,  is_train=False)

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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

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
        print(f"    Bieu do        : {chart_path}")

    except ImportError:
        print("    [INFO] Cai matplotlib de xuat bieu do: pip install matplotlib")

    print(f"\n  Huan luyen hoan tat!\n")


if __name__ == "__main__":
    main()
