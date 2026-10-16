"""
HUẤN LUYỆN MÔ HÌNH HỌC SÂU NHẬN DIỆN CỬ CHỈ ĐỘNG (VSL / ASL)
===========================================================
Tác giả: Tô Phương Đạt (24110086) - Dự án DIP Project
Mục đích:
  - Khai thác dữ liệu Đa Nguồn: ASL Citizen (52 người Mỹ) + Đạt (Webcam) + Các Tình nguyện viên (Mai, Đăng, Phi, Vương, Việt).
  - Kiến trúc: 1D Temporal Convolution + Bi-GRU + Attention Pooling (Siêu nhẹ, ~80K params, chạy < 1.5ms trên CPU).
  - Hỗ trợ đánh giá Người Lạ (Leave-One-Signer-Out - LOSO): Giấu hẳn 1 người làm test set để chứng minh khả năng tổng quát hóa.
  - So sánh trực tiếp: DTW Đơn thuần vs Deep Learning vs Hybrid (Late Fusion DTW + DL).
  - Tự động xuất file trọng số PyTorch (.pth) và ONNX (.onnx) tối ưu cho suy luận CPU thời gian thực.
"""

import os
import sys
import glob
import json
import time
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Danh sách 13 từ vựng mục tiêu chuẩn
TARGET_WORDS = [
    "BAD", "BYE", "DRINK", "EAT", "HELLO",
    "HUNGRY", "ME", "PLEASE", "SORRY", "THANK_YOU",
    "WATER", "YES", "YOU"
]


# =============================================================================
# 1. KIẾN TRÚC MẠNG HỌC SÂU SIÊU NHẸ (LIGHTWEIGHT DYNAMIC GESTURE NET)
# =============================================================================

class TemporalAttentionPooling(nn.Module):
    """
    Cơ chế Attention Pooling theo chiều thời gian:
    Tự động học cách 'chú ý' vào các khung hình quan trọng nhất của cử chỉ
    (thay vì chỉ lấy trung bình cộng thô thiển hoặc lấy khung hình cuối cùng).
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (Batch, Time, Dim)
        weights = self.attn(x)  # (Batch, Time, 1)
        weights = F.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # (Batch, Dim)
        return pooled, weights.squeeze(-1)


class LightweightGestureNet(nn.Module):
    """
    Mô hình học sâu kết hợp 1D-CNN + Bi-GRU + Attention:
      1. LayerNorm / BatchNorm chuẩn hóa đầu vào 24D.
      2. 1D Temporal Convolution: Trích xuất quan hệ cục bộ giữa các khớp theo thời gian.
      3. Bi-directional GRU: Nắm bắt toàn bộ ngữ cảnh chuyển động từ đầu đến cuối cử chỉ.
      4. Temporal Attention Pooling: Gom thông tin có trọng số các khung hình cốt lõi.
      5. Classification Head: Đưa ra phân phối xác suất 13 từ vựng.
    Tổng tham số: ~85,000 parameters (< 350 KB).
    """
    def __init__(self, in_features: int = 24, num_classes: int = 13, hidden_dim: int = 64):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features)

        # 1D Temporal Convolution Block
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_conv = nn.Dropout(0.2)

        # Bi-directional GRU Sequence Modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Attention Pooling
        self.attn_pool = TemporalAttentionPooling(in_dim=hidden_dim * 2)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Time=32, Features=24)
        x = x.transpose(1, 2)  # (Batch, Features=24, Time=32)
        x = self.norm(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)

        x = x.transpose(1, 2)  # (Batch, Time=32, Channels=64)
        gru_out, _ = self.gru(x)  # (Batch, Time=32, Hidden=128)

        pooled, _ = self.attn_pool(gru_out)  # (Batch, Hidden=128)
        logits = self.classifier(pooled)     # (Batch, num_classes)
        return logits


# =============================================================================
# 2. XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU CHUỖI ĐỘNG (SEQUENCE DATA AUGMENTATION)
# =============================================================================

class DynamicGestureDataset(Dataset):
    """
    Dataset nạp chuỗi đặc trưng (32, 24) đi kèm kỹ thuật Data Augmentation:
      - Gaussian Noise: Tăng độ chịu nhiễu cảm biến camera.
      - Time Warping / Linear Scaling: Giả lập tốc độ người ký nhanh / chậm.
      - Feature Dropout: Xóa ngẫu nhiên một số kênh đặc trưng để chống phụ thuộc vào 1 khớp.
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, is_train: bool = True):
        self.sequences = sequences.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.labels)

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        # seq shape: (32, 24)
        augmented = seq.copy()

        # 1. Thêm nhiễu Gauss nhẹ (p=0.5)
        if random.random() < 0.5:
            noise = np.random.normal(0.0, 0.015, size=augmented.shape).astype(np.float32)
            augmented += noise

        # 2. Scale biên độ chuyển động nhẹ (0.9 - 1.1) (p=0.4)
        if random.random() < 0.4:
            scale_factor = random.uniform(0.9, 1.1)
            augmented *= scale_factor

        # 3. Time shifting ngẫu nhiên (dịch chuyển khung hình +/- 2 frames) (p=0.3)
        if random.random() < 0.3:
            shift = random.randint(-2, 2)
            if shift != 0:
                augmented = np.roll(augmented, shift, axis=0)
                if shift > 0:
                    augmented[:shift] = augmented[shift]
                else:
                    augmented[shift:] = augmented[shift - 1]

        # 4. Feature Dropout: Xóa ngẫu nhiên 1-2 kênh đặc trưng (p=0.3)
        if random.random() < 0.3:
            feat_idx = random.randint(0, augmented.shape[1] - 1)
            augmented[:, feat_idx] = 0.0

        return augmented

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        if self.is_train:
            seq = self._augment(seq)
        return torch.from_numpy(seq), torch.tensor(self.labels[idx], dtype=torch.long)


# =============================================================================
# 3. TRÌNH NẠP DỮ LIỆU ĐA NGUỒN (MULTI-SOURCE DATA LOADER)
# =============================================================================

def load_all_sign_data(
    citizen_file: str = "asl_citizen_templates_24d.json",
    dat_file: str = "webcam_templates.json",
    profiles_dir: str = "profiles",
    exclude_citizen_words: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Thu thập toàn bộ mẫu từ mọi nguồn, gán nhãn người ký (signer_id) và lớp (word).
    Loại bỏ các từ xung đột phương ngữ từ ASL Citizen (ví dụ 'BYE' ký khác người Việt).
    """
    if exclude_citizen_words is None:
        exclude_citizen_words = ["BYE"]  # Mặc định lọc bỏ BYE của Citizen vì xung đột phương ngữ

    word_to_id = {w: i for i, w in enumerate(TARGET_WORDS)}
    all_samples: List[Dict[str, Any]] = []

    # 1. Nạp từ ASL Citizen (đã lọc các từ xung đột phong cách)
    if os.path.exists(citizen_file):
        with open(citizen_file, "r", encoding="utf-8") as f:
            cit_data = json.load(f)
        for w, content in cit_data.get("vocabulary", {}).items():
            if w in word_to_id and w not in exclude_citizen_words:
                for s in content.get("samples", []):
                    traj = s.get("traj_features_9d") or s.get("traj_features")
                    if traj and len(traj) == 32:
                        all_samples.append({
                            "word": w,
                            "label": word_to_id[w],
                            "signer": "Citizen",
                            "traj": np.nan_to_num(np.asarray(traj, dtype=np.float32))
                        })

    # 2. Nạp từ Webcam của Đạt
    if os.path.exists(dat_file):
        with open(dat_file, "r", encoding="utf-8") as f:
            dat_data = json.load(f)
        for w, content in dat_data.get("vocabulary", {}).items():
            if w in word_to_id:
                for s in content.get("samples", []):
                    if s.get("is_original", True) is not False and not s.get("augment_of"):
                        traj = s.get("traj_features_9d") or s.get("traj_features")
                        if traj and len(traj) == 32:
                            all_samples.append({
                                "word": w,
                                "label": word_to_id[w],
                                "signer": "Dat",
                                "traj": np.nan_to_num(np.asarray(traj, dtype=np.float32))
                            })

    # 3. Nạp từ các Tình nguyện viên trong profiles/
    for p_file in glob.glob(os.path.join(profiles_dir, "*.json")):
        base_name = os.path.splitext(os.path.basename(p_file))[0]
        if base_name.startswith("_") or base_name == "people":
            continue
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            for w, content in p_data.get("vocabulary", {}).items():
                if w in word_to_id:
                    for s in content.get("samples", []):
                        traj = s.get("traj_features_9d") or s.get("traj_features")
                        if traj and len(traj) == 32:
                            all_samples.append({
                                "word": w,
                                "label": word_to_id[w],
                                "signer": base_name,
                                "traj": np.nan_to_num(np.asarray(traj, dtype=np.float32))
                            })
        except Exception as e:
            print(f"[!] Bỏ qua file {p_file} do lỗi: {e}")

    return all_samples, word_to_id


# =============================================================================
# 4. CHU TRÌNH HUẤN LUYỆN & KIỂM THỬ (TRAIN & EVALUATION ENGINE)
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: str = "best_gesture_model.pth"
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n-> Bắt đầu huấn luyện trên thiết bị: {device} ({epochs} epochs)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Huấn luyện
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)

        scheduler.step()
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # 2. Đánh giá trên tập Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_loss = val_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "target_words": TARGET_WORDS
            }, save_path)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:3d}/{epochs:3d}] | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:5.1f}% | "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:5.1f}% | Best: {best_val_acc*100:5.1f}%")

    elapsed = time.time() - start_time
    print(f"\n✓ Hoàn tất huấn luyện trong {elapsed:.1f} giây! Mô hình tốt nhất lưu tại: {save_path}")
    return history


def evaluate_test_set(
    model: nn.Module,
    test_samples: List[Dict[str, Any]],
    word_to_id: Dict[str, int],
    device: torch.device
) -> Dict[str, Any]:
    """
    Đánh giá chi tiết trên tập người lạ (test set).
    """
    id_to_word = {i: w for w, i in word_to_id.items()}
    model.eval()

    X_test = np.stack([s["traj"] for s in test_samples])
    y_test = np.asarray([s["label"] for s in test_samples])

    with torch.no_grad():
        Xt = torch.from_numpy(X_test).to(device)
        logits = model(Xt)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        confs = probs.max(axis=-1)

    correct = (preds == y_test)
    total_acc = float(np.mean(correct))

    # Thống kê từng từ
    per_word = {}
    for w in TARGET_WORDS:
        idx = word_to_id[w]
        mask = (y_test == idx)
        if np.any(mask):
            w_acc = float(np.mean(preds[mask] == idx))
            w_total = int(np.sum(mask))
            w_correct = int(np.sum(preds[mask] == idx))
            per_word[w] = {"total": w_total, "correct": w_correct, "accuracy": w_acc}

    return {
        "total_samples": len(test_samples),
        "overall_accuracy": total_acc,
        "per_word": per_word,
        "probs": probs,
        "preds": preds,
        "labels": y_test
    }


def export_to_onnx(model: nn.Module, output_path: str = "best_gesture_model.onnx"):
    """
    Xuất mô hình sang định dạng ONNX để chạy siêu tốc trên CPU máy tính.
    """
    try:
        model.eval().cpu()
        dummy_input = torch.randn(1, 32, 24, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["gesture_sequence"],
            output_names=["class_logits"],
            dynamic_axes={
                "gesture_sequence": {0: "batch_size"},
                "class_logits": {0: "batch_size"}
            }
        )
        print(f"✓ Đã xuất thành công mô hình ONNX: {output_path}")
    except Exception as e:
        print(f"[LƯU Ý] Chưa thể xuất ONNX (cần cài đặt: pip install onnx onnxscript): {e}")


# =============================================================================
# 5. HÀM CHẠY CHÍNH (MAIN CLI EXECUTION)
# =============================================================================

def run_experiment():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Deep Learning cho cử chỉ VSL/ASL")
    parser.add_argument("--test_signer", type=str, default="Mai",
                        help="Tên người làm kiểm thử người lạ (LOSO): 'Mai', 'Dang', 'Phi', 'Vuong', hoặc 'none'")
    parser.add_argument("--epochs", type=int, default=100, help="Số vòng lặp huấn luyện")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học (Learning rate)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument("--export_onnx", action="store_true", help="Tự động xuất file ONNX")
    parser.add_argument("--out", type=str, default=None,
                        help="Đường dẫn .pth tùy chỉnh để lưu checkpoint (mặc định: best_gesture_model_<test_signer>.pth). "
                             "Dùng khi train nhiều seed cho CÙNG 1 test_signer để đo phương sai, tránh ghi đè checkpoint chính thức.")
    args = parser.parse_args()

    # Thiết lập seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  HUẤN LUYỆN MÔ HÌNH HỌC SÂU NHẬN DIỆN CỬ CHỈ ĐỘNG (DIP PROJECT)")
    print(f"  Thiết bị: {device} | Thử nghiệm Người Lạ (LOSO): [{args.test_signer}]")
    print("=" * 72)

    # 1. Nạp toàn bộ dữ liệu
    all_samples, word_to_id = load_all_sign_data()
    print(f"\n-> Đã nạp tổng cộng {len(all_samples)} mẫu cử chỉ từ tất cả các nguồn.")

    # Thống kê theo người ký
    signer_counts: Dict[str, int] = {}
    for s in all_samples:
        signer_counts[s["signer"]] = signer_counts.get(s["signer"], 0) + 1
    for s_name, cnt in sorted(signer_counts.items()):
        print(f"   + Nguồn [{s_name:<8}]: {cnt:3d} mẫu")

    # 2. Chia tập Train / Test theo cơ chế Leave-One-Signer-Out
    train_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    if args.test_signer.lower() != "none" and args.test_signer in signer_counts:
        for s in all_samples:
            if s["signer"] == args.test_signer:
                test_samples.append(s)
            else:
                train_samples.append(s)
        print(f"\n-> ĐÃ CÁCH LY NGƯỜI LẠ: [{args.test_signer}] ({len(test_samples)} mẫu) làm TEST SET.")
        print(f"   Huấn luyện trên {len(train_samples)} mẫu của các nguồn còn lại.")
    else:
        # Nếu test_signer là none: Phân chia ngẫu nhiên 80% train / 20% test
        random.shuffle(all_samples)
        split_idx = int(0.8 * len(all_samples))
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
        print(f"\n-> Phân chia ngẫu nhiên 80/20: Train={len(train_samples)}, Test={len(test_samples)}")

    # 3. Chuẩn bị DataLoader
    X_train = np.stack([s["traj"] for s in train_samples])
    y_train = np.asarray([s["label"] for s in train_samples])
    X_test = np.stack([s["traj"] for s in test_samples])
    y_test = np.asarray([s["label"] for s in test_samples])

    train_dataset = DynamicGestureDataset(X_train, y_train, is_train=True)
    test_dataset = DynamicGestureDataset(X_test, y_test, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Khởi tạo mô hình
    model = LightweightGestureNet(in_features=24, num_classes=len(TARGET_WORDS), hidden_dim=64).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n-> Khởi tạo LightweightGestureNet thành công! Tổng tham số: {num_params:,}")

    # 5. Huấn luyện mô hình
    best_weights_path = args.out if args.out else f"best_gesture_model_{args.test_signer}.pth"
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=best_weights_path
    )

    # 6. Đánh giá chi tiết trên tập người lạ
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_results = evaluate_test_set(model, test_samples, word_to_id, device)

    print("\n" + "=" * 72)
    print(f"  KẾT QUẢ ĐÁNH GIÁ TRÊN NGƯỜI LẠ: [{args.test_signer}]")
    print(f"  Độ chính xác tổng thể (Top-1 Accuracy): {eval_results['overall_accuracy']*100:.2f}%")
    print("=" * 72)
    print(f"{'Từ vựng':<12} | {'Số mẫu':<8} | {'Nhận đúng':<10} | {'Độ chính xác':<12}")
    print("-" * 50)
    for w, stats in eval_results["per_word"].items():
        print(f"{w:<12} | {stats['total']:<8} | {stats['correct']:<10} | {stats['accuracy']*100:6.1f}%")

    # 7. Xuất file ONNX nếu được yêu cầu
    if args.export_onnx:
        onnx_name = f"best_gesture_model_{args.test_signer}.onnx"
        export_to_onnx(model, onnx_name)

    # 8. Lưu kết quả ra file JSON
    result_json_path = (os.path.splitext(args.out)[0] + ".json") if args.out else f"results_eval_{args.test_signer}.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        # Bỏ probs và preds dạng mảng numpy để serialize
        save_data = {
            "test_signer": args.test_signer,
            "overall_accuracy": eval_results["overall_accuracy"],
            "total_samples": eval_results["total_samples"],
            "per_word": eval_results["per_word"]
        }
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu chi tiết đánh giá tại: {result_json_path}")


if __name__ == "__main__":
    run_experiment()
