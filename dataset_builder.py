"""
MODULE TẠO VÀ QUẢN LÝ TẬP DỮ LIỆU THỦ NGỮ (DIP DATASET BUILDER)
============================================================
Sinh viên: Tô Phương Đạt
Dự án: Nhận diện Thủ ngữ Tiếng Việt (VSL)

Chức năng:
- Thu thập Tensor đa kênh (128, 128, 3) từ improved_dip_pipeline
- Lưu trữ mẫu dưới dạng mảng nhị phân nén NumPy (.npy)
- Quản lý nhãn và danh mục ký hiệu (A, B, hello, thank_you...)
- Tự động phân chia tập Train / Val / Test (70% - 15% - 15%)
"""

import os
import json
import time
import random
import numpy as np


class DIPDatasetBuilder:
    def __init__(self, output_dir="./vsl_dataset"):
        self.output_dir = output_dir
        self.samples_dir = os.path.join(output_dir, "samples")
        self.splits_dir = os.path.join(output_dir, "splits")
        
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)

        self.classes = {}
        self.samples = []
        self.sample_counts = {}

    def register_class(self, class_name, description=""):
        """Đăng ký một nhãn ký hiệu thủ ngữ mới."""
        if class_name not in self.classes:
            self.classes[class_name] = {
                "id": len(self.classes),
                "description": description,
                "created_at": time.time()
            }
            self.sample_counts[class_name] = 0
            # Tạo thư mục con cho class
            class_folder = os.path.join(self.samples_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            print(f"[DatasetBuilder] Đã đăng ký nhãn: '{class_name}' ({description})")

    def save_sample(self, multi_channel_tensor, class_name):
        """
        Lưu một tensor đặc trưng đa kênh (128, 128, 3) vào đĩa.
        """
        if class_name not in self.classes:
            self.register_class(class_name)

        count = self.sample_counts[class_name]
        filename = f"{class_name}_{count:05d}.npy"
        relative_path = os.path.join("samples", class_name, filename).replace("\\", "/")
        full_path = os.path.join(self.samples_dir, class_name, filename)

        # Lưu tensor dưới dạng npy
        np.save(full_path, multi_channel_tensor.astype(np.uint8))
        self.sample_counts[class_name] += 1

        # Lưu thông tin mẫu vào danh sách
        sample_info = {
            "path": relative_path,
            "class": class_name,
            "class_id": self.classes[class_name]["id"],
            "shape": list(multi_channel_tensor.shape),
            "timestamp": time.time()
        }
        self.samples.append(sample_info)
        return relative_path

    def save_metadata(self):
        """Lưu file cấu hình metadata tổng thể của dataset."""
        metadata = {
            "total_samples": len(self.samples),
            "classes": self.classes,
            "sample_counts": self.sample_counts,
            "created_at": time.time()
        }
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"[DatasetBuilder] Đã lưu metadata vào: {meta_path} (Tổng: {len(self.samples)} mẫu)")

    def create_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Tự động phân chia ngẫu nhiên dữ liệu thành 3 tập Train / Val / Test.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Tổng tỉ lệ phải bằng 1.0"
        random.seed(seed)

        # Gom mẫu theo từng class để phân chia cân bằng (Stratified Split)
        class_samples = {}
        for sample in self.samples:
            c = sample["class"]
            if c not in class_samples:
                class_samples[c] = []
            class_samples[c].append(sample)

        train_samples = []
        val_samples = []
        test_samples = []

        for c, s_list in class_samples.items():
            random.shuffle(s_list)
            n = len(s_list)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_samples.extend(s_list[:n_train])
            val_samples.extend(s_list[n_train:n_train + n_val])
            test_samples.extend(s_list[n_train + n_val:])

        # Lưu các file split
        splits = {
            "train": (train_samples, "train_split.json"),
            "val": (val_samples, "val_split.json"),
            "test": (test_samples, "test_split.json")
        }

        for split_name, (split_data, fname) in splits.items():
            out_path = os.path.join(self.splits_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "split": split_name,
                    "count": len(split_data),
                    "samples": split_data
                }, f, indent=4, ensure_ascii=False)
            print(f"[DatasetBuilder] Đã tạo tập {split_name.upper()}: {len(split_data)} mẫu -> {fname}")
