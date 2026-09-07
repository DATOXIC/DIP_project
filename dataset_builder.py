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

        # 1. Đọc metadata.json cũ nếu tồn tại
        meta_path = os.path.join(self.output_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.classes = meta.get("classes", {})
                    self.samples = meta.get("samples", [])
                    self.sample_counts = meta.get("sample_counts", {})
            except Exception as e:
                print(f"[DatasetBuilder] Không thể đọc metadata cũ ({e}), khởi tạo mới.")

        # 2. Quét thực tế trên đĩa để đồng bộ số lượng chính xác tuyệt đối
        self._sync_with_disk()

    def _scan_disk_for_class(self, class_name: str) -> int:
        """Đếm số file .npy thực tế đang tồn tại trong thư mục của class."""
        class_folder = os.path.join(self.samples_dir, class_name)
        if not os.path.exists(class_folder):
            return 0
        return len([f for f in os.listdir(class_folder) if f.endswith(".npy")])

    def _get_next_index_on_disk(self, class_name: str, prefix: str = "") -> int:
        """
        Tìm số thứ tự (index) lớn nhất của prefix trong thư mục để tránh ghi đè.
        Ví dụ: nếu đã có webcam_A_00031.npy thì số tiếp theo sẽ là 32.
        """
        class_folder = os.path.join(self.samples_dir, class_name)
        if not os.path.exists(class_folder):
            return 0

        max_idx = -1
        prefix_tag = f"{prefix}_{class_name}_" if prefix else f"{class_name}_"

        for fname in os.listdir(class_folder):
            if not fname.endswith(".npy"):
                continue
            if fname.startswith(prefix_tag):
                num_part = fname[len(prefix_tag):-4]
                try:
                    val = int(num_part)
                    if val > max_idx:
                        max_idx = val
                except ValueError:
                    pass
            elif not prefix:
                parts = fname[:-4].split("_")
                try:
                    val = int(parts[-1])
                    if val > max_idx:
                        max_idx = val
                except ValueError:
                    pass

        return max_idx + 1

    def _sync_with_disk(self):
        """Quét toàn bộ thư mục samples để đồng bộ sample_counts và tái tạo danh sách samples."""
        if not os.path.exists(self.samples_dir):
            return

        existing_paths = {s.get("path") for s in self.samples if isinstance(s, dict)}

        for c_name in os.listdir(self.samples_dir):
            c_folder = os.path.join(self.samples_dir, c_name)
            if not os.path.isdir(c_folder):
                continue

            npy_files = [f for f in os.listdir(c_folder) if f.endswith(".npy")]
            self.sample_counts[c_name] = len(npy_files)

            if c_name not in self.classes:
                self.classes[c_name] = {
                    "id": len(self.classes),
                    "description": f"Class {c_name}",
                    "created_at": time.time()
                }

            c_id = self.classes[c_name]["id"]
            for f in npy_files:
                rel_path = f"samples/{c_name}/{f}".replace("\\", "/")
                if rel_path not in existing_paths:
                    self.samples.append({
                        "path": rel_path,
                        "class": c_name,
                        "class_id": c_id,
                        "shape": [128, 128, 3],
                        "timestamp": os.path.getmtime(os.path.join(c_folder, f))
                    })
                    existing_paths.add(rel_path)

    def register_class(self, class_name, description=""):
        """Đăng ký một nhãn ký hiệu thủ ngữ mới."""
        class_folder = os.path.join(self.samples_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

        if class_name not in self.classes:
            self.classes[class_name] = {
                "id": len(self.classes),
                "description": description,
                "created_at": time.time()
            }
            print(f"[DatasetBuilder] Đã đăng ký nhãn mới: '{class_name}' ({description})")
        elif description and not self.classes[class_name].get("description"):
            self.classes[class_name]["description"] = description

        # Luôn đồng bộ số lượng mẫu thực tế trên đĩa
        self.sample_counts[class_name] = self._scan_disk_for_class(class_name)

    def save_sample(self, multi_channel_tensor, class_name, prefix=""):
        """
        Lưu một tensor đặc trưng đa kênh (128, 128, 3) vào đĩa.
        prefix: Tiền tố nhận diện nguồn (ví dụ: 'webcam', 'kaggle', 'ext').
        ĐẢM BẢO 100% KHÔNG BAO GIỜ GHI ĐÈ FILE ĐÃ CÓ.
        """
        if class_name not in self.classes:
            self.register_class(class_name)

        # Lấy số thứ tự kế tiếp an toàn trên đĩa
        next_idx = self._get_next_index_on_disk(class_name, prefix)

        if prefix:
            filename = f"{prefix}_{class_name}_{next_idx:05d}.npy"
        else:
            filename = f"{class_name}_{next_idx:05d}.npy"

        full_path = os.path.join(self.samples_dir, class_name, filename)

        # Khóa an toàn 2 lớp: Nếu file vẫn tồn tại (do xung đột) thì tăng tiếp
        while os.path.exists(full_path):
            next_idx += 1
            filename = f"{prefix}_{class_name}_{next_idx:05d}.npy" if prefix else f"{class_name}_{next_idx:05d}.npy"
            full_path = os.path.join(self.samples_dir, class_name, filename)

        relative_path = os.path.join("samples", class_name, filename).replace("\\", "/")

        # Lưu tensor dưới dạng npy
        np.save(full_path, multi_channel_tensor.astype(np.uint8))
        
        # Cập nhật số lượng mẫu thực tế
        self.sample_counts[class_name] = self._scan_disk_for_class(class_name)

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
        # Đồng bộ lại số lượng thực tế của tất cả các class
        for c_name in list(self.classes.keys()):
            self.sample_counts[c_name] = self._scan_disk_for_class(c_name)

        metadata = {
            "total_samples": sum(self.sample_counts.values()),
            "classes": self.classes,
            "sample_counts": self.sample_counts,
            "samples": self.samples,
            "updated_at": time.time()
        }
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"[DatasetBuilder] Đã lưu metadata vào: {meta_path} (Tổng: {metadata['total_samples']} mẫu trên đĩa)")

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
