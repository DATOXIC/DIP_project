"""
CÔNG CỤ TỰ ĐỘNG DÒ NGƯỠNG THEO TỪNG TỪ (PER-WORD THRESHOLD TUNER)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Mục đích:
  Thay thế ngưỡng "threshold: 1.25" dùng chung cho MỌI từ trong
  dynamic_templates.json bằng ngưỡng riêng cho từng từ, tính toán trực tiếp
  từ dữ liệu mẫu thực tế đã thu thập — dùng đúng hàm khoảng cách DTW 12D
  (compute_golden_triad_dtw_distance) và trọng số Per-Sign (_get_weights_for_sign)
  mà production (predict_sentence_webcam.py) dùng để so khớp, đảm bảo ngưỡng
  đề xuất phản ánh chính xác hành vi thật của engine.

Công thức:
  - intra_max = khoảng cách DTW LỚN NHẤT giữa các mẫu CÙNG 1 từ (Leave-One-Out,
    so từng cặp mẫu với nhau, không so mẫu với chính nó).
  - inter_min = khoảng cách DTW NHỎ NHẤT từ bất kỳ mẫu nào của từ này đến bất kỳ
    mẫu nào của TỪ KHÁC (toàn bộ từ điển), dùng trọng số của ứng viên (candidate)
    đang so (other_key) thay vì từ đang được đánh giá (word_key), phản ánh đúng cách
    production đánh giá từng ứng viên.
  - proposed_threshold = min(intra_max * 1.15, inter_min * 0.85), sau đó kẹp vào
    khoảng [0.6, 2.0] để chống giá trị bệnh lý khi dữ liệu quá thưa.

Cách dùng:
  python tune_thresholds.py                       # Chỉ xem trước bảng đề xuất, KHÔNG ghi file
  python tune_thresholds.py --apply                # Sao lưu + ghi ngưỡng mới vào file templates
  python tune_thresholds.py --templates <path>      # Chỉ định file templates khác mặc định
"""

import os
import sys
import json
import shutil
import time
import argparse
import math
from typing import Any, Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from dynamic_dtw_engine import GoldenTriadDTWMatcher, compute_golden_triad_dtw_distance, canonicalize_chirality

try:
    from predict_sentence_webcam import CONFIG
except Exception:
    CONFIG = {}

DEFAULT_TEMPLATES_PATH = "dynamic_templates.json"

# Hằng số công thức đề xuất ngưỡng
INTRA_MARGIN = 1.15    # Nới rộng 15% trên khoảng cách nội-từ lớn nhất (chống hụt ACCEPT)
INTER_MARGIN = 0.85    # Thu hẹp 15% dưới khoảng cách liên-từ nhỏ nhất (chống nhầm từ khác)
THRESH_FLOOR = 0.6     # Ngưỡng sàn an toàn, chống giá trị quá thấp do dữ liệu thưa
THRESH_CEIL = 2.0      # Ngưỡng trần an toàn, chống giá trị quá cao do dữ liệu bệnh lý


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dò ngưỡng threshold động theo từng từ vựng từ dữ liệu mẫu thực tế (Golden Triad 12D)."
    )
    parser.add_argument(
        "--templates", default=DEFAULT_TEMPLATES_PATH,
        help=f"Đường dẫn file templates JSON (mặc định: {DEFAULT_TEMPLATES_PATH})"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Ghi ngưỡng mới trực tiếp vào file templates (mặc định chỉ xem trước, KHÔNG ghi gì)."
    )
    return parser.parse_args()


def load_sample_traj(sample: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """Trích xuất (traj, shapes) của 1 mẫu, ưu tiên traj_features_9d rồi mới đến traj_features."""
    traj_raw = sample.get("traj_features_9d")
    if traj_raw is None:
        traj_raw = sample.get("traj_features", [])
    traj = np.array(traj_raw, dtype=np.float32)
    # CHUẨN HÓA CHIRALITY bằng CÙNG hàm mà GoldenTriadDTWMatcher.load_templates() dùng,
    # để ngưỡng/audit tính offline nằm trong ĐÚNG không gian dấu mà production so khớp.
    # Nếu bỏ bước này, mọi con số dưới đây đo trên một không gian khác với lúc chạy thật.
    if not sample.get("chirality_canonical", False):
        traj, _ = canonicalize_chirality(traj, handedness=sample.get("handedness"))
    shapes = sample.get("hand_shapes", ["Unknown"] * len(traj))
    return traj, shapes


def main() -> None:
    args = parse_args()
    templates_path = args.templates

    if not os.path.exists(templates_path):
        print(f"[!] Không tìm thấy file templates: {templates_path}")
        sys.exit(1)

    # Dùng GoldenTriadDTWMatcher (strict_weights=False vì mục đích chẩn đoán, không
    # muốn crash cứng ngay cả khi vocabulary/weights đang tạm không nhất quán) để nạp
    # templates + trọng số Per-Sign ASL-LEX, đảm bảo khoảng cách tính ở đây GIỐNG HỆT
    # khoảng cách production thật sự dùng khi match.
    matcher = GoldenTriadDTWMatcher(
        templates_path=templates_path,
        strict_weights=False,
        w_p=CONFIG.get("w_wrist_pos", 0.4),
        w_a=CONFIG.get("w_finger_angles", 0.8),
        w_m=CONFIG.get("w_wrist_dir", 0.5),
        w_n=CONFIG.get("w_palm_normal", 0.5),
        lambda_shape=CONFIG.get("lambda_shape", 0.0),
        asllex_blend_alpha=CONFIG.get("asllex_blend_alpha", 1.0),
        asllex_max_deviation=CONFIG.get("asllex_max_deviation", None),
    )
    vocabulary = matcher.vocabulary

    if not vocabulary:
        print("[!] Từ điển templates rỗng, không có gì để dò ngưỡng.")
        sys.exit(0)

    # Nạp trước toàn bộ mẫu (traj, shapes) hợp lệ của mọi từ vào bộ nhớ 1 lần duy nhất.
    word_samples: Dict[str, List[Tuple[np.ndarray, List[str]]]] = {}
    for word_key, word_data in vocabulary.items():
        parsed: List[Tuple[np.ndarray, List[str]]] = []
        for s in word_data.get("samples", []):
            traj, shapes = load_sample_traj(s)
            if len(traj) == 32:
                parsed.append((traj, shapes))
        word_samples[word_key] = parsed

    rows: List[Dict[str, Any]] = []
    skipped_words: List[Tuple[str, str, int]] = []

    for word_key, word_data in vocabulary.items():
        display_name = word_data.get("display_name", word_key)
        old_threshold = float(word_data.get("threshold", 1.25))
        samples = word_samples.get(word_key, [])
        num_samples = len(samples)

        if num_samples < 2:
            skipped_words.append((word_key, display_name, num_samples))
            continue

        wp, wa, wm, wn = matcher._get_weights_for_sign(word_key)

        # intra_max: khoảng cách DTW lớn nhất giữa các mẫu CÙNG 1 từ (Leave-One-Out)
        intra_max = 0.0
        for i in range(num_samples):
            for j in range(num_samples):
                if i == j:
                    continue
                q_traj, q_shapes = samples[i]
                t_traj, t_shapes = samples[j]
                dist = compute_golden_triad_dtw_distance(
                    q_traj=q_traj, q_shapes=q_shapes, t_traj=t_traj, t_shapes=t_shapes,
                    w_p=wp, w_a=wa, w_m=wm, w_n=wn,
                    lambda_shape=matcher.lambda_shape,
                    lambda_shape_global=matcher.lambda_shape_global
                )
                intra_max = max(intra_max, dist)

        # inter_min: khoảng cách DTW nhỏ nhất từ mẫu của từ này đến mẫu của TỪ KHÁC.
        # QUAN TRỌNG: dùng trọng số của ỨNG VIÊN đang so (other_key), đúng như cách
        # GoldenTriadDTWMatcher.match() thật sự đánh giá từng ứng viên trong vocabulary —
        # KHÔNG dùng trọng số của word_key (đó chỉ đúng cho intra_max ở trên).
        inter_min = math.inf
        for other_key, other_samples in word_samples.items():
            if other_key == word_key or not other_samples:
                continue
            wp_o, wa_o, wm_o, wn_o = matcher._get_weights_for_sign(other_key)
            for q_traj, q_shapes in samples:
                for t_traj, t_shapes in other_samples:
                    dist = compute_golden_triad_dtw_distance(
                        q_traj=q_traj, q_shapes=q_shapes, t_traj=t_traj, t_shapes=t_shapes,
                        w_p=wp_o, w_a=wa_o, w_m=wm_o, w_n=wn_o,
                        lambda_shape=matcher.lambda_shape,
                        lambda_shape_global=matcher.lambda_shape_global
                    )
                    inter_min = min(inter_min, dist)

        # Nếu không có từ nào khác có mẫu để so sánh, không giới hạn theo inter_min
        # (min(...) với inf tự động bỏ qua nhánh này).
        proposed_raw = min(intra_max * INTRA_MARGIN, inter_min * INTER_MARGIN)
        proposed = max(THRESH_FLOOR, min(THRESH_CEIL, proposed_raw))

        rows.append({
            "word_key": word_key,
            "display_name": display_name,
            "num_samples": num_samples,
            "old_threshold": old_threshold,
            "intra_max": intra_max,
            "inter_min": inter_min,
            "proposed_threshold": proposed
        })

    # ------------------------------------------------------------------
    # In bảng kết quả xem trước
    # ------------------------------------------------------------------
    header = f"{'word_key':<14}{'display_name':<26}{'n':>4}{'old_thr':>10}{'intra_max':>12}{'inter_min':>12}{'proposed':>12}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        inter_str = f"{r['inter_min']:.4f}" if math.isfinite(r["inter_min"]) else "N/A"
        print(
            f"{r['word_key']:<14}{r['display_name'][:24]:<26}{r['num_samples']:>4}"
            f"{r['old_threshold']:>10.3f}{r['intra_max']:>12.4f}{inter_str:>12}{r['proposed_threshold']:>12.4f}"
        )
    print("=" * len(header))

    if skipped_words:
        print("\n[!] Các từ CHƯA ĐỦ mẫu (< 2 mẫu) để dò ngưỡng — GIỮ NGUYÊN threshold hiện tại:")
        for word_key, display_name, num_samples in skipped_words:
            print(f"    - {word_key} ({display_name}): {num_samples} mẫu -> cần thu thập thêm qua collect_dynamic_templates.py")

    if not args.apply:
        print("\n[Preview Mode] Chưa ghi gì vào file. Chạy lại với --apply để áp dụng ngưỡng mới.")
        return

    if not rows:
        print("\n[!] Không có từ nào đủ điều kiện để cập nhật ngưỡng. Không ghi file.")
        return

    # ------------------------------------------------------------------
    # Sao lưu file gốc trước khi ghi đè (theo quy ước đặt tên đã dùng trong repo,
    # VD: dynamic_templates.backup_v51.json)
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"dynamic_templates.backup_{timestamp}.json"
    shutil.copy2(templates_path, backup_path)
    print(f"\n[Backup] Đã sao lưu file gốc sang: {backup_path}")

    with open(templates_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0
    for r in rows:
        word_key = r["word_key"]
        if word_key in data.get("vocabulary", {}):
            data["vocabulary"][word_key]["threshold"] = round(r["proposed_threshold"], 3)
            updated_count += 1

    data.setdefault("metadata", {})["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(templates_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Apply] Đã ghi {updated_count} ngưỡng mới vào {templates_path}.")


if __name__ == "__main__":
    main()
