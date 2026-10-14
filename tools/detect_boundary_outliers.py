"""
PHÁT HIỆN MẪU GÂY NHIỄU BIÊN GIỚI + AUDIT ĐỘ TIN CẬY DÁNG TAY (c_shape)
=========================================================================
Phần 1: Với mỗi mẫu, kiểm tra mẫu này có đang GẦN mẫu của TỪ KHÁC hơn là
gần mẫu CÙNG TỪ với nó hay không (mẫu "lạc" kéo 2 từ lại gần nhau).

Phần 2: Với mỗi từ, thống kê tỷ lệ frame "Unknown" và độ nhất quán nhãn
dáng tay chiếm đa số — để quyết định có nên tiếp tục tin cậy lambda_shape
hay nên tắt/giảm nó cho từ đó (hoặc toàn hệ thống).

Cách dùng:
  python detect_boundary_outliers.py --templates dynamic_templates.json
  python detect_boundary_outliers.py --templates asl_citizen_templates.json
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple, Any

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


def load_sample_traj(sample: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    traj_raw = sample.get("traj_features_9d") or sample.get("traj_features", [])
    traj = np.array(traj_raw, dtype=np.float32)
    # CHUẨN HÓA CHIRALITY bằng CÙNG hàm mà GoldenTriadDTWMatcher.load_templates() dùng,
    # để ngưỡng/audit tính offline nằm trong ĐÚNG không gian dấu mà production so khớp.
    # Nếu bỏ bước này, mọi con số dưới đây đo trên một không gian khác với lúc chạy thật.
    if not sample.get("chirality_canonical", False):
        traj, _ = canonicalize_chirality(traj, handedness=sample.get("handedness"))
    shapes = sample.get("hand_shapes", ["Unknown"] * len(traj))
    return traj, shapes


def audit_shape_reliability(word_samples: Dict[str, List[Tuple[np.ndarray, List[str]]]]) -> None:
    """PHẦN 2: Thống kê độ tin cậy nhãn dáng tay (c_shape) theo từng từ."""
    header = f"{'word':<14}{'n_samples':>10}{'unknown_pct':>14}{'dominant_shape':<16}{'consistency_pct':>17}"
    print("\n" + "=" * len(header))
    print("AUDIT ĐỘ TIN CẬY C_SHAPE (dáng tay từ CNN Giai đoạn 1)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    low_confidence_words = []

    for word_key, samples in word_samples.items():
        if not samples:
            continue
        all_shapes = []
        for _, shapes in samples:
            all_shapes.extend(shapes)

        total = len(all_shapes)
        if total == 0:
            continue

        unknown_count = sum(1 for s in all_shapes if s == "Unknown")
        unknown_pct = (unknown_count / total) * 100.0

        known_shapes = [s for s in all_shapes if s != "Unknown"]
        if known_shapes:
            counter = Counter(known_shapes)
            dominant_shape, dominant_count = counter.most_common(1)[0]
            consistency_pct = (dominant_count / len(known_shapes)) * 100.0
        else:
            dominant_shape, consistency_pct = "N/A", 0.0

        print(f"{word_key:<14}{len(samples):>10}{unknown_pct:>13.1f}%  {dominant_shape:<14}{consistency_pct:>17.1f}%")

        # Ngưỡng nghi vấn: nhiều Unknown HOẶC dáng tay không nhất quán
        if unknown_pct > 15.0 or consistency_pct < 60.0:
            low_confidence_words.append((word_key, unknown_pct, consistency_pct))

    print("=" * len(header))

    if low_confidence_words:
        print(f"\n[!] {len(low_confidence_words)} từ có tín hiệu c_shape KHÔNG đáng tin cậy:")
        for word_key, u_pct, c_pct in low_confidence_words:
            print(f"    - {word_key}: Unknown={u_pct:.1f}%, độ nhất quán={c_pct:.1f}%")
        print("    Gợi ý: cân nhắc giảm/tắt lambda_shape cho các từ này, hoặc toàn hệ thống")
        print("    nếu phần lớn từ đều nằm trong danh sách này.")
    else:
        print("\n[OK] c_shape có vẻ đáng tin cậy cho toàn bộ từ vựng hiện tại.")


def detect_outliers(matcher: GoldenTriadDTWMatcher, word_samples: Dict[str, List[Tuple[np.ndarray, List[str]]]]) -> None:
    """PHẦN 1: Phát hiện mẫu gây nhiễu biên giới (như đã đặc tả trước đó)."""
    suspicious_rows = []

    for word_key, samples in word_samples.items():
        if len(samples) < 2:
            continue
        wp_own, wa_own, wm_own, wn_own = matcher._get_weights_for_sign(word_key)

        for idx, (q_traj, q_shapes) in enumerate(samples):
            own_dists = []
            for j, (t_traj, t_shapes) in enumerate(samples):
                if j == idx:
                    continue
                d = compute_golden_triad_dtw_distance(
                    q_traj=q_traj, q_shapes=q_shapes, t_traj=t_traj, t_shapes=t_shapes,
                    w_p=wp_own, w_a=wa_own, w_m=wm_own, w_n=wn_own,
                    lambda_shape=matcher.lambda_shape,
                    lambda_shape_global=matcher.lambda_shape_global
                )
                own_dists.append(d)
            if not own_dists:
                continue
            dist_own_min = min(own_dists)
            dist_own_avg = sum(own_dists) / len(own_dists)

            best_other_word = None
            dist_other_min = math.inf
            for other_key, other_samples in word_samples.items():
                if other_key == word_key or not other_samples:
                    continue
                wp_o, wa_o, wm_o, wn_o = matcher._get_weights_for_sign(other_key)
                for t_traj, t_shapes in other_samples:
                    d = compute_golden_triad_dtw_distance(
                        q_traj=q_traj, q_shapes=q_shapes, t_traj=t_traj, t_shapes=t_shapes,
                        w_p=wp_o, w_a=wa_o, w_m=wm_o, w_n=wn_o,
                        lambda_shape=matcher.lambda_shape,
                        lambda_shape_global=matcher.lambda_shape_global
                    )
                    if d < dist_other_min:
                        dist_other_min = d
                        best_other_word = other_key

            if best_other_word is not None and dist_other_min < dist_own_min:
                src_video = "N/A"
                if word_key in matcher.vocabulary and "samples" in matcher.vocabulary[word_key]:
                    raw_samples = matcher.vocabulary[word_key]["samples"]
                    if idx < len(raw_samples):
                        src_video = raw_samples[idx].get("source_video", "N/A")

                suspicious_rows.append({
                    "word": word_key, "sample_idx": idx,
                    "dist_own": dist_own_min, "dist_other_min": dist_other_min,
                    "nearest_other_word": best_other_word,
                    "source_video": src_video
                })

    header = f"{'word':<12}{'idx':>5}{'dist_own_min':>14}{'dist_other_min':>16}{'nearest_other':<16}{'source_video':<30}"
    print("=" * len(header))
    print("PHÁT HIỆN MẪU GÂY NHIỄU BIÊN GIỚI")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    if not suspicious_rows:
        print("[OK] Không phát hiện mẫu nào bị lệch gần từ khác hơn từ của chính nó.")
    else:
        for r in suspicious_rows:
            print(f"{r['word']:<12}{r['sample_idx']:>5}{r['dist_own']:>10.4f}{r['dist_other_min']:>12.4f}"
                  f"  {r['nearest_other_word']:<14}{str(r['source_video'])[:28]:<30}")
        print("-" * len(header))
        print(f"[!] Tổng {len(suspicious_rows)} mẫu nghi vấn. Xem xét xóa qua collect_dynamic_templates.py")
        print("    (Undo/xóa) hoặc manage_asl_citizen_samples.py --delete <idx>.")
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", default="dynamic_templates.json")
    args = parser.parse_args()

    if not os.path.exists(args.templates):
        print(f"[!] Không tìm thấy file template: {args.templates}")
        sys.exit(1)

    matcher = GoldenTriadDTWMatcher(templates_path=args.templates, strict_weights=False)
    vocabulary = matcher.vocabulary

    word_samples: Dict[str, List[Tuple[np.ndarray, List[str]]]] = {}
    for word_key, word_data in vocabulary.items():
        parsed = []
        for s in word_data.get("samples", []):
            traj, shapes = load_sample_traj(s)
            if len(traj) == 32:
                parsed.append((traj, shapes))
        word_samples[word_key] = parsed

    detect_outliers(matcher, word_samples)
    audit_shape_reliability(word_samples)


if __name__ == "__main__":
    main()
