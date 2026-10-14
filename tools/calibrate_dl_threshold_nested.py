"""
HIỆU CHỈNH NGƯỠNG DL BẰNG NESTED CROSS-VALIDATION (không rò rỉ).

Bối cảnh: DEFAULT_CONF_THRESHOLD=0.32 trong dl_fusion.py trước đó được chọn bằng cách sweep
trên CHÍNH 3/4 người (Dang/Mai/Vuong) rồi lấy trung bình — tức ngưỡng không hoàn toàn độc lập
với chính những người dùng để báo cáo kết quả cuối (circularity nhẹ). Script này làm đúng cách:
với mỗi người lạ T, chọn ngưỡng CHỈ bằng dữ liệu của 3 người CÒN LẠI (không đụng vào T), rồi
đo accuracy Hybrid trên T bằng đúng ngưỡng đó — T không hề "thấy" được dùng để tự chọn ngưỡng
cho chính mình. Đây là ước lượng trung thực, không lạc quan hơn giá trị thật.

Cách dùng:  py calibrate_dl_threshold_nested.py
Kết quả: results/baselines/dl_threshold_nested_cv.json
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import json
import os
import sys
from typing import Any, Dict, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import dl_fusion
from train_dynamic_dl import load_all_sign_data
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config

PEOPLE = ["Dang", "Mai", "Phi", "Vuong"]


def collect_rows(person: str, cfg: Any, dat_bank: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chạy DTW + DL trên toàn bộ mẫu của `person`, dùng checkpoint LOSO CỦA CHÍNH họ
    (best_gesture_model_<person>.pth - không thấy mẫu của họ lúc train)."""
    weights_path = f"best_gesture_model_{person}.pth"
    bundle = dl_fusion.load_model(weights_path)
    if bundle is None:
        print(f"[!] Thiếu {weights_path}, bỏ qua {person}.")
        return []
    matcher = build_matcher_from_config(cfg, dat_bank)

    all_samples, _ = load_all_sign_data()
    samples = [s for s in all_samples if s["signer"] == person]

    rows: List[Dict[str, Any]] = []
    for s in samples:
        traj = s["traj"]
        true_word = s["word"]
        dl_pred = dl_fusion.predict(bundle, traj)
        dl_word, dl_conf = (dl_pred[0], dl_pred[1]) if dl_pred is not None else (None, 0.0)

        shapes = s.get("hand_shapes", ["Unknown"] * 32)
        handed = s.get("handedness", "Right")
        chir = bool(s.get("chirality_canonical", True))
        has_body = s.get("body_anchor_valid", True)
        rep = s.get("rep_count")
        r = matcher.match(
            query_traj_9d=np.asarray(traj, dtype=np.float32),
            query_shapes=shapes,
            handedness=("Right" if chir else handed),
            query_has_body=has_body,
            query_rep_count=rep,
        )
        rows.append({"status": r.status, "true_word": true_word, "dtw_word": r.top_word,
                     "dl_word": dl_word, "dl_conf": dl_conf})
    return rows


def accuracy_at_threshold(rows: List[Dict[str, Any]], threshold: float) -> float:
    if not rows:
        return 0.0
    correct = sum(
        1 for r in rows
        if dl_fusion.gated_prediction(r["status"], r["dtw_word"], r["dl_word"], r["dl_conf"], threshold) == r["true_word"]
    )
    return correct / len(rows)


def main() -> None:
    cfg, _ = validate_and_hash_config("configs/G1p90.json")
    dat_bank, _ = load_bank("webcam_templates.json")

    print("Đang thu thập DTW+DL trên từng người (dùng checkpoint LOSO của chính họ)...")
    rows_by_person: Dict[str, List[Dict[str, Any]]] = {}
    for p in PEOPLE:
        rows_by_person[p] = collect_rows(p, cfg, dat_bank)
        print(f"  {p}: {len(rows_by_person[p])} mẫu")

    out: Dict[str, Any] = {"per_target": {}}
    print("\n" + "=" * 78)
    print("NESTED CV: ngưỡng cho mỗi người CHỈ chọn từ 3 người CÒN LẠI (không rò rỉ)")
    print("=" * 78)
    print(f"{'Người lạ':<10}{'Nguong ngoai (nested)':>22}{'Acc @ nested':>14}"
          f"{f'Acc @ {dl_fusion.DEFAULT_CONF_THRESHOLD:.2f} (global)':>20}")

    nested_accs = []
    global_accs = []
    for target in PEOPLE:
        target_rows = rows_by_person[target]
        if not target_rows:
            continue
        other_rows: List[Dict[str, Any]] = []
        for o in PEOPLE:
            if o != target:
                other_rows += rows_by_person[o]

        sweep = dl_fusion.sweep_confidence_threshold(other_rows)
        nested_thr = sweep["best"]["threshold"]
        acc_nested = accuracy_at_threshold(target_rows, nested_thr)
        acc_global = accuracy_at_threshold(target_rows, dl_fusion.DEFAULT_CONF_THRESHOLD)

        nested_accs.append(acc_nested)
        global_accs.append(acc_global)

        print(f"{target:<10}{nested_thr:>22.2f}{100*acc_nested:>13.1f}%{100*acc_global:>19.1f}%")
        out["per_target"][target] = {
            "n": len(target_rows), "nested_threshold": nested_thr,
            "accuracy_at_nested_threshold": acc_nested,
            "accuracy_at_global_default": acc_global,
        }

    print("-" * 78)
    print(f"{'TRUNG BINH':<10}{'':>22}{100*np.mean(nested_accs):>13.1f}%{100*np.mean(global_accs):>19.1f}%")
    out["summary"] = {
        "mean_accuracy_nested_cv": float(np.mean(nested_accs)),
        "mean_accuracy_global_default": float(np.mean(global_accs)),
    }
    print("\nDiễn giải: cột 'Acc @ nested' là ước lượng TRUNG THỰC (không rò rỉ) cho accuracy Hybrid")
    print("nếu triển khai thật cho một người lạ hoàn toàn mới, chọn ngưỡng dựa trên dữ liệu của")
    print("những người đã có sẵn (không phải người đó). Nếu 2 cột gần nhau -> ngưỡng 0.32 KHÔNG")
    print("bị overfit vào các người cụ thể đã dùng để chọn nó.")

    out_path = os.path.join("results", "baselines", "dl_threshold_nested_cv.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {out_path}")


if __name__ == "__main__":
    main()
