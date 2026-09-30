"""
ĐÁNH GIÁ SO SÁNH HYBRID (DTW vs DEEP LEARNING vs HYBRID GATED) TRÊN NGƯỜI LẠ
==============================================================================
Tác giả: Tô Phương Đạt (24110086) - Dự án DIP Project

SỬA (2026-09-30): bản cũ blend xác suất theo alpha CỐ ĐỊNH (hybrid = 0.65*dl_probs +
0.35*dtw_scores, áp cho MỌI mẫu) và BỎ QUA HOÀN TOÀN result.status của DTW. Hậu quả đo
được trên results_hybrid_Dang.json: ở các từ DTW ĐÚNG còn DL SAI (BYE 3/4 dtw vs 2/4 dl,
EAT 4/4 vs 2/4, DRINK 4/5 vs 2/5), hybrid_accuracy TRÙNG KHỚP TUYỆT ĐỐI dl_accuracy - tức
alpha=0.65 luôn thắng, ghi đè oan DTW đúng ở mọi trường hợp, không hề "cân đo" gì cả.
Bản này thay bằng GATE theo trạng thái (xem dl_fusion.py, cùng luật đã kiểm chứng cho
handshape_fusion.py): DTW ACCEPTED -> luôn giữ DTW; chỉ khi DTW UNCERTAIN/REJECTED mới
xét DL, và chỉ nhận DL nếu độ tin cậy softmax của nó vượt ngưỡng đã hiệu chỉnh.

Mục đích:
  - Kiểm chứng khoa học giả thuyết: "Deep Learning + DTW giúp giải quyết triệt để bài toán Người Lạ".
  - So sánh 3 phương pháp trên CÙNG 1 tập dữ liệu kiểm thử (unseen signer):
      1. DTW thuần túy (GoldenTriadDTWMatcher - cấu hình G1p90, bank của Đạt)
      2. Deep Learning thuần túy (LightweightGestureNet - 1D-CNN + Bi-GRU + Attention)
      3. Hybrid GATED: chỉ dùng DL khi DTW không tự tin, có ngưỡng tin cậy.
  - In bảng đối sánh + quét ngưỡng tham khảo, xuất file báo cáo JSON.

Cách dùng:  py eval_dynamic_hybrid.py --test_signer Dang --conf-threshold 0.55 --sweep
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import dl_fusion
from train_dynamic_dl import load_all_sign_data
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config


def evaluate_hybrid_on_signer(
    test_signer: str = "Dang",
    model_weights_path: Optional[str] = None,
    dtw_config_path: str = "configs/G1p90.json",
    bank_path: str = "webcam_templates.json",
    conf_threshold: float = dl_fusion.DEFAULT_CONF_THRESHOLD,
    sweep: bool = False,
) -> None:
    print("=" * 75)
    print(f"  ĐÁNH GIÁ SO SÁNH TRÊN NGƯỜI LẠ: [{test_signer}]")
    print(f"  Cấu hình: Hybrid GATED, ngưỡng tin cậy DL = {conf_threshold:.2f}")
    print("=" * 75)

    # 1. Nạp toàn bộ dữ liệu và lọc ra tập test của test_signer
    all_samples, _ = load_all_sign_data()
    test_samples = [s for s in all_samples if s["signer"] == test_signer]
    if len(test_samples) == 0:
        print(f"[LỖI] Không tìm thấy mẫu nào của người ký [{test_signer}]!")
        return
    print(f"-> Đã tìm thấy {len(test_samples)} mẫu cử chỉ của [{test_signer}].")

    # 2. Nạp mô hình Deep Learning (qua dl_fusion, cache theo đường dẫn)
    if model_weights_path is None or not os.path.exists(model_weights_path):
        model_weights_path = f"best_gesture_model_{test_signer}.pth"
    bundle = dl_fusion.load_model(model_weights_path)
    if bundle is None:
        print(f"[!] Chưa có trọng số mô hình tại {model_weights_path}.")
        print(f"    Vui lòng chạy: py train_dynamic_dl.py --test_signer {test_signer} trước!")
        return
    print(f"-> Đã nạp thành công mô hình Deep Learning: {model_weights_path}")

    # 3. Khởi tạo bộ so khớp DTW (bank Đạt, cấu hình G1p90 - giống baseline sản xuất)
    if not (os.path.exists(dtw_config_path) and os.path.exists(bank_path)):
        print(f"[LỖI] Thiếu {dtw_config_path} hoặc {bank_path}.")
        return
    cfg, _ = validate_and_hash_config(dtw_config_path)
    dat_bank, _ = load_bank(bank_path)
    dtw_matcher = build_matcher_from_config(cfg, dat_bank)
    print(f"-> Đã nạp thành công bộ so khớp DTW: {dtw_config_path} + {bank_path}")

    # 4. Thực hiện đánh giá từng mẫu
    dtw_correct = dl_correct = hybrid_correct = 0
    per_word_stats: Dict[str, Dict[str, int]] = {}
    sweep_rows: List[Dict[str, Any]] = []

    for s in test_samples:
        true_word = s["word"]
        traj = s["traj"]  # (32, 24), đã chuẩn hóa chirality tại nguồn (xem load_all_sign_data)
        st = per_word_stats.setdefault(true_word, {"total": 0, "dtw": 0, "dl": 0, "hybrid": 0})
        st["total"] += 1

        # A. Dự đoán qua Deep Learning
        dl_pred = dl_fusion.predict(bundle, traj)
        if dl_pred is not None:
            dl_word, dl_conf, _dl_probs = dl_pred
        else:
            dl_word, dl_conf = None, 0.0
        if dl_word == true_word:
            dl_correct += 1
            st["dl"] += 1

        # B. Dự đoán qua DTW (đúng convention _dtw_call: query đã canonical -> ép handedness="Right")
        shapes = s.get("hand_shapes", ["Unknown"] * 32)
        handed = s.get("handedness", "Right")
        chir = bool(s.get("chirality_canonical", True))
        has_body = s.get("body_anchor_valid", True)
        rep = s.get("rep_count")
        dtw_res = dtw_matcher.match(
            query_traj_9d=np.asarray(traj, dtype=np.float32),
            query_shapes=shapes,
            handedness=("Right" if chir else handed),
            query_has_body=has_body,
            query_rep_count=rep,
        )
        dtw_word = dtw_res.top_word
        if dtw_word == true_word:
            dtw_correct += 1
            st["dtw"] += 1

        # C. Hybrid GATED: DTW ACCEPTED -> giữ DTW; chỉ khi không ACCEPTED mới xét DL theo ngưỡng
        hybrid_word = dl_fusion.gated_prediction(dtw_res.status, dtw_word, dl_word, dl_conf, conf_threshold)
        if hybrid_word == true_word:
            hybrid_correct += 1
            st["hybrid"] += 1

        sweep_rows.append({"status": dtw_res.status, "true_word": true_word,
                            "dtw_word": dtw_word, "dl_word": dl_word, "dl_conf": dl_conf})

    total = len(test_samples)
    dtw_acc = 100.0 * dtw_correct / total
    dl_acc = 100.0 * dl_correct / total
    hybrid_acc = 100.0 * hybrid_correct / total

    # 5. In bảng kết quả so sánh
    print("\n" + "=" * 75)
    print(f"  BẢNG SO SÁNH ĐỘ CHÍNH XÁC (TOP-1 ACCURACY) TRÊN [{test_signer}]")
    print("=" * 75)
    print(f"  1. DTW Đơn thuần (Bank Đạt)         : {dtw_correct}/{total} mẫu ({dtw_acc:5.2f}%)")
    print(f"  2. Deep Learning Đa nguồn (CNN+GRU)  : {dl_correct}/{total} mẫu ({dl_acc:5.2f}%)")
    print(f"  3. HYBRID GATED (ngưỡng={conf_threshold:.2f})       : {hybrid_correct}/{total} mẫu ({hybrid_acc:5.2f}%)")
    print("-" * 75)
    print(f"{'Từ vựng':<12} | {'Tổng mẫu':<9} | {'DTW':<12} | {'Deep Learning':<15} | {'HYBRID GATED':<15}")
    print("-" * 75)

    for w, st in sorted(per_word_stats.items()):
        n = st["total"]
        dtw_str = f"{st['dtw']}/{n} ({(st['dtw']/n)*100:4.0f}%)"
        dl_str = f"{st['dl']}/{n} ({(st['dl']/n)*100:4.0f}%)"
        hy_str = f"{st['hybrid']}/{n} ({(st['hybrid']/n)*100:4.0f}%)"
        print(f"{w:<12} | {n:<9} | {dtw_str:<12} | {dl_str:<15} | {hy_str:<15}")

    print("=" * 75)

    n_worse = sum(1 for st in per_word_stats.values() if st["hybrid"] < max(st["dtw"], st["dl"]))
    if n_worse:
        print(f"[CẢNH BÁO] {n_worse} từ có Hybrid TỆ HƠN cả DTW lẫn DL riêng lẻ — kiểm tra lại gate!")
    else:
        print("[OK] Hybrid không tệ hơn max(DTW, DL) ở bất kỳ từ nào (đúng kỳ vọng của gate).")

    # 6. Quét ngưỡng tham khảo (KHÔNG dùng để chọn ngưỡng sản xuất từ chính tập test này)
    sweep_result = None
    if sweep:
        sweep_result = dl_fusion.sweep_confidence_threshold(sweep_rows)
        print(f"\n=== QUÉT NGƯỠNG TIN CẬY DL (chỉ THAM KHẢO — đây là tập test của [{test_signer}], "
              f"đừng lấy ngưỡng tốt nhất tìm được ở đây làm ngưỡng sản xuất, phải hiệu chỉnh trên "
              f"tập validation RIÊNG, ví dụ gộp vài người khác) ===")
        print(f"  Ngưỡng tốt nhất trên tập này: {sweep_result['best']['threshold']:.2f} "
              f"(accuracy {sweep_result['best']['accuracy']*100:.1f}%)")

    # 7. Ghi kết quả so sánh vào file JSON
    out_file = f"results_hybrid_{test_signer}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_signer": test_signer,
            "total_samples": total,
            "conf_threshold": conf_threshold,
            "dtw_accuracy": dtw_acc,
            "dl_accuracy": dl_acc,
            "hybrid_accuracy": hybrid_acc,
            "per_word": per_word_stats,
            "threshold_sweep": sweep_result,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu chi tiết so sánh tại: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh DTW vs Deep Learning vs Hybrid GATED trên người lạ")
    parser.add_argument("--test_signer", type=str, default="Dang", help="Tên người ký cần đánh giá (LOSO)")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file trọng số .pth")
    parser.add_argument("--conf-threshold", type=float, default=dl_fusion.DEFAULT_CONF_THRESHOLD,
                         help="Ngưỡng tin cậy softmax tối thiểu để DL được nhận khi DTW không ACCEPTED")
    parser.add_argument("--sweep", action="store_true", help="In thêm bảng quét ngưỡng tham khảo")
    args = parser.parse_args()

    evaluate_hybrid_on_signer(
        test_signer=args.test_signer,
        model_weights_path=args.weights,
        conf_threshold=args.conf_threshold,
        sweep=args.sweep,
    )
