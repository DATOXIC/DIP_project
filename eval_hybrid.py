"""
Thử kết hợp GRU (chọn từ) + DTW (quyết định nhận/từ chối), trên CÙNG bank, cùng cách chia như eval_baselines.py.

Lý do: eval_baselines/eval_learning_curve cho thấy GRU chọn đúng từ ngang/hơn DTW, còn DTW từ chối từ lạ tốt hơn GRU
(ở dữ liệu đầy đủ). Câu hỏi: lấy nhãn của GRU và cổng từ chối của DTW có được cả hai ưu điểm không?

Các biến thể (cùng phép đo LOSO / Mai / LOWO / NEG). Thêm cuu_UNC@t / cuu_moi@t: DTW chưa nhận nhưng GRU đồng ý từ DTW chọn với xác suất >= t thì nhận.
  dtw           : nhãn = DTW, nhận nếu DTW ACCEPTED (hệ thống hiện tại).
  gru@ngưỡng    : nhãn = GRU, nhận nếu xác suất tối đa >= ngưỡng (chỉnh để LOSO nhận-đúng = dtw).
  hybrid_cong   : nhãn = GRU, nhận nếu DTW ACCEPTED (cổng của DTW, bất kể DTW chọn từ nào).
  hybrid_dong_y : nhãn = GRU, nhận nếu DTW ACCEPTED VÀ DTW chọn cùng từ với GRU (hai bên đồng ý).
Chỉ số: nhận-đúng / nhận-sai trên LOSO và Mai; nhận nhầm trên LOWO (từ lạ) và NEG (nét âm); top-1 ép chọn của nhãn GRU.
GRU: trung bình qua nhiều hạt giống (mặc định 3).

Cách dùng:  py eval_hybrid.py --dtw-config configs/G1p90.json --seeds 3
Kết quả: results/baselines/hybrid.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from eval_baselines import MODELS, _dtw_call, _is_orig, _traj, bank_arrays
from scoreboard import (
    build_matcher_from_config,
    load_bank,
    load_negative_strokes,
    split_dataset,
    validate_and_hash_config,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="GRU chọn từ + DTW quyết định nhận/từ chối.")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--neg", default="negative_strokes.json")
    p.add_argument("--signer", default="Mai")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--dtw-config", default="configs/G1p90.json")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default=os.path.join("results", "baselines", "hybrid.json"))
    args = p.parse_args()

    bank, _ = load_bank(args.bank)
    neg, _ = load_negative_strokes(args.neg)
    sp = os.path.join(args.profiles_dir, f"{args.signer}.json")
    signer_vocab = None
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as fh:
            signer_vocab = json.load(fh).get("vocabulary", {})
    cfg, _ = validate_and_hash_config(args.dtw_config)
    words = sorted(bank["vocabulary"].keys())
    idx = {w: i for i, w in enumerate(words)}
    C = len(words)
    _, folds = split_dataset(bank)

    # ---- danh sách truy vấn theo nhóm mô hình: (tập, nhóm, từ_thật hoặc None, traj, kwargs cho DTW)
    def dkw(s):
        return dict(shapes=s.get("hand_shapes", ["Unknown"] * 32), handed=s.get("handedness"),
                    chir=bool(s.get("chirality_canonical")), has_body=s.get("body_anchor_valid", True), rep=s.get("rep_count"))

    Q: List[Dict[str, Any]] = []
    for k, f in enumerate(folds):
        for q in f.queries:
            Q.append(dict(set="loso", group=("fold", k), true=q.true_word, traj=np.nan_to_num(np.asarray(q.traj, np.float32)),
                          d=dict(shapes=q.hand_shapes, handed=q.handedness, chir=q.chirality_canonical, has_body=q.has_body, rep=q.rep_count)))
    if signer_vocab:
        for w, d in signer_vocab.items():
            for s in d.get("samples", []):
                Q.append(dict(set="v01", group=("full",), true=w, traj=_traj(s), d=dkw(s)))
    for s in neg:
        Q.append(dict(set="neg", group=("full",), true=None, traj=_traj(s), d=dkw(s)))
    for W in words:
        for s in bank["vocabulary"][W]["samples"]:
            if _is_orig(s):
                Q.append(dict(set="lowo", group=("lowo", W), true=None, traj=_traj(s), d=dkw(s)))

    def ref_bank(group):
        if group[0] == "fold":
            return folds[group[1]].reference_bank
        if group[0] == "full":
            return bank
        return {"metadata": bank.get("metadata", {}),
                "vocabulary": {k: v for k, v in bank["vocabulary"].items() if k != group[1]}}

    groups = sorted({q["group"] for q in Q}, key=lambda g: (g[0], str(g[1]) if len(g) > 1 else ""))
    t0 = time.time()

    # ---- DTW (tất định)
    for g in groups:
        m = build_matcher_from_config(cfg, ref_bank(g))
        for q in Q:
            if q["group"] != g:
                continue
            r = _dtw_call(m, q["traj"], q["d"]["shapes"], q["d"]["handed"], q["d"]["chir"], q["d"]["has_body"], q["d"]["rep"])
            q["dtw_top"] = r.candidates[0].word_key if r.candidates else None
            q["dtw_acc"] = bool(r.status == "ACCEPTED")
            q["dtw_stat"] = str(r.status)
    print(f"[{time.time() - t0:4.0f}s] DTW xong ({len(Q)} truy vấn)", flush=True)

    set_of = np.array([q["set"] for q in Q])
    true = np.array([q["true"] if q["true"] is not None else "" for q in Q])
    dtw_top = np.array([q["dtw_top"] or "" for q in Q])
    dtw_acc = np.array([q["dtw_acc"] for q in Q])
    dtw_stat = np.array([q["dtw_stat"] for q in Q])

    def counts(pred: np.ndarray, acc: np.ndarray) -> Dict[str, float]:
        out = {}
        for s in ("loso", "v01"):
            m = set_of == s
            ok = pred[m] == true[m]
            out[f"{s}_n"] = int(m.sum())
            out[f"{s}_top1"] = float(ok.mean())
            out[f"{s}_acc_ok"] = int((acc[m] & ok).sum())
            out[f"{s}_acc_wrong"] = int((acc[m] & ~ok).sum())
        for s in ("lowo", "neg"):
            m = set_of == s
            out[f"{s}_n"] = int(m.sum())
            out[f"{s}_fa"] = int(acc[m].sum())
        return out

    res: Dict[str, List[Dict[str, float]]] = {"dtw": [counts(dtw_top, dtw_acc)]}
    target = res["dtw"][0]["loso_acc_ok"]

    for sd in range(args.seeds):
        pred = np.zeros(len(Q), dtype=object)
        conf = np.zeros(len(Q))
        for g in groups:
            X, y = bank_arrays(ref_bank(g), words)
            m = MODELS["gru"]().fit(X, y, C, 1000 + sd)
            sel = [i for i, q in enumerate(Q) if q["group"] == g]
            pi, sc = m.predict(np.stack([Q[i]["traj"] for i in sel]))
            for i, a, b in zip(sel, pi, sc):
                pred[i] = words[int(a)]
                conf[i] = float(b)
        pred = pred.astype(str)
        # gru@ngưỡng: chỉnh để LOSO nhận-đúng = dtw
        ml = set_of == "loso"
        cs = np.sort(conf[ml & (pred == true)])[::-1]
        tau = cs[min(target, len(cs)) - 1]
        res.setdefault("gru@ngưỡng", []).append(counts(pred, conf >= tau))
        res.setdefault("hybrid_cong", []).append(counts(pred, dtw_acc))
        res.setdefault("hybrid_dong_y", []).append(counts(pred, dtw_acc & (dtw_top == pred)))
        agree = dtw_top == pred
        for tg in (0.6, 0.8, 0.9):
            # cứu: DTW chưa nhận (UNCERTAIN) nhưng GRU đồng ý với từ DTW chọn và rất tự tin
            res.setdefault(f"cuu_UNC@{tg}", []).append(counts(pred, dtw_acc | ((dtw_stat == "UNCERTAIN") & agree & (conf >= tg))))
            # cứu rộng: cả UNCERTAIN lẫn REJECTED
            res.setdefault(f"cuu_moi@{tg}", []).append(counts(pred, dtw_acc | (agree & (conf >= tg))))
        print(f"[{time.time() - t0:4.0f}s] GRU hạt {sd} xong", flush=True)

    def m(name: str, key: str) -> float:
        return float(np.mean([r[key] for r in res[name]]))

    print("\nĐơn vị: số truy vấn (trung bình qua hạt giống với GRU). Nhận nhầm càng thấp càng tốt.")
    n = res["dtw"][0]
    print(f"{'biến thể':<16}{'LOSO top-1':>11}{'LOSO đúng':>11}{'LOSO sai':>10}{'Mai top-1':>11}"
          f"{'Mai đúng':>10}{'Mai sai':>9}{'từ lạ nhầm':>12}{'nét âm nhầm':>13}")
    for name in res:
        print(f"{name:<16}{100 * m(name, 'loso_top1'):>10.1f}%{m(name, 'loso_acc_ok'):>8.1f}/{n['loso_n']:<3}{m(name, 'loso_acc_wrong'):>8.1f}"
              f"{100 * m(name, 'v01_top1'):>10.1f}%{m(name, 'v01_acc_ok'):>7.1f}/{n['v01_n']:<3}{m(name, 'v01_acc_wrong'):>7.1f}"
              f"{m(name, 'lowo_fa'):>8.1f}/{n['lowo_n']:<4}{m(name, 'neg_fa'):>8.1f}/{n['neg_n']:<4}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"dtw_config": args.dtw_config, "seeds": args.seeds, "results": res}, fh, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}  (tổng {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
