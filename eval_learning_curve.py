"""
Đường cong học: khi mỗi từ chỉ có n mẫu tham chiếu/huấn luyện (n = 1, 2, 3, 5, 8, ...), phương pháp nào còn nhận đúng?

Giả thuyết cần kiểm: DTW (so khớp mẫu, không huấn luyện) hơn mô hình học máy (GRU/MLP) KHI ÍT MẪU — lý do khoa học để
"cá nhân hóa chỉ cần vài mẫu". Nếu giả thuyết sai, kết quả cũng được báo cáo như vậy.

Thiết kế (khớp eval_baselines.py):
  - Cùng 7 lượt LOSO theo phiên; trong MỖI lượt, phía tham chiếu/huấn luyện chỉ giữ ngẫu nhiên n mẫu GỐC mỗi từ
    (query giữ nguyên toàn bộ). Người lạ Mai và nét âm dùng bank đầy đủ đã lấy n mẫu/từ.
  - Mỗi (n, lần lặp) dùng CÙNG tập mẫu con cho mọi phương pháp (hạt giống tất định), lặp R lần rồi lấy trung bình ± độ lệch chuẩn.
  - Chỉ số: LOSO top-1, Mai top-1, AUROC (từ đã biết = LOSO, từ lạ = nét âm). KHÔNG đo LOWO ở đây (phải huấn luyện lại 13 lần,
    quá chậm cho GRU); AUROC nét âm là đại diện của khả năng từ chối.
  - Riêng DTW còn báo tỉ lệ "nhận đúng" (ACCEPTED và đúng từ) và số nét âm bị nhận nhầm ở ngưỡng riêng của nó.
LƯU Ý: với n nhỏ độ rộng nội lớp của DTW ước lượng kém (n=1 không tính được) — ngưỡng chấp nhận vì thế không đáng tin ở n nhỏ;
top-1 và AUROC là phép đo công bằng hơn.

Cách dùng:
    py eval_learning_curve.py --ns 1 2 3 5 8 --repeats 3
Kết quả: results/baselines/learning_curve.json (+ learning_curve.png nếu có matplotlib)
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from eval_baselines import (
    MODELS,
    _dtw_call,
    _is_orig,
    _traj,
    auroc,
    bank_arrays,
)
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


def subsample_bank(bank: Dict[str, Any], n: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Bản sao bank chỉ giữ n mẫu GỐC ngẫu nhiên mỗi từ (từ có ít hơn n mẫu thì giữ hết)."""
    out = {k: v for k, v in bank.items() if k != "vocabulary"}
    out["vocabulary"] = {}
    for w, d in bank["vocabulary"].items():
        orig = [s for s in d.get("samples", []) if _is_orig(s)]
        pick = sorted(rng.choice(len(orig), size=min(n, len(orig)), replace=False)) if orig else []
        nd = {k: v for k, v in d.items() if k != "samples"}
        nd["samples"] = [copy.deepcopy(orig[i]) for i in pick]
        nd["num_samples"] = len(nd["samples"])
        out["vocabulary"][w] = nd
    return out


def run_once(n: int, rep: int, methods: List[str], cfg, bank, folds, neg, signer_vocab, words) -> Dict[str, Dict[str, float]]:
    C = len(words)
    idx = {w: i for i, w in enumerate(words)}
    res: Dict[str, Dict[str, float]] = {}
    subs = [subsample_bank(f.reference_bank, n, np.random.default_rng([20260928, rep, n, k])) for k, f in enumerate(folds)]
    full_sub = subsample_bank(bank, n, np.random.default_rng([20260928, rep, n, 999]))

    v_items = []
    if signer_vocab:
        for w, d in signer_vocab.items():
            for s in d.get("samples", []):
                v_items.append((w, s))
    Xv = np.stack([_traj(s) for _, s in v_items]) if v_items else None
    Xn = np.stack([_traj(s) for s in neg]) if neg else None

    for mn in methods:
        loso_ok, loso_s, loso_acc = [], [], []
        if mn == "dtw":
            for f, sub in zip(folds, subs):
                m = build_matcher_from_config(cfg, sub)
                for q in f.queries:
                    r = _dtw_call(m, q.traj, q.hand_shapes, q.handedness, q.chirality_canonical, q.has_body, q.rep_count)
                    top = r.candidates[0].word_key if r.candidates else None
                    loso_ok.append(top == q.true_word); loso_s.append(-float(r.r_ratio))
                    loso_acc.append(r.status == "ACCEPTED" and top == q.true_word)
            fm = build_matcher_from_config(cfg, full_sub)

            def dcall(s):
                return _dtw_call(fm, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                                 bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
            v_ok, v_acc = [], []
            for w, s in v_items:
                r = dcall(s)
                top = r.candidates[0].word_key if r.candidates else None
                v_ok.append(top == w); v_acc.append(r.status == "ACCEPTED" and top == w)
            n_s, n_acc = [], []
            for s in neg:
                r = dcall(s)
                n_s.append(-float(r.r_ratio)); n_acc.append(r.status == "ACCEPTED")
            res[mn] = {"loso_top1": float(np.mean(loso_ok)), "v01_top1": float(np.mean(v_ok)) if v_ok else float("nan"),
                       "auroc_neg": auroc(np.asarray(loso_s), np.asarray(n_s)),
                       "loso_accept_correct": float(np.mean(loso_acc)),
                       "v01_accept_correct": float(np.mean(v_acc)) if v_acc else float("nan"),
                       "neg_false_accept": float(np.sum(n_acc))}
        else:
            seed = 1000 + rep
            for f, sub in zip(folds, subs):
                X, y = bank_arrays(sub, words)
                m = MODELS[mn]().fit(X, y, C, seed)
                Xq = np.stack([np.nan_to_num(np.asarray(q.traj, dtype=np.float32)) for q in f.queries])
                pred, sc = m.predict(Xq)
                loso_ok += [bool(p == idx.get(q.true_word, -1)) for p, q in zip(pred, f.queries)]
                loso_s += list(sc)
            X, y = bank_arrays(full_sub, words)
            fm = MODELS[mn]().fit(X, y, C, seed)
            v_ok = []
            if Xv is not None:
                pred, _ = fm.predict(Xv)
                v_ok = [bool(p == idx.get(w, -1)) for p, (w, _) in zip(pred, v_items)]
            n_s = list(fm.predict(Xn)[1]) if Xn is not None else []
            res[mn] = {"loso_top1": float(np.mean(loso_ok)), "v01_top1": float(np.mean(v_ok)) if v_ok else float("nan"),
                       "auroc_neg": auroc(np.asarray(loso_s), np.asarray(n_s))}
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="Đường cong học: DTW vs baseline theo số mẫu mỗi từ.")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--neg", default="negative_strokes.json")
    p.add_argument("--signer", default="Mai")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--dtw-config", default="configs/G1p90.json")
    p.add_argument("--models", nargs="+", default=["dtw", "knn", "mlp", "gru"], choices=["dtw", "knn", "mlp", "gru"])
    p.add_argument("--ns", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=os.path.join("results", "baselines", "learning_curve.json"))
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
    _, folds = split_dataset(bank)
    print(f"{len(words)} từ | {len(folds)} lượt LOSO | n = {args.ns} | lặp {args.repeats} lần | phương pháp: {args.models}", flush=True)

    t0 = time.time()
    raw: Dict[str, Dict[int, List[Dict[str, float]]]] = {m: {n: [] for n in args.ns} for m in args.models}
    for n in args.ns:
        for rep in range(args.repeats):
            r = run_once(n, rep, args.models, cfg, bank, folds, neg, signer_vocab, words)
            for m, v in r.items():
                raw[m][n].append(v)
            print(f"[{time.time() - t0:5.0f}s] n={n} lặp {rep} xong", flush=True)

    keys = ["loso_top1", "v01_top1", "auroc_neg"]
    names = {"loso_top1": "LOSO top-1 (%)", "v01_top1": "Mai (người lạ) top-1 (%)", "auroc_neg": "AUROC nét âm"}
    summary: Dict[str, Any] = {}
    for k in keys + ["loso_accept_correct", "v01_accept_correct", "neg_false_accept"]:
        summary[k] = {}
        for m in args.models:
            summary[k][m] = {}
            for n in args.ns:
                v = [x[k] for x in raw[m][n] if k in x and not np.isnan(x[k])]
                if v:
                    summary[k][m][n] = (float(np.mean(v)), float(np.std(v)))
    for k in keys:
        pct = k != "auroc_neg"
        print(f"\n=== {names[k]} theo số mẫu mỗi từ (n) ===")
        print(f"{'n':<6}" + "".join(f"{m:>18}" for m in args.models))
        for n in args.ns:
            row = f"{n:<6}"
            for m in args.models:
                mu, sd = summary[k][m].get(n, (float("nan"), 0.0))
                row += f"{(f'{100 * mu:.1f}±{100 * sd:.1f}' if pct else f'{mu:.3f}±{sd:.3f}'):>18}"
            print(row)
    if "dtw" in args.models:
        print("\n=== DTW ở ngưỡng riêng (nhận-đúng = ACCEPTED và đúng từ; ngưỡng kém tin cậy khi n nhỏ) ===")
        print(f"{'n':<6}{'LOSO nhận đúng':>18}{'Mai nhận đúng':>18}{'nét âm nhận nhầm/41':>22}")
        for n in args.ns:
            a = summary["loso_accept_correct"]["dtw"].get(n, (float("nan"), 0))
            b = summary["v01_accept_correct"]["dtw"].get(n, (float("nan"), 0))
            c = summary["neg_false_accept"]["dtw"].get(n, (float("nan"), 0))
            print(f"{n:<6}{f'{100 * a[0]:.1f}%':>18}{f'{100 * b[0]:.1f}%':>18}{f'{c[0]:.1f}':>22}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"ns": args.ns, "repeats": args.repeats, "raw": {m: {str(n): v for n, v in d.items()} for m, d in raw.items()},
                   "summary": {k: {m: {str(n): list(x) for n, x in d.items()} for m, d in mm.items()} for k, mm in summary.items()}},
                  fh, ensure_ascii=False, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        for ax, k in zip(axs, keys):
            for m in args.models:
                d = summary[k][m]
                ns = sorted(d)
                mu = np.array([d[n][0] for n in ns]); sd = np.array([d[n][1] for n in ns])
                sc = 100 if k != "auroc_neg" else 1
                ax.errorbar(ns, mu * sc, yerr=sd * sc, marker="o", capsize=3, label=m)
            ax.set_xlabel("số mẫu mỗi từ (n)"); ax.set_title(names[k]); ax.grid(alpha=0.3)
        axs[0].legend()
        fig.tight_layout()
        png = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png, dpi=130)
        print(f"Đã lưu biểu đồ: {png}")
    except Exception as e:
        print(f"(bỏ qua biểu đồ: {type(e).__name__})")
    print(f"\nĐã lưu: {args.out}   (tổng {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
