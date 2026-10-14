"""
Mô phỏng "tự thích nghi khi dùng": bank khởi đầu = bank Đạt; người lạ ký lần lượt các mẫu của họ (thứ tự ngẫu nhiên).
Mỗi mẫu được nhận diện bằng bank HIỆN TẠI (quyết định trước khi học); nếu hệ thống ACCEPTED và r_ratio <= g
thì mẫu đó được thêm vào bank dưới NHÃN DỰ ĐOÁN (không dùng nhãn thật), tối đa `cap` mẫu tự thêm mỗi từ.
So với tĩnh (không thích nghi) trên cùng thứ tự. Đo: nhận đúng / nhận sai trên cả dòng và nửa sau, số mẫu thêm, số thêm nhầm nhãn,
và số nét âm nhận nhầm với bank cuối. g = 9 nghĩa là mọi mẫu ACCEPTED đều được học.
Cách dùng:  py eval_adapt.py --targets Mai Phi --gates 9 1.2 0.9 --cap 3 --repeats 5
Kết quả: results/baselines/adapt.json
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from eval_enroll import make_bank, metrics, person_samples, run_queries
from scoreboard import build_matcher_from_config, load_bank, load_negative_strokes, validate_and_hash_config
from eval_baselines import _dtw_call, _traj

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def one(matcher_of, s):
    r = _dtw_call(matcher_of, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                  bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
    return (r.candidates[0].word_key if r.candidates else None), r.status, float(r.r_ratio)


def simulate(cfg, dat, stream, gate, cap):
    added: Dict[str, List[Any]] = {}
    m = build_matcher_from_config(cfg, make_bank(dat, True, []))
    out, n_add, n_bad = [], 0, 0
    for q in stream:
        w, st, rr = one(m, q["s"])
        out.append((w, st))
        if st == "ACCEPTED" and rr <= gate and len(added.get(w, [])) < cap:
            added.setdefault(w, []).append(q["s"])
            n_add += 1
            n_bad += (w != q["w"])
            m = build_matcher_from_config(cfg, make_bank(dat, True, [added]))
    return out, m, n_add, n_bad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=["Mai", "Phi"])
    p.add_argument("--gates", nargs="+", type=float, default=[9.0, 1.2, 0.9])
    p.add_argument("--cap", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--config", default="configs/G1p90.json")
    p.add_argument("--out", default=os.path.join("results", "baselines", "adapt.json"))
    a = p.parse_args()
    dat, _ = load_bank("webcam_templates.json")
    neg, _ = load_negative_strokes("negative_strokes.json")
    cfg, _ = validate_and_hash_config(a.config)
    words = list(dat["vocabulary"])
    t0, res = time.time(), {}
    for tgt in a.targets:
        own = person_samples(tgt, words, "profiles")
        allq = [{"w": w, "s": s} for w, ss in own.items() for s in ss]
        m0 = build_matcher_from_config(cfg, make_bank(dat, True, []))
        cache = run_queries(m0, allq)
        rows: Dict[str, List[Dict[str, float]]] = {"tinh": []}
        for g in a.gates:
            rows[f"g={g:g}"] = []
        for rep in range(a.repeats):
            order = np.random.default_rng([20260928, rep, sum(map(ord, tgt))]).permutation(len(allq))
            stream = [allq[i] for i in order]
            static = [cache[i] for i in order]
            h = len(stream) // 2
            def stats(r, lo=0, hi=None):
                qs = stream[lo:hi]
                return metrics(r[lo:hi], qs, 0)
            st = {"all": stats(static), "half2": stats(static, h), "adds": 0, "bad": 0, "neg_fa": None}
            rows["tinh"].append(st)
            for g in a.gates:
                r, m, n_add, n_bad = simulate(cfg, dat, stream, g, a.cap)
                fa = sum(1 for x in run_queries(m, [{"s": s, "w": None} for s in neg]) if x[1] == "ACCEPTED")
                rows[f"g={g:g}"].append({"all": stats(r), "half2": stats(r, h), "adds": n_add, "bad": n_bad, "neg_fa": fa})
            print(f"[{time.time() - t0:5.0f}s] {tgt} lặp {rep} xong ({len(stream)} mẫu)", flush=True)
        res[tgt] = rows
        print(f"\n=== {tgt} ({len(allq)} mẫu, cap {a.cap}/từ) ===")
        print(f"{'cách':<9}{'nhận đúng':>10}{'nửa sau':>9}{'nhận sai':>10}{'thêm':>6}{'nhầm nhãn':>10}{'nét âm/41':>10}")
        for k, rs in rows.items():
            f = lambda key, sub: 100 * np.mean([r[sub][key] for r in rs])
            neg_s = "-" if rs[0]["neg_fa"] is None else f"{np.mean([r['neg_fa'] for r in rs]):.1f}"
            print(f"{k:<9}{f('acc_ok', 'all'):>9.1f}%{f('acc_ok', 'half2'):>8.1f}%{f('acc_wrong', 'all'):>9.1f}%"
                  f"{np.mean([r['adds'] for r in rs]):>6.1f}{np.mean([r['bad'] for r in rs]):>10.1f}{neg_s:>10}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
