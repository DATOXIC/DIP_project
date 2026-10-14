"""
Mô phỏng một PHIÊN với chế độ hỏi xác nhận (session_enroll.py), người dùng trả lời ĐÚNG theo nhãn thật.

Bank khởi đầu = bank Đạt; người lạ ký lần lượt các mẫu của họ (thứ tự ngẫu nhiên). Mỗi mẫu được nhận diện bằng bank HIỆN TẠI
(quyết định trước khi hỏi). Nếu kết quả không phải ACCEPTED (và, tùy `hoi`, gồm cả REJECTED) và có ứng viên thì hệ thống HỎI:
  - đáp án thật nằm trong top-2 và từ đó chưa đủ `cap` mẫu -> người dùng xác nhận, mẫu được thêm;
  - ngược lại -> người dùng bấm [n] (vẫn tính là một câu hỏi).
Đo: nhận đúng / nhận sai tính trên quyết định TRƯỚC khi hỏi, chia 4 phần theo tiến trình phiên, số câu hỏi, số mẫu thêm,
và số nét âm nhận nhầm với bank cuối phiên.
Cách dùng:  py eval_confirm.py --targets Mai Phi --repeats 3
Kết quả: results/baselines/confirm.json
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from eval_baselines import _dtw_call, _traj
from eval_enroll import make_bank, person_samples
from scoreboard import build_matcher_from_config, load_bank, load_negative_strokes, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def call(m, s):
    r = _dtw_call(m, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                  bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
    return r.status, [c.word_key for c in r.candidates[:2]]


def simulate(cfg, dat, stream, cap, hoi):
    added: Dict[str, List[Any]] = {}
    m = build_matcher_from_config(cfg, make_bank(dat, True, []))
    ok, wrong, asked, adds = [], [], 0, 0
    for q in stream:
        st, cands = call(m, q["s"])
        top = cands[0] if cands else None
        ok.append(st == "ACCEPTED" and top == q["w"])
        wrong.append(st == "ACCEPTED" and top != q["w"])
        if st != "ACCEPTED" and cands and (hoi == "unc+rej" or st == "UNCERTAIN"):
            asked += 1
            if q["w"] in cands and len(added.get(q["w"], [])) < cap:
                added.setdefault(q["w"], []).append(q["s"])
                adds += 1
                m = build_matcher_from_config(cfg, make_bank(dat, True, [added]))
    return np.array(ok), np.array(wrong), asked, adds, m


def quarters(x):
    return [float(np.mean(c)) for c in np.array_split(x, 4)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=["Mai", "Phi"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--config", default="configs/G1p90.json")
    p.add_argument("--out", default=os.path.join("results", "baselines", "confirm.json"))
    a = p.parse_args()
    dat, _ = load_bank("webcam_templates.json")
    neg, _ = load_negative_strokes("negative_strokes.json")
    cfg, _ = validate_and_hash_config(a.config)
    words = list(dat["vocabulary"])
    variants = [("tinh", 0, "unc+rej"), ("cap3 unc+rej", 3, "unc+rej"), ("cap2 unc+rej", 2, "unc+rej"),
                ("cap3 chỉ unc", 3, "unc")]
    t0, res = time.time(), {}
    for tgt in a.targets:
        own = person_samples(tgt, words, "profiles")
        allq = [{"w": w, "s": s} for w, ss in own.items() for s in ss]
        rows: Dict[str, List[Dict[str, Any]]] = {v[0]: [] for v in variants}
        for rep in range(a.repeats):
            order = np.random.default_rng([20260928, rep, sum(map(ord, tgt))]).permutation(len(allq))
            stream = [allq[i] for i in order]
            for name, cap, hoi in variants:
                ok, wrong, asked, adds, m = simulate(cfg, dat, stream, cap, hoi)
                fa = sum(1 for s in neg if call(m, s)[0] == "ACCEPTED")
                rows[name].append({"ok": ok.mean(), "wrong": wrong.mean(), "q_ok": quarters(ok), "asked": asked,
                                   "adds": adds, "neg_fa": fa})
            print(f"[{time.time() - t0:5.0f}s] {tgt} lặp {rep} xong ({len(stream)} mẫu)", flush=True)
        res[tgt] = rows
        print(f"\n=== {tgt} ({len(allq)} mẫu/phiên) — nhận đúng theo 4 phần của phiên ===")
        print(f"{'cách':<15}{'cả phiên':>9}{'P1':>7}{'P2':>7}{'P3':>7}{'P4':>7}{'nhận sai':>10}{'hỏi':>6}{'thêm':>6}{'nét âm':>8}")
        for k, rs in rows.items():
            q = np.mean([r["q_ok"] for r in rs], axis=0) * 100
            print(f"{k:<15}{100 * np.mean([r['ok'] for r in rs]):>8.1f}%" + "".join(f"{x:>6.0f}%" for x in q)
                  + f"{100 * np.mean([r['wrong'] for r in rs]):>9.1f}%{np.mean([r['asked'] for r in rs]):>6.1f}"
                  + f"{np.mean([r['adds'] for r in rs]):>6.1f}{np.mean([r['neg_fa'] for r in rs]):>8.1f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"\nĐã lưu: {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
