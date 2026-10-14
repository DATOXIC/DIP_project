"""
MA TRAN NHAM LAN THAT CUA DTW TREN 13 TU (LOSO) -- de tra loi cau hoi: DTW dang
nham lan cap tu nao voi nhau bang QUY DAO thuan, truoc khi quyet dinh dau tu train
them chu cai handshape cho tu nao. Chi nen them handshape cho tu thuoc mot cap
DTW dang thuc su nham, khong phai suy doan theo "chu cai nhin giong nhau".

Cach dung:  py eval_word_confusion.py --dtw-config configs/G1p90.json
Ket qua: results/baselines/word_confusion.json
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval_baselines import _dtw_call
from scoreboard import build_matcher_from_config, load_bank, split_dataset, validate_and_hash_config


def main() -> None:
    p = argparse.ArgumentParser(description="Ma tran nham lan that cua DTW tren cac tu (LOSO).")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--dtw-config", default="configs/G1p90.json")
    p.add_argument("--out", default=os.path.join("results", "baselines", "word_confusion.json"))
    args = p.parse_args()

    bank, _ = load_bank(args.bank)
    cfg, _ = validate_and_hash_config(args.dtw_config)
    words = sorted(bank["vocabulary"].keys())
    mode, folds = split_dataset(bank)
    print(f"Che do chia: {mode} | {len(folds)} luot")

    confusion: Dict[str, Counter] = {w: Counter() for w in words}
    n_by_word = Counter()
    wrong_pairs = Counter()

    for fold in folds:
        matcher = build_matcher_from_config(cfg, fold.reference_bank)
        for q in fold.queries:
            n_by_word[q.true_word] += 1
            r = _dtw_call(matcher, q.traj, q.hand_shapes, q.handedness, q.chirality_canonical,
                          q.has_body, q.rep_count)
            top = r.candidates[0].word_key if r.candidates else None
            confusion[q.true_word][top] += 1
            if top is not None and top != q.true_word:
                pair = tuple(sorted([q.true_word, top]))
                wrong_pairs[pair] += 1

    print(f"\n{'tu that':<12}{'n':>4}  du doan (dang: so lan)")
    for w in words:
        n = n_by_word[w]
        acc = 100.0 * confusion[w][w] / max(1, n)
        preds = ", ".join(f"{k}={v}" for k, v in confusion[w].most_common() if k != w)
        print(f"{w:<12}{n:>4}  acc={acc:5.1f}%  " + (f"nham: {preds}" if preds else "(khong nham)"))

    print("\n=== CAP TU BI NHAM LAN NHIEU NHAT (2 chieu gop lai) ===")
    for pair, cnt in wrong_pairs.most_common(15):
        print(f"  {pair[0]:<12} <-> {pair[1]:<12}  {cnt} lan")

    out = {
        "mode": mode,
        "n_by_word": dict(n_by_word),
        "confusion": {w: dict(c) for w, c in confusion.items()},
        "wrong_pairs": {f"{a}|{b}": c for (a, b), c in wrong_pairs.most_common()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nDa luu: {args.out}")


if __name__ == "__main__":
    main()
