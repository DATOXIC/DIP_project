"""
PILOT: liệu một mạng sâu HUẤN LUYỆN TRÊN DỮ LIỆU GỘP NHIỀU NGƯỜI có nhận người lạ (0 mẫu enroll) tốt hơn DTW không.

Vì sao cần phép đo này: eval_hybrid.py trước đây huấn luyện GRU CHỈ trên bank của Đạt (LOSO theo PHIÊN của
CÙNG MỘT NGƯỜI) -- mạng chưa từng thấy tay/dáng ký của người khác trong lúc train, nên kết luận "GRU không hơn DTW"
KHÔNG phải bằng chứng "deep learning không giúp cho người lạ". Ở đây sửa đúng chỗ đó: với mỗi người lạ target,
GRU được train trên dữ liệu GỘP của 2 người KHÁC (không phải target) -- rồi test trên 100% mẫu của target (0 mẫu
enroll của target). So target cùng lúc với DTW-bank-Đạt (mặc định) và DTW-bank-gộp (2 người khác), CÙNG một tập
mẫu test, để so sánh công bằng.

Chỉ có 3 người (Đạt/Mai/Phi) nên đây là bằng chứng SƠ BỘ, không phải kết luận cuối; nhưng nó rẻ (vài phút, CPU,
không cần Colab/tải dữ liệu ngoài) và trả lời đúng câu hỏi "huấn luyện trên nhiều người có giúp không" bằng
chính dữ liệu đã có.

Cách dùng:  py eval_deep_pilot.py --seeds 5
Kết quả: results/baselines/deep_pilot.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np

from eval_baselines import MODELS, bank_arrays
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config
from eval_baselines import _dtw_call, _traj, _is_orig

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_profile_vocab(pid: str, profiles_dir: str) -> Dict[str, Any]:
    p = os.path.join(profiles_dir, f"{pid}.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("vocabulary", {})


def dtw_top1(cfg, ref_bank: Dict[str, Any], target_vocab: Dict[str, Any]) -> Dict[str, float]:
    m = build_matcher_from_config(cfg, ref_bank)
    n = ok = acc_ok = acc_wrong = 0
    for w, d in target_vocab.items():
        for s in d.get("samples", []):
            if not _is_orig(s):
                continue
            n += 1
            r = _dtw_call(m, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                          bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
            top = r.candidates[0].word_key if r.candidates else None
            correct = top == w
            ok += correct
            acc_ok += correct and r.status == "ACCEPTED"
            acc_wrong += (not correct) and r.status == "ACCEPTED"
    return {"n": n, "top1": ok / n, "acc_ok": acc_ok / n, "acc_wrong": acc_wrong / n}


def gru_top1(train_bank: Dict[str, Any], target_vocab: Dict[str, Any], words, seeds: int) -> Dict[str, float]:
    idx = {w: i for i, w in enumerate(words)}
    C = len(words)
    X, y = bank_arrays(train_bank, words)
    Xq, yq = [], []
    for w, d in target_vocab.items():
        if w not in idx:
            continue
        for s in d.get("samples", []):
            if _is_orig(s):
                Xq.append(_traj(s))
                yq.append(idx[w])
    Xq = np.stack(Xq)
    yq = np.asarray(yq)
    tops = []
    for sd in range(seeds):
        m = MODELS["gru"]().fit(X, y, C, 2000 + sd)
        pred, _ = m.predict(Xq)
        tops.append(float((pred == yq).mean()))
    return {"n": len(yq), "top1_mean": float(np.mean(tops)), "top1_sd": float(np.std(tops)), "per_seed": tops}


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot: GRU train trên dữ liệu gộp nhiều người, test người lạ 0-shot.")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--dtw-config", default="configs/G1p90.json")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default=os.path.join("results", "baselines", "deep_pilot.json"))
    args = p.parse_args()

    dat_bank, _ = load_bank(args.bank)
    words = sorted(dat_bank["vocabulary"].keys())
    cfg, _ = validate_and_hash_config(args.dtw_config)
    people = ["Mai", "Phi"]
    vocabs = {pid: load_profile_vocab(pid, args.profiles_dir) for pid in people}

    t0 = time.time()
    out: Dict[str, Any] = {"dtw_config": args.dtw_config, "seeds": args.seeds, "by_target": {}}

    for target in people:
        others = [q for q in people if q != target]
        print(f"\n=== target lạ: {target}  (huấn luyện/tham chiếu: Đạt + {' + '.join(others)}, KHÔNG dùng mẫu của {target}) ===")
        target_vocab = vocabs[target]

        pooled_vocab: Dict[str, Any] = {w: {"samples": list(d.get("samples", []))} for w, d in dat_bank["vocabulary"].items()}
        for o in others:
            for w, d in vocabs[o].items():
                pooled_vocab.setdefault(w, {"samples": []})["samples"] += d.get("samples", [])
        pooled_bank = {"metadata": dat_bank.get("metadata", {}), "vocabulary": pooled_vocab}

        r_dat = dtw_top1(cfg, dat_bank, target_vocab)
        print(f"  dtw (chỉ bank Đạt)      : top1 {100*r_dat['top1']:.1f}%  nhận-đúng {100*r_dat['acc_ok']:.1f}%  nhận-sai {100*r_dat['acc_wrong']:.1f}%  (n={r_dat['n']})")

        r_pool = dtw_top1(cfg, pooled_bank, target_vocab)
        print(f"  dtw (gộp Đạt+{'+'.join(others):<10}): top1 {100*r_pool['top1']:.1f}%  nhận-đúng {100*r_pool['acc_ok']:.1f}%  nhận-sai {100*r_pool['acc_wrong']:.1f}%  (n={r_pool['n']})")

        r_gru_dat = gru_top1(dat_bank, target_vocab, words, args.seeds)
        print(f"  gru (train chỉ Đạt)     : top1 {100*r_gru_dat['top1_mean']:.1f}%±{100*r_gru_dat['top1_sd']:.1f}  (n={r_gru_dat['n']}, {args.seeds} hạt)")

        r_gru_pool = gru_top1(pooled_bank, target_vocab, words, args.seeds)
        print(f"  gru (train gộp nhiều người): top1 {100*r_gru_pool['top1_mean']:.1f}%±{100*r_gru_pool['top1_sd']:.1f}  (n={r_gru_pool['n']}, {args.seeds} hạt)")

        out["by_target"][target] = {
            "dtw_dat_only": r_dat, "dtw_pooled": r_pool,
            "gru_train_dat_only": r_gru_dat, "gru_train_pooled": r_gru_pool,
        }
        print(f"  [{time.time()-t0:.0f}s]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}  (tổng {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
