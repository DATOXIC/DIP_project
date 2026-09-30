"""
PILOT 4: lặp lại kiểm định "đa dạng người giúp gì" bằng 5 NGƯỜI THẬT (Đạt/Mai/Phi/Đăng/Vương) thay vì
Citizen -- không còn dính lệch miền (tất cả cùng pipeline thu). Với mỗi người bị giữ lại làm "người lạ",
đo top-1 (DTW và GRU) khi tập huấn luyện/tham chiếu gồm 1, 2, 3, hoặc CẢ 4 người còn lại -- vẽ đường cong
"số người tham chiếu -> độ chính xác trên người lạ", trả lời trực tiếp câu hỏi "cần bao nhiêu người".

Cách dùng:  py eval_deep_pilot4.py --seeds 3 --draws 2
Kết quả: results/baselines/deep_pilot4.json
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np

from eval_baselines import MODELS, bank_arrays, _dtw_call, _traj, _is_orig
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PEOPLE = ["Dat", "Mai", "Phi", "Dang", "Vuong"]


def load_profile_vocab(pid: str, profiles_dir: str) -> Dict[str, Any]:
    p = os.path.join(profiles_dir, f"{pid}.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("vocabulary", {})


def merge_vocabs(*vocabs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for v in vocabs:
        for w, d in v.items():
            out.setdefault(w, {"samples": []})["samples"] += [s for s in d.get("samples", []) if _is_orig(s)]
    return out


def dtw_top1(cfg, pool_vocab: Dict[str, Any], meta, target_vocab: Dict[str, Any]) -> Dict[str, float]:
    m = build_matcher_from_config(cfg, {"metadata": meta, "vocabulary": pool_vocab})
    n = ok = acc_ok = 0
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
    return {"n": n, "top1": ok / max(1, n), "acc_ok": acc_ok / max(1, n)}


def gru_top1(pool_vocab: Dict[str, Any], target_vocab: Dict[str, Any], words: List[str], seeds: int) -> Dict[str, float]:
    idx = {w: i for i, w in enumerate(words)}
    C = len(words)
    X, y = bank_arrays({"vocabulary": pool_vocab}, words)
    Xq, yq = [], []
    for w, d in target_vocab.items():
        if w not in idx:
            continue
        for s in d.get("samples", []):
            if _is_orig(s):
                Xq.append(_traj(s))
                yq.append(idx[w])
    Xq, yq = np.stack(Xq), np.asarray(yq)
    tops = []
    for sd in range(seeds):
        m = MODELS["gru"]().fit(X, y, C, 6000 + sd)
        pred, _ = m.predict(Xq)
        tops.append(float((pred == yq).mean()))
    return {"n_train": len(y), "top1_mean": float(np.mean(tops)), "top1_sd": float(np.std(tops))}


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot 4: đường cong độ chính xác theo số người tham chiếu (5 người thật).")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--dtw-config", default="configs/G1p90.json")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--draws", type=int, default=2, help="số tổ hợp ngẫu nhiên lấy mẫu cho mỗi kích cỡ k (trừ k lớn nhất)")
    p.add_argument("--out", default=os.path.join("results", "baselines", "deep_pilot4.json"))
    args = p.parse_args()

    dat_bank, _ = load_bank(args.bank)
    words = sorted(dat_bank["vocabulary"].keys())
    cfg, _ = validate_and_hash_config(args.dtw_config)
    meta = dat_bank.get("metadata", {})

    vocabs = {"Dat": {w: {"samples": list(d.get("samples", []))} for w, d in dat_bank["vocabulary"].items()}}
    for pid in ["Mai", "Phi", "Dang", "Vuong"]:
        vocabs[pid] = load_profile_vocab(pid, args.profiles_dir)
    for pid, v in vocabs.items():
        n = sum(len(d.get("samples", [])) for d in v.values())
        print(f"{pid}: {len(v)} tu, {n} mau")

    rng = random.Random(42)
    t0 = time.time()
    out: Dict[str, Any] = {"seeds": args.seeds, "draws": args.draws, "by_target": {}}

    for target in PEOPLE:
        others = [p for p in PEOPLE if p != target]
        target_vocab = vocabs[target]
        print(f"\n=== target lạ: {target}  (ứng viên tham chiếu: {others}) ===")
        by_k: Dict[int, List[Dict[str, Any]]] = {}
        for k in range(1, len(others) + 1):
            all_subsets = list(itertools.combinations(others, k))
            chosen = all_subsets if len(all_subsets) <= args.draws else rng.sample(all_subsets, args.draws)
            rows = []
            for subset in chosen:
                pool_vocab = merge_vocabs(*[vocabs[p] for p in subset])
                r_dtw = dtw_top1(cfg, pool_vocab, meta, target_vocab)
                r_gru = gru_top1(pool_vocab, target_vocab, words, args.seeds)
                rows.append({"subset": list(subset), "dtw_top1": r_dtw["top1"], "dtw_acc_ok": r_dtw["acc_ok"],
                             "gru_top1_mean": r_gru["top1_mean"], "gru_top1_sd": r_gru["top1_sd"], "n_train": r_gru["n_train"]})
            dtw_m = float(np.mean([x["dtw_top1"] for x in rows]))
            gru_m = float(np.mean([x["gru_top1_mean"] for x in rows]))
            n_m = float(np.mean([x["n_train"] for x in rows]))
            print(f"  k={k}  (n_train~{n_m:.0f}, {len(chosen)} tổ hợp): dtw top1 {100*dtw_m:.1f}%   gru top1 {100*gru_m:.1f}%")
            by_k[k] = rows
        out["by_target"][target] = by_k
        print(f"  [{time.time()-t0:.0f}s]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}  (tổng {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
