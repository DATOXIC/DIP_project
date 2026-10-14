"""
PILOT 2: mở rộng eval_deep_pilot.py — thêm ASL Citizen (12 từ trùng, ~86 mẫu, nhiều signer ẩn danh) vào
tập huấn luyện GỘP, để xem có giúp GRU tổng quát hoá sang người lạ tốt hơn không, so với chỉ gộp
Đạt + 1 tình nguyện viên. Test LẦN LƯỢT cả 3 người (Đạt/Mai/Phi) làm "người lạ" giữ lại.

Với mỗi target trong {Đạt, Mai, Phi}: train/tham chiếu = 2 người CÒN LẠI + Citizen (12 từ chung),
KHÔNG dùng bất kỳ mẫu nào của target. So 4 kịch bản trên CÙNG tập test:
  dtw_others      : DTW, bank = 2 người còn lại (không Citizen)
  dtw_others_cit  : DTW, bank = 2 người còn lại + Citizen
  gru_others      : GRU train trên 2 người còn lại
  gru_others_cit  : GRU train trên 2 người còn lại + Citizen

Cách dùng:  py eval_deep_pilot2.py --seeds 5
Kết quả: results/baselines/deep_pilot2.json
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import json
import os
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


def gru_top1(train_bank: Dict[str, Any], target_vocab: Dict[str, Any], words: List[str], seeds: int) -> Dict[str, float]:
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
        m = MODELS["gru"]().fit(X, y, C, 3000 + sd)
        pred, _ = m.predict(Xq)
        tops.append(float((pred == yq).mean()))
    return {"n": len(yq), "top1_mean": float(np.mean(tops)), "top1_sd": float(np.std(tops))}


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot 2: GRU/DTW train gộp + Citizen, test từng người lạ.")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--citizen", default="asl_citizen_templates_24d.json")
    p.add_argument("--dtw-config", default="configs/G1p90.json")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default=os.path.join("results", "baselines", "deep_pilot2.json"))
    args = p.parse_args()

    dat_bank, _ = load_bank(args.bank)
    words = sorted(dat_bank["vocabulary"].keys())
    cfg, _ = validate_and_hash_config(args.dtw_config)

    with open(args.citizen, "r", encoding="utf-8") as f:
        cit_vocab_raw = json.load(f)["vocabulary"]
    cit_vocab = {w: d for w, d in cit_vocab_raw.items() if w in words}
    n_cit = sum(len(d["samples"]) for d in cit_vocab.values())
    print(f"Citizen dùng được: {len(cit_vocab)} từ, {n_cit} mẫu (trong 13 từ của bank).")

    people = {
        "Dat": {w: {"samples": list(d.get("samples", []))} for w, d in dat_bank["vocabulary"].items()},
        "Mai": load_profile_vocab("Mai", args.profiles_dir),
        "Phi": load_profile_vocab("Phi", args.profiles_dir),
    }

    t0 = time.time()
    out: Dict[str, Any] = {"dtw_config": args.dtw_config, "seeds": args.seeds, "n_citizen": n_cit, "by_target": {}}

    for target in ["Dat", "Mai", "Phi"]:
        others = [q for q in people if q != target]
        target_vocab = people[target]
        print(f"\n=== target lạ: {target}  (train: {' + '.join(others)} {'+ Citizen' if True else ''}) ===")

        pool_others = merge_vocabs(*[people[o] for o in others])
        pool_others_cit = merge_vocabs(*[people[o] for o in others], cit_vocab)
        bank_others = {"metadata": dat_bank.get("metadata", {}), "vocabulary": pool_others}
        bank_others_cit = {"metadata": dat_bank.get("metadata", {}), "vocabulary": pool_others_cit}

        r_dtw_o = dtw_top1(cfg, bank_others, target_vocab)
        r_dtw_oc = dtw_top1(cfg, bank_others_cit, target_vocab)
        r_gru_o = gru_top1(bank_others, target_vocab, words, args.seeds)
        r_gru_oc = gru_top1(bank_others_cit, target_vocab, words, args.seeds)

        print(f"  dtw  (2 người)        : top1 {100*r_dtw_o['top1']:.1f}%  nhận-đúng {100*r_dtw_o['acc_ok']:.1f}%  nhận-sai {100*r_dtw_o['acc_wrong']:.1f}%")
        print(f"  dtw  (2 người+Citizen): top1 {100*r_dtw_oc['top1']:.1f}%  nhận-đúng {100*r_dtw_oc['acc_ok']:.1f}%  nhận-sai {100*r_dtw_oc['acc_wrong']:.1f}%")
        print(f"  gru  (2 người)        : top1 {100*r_gru_o['top1_mean']:.1f}%±{100*r_gru_o['top1_sd']:.1f}")
        print(f"  gru  (2 người+Citizen): top1 {100*r_gru_oc['top1_mean']:.1f}%±{100*r_gru_oc['top1_sd']:.1f}")
        print(f"  (n test = {r_dtw_o['n']})  [{time.time()-t0:.0f}s]")

        out["by_target"][target] = {
            "n_test": r_dtw_o["n"],
            "dtw_others": r_dtw_o, "dtw_others_citizen": r_dtw_oc,
            "gru_others": r_gru_o, "gru_others_citizen": r_gru_oc,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}  (tổng {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
