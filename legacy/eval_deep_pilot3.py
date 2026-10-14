"""
PILOT 3 (kiểm định tách bạch): pilot2 cho thấy thêm Citizen giúp GRU tăng mạnh khi target = Đạt
(61.4% -> 86.1%). Câu hỏi: đó là nhờ TÍNH ĐA DẠNG của Citizen (nhiều signer khác) hay chỉ đơn giản
là hiệu ứng "có thêm dữ liệu" (GRU đang đói dữ liệu, thêm bao nhiêu mẫu cũng tăng)?

Cách tách: với mỗi target, so 3 kịch bản CÙNG một GRU, CÙNG một tập test:
  others      : chỉ 2 người còn lại (như pilot2)
  others_cit  : + Citizen (đúng số mẫu/từ như pilot2)
  others_dup  : + nhân bản NGẪU NHIÊN từ chính 2 người còn lại, ĐÚNG BẰNG số mẫu Citizen thêm vào
                mỗi từ (không thêm dữ liệu mới, chỉ tăng khối lượng) -- đối chứng "chỉ nhiều dữ liệu hơn"

Nếu others_cit >> others_dup: Citizen đóng góp THẬT (đa dạng signer), đáng đầu tư cắt thêm.
Nếu others_cit ~ others_dup: chỉ là hiệu ứng khối lượng dữ liệu, không phải do Citizen đặc biệt.

Cách dùng:  py eval_deep_pilot3.py --seeds 8
Kết quả: results/baselines/deep_pilot3.json
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np

from eval_baselines import MODELS, bank_arrays, _traj, _is_orig
from scoreboard import load_bank

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


def gru_top1(train_vocab: Dict[str, Any], target_vocab: Dict[str, Any], words: List[str], seeds: int) -> Dict[str, float]:
    idx = {w: i for i, w in enumerate(words)}
    C = len(words)
    X, y = bank_arrays({"vocabulary": train_vocab}, words)
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
        m = MODELS["gru"]().fit(X, y, C, 4000 + sd)
        pred, _ = m.predict(Xq)
        tops.append(float((pred == yq).mean()))
    return {"n_train": len(y), "top1_mean": float(np.mean(tops)), "top1_sd": float(np.std(tops))}


def build_dup_pool(others_vocab: Dict[str, Any], cit_vocab: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Nhân bản ngẫu nhiên từ others_vocab, đúng số mẫu Citizen thêm vào mỗi từ (không thêm dữ liệu mới)."""
    rng = random.Random(seed)
    out = {w: {"samples": list(d["samples"])} for w, d in others_vocab.items()}
    for w, d in cit_vocab.items():
        n_add = len(d.get("samples", []))
        pool = others_vocab.get(w, {}).get("samples", [])
        if n_add == 0 or not pool:
            continue
        out.setdefault(w, {"samples": []})
        out[w]["samples"] += [rng.choice(pool) for _ in range(n_add)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot 3: tách hiệu ứng 'đa dạng Citizen' khỏi hiệu ứng 'nhiều dữ liệu'.")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--citizen", default="asl_citizen_templates_24d.json")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--dup-trials", type=int, default=3, help="số lần lặp nhân-bản-ngẫu-nhiên khác nhau, lấy trung bình")
    p.add_argument("--exclude-words", nargs="*", default=[], help="loại các từ này khỏi Citizen trước khi gộp (vd BYE YOU)")
    p.add_argument("--out", default=os.path.join("results", "baselines", "deep_pilot3.json"))
    p.add_argument("--tag", default="", help="hậu tố tên biến trong log để phân biệt lần chạy")
    args = p.parse_args()

    dat_bank, _ = load_bank(args.bank)
    words = sorted(dat_bank["vocabulary"].keys())

    with open(args.citizen, "r", encoding="utf-8") as f:
        cit_vocab_raw = json.load(f)["vocabulary"]
    cit_vocab = {w: d for w, d in cit_vocab_raw.items() if w in words and w not in args.exclude_words}
    if args.exclude_words:
        print(f"Đã loại khỏi Citizen: {args.exclude_words}")

    people = {
        "Dat": {w: {"samples": list(d.get("samples", []))} for w, d in dat_bank["vocabulary"].items()},
        "Mai": load_profile_vocab("Mai", args.profiles_dir),
        "Phi": load_profile_vocab("Phi", args.profiles_dir),
    }

    t0 = time.time()
    out: Dict[str, Any] = {"seeds": args.seeds, "dup_trials": args.dup_trials, "by_target": {}}

    for target in ["Dat", "Mai", "Phi"]:
        others = [q for q in people if q != target]
        target_vocab = people[target]
        others_vocab = merge_vocabs(*[people[o] for o in others])
        others_cit_vocab = merge_vocabs(others_vocab, cit_vocab)

        print(f"\n=== target lạ: {target}  (nền: {' + '.join(others)}) ===")
        r_o = gru_top1(others_vocab, target_vocab, words, args.seeds)
        print(f"  others            (n_train={r_o['n_train']:>3}): top1 {100*r_o['top1_mean']:.1f}%±{100*r_o['top1_sd']:.1f}")

        r_oc = gru_top1(others_cit_vocab, target_vocab, words, args.seeds)
        print(f"  others + Citizen  (n_train={r_oc['n_train']:>3}): top1 {100*r_oc['top1_mean']:.1f}%±{100*r_oc['top1_sd']:.1f}")

        dup_tops = []
        n_train_dup = None
        for dt in range(args.dup_trials):
            dup_vocab = build_dup_pool(others_vocab, cit_vocab, seed=5000 + dt)
            r_dup = gru_top1(dup_vocab, target_vocab, words, args.seeds)
            dup_tops.append(r_dup["top1_mean"])
            n_train_dup = r_dup["n_train"]
        dup_mean, dup_sd = float(np.mean(dup_tops)), float(np.std(dup_tops))
        print(f"  others + nhân-bản (n_train={n_train_dup:>3}): top1 {100*dup_mean:.1f}%±{100*dup_sd:.1f}  (đối chứng: cùng khối lượng, không dữ liệu mới)")
        print(f"  [{time.time()-t0:.0f}s]")

        gain_cit = r_oc["top1_mean"] - r_o["top1_mean"]
        gain_dup = dup_mean - r_o["top1_mean"]
        print(f"  => tăng nhờ Citizen: {100*gain_cit:+.1f} điểm  |  tăng nhờ CHỈ-THÊM-KHỐI-LƯỢNG: {100*gain_dup:+.1f} điểm"
              f"  |  phần THẬT do đa dạng Citizen ~ {100*(gain_cit-gain_dup):+.1f} điểm")

        out["by_target"][target] = {
            "others": r_o, "others_citizen": r_oc,
            "others_dup": {"n_train": n_train_dup, "top1_mean": dup_mean, "top1_sd": dup_sd, "per_trial": dup_tops},
            "gain_citizen_pct": 100 * gain_cit, "gain_dup_only_pct": 100 * gain_dup,
            "diversity_contribution_pct": 100 * (gain_cit - gain_dup),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}  (tổng {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
