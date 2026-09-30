# -*- coding: utf-8 -*-
"""
Dieu tra vi sao HUNGRY sai nhieu nhat o nguoi la (17/27 loi = 63%, xem
scratch/stranger_handshape_rescue_test.py). HUNGRY LOSO tren chinh Dat la 100%
(eval_word_confusion.py) -> day la loi TONG QUAT HOA CHEO NGUOI KY, khong phai
loi quy dao mo ho noi bo cua Dat.

In chi tiet tung truy van HUNGRY cua 4 nguoi la: tu doan sai la gi, khoang cach
toi HUNGRY that vs toi tu doan sai, va cac thanh phan dong gop (vi position/
angle/normal/body) neu lay duoc, de biet loi nam o thanh phan nao.

Chay: python scratch/investigate_hungry.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from eval_baselines import _dtw_call, _is_orig, _traj
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PEOPLE = ["Dang", "Mai", "Phi", "Vuong"]
WORD = "HUNGRY"


def main() -> None:
    dat_bank, _ = load_bank("webcam_templates.json")
    cfg, _ = validate_and_hash_config("configs/G1p90.json")
    matcher = build_matcher_from_config(cfg, dat_bank)

    n_dat = len(dat_bank["vocabulary"][WORD]["samples"])
    print(f"So mau HUNGRY cua Dat trong bank: {n_dat}")

    confuse_with = Counter()
    total = 0
    wrong = 0

    for person in PEOPLE:
        d = json.load(open(f"profiles/{person}.json", encoding="utf-8"))
        samples = [s for s in d.get("vocabulary", {}).get(WORD, {}).get("samples", []) if _is_orig(s)]
        if not samples:
            print(f"\n[{person}] khong co mau HUNGRY")
            continue
        print(f"\n=== {person}: {len(samples)} mau HUNGRY ===")
        for i, s in enumerate(samples):
            traj = _traj(s)
            r = _dtw_call(matcher, traj, s.get("hand_shapes", ["Unknown"] * 32),
                          s.get("handedness"), bool(s.get("chirality_canonical")),
                          s.get("body_anchor_valid", True), s.get("rep_count"))
            total += 1
            cands = r.candidates[:3] if r.candidates else []
            top = cands[0].word_key if cands else None
            is_ok = (top == WORD)
            if not is_ok:
                wrong += 1
                confuse_with[top] += 1
            cand_str = ", ".join(f"{c.word_key}={c.distance:.3f}" for c in cands)
            body_ok = s.get("body_anchor_valid", True)
            print(f"  mau {i}: dtw_top={top!s:<10} status={r.status:<10} "
                  f"body_anchor_valid={body_ok}  top3=[{cand_str}]"
                  f"{'  <-- SAI' if not is_ok else ''}")

    print(f"\nTong: {wrong}/{total} sai ({100*wrong/max(1,total):.1f}%)")
    print("Nham voi:", dict(confuse_with))


if __name__ == "__main__":
    main()
