"""
Do kha nang TU CHOI TU LA va ROBUSTNESS CHEO NGUOI KY cho mot hoac nhieu cau hinh scoreboard.

Bon phep do tren CUNG bank (khong thu them du lieu):
  LOSO  : query = mau goc cua phien bi giu lai, tham chieu = cac phien con lai (do chap nhan dung / nhan nham).
  Mai   : query = mau cua tinh nguyen vien (profiles/<signer>.json), tham chieu = ca bank. CHI DE DO, khong huan luyen.
  LOWO  : (leave-one-word-out) bo han mot tu khoi bank, query = cac mau goc cua tu do => day la TU LA that su;
          bat ky ket qua ACCEPTED nao la nhan nham tu la.
  NEG   : mau am trong negative_strokes.json (cu chi khong phai tu vung) tren bank day du.

Cach dung:
    py eval_open_set.py --config configs/C11_base.json --config configs/G2p00.json
    py eval_open_set.py --config configs/G2p00.json --signer Mai --bank webcam_templates.json
"""

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

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


def _query_kwargs(s: Dict[str, Any]) -> Dict[str, Any]:
    raw = s.get("traj_features_9d") if s.get("traj_features_9d") is not None else s.get("traj_features", [])
    return dict(
        query_traj_9d=np.array(raw, dtype=np.float32),
        query_shapes=s.get("hand_shapes", ["Unknown"] * 32),
        handedness=("Right" if s.get("chirality_canonical") else s.get("handedness")),
        query_has_body=s.get("body_anchor_valid", True),
        query_rep_count=s.get("rep_count"),
    )


def evaluate_config(cfg: Dict[str, Any], bank: Dict[str, Any], neg: List[Dict[str, Any]],
                    signer_vocab: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # --- LOSO
    _, folds = split_dataset(bank)
    n = acc = wrong = 0
    for f in folds:
        m = build_matcher_from_config(cfg, f.reference_bank)
        for q in f.queries:
            r = m.match(query_traj_9d=q.traj, query_shapes=q.hand_shapes,
                        handedness=("Right" if q.chirality_canonical else q.handedness),
                        query_has_body=q.has_body, query_rep_count=q.rep_count)
            n += 1
            acc += (r.status == "ACCEPTED" and r.top_word == q.true_word)
            wrong += (r.status == "ACCEPTED" and r.top_word != q.true_word)
    out["loso"] = (n, acc, wrong)

    full = build_matcher_from_config(cfg, bank)

    # --- Mai (nguoi la)
    if signer_vocab:
        n = t1 = acc = wrong = 0
        for w, d in signer_vocab.items():
            for s in d.get("samples", []):
                r = full.match(**_query_kwargs(s))
                n += 1
                t1 += (r.top_word == w)
                acc += (r.status == "ACCEPTED" and r.top_word == w)
                wrong += (r.status == "ACCEPTED" and r.top_word != w)
        out["signer"] = (n, t1, acc, wrong)

    # --- NEG
    out["neg"] = (len(neg), sum(full.match(**_query_kwargs(s)).status == "ACCEPTED" for s in neg))

    # --- LOWO
    n = fa = 0
    per_word = collections.Counter()
    for W in bank["vocabulary"]:
        sub = {"metadata": bank.get("metadata", {}), "vocabulary": {k: v for k, v in bank["vocabulary"].items() if k != W}}
        m = build_matcher_from_config(cfg, sub)
        for s in bank["vocabulary"][W]["samples"]:
            if s.get("is_original", True) is False:
                continue
            n += 1
            if m.match(**_query_kwargs(s)).status == "ACCEPTED":
                fa += 1
                per_word[W] += 1
    out["lowo"] = (n, fa, dict(per_word))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Do chap nhan dung / tu choi tu la / nguoi la cho cac cau hinh scoreboard.")
    p.add_argument("--config", action="append", required=True, help="File cau hinh scoreboard (lap duoc).")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--neg", default="negative_strokes.json")
    p.add_argument("--signer", default="Mai", help="Ma tinh nguyen vien; doc profiles/<ma>.json (bo qua neu khong co).")
    p.add_argument("--profiles-dir", default="profiles")
    args = p.parse_args()

    bank, _ = load_bank(args.bank)
    neg, _ = load_negative_strokes(args.neg)
    sp = os.path.join(args.profiles_dir, f"{args.signer}.json")
    signer_vocab = None
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            signer_vocab = json.load(f).get("vocabulary", {})
    else:
        print(f"[!] Khong thay '{sp}' -> bo qua phep do nguoi la.")

    rows = []
    for cp in args.config:
        cfg, _ = validate_and_hash_config(cp)
        r = evaluate_config(cfg, bank, neg, signer_vocab)
        rows.append((cfg.get("name", cp), cfg.get("thresholds"), r))

    print(f"\nBank: {args.bank} | nguoi la: {args.signer if signer_vocab else '(khong co)'}")
    print(f"{'cau hinh':<14}{'thresholds':<16}{'LOSO chap nhan':>16}{'LOSO sai':>9}{'nguoi la top1':>15}{'nguoi la chap nhan':>20}{'LOWO nhan nham':>16}{'NEG nhan nham':>15}")
    for name, thr, r in rows:
        n, a, w = r["loso"]
        s = r.get("signer")
        s_t1 = f"{s[1]}/{s[0]}" if s else "-"
        s_ac = f"{s[2]}/{s[0]} ({100 * s[2] / max(s[0], 1):.0f}%)" if s else "-"
        ln, lf, _ = r["lowo"]
        print(f"{name:<14}{str(thr):<16}{f'{a}/{n} ({100 * a / n:.1f}%)':>16}{w:>9}{s_t1:>15}{s_ac:>20}{f'{lf}/{ln}':>16}{f'{r['neg'][1]}/{r['neg'][0]}':>15}")
    for name, _, r in rows:
        if r["lowo"][2]:
            print(f"  [{name}] LOWO nhan nham theo tu: {r['lowo'][2]}")


if __name__ == "__main__":
    main()
