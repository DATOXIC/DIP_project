"""
Kiem tra do tin cay cua negative_strokes.json truoc khi dung lam mau am cho Binary Spotter.

Ba phep do:
  1. THONG KE   : so mau, tay trai/phai, phien thu, so cu chi khac nhau (rep_count), thoi luong, do lon chuyen dong.
  2. GAN TU VUNG: moi mau am chay qua DTW voi bank day du -> tu gan nhat, khoang cach, r_ratio, trang thai.
                  Mau co r_ratio thap nhu mau duong that = NGHI LA MAU DUONG (can xem lai hoac loai).
  3. DOI CHIEU  : phan phoi r_ratio cua mau duong that (LOSO, giu lai 1 phien) de biet "thap" la bao nhieu.

Cach dung:
    py audit_negatives.py
    py audit_negatives.py --config configs/G1p90.json --bank webcam_templates.json --top 15
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import collections
import sys
from typing import Any, Dict, List

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


def motion_energy(s: Dict[str, Any]) -> float:
    """Tong thay doi giua cac frame lien tiep cua vector dac trung (dai luong tuong doi, chi de so sanh)."""
    f = np.array(s["traj_features"], dtype=np.float32)
    return float(np.linalg.norm(np.diff(f, axis=0), axis=1).sum())


def pct(a: List[float], q: float) -> float:
    return float(np.percentile(a, q)) if a else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Kiem tra do tin cay cua mau am.")
    p.add_argument("--config", default="configs/G1p90.json")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--neg", default="negative_strokes.json")
    p.add_argument("--top", type=int, default=12, help="So mau am nghi ngo nhat de liet ke.")
    args = p.parse_args()

    bank, _ = load_bank(args.bank)
    neg, _ = load_negative_strokes(args.neg)
    cfg, _ = validate_and_hash_config(args.config)

    # ---- 1. THONG KE
    print(f"\n=== 1. THONG KE ({len(neg)} mau am) ===")
    print("tay (handedness goc):", dict(collections.Counter(s.get("handedness") for s in neg)))
    print("phien thu           :", dict(collections.Counter(s.get("session_id") for s in neg)))
    print("rep_count           :", dict(sorted(collections.Counter(s.get("rep_count") for s in neg).items())))
    dur = [float(s.get("duration_sec", 0)) for s in neg]
    en = [motion_energy(s) for s in neg]
    print(f"thoi luong (giay)   : min {min(dur):.2f} | trung vi {pct(dur, 50):.2f} | max {max(dur):.2f}")
    print(f"nang luong chuyen dong: min {min(en):.2f} | trung vi {pct(en, 50):.2f} | max {max(en):.2f}")

    # ---- 3. mau duong that (LOSO) lam moc so sanh
    _, folds = split_dataset(bank)
    pos_r: List[float] = []
    pos_d: List[float] = []
    pos_dur: List[float] = []
    pos_en: List[float] = []
    for f in folds:
        m = build_matcher_from_config(cfg, f.reference_bank)
        for q in f.queries:
            r = m.match(query_traj_9d=q.traj, query_shapes=q.hand_shapes,
                        handedness=("Right" if q.chirality_canonical else q.handedness),
                        query_has_body=q.has_body, query_rep_count=q.rep_count)
            pos_r.append(r.r_ratio)
            pos_d.append(r.best_distance)
    for w, d in bank["vocabulary"].items():
        for s in d["samples"]:
            if s.get("is_original", True) is False:
                continue
            if "duration_sec" in s:
                pos_dur.append(float(s["duration_sec"]))
            if s.get("traj_features"):
                pos_en.append(motion_energy(s))

    # ---- 2. GAN TU VUNG
    full = build_matcher_from_config(cfg, bank)
    rows = []
    for i, s in enumerate(neg):
        r = full.match(**_query_kwargs(s))
        rows.append(dict(i=i, word=r.top_word, d=r.best_distance, r=r.r_ratio, st=r.status,
                         hand=s.get("handedness"), dur=s.get("duration_sec"), rep=s.get("rep_count"),
                         sess=s.get("session_id")))
    neg_r = [x["r"] for x in rows]
    neg_d = [x["d"] for x in rows]
    st = collections.Counter(x["st"] for x in rows)

    print(f"\n=== 2. MAU AM TREN BANK DAY DU (config {cfg.get('name', args.config)}) ===")
    print("trang thai:", dict(st), "  <- ACCEPTED = DTW nhan nham mau am thanh tu vung")
    print("tu gan nhat nhieu nhat:", collections.Counter(x["word"] for x in rows).most_common(6))

    print(f"\n=== 3. DOI CHIEU VOI MAU DUONG THAT (LOSO, {len(pos_r)} mau) ===")
    print(f"{'':<14}{'p10':>8}{'p25':>8}{'trung vi':>10}{'p75':>8}{'p90':>8}")
    for name, a in (("r_ratio duong", pos_r), ("r_ratio am", neg_r), ("khoang cach duong", pos_d), ("khoang cach am", neg_d)):
        print(f"{name:<18}{pct(a,10):>8.2f}{pct(a,25):>8.2f}{pct(a,50):>10.2f}{pct(a,75):>8.2f}{pct(a,90):>8.2f}")
    if pos_dur:
        print(f"thoi luong duong: trung vi {pct(pos_dur,50):.2f}s | am: trung vi {pct(dur,50):.2f}s")
    if pos_en:
        print(f"nang luong duong: trung vi {pct(pos_en,50):.2f} | am: trung vi {pct(en,50):.2f}")

    # nguong nghi ngo: mau am nao co r_ratio <= p75 cua mau duong => khong tach noi khoi mau duong
    thr = pct(pos_r, 75)
    suspects = sorted([x for x in rows if x["r"] <= thr], key=lambda x: x["r"])
    print(f"\nNGHI LA MAU DUONG: {len(suspects)}/{len(rows)} mau am co r_ratio <= {thr:.2f} (p75 cua mau duong)")
    print(f"{'#':>3} {'tu gan nhat':<18}{'r_ratio':>8}{'khoang cach':>12}{'tt':>11}{'tay':>7}{'dai(s)':>8}{'rep':>5}  phien")
    for x in suspects[: args.top]:
        print(f"{x['i']:>3} {str(x['word']):<18}{x['r']:>8.2f}{x['d']:>12.3f}{x['st']:>11}{str(x['hand']):>7}{float(x['dur'] or 0):>8.2f}{str(x['rep']):>5}  {x['sess']}")

    # nhom theo (tay, thoi luong ~, tu gan nhat) de uoc luong so cu chi KHAC NHAU that
    groups = collections.defaultdict(list)
    for x in rows:
        groups[(x["sess"], x["word"])].append(x["i"])
    print(f"\nSo nhom (phien, tu gan nhat) = {len(groups)} -> uoc luong so cu chi khac nhau that nho hon 41")
    print("\nDoc ket qua: r_ratio am THAP sat/duoi r_ratio duong = mau do nhin giong tu that -> xem lai bang mat.")
    print("            r_ratio am CAO hon han mau duong = mau am sach, tach duoc.")


if __name__ == "__main__":
    main()
