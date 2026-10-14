"""
Do bo phan doan (KineticPauseSegmenter) khi FPS that lech 30 va khi camera lap khung.

Khong can camera, khong can matcher: dung cac stroke tho trong raw_strokes/ (chi nhung stroke con nam trong bank),
tao bien the "cung chuyen dong vat ly, khac fps" (tai dung ham cua fps_invariance_test.py), dua qua 3 che do:
  legacy  : make_segmenter(CONFIG)                         -> ngưỡng theo so khung (cach cu, khong fast_chain)
  prod    : make_segmenter(CONFIG, fast_chain)             -> production hien tai (pause/min_stroke/ready theo GIAY)
  prod+tn : prod + time_norm.enabled (van toc + Swipe theo THOI GIAN)
Bao cao: ti le "dung 1 stroke" / "cat doi" / "nho" va so lan Swipe-Left BAO NHAM (stroke tu vung khong duoc phep
kich hoat Swipe), roi mot bai thu Swipe-Left GIA LAP (tong hop) o nhieu fps de dam bao khong mat Swipe that.

Cach dung:
    py eval_segmenter_fps.py
    py eval_segmenter_fps.py --limit 60
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import copy
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

import fps_invariance_test as fit
from augment_templates import load_raw_strokes, match_bank_to_raw
from pipeline_common import make_segmenter
from predict_sentence_webcam import CONFIG

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

VARIANTS = ["a_goc", "b_60fps", "c_15fps", "d_trung", "d_dedup", "e_45fps", "f_bo_trung"]
MODES = ["legacy", "prod", "prod+tn", "prod+tn_hi"]   # tn_hi = time_norm chi bu van toc khi fps CAO hon 30 (v_scale_min=1)


def build_segmenter(mode: str):
    cfg = copy.deepcopy(CONFIG)
    if mode == "legacy":
        return make_segmenter(cfg)
    tn = dict(cfg.get("time_norm", {}), enabled=mode.startswith("prod+tn"))
    if mode == "prod+tn_hi":
        tn["v_scale_min"] = 1.0
    cfg["time_norm"] = tn
    return make_segmenter(cfg, cfg["fast_chain"])


def make_frames(raw: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    fr = raw["frames"]
    if name == "d_dedup":                       # camera lap khung roi bo khung trung (co dedupe_frames)
        return fit.variant_dedup(fit.variant_duplicate(fr))
    return fit.make_variant(raw, name)["frames"]


def run(mode: str, stream: List[Dict[str, Any]]) -> Tuple[int, int]:
    seg = build_segmenter(mode)
    n_stroke = n_swipe = 0
    for i, f in enumerate(stream):
        wx, wy = f["wrist_smoothed"]
        _, done, is_swipe, _ = seg.update(wrist_x=float(wx), wrist_y=float(wy), palm_len=float(f["palm_len"]),
                                          hand_shape=str(i), timestamp=float(f["timestamp"]))
        n_stroke += done is not None
        n_swipe += bool(is_swipe)
    return n_stroke, n_swipe


def synthetic_swipe(fps: float, speed_palm_s: float = 9.0, dup: bool = False) -> List[Dict[str, Any]]:
    """Cu quet trai tong hop: dung yen 0.6 s -> quet 0.30 s -> dung yen 1.0 s. palm = 100 px. dup=True: cu 2 lan doc chi 1 khung moi."""
    dt = 1.0 / fps
    n = int(round(1.9 * fps))
    x, out, last = 500.0, [], None
    for i in range(n):
        t = i * dt
        if 0.6 <= t < 0.9:
            x -= speed_palm_s * 100.0 * dt
        cur = x
        if dup and i % 2 == 1 and last is not None:
            cur = last
        last = cur
        out.append({"wrist_smoothed": [cur, 300.0], "palm_len": 100.0, "timestamp": 1000.0 + t})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Do bo phan doan theo FPS that va khung lap.")
    ap.add_argument("--bank", default="webcam_templates.json")
    ap.add_argument("--raw-dir", default="raw_strokes")
    ap.add_argument("--limit", type=int, default=0, help="Chi chay N stroke dau (0 = tat ca stroke con trong bank).")
    args = ap.parse_args()

    import json
    with open(args.bank, "r", encoding="utf-8") as f:
        bank = json.load(f)
    raws = load_raw_strokes(args.raw_dir)
    matched = match_bank_to_raw(bank["vocabulary"], raws)
    strokes = [m[1] for m in matched.values()]
    if args.limit:
        strokes = strokes[:args.limit]
    print(f"[Segmenter] {len(strokes)} stroke (con trong bank) x {len(VARIANTS)} bien the x {len(MODES)} che do")

    res = {(m, v): [0, 0, 0, 0] for m in MODES for v in VARIANTS}     # dung1, cat doi, nho, swipe
    for raw in strokes:
        for v in VARIANTS:
            frames = make_frames(raw, v)
            if len(frames) < 2:
                continue
            stream = fit.build_stream(frames)
            for m in MODES:
                n_s, n_sw = run(m, stream)
                r = res[(m, v)]
                r[0] += (n_s == 1)
                r[1] += (n_s > 1)
                r[2] += (n_s == 0)
                r[3] += n_sw
    n = len(strokes)
    print(f"\n{'bien the':<12}" + "".join(f"| {m:^32}" for m in MODES))
    print(f"{'':<12}" + "".join("| dung1  cat-doi  nho  swipe-nham  " for _ in MODES))
    for v in VARIANTS:
        line = f"{v:<12}"
        for m in MODES:
            d1, cd, nh, sw = res[(m, v)]
            line += f"| {100 * d1 / n:5.1f}% {100 * cd / n:6.1f}% {100 * nh / n:5.1f}% {sw:6d}       "
        print(line)

    print("\nSwipe-Left GIA LAP (toc do 9 palm/s, 0.30 s): so lan phat hien (mong doi = 1)")
    print(f"{'fps':<8}{'kieu':<8}" + "".join(f"{m:>10}" for m in MODES))
    for fps in (8, 10, 14, 20, 30, 45, 60):
        for dup in (False, True):
            if dup and fps < 20:
                continue
            row = f"{fps:<8}{'lap' if dup else 'that':<8}"
            for m in MODES:
                seg = build_segmenter(m)
                cnt = 0
                for i, f in enumerate(synthetic_swipe(fps, dup=dup)):
                    _, _, sw, _ = seg.update(wrist_x=f["wrist_smoothed"][0], wrist_y=300.0, palm_len=100.0,
                                             hand_shape="x", timestamp=f["timestamp"])
                    cnt += bool(sw)
                row += f"{cnt:>10}"
            print(row)


if __name__ == "__main__":
    main()
