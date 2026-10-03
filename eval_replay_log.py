"""
Cham diem log phat lai (predict_sentence_webcam.py --video <session.mp4> --video-clock --log-session <csv>)
bang nhan that cua phien thu lien tuc (events.json + frames.npz).

Moi dong ACCEPTED trong log la mot TU DA CHOT (se vao cau). Phan loai theo thoi diem stroke:
  DUNG      : stroke giao voi mot tu that VA trung tu do, la lan chot DAU TIEN cho tu that nay.
  TRUNG     : them mot lan chot nua trong cung mot tu that (cung tu hoac khac tu)  <- loi "doc 2-4 lan".
  SAI TU    : giao voi tu that nhung doc ra tu khac (lan dau cho tu that nay).
  BAO NHAM  : khong giao tu that nao; phan theo nhan tai giua stroke: WANDER / REST / TRANSITION / khac.
  BO SOT    : tu that khong co lan chot DUNG nao.
Tu that = SIGN don, hoac moi tu trong cap (cat boi hai lan bam m). Chi tinh attempt khong bi 'x' va khong bi bo.

Cach dung:
    py eval_replay_log.py --events scratch/replay/events_103331.json --frames scratch/replay/frames_103331.npz \
        --run-txt scratch/replay/run_vc.txt --log base=scratch/replay/vc_base.csv --log dl=scratch/replay/vc_dl.csv
"""

import argparse
import collections
import csv
import json
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from collect_continuous import (
    P_PAIR,
    P_SIGN,
    REST,
    SIGN,
    TRANSITION,
    WANDER,
    _collect_attempts,
    _events_list,
    derive_frame_labels,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ground_truth_words(events: Any, t_end: float) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    for a in _collect_attempts(_events_list(events), t_end).values():
        if a["bad"] or a["discarded"] or not a["signs"]:
            continue
        ss, se = a["signs"][0]
        words = a.get("words")
        if a["kind"] == P_SIGN and words:
            out.append((words[0], ss, se))
        elif a["kind"] == P_PAIR and words and len(words) == 2 and len(a["boundaries"]) >= 2:
            b0, b1 = sorted(a["boundaries"])[:2]
            out.append((words[0], ss, b0))
            out.append((words[1], b1, se))
    return out


def attach_words(events: Any) -> Dict[int, List[str]]:
    return {e["aid"]: list(e.get("words") or []) for e in _events_list(events) if e.get("type") == "prompt_begin"}


def parse_t0(run_txt: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mode = None
    for line in open(run_txt, encoding="utf-8"):
        m = re.match(r"#+\s*(\S+)", line)
        if m:
            mode = m.group(1)
        m = re.search(r"VIDEO_CLOCK_T0=([0-9.]+)", line)
        if m and mode:
            out[mode] = float(m.group(1))
    return out


def classify(rows: List[Dict[str, Any]], gt: List[Tuple[str, float, float]], t: np.ndarray, labels: np.ndarray,
             t0: float, fps: float) -> Dict[str, Any]:
    t_first = float(t[0])
    used: Dict[int, int] = {}
    res = collections.Counter()
    false_by = collections.Counter()
    correct_idx = set()
    for r in rows:
        if r["event"] != "ACCEPTED":
            continue
        ts = float(r["t_start"]) - t0                     # giay tren truc video
        te = ts + float(r["duration_sec"])
        ts_s, te_s = t_first + ts - 1.0 / fps, t_first + te - 1.0 / fps    # sang truc phien
        hit = [i for i, (_, a, b) in enumerate(gt) if min(b, te_s) - max(a, ts_s) > 0.0]
        if hit:
            i = max(hit, key=lambda k: min(gt[k][2], te_s) - max(gt[k][1], ts_s))
            if used.get(i, 0) >= 1:
                res["TRUNG"] += 1
            elif gt[i][0].upper() == r["predicted_word"].upper():
                res["DUNG"] += 1
                correct_idx.add(i)
            else:
                res["SAI TU"] += 1
            used[i] = used.get(i, 0) + 1
        else:
            res["BAO NHAM"] += 1
            mid = 0.5 * (ts_s + te_s)
            j = int(np.clip(np.searchsorted(t, mid), 0, len(t) - 1))
            lab = int(labels[j])
            false_by[{WANDER: "WANDER", REST: "REST", TRANSITION: "TRANSITION", 0: "ngoai nhan"}.get(lab, "SIGN?")] += 1
    res["BO SOT"] = len(gt) - len(correct_idx)
    res["TONG TU THAT"] = len(gt)
    return {"res": res, "false_by": false_by}


def main() -> None:
    p = argparse.ArgumentParser(description="Cham diem log phat lai bang nhan that.")
    p.add_argument("--events", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--run-txt", required=True)
    p.add_argument("--log", action="append", required=True, help="ten=duong_dan_csv (lap duoc)")
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    ev = json.load(open(args.events, encoding="utf-8"))
    with np.load(args.frames, allow_pickle=False) as z:
        t = z["t"].astype(np.float64)
    t_end = float(t[-1])
    labels = derive_frame_labels(t, ev)
    words_by_aid = attach_words(ev)
    atts = _collect_attempts(_events_list(ev), t_end)
    for aid, a in atts.items():
        a["words"] = words_by_aid.get(aid, [])
    gt: List[Tuple[str, float, float]] = []
    for a in atts.values():
        if a["bad"] or a["discarded"] or not a["signs"]:
            continue
        ss, se = a["signs"][0]
        if a["kind"] == P_SIGN and a["words"]:
            gt.append((a["words"][0], ss, se))
        elif a["kind"] == P_PAIR and len(a["words"]) == 2 and len(a["boundaries"]) >= 2:
            b0, b1 = sorted(a["boundaries"])[:2]
            gt.append((a["words"][0], ss, b0))
            gt.append((a["words"][1], b1, se))
    t0s = parse_t0(args.run_txt)
    neg_minutes = float(((labels == WANDER) | (labels == REST) | (labels == TRANSITION)).sum()) / args.fps / 60.0
    wander_min = float((labels == WANDER).sum()) / args.fps / 60.0

    print(f"Tu that: {len(gt)} (SIGN don + moi tu cua cap co du 2 lan bam m). "
          f"Thoi gian khong ky co nhan: {neg_minutes:.1f} phut (WANDER {wander_min:.1f} phut).")
    print(f"\n{'che do':<8}{'chot':>6}{'DUNG':>6}{'TRUNG':>7}{'SAI TU':>8}{'BAO NHAM':>10}{'BO SOT':>8}   bao nham theo lop / phut khong ky")
    for spec in args.log:
        name, path = spec.split("=", 1)
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if name not in t0s:
            print(f"[!] Khong co VIDEO_CLOCK_T0 cho '{name}' trong {args.run_txt}")
            continue
        r = classify(rows, gt, t, labels, t0s[name], args.fps)
        c, fb = r["res"], r["false_by"]
        total = c["DUNG"] + c["TRUNG"] + c["SAI TU"] + c["BAO NHAM"]
        fb_txt = ", ".join(f"{k} {v}" for k, v in sorted(fb.items())) or "-"
        print(f"{name:<8}{total:>6}{c['DUNG']:>6}{c['TRUNG']:>7}{c['SAI TU']:>8}{c['BAO NHAM']:>10}{c['BO SOT']:>8}   "
              f"{fb_txt}  ({c['BAO NHAM'] / max(neg_minutes, 1e-6):.1f}/phut)")
    print("\nDUNG cao + TRUNG/BAO NHAM/BO SOT thap la tot. 'chot' = so tu da vao cau (dong ACCEPTED).")


if __name__ == "__main__":
    main()
