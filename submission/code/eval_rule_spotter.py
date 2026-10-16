"""
Do RuleSpotter (spotter_gate.py) tren cac phien thu lien tuc (collect_continuous.py) co nhan tung khung.

v_norm duoc tinh DUNG nhu luong that: landmark co tay -> compute_palm_len -> bo lam muot co tay (1 Euro)
-> KineticPauseSegmenter.update (cung v_norm ma predict_sentence_webcam dua cho spotter). Mat tay thi
reset bo lam muot + segmenter va goi spotter.update(t, False, 0.0), giong vong lap that.

Cac phep do (chi tren khung co nhan SIGN/REST/WANDER/TRANSITION, bo qua IGNORE):
  1. Ti le gate MO theo tung lop      : SIGN muon cao, REST/WANDER/TRANSITION muon thap.
  2. Tung tu (mot doan SIGN lien tuc)  : bi bo sot / do tre mo / so span gate tren mot tu (phan manh).
  3. Span gate "thua"                  : span khong cham khung SIGN nao (bao nham o REST/WANDER/TRANSITION).
  4. Cap 2 tu (co du 2 lan bam m)      : gate co TACH thanh >= 2 span giua hai tu hay khong.
                                         1 span = hai tu dinh nhau => OneShotLock nuot tu thu hai.
  5. Quet tham so open_mult x close_hold_sec de thay duoc moi quan he danh doi.

Cach dung:
    py eval_rule_spotter.py                                  # moi phien trong continuous_data/
    py eval_rule_spotter.py --session continuous_data/Dat/20261003_103331
    py eval_rule_spotter.py --margin 0.15 --no-sweep
"""

import argparse
import glob
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from collect_continuous import (
    DEFAULT_MARGIN_SEC,
    P_PAIR,
    SIGN,
    REST,
    TRANSITION,
    WANDER,
    _collect_attempts,
    _events_list,
    find_session_dirs,
    load_session,
)
from dynamic_features import compute_palm_len
from pipeline_common import make_segmenter, make_wrist_smoother
from spotter_gate import OneShotLock, RuleSpotter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CLASS_NAMES = {SIGN: "SIGN", REST: "REST", WANDER: "WANDER", TRANSITION: "TRANSITION"}
OPEN_MULTS = (1.0, 1.5, 2.5)
CLOSE_HOLDS = (0.10, 0.15, 0.25, 0.40)
DEFAULT_OPEN_MULT = 1.5
DEFAULT_CLOSE_HOLD = 0.25
OPEN_HOLD_SEC = 0.10


# ===========================================================================
# 1. v_norm theo luong that
# ===========================================================================

def compute_vnorm_series(session: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Tra ve (hand_found[T] bool, v_norm[T] float, v_stop_ratio cua segmenter)."""
    z_t = np.asarray(session["t"], dtype=np.float64)
    lm = np.asarray(session["landmarks"], dtype=np.float64)          # (T, 2, 21, 3) toa do chuan hoa
    score = np.asarray(session["hand_score"], dtype=np.float64)       # (T, 2)
    meta = session.get("meta", {}) or {}
    w_f = float(meta.get("frame_width", 640) or 640)
    h_f = float(meta.get("frame_height", 480) or 480)

    segmenter = make_segmenter(None)
    smoother = make_wrist_smoother()
    T = z_t.shape[0]
    found = np.zeros(T, dtype=bool)
    v = np.zeros(T, dtype=np.float64)
    lost = True
    for i in range(T):
        vis = ~np.isnan(lm[i, :, 0, 0])
        if not vis.any():
            if not lost:
                smoother.reset()
                segmenter.reset()
            lost = True
            continue
        slot = int(np.argmax(np.where(vis, np.nan_to_num(score[i], nan=0.0) + 1e-6, -1.0)))
        pts = lm[i, slot]
        p = lambda k: SimpleNamespace(x=float(pts[k, 0]), y=float(pts[k, 1]), z=float(pts[k, 2]))   # noqa: E731
        palm = compute_palm_len(p(0), p(5), p(9), p(17), w_f, h_f)
        wx, wy = smoother.update(float(pts[0, 0]) * w_f, float(pts[0, 1]) * h_f, timestamp=float(z_t[i]))
        _, _, _, vn = segmenter.update(wx, wy, palm, timestamp=float(z_t[i]))
        found[i] = True
        v[i] = float(vn)
        lost = False
    return found, v, float(segmenter.v_stop_ratio)


# ===========================================================================
# 2. Chay spotter + do
# ===========================================================================

def run_gate(t: np.ndarray, found: np.ndarray, v: np.ndarray, v_stop: float,
             open_mult: float, close_hold: float) -> Tuple[np.ndarray, List[Tuple[float, Optional[float]]]]:
    """Chay RuleSpotter + OneShotLock qua ca phien. Tra ve (gate_open[T], danh sach span (t_open, t_close|None))."""
    sp = RuleSpotter(open_v=v_stop * open_mult, close_v=v_stop,
                     open_hold_sec=OPEN_HOLD_SEC, close_hold_sec=close_hold)
    lock = OneShotLock(max_spans=10 ** 6)
    gate = np.zeros(t.shape[0], dtype=bool)
    for i in range(t.shape[0]):
        g = sp.update(float(t[i]), bool(found[i]), float(v[i]) if found[i] else 0.0)
        lock.observe(float(t[i]), g)
        gate[i] = g
    spans = [(s[1], s[2]) for s in lock.spans]
    return gate, spans


def label_runs(labels: np.ndarray, cls: int) -> List[Tuple[int, int]]:
    """Cac doan lien tuc [i0, i1] (bao gom hai dau) cua nhan cls."""
    sel = labels == cls
    runs: List[Tuple[int, int]] = []
    i = 0
    n = sel.shape[0]
    while i < n:
        if sel[i]:
            j = i
            while j + 1 < n and sel[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def span_mask(t: np.ndarray, span: Tuple[float, Optional[float]]) -> np.ndarray:
    t_open, t_close = span
    end = t[-1] + 1.0 if t_close is None else t_close
    return (t >= t_open) & (t < end)


def measure(t: np.ndarray, labels: np.ndarray, events: Any, gate: np.ndarray,
            spans: List[Tuple[float, Optional[float]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # 1. ti le mo theo lop
    cls_open: Dict[str, Tuple[int, int]] = {}
    for c, name in CLASS_NAMES.items():
        sel = labels == c
        cls_open[name] = (int(gate[sel].sum()), int(sel.sum()))
    out["class_open"] = cls_open

    # 2. tung tu (doan SIGN)
    words = label_runs(labels, SIGN)
    detected = 0
    latencies: List[float] = []
    spans_per_word: List[int] = []
    for i0, i1 in words:
        frac = float(gate[i0:i1 + 1].mean())
        if frac >= 0.5:
            detected += 1
        # do tre mo: tu dau doan toi luc gate mo (cho phep gate da mo san -> 0)
        lo = max(0, i0 - int(0.5 / max(float(np.median(np.diff(t))), 1e-3)))
        idx = np.nonzero(gate[lo:i1 + 1])[0]
        if idx.size:
            latencies.append(max(0.0, float(t[lo + idx[0]] - t[i0])))
        n_sp = 0
        for sp in spans:
            m = span_mask(t, sp)
            if m[i0:i1 + 1].any():
                n_sp += 1
        spans_per_word.append(n_sp)
    out["words"] = len(words)
    out["words_detected"] = detected
    out["latency_median"] = float(np.median(latencies)) if latencies else float("nan")
    out["spans_per_word_mean"] = float(np.mean(spans_per_word)) if spans_per_word else float("nan")
    out["words_fragmented"] = int(sum(1 for n in spans_per_word if n > 1))

    # 3. span thua: khong cham khung SIGN nao
    spurious = {"WANDER": 0, "REST": 0, "TRANSITION": 0, "IGNORE": 0}
    n_spurious = 0
    for sp in spans:
        m = span_mask(t, sp)
        if (labels[m] == SIGN).any():
            continue
        n_spurious += 1
        lab = labels[m]
        if lab.size == 0:
            spurious["IGNORE"] += 1
            continue
        counts = {name: int((lab == c).sum()) for c, name in CLASS_NAMES.items() if c != SIGN}
        counts["IGNORE"] = int((lab == 0).sum())
        spurious[max(counts, key=counts.get)] += 1
    minutes = max((t[-1] - t[0]) / 60.0, 1e-6)
    out["spurious_spans"] = n_spurious
    out["spurious_by_class"] = spurious
    out["spurious_per_min"] = n_spurious / minutes
    out["n_spans"] = len(spans)

    # 4. cap 2 tu co du 2 lan bam m
    pairs_total = pairs_split = 0
    for a in _collect_attempts(_events_list(events), float(t[-1])).values():
        if a["kind"] != P_PAIR or a["bad"] or a["discarded"] or len(a["boundaries"]) < 2 or not a["signs"]:
            continue
        ss, se = a["signs"][0]
        n_sp = sum(1 for sp in spans if (span_mask(t, sp) & (t >= ss) & (t <= se)).any())
        pairs_total += 1
        pairs_split += int(n_sp >= 2)
    out["pairs_total"] = pairs_total
    out["pairs_split"] = pairs_split
    return out


def merge_results(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cong gop ket qua nhieu phien (cong tong dem, tinh lai cac ti le)."""
    m: Dict[str, Any] = {"class_open": {k: (0, 0) for k in CLASS_NAMES.values()}, "words": 0, "words_detected": 0,
                         "words_fragmented": 0, "spurious_spans": 0, "n_spans": 0, "pairs_total": 0, "pairs_split": 0,
                         "spurious_by_class": {"WANDER": 0, "REST": 0, "TRANSITION": 0, "IGNORE": 0},
                         "minutes": 0.0, "lat": [], "spw": []}
    for r in rs:
        for k, (a, b) in r["class_open"].items():
            x, y = m["class_open"][k]
            m["class_open"][k] = (x + a, y + b)
        for k in ("words", "words_detected", "words_fragmented", "spurious_spans", "n_spans", "pairs_total", "pairs_split"):
            m[k] += r[k]
        for k, v in r["spurious_by_class"].items():
            m["spurious_by_class"][k] += v
        m["minutes"] += r["minutes"]
        if r["words"]:
            m["lat"].append((r["latency_median"], r["words"]))
            m["spw"].append((r["spans_per_word_mean"], r["words"]))
    wl = sum(w for _, w in m["lat"])
    m["latency_median"] = (sum(x * w for x, w in m["lat"] if x == x) / wl) if wl else float("nan")
    ww = sum(w for _, w in m["spw"])
    m["spans_per_word_mean"] = (sum(x * w for x, w in m["spw"]) / ww) if ww else float("nan")
    m["spurious_per_min"] = m["spurious_spans"] / max(m["minutes"], 1e-6)
    return m


def pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:5.1f}%" if b else "   -  "


# ===========================================================================
# 3. Chay tren cac phien
# ===========================================================================

def load_all(session_dirs: List[str], margin: float) -> List[Dict[str, Any]]:
    from collect_continuous import derive_frame_labels
    out = []
    for d in session_dirs:
        s = load_session(d, margin)
        t = np.asarray(s["t"], dtype=np.float64)
        labels = np.asarray(s["labels"])
        print(f"[nap] {d}: {t.shape[0]} khung, {t[-1]:.0f}s - dang tinh v_norm ...")
        found, v, v_stop = compute_vnorm_series(s)
        out.append(dict(dir=d, t=t, labels=labels, events=s["events"], found=found, v=v, v_stop=v_stop))
    return out


def evaluate(data: List[Dict[str, Any]], open_mult: float, close_hold: float) -> Dict[str, Any]:
    rs = []
    for d in data:
        gate, spans = run_gate(d["t"], d["found"], d["v"], d["v_stop"], open_mult, close_hold)
        r = measure(d["t"], d["labels"], d["events"], gate, spans)
        r["minutes"] = float((d["t"][-1] - d["t"][0]) / 60.0)
        rs.append(r)
    return merge_results(rs)


def print_detail(m: Dict[str, Any], open_mult: float, close_hold: float, v_stop: float) -> None:
    print(f"\n=== CHI TIET: open_v = {open_mult:g} x v_stop ({v_stop * open_mult:.3f}), close_v = {v_stop:.3f}, "
          f"close_hold = {close_hold:.2f}s, open_hold = {OPEN_HOLD_SEC:.2f}s ===")
    print("1. Ti le gate MO theo lop (SIGN muon CAO; cac lop con lai muon THAP):")
    for name in ("SIGN", "REST", "WANDER", "TRANSITION"):
        a, b = m["class_open"][name]
        print(f"   {name:<11}{pct(a, b)}  ({a}/{b} khung)")
    print(f"2. Tung tu: {m['words']} tu | phat hien dung {m['words_detected']}/{m['words']} "
          f"({pct(m['words_detected'], m['words']).strip()}) | tre mo trung binh {m['latency_median']:.2f}s | "
          f"span/tu {m['spans_per_word_mean']:.2f} | tu bi phan manh (>1 span): {m['words_fragmented']}")
    sb = m["spurious_by_class"]
    print(f"3. Span gate thua (khong cham SIGN nao): {m['spurious_spans']}/{m['n_spans']} span = {m['spurious_per_min']:.1f}/phut "
          f"| theo lop: WANDER {sb['WANDER']}, REST {sb['REST']}, TRANSITION {sb['TRANSITION']}, khac {sb['IGNORE']}")
    print(f"4. Cap 2 tu: gate TACH thanh >= 2 span o {m['pairs_split']}/{m['pairs_total']} cap "
          f"({pct(m['pairs_split'], m['pairs_total']).strip()}) -> cap con lai bi dinh nhau, tu thu hai se bi OneShotLock nuot")


def main() -> None:
    p = argparse.ArgumentParser(description="Do RuleSpotter tren phien thu lien tuc co nhan.")
    p.add_argument("--session", action="append", default=None, help="Thu muc phien (lap duoc). Mac dinh: moi phien trong --root.")
    p.add_argument("--root", default="continuous_data")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN_SEC)
    p.add_argument("--no-sweep", action="store_true", help="Chi in chi tiet cau hinh mac dinh, bo bang quet tham so.")
    args = p.parse_args()

    dirs = args.session or find_session_dirs(args.root)
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "frames.npz"))]
    if not dirs:
        print(f"[!] Khong thay phien nao trong '{args.root}'.")
        sys.exit(1)
    data = load_all(dirs, args.margin)
    v_stop = data[0]["v_stop"]

    m = evaluate(data, DEFAULT_OPEN_MULT, DEFAULT_CLOSE_HOLD)
    print_detail(m, DEFAULT_OPEN_MULT, DEFAULT_CLOSE_HOLD, v_stop)
    if args.no_sweep:
        return

    print("\n=== QUET THAM SO (open_mult x close_hold) ===")
    print("SIGN% = gate mo trong SIGN (cao tot) | WAN/REST/TRN% = gate mo o lop am (thap tot) | "
          "tach = so cap 2 tu duoc tach | thua/ph = span thua moi phut")
    print(f"{'open_mult':>9}{'close_hold':>11}{'SIGN%':>8}{'WAN%':>8}{'REST%':>8}{'TRN%':>8}{'tu_dung':>9}{'tach':>8}{'thua/ph':>9}")
    for om in OPEN_MULTS:
        for ch in CLOSE_HOLDS:
            r = evaluate(data, om, ch)
            co = r["class_open"]
            print(f"{om:>9g}{ch:>11.2f}{pct(*co['SIGN']):>8}{pct(*co['WANDER']):>8}{pct(*co['REST']):>8}"
                  f"{pct(*co['TRANSITION']):>8}{pct(r['words_detected'], r['words']):>9}"
                  f"{f'{r['pairs_split']}/{r['pairs_total']}':>8}{r['spurious_per_min']:>9.1f}")
    print("\nDoc: muon SIGN% cao, WAN%/REST%/TRN% thap, 'tach' = tat ca cap, 'thua/ph' thap. "
          "Neu khong cau hinh nao dat -> RuleSpotter khong du, can Spotter DL.")


if __name__ == "__main__":
    main()
