# -*- coding: utf-8 -*-
"""
Kiem tra THUC NGHIEM (khong can camera, dung du lieu raw_strokes/ da co san) xem
tu nao trong 13 tu dang co trong bank san xuat (webcam_templates.json) co CHU KY
DANG TAY (flexion_angles, 5D, da tinh san luc thu mau) ON DINH-TRONG-TU va
PHAN BIET-DUOC-VOI-TU-KHAC.

So sanh 3 CACH lay chu ky dang tay tu 1 sample (list frame theo thoi gian):
  - full     : trung vi flexion_angles qua TOAN BO khung cua sample (baseline ban dau).
  - boundary : chi lay khung o 2 DAU MUT (dau + cuoi, moi ben BOUNDARY_FRAC%) - gia
               thuyet nguoi dung: do protocol thu mau la "settle-then-sign" (dung yen
               truoc/sau chuyen dong), 2 dau mut la luc dang tay ON DINH nhat, giua
               chung la luc tay dang BIEN HINH de di chuyen (nhieu, khong dai dien).
  - lowvel   : lay N% khung co VAN TOC CO TAY THAP NHAT (khong phan biet vi tri), tinh
               v_norm = |wrist_smoothed[i] - wrist_smoothed[i-1]| / palm_len[i].

Chay: python scratch/handshape_signature_audit.py
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Callable, Dict, List, Sequence

import numpy as np

RAW_DIR = "raw_strokes"
BOUNDARY_FRAC = 0.25   # 25% dau + 25% cuoi
LOWVEL_FRAC = 0.30     # 30% khung van toc thap nhat


def _compute_v_norm(frames: List[dict]) -> List[float]:
    v = [0.0] * len(frames)
    for i in range(1, len(frames)):
        wa = frames[i].get("wrist_smoothed") or frames[i].get("wrist_raw")
        wb = frames[i - 1].get("wrist_smoothed") or frames[i - 1].get("wrist_raw")
        palm = max(float(frames[i].get("palm_len", 1.0) or 1.0), 1e-6)
        if wa is None or wb is None:
            continue
        dx, dy = wa[0] - wb[0], wa[1] - wb[1]
        v[i] = math.hypot(dx, dy) / palm
    return v


def _select_full(frames: List[dict]) -> List[dict]:
    return frames


def _select_boundary(frames: List[dict]) -> List[dict]:
    n = len(frames)
    k = max(1, int(round(n * BOUNDARY_FRAC)))
    if 2 * k >= n:
        return frames
    return frames[:k] + frames[-k:]


def _select_lowvel(frames: List[dict]) -> List[dict]:
    n = len(frames)
    if n <= 2:
        return frames
    v = _compute_v_norm(frames)
    k = max(1, int(round(n * LOWVEL_FRAC)))
    order = sorted(range(n), key=lambda i: v[i])[:k]
    return [frames[i] for i in order]


SELECTORS: Dict[str, Callable[[List[dict]], List[dict]]] = {
    "full": _select_full,
    "boundary": _select_boundary,
    "lowvel": _select_lowvel,
}


def load_word_signatures(raw_dir: str, selector: Callable[[List[dict]], List[dict]]) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {}
    for word_dir in sorted(glob.glob(os.path.join(raw_dir, "*"))):
        if not os.path.isdir(word_dir):
            continue
        word = os.path.basename(word_dir)
        sigs: List[np.ndarray] = []
        for fp in sorted(glob.glob(os.path.join(word_dir, "*.json"))):
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            frames = d.get("frames", [])
            picked = selector(frames)
            angles = [f["flexion_angles"] for f in picked if f.get("flexion_angles") is not None]
            if not angles:
                continue
            arr = np.asarray(angles, dtype=np.float64)  # (n_picked, 5)
            sigs.append(np.median(arr, axis=0))
        if sigs:
            out[word] = sigs
    return out


def compute_separability(sig_map: Dict[str, List[np.ndarray]]) -> Dict[str, tuple]:
    centroids = {w: np.mean(np.stack(s), axis=0) for w, s in sig_map.items()}
    words = list(sig_map.keys())
    result: Dict[str, tuple] = {}
    for w in words:
        sigs = np.stack(sig_map[w])
        c = centroids[w]
        intra_std = float(np.mean(np.linalg.norm(sigs - c, axis=1)))
        others = [o for o in words if o != w]
        other_centroids = np.stack([centroids[o] for o in others])
        dists = np.linalg.norm(other_centroids - c, axis=1)
        inter_gap = float(dists.min()) if len(dists) else float("nan")
        nearest = others[int(np.argmin(dists))] if others else "-"
        sep = inter_gap / (intra_std + 1e-6)
        result[w] = (len(sigs), intra_std, inter_gap, nearest, sep)
    return result


def verdict(sep: float) -> str:
    return "TOT" if sep >= 2.0 else ("TAM" if sep >= 1.2 else "KEM")


def main() -> None:
    per_mode: Dict[str, Dict[str, tuple]] = {}
    for mode, selector in SELECTORS.items():
        sig_map = load_word_signatures(RAW_DIR, selector)
        if not sig_map:
            print(f"[!] Khong tim thay du lieu cho mode={mode}")
            continue
        per_mode[mode] = compute_separability(sig_map)

    words = sorted(next(iter(per_mode.values())).keys())

    print("=" * 118)
    print("SO SANH 3 CACH LAY CHU KY DANG TAY: full (toan bo khung) vs boundary (2 dau mut) vs lowvel (van toc thap nhat)")
    print("=" * 118)
    header = f"{'Tu':<12}"
    for mode in SELECTORS:
        header += f"{'sep_' + mode:<14}"
    header += "Ket luan (mode tot nhat)"
    print(header)
    print("-" * 118)

    rows = []
    for w in words:
        seps = {mode: per_mode[mode][w][4] for mode in SELECTORS if w in per_mode.get(mode, {})}
        best_mode = max(seps, key=lambda m: seps[m]) if seps else "-"
        rows.append((w, seps, best_mode))

    rows.sort(key=lambda r: -max(r[1].values()) if r[1] else 0)

    for w, seps, best_mode in rows:
        line = f"{w:<12}"
        for mode in SELECTORS:
            v = seps.get(mode, float("nan"))
            line += f"{v:<14.2f}"
        line += f"{best_mode} ({verdict(seps.get(best_mode, 0))})"
        print(line)

    # Tom tat gia thuyet: boundary/lowvel co thang full mot cach he thong khong?
    print("\n--- Tom tat gia thuyet 'dau/cuoi on dinh hon' ---")
    n_boundary_better = sum(1 for w in words if per_mode["boundary"][w][4] > per_mode["full"][w][4])
    n_lowvel_better = sum(1 for w in words if per_mode["lowvel"][w][4] > per_mode["full"][w][4])
    print(f"boundary > full o {n_boundary_better}/{len(words)} tu")
    print(f"lowvel   > full o {n_lowvel_better}/{len(words)} tu")
    n_good_full = sum(1 for w in words if per_mode["full"][w][4] >= 2.0)
    n_good_boundary = sum(1 for w in words if per_mode["boundary"][w][4] >= 2.0)
    n_good_lowvel = sum(1 for w in words if per_mode["lowvel"][w][4] >= 2.0)
    print(f"So tu dat nguong TOT (sep>=2.0): full={n_good_full}, boundary={n_good_boundary}, lowvel={n_good_lowvel}")


if __name__ == "__main__":
    main()
