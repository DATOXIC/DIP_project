# -*- coding: utf-8 -*-
"""
Buoc 2 (sau audit chu ky): xay bo phan loai dang tay DON GIAN (centroid + KNN, tren
flexion_angles 5D, chu ky trich theo mode 'boundary' - da xac dinh la lua chon tot
nhat o audit truoc) CHI cho tap tu da qua sang loc: BAD, DRINK, PLEASE, THANK_YOU.

Danh gia bang Leave-One-Out (LOO) tren chinh du lieu that trong raw_strokes/ - dung
phuong phap da dung cho DTW truoc day (xem project_domain_shift_findings), khong
dung random train_test_split de tranh danh gia lac quan gia.

In ra:
  - Accuracy tong the (centroid 1-NN va KNN k=3)
  - Ma tran nham lan 4x4 (quan trong nhat: PLEASE vs THANK_YOU co con nham khong?)

Chay: python scratch/handshape_classifier_loo.py
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

RAW_DIR = "raw_strokes"
TARGET_WORDS = ["BAD", "DRINK", "PLEASE", "THANK_YOU"]
BOUNDARY_FRAC = 0.25


def _select_boundary(frames: List[dict]) -> List[dict]:
    n = len(frames)
    k = max(1, int(round(n * BOUNDARY_FRAC)))
    if 2 * k >= n:
        return frames
    return frames[:k] + frames[-k:]


def load_signatures(words: List[str]) -> Tuple[np.ndarray, List[str]]:
    X, y = [], []
    for word in words:
        word_dir = os.path.join(RAW_DIR, word)
        for fp in sorted(glob.glob(os.path.join(word_dir, "*.json"))):
            d = json.load(open(fp, encoding="utf-8"))
            frames = d.get("frames", [])
            picked = _select_boundary(frames)
            angles = [f["flexion_angles"] for f in picked if f.get("flexion_angles") is not None]
            if not angles:
                continue
            sig = np.median(np.asarray(angles, dtype=np.float64), axis=0)
            X.append(sig)
            y.append(word)
    return np.stack(X), y


def loo_eval(X: np.ndarray, y: List[str], k_nn: int = 3) -> None:
    n = len(y)
    words = sorted(set(y))
    idx = {w: i for i, w in enumerate(words)}
    conf_1nn = np.zeros((len(words), len(words)), dtype=int)
    conf_knn = np.zeros((len(words), len(words)), dtype=int)
    correct_1nn = 0
    correct_knn = 0

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr, ytr = X[mask], [y[j] for j in range(n) if mask[j]]
        q = X[i]
        dists = np.linalg.norm(Xtr - q, axis=1)

        # centroid-nearest-neighbor duoi dang 1-NN thuan
        nn = int(np.argmin(dists))
        pred_1nn = ytr[nn]
        conf_1nn[idx[y[i]], idx[pred_1nn]] += 1
        correct_1nn += int(pred_1nn == y[i])

        # KNN k=3 (bo phieu)
        order = np.argsort(dists)[:k_nn]
        votes: Dict[str, int] = {}
        for o in order:
            votes[ytr[o]] = votes.get(ytr[o], 0) + 1
        pred_knn = max(votes, key=lambda w: votes[w])
        conf_knn[idx[y[i]], idx[pred_knn]] += 1
        correct_knn += int(pred_knn == y[i])

    print("=" * 70)
    print(f"LOO tren {n} mau, {len(words)} lop: {words}")
    print("=" * 70)
    print(f"1-NN accuracy : {correct_1nn}/{n} = {100*correct_1nn/n:.1f}%")
    print(f"KNN(k={k_nn}) accuracy : {correct_knn}/{n} = {100*correct_knn/n:.1f}%")

    def print_conf(name: str, conf: np.ndarray) -> None:
        print(f"\nMa tran nham lan ({name}) - hang=that, cot=du_doan:")
        header = "".ljust(12) + "".join(w[:8].ljust(10) for w in words)
        print(header)
        for i, w in enumerate(words):
            row = w.ljust(12) + "".join(str(conf[i, j]).ljust(10) for j in range(len(words)))
            print(row)

    print_conf("1-NN", conf_1nn)
    print_conf(f"KNN k={k_nn}", conf_knn)

    if "PLEASE" in idx and "THANK_YOU" in idx:
        p, t = idx["PLEASE"], idx["THANK_YOU"]
        cross = conf_knn[p, t] + conf_knn[t, p]
        print(f"\n>>> PLEASE<->THANK_YOU nham lan (KNN): {cross} mau trong tong "
              f"{conf_knn[p].sum() + conf_knn[t].sum()} mau cua 2 tu nay.")


def main() -> None:
    X, y = load_signatures(TARGET_WORDS)
    loo_eval(X, y)


if __name__ == "__main__":
    main()
