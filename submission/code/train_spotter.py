"""
Huan luyen + do Binary Spotter (ky / khong ky) tren cac phien thu lien tuc (collect_continuous.py).

Nhan nhi phan:  SIGN = 1;  REST / WANDER / TRANSITION = 0;  IGNORE bi loai khoi huan luyen va do.
Dau vao:        cua so 16 khung dac trung tung khung (spotter_features.py), CHI nhin qua khu.
Mo hinh:        LR (hoi quy logistic), HGB (cay tang cuong), CNN (mang 1D-CNN nho) - LR/HGB la moc don gian.

DANH GIA (khong dung du lieu da train de cham diem):
    Chia K khoi. Cac khung cua cung mot lan thu (attempt) luon nam cung mot khoi; WANDER duoc cat thanh
    cac doan 6 giay de du doan ra nhieu khoi. Moi khung duoc du doan boi mo hinh KHONG thay khoi chua no
    (out-of-fold). So sanh voi RuleSpotter bang CUNG cac chi so cua eval_rule_spotter.py.
    Canh bao: du lieu cua MOT nguoi, cac khoi van cung thoi quen ky -> day la can tren cho nguoi la.

Cach dung:
    py train_spotter.py                    # do CV, in bang so sanh
    py train_spotter.py --save             # sau do huan luyen tren TOAN BO du lieu va luu spotter_model.pth
    py train_spotter.py --folds 5 --epochs 60 --seed 0
"""

import argparse
import collections
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from collect_continuous import (
    DEFAULT_MARGIN_SEC,
    P_PAIR,
    P_WANDER,
    REST,
    SIGN,
    TRANSITION,
    WANDER,
    _collect_attempts,
    _events_list,
    find_session_dirs,
    load_session,
)
from eval_rule_spotter import (
    DEFAULT_CLOSE_HOLD,
    DEFAULT_OPEN_MULT,
    compute_vnorm_series,
    measure,
    merge_results,
    pct,
    run_gate,
    span_mask,
)
from spotter_features import (
    FEAT_DIM,
    WINDOW,
    session_features,
    window_stats,
    windows_from_features,
)
from spotter_dl import build_net
from spotter_gate import OneShotLock, ProbGate

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WANDER_CHUNK_SEC = 6.0


# ===========================================================================
# 1. Du lieu
# ===========================================================================

def build_session_data(session_dir: str, margin: float) -> Dict[str, Any]:
    s = load_session(session_dir, margin)
    t = np.asarray(s["t"], dtype=np.float64)
    meta = s.get("meta", {}) or {}
    fw = float(meta.get("frame_width", 640) or 640)
    fh = float(meta.get("frame_height", 480) or 480)
    feats = session_features(np.asarray(s["landmarks"], dtype=np.float64), np.asarray(s["hand_score"], dtype=np.float64),
                             t, fw, fh)
    labels = np.asarray(s["labels"])
    found, v, v_stop = compute_vnorm_series(s)

    # nhom (attempt, doan) cho tung khung -> dung de chia khoi
    atts = _collect_attempts(_events_list(s["events"]), float(t[-1]))
    group = np.full(t.shape[0], "", dtype=object)
    kind_of: Dict[str, str] = {}
    order = sorted(atts.items(), key=lambda kv: kv[1]["tb"])
    for aid, a in order:
        sel = (t >= a["tb"]) & (t <= a["te"])
        for i in np.nonzero(sel)[0]:
            chunk = int((t[i] - a["tb"]) // WANDER_CHUNK_SEC) if a["kind"] == P_WANDER else 0
            g = f"{os.path.basename(session_dir)}:{aid}:{chunk}"
            group[i] = g
            kind_of[g] = a["kind"]
    # khung nam ngoai moi attempt: gan theo nhom gan nhat phia truoc (hoac nhom dau tien)
    last = ""
    first = next((g for g in group if g), "")
    for i in range(t.shape[0]):
        if group[i]:
            last = group[i]
        else:
            group[i] = last or first
    return dict(dir=session_dir, t=t, feats=feats, labels=labels, events=s["events"], found=found, v=v,
                v_stop=v_stop, group=group, kind_of=kind_of)


def assign_folds(sessions: List[Dict[str, Any]], k: int) -> Dict[str, int]:
    """Moi loai prompt xep theo thoi gian roi chia vong tron vao K khoi -> moi khoi co du loai."""
    by_kind: Dict[str, List[str]] = collections.defaultdict(list)
    for d in sessions:
        seen = []
        for g in d["group"]:
            if g not in seen:
                seen.append(g)
        for g in seen:
            by_kind[d["kind_of"].get(g, "REST")].append(g)
    fold_of: Dict[str, int] = {}
    for kind, gs in by_kind.items():
        for r, g in enumerate(gs):
            fold_of[g] = r % k
    return fold_of


def stack_all(sessions: List[Dict[str, Any]], fold_of: Dict[str, int]) -> Dict[str, np.ndarray]:
    wins, y, lab, fold, sid, idx = [], [], [], [], [], []
    for si, d in enumerate(sessions):
        w = windows_from_features(d["feats"], WINDOW)
        wins.append(w)
        lab.append(d["labels"])
        y.append((d["labels"] == SIGN).astype(np.int64))
        fold.append(np.array([fold_of.get(g, 0) for g in d["group"]]))
        sid.append(np.full(d["t"].shape[0], si))
        idx.append(np.arange(d["t"].shape[0]))
    return dict(win=np.concatenate(wins), y=np.concatenate(y), lab=np.concatenate(lab),
                fold=np.concatenate(fold), sid=np.concatenate(sid), idx=np.concatenate(idx))


# ===========================================================================
# 2. Mo hinh
# ===========================================================================

class SpotterCNN:
    """Boc mo hinh torch 1D-CNN nho + chuan hoa; fit/predict_proba giong sklearn."""

    def __init__(self, epochs: int = 40, seed: int = 0, lr: float = 2e-3, wd: float = 1e-3):
        self.epochs, self.seed, self.lr, self.wd = epochs, seed, lr, wd
        self.net = None
        self.mean = None
        self.std = None

    def fit(self, win: np.ndarray, y: np.ndarray) -> "SpotterCNN":
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        flat = win.reshape(-1, FEAT_DIM)
        self.mean = flat.mean(0).astype(np.float32)
        self.std = (flat.std(0) + 1e-3).astype(np.float32)
        X = torch.from_numpy(((win - self.mean) / self.std).astype(np.float32))
        Y = torch.from_numpy(y.astype(np.float32))
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos))
        self.net = build_net()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        n = X.shape[0]
        for _ in range(self.epochs):
            self.net.train()
            perm = rng.permutation(n)
            for b in range(0, n, 256):
                ib = perm[b:b + 256]
                xb = X[ib]
                xb = xb + 0.05 * torch.randn_like(xb)          # nhieu nhe chong hoc thuoc
                opt.zero_grad()
                loss = loss_fn(self.net(xb), Y[ib])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, win: np.ndarray) -> np.ndarray:
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, win.shape[0], 2048):
                xb = torch.from_numpy(((win[b:b + 2048] - self.mean) / self.std).astype(np.float32))
                out.append(torch.sigmoid(self.net(xb)).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def save(self, path: str) -> None:
        import torch
        torch.save({"state_dict": self.net.state_dict(), "mean": self.mean, "std": self.std,
                    "window": WINDOW, "feat_dim": FEAT_DIM, "version": 1}, path)


class StatsModel:
    """LR / HGB tren thong ke cua so (spotter_features.window_stats)."""

    def __init__(self, kind: str, seed: int = 0):
        self.kind, self.seed = kind, seed
        self.m = None

    def fit(self, win: np.ndarray, y: np.ndarray) -> "StatsModel":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = window_stats(win)
        if self.kind == "LR":
            self.m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))
        else:
            self.m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150,
                                                    class_weight="balanced", random_state=self.seed)
        self.m.fit(X, y)
        return self

    def predict_proba(self, win: np.ndarray) -> np.ndarray:
        return self.m.predict_proba(window_stats(win))[:, 1]


def make_model(name: str, epochs: int, seed: int):
    if name == "CNN":
        return SpotterCNN(epochs=epochs, seed=seed)
    return StatsModel(name, seed)


# ===========================================================================
# 3. Do
# ===========================================================================

def oof_probs(name: str, data: Dict[str, np.ndarray], k: int, epochs: int, seed: int) -> np.ndarray:
    """Xac suat out-of-fold cho MOI khung (ke ca IGNORE): mo hinh du doan khoi f khong thay khoi f."""
    p = np.zeros(data["y"].shape[0], dtype=np.float32)
    labelled = data["lab"] != 0
    for f in range(k):
        tr = labelled & (data["fold"] != f)
        te = data["fold"] == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        m = make_model(name, epochs, seed).fit(data["win"][tr], data["y"][tr])
        p[te] = m.predict_proba(data["win"][te])
    return p


def gate_from_prob(t: np.ndarray, prob: np.ndarray, **kw) -> Tuple[np.ndarray, List[Tuple[float, Optional[float]]]]:
    g = ProbGate(**kw)
    lock = OneShotLock(max_spans=10 ** 6)
    gate = np.zeros(t.shape[0], dtype=bool)
    for i in range(t.shape[0]):
        o = g.update(float(t[i]), float(prob[i]))
        lock.observe(float(t[i]), o)
        gate[i] = o
    return gate, [(s[1], s[2]) for s in lock.spans]


def pairs_ok(t: np.ndarray, gate: np.ndarray, events: Any) -> Tuple[int, int]:
    """Cap 2 tu dung nghia: gate MO trong tu 1, DONG o giua (doan chuyen tiep), MO lai trong tu 2."""
    ok = tot = 0
    for a in _collect_attempts(_events_list(events), float(t[-1])).values():
        if a["kind"] != P_PAIR or a["bad"] or a["discarded"] or len(a["boundaries"]) < 2 or not a["signs"]:
            continue
        ss, se = a["signs"][0]
        b0, b1 = sorted(a["boundaries"])[:2]
        fr = lambda lo, hi: float(gate[(t >= lo) & (t <= hi)].mean()) if ((t >= lo) & (t <= hi)).any() else 0.0   # noqa: E731
        tot += 1
        ok += int(fr(ss, b0) >= 0.5 and fr(b1, se) >= 0.5 and fr(b0, b1) < 0.5)
    return ok, tot


def score_frames(prob: np.ndarray, data: Dict[str, np.ndarray]) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    m = data["lab"] != 0
    y = data["y"][m]
    pr = prob[m]
    out = {"auc": float(roc_auc_score(y, pr)) if 0 < y.sum() < len(y) else float("nan")}
    pred = pr >= 0.5
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    out["bacc"] = 0.5 * (tpr + tnr)
    return out


def evaluate_gate(sessions: List[Dict[str, Any]], data: Dict[str, np.ndarray], prob: Optional[np.ndarray],
                  rule: bool = False, **gate_kw) -> Dict[str, Any]:
    rs = []
    pok = ptot = 0
    for si, d in enumerate(sessions):
        if rule:
            gate, spans = run_gate(d["t"], d["found"], d["v"], d["v_stop"], DEFAULT_OPEN_MULT, DEFAULT_CLOSE_HOLD)
        else:
            sel = data["sid"] == si
            gate, spans = gate_from_prob(d["t"], prob[sel], **gate_kw)
        r = measure(d["t"], d["labels"], d["events"], gate, spans)
        r["minutes"] = float((d["t"][-1] - d["t"][0]) / 60.0)
        rs.append(r)
        a, b = pairs_ok(d["t"], gate, d["events"])
        pok += a
        ptot += b
    m = merge_results(rs)
    m["pairs_ok"], m["pairs_ok_total"] = pok, ptot
    return m


def print_table(rows: List[Tuple[str, Dict[str, Any], Optional[Dict[str, float]]]]) -> None:
    print(f"\n{'mo hinh':<10}{'AUC':>7}{'SIGN%':>8}{'WAN%':>8}{'TRN%':>8}{'REST%':>8}{'tu_dung':>9}{'thua/ph':>9}{'cap_dung':>10}{'tre(s)':>8}")
    for name, m, fs in rows:
        co = m["class_open"]
        auc = f"{fs['auc']:.3f}" if fs else "  -  "
        print(f"{name:<10}{auc:>7}{pct(*co['SIGN']):>8}{pct(*co['WANDER']):>8}{pct(*co['TRANSITION']):>8}{pct(*co['REST']):>8}"
              f"{pct(m['words_detected'], m['words']):>9}{m['spurious_per_min']:>9.1f}"
              f"{f'{m['pairs_ok']}/{m['pairs_ok_total']}':>10}{m['latency_median']:>8.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Huan luyen va do Binary Spotter.")
    p.add_argument("--root", default="continuous_data")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN_SEC)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", default="LR,HGB,CNN")
    p.add_argument("--open-thr", type=float, default=0.5)
    p.add_argument("--close-thr", type=float, default=0.35)
    p.add_argument("--close-hold", type=float, default=0.20)
    p.add_argument("--save", action="store_true", help="Sau khi do, huan luyen CNN tren TOAN BO du lieu va luu spotter_model.pth")
    p.add_argument("--out", default="spotter_model.pth")
    p.add_argument("--exclude", action="append", default=[], metavar="TEN_PHIEN",
                   help="Bo cac phien co duong dan chua chuoi nay (lap duoc) - de GIU LAI phien kiem tra doc lap (xem eval_negative_session.py)")
    args = p.parse_args()

    dirs = [d for d in find_session_dirs(args.root) if os.path.exists(os.path.join(d, "frames.npz"))]
    if args.exclude:
        left_out = [d for d in dirs if any(x in d for x in args.exclude)]
        dirs = [d for d in dirs if d not in left_out]
        print(f"[du lieu] giu lai (KHONG dung de huan luyen): {[os.path.basename(d) for d in left_out] or 'khong khop phien nao'}")
    if not dirs:
        print(f"[!] Khong thay phien nao trong '{args.root}'.")
        sys.exit(1)
    print(f"[du lieu] {len(dirs)} phien; dang trich dac trung + v_norm ...")
    sessions = [build_session_data(d, args.margin) for d in dirs]
    fold_of = assign_folds(sessions, args.folds)
    data = stack_all(sessions, fold_of)
    lab = data["lab"]
    print(f"[du lieu] khung co nhan: SIGN {(lab == SIGN).sum()}, REST {(lab == REST).sum()}, "
          f"WANDER {(lab == WANDER).sum()}, TRANSITION {(lab == TRANSITION).sum()} | khoi: "
          f"{[int(((data['fold'] == f) & (lab != 0)).sum()) for f in range(args.folds)]} khung/khoi")

    gate_kw = dict(open_thr=args.open_thr, close_thr=args.close_thr, close_hold_sec=args.close_hold)
    rows: List[Tuple[str, Dict[str, Any], Optional[Dict[str, float]]]] = []
    m_rule = evaluate_gate(sessions, data, None, rule=True)
    rows.append(("Rule", m_rule, None))
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        t0 = time.time()
        prob = oof_probs(name, data, args.folds, args.epochs, args.seed)
        fs = score_frames(prob, data)
        rows.append((name, evaluate_gate(sessions, data, prob, **gate_kw), fs))
        print(f"[{name}] xong {time.time() - t0:.0f}s | AUC {fs['auc']:.3f} | do chinh xac can bang {fs['bacc']:.3f}")

    print("\n=== SO SANH TREN KHOI GIU LAI (out-of-fold) ===")
    print("SIGN% cao tot | WAN%/TRN%/REST% = gate mo o lop am, thap tot | tu_dung = tu duoc gate mo >= 50% | "
          "thua/ph = span thua moi phut | cap_dung = cap 2 tu co gate mo-DONG-mo dung cho")
    print_table(rows)
    print(f"\ncong gate DL: mo khi p>={args.open_thr} , dong khi p<{args.close_thr} giu {args.close_hold}s")
    print("Luu y: du lieu MOT nguoi -> day la CAN TREN cho nguoi la, khong phai do chinh xac thuc te.")

    if args.save:
        t0 = time.time()
        labelled = data["lab"] != 0
        m = SpotterCNN(epochs=args.epochs, seed=args.seed).fit(data["win"][labelled], data["y"][labelled])
        m.save(args.out)
        print(f"[luu] {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB) - huan luyen tren TOAN BO du lieu, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
