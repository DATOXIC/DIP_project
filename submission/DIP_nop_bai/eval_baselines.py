"""
So sánh DTW hiện tại với các phương pháp cơ sở (baseline) học máy trên CÙNG bank, CÙNG cách chia, CÙNG đặc trưng 24D.

Mục đích khoa học: trả lời câu hỏi người phản biện sẽ hỏi — "Tại sao không dùng học sâu?" — bằng số liệu.
Không cần dữ liệu mới, không dùng ảnh (chỉ quỹ đạo 32x24 đã có trong bank), chạy được trên CPU.

Phương pháp:
  dtw      : GoldenTriadDTWMatcher với cấu hình scoreboard (mặc định configs/G1p90.json). Điểm "biết" = -r_ratio (d1/độ rộng).
  knn      : 1-NN khoảng cách Euclid trên đặc trưng làm phẳng đã chuẩn hóa (không DTW). Điểm "biết" = -d1.
  mlp      : MLP nhỏ (768-128-C), softmax. Điểm "biết" = xác suất lớn nhất.
  gru      : GRU hai chiều nhỏ (24-48x2-C), softmax. Điểm "biết" = xác suất lớn nhất.

Bốn phép đo (giống eval_open_set.py, để so sánh trực tiếp):
  LOSO : query = mẫu của phiên bị giữ lại, huấn luyện/tham chiếu = các phiên còn lại.
  Mai  : query = người lạ (profiles/Mai.json), huấn luyện/tham chiếu = cả bank. CHỈ ĐỂ ĐO.
  LOWO : bỏ hẳn một từ khỏi bank, query = mẫu của từ đó (từ lạ thật sự).
  NEG  : nét âm trong negative_strokes.json, huấn luyện/tham chiếu = cả bank.

Chỉ số:
  - top-1 đóng (LOSO, Mai): đoán đúng từ khi ép chọn một từ.
  - AUROC mở: khả năng phân biệt "từ đã biết" (LOSO) với "từ lạ" (LOWO + NEG) bằng điểm tin cậy, KHÔNG phụ thuộc ngưỡng.
  - Điểm vận hành: ngưỡng của baseline được chọn sao cho số câu LOSO nhận-đúng BẰNG dtw; sau đó xem người lạ nhận đúng,
    nhận nhầm từ lạ (LOWO), nhận nhầm nét âm (NEG). (Ngưỡng chọn trên chính LOSO — cũng như k của dtw đã chỉnh trên LOSO.)

Cách dùng:
    py eval_baselines.py
    py eval_baselines.py --models dtw knn mlp gru --seeds 3 --dtw-config configs/G1p90.json --dtw-config configs/C11_base.json
Kết quả JSON: results/baselines/eval_baselines.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# ============================================================================ dữ liệu

def _traj(s: Dict[str, Any]) -> np.ndarray:
    raw = s.get("traj_features_9d") if s.get("traj_features_9d") is not None else s.get("traj_features")
    return np.nan_to_num(np.asarray(raw, dtype=np.float32))


def _is_orig(s: Dict[str, Any]) -> bool:
    return s.get("is_original", True) is not False and not s.get("augment_of")


def bank_arrays(bank: Dict[str, Any], words: Sequence[str], skip_word: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(N,T,D) và nhãn (N,) từ các mẫu GỐC của bank, có thể bỏ một từ."""
    idx = {w: i for i, w in enumerate(words)}
    X, y = [], []
    for w, d in bank["vocabulary"].items():
        if w == skip_word or w not in idx:
            continue
        for s in d.get("samples", []):
            if _is_orig(s):
                X.append(_traj(s))
                y.append(idx[w])
    return np.stack(X), np.asarray(y, dtype=np.int64)


# ============================================================================ mô hình học máy

class _Std:
    def fit(self, X: np.ndarray) -> "_Std":
        flat = X.reshape(-1, X.shape[-1])
        self.mu = flat.mean(0, keepdims=True)
        self.sd = np.maximum(flat.std(0, keepdims=True), 1e-6)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)


class KNN:
    name = "knn"

    def fit(self, X: np.ndarray, y: np.ndarray, C: int, seed: int) -> "KNN":
        self.std = _Std().fit(X)
        self.Xf = self.std(X).reshape(len(X), -1)
        self.y = y
        return self

    def predict(self, Xq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Q = self.std(Xq).reshape(len(Xq), -1)
        d = np.sqrt(((Q[:, None, :] - self.Xf[None, :, :]) ** 2).sum(-1))
        j = d.argmin(1)
        return self.y[j], -d[np.arange(len(Q)), j]


class _TorchModel:
    """Khung huấn luyện chung: toàn bộ lô, AdamW, làm mịn nhãn, nhiễu Gaussian nhẹ (cùng cho mọi mạng)."""
    epochs = 200

    def _build(self, T: int, D: int, C: int):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, C: int, seed: int):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.std = _Std().fit(X)
        Xt = torch.from_numpy(self.std(X))
        yt = torch.from_numpy(y)
        self.net = self._build(X.shape[1], X.shape[2], C)
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = lossf(self.net(Xt + 0.05 * torch.randn_like(Xt)), yt)
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    def predict(self, Xq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            p = torch.softmax(self.net(torch.from_numpy(self.std(Xq))), dim=-1).numpy()
        return p.argmax(1), p.max(1)


class MLP(_TorchModel):
    name = "mlp"

    def _build(self, T, D, C):
        import torch.nn as nn
        return nn.Sequential(nn.Flatten(), nn.Linear(T * D, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, C))


class GRU(_TorchModel):
    name = "gru"

    def _build(self, T, D, C):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(D, 48, batch_first=True, bidirectional=True)
                self.drop = nn.Dropout(0.3)
                self.head = nn.Linear(96, C)

            def forward(self, x):
                o, _ = self.rnn(x)
                return self.head(self.drop(o.mean(1)))
        return Net()


MODELS = {"knn": KNN, "mlp": MLP, "gru": GRU}


# ============================================================================ thu điểm cho từng phép đo

def _fold_arrays(f) -> Tuple[np.ndarray, np.ndarray]:
    Xq = np.stack([np.nan_to_num(np.asarray(q.traj, dtype=np.float32)) for q in f.queries])
    return Xq, np.asarray([q.true_word for q in f.queries])


def collect_ml(model_cls, bank, folds, neg, signer_vocab, words, seed) -> Dict[str, np.ndarray]:
    C = len(words)
    idx = {w: i for i, w in enumerate(words)}
    out: Dict[str, List[Any]] = {k: [] for k in ("loso_ok", "loso_s", "v01_ok", "v01_s", "lowo_s", "neg_s")}

    for f in folds:
        X, y = bank_arrays(f.reference_bank, words)
        m = model_cls().fit(X, y, C, seed)
        Xq, tw = _fold_arrays(f)
        pred, sc = m.predict(Xq)
        out["loso_ok"] += [bool(p == idx.get(t, -1)) for p, t in zip(pred, tw)]
        out["loso_s"] += list(sc)

    X, y = bank_arrays(bank, words)
    full = model_cls().fit(X, y, C, seed)
    if signer_vocab:
        Xq, tw = [], []
        for w, d in signer_vocab.items():
            for s in d.get("samples", []):
                Xq.append(_traj(s))
                tw.append(w)
        pred, sc = full.predict(np.stack(Xq))
        out["v01_ok"] += [bool(p == idx.get(t, -1)) for p, t in zip(pred, tw)]
        out["v01_s"] += list(sc)
    if neg:
        _, sc = full.predict(np.stack([_traj(s) for s in neg]))
        out["neg_s"] += list(sc)

    for W in words:
        Xw = np.stack([_traj(s) for s in bank["vocabulary"][W]["samples"] if _is_orig(s)])
        X, y = bank_arrays(bank, words, skip_word=W)
        m = model_cls().fit(X, y, C, seed)
        _, sc = m.predict(Xw)
        out["lowo_s"] += list(sc)
    return {k: np.asarray(v) for k, v in out.items()}


def _dtw_call(m, traj, shapes, handed, chir, has_body, rep):
    return m.match(query_traj_9d=np.asarray(traj, dtype=np.float32), query_shapes=shapes,
                   handedness=("Right" if chir else handed), query_has_body=has_body, query_rep_count=rep)


def _dtw_pack(r, true_word: Optional[str]) -> Tuple[bool, float, bool]:
    top = r.candidates[0].word_key if r.candidates else None
    return (top == true_word), -float(r.r_ratio), (r.status == "ACCEPTED")


def collect_dtw(cfg, bank, folds, neg, signer_vocab, words) -> Dict[str, np.ndarray]:
    out: Dict[str, List[Any]] = {k: [] for k in ("loso_ok", "loso_s", "loso_acc", "v01_ok", "v01_s", "v01_acc",
                                                  "lowo_s", "lowo_acc", "neg_s", "neg_acc")}
    for f in folds:
        m = build_matcher_from_config(cfg, f.reference_bank)
        for q in f.queries:
            ok, s, acc = _dtw_pack(_dtw_call(m, q.traj, q.hand_shapes, q.handedness, q.chirality_canonical,
                                             q.has_body, q.rep_count), q.true_word)
            out["loso_ok"].append(ok); out["loso_s"].append(s); out["loso_acc"].append(acc)
    full = build_matcher_from_config(cfg, bank)

    def call_dict(s, true_word):
        r = _dtw_call(full, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                      bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
        return _dtw_pack(r, true_word)

    if signer_vocab:
        for w, d in signer_vocab.items():
            for s in d.get("samples", []):
                ok, sc, acc = call_dict(s, w)
                out["v01_ok"].append(ok); out["v01_s"].append(sc); out["v01_acc"].append(acc)
    for s in neg:
        _, sc, acc = call_dict(s, None)
        out["neg_s"].append(sc); out["neg_acc"].append(acc)
    for W in words:
        sub = {"metadata": bank.get("metadata", {}),
               "vocabulary": {k: v for k, v in bank["vocabulary"].items() if k != W}}
        m = build_matcher_from_config(cfg, sub)
        for s in bank["vocabulary"][W]["samples"]:
            if not _is_orig(s):
                continue
            r = _dtw_call(m, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                          bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
            out["lowo_s"].append(-float(r.r_ratio)); out["lowo_acc"].append(r.status == "ACCEPTED")
    return {k: np.asarray(v) for k, v in out.items()}


# ============================================================================ chỉ số

def auroc(known: np.ndarray, unknown: np.ndarray) -> float:
    """P(điểm biết > điểm lạ) (hòa = 1/2). Không phụ thuộc ngưỡng."""
    if len(known) == 0 or len(unknown) == 0:
        return float("nan")
    k = np.asarray(known)[:, None]
    u = np.asarray(unknown)[None, :]
    return float(((k > u).sum() + 0.5 * (k == u).sum()) / (len(known) * len(unknown)))


def operating_point(d: Dict[str, np.ndarray], target_correct: int) -> Dict[str, Any]:
    """Ngưỡng tau sao cho số câu LOSO (đoán đúng và điểm >= tau) = target_correct; rồi đo các phép còn lại."""
    ok, s = d["loso_ok"].astype(bool), d["loso_s"]
    cs = np.sort(s[ok])[::-1]
    tau = cs[min(max(target_correct, 1), len(cs)) - 1]
    acc = s >= tau
    res = {
        "loso_accept_correct": int((acc & ok).sum()),
        "loso_accept_wrong": int((acc & ~ok).sum()),
    }
    if len(d["v01_s"]):
        res["v01_accept_correct"] = int(((d["v01_s"] >= tau) & d["v01_ok"].astype(bool)).sum())
        res["v01_n"] = int(len(d["v01_s"]))
    res["lowo_false_accept"] = int((d["lowo_s"] >= tau).sum()); res["lowo_n"] = int(len(d["lowo_s"]))
    res["neg_false_accept"] = int((d["neg_s"] >= tau).sum()); res["neg_n"] = int(len(d["neg_s"]))
    return res


def open_metrics(d: Dict[str, np.ndarray]) -> Dict[str, Any]:
    unknown = np.concatenate([d["lowo_s"], d["neg_s"]])
    m = {
        "loso_top1": float(d["loso_ok"].mean()),
        "loso_n": int(len(d["loso_ok"])),
        "auroc_all": auroc(d["loso_s"], unknown),
        "auroc_lowo": auroc(d["loso_s"], d["lowo_s"]),
        "auroc_neg": auroc(d["loso_s"], d["neg_s"]),
    }
    if len(d["v01_ok"]):
        m["v01_top1"] = float(d["v01_ok"].mean()); m["v01_n"] = int(len(d["v01_ok"]))
    return m


# ============================================================================ chạy

def main() -> None:
    p = argparse.ArgumentParser(description="So sánh DTW với baseline học máy (LOSO/Mai/LOWO/NEG).")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--neg", default="negative_strokes.json")
    p.add_argument("--signer", default="Mai")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--models", nargs="+", default=["dtw", "knn", "mlp", "gru"], choices=["dtw", "knn", "mlp", "gru"])
    p.add_argument("--dtw-config", action="append", default=None, help="Lặp được; mặc định configs/G1p90.json")
    p.add_argument("--seeds", type=int, default=3, help="Số hạt giống cho mlp/gru (kết quả = trung bình).")
    p.add_argument("--out", default=os.path.join("results", "baselines", "eval_baselines.json"))
    args = p.parse_args()

    bank, _ = load_bank(args.bank)
    neg, _ = load_negative_strokes(args.neg)
    sp = os.path.join(args.profiles_dir, f"{args.signer}.json")
    signer_vocab = None
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as fh:
            signer_vocab = json.load(fh).get("vocabulary", {})
    words = sorted(bank["vocabulary"].keys())
    _, folds = split_dataset(bank)
    print(f"Bank {args.bank}: {len(words)} từ | {len(folds)} lượt LOSO | người lạ: {args.signer if signer_vocab else '(không có)'} | nét âm: {len(neg)}")

    rows: List[Dict[str, Any]] = []
    dtw_target: Optional[int] = None
    t0 = time.time()

    if "dtw" in args.models:
        for cp in (args.dtw_config or ["configs/G1p90.json"]):
            cfg, _ = validate_and_hash_config(cp)
            d = collect_dtw(cfg, bank, folds, neg, signer_vocab, words)
            m = open_metrics(d)
            acc_ok = d["loso_acc"].astype(bool) & d["loso_ok"].astype(bool)
            m["own_op"] = {
                "loso_accept_correct": int(acc_ok.sum()),
                "loso_accept_wrong": int((d["loso_acc"].astype(bool) & ~d["loso_ok"].astype(bool)).sum()),
                "v01_accept_correct": int((d["v01_acc"].astype(bool) & d["v01_ok"].astype(bool)).sum()) if len(d["v01_ok"]) else None,
                "v01_n": int(len(d["v01_ok"])),
                "lowo_false_accept": int(d["lowo_acc"].sum()), "lowo_n": int(len(d["lowo_acc"])),
                "neg_false_accept": int(d["neg_acc"].sum()), "neg_n": int(len(d["neg_acc"])),
            }
            name = "dtw:" + os.path.splitext(os.path.basename(cp))[0]
            rows.append({"name": name, "seeds": 1, "metrics": [m], "matched": None})
            if dtw_target is None:
                dtw_target = int(acc_ok.sum())
            print(f"[{time.time() - t0:5.0f}s] {name} xong")
    if dtw_target is None:
        dtw_target = 155

    for mn in [x for x in args.models if x != "dtw"]:
        per_seed, per_seed_op = [], []
        for sd in range(args.seeds if mn != "knn" else 1):
            d = collect_ml(MODELS[mn], bank, folds, neg, signer_vocab, words, seed=1000 + sd)
            per_seed.append(open_metrics(d))
            per_seed_op.append(operating_point(d, dtw_target))
            print(f"[{time.time() - t0:5.0f}s] {mn} hạt {sd} xong")
        rows.append({"name": mn, "seeds": len(per_seed), "metrics": per_seed, "matched": per_seed_op})

    # --------------------------------------------------------------- in bảng
    def mean(rs: List[Dict[str, Any]], k: str) -> float:
        v = [r[k] for r in rs if r.get(k) is not None]
        return float(np.mean(v)) if v else float("nan")

    def sd(rs: List[Dict[str, Any]], k: str) -> float:
        v = [r[k] for r in rs if r.get(k) is not None]
        return float(np.std(v)) if len(v) > 1 else 0.0

    print("\n=== (1) ĐÓNG + AUROC MỞ (không phụ thuộc ngưỡng; AUROC cao = tách từ lạ tốt) ===")
    print(f"{'phương pháp':<20}{'LOSO top-1':>12}{'Mai top-1':>12}{'AUROC tất cả':>14}{'AUROC từ lạ':>13}{'AUROC nét âm':>14}")
    for r in rows:
        ms = r["metrics"]
        def f(k, pct=False):
            v, s_ = mean(ms, k), sd(ms, k)
            if pct:
                return f"{100 * v:.1f}%" + (f"±{100 * s_:.1f}" if len(ms) > 1 else "")
            return f"{v:.3f}" + (f"±{s_:.3f}" if len(ms) > 1 else "")
        print(f"{r['name']:<20}{f('loso_top1', True):>12}{f('v01_top1', True):>12}{f('auroc_all'):>14}{f('auroc_lowo'):>13}{f('auroc_neg'):>14}")

    print(f"\n=== (2) ĐIỂM VẬN HÀNH: baseline được chỉnh ngưỡng để LOSO nhận-đúng = {dtw_target} (bằng dtw đầu tiên) ===")
    print(f"{'phương pháp':<20}{'LOSO đúng':>11}{'LOSO sai':>10}{'Mai nhận đúng':>15}{'từ lạ nhận nhầm':>17}{'nét âm nhận nhầm':>18}")
    for r in rows:
        if r["matched"] is None:
            o = r["metrics"][0]["own_op"]
            lab = "(ngưỡng riêng)"
            vals = [o]
        else:
            lab = ""
            vals = r["matched"]
        def g(k):
            return float(np.mean([v[k] for v in vals if v.get(k) is not None]))
        n_v01 = vals[0].get("v01_n", 0)
        print(f"{r['name'] + lab:<20}{g('loso_accept_correct'):>11.1f}{g('loso_accept_wrong'):>10.1f}"
              f"{g('v01_accept_correct'):>10.1f}/{n_v01:<4}{g('lowo_false_accept'):>12.1f}/{vals[0]['lowo_n']:<4}"
              f"{g('neg_false_accept'):>13.1f}/{vals[0]['neg_n']:<4}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"bank": args.bank, "dtw_target": dtw_target, "rows": rows}, fh, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}   (tổng {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
