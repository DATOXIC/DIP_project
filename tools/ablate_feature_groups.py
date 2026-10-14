"""
CỔNG ĐO TỪNG NHÓM ĐẶC TRƯNG — ABLATION LOO TRÊN BANK ASL CITIZEN 24D (SPEC 03, NHIỆM VỤ 5)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

MỤC ĐÍCH
--------
Thêm chiều KHÔNG đơn điệu tăng độ chính xác (bằng chứng: thêm nhãn C_Shape làm 79.9% -> 54.7%).
Hàm chi phí là tổng có trọng số nên một nhóm nhiễu sẽ PHA LOÃNG ngân sách của các nhóm có ích.
Script này đo LOO top-1 + margin trung bình cho TỪNG nhóm mới một cách riêng rẽ, rồi áp QUY TẮC
QUYẾT ĐỊNH của spec: nhóm nào KHÔNG làm tăng margin trung bình so với mốc 17D thì bị ĐỀ XUẤT LOẠI
(đặt trọng số 0.0 ở cấu hình production; code vẫn giữ để bật lại thử nghiệm sau).

Trọng số KHỞI ĐIỂM (KHÔNG dò tối ưu — việc đó thuộc SPEC 04): w_z=0.5, w_b=0.8, w_s=0.4, w_r=0.15.

PHÉP ĐO
-------
  - LOO: mỗi mẫu làm query, template = mọi mẫu còn lại (bỏ chính nó).
  - top-1 = từ có template gần nhất trùng nhãn.
  - margin = d(từ sai gần nhất) - d(từ đúng gần nhất), lấy trung bình trên mọi query.
  - Trọng số 4 kênh cũ: mặc định "global" (w_p=0.6 w_a=1.0 w_m=0.4 w_n=0.8 — cấu hình đã tái hiện đúng mốc 79.9%); "persign" = như production (ASL-LEX + pair override qua
    GoldenTriadDTWMatcher._get_weights_for_sign), cho kết quả thấp hơn nhiều (mốc 17D chỉ 62.6%).
  - Nhóm mốc cơ thể (w_b) tự vô hiệu cho cặp nào có một phía body_anchor_valid=False.
  - Chạy song song nhiều tiến trình (mỗi query một tác vụ).

CÁCH DÙNG
---------
    py ablate_feature_groups.py                                   # asl_citizen_templates_24d.json
    py ablate_feature_groups.py --bank X.json --weights global
    py ablate_feature_groups.py --extra                           # thêm các dòng tham khảo
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from dynamic_dtw_engine import (
    GoldenTriadDTWMatcher,
    compute_golden_triad_dtw_distance,
    canonicalize_chirality,
)
from dynamic_features import NEW_WEIGHT_KEYS

START_WEIGHTS = {"w_z": 0.5, "w_b": 0.8, "w_s": 0.4, "w_r": 0.15}
GLOBAL_WEIGHTS = (0.6, 1.0, 0.4, 0.8)   # w_p, w_a, w_m, w_n
WORDS_OF_INTEREST = ["FATHER", "MOTHER", "THANK_YOU", "GOOD", "YES", "BYE", "ME", "COME", "YOU"]

# (nhãn cấu hình, dict trọng số mới)
CONFIGS: List[Tuple[str, Dict[str, float]]] = [
    ("Mốc 17D (w_z=w_b=w_s=w_r=0)", {"w_z": 0.0, "w_b": 0.0, "w_s": 0.0, "w_r": 0.0}),
    ("chỉ z_norm (w_z=0.5)",        {"w_z": 0.5, "w_b": 0.0, "w_s": 0.0, "w_r": 0.0}),
    ("chỉ mốc cơ thể (w_b=0.8)",    {"w_z": 0.0, "w_b": 0.8, "w_s": 0.0, "w_r": 0.0}),
    ("chỉ độ xòe ngón (w_s=0.4)",   {"w_z": 0.0, "w_b": 0.0, "w_s": 0.4, "w_r": 0.0}),
    ("chỉ số nhịp lặp (w_r=0.15)",  {"w_z": 0.0, "w_b": 0.0, "w_s": 0.0, "w_r": 0.15}),
    ("tất cả (0.5/0.8/0.4/0.15)",   dict(START_WEIGHTS)),
]
GROUP_OF_CONFIG = {1: "z_norm", 2: "mốc cơ thể", 3: "độ xòe ngón", 4: "số nhịp lặp"}

# ---------------------------------------------------------------------------
# Trạng thái toàn cục của tiến trình con (nạp 1 lần qua initializer)
# ---------------------------------------------------------------------------
_G: Dict[str, Any] = {}


def _init_worker(samples, word_list, weights_by_word, global_w):
    _G["samples"] = samples
    _G["word_list"] = word_list
    _G["weights_by_word"] = weights_by_word
    _G["global_w"] = global_w


def _pair_distance(q: Dict[str, Any], t: Dict[str, Any], cfg: Dict[str, float], base_w: Tuple[float, float, float, float]) -> float:
    wb_eff = cfg["w_b"] if (q["body_ok"] and t["body_ok"]) else 0.0
    return compute_golden_triad_dtw_distance(
        q_traj=q["traj"], q_shapes=q["shapes"], t_traj=t["traj"], t_shapes=t["shapes"],
        w_p=base_w[0], w_a=base_w[1], w_m=base_w[2], w_n=base_w[3],
        lambda_shape=0.0, lambda_shape_global=0.0,
        w_z=cfg["w_z"], w_b=wb_eff, w_s=cfg["w_s"], w_r=cfg["w_r"],
        q_rep_count=q["rep"], t_rep_count=t["rep"],
    )


def _query_task(args) -> Tuple[int, np.ndarray]:
    """Với query i: trả về vector khoảng cách tối thiểu tới từng TỪ (bỏ chính nó)."""
    i, cfg = args
    samples = _G["samples"]
    word_list = _G["word_list"]
    q = samples[i]
    best = np.full(len(word_list), np.inf)
    for j, t in enumerate(samples):
        if j == i:
            continue
        base_w = _G["weights_by_word"].get(t["word"], _G["global_w"])  # trọng số của ỨNG VIÊN (như production)
        d = _pair_distance(q, t, cfg, base_w)
        k = word_list.index(t["word"])
        if d < best[k]:
            best[k] = d
    return i, best


def load_bank(path: str, chirality: str = "stored", body_flip_variant: bool = False) -> List[Dict[str, Any]]:
    """
    Nạp bank 24D. chirality:
      "stored"    : dùng đúng vector đã lưu (đã chuẩn hóa bằng nhãn tay thuận lúc trích xuất).
      "heuristic" : hoàn tác phép lật đã lưu rồi chuẩn hóa lại bằng heuristic max|nx| (handedness=None) —
                    chỉ dùng để đối chiếu với mốc 79.9% đo trên bank cũ (vốn chuẩn hóa bằng heuristic).
    body_flip_variant=True: KHÔNG lật cột 18 (mốc cơ thể x) khi lật chirality — dòng tham khảo cho quyết định thiết kế.
    """
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)["vocabulary"]
    out: List[Dict[str, Any]] = []
    for word, entry in vocab.items():
        for s in entry.get("samples", []):
            traj = np.array(s["traj_features"], dtype=np.float32)
            if traj.shape != (32, 24):
                continue
            flipped = bool(s.get("chirality_flipped_on_save", False))
            if chirality == "heuristic" and flipped:
                for c in (0, 2, 9, 18):
                    traj[:, c] = -traj[:, c]
                traj, _ = canonicalize_chirality(traj, handedness=None)
            elif body_flip_variant and flipped:
                traj[:, 18] = -traj[:, 18]   # hoàn tác lật cột 18 (biến thể không lật mốc cơ thể x)
            out.append({
                "word": word, "traj": traj,
                "shapes": s.get("hand_shapes", ["Unknown"] * 32),
                "rep": s.get("rep_count"),
                "body_ok": bool(s.get("body_anchor_valid", True)),
                "src": s.get("source_video", ""),
            })
    return out


def run_loo(samples: List[Dict[str, Any]], cfg: Dict[str, float], weights_by_word, workers: int) -> Dict[str, Any]:
    word_list = sorted({s["word"] for s in samples})
    n = len(samples)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(samples, word_list, weights_by_word, GLOBAL_WEIGHTS)) as ex:
        results = list(ex.map(_query_task, [(i, cfg) for i in range(n)], chunksize=2))
    results.sort(key=lambda r: r[0])

    hits = 0
    margins: List[float] = []
    dcor: List[float] = []
    per_word: Dict[str, List[int]] = {w: [0, 0] for w in word_list}
    per_word_margin: Dict[str, List[float]] = {w: [] for w in word_list}
    confusions: Dict[Tuple[str, str], int] = {}
    for i, best in results:
        w = samples[i]["word"]
        k = word_list.index(w)
        top = word_list[int(np.argmin(best))]
        per_word[w][1] += 1
        if top == w:
            hits += 1
            per_word[w][0] += 1
        else:
            confusions[(w, top)] = confusions.get((w, top), 0) + 1
        d_corr = best[k]
        wrong = np.delete(best, k)
        d_wrong = float(np.min(wrong)) if wrong.size else float("inf")
        margins.append(d_wrong - d_corr)
        per_word_margin[w].append(d_wrong - d_corr)
        dcor.append(d_corr)
    return {
        "hits": hits, "n": n, "top1": hits / n, "margin": float(np.mean(margins)),
        "d_correct": float(np.mean(dcor)), "per_word": per_word,
        "per_word_margin": {w: float(np.mean(v)) if v else float("nan") for w, v in per_word_margin.items()},
        "confusions": confusions,
    }


def build_weights_by_word(bank_path: str, mode: str) -> Dict[str, Tuple[float, float, float, float]]:
    if mode == "global":
        return {}
    m = GoldenTriadDTWMatcher(templates_path=bank_path, strict_weights=False)
    return {w: m._get_weights_for_sign(w) for w in m.vocabulary.keys()}


def fmt_row(label: str, r: Dict[str, Any], words: List[str]) -> str:
    cells = []
    for w in words:
        c, t = r["per_word"].get(w, [0, 0])
        cells.append(f"{c}/{t}" if t else "-")
    return (f"{label:<32}{r['hits']:>4}/{r['n']:<4}{100*r['top1']:>6.1f}%{r['margin']:>+9.3f}{r['d_correct']:>9.3f}  "
            + " ".join(f"{c:>7}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation LOO tung nhom dac trung tren bank Citizen 24D.")
    ap.add_argument("--bank", default="asl_citizen_templates_24d.json")
    ap.add_argument("--weights", choices=["persign", "global"], default="global")
    ap.add_argument("--chirality", choices=["stored", "heuristic"], default="stored")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--extra", action="store_true", help="Them cac dong tham khao (khong lat cot 18, chi z=0.2...).")
    args = ap.parse_args()

    samples = load_bank(args.bank, args.chirality)
    n_body = sum(1 for s in samples if s["body_ok"])
    print("=" * 130)
    print(f"  ABLATION LOO — bank: {args.bank} | {len(samples)} mau / {len({s['word'] for s in samples})} tu | "
          f"trong so 4 kenh cu: {args.weights} | chirality: {args.chirality}")
    print(f"  Mau co moc co the hop le: {n_body}/{len(samples)} | Mau thieu rep_count: {sum(1 for s in samples if s['rep'] is None)}")
    print("=" * 130)
    wbw = build_weights_by_word(args.bank, args.weights)

    words = [w for w in WORDS_OF_INTEREST if any(s["word"] == w for s in samples)]
    header = f"{'Cau hinh':<32}{'top1':>9}{'%':>7}{'margin':>9}{'d_dung':>9}  " + " ".join(f"{w:>7}" for w in words)
    print(header)
    print("-" * len(header))

    results = []
    configs = list(CONFIGS)
    for label, cfg in configs:
        r = run_loo(samples, cfg, wbw, args.workers)
        results.append(r)
        print(fmt_row(label, r, words), flush=True)

    extra_results = []
    if args.extra:
        print("-" * len(header))
        print("  [Dong tham khao — KHONG dung de quyet dinh]")
        s_noflip = load_bank(args.bank, args.chirality, body_flip_variant=True)
        r = run_loo(s_noflip, {"w_z": 0, "w_b": 0.8, "w_s": 0, "w_r": 0}, wbw, args.workers)
        print(fmt_row("chi moc co the, KHONG lat cot 18", r, words), flush=True)
        extra_results.append(("chi moc co the, KHONG lat cot 18", r))

    base = results[0]
    print("\n" + "=" * 130)
    print("  QUY TAC QUYET DINH: nhom nao KHONG lam tang margin trung binh so voi moc 17D thi DE XUAT LOAI (w = 0.0)")
    print("=" * 130)
    print(f"  Moc 17D: top-1 {100*base['top1']:.1f}% | margin TB {base['margin']:+.4f}")
    for idx, gname in GROUP_OF_CONFIG.items():
        r = results[idx]
        d_margin = r["margin"] - base["margin"]
        d_top1 = r["hits"] - base["hits"]
        verdict = "GIU" if d_margin > 0 else "DE XUAT LOAI"
        print(f"  {gname:<14} margin {r['margin']:+.4f} (delta {d_margin:+.4f}) | top-1 {r['hits']}/{r['n']} (delta {d_top1:+d}) -> {verdict}")
    r_all = results[5]
    print(f"  {'tat ca':<14} margin {r_all['margin']:+.4f} (delta {r_all['margin']-base['margin']:+.4f}) | "
          f"top-1 {r_all['hits']}/{r_all['n']} (delta {r_all['hits']-base['hits']:+d})")

    print("\n  Confusion cua moc 17D (thuc -> doan, so mau):")
    for (a, b), c in sorted(base["confusions"].items(), key=lambda kv: -kv[1])[:12]:
        print(f"     {a:<10} -> {b:<10} x{c}")
    print("\n  Margin TB theo tu (moc 17D | chi z | chi body | chi xoe | chi rep | tat ca):")
    for w in words:
        vals = [r["per_word_margin"].get(w, float("nan")) for r in results]
        print(f"     {w:<10} " + " | ".join(f"{v:+.3f}" for v in vals))


if __name__ == "__main__":
    main()
