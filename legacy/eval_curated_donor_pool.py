# -*- coding: utf-8 -*-
"""
Thí nghiệm "CHỌN LỌC" mẫu tình nguyện viên rồi thêm vào bank Đạt — khác với eval_enroll.py
(điều kiện "pool": gộp TOÀN BỘ mẫu người khác, đã đo là CÓ HẠI, ví dụ Phi 58%->47%).

Câu hỏi: gộp thô thì hại, nhưng nếu CHỌN LỌC kỹ (không nhét hết) thì có khác không? Đây là
thí nghiệm CHƯA từng đo — kết quả có thể theo bất kỳ hướng nào, không giả định trước.

Với mỗi người mục tiêu T (một "người lạ" thật, giữ nguyên toàn bộ mẫu của họ làm truy vấn,
KHÔNG enroll bất kỳ mẫu nào của T — đây là phép thử "người lạ" nghiêm ngặt nhất), nguồn mẫu
"tình nguyện viên khác" (donors) = tất cả người trong --targets trừ T. Với mỗi từ, chấm điểm
"độ tương thích" của từng mẫu donor bằng khoảng cách DTW của chính mẫu đó tới CỤM MẪU CỦA ĐẠT
cho từ đó (dùng matcher thật, cùng trọng số/threshold đang chạy production) — không phải máy
móc chọn ngẫu nhiên.

Các điều kiện so sánh (n mẫu donor được thêm mỗi từ, mặc định n=2):
  dat                : bank Đạt (webcam_templates.json) — mức hiện tại, không có gì thêm.
  pool_full          : bank Đạt + TOÀN BỘ mẫu donor (mốc đối chứng, tái lập kết quả có hại đã biết).
  random_n           : bank Đạt + n mẫu donor NGẪU NHIÊN mỗi từ (lặp lại nhiều lần lấy trung bình) —
                        đối chứng để tách "số lượng ít hơn" khỏi "chọn lọc có mục đích".
  curated_close_n    : bank Đạt + n mẫu donor GẦN cụm Đạt nhất mỗi từ (giống phong cách Đạt nhất).
  curated_diverse_n  : bank Đạt + n mẫu donor XA cụm Đạt nhất mỗi từ (phong cách khác biệt nhất,
                        nhưng vẫn đúng từ theo nhãn gốc của donor).
  curated_weak_words : như curated_close_n nhưng CHỈ áp dụng cho các từ đang yếu nhất (accuracy
                        thấp nhất ở điều kiện "dat"), các từ còn lại giữ nguyên bank Đạt thuần.

Đo trên MỖI điều kiện: top-1, nhận-đúng (ACCEPTED đúng từ), nhận-sai (ACCEPTED sai từ), và số
nét âm (negative_strokes.json) bị nhận nhầm — để biết chọn lọc có làm bank "dễ nhận nhầm" hơn không.

Cách dùng:  py eval_curated_donor_pool.py --targets Dang Mai Phi Vuong --n 2 --repeats 5
Kết quả: results/baselines/curated_donor_pool.json
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from eval_baselines import _dtw_call, _is_orig, _traj
from eval_enroll import make_bank, metrics, person_samples, run_queries
from scoreboard import build_matcher_from_config, load_bank, load_negative_strokes, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

FIXED_CONDS = ["dat", "pool_full", "curated_close_n", "curated_diverse_n", "curated_weak_words"]


def word_distance(matcher, sample: Dict[str, Any], word_key: str) -> float:
    """Khoảng cách DTW của MỘT mẫu tới đúng từ `word_key` trong bank hiện tại của `matcher`
    (dùng để chấm "độ tương thích phong cách" của mẫu donor với cụm mẫu Đạt đang có)."""
    r = _dtw_call(matcher, _traj(sample), sample.get("hand_shapes", ["Unknown"] * 32), sample.get("handedness"),
                  bool(sample.get("chirality_canonical")), sample.get("body_anchor_valid", True), sample.get("rep_count"))
    for c in r.candidates:
        if c.word_key == word_key:
            return float(c.distance)
    return float("inf")


def rank_donors_by_compat(m_dat, donor_by_word: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Tuple[float, Dict[str, Any]]]]:
    """Với mỗi từ, trả về danh sách (khoảng_cách_tới_cụm_Đạt, mẫu) đã SẮP XẾP TĂNG DẦN
    (đầu danh sách = giống phong cách Đạt nhất, cuối danh sách = khác biệt nhất)."""
    ranked: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
    for w, samples in donor_by_word.items():
        scored = [(word_distance(m_dat, s, w), s) for s in samples]
        scored.sort(key=lambda t: t[0])
        ranked[w] = scored
    return ranked


def pick_n(ranked: Dict[str, List[Tuple[float, Dict[str, Any]]]], n: int, mode: str,
           words_filter: Any = None) -> Dict[str, List[Dict[str, Any]]]:
    """mode='close' -> n mẫu đầu (gần Đạt nhất) | mode='diverse' -> n mẫu cuối (xa Đạt nhất).
    words_filter: nếu truyền, CHỈ chọn cho các từ trong tập này (từ còn lại bị bỏ qua hoàn toàn)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for w, scored in ranked.items():
        if words_filter is not None and w not in words_filter:
            continue
        if not scored:
            continue
        chosen = scored[:n] if mode == "close" else scored[-n:]
        out[w] = [s for _, s in chosen]
    return out


def pick_random_n(donor_by_word: Dict[str, List[Dict[str, Any]]], n: int, rng: np.random.Generator) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for w, samples in donor_by_word.items():
        if not samples:
            continue
        k = min(n, len(samples))
        idx = rng.choice(len(samples), size=k, replace=False)
        out[w] = [samples[i] for i in idx]
    return out


def per_word_accuracy(res: List[Any], queries: List[Dict[str, Any]]) -> Dict[str, float]:
    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for r, q in zip(res, queries):
        total[q["w"]] += 1
        if r[1] == "ACCEPTED" and r[0] == q["w"]:
            correct[q["w"]] += 1
    return {w: correct[w] / total[w] for w in total}


def merge_donor_pools(pools: List[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pool in pools:
        for w, ss in pool.items():
            merged[w].extend(ss)
    return dict(merged)


def main() -> None:
    p = argparse.ArgumentParser(description="Chọn lọc mẫu tình nguyện viên thêm vào bank Đạt (thay vì gộp thô).")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--neg", default="negative_strokes.json")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--targets", nargs="+", default=["Dang", "Mai", "Phi", "Vuong"])
    p.add_argument("--n", type=int, default=2, help="Số mẫu donor thêm mỗi từ ở các điều kiện curated/random.")
    p.add_argument("--weak-k", type=int, default=5, help="Số từ 'yếu nhất' (accuracy thấp nhất ở dat) được curated_weak_words nhắm tới.")
    p.add_argument("--repeats", type=int, default=5, help="Số lần lặp cho điều kiện random_n (lấy trung bình).")
    p.add_argument("--config", default="configs/G1p90.json")
    p.add_argument("--out", default=os.path.join("results", "baselines", "curated_donor_pool.json"))
    args = p.parse_args()

    dat, _ = load_bank(args.bank)
    neg, _ = load_negative_strokes(args.neg)
    cfg, _ = validate_and_hash_config(args.config)
    words = list(dat["vocabulary"].keys())
    people = {t: person_samples(t, words, args.profiles_dir) for t in args.targets}
    t0 = time.time()

    def neg_false_accepts(matcher) -> int:
        return sum(1 for r in run_queries(matcher, [{"s": s, "w": None} for s in neg]) if r[1] == "ACCEPTED")

    results: Dict[str, Any] = {}
    for tgt in args.targets:
        own = people[tgt]
        if not own:
            print(f"[!] {tgt}: không có mẫu nào trong profiles/{tgt}.json khớp từ vựng của bank -> bỏ qua.")
            continue
        donors = [t for t in args.targets if t != tgt]
        donor_pool = merge_donor_pools([people[d] for d in donors])
        if not donor_pool:
            print(f"[!] {tgt}: không có donor nào khác trong --targets -> bỏ qua.")
            continue

        queries = [{"w": w, "s": s, "id": (w, i)} for w, ss in own.items() for i, s in enumerate(ss)]

        m_dat = build_matcher_from_config(cfg, make_bank(dat, True, []))
        fa_dat = neg_false_accepts(m_dat)
        res_dat = run_queries(m_dat, queries)
        met_dat = metrics(res_dat, queries, fa_dat)
        acc_by_word_dat = per_word_accuracy(res_dat, queries)
        print(f"[{time.time() - t0:5.0f}s] {tgt}: {len(queries)} truy vấn ({len(own)} từ), "
              f"{sum(len(v) for v in donor_pool.values())} mẫu donor từ {donors}; "
              f"dat top1={100*met_dat['top1']:.1f}% nét âm nhầm={fa_dat}", flush=True)

        m_pool_full = build_matcher_from_config(cfg, make_bank(dat, True, [donor_pool]))
        fa_pool = neg_false_accepts(m_pool_full)
        met_pool_full = metrics(run_queries(m_pool_full, queries), queries, fa_pool)

        ranked = rank_donors_by_compat(m_dat, donor_pool)

        curated_close = pick_n(ranked, args.n, "close")
        m_close = build_matcher_from_config(cfg, make_bank(dat, True, [curated_close]))
        fa_close = neg_false_accepts(m_close)
        met_close = metrics(run_queries(m_close, queries), queries, fa_close)

        curated_diverse = pick_n(ranked, args.n, "diverse")
        m_diverse = build_matcher_from_config(cfg, make_bank(dat, True, [curated_diverse]))
        fa_diverse = neg_false_accepts(m_diverse)
        met_diverse = metrics(run_queries(m_diverse, queries), queries, fa_diverse)

        weak_words = {w for w, _ in sorted(acc_by_word_dat.items(), key=lambda kv: kv[1])[:args.weak_k]}
        curated_weak = pick_n(ranked, args.n, "close", words_filter=weak_words)
        m_weak = build_matcher_from_config(cfg, make_bank(dat, True, [curated_weak]))
        fa_weak = neg_false_accepts(m_weak)
        met_weak = metrics(run_queries(m_weak, queries), queries, fa_weak)

        random_rows: List[Dict[str, float]] = []
        for rep in range(args.repeats):
            rng = np.random.default_rng([20260930, rep, sum(map(ord, tgt))])
            random_pick = pick_random_n(donor_pool, args.n, rng)
            m_rand = build_matcher_from_config(cfg, make_bank(dat, True, [random_pick]))
            fa_rand = neg_false_accepts(m_rand)
            random_rows.append(metrics(run_queries(m_rand, queries), queries, fa_rand))

        results[tgt] = {
            "n_queries": len(queries),
            "n_words": len(own),
            "donors": donors,
            "weak_words": sorted(weak_words),
            "dat": met_dat,
            "pool_full": met_pool_full,
            "curated_close_n": met_close,
            "curated_diverse_n": met_diverse,
            "curated_weak_words": met_weak,
            "random_n": random_rows,
        }
        print(f"[{time.time() - t0:5.0f}s] {tgt}: xong tất cả điều kiện.", flush=True)

    print("\n" + "=" * 90)
    print(f"Mỗi từ thêm tối đa {args.n} mẫu donor (trừ 'dat' và 'pool_full'). weak_words = {args.weak_k} từ yếu nhất ở 'dat'.")
    header = f"{'điều kiện':<20}{'top-1':>8}{'nhận đúng':>12}{'nhận sai':>10}{'nét âm nhầm/41':>16}"
    for tgt, r in results.items():
        print(f"\n=== {tgt} (donors: {', '.join(r['donors'])}; từ yếu: {', '.join(r['weak_words'])}) ===")
        print(header)
        for cname in ("dat", "pool_full", "curated_close_n", "curated_diverse_n", "curated_weak_words"):
            m = r[cname]
            print(f"{cname:<20}{100*m['top1']:>7.1f}%{100*m['acc_ok']:>11.1f}%{100*m['acc_wrong']:>9.1f}%{m['neg_fa']:>16.1f}")
        rr = r["random_n"]
        ao = [m["acc_ok"] for m in rr]
        print(f"{'random_n':<20}{100*np.mean([m['top1'] for m in rr]):>7.1f}%"
              f"{f'{100*np.mean(ao):.1f}%±{100*np.std(ao):.1f}':>11}"
              f"{100*np.mean([m['acc_wrong'] for m in rr]):>9.1f}%{np.mean([m['neg_fa'] for m in rr]):>16.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}  (tổng {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
