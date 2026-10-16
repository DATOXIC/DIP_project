"""
Thí nghiệm "người lạ": thêm mẫu của chính họ (enroll) và/hoặc mẫu của người khác vào bank có giúp nhận đúng hơn không?

Với mỗi người mục tiêu P (mặc định Mai, Phi) và mỗi k (số mẫu tự thêm mỗi từ), lặp R lần chọn ngẫu nhiên k mẫu/từ của P
làm "mẫu enroll"; các mẫu còn lại của P là truy vấn (cùng một tập cho MỌI điều kiện trong lần lặp đó). Các điều kiện:

  dat        : bank của Đạt (webcam_templates.json) — mức hiện tại với người lạ.
  dat+own    : bank Đạt + k mẫu enroll của P.
  own        : CHỈ k mẫu enroll của P ("hồ sơ riêng", như `py dip.py nhan-dien --ho-so P`).
  pool       : bank Đạt + TOÀN BỘ mẫu của người khác (bank gộp), không có mẫu của P.
  pool+own   : bank gộp + k mẫu enroll của P.

Chỉ số trên các truy vấn: top-1, nhận-đúng (ACCEPTED và đúng từ), nhận-sai (ACCEPTED nhưng sai từ), và số nét âm
(negative_strokes.json, 41 nét) bị nhận nhầm ở bank đó.
Giới hạn: mọi mẫu của một người thu gần như cùng buổi (khác từ khác buổi), nên không tách được theo phiên;
enroll và truy vấn cùng buổi có thể lạc quan hơn dùng thật vài ngày sau. Chỉ từ nào P có > k mẫu mới được dùng.

Cách dùng:  py eval_enroll.py --targets Mai Phi --ks 1 2 3 --repeats 5
Kết quả: results/baselines/enroll.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from eval_baselines import _dtw_call, _is_orig, _traj
from scoreboard import build_matcher_from_config, load_bank, load_negative_strokes, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONDS = ["dat", "dat+own", "own", "pool", "pool+own"]


def person_samples(pid: str, vocab_words: List[str], profiles_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(os.path.join(profiles_dir, f"{pid}.json"), "r", encoding="utf-8") as f:
        v = json.load(f).get("vocabulary", {})
    return {w: [s for s in v[w].get("samples", []) if _is_orig(s)] for w in v if w in vocab_words and v[w].get("samples")}


def make_bank(dat: Dict[str, Any], use_dat: bool, extra: List[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    """Bank = (Đạt nếu use_dat) + các nhóm mẫu bổ sung (mỗi nhóm: {từ: [mẫu]})."""
    voc: Dict[str, Dict[str, Any]] = {}
    for w, d in dat["vocabulary"].items():
        base = {k: v for k, v in d.items() if k != "samples"}
        voc[w] = dict(base, samples=list(d["samples"]) if use_dat else [])
    for grp in extra:
        for w, ss in grp.items():
            if w in voc:
                voc[w]["samples"] = voc[w]["samples"] + list(ss)
    voc = {w: dict(d, num_samples=len(d["samples"])) for w, d in voc.items() if d["samples"]}
    return {"metadata": dat.get("metadata", {}), "vocabulary": voc}


def run_queries(matcher, queries: List[Dict[str, Any]]) -> List[Any]:
    out = []
    for q in queries:
        s = q["s"]
        r = _dtw_call(matcher, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                      bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
        out.append((r.candidates[0].word_key if r.candidates else None, r.status))
    return out


def metrics(res, queries, neg_fa: int) -> Dict[str, float]:
    n = len(queries)
    top1 = np.mean([r[0] == q["w"] for r, q in zip(res, queries)])
    acc_ok = np.mean([(r[1] == "ACCEPTED" and r[0] == q["w"]) for r, q in zip(res, queries)])
    acc_bad = np.mean([(r[1] == "ACCEPTED" and r[0] != q["w"]) for r, q in zip(res, queries)])
    return {"n": n, "top1": float(top1), "acc_ok": float(acc_ok), "acc_wrong": float(acc_bad), "neg_fa": float(neg_fa)}


def main() -> None:
    p = argparse.ArgumentParser(description="Enroll / bank gộp cho người lạ.")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--neg", default="negative_strokes.json")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--targets", nargs="+", default=["Mai", "Phi"])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--config", default="configs/G1p90.json")
    p.add_argument("--out", default=os.path.join("results", "baselines", "enroll.json"))
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
        others = [t for t in args.targets if t != tgt]
        other_pool = [people[o] for o in others]
        # Điều kiện không phụ thuộc lần lặp: dat / pool (chỉ cần một matcher, truy vấn thay đổi theo lần lặp)
        m_dat = build_matcher_from_config(cfg, make_bank(dat, True, []))
        m_pool = build_matcher_from_config(cfg, make_bank(dat, True, other_pool))
        fa_dat, fa_pool = neg_false_accepts(m_dat), neg_false_accepts(m_pool)
        all_q = [{"w": w, "s": s, "id": (w, i)} for w, ss in own.items() for i, s in enumerate(ss)]
        cache_dat = dict(zip([q["id"] for q in all_q], run_queries(m_dat, all_q)))
        cache_pool = dict(zip([q["id"] for q in all_q], run_queries(m_pool, all_q)))
        print(f"[{time.time() - t0:5.0f}s] {tgt}: {len(all_q)} mẫu ({len(own)} từ); nét âm nhận nhầm dat={fa_dat} pool={fa_pool}", flush=True)

        results[tgt] = {}
        for k in args.ks:
            rows: Dict[str, List[Dict[str, float]]] = {c: [] for c in CONDS}
            for rep in range(args.repeats):
                rng = np.random.default_rng([20260928, k, rep, sum(map(ord, tgt))])
                enroll: Dict[str, List[Dict[str, Any]]] = {}
                queries: List[Dict[str, Any]] = []
                for w, ss in own.items():
                    if len(ss) <= k:
                        continue
                    pick = set(rng.choice(len(ss), size=k, replace=False).tolist())
                    enroll[w] = [ss[i] for i in sorted(pick)]
                    queries += [{"w": w, "s": s, "id": (w, i)} for i, s in enumerate(ss) if i not in pick]
                if not queries:
                    continue
                rows["dat"].append(metrics([cache_dat[q["id"]] for q in queries], queries, fa_dat))
                rows["pool"].append(metrics([cache_pool[q["id"]] for q in queries], queries, fa_pool))
                for cname, use_dat, extra in (("dat+own", True, [enroll]), ("own", False, [enroll]),
                                              ("pool+own", True, other_pool + [enroll])):
                    m = build_matcher_from_config(cfg, make_bank(dat, use_dat, extra))
                    rows[cname].append(metrics(run_queries(m, queries), queries, neg_false_accepts(m)))
                print(f"[{time.time() - t0:5.0f}s] {tgt} k={k} lặp {rep} xong ({len(queries)} truy vấn)", flush=True)
            results[tgt][k] = rows

    print("\nMỗi ô = trung bình các lần lặp (nhận-đúng ± độ lệch chuẩn). Truy vấn = mẫu của người đó KHÔNG dùng để enroll.")
    for tgt in args.targets:
        for k in args.ks:
            rows = results[tgt][k]
            n = int(np.mean([r["n"] for r in rows["dat"]])) if rows["dat"] else 0
            print(f"\n=== {tgt} — thêm {k} mẫu/từ của chính họ (~{n} truy vấn) ===")
            print(f"{'điều kiện':<11}{'top-1':>8}{'nhận đúng':>16}{'nhận sai':>10}{'nét âm nhầm/41':>16}")
            for c in CONDS:
                rs = rows[c]
                if not rs:
                    continue
                ao = [r["acc_ok"] for r in rs]
                print(f"{c:<11}{100 * np.mean([r['top1'] for r in rs]):>7.1f}%"
                      f"{f'{100 * np.mean(ao):.1f}% ±{100 * np.std(ao):.1f}':>16}"
                      f"{100 * np.mean([r['acc_wrong'] for r in rs]):>9.1f}%{np.mean([r['neg_fa'] for r in rs]):>16.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({str(t): {str(k): v for k, v in d.items()} for t, d in results.items()}, fh, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}  (tổng {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
