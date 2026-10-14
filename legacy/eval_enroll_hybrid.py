"""
ENROLLMENT + HYBRID GATED CỘNG GỘP: 2 đòn bẩy đã đo RIÊNG (eval_enroll.py cho enrollment,
eval_dynamic_hybrid.py cho DL-gate) — script này đo khi dùng CÙNG LÚC trên cùng 1 người lạ, để
biết chúng CỘNG HƯỞNG (mỗi cái sửa lỗi khác nhau) hay TRÙNG LẶP (enrollment đã che hết lỗi mà
DL-gate lẽ ra sửa được).

Điều kiện so sánh, mô phỏng đúng luồng production (session_enroll.py): thêm k mẫu/từ CỦA CHÍNH
người lạ vào bank Đạt (điều kiện "dat+own" — production KHÔNG gộp pool người khác), model DL
KHÔNG train lại (checkpoint LOSO tĩnh, đúng thực tế "model không tự học lại giữa phiên"):
  dtw_only   : DTW trên bank Đạt+k mẫu enroll (không có DL).
  hybrid     : DTW trên bank Đạt+k mẫu enroll, CỘNG THÊM DL-gate (dl_fusion, ngưỡng DEFAULT).

Cách dùng:  py eval_enroll_hybrid.py --targets Dang Mai Phi Vuong --ks 1 2 3 --repeats 5
Kết quả: results/baselines/enroll_hybrid.json
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

import dl_fusion
from eval_baselines import _dtw_call, _is_orig, _traj
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def person_samples(pid: str, vocab_words: List[str], profiles_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(os.path.join(profiles_dir, f"{pid}.json"), "r", encoding="utf-8") as f:
        v = json.load(f).get("vocabulary", {})
    return {w: [s for s in v[w].get("samples", []) if _is_orig(s)] for w in v if w in vocab_words and v[w].get("samples")}


def make_bank(dat: Dict[str, Any], enroll: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    voc: Dict[str, Any] = {}
    for w, d in dat["vocabulary"].items():
        base = {k: v for k, v in d.items() if k != "samples"}
        voc[w] = dict(base, samples=list(d["samples"]))
    for w, ss in enroll.items():
        if w in voc:
            voc[w]["samples"] = voc[w]["samples"] + list(ss)
    voc = {w: dict(d, num_samples=len(d["samples"])) for w, d in voc.items() if d["samples"]}
    return {"metadata": dat.get("metadata", {}), "vocabulary": voc}


def run_queries(matcher, queries: List[Dict[str, Any]], bundle) -> List[Dict[str, Any]]:
    """Trả list {'dtw_only_correct', 'hybrid_correct'} cho từng truy vấn."""
    out = []
    for q in queries:
        s = q["s"]
        r = _dtw_call(matcher, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                      bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
        dtw_word = r.candidates[0].word_key if r.candidates else None
        dtw_only_correct = (dtw_word == q["w"])

        dl_word, dl_conf = None, 0.0
        if bundle is not None:
            traj24 = _traj(s)
            pred = dl_fusion.predict(bundle, traj24)
            if pred is not None:
                dl_word, dl_conf = pred[0], pred[1]
        hybrid_word = dl_fusion.gated_prediction(r.status, dtw_word, dl_word, dl_conf, dl_fusion.DEFAULT_CONF_THRESHOLD)
        out.append({"dtw_only_correct": dtw_only_correct, "hybrid_correct": hybrid_word == q["w"]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Enrollment + Hybrid GATED cộng gộp trên người lạ.")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--profiles-dir", default="profiles")
    p.add_argument("--targets", nargs="+", default=["Dang", "Mai", "Phi", "Vuong"])
    p.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--config", default="configs/G1p90.json")
    p.add_argument("--out", default=os.path.join("results", "baselines", "enroll_hybrid.json"))
    args = p.parse_args()

    dat, _ = load_bank(args.bank)
    cfg, _ = validate_and_hash_config(args.config)
    words = list(dat["vocabulary"].keys())
    t0 = time.time()

    results: Dict[str, Any] = {}
    for tgt in args.targets:
        bundle = dl_fusion.load_model(f"best_gesture_model_{tgt}.pth")
        if bundle is None:
            print(f"[!] Thiếu checkpoint LOSO cho {tgt}, bỏ qua.")
            continue
        own = person_samples(tgt, words, args.profiles_dir)
        results[tgt] = {}
        for k in args.ks:
            dtw_accs, hybrid_accs = [], []
            for rep in range(args.repeats):
                rng = np.random.default_rng([20260930, k, rep, sum(map(ord, tgt))])
                enroll: Dict[str, List[Dict[str, Any]]] = {}
                queries: List[Dict[str, Any]] = []
                for w, ss in own.items():
                    if k > 0 and len(ss) <= k:
                        continue
                    if k > 0:
                        pick = set(rng.choice(len(ss), size=k, replace=False).tolist())
                        enroll[w] = [ss[i] for i in sorted(pick)]
                    else:
                        pick = set()
                    queries += [{"w": w, "s": s} for i, s in enumerate(ss) if i not in pick]
                if not queries:
                    continue
                matcher = build_matcher_from_config(cfg, make_bank(dat, enroll))
                rows = run_queries(matcher, queries, bundle)
                dtw_accs.append(np.mean([r["dtw_only_correct"] for r in rows]))
                hybrid_accs.append(np.mean([r["hybrid_correct"] for r in rows]))
            results[tgt][str(k)] = {
                "dtw_only_mean": float(np.mean(dtw_accs)), "dtw_only_sd": float(np.std(dtw_accs)),
                "hybrid_mean": float(np.mean(hybrid_accs)), "hybrid_sd": float(np.std(hybrid_accs)),
                "n_queries_approx": int(np.mean([len(queries)] * 1)),
            }
            print(f"[{time.time()-t0:5.0f}s] {tgt} k={k}: dtw_only={100*np.mean(dtw_accs):5.1f}%  "
                  f"hybrid={100*np.mean(hybrid_accs):5.1f}%  (delta={100*(np.mean(hybrid_accs)-np.mean(dtw_accs)):+.1f}pt)",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nĐã lưu: {args.out}  (tổng {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
