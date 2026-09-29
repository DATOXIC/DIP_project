"""
Đo nhanh: mẫu ASL Citizen (24D, đã cắt) so với bank Đạt bằng pipeline hiện tại (cấu hình G1p90). Chỉ để ĐO, không chỉnh gì.
Từ trùng với bank Đạt -> nhận đúng / nhận sai / từ chối; từ ngoài bank -> tỉ lệ bị nhận nhầm (ACCEPTED).
Cách dùng:  py eval_citizen.py     Kết quả: results/baselines/citizen.json
"""
import json
import os
import sys
from collections import Counter

from eval_baselines import _is_orig
from eval_enroll import run_queries
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    dat, _ = load_bank("webcam_templates.json")
    cfg, _ = validate_and_hash_config("configs/G1p90.json")
    with open("asl_citizen_templates_24d.json", "r", encoding="utf-8") as f:
        cit = json.load(f)["vocabulary"]
    m = build_matcher_from_config(cfg, dat)
    inv, oov = [], []
    for w, d in cit.items():
        for s in d["samples"]:
            if _is_orig(s):
                (inv if w in dat["vocabulary"] else oov).append({"w": w, "s": s})
    ri, ro = run_queries(m, inv), run_queries(m, oov)
    n = len(inv)
    ok = sum(r[1] == "ACCEPTED" and r[0] == q["w"] for r, q in zip(ri, inv))
    bad = sum(r[1] == "ACCEPTED" and r[0] != q["w"] for r, q in zip(ri, inv))
    top1 = sum(r[0] == q["w"] for r, q in zip(ri, inv))
    unc = sum(r[1] == "UNCERTAIN" for r in ri)
    print(f"Từ trùng bank Đạt ({sorted({q['w'] for q in inv})}): {n} mẫu")
    print(f"  top-1 {100*top1/n:.1f}% | nhận đúng {100*ok/n:.1f}% | nhận sai {100*bad/n:.1f}% | không chắc {100*unc/n:.1f}% | từ chối {100*(n-ok-bad-unc)/n:.1f}%")
    print("  theo từ (top-1 / nhận đúng):")
    per = {}
    for r, q in zip(ri, inv):
        c = per.setdefault(q["w"], [0, 0, 0])
        c[0] += 1; c[1] += r[0] == q["w"]; c[2] += r[1] == "ACCEPTED" and r[0] == q["w"]
    for w, c in sorted(per.items()):
        print(f"    {w:<10}{c[1]}/{c[0]}  {c[2]}/{c[0]}")
    fa = sum(r[1] == "ACCEPTED" for r in ro)
    print(f"Từ ngoài bank ({len(set(q['w'] for q in oov))} từ): {len(oov)} mẫu, nhận nhầm {fa} ({100*fa/max(1,len(oov)):.1f}%)")
    print("  nhận nhầm thành:", dict(Counter(r[0] for r in ro if r[1] == 'ACCEPTED')))
    os.makedirs("results/baselines", exist_ok=True)
    with open("results/baselines/citizen.json", "w", encoding="utf-8") as f:
        json.dump({"n_in": n, "top1": top1 / n, "acc_ok": ok / n, "acc_wrong": bad / n, "uncertain": unc / n,
                   "n_oov": len(oov), "oov_false_accept": fa / max(1, len(oov)), "per_word": per}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
