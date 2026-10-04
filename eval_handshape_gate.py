# -*- coding: utf-8 -*-
"""
BẰNG CHỨNG ĐO: bộ lọc dáng tay có làm pipeline chính xác hơn không?

Thiết lập (điều kiện thật của dự án: NGƯỜI LẠ ký, bank chỉ có chủ dự án, không enroll):
  - DTW: GoldenTriadDTWMatcher với bank của chủ dự án (webcam_templates.json), cấu hình configs/G1p90.json.
  - Truy vấn: mẫu gốc của người lạ trong profiles/<người>.json (Dang, Mai, Phi, Vuong).
  - Bằng chứng dáng tay: chạy nhánh landmark trên landmark từng khung của chính cử chỉ đó (raw_volunteers/),
    ghép với mẫu theo (từ, phiên, thời lượng). Mẫu không ghép được bị bỏ (in rõ độ phủ).
  - Bảng từ -> nhóm dáng tay: HỌC CHỈ từ bản ghi của chủ dự án (raw_strokes/) => không rò rỉ sang người lạ.

So sánh 3 hệ:
  baseline     : DTW thuần.
  gate         : DTW + bộ lọc (chỉ can thiệp khi status != ACCEPTED)   <- đề xuất.
  gate_hard    : ràng buộc cứng, can thiệp cả khi ACCEPTED             <- ý tưởng "chỉ được nhầm cùng chữ cái", để ĐO RỦI RO.

Chỉ số: top-1 đúng; "chấp nhận đúng" (ACCEPTED & đúng); "chấp nhận SAI" (ACCEPTED & sai — lỗi nguy hiểm nhất, người
dùng không biết để thử lại); số ca sửa được (sai -> đúng) và số ca làm hỏng (đúng -> sai).

Chạy:  py eval_handshape_gate.py [--model letter_landmark_model_image.pt] [--sweep]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eval_baselines import _dtw_call, _is_orig, _traj
from handshape_gate import (SHAPE_GROUPS, apply_shape_gate, boundary_group_evidence, group_names, learn_word_table)
from letter_landmark_classifier import LetterLandmarkClassifier
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORDS = ["BAD", "BYE", "DRINK", "EAT", "HELLO", "HUNGRY", "ME", "PLEASE", "SORRY", "THANK_YOU", "WATER", "YES", "YOU"]
PEOPLE = ["Dang", "Mai", "Phi", "Vuong"]


def stroke_frame_probs(clf: LetterLandmarkClassifier, stroke: Dict[str, Any]) -> np.ndarray:
    """(T, C) xác suất lớp cho từng khung của bản ghi thô (landmarks_px, nhãn tay từng khung)."""
    from hand_pose_features import landmarks_to_features
    fallback = stroke.get("dominant_handedness")
    feats = []
    for fr in stroke["frames"]:
        lm = np.asarray(fr["landmarks_px"], dtype=np.float32)
        feats.append(landmarks_to_features(lm, fr.get("handedness") or fallback))
    return clf.predict_features(np.stack(feats))


def owner_evidence(clf: LetterLandmarkClassifier) -> Dict[str, List[Dict[str, float]]]:
    out: Dict[str, List[Dict[str, float]]] = collections.defaultdict(list)
    for f in glob.glob("raw_strokes/*/*.json"):
        d = json.load(open(f, encoding="utf-8"))
        w = d["word_key"]
        if w not in WORDS or len(d["frames"]) < 6:
            continue
        out[w].append(boundary_group_evidence(stroke_frame_probs(clf, d), clf.classes))
    return out


def index_volunteer_raw() -> Dict[Tuple[str, str], List[Tuple[float, str]]]:
    idx: Dict[Tuple[str, str], List[Tuple[float, str]]] = collections.defaultdict(list)
    for f in glob.glob("raw_volunteers/**/*.json", recursive=True):
        d = json.load(open(f, encoding="utf-8"))
        idx[(d["word_key"], d["session_id"])].append((float(d["stroke_duration_sec"]), f))
    return idx


def build_queries(raw_idx) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[int, int]]]:
    queries, coverage = [], {}
    for person in PEOPLE:
        voc = json.load(open(f"profiles/{person}.json", encoding="utf-8")).get("vocabulary", {})
        tot = matched = 0
        for w in WORDS:
            for s in voc.get(w, {}).get("samples", []):
                if not _is_orig(s):
                    continue
                tot += 1
                cands = [f for dur, f in raw_idx.get((w, s.get("session_id")), []) if abs(dur - s["duration_sec"]) < 0.002]
                if cands:
                    matched += 1
                    queries.append({"person": person, "word": w, "sample": s, "raw": cands[0]})
        coverage[person] = (matched, tot)
    return queries, coverage


def outcome(res: Any, truth: str) -> Tuple[bool, bool]:
    top = res.candidates[0].word_key if res.candidates else None
    return top == truth, res.status == "ACCEPTED"


def summarize(name: str, rows: List[Dict[str, Any]], base_rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    acc_ok = sum(r["ok"] and r["acc"] for r in rows)
    acc_bad = sum((not r["ok"]) and r["acc"] for r in rows)
    out = {"n": n, "top1": ok, "accepted_correct": acc_ok, "accepted_wrong": acc_bad,
           "not_accepted": sum(not r["acc"] for r in rows)}
    if base_rows is not None:
        out["fixed"] = sum((not b["ok"]) and r["ok"] for b, r in zip(base_rows, rows))
        out["broken"] = sum(b["ok"] and (not r["ok"]) for b, r in zip(base_rows, rows))
        out["acc_wrong_delta"] = acc_bad - sum((not b["ok"]) and b["acc"] for b in base_rows)
    return out


def print_table(title: str, results: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(f"{'hệ':<12}{'n':>5}{'top-1':>14}{'ACCEPTED đúng':>16}{'ACCEPTED SAI':>14}{'không chấp nhận':>17}{'sửa được':>10}{'làm hỏng':>10}")
    for k, v in results.items():
        n = v["n"]
        print(f"{k:<12}{n:>5}{v['top1']:>7} ({100 * v['top1'] / n:4.1f}%){v['accepted_correct']:>10}{v['accepted_wrong']:>14}"
              f"{v['not_accepted']:>17}{v.get('fixed', '-'):>10}{v.get('broken', '-'):>10}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="letter_landmark_model_image.pt")
    ap.add_argument("--config", default="configs/G1p90.json")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--compat", type=float, default=0.15)
    ap.add_argument("--reliable-top", type=float, default=0.6, help="nhóm trội phải chiếm >= tỉ lệ này thì từ mới bị ràng buộc")
    ap.add_argument("--no-strict-promote", action="store_true", help="đối chứng: nâng ACCEPTED dù còn từ tự do trong top-K")
    ap.add_argument("--sweep", action="store_true", help="quét ngưỡng conf/compat để xem độ nhạy")
    ap.add_argument("--out", default="results/handshape_gate_eval.json")
    a = ap.parse_args()

    clf = LetterLandmarkClassifier(a.model, min_conf=0.0)
    if clf.space != "image":
        print(f"[!] Model '{a.model}' học landmark '{clf.space}' nhưng bản ghi cũ chỉ có landmark ảnh: kết quả sẽ kém.")
    print("Nhóm dáng tay:", {g: [c for c in clf.classes if (SHAPE_GROUPS.get(c, c) == g)] for g in group_names(clf.classes)})

    ev = owner_evidence(clf)
    table = learn_word_table(ev, reliable_top=a.reliable_top)
    print("\nBảng từ -> nhóm dáng tay (học từ chủ dự án):")
    for w in WORDS:
        ws = table.get(w)
        if ws is None:
            continue
        prof = ", ".join(f"{g}={m * 100:.0f}%" for g, m in sorted(ws.profile.items(), key=lambda kv: -kv[1])[:3])
        print(f"  {w:<10} {'RÀNG BUỘC ' if ws.reliable else 'tự do     '} cho phép={sorted(ws.allowed)}  ({prof}; {len(ev[w])} mẫu)")

    dat_bank, _ = load_bank("webcam_templates.json")
    cfg, _ = validate_and_hash_config(a.config)
    matcher = build_matcher_from_config(cfg, dat_bank)

    queries, coverage = build_queries(index_volunteer_raw())
    print("\nĐộ phủ ghép bản ghi thô (có bằng chứng dáng tay / tổng mẫu gốc):",
          {p: f"{m}/{t}" for p, (m, t) in coverage.items()})

    dtw_res, evidences = [], []
    for q in queries:
        s = q["sample"]
        r = _dtw_call(matcher, _traj(s), s.get("hand_shapes", ["Unknown"] * 32), s.get("handedness"),
                      bool(s.get("chirality_canonical")), s.get("body_anchor_valid", True), s.get("rep_count"))
        dtw_res.append(r)
        raw = json.load(open(q["raw"], encoding="utf-8"))
        evidences.append(boundary_group_evidence(stroke_frame_probs(clf, raw), clf.classes))

    def run_variant(conf: float, comp: float, hard: bool) -> List[Dict[str, Any]]:
        rows = []
        for q, r, e in zip(queries, dtw_res, evidences):
            r2 = apply_shape_gate(r, e, table, conf_thresh=conf, compat_thresh=comp, gate_accepted=hard, strict_promote=not a.no_strict_promote)
            ok, acc = outcome(r2, q["word"])
            rows.append({"person": q["person"], "word": q["word"], "ok": ok, "acc": acc, "note": getattr(r2, "fusion_note", ""),
                         "dtw_top": r.candidates[0].word_key if r.candidates else None, "new_top": r2.top_word})
        return rows

    base = run_variant(a.conf, a.compat, False)  # placeholder, replaced below
    base = []
    for q, r in zip(queries, dtw_res):
        ok, acc = outcome(r, q["word"])
        base.append({"person": q["person"], "word": q["word"], "ok": ok, "acc": acc, "note": "", "dtw_top": r.candidates[0].word_key if r.candidates else None,
                     "new_top": r.top_word})
    gate = run_variant(a.conf, a.compat, False)
    hard = run_variant(a.conf, a.compat, True)

    pairs = collections.Counter((q["word"], b["dtw_top"]) for q, b in zip(queries, base) if not b["ok"])
    print("\nCặp (đúng, DTW đoán) mà DTW thuần nhầm:", dict(pairs.most_common()))
    res_all = {"baseline": summarize("baseline", base, None), "gate": summarize("gate", gate, base),
               "gate_hard": summarize("gate_hard", hard, base)}
    print_table(f"TỔNG ({len(queries)} truy vấn người lạ, conf>={a.conf}, compat<{a.compat})", res_all)
    for p in PEOPLE:
        idx = [i for i, q in enumerate(queries) if q["person"] == p]
        if not idx:
            continue
        sub = lambda rows: [rows[i] for i in idx]
        print_table(f"  {p}", {"baseline": summarize("b", sub(base), None), "gate": summarize("g", sub(gate), sub(base)),
                               "gate_hard": summarize("h", sub(hard), sub(base))})

    print("\nChi tiết các ca bộ lọc THAY ĐỔI kết quả (gate / gate_hard):")
    for name, rows in (("gate", gate), ("gate_hard", hard)):
        for q, b, r in zip(queries, base, rows):
            if b["new_top"] != r["new_top"] or b["acc"] != r["acc"]:
                tag = "SỬA" if (not b["ok"] and r["ok"]) else ("HỎNG" if (b["ok"] and not r["ok"]) else "đổi")
                print(f"  [{name}] {q['person']:<6} đúng={q['word']:<10} DTW={b['dtw_top']}({'ACC' if b['acc'] else 'không'}) -> {r['new_top']}({'ACC' if r['acc'] else 'không'}) {tag}  {r['note']}")

    if a.sweep:
        print("\nĐộ nhạy ngưỡng (chế độ gate đề xuất): conf x compat -> sửa/hỏng/ACCEPTED-sai")
        for conf in (0.5, 0.6, 0.7, 0.8):
            line = []
            for comp in (0.05, 0.10, 0.15, 0.25):
                s = summarize("x", run_variant(conf, comp, False), base)
                line.append(f"{comp:.2f}:+{s['fixed']}/-{s['broken']}/{s['accepted_wrong']}")
            print(f"  conf={conf:.1f}  " + "   ".join(line))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"coverage": coverage, "summary": res_all,
               "table": {w: {"profile": v.profile, "allowed": sorted(v.allowed), "reliable": v.reliable} for w, v in table.items()}},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
