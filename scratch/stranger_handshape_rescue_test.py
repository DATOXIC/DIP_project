# -*- coding: utf-8 -*-
"""
Kiem tra DUNG MUC TIEU THAT (theo yeu cau nguoi dung): du an nay bao ve dua tren kha
nang nhan dung NGUOI LA ky (Mai, Phi), khong phai closed-set LOSO (da 100% san roi -
khong con gi de handshape sua o do). Cau hoi that su can tra loi:

  "Voi cac tu ma dang tay TINH da duoc xac nhan la on dinh + phan biet duoc
   (DRINK, PLEASE, THANK_YOU - xem scratch/handshape_signature_audit.py va
   handshape_classifier_loo.py), khi NGUOI LA (Mai/Phi) ky va DTW hien tai
   (bank chi co Dat, KHONG enroll - dieu kien 'dat' trong eval_enroll.py,
   phan anh dung hien trang truoc khi ap dung bat ky ky thuat cai thien nao)
   doan SAI, thi tin hieu dang tay tinh (train tren du lieu cua Dat) co
   doan DUNG duoc khong? Neu co -> fusion co gia tri that cho dung muc tieu
   'nguoi la ky van nhan dien duoc'."

Doi chieu voi phat hien cu (eval_hybrid.py, 2026-09-28): GRU+DTW rescue giup LOSO
nhung KHONG giup nguoi la V01. Script nay lam lai phep do tuong tu nhung voi
handshape (angles tinh, dim 4:9 cua vector 24D) thay vi GRU, tren dung 3 tu da
qua sang loc, de xem co lap lai ket luan "khong giup nguoi la" hay khong.

Chay: python scratch/stranger_handshape_rescue_test.py
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np

from eval_baselines import _dtw_call, _is_orig, _traj
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORDS = ["HELLO", "BYE", "ME", "THANK_YOU", "EAT", "DRINK", "WATER", "SORRY", "PLEASE", "HUNGRY"]
PEOPLE = ["Dang", "Mai", "Phi", "Vuong"]  # Viet loai vi chi co 4 mau (2 tu)
BOUNDARY_FRAC = 0.25
ANGLES_SLICE = slice(4, 9)  # FEATURE_GROUPS: ("angles", 4, 9, ...)


def boundary_signature(traj: np.ndarray) -> np.ndarray:
    """traj: (T=32, D=24). Lay median cua dim 'angles' o 25% dau + 25% cuoi khung."""
    T = traj.shape[0]
    k = max(1, int(round(T * BOUNDARY_FRAC)))
    picked = np.concatenate([traj[:k], traj[-k:]], axis=0) if 2 * k < T else traj
    return np.median(picked[:, ANGLES_SLICE], axis=0)


def build_reference(dat_bank: Dict[str, Any], words: List[str]) -> Dict[str, np.ndarray]:
    """Chu ky dang tay tham chieu cua Dat (mau goc trong webcam_templates.json)."""
    ref: Dict[str, List[np.ndarray]] = {w: [] for w in words}
    for w in words:
        for s in dat_bank["vocabulary"].get(w, {}).get("samples", []):
            if _is_orig(s):
                ref[w].append(boundary_signature(_traj(s)))
    return {w: np.stack(v) for w, v in ref.items() if v}


def knn_predict(sig: np.ndarray, ref: Dict[str, np.ndarray], k: int = 3) -> str:
    dists = []
    for w, sigs in ref.items():
        d = np.linalg.norm(sigs - sig, axis=1)
        dists += [(dd, w) for dd in d]
    dists.sort(key=lambda x: x[0])
    votes: Dict[str, int] = {}
    for _, w in dists[:k]:
        votes[w] = votes.get(w, 0) + 1
    return max(votes, key=lambda w: votes[w])


def person_queries(profile_path: str, words: List[str]) -> List[Dict[str, Any]]:
    import json
    d = json.load(open(profile_path, encoding="utf-8"))
    voc = d.get("vocabulary", {})
    out = []
    for w in words:
        for s in voc.get(w, {}).get("samples", []):
            if _is_orig(s):
                out.append({"w": w, "s": s})
    return out


def main() -> None:
    dat_bank, _ = load_bank("webcam_templates.json")
    cfg, _ = validate_and_hash_config("configs/G1p90.json")
    reference = build_reference(dat_bank, WORDS)
    print("So mau tham chieu (Dat) theo tu:", {w: len(v) for w, v in reference.items()})

    matcher = build_matcher_from_config(cfg, dat_bank)  # dieu kien 'dat': KHONG enroll

    grand_n = 0
    grand_dtw_correct = 0
    grand_rescued = 0
    grand_both_wrong = 0
    grand_broke_gated = 0     # DTW DUNG nhung status != ACCEPTED, handshape noi khac (vung "nguy hiem" neu ap dung veto)
    grand_broke_accepted = 0  # DTW DUNG va status == ACCEPTED, handshape noi khac (KHONG bao gio bi cham vao neu gate dung)
    per_word_wrong: Dict[str, List[str]] = {w: [] for w in WORDS}   # tu -> danh sach ket qua rescue ("rescued"/"both_wrong")
    detail_rows = []

    for person in PEOPLE:
        queries = person_queries(f"profiles/{person}.json", WORDS)
        if not queries:
            print(f"\n[{person}] khong co mau cho cac tu nay, bo qua.")
            continue
        n = len(queries)
        dtw_correct = 0
        shape_correct = 0
        rescued = 0
        both_wrong = 0
        broke_gated = 0
        broke_accepted = 0

        for q in queries:
            traj = _traj(q["s"])
            r = _dtw_call(matcher, traj, q["s"].get("hand_shapes", ["Unknown"] * 32),
                          q["s"].get("handedness"), bool(q["s"].get("chirality_canonical")),
                          q["s"].get("body_anchor_valid", True), q["s"].get("rep_count"))
            dtw_top = r.candidates[0].word_key if r.candidates else None
            sig = boundary_signature(traj)
            shape_top = knn_predict(sig, reference) if q["w"] in reference else None

            is_dtw_ok = (dtw_top == q["w"])
            is_shape_ok = (shape_top == q["w"])
            dtw_correct += int(is_dtw_ok)
            shape_correct += int(is_shape_ok)

            if not is_dtw_ok and is_shape_ok:
                rescued += 1
                per_word_wrong[q["w"]].append("rescued")
            elif not is_dtw_ok and not is_shape_ok:
                both_wrong += 1
                per_word_wrong[q["w"]].append("both_wrong")
            elif is_dtw_ok and shape_top != dtw_top:
                if r.status == "ACCEPTED":
                    broke_accepted += 1
                else:
                    broke_gated += 1

            detail_rows.append((person, q["w"], dtw_top, shape_top, r.status))

        n_dtw_wrong = n - dtw_correct
        print(f"\n=== {person} ({n} truy vấn) — điều kiện 'dat' (KHÔNG enroll) ===")
        print(f"DTW top-1 accuracy      : {dtw_correct}/{n} = {100*dtw_correct/n:.1f}%")
        print(f"Handshape-only accuracy : {shape_correct}/{n} = {100*shape_correct/n:.1f}%")
        print(f"Trong {n_dtw_wrong} ca DTW SAI: cứu được {rescued}, cả hai cùng sai {both_wrong}")
        print(f"Trong {dtw_correct} ca DTW ĐÚNG: handshape nói khác nhưng bị chặn (status!=ACCEPTED, an toàn) "
              f"= {broke_gated}; nói khác NGAY CẢ KHI status=ACCEPTED (rủi ro thật nếu không gate) = {broke_accepted}")

        grand_n += n
        grand_dtw_correct += dtw_correct
        grand_rescued += rescued
        grand_both_wrong += both_wrong
        grand_broke_gated += broke_gated
        grand_broke_accepted += broke_accepted

    print("\n" + "=" * 78)
    print(f"TỔNG HỢP {len(PEOPLE)} người lạ, {len(WORDS)} từ, {grand_n} truy vấn")
    print("=" * 78)
    print(f"DTW top-1 accuracy tổng      : {grand_dtw_correct}/{grand_n} = {100*grand_dtw_correct/grand_n:.1f}%")
    n_wrong_total = grand_n - grand_dtw_correct
    print(f"Trong {n_wrong_total} ca DTW SAI: handshape cứu được {grand_rescued} "
          f"({100*grand_rescued/max(1,n_wrong_total):.1f}%), cả hai cùng sai {grand_both_wrong}")
    print(f"Rủi ro nếu tích hợp KHÔNG gate theo status: {grand_broke_accepted} ca ĐANG ĐÚNG bị nói khác dù ACCEPTED")
    print(f"Nếu CÓ gate (chỉ can thiệp khi status != ACCEPTED): {grand_broke_gated} ca bị nói khác nhưng vô hại "
          f"(không áp dụng vì đã bị chặn)")

    print("\n--- Theo từng từ (số ca DTW sai / rescued / both_wrong) ---")
    for w in WORDS:
        outcomes = per_word_wrong[w]
        if not outcomes:
            print(f"{w:<12} khong co loi DTW nao trong du lieu nguoi la (khong danh gia duoc)")
            continue
        n_r = outcomes.count("rescued")
        n_b = outcomes.count("both_wrong")
        print(f"{w:<12} {len(outcomes)} lỗi DTW -> rescued={n_r}, both_wrong={n_b}")


if __name__ == "__main__":
    main()
