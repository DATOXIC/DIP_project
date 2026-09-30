# -*- coding: utf-8 -*-
"""
FUSION DÁNG TAY TĨNH cho DTW động — PHẠM VI HẸP, CÓ CHỦ ĐÍCH: chỉ DRINK/WATER.

Nguồn gốc: điều tra thực nghiệm (scratch/handshape_signature_audit.py,
scratch/handshape_classifier_loo.py, scratch/stranger_handshape_rescue_test.py) trên
13 từ trong webcam_templates.json, đo trên lỗi THẬT của người lạ (Dang/Mai/Phi/Vuong,
242 truy vấn, điều kiện KHÔNG enroll). Kết luận:
  - Chỉ DRINK và WATER có lợi ích thật đo được (11/13 lỗi DTW được dáng tay cứu đúng
    = 84.6%). Các từ khác (PLEASE, HUNGRY, ...) KHÔNG được cứu (0%) hoặc mẫu quá nhỏ
    để kết luận — KHÔNG đưa vào đây, tránh áp dụng tràn lan không có bằng chứng.
  - Bộ phân loại dáng tay (train trên mẫu gốc của Đạt) có SAI LỆCH HỆ THỐNG với người
    lạ ở một số từ khác (không phải DRINK/WATER) — nên TUYỆT ĐỐI không áp dụng khi
    DTW đang ACCEPTED (tự tin): chỉ được đọc là "ứng viên phụ" khi DTW tự báo không
    chắc (UNCERTAIN/REJECTED), không bao giờ ghi đè một quyết định DTW đang tự tin.

Đặc trưng dùng: cột "angles" (chỉ số 4:9 trong FEATURE_GROUPS của dynamic_features.py)
của vector 24D đã resample K=32 khung — góc cong từng ngón tay, không phụ thuộc màu da/
ánh sáng (khác CNN CIELab cũ đã bị tắt). Chữ ký = trung vị cột "angles" ở 25% khung ĐẦU
+ 25% khung CUỐI (khung biên, lúc dáng tay đã "chốt" theo protocol thu mẫu
settle-then-sign) — đã đo là nhỉnh hơn dùng toàn bộ khung.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FUSION_WORDS: Tuple[str, ...] = ("DRINK", "WATER")
ANGLES_SLICE = slice(4, 9)   # FEATURE_GROUPS: ("angles", 4, 9, "w_a", "euclid", SQRT5)
BOUNDARY_FRAC = 0.25
KNN_K = 3


def _is_orig(s: Dict[str, Any]) -> bool:
    return s.get("is_original", True) is not False and not s.get("augment_of")


def boundary_signature(traj_24d: np.ndarray) -> np.ndarray:
    """Chữ ký dáng tay (5D) = trung vị cột 'angles' ở 25% khung đầu + 25% khung cuối."""
    traj = np.asarray(traj_24d, dtype=np.float32)
    T = traj.shape[0]
    k = max(1, int(round(T * BOUNDARY_FRAC)))
    picked = np.concatenate([traj[:k], traj[-k:]], axis=0) if 2 * k < T else traj
    return np.median(picked[:, ANGLES_SLICE], axis=0)


def build_reference_signatures(vocabulary: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """{tu: (N,5) mang chu ky} — CHI cho FUSION_WORDS, chi dung mau GOC (khong dung
    mau augment) cua chinh bank dang truyen vao (thuong la bank cua Dat)."""
    ref: Dict[str, List[np.ndarray]] = {}
    for w in FUSION_WORDS:
        sigs: List[np.ndarray] = []
        for s in vocabulary.get(w, {}).get("samples", []):
            if not _is_orig(s):
                continue
            traj = s.get("traj_features_9d")
            if traj is None:
                traj = s.get("traj_features")
            if traj is None or len(traj) != 32:
                continue
            sigs.append(boundary_signature(np.asarray(traj, dtype=np.float32)))
        if sigs:
            ref[w] = np.stack(sigs)
    return ref


REJECT_MULT = 1.4  # he so an toan nhan voi "thang do trong-lop" de cho phep TU CHOI (abstain)
# Do bang scratch/ (sweep 1.2-2.4 tren 242 truy van nguoi la, 4 nguoi, 10 tu): 1.4 la diem
# toi uu duoc do - fixed=6, broken=0 (thap hon hoac cao hon deu KEM hon, xem lich su sua doi).


def _intra_class_scale(reference: Dict[str, np.ndarray]) -> float:
    """
    Uoc luong 'thang do khoang cach trong-lop' bang LOO-NN NGAY TREN chinh cac mau tham
    chieu (moi mau, khoang cach toi mau GAN NHAT CUNG LOP con lai). Dung lam nguong TUYET
    DOI de biet mot query co thuc su gan cum DRINK/WATER hay khong - khac voi bo phieu
    KNN thuan tuy (chi so sanh TUONG DOI giua 2 lop, luon bi ep chon 1 trong 2 du query
    la tu hoan toan khac, vd YOU - day la loi da bat duoc khi hoi quy tren LOSO).
    """
    all_nn: List[float] = []
    for sigs in reference.values():
        n = len(sigs)
        if n < 2:
            continue
        for i in range(n):
            rest = np.delete(sigs, i, axis=0)
            d = np.linalg.norm(rest - sigs[i], axis=1)
            all_nn.append(float(d.min()))
    if not all_nn:
        return 0.15  # fallback an toan (xap xi intra_std do duoc o DRINK trong audit)
    return float(np.percentile(all_nn, 90))


def classify_handshape(query_traj_24d: Optional[np.ndarray],
                        reference: Dict[str, np.ndarray],
                        k: int = KNN_K) -> Optional[Tuple[str, int, int]]:
    """KNN bo phieu, CHI tren FUSION_WORDS, CO THE TU CHOI (tra ve None) neu query o qua
    xa ca 2 cum DRINK/WATER - bat buoc phai co buoc nay vi bo phan loai chi co 2 lop nen
    neu khong se LUON gan mot nhan cho moi query, ke ca tu hoan toan khac (da xac nhan
    bang hoi quy: YOU bi gan nham DRINK truoc khi them buoc tu choi nay).
    Tra ve (tu_du_doan, so_phieu_thang, k) de goi biet do "chac chan" the nao."""
    if not reference or query_traj_24d is None:
        return None
    traj = np.asarray(query_traj_24d, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] != 32:
        return None
    sig = boundary_signature(traj)

    dists: List[Tuple[float, str]] = []
    for w, sigs in reference.items():
        d = np.linalg.norm(sigs - sig, axis=1)
        dists += [(float(dd), w) for dd in d]
    if not dists:
        return None
    dists.sort(key=lambda x: x[0])

    reject_thresh = _intra_class_scale(reference) * REJECT_MULT
    if dists[0][0] > reject_thresh:
        return None  # qua xa ca 2 cum -> tu choi, khong ep chon

    top_k = dists[:k]
    votes: Dict[str, int] = {}
    for _, w in top_k:
        votes[w] = votes.get(w, 0) + 1
    best_word = max(votes, key=lambda w: votes[w])
    return best_word, votes[best_word], len(top_k)


def apply_drink_water_veto(result: Any, query_traj_24d: Optional[np.ndarray],
                            reference: Dict[str, np.ndarray]) -> Any:
    """
    LUẬT GATE (bắt buộc, không được nới lỏng):
      1. CHỈ can thiệp khi result.status != "ACCEPTED" — không bao giờ ghi đè quyết
         định mà DTW đang tự tin, vì đã đo được bộ phân loại này có sai lệch hệ thống
         với người lạ ở một số từ (xem docstring module).
      2. CHỈ xét khi ít nhất 1 trong top-3 candidate của DTW đã thuộc {DRINK, WATER} —
         không bao giờ "bịa" ra một từ nằm ngoài danh sách DTW đã đề xuất.
      3. Dáng tay phải thắng phiếu ĐA SỐ TUYỆT ĐỐI (>k/2), không chỉ nhiều phiếu nhất.
    Nếu cả 3 điều kiện thỏa và từ dáng tay chọn KHÁC top-1 hiện tại của DTW: xếp lại
    candidate đó lên đầu, nâng status lên "ACCEPTED" (dáng tay đã gỡ được nghi vấn).
    Ngược lại: trả về result NGUYÊN VẸN, không đổi gì.
    """
    if result.status == "ACCEPTED" or not result.candidates:
        return result

    top3_keys = {c.word_key for c in result.candidates[:3]}
    if not (top3_keys & set(FUSION_WORDS)):
        return result

    pred = classify_handshape(query_traj_24d, reference)
    if pred is None:
        return result
    shape_word, votes, k = pred
    if shape_word not in FUSION_WORDS or votes <= k // 2:
        return result  # khong du da so phieu chac chan -> khong can thiep

    cand_map = {c.word_key: c for c in result.candidates}
    chosen = cand_map.get(shape_word)
    if chosen is None or chosen is result.candidates[0]:
        return result  # tu dang tay chon khong nam trong candidate cua DTW, hoac da la top1

    new_candidates = [chosen] + [c for c in result.candidates if c is not chosen]
    note = f"handshape_veto:{shape_word}({votes}/{k})<-{result.top_word}"
    return dataclasses.replace(
        result,
        status="ACCEPTED",
        top_word=chosen.word_key,
        top_display=chosen.display_name,
        best_distance=chosen.distance,
        best_confidence=chosen.confidence,
        candidates=new_candidates,
        match_quality=chosen.match_quality,
        r_ratio=chosen.r_ratio,
        fusion_note=note,
    )
