# -*- coding: utf-8 -*-
"""
BỘ LỌC ỨNG VIÊN THEO DÁNG TAY (handshape gate) — dùng nhánh landmark 3D chuẩn hóa.

Ý tưởng: DTW so khớp QUỸ ĐẠO; dáng tay (chữ cái/số) là bằng chứng ĐỘC LẬP. Hai từ có dáng tay khác nhau
(PLEASE = B, SORRY = A) không thể bị nhầm với nhau nếu dáng tay đo được đủ rõ. Chỉ các từ CÙNG NHÓM dáng tay
mới còn được phép nhầm nhau.

Quy trình:
  1. Mỗi khung -> xác suất theo lớp (letter_landmark_classifier).
  2. Gộp các lớp gần nhau thành NHÓM dáng tay (SHAPE_GROUPS): A/S/E (nắm đấm), B/5 (bàn tay mở), D/1 (chỉ một ngón).
  3. Bằng chứng của cả cử chỉ = trung bình xác suất nhóm ở 25% khung ĐẦU + 25% khung CUỐI (lúc tay đã chốt dáng).
  4. Bảng từ -> nhóm cho phép được HỌC từ bản ghi của chủ dự án (learn_word_table), không lấy từ lý thuyết ASL.
     Từ nào dáng tay đo ra không ổn định thì đánh dấu 'reliable=False' => KHÔNG ràng buộc.
  5. apply_shape_gate loại các ứng viên trong top-K của DTW mà MÂU THUẪN rõ với dáng tay đo được.

An toàn: mặc định chỉ can thiệp khi DTW KHÔNG tự tin (status != ACCEPTED); chế độ ràng buộc cứng
(gate_accepted=True, đúng ý "chỉ được nhầm giữa các từ cùng chữ cái") có sẵn để ĐO rủi ro, không bật mặc định.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Lớp -> nhóm. Lớp không liệt kê tự là nhóm riêng (tên nhóm = tên lớp).
SHAPE_GROUPS: Dict[str, str] = {
    "a": "fist", "s": "fist", "e": "fist",
    "b": "open", "5": "open",
    "d": "point", "1": "point",
}
BOUNDARY_FRAC = 0.25

# Từ có dáng tay KHÔNG thuộc bảng chữ cái/số đã train (EAT: các đầu ngón chụm lại; HUNGRY: bàn tay khum kéo xuống).
# Model chỉ "đoán bừa" ở các từ này (đã đo: EAT bị đọc là B tới 98%), nên KHÔNG BAO GIỜ được ràng buộc dù bảng học
# từ dữ liệu có vẻ ổn định. Thêm từ vào đây khi dáng tay của nó không có lớp tương ứng.
NO_LETTER_HANDSHAPE = frozenset({"EAT", "HUNGRY"})


def group_of(cls: str) -> str:
    return SHAPE_GROUPS.get(cls, cls)


def group_names(classes: Sequence[str]) -> List[str]:
    seen: List[str] = []
    for c in classes:
        g = group_of(c)
        if g not in seen:
            seen.append(g)
    return seen


def to_group_probs(probs: np.ndarray, classes: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """(..., C) xác suất lớp -> (..., G) xác suất nhóm (cộng dồn các lớp cùng nhóm)."""
    names = group_names(classes)
    idx = {g: i for i, g in enumerate(names)}
    out = np.zeros(probs.shape[:-1] + (len(names),), dtype=np.float32)
    for j, c in enumerate(classes):
        out[..., idx[group_of(c)]] += probs[..., j]
    return out, names


def boundary_group_evidence(frame_probs: np.ndarray, classes: Sequence[str]) -> Dict[str, float]:
    """frame_probs: (T, C). Trả về {nhóm: xác suất trung bình ở 25% khung đầu + 25% khung cuối}."""
    p = np.asarray(frame_probs, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] == 0:
        return {}
    T = p.shape[0]
    k = max(1, int(round(T * BOUNDARY_FRAC)))
    picked = np.concatenate([p[:k], p[-k:]], axis=0) if 2 * k < T else p
    g, names = to_group_probs(picked.mean(axis=0), classes)
    return {n: float(v) for n, v in zip(names, g)}


@dataclasses.dataclass
class WordShape:
    profile: Dict[str, float]      # nhóm -> khối lượng trung bình
    allowed: Set[str]              # nhóm được phép
    reliable: bool                 # True => được dùng để loại ứng viên


def learn_word_table(evidence_by_word: Dict[str, List[Dict[str, float]]], min_mass: float = 0.15,
                     reliable_top: float = 0.6, min_samples: int = 5,
                     never_constrain: Sequence[str] = NO_LETTER_HANDSHAPE) -> Dict[str, WordShape]:
    """
    Học bảng từ -> nhóm dáng tay từ các bằng chứng của MỘT chủ thể tin cậy (chủ dự án).
      allowed  = nhóm có khối lượng trung bình >= min_mass
      reliable = có >= min_samples mẫu VÀ nhóm trội nhất >= reliable_top (dáng tay đo ổn định)
    """
    table: Dict[str, WordShape] = {}
    for w, evs in evidence_by_word.items():
        evs = [e for e in evs if e]
        if not evs:
            continue
        groups = sorted({g for e in evs for g in e})
        mean = {g: float(np.mean([e.get(g, 0.0) for e in evs])) for g in groups}
        allowed = {g for g, m in mean.items() if m >= min_mass}
        reliable = len(evs) >= min_samples and max(mean.values()) >= reliable_top and w not in never_constrain
        table[w] = WordShape(profile=mean, allowed=allowed, reliable=reliable)
    return table


def compat(word: str, evidence: Dict[str, float], table: Dict[str, WordShape]) -> Optional[float]:
    """Khối lượng bằng chứng nằm trong nhóm cho phép của từ; None nếu từ không bị ràng buộc."""
    ws = table.get(word)
    if ws is None or not ws.reliable:
        return None
    return float(sum(evidence.get(g, 0.0) for g in ws.allowed))


def apply_shape_gate(result: Any, evidence: Optional[Dict[str, float]], table: Dict[str, WordShape], *,
                     conf_thresh: float = 0.6, compat_thresh: float = 0.15, top_k: int = 3,
                     gate_accepted: bool = False, strict_promote: bool = True) -> Any:
    """
    Loại ứng viên mâu thuẫn dáng tay khỏi top-K của DTW.

    Điều kiện can thiệp (tất cả phải đúng):
      - có bằng chứng và nhóm trội nhất của bằng chứng >= conf_thresh (đo đủ chắc);
      - status != ACCEPTED, trừ khi gate_accepted=True (chế độ cứng, chỉ để đo);
      - một ứng viên top-K bị ràng buộc (reliable) và compat(w) < compat_thresh.
    Không bao giờ tạo từ ngoài danh sách DTW; nếu mọi ứng viên top-K đều mâu thuẫn thì trả về nguyên vẹn.
    Nếu top-1 đổi: giữ status UNCERTAIN (không nâng lên ACCEPTED). Nếu top-1 giữ nguyên nhưng đối thủ bị loại và
    status đang UNCERTAIN: nâng lên ACCEPTED (nghi vấn nhập nhằng đã được gỡ).
    """
    if result is None or not result.candidates or not evidence:
        return result
    if max(evidence.values()) < conf_thresh:
        return result
    if result.status == "ACCEPTED" and not gate_accepted:
        return result

    cands = list(result.candidates)
    head = cands[:top_k]
    removed = []
    for c in head:
        cp = compat(c.word_key, evidence, table)
        if cp is not None and cp < compat_thresh:
            removed.append(c)
    if not removed:
        return result
    survivors = [c for c in head if c not in removed]
    if not survivors:
        return result
    new_cands = survivors + [c for c in cands[top_k:]]
    top = new_cands[0]
    changed = top.word_key != result.top_word
    if changed:
        status = "ACCEPTED" if (gate_accepted and result.status == "ACCEPTED") else "UNCERTAIN"
    else:
        # Chỉ nâng lên ACCEPTED khi KHÔNG còn từ "tự do" nào trong top-K: từ không ràng buộc (EAT, HUNGRY...) không thể
        # bị dáng tay loại trừ nên vẫn là đối thủ chưa gỡ được nghi vấn.
        free_left = any(compat(c.word_key, evidence, table) is None for c in survivors[1:])
        status = "ACCEPTED" if (result.status == "UNCERTAIN" and not (strict_promote and free_left)) else result.status
    note = "shape_gate:" + ",".join(c.word_key for c in removed) + (f"->{top.word_key}" if changed else "")
    return dataclasses.replace(
        result, status=status, top_word=top.word_key, top_display=top.display_name,
        best_distance=top.distance, best_confidence=top.confidence, candidates=new_cands,
        match_quality=top.match_quality, r_ratio=top.r_ratio, fusion_note=note,
    )


# =============================================================================
# LỚP CHẠY THẬT: nạp model + bảng, tính bằng chứng từ các bản ghi khung của cử chỉ vừa chốt
# =============================================================================

def save_word_table(table: Dict[str, WordShape], path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump({w: {"profile": v.profile, "allowed": sorted(v.allowed), "reliable": v.reliable} for w, v in table.items()},
                  f, ensure_ascii=False, indent=2)


def load_word_table(path: str) -> Dict[str, WordShape]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {w: WordShape(profile={k: float(x) for k, x in v["profile"].items()}, allowed=set(v["allowed"]),
                         reliable=bool(v["reliable"])) for w, v in raw.items()}


class HandshapeGateRuntime:
    """
    Dùng trong predict_sentence_webcam.py. Chỉ chạy MỘT lần mỗi cử chỉ lúc chốt (vài ms), dùng landmark đã có sẵn
    trong các bản ghi khung (khóa 'landmarks': (21,3) pixel, z*bề rộng — đúng không gian model 'image' đã học).
    """

    def __init__(self, model_path: str, table_path: str, conf_thresh: float = 0.6, compat_thresh: float = 0.15,
                 top_k: int = 3, min_frames: int = 6) -> None:
        from letter_landmark_classifier import LetterLandmarkClassifier
        self.clf = LetterLandmarkClassifier(model_path, min_conf=0.0)
        if self.clf.space != "image":
            raise ValueError(f"Model {model_path} học landmark '{self.clf.space}', pipeline động cần model huấn luyện với --space image")
        self.table = load_word_table(table_path)
        self.conf_thresh, self.compat_thresh, self.top_k, self.min_frames = conf_thresh, compat_thresh, top_k, min_frames
        self.last_evidence: Optional[Dict[str, float]] = None

    def evidence(self, records: Sequence[Dict[str, Any]], handedness: Optional[str]) -> Optional[Dict[str, float]]:
        from hand_pose_features import landmarks_to_features
        feats = [landmarks_to_features(np.asarray(r["landmarks"], dtype=np.float32), r.get("handedness") or handedness)
                 for r in records if r.get("landmarks") is not None]
        if len(feats) < self.min_frames:
            return None
        return boundary_group_evidence(self.clf.predict_features(np.stack(feats)), self.clf.classes)

    def apply(self, result: Any, records: Sequence[Dict[str, Any]], handedness: Optional[str]) -> Any:
        """Luôn an toàn: mọi lỗi/thiếu bằng chứng => trả về result nguyên vẹn."""
        try:
            ev = self.evidence(records, handedness)
            self.last_evidence = ev
            return apply_shape_gate(result, ev, self.table, conf_thresh=self.conf_thresh,
                                    compat_thresh=self.compat_thresh, top_k=self.top_k)
        except Exception:
            return result
