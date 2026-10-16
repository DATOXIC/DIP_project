# -*- coding: utf-8 -*-
"""
LEAD TRIM - CẮT PHẦN DẪN ĐẦU THÍCH NGHI (SPEC 08B Nhiệm vụ C)
============================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

Mục đích:
    Trong chế độ ký nhanh (--fast-chain), sau khi chốt từ trước, chuyển động chuyển tiếp
    sang từ kế tiếp ("movement epenthesis") thường bị ghi vào đầu stroke kế.
    Module này thử nghiệm nhiều điểm cắt tau trong [0, max_sec] giây đầu dựa trên các
    cực tiểu vận tốc và lưới thời gian, chọn cách cắt cho kết quả so khớp bank tốt nhất.

Nguyên tắc an toàn:
    - Đường nhanh (Fast Path): nếu stroke gốc (tau=0) đã ACCEPTED với r_ratio <= good_r (0.80)
      thì KHÔNG tìm kiếm thêm (chi phí = 0 ms).
    - Điểm phạt: score = r_ratio + trim_penalty_per_sec * tau (mặc định 0.30/s).
    - Chỉ ứng viên ACCEPTED mới được xem xét.
    - tau > 0 chỉ thắng nếu cải thiện hơn tau=0 ít nhất min_gain (0.10), hoặc min_gain_other_word (0.25)
      nếu đổi sang từ khác.
    - Không có ứng viên nào đạt -> giữ nguyên kết quả tau=0 (hành vi cũ).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dynamic_dtw_engine import is_movement_epenthesis, is_stroke_duration_valid
from dynamic_features import build_segment_features, extend_still_tail, records_to_gesture_frames

DEFAULT_LEAD_TRIM_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "max_sec": 0.45,
    "max_variants": 6,
    "good_r": 0.80,
    "trim_penalty_per_sec": 0.30,
    "min_gain": 0.10,
    "min_gain_other_word": 0.25,
    "min_after_trim_sec": 0.30,
    "max_trim_fraction": 0.40,
}


@dataclass
class LeadTrimResult:
    """Kết quả phân tích cắt phần dẫn đầu thích nghi."""
    match_result: Any              # MatchResult trả về từ matcher.match()
    trim_sec: float                # Thời gian cắt tau được chọn (giây; 0.0 nếu không cắt)
    n_tried: int                   # Số ứng viên đã thử nghiệm
    info: Dict[str, Any]           # Thông tin bổ sung (chi tiết các ứng viên, fast path, v.v.)


def compute_frame_velocities(records: Sequence[Dict[str, Any]]) -> np.ndarray:
    """
    Tính mảng vận tốc chuẩn hóa v_norm từng khung giống KineticPauseSegmenter:
    v_norm[i] = hypot(dx, dy) / safe_palm.
    """
    n = len(records)
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    v_norms = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        dx = records[i]["x"] - records[i - 1]["x"]
        dy = records[i]["y"] - records[i - 1]["y"]
        safe_palm = max(records[i].get("palm_len", 50.0), 15.0)
        v_norms[i] = math.hypot(dx, dy) / safe_palm
    v_norms[0] = v_norms[1]
    return v_norms


def extract_trim_candidates(
    records: Sequence[Dict[str, Any]],
    max_sec: float = 0.45,
    max_variants: int = 6,
    min_after_trim_sec: float = 0.30,
    max_trim_fraction: float = 0.40,
) -> List[Tuple[int, float]]:
    """
    Trích xuất danh sách ứng viên cắt (frame_index, tau_seconds).
    tau = 0.0 (index 0) luôn có mặt đầu tiên.
    Dò cực tiểu vận tốc trong [0.05, max_sec] s kết hợp lưới thô {0.10, 0.20, 0.30, 0.40} <= max_sec.
    """
    n = len(records)
    if n < 4:
        return [(0, 0.0)]

    t0 = records[0]["t"]
    total_dur = records[-1]["t"] - t0
    if total_dur <= 0.0:
        return [(0, 0.0)]

    v_norms = compute_frame_velocities(records)
    frame_times = np.array([r["t"] - t0 for r in records], dtype=np.float32)

    # 1. Dò cực tiểu vận tốc cục bộ trong [0.05, max_sec]
    local_minima: List[Tuple[float, int, float]] = []  # (v_norm, idx, tau)
    for i in range(1, n - 1):
        tau = float(frame_times[i])
        if 0.05 <= tau <= max_sec:
            if v_norms[i] <= v_norms[i - 1] and v_norms[i] <= v_norms[i + 1]:
                # Kiểm tra điều kiện thời lượng
                rem_dur = total_dur - tau
                if rem_dur >= min_after_trim_sec and (tau / total_dur) <= max_trim_fraction:
                    local_minima.append((float(v_norms[i]), i, tau))

    # Sắp xếp cực tiểu theo độ sâu (v_norm nhỏ nhất trước)
    local_minima.sort(key=lambda x: x[0])

    # 2. Lưới thô {0.10, 0.20, 0.30, 0.40} <= max_sec
    coarse_grid = [0.10, 0.20, 0.30, 0.40]
    coarse_candidates: List[Tuple[int, float]] = []
    for g_tau in coarse_grid:
        if g_tau <= max_sec:
            rem_dur = total_dur - g_tau
            if rem_dur >= min_after_trim_sec and (g_tau / total_dur) <= max_trim_fraction:
                idx = int(np.searchsorted(frame_times, g_tau))
                idx = min(max(idx, 0), n - 1)
                actual_tau = float(frame_times[idx])
                coarse_candidates.append((idx, actual_tau))

    # 3. Tổng hợp ứng viên, loại trùng lặp theo chỉ số khung
    seen_indices = {0}
    candidates: List[Tuple[int, float]] = [(0, 0.0)]

    # Ưu tiên cực tiểu vận tốc sâu
    for _v, idx, tau in local_minima:
        if idx not in seen_indices and len(candidates) < max_variants:
            seen_indices.add(idx)
            candidates.append((idx, tau))

    # Bổ sung từ lưới thô nếu còn chỗ
    for idx, tau in coarse_candidates:
        if idx not in seen_indices and len(candidates) < max_variants:
            seen_indices.add(idx)
            candidates.append((idx, tau))

    # Sắp xếp các ứng viên sau tau=0 theo thứ tự thời gian tăng dần
    tail_candidates = sorted(candidates[1:], key=lambda x: x[1])
    return [candidates[0]] + tail_candidates


def _eval_one_trim(
    records: Sequence[Dict[str, Any]],
    start_idx: int,
    matcher: Any,
    handedness: str,
    query_has_body: bool,
    tail_pad_sec: Optional[float],
    v_stop: float,
) -> Tuple[Optional[Any], str]:
    """Đánh giá một phân đoạn từ start_idx đến cuối."""
    sub_recs = list(records[start_idx:])
    if len(sub_recs) < 3:
        return None, "too_short"

    if tail_pad_sec:
        sub_recs, _ = extend_still_tail(sub_recs, float(tail_pad_sec), v_stop, replay_jitter=True)
        dur = sub_recs[-1]["t"] - sub_recs[0]["t"]
        epi_frames = records_to_gesture_frames(sub_recs)
    else:
        dur = sub_recs[-1]["t"] - sub_recs[0]["t"]
        epi_frames = records_to_gesture_frames(sub_recs)

    if not is_stroke_duration_valid(dur):
        return None, "duration_gate"
    if is_movement_epenthesis(epi_frames):
        return None, "epenthesis"

    traj, shp, body_ok, _dom = build_segment_features(sub_recs)
    if traj is None:
        return None, "build_features_failed"

    effective_body = body_ok if query_has_body else False
    m = matcher.match(traj, shp, handedness=handedness, query_has_body=effective_body)
    return m, m.status


def select_best_lead_trim(
    records: Sequence[Dict[str, Any]],
    matcher: Any,
    cfg: Optional[Dict[str, Any]] = None,
    handedness: str = "Right",
    query_has_body: bool = True,
    tail_pad_sec: Optional[float] = None,
    v_stop: float = 0.040,
) -> LeadTrimResult:
    """
    Thuật toán chính cắt phần dẫn đầu thích nghi (SPEC 08B Nhiệm vụ C).
    
    Args:
        records: Danh sách bản ghi từng khung của stroke sắp chốt
        matcher: Đối tượng GoldenTriadDTWMatcher hoặc LooMatcher
        cfg: Dict cấu hình (lấy từ CONFIG["fast_chain"]["lead_trim"])
        handedness: Nhãn tay thuận ("Right" hoặc "Left")
        query_has_body: Cờ mốc cơ thể hợp lệ
        tail_pad_sec: Kéo dài đuôi yên (nếu fast_chain có bật)
        v_stop: Ngưỡng dừng vận tốc chuẩn hóa
    """
    c = dict(DEFAULT_LEAD_TRIM_CONFIG)
    if cfg:
        c.update({k: v for k, v in cfg.items() if v is not None})

    max_sec = float(c["max_sec"])
    max_variants = int(c["max_variants"])
    good_r = float(c["good_r"])
    trim_penalty_per_sec = float(c["trim_penalty_per_sec"])
    min_gain = float(c["min_gain"])
    min_gain_other_word = float(c["min_gain_other_word"])
    min_after_trim_sec = float(c["min_after_trim_sec"])
    max_trim_fraction = float(c["max_trim_fraction"])

    # 1. Đánh giá ứng viên tau = 0.0
    m_0, status_0 = _eval_one_trim(
        records, 0, matcher, handedness, query_has_body, tail_pad_sec, v_stop
    )

    if m_0 is None:
        # Stroke gốc không qua cổng cơ bản
        fallback_result = getattr(matcher, "_empty_match_result", None)
        if callable(fallback_result):
            empty_m = fallback_result()
        else:
            from dynamic_dtw_engine import MatchResult
            empty_m = MatchResult(status="REJECTED", top_word=None, top_display=None, best_distance=999.0, best_confidence=0.0)
        return LeadTrimResult(match_result=empty_m, trim_sec=0.0, n_tried=1, info={"status": status_0})

    r_0 = float(getattr(m_0, "r_ratio", 0.0))

    # 2. ĐƯỜNG NHANH (Fast Path): nếu tau=0 đã ACCEPTED với r <= good_r thì KHÔNG tìm kiếm
    if m_0.status == "ACCEPTED" and r_0 <= good_r:
        return LeadTrimResult(
            match_result=m_0,
            trim_sec=0.0,
            n_tried=1,
            info={"fast_path": True, "r_0": r_0, "word": m_0.top_word}
        )

    # 3. Trích xuất các ứng viên cắt tau > 0
    candidates = extract_trim_candidates(
        records,
        max_sec=max_sec,
        max_variants=max_variants,
        min_after_trim_sec=min_after_trim_sec,
        max_trim_fraction=max_trim_fraction,
    )

    tried_info: List[Dict[str, Any]] = [
        {"tau": 0.0, "status": m_0.status, "word": m_0.top_word, "r": r_0, "score": r_0}
    ]

    best_tau = 0.0
    best_m = m_0
    best_score = r_0 if m_0.status == "ACCEPTED" else float("inf")

    # 4. Thử nghiệm các ứng viên tau > 0
    n_tried = 1
    for idx, tau in candidates[1:]:
        n_tried += 1
        m_tau, status_tau = _eval_one_trim(
            records, idx, matcher, handedness, query_has_body, tail_pad_sec, v_stop
        )
        if m_tau is None or m_tau.status != "ACCEPTED":
            tried_info.append({"tau": tau, "status": status_tau or "failed", "word": None, "r": None, "score": None})
            continue

        r_tau = float(getattr(m_tau, "r_ratio", 0.0))
        score_tau = r_tau + trim_penalty_per_sec * tau
        tried_info.append({"tau": tau, "status": m_tau.status, "word": m_tau.top_word, "r": r_tau, "score": score_tau})

        # 5. Quy tắc chọn:
        # Chỉ ứng viên có status == "ACCEPTED" mới được xem xét.
        # tau > 0 chỉ thắng nếu tốt hơn tau=0 ít nhất min_gain (0.10);
        # nếu top-1 khác từ của tau=0 thì cần mức cao hơn min_gain_other_word (0.25).
        if m_0.status == "ACCEPTED":
            req_gain = min_gain_other_word if (m_tau.top_word != m_0.top_word) else min_gain
            # Điểm điều chỉnh score_tau phải tốt hơn (nhỏ hơn) r_0 ít nhất req_gain
            if (r_0 - score_tau) >= req_gain and score_tau < best_score:
                best_score = score_tau
                best_tau = tau
                best_m = m_tau
        else:
            # tau=0 KHÔNG accepted, nhưng m_tau ACCEPTED!
            if score_tau < best_score:
                best_score = score_tau
                best_tau = tau
                best_m = m_tau

    return LeadTrimResult(
        match_result=best_m,
        trim_sec=best_tau,
        n_tried=n_tried,
        info={"fast_path": False, "tried": tried_info, "selected_tau": best_tau, "selected_word": best_m.top_word}
    )
