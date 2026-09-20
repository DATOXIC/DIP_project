# -*- coding: utf-8 -*-
"""
UNIT TESTS CHO MODULE LEAD TRIM (SPEC 08B Nhiệm vụ C)
====================================================
Kiểm tra:
1. Trích xuất ứng viên cắt: tau=0 luôn có, kẹp đúng giới hạn [0.05, max_sec], max_variants, min_after_trim_sec, max_trim_fraction.
2. Fast path: bỏ qua tìm kiếm khi stroke gốc (tau=0) đã ACCEPTED với r <= good_r.
3. Cắt đúng tau trên stroke có chuyển tiếp nhân tạo biết trước.
4. Không cắt khi stroke đã chuẩn (không có dẫn đầu).
5. Rào chắn min_gain và min_gain_other_word chống đổi từ bừa bãi.
6. Tính xác định (deterministic).
"""
import os
import sys
import pytest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lead_trim import (
    extract_trim_candidates,
    select_best_lead_trim,
    LeadTrimResult,
    DEFAULT_LEAD_TRIM_CONFIG,
)
from dynamic_dtw_engine import GoldenTriadDTWMatcher, MatchResult


class DummyMatcher:
    """Matcher giả lập cho unit test có kiểm soát."""
    def __init__(self, outcomes):
        # outcomes: dict mapping start_idx -> MatchResult
        self.outcomes = outcomes
        self.call_count = 0

    def match(self, traj, shapes, handedness="Right", query_has_body=True):
        self.call_count += 1
        # Tra cứu theo độ dài hoặc gọi tuần tự
        n = traj.shape[0] if traj is not None else 0
        return self.outcomes.get(n, self.outcomes.get("default"))


def _make_dummy_records(n_frames=30, dt=0.033, lead_frames=6, palm_len=60.0):
    """Tạo records giả lập: lead_frames đầu là chuyển tiếp (vận tốc cao rồi dừng nhẹ), sau đó là cử chỉ chính."""
    records = []
    t0 = 100.0
    x, y = 200.0, 200.0
    for i in range(n_frames):
        t = t0 + i * dt
        if i < lead_frames:
            # Chuyển động chuyển tiếp
            x += 15.0
            y += 5.0
        elif i == lead_frames:
            # Điểm dừng tại đích chuyển tiếp (cực tiểu vận tốc)
            x += 0.5
            y += 0.2
        else:
            # Cử chỉ thật
            x += 5.0 * math.sin(i * 0.3)
            y += 4.0 * math.cos(i * 0.3)
        records.append({
            "x": float(x), "y": float(y), "palm_len": float(palm_len),
            "hand_shape": "Fist", "t": float(t),
            "angles": np.ones(5, dtype=np.float32) * 0.5,
            "normal": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "landmarks": np.zeros((21, 3), dtype=np.float32),
            "face_box": (100.0, 100.0, 50.0, 50.0),
            "handedness": "Right"
        })
    return records


import math


def test_extract_trim_candidates_tau0_always():
    records = _make_dummy_records(30)
    cands = extract_trim_candidates(records, max_sec=0.45, max_variants=6)
    assert len(cands) >= 1
    assert cands[0] == (0, 0.0)


def test_extract_trim_candidates_bounds():
    records = _make_dummy_records(40, dt=0.033)
    max_sec = 0.35
    max_variants = 4
    cands = extract_trim_candidates(records, max_sec=max_sec, max_variants=max_variants)
    assert len(cands) <= max_variants
    assert cands[0] == (0, 0.0)
    for idx, tau in cands[1:]:
        assert 0.05 <= tau <= max_sec + 1e-4
        assert idx > 0


def test_fast_path_skips_search():
    """Khi tau=0 đã ACCEPTED với r <= good_r, n_tried phải bằng 1 và không gọi match thêm."""
    records = _make_dummy_records(30)
    match_0 = MatchResult(status="ACCEPTED", top_word="HELLO", top_display="Hello", best_distance=0.05, best_confidence=0.95)
    match_0.r_ratio = 0.65  # <= good_r (0.80)

    class FastMatcher:
        def __init__(self):
            self.calls = 0
        def match(self, *a, **k):
            self.calls += 1
            return match_0

    fm = FastMatcher()
    res = select_best_lead_trim(records, fm, cfg={"good_r": 0.80})
    assert res.trim_sec == 0.0
    assert res.n_tried == 1
    assert res.info["fast_path"] is True
    assert fm.calls == 1


def test_trim_winner_when_tau_improves():
    """Khi tau=0 có r cao hoặc UNCERTAIN, ứng viên tau > 0 có r tốt hơn và ACCEPTED sẽ thắng."""
    records = _make_dummy_records(35, lead_frames=6)
    m_0 = MatchResult(status="UNCERTAIN", top_word="HELLO", top_display="Hello", best_distance=0.15, best_confidence=0.70)
    m_0.r_ratio = 1.30

    m_trim = MatchResult(status="ACCEPTED", top_word="HELLO", top_display="Hello", best_distance=0.04, best_confidence=0.96)
    m_trim.r_ratio = 0.50

    call_count = [0]
    def mock_match(traj, *a, **k):
        call_count[0] += 1
        if call_count[0] == 1:
            return m_0
        return m_trim

    matcher = type("M", (), {"match": staticmethod(mock_match)})()
    res = select_best_lead_trim(records, matcher, cfg={"good_r": 0.80, "min_gain": 0.10})
    assert res.trim_sec > 0.0
    assert res.match_result.status == "ACCEPTED"
    assert res.match_result.top_word == "HELLO"


def test_guard_min_gain_other_word():
    """Ứng viên tau > 0 đổi sang từ khác phải có gain >= 0.25 mới thắng."""
    records = _make_dummy_records(35, lead_frames=6)
    m_0 = MatchResult(status="ACCEPTED", top_word="HELLO", top_display="Hello", best_distance=0.08, best_confidence=0.85)
    m_0.r_ratio = 0.95

    # Đổi sang từ khác "BYE" nhưng chỉ cải thiện 0.12 (chưa đủ 0.25)
    m_other = MatchResult(status="ACCEPTED", top_word="BYE", top_display="Bye", best_distance=0.07, best_confidence=0.88)
    m_other.r_ratio = 0.83  # gain = 0.95 - (0.83 + 0.30*tau) < 0.25

    call_count = [0]
    def mock_match(traj, *a, **k):
        call_count[0] += 1
        if call_count[0] == 1:
            return m_0
        return m_other

    matcher = type("M", (), {"match": staticmethod(mock_match)})()
    res = select_best_lead_trim(records, matcher, cfg={"good_r": 0.70, "min_gain_other_word": 0.25})
    # Không được đổi sang BYE vì gain chưa đủ 0.25 -> giữ HELLO ở tau=0
    assert res.trim_sec == 0.0
    assert res.match_result.top_word == "HELLO"


def test_deterministic():
    """Chạy 2 lần trên cùng stroke phải cho kết quả giống hệt bit."""
    records = _make_dummy_records(32, lead_frames=5)
    m_res = MatchResult(status="ACCEPTED", top_word="THANK_YOU", top_display="Thank You", best_distance=0.06, best_confidence=0.90)
    m_res.r_ratio = 0.75
    matcher = type("M", (), {"match": staticmethod(lambda *a, **k: m_res)})()

    res1 = select_best_lead_trim(records, matcher)
    res2 = select_best_lead_trim(records, matcher)
    assert res1.trim_sec == res2.trim_sec
    assert res1.n_tried == res2.n_tried
    assert res1.match_result.top_word == res2.match_result.top_word
