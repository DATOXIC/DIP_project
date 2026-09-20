# -*- coding: utf-8 -*-
"""
Kiểm thử tương đương giữa đường DTW nhanh (vector hóa) và đường gốc của GoldenTriadDTWMatcher.match.
Dùng bank thật webcam_templates.json (bỏ qua nếu không có file).
"""
import os
import copy
import json
import numpy as np
import pytest

from dynamic_dtw_engine import (
    GoldenTriadDTWMatcher,
    compute_golden_triad_dtw_distance,
    canonicalize_chirality,
    _FastDTWBank,
)

BANK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webcam_templates.json")
pytestmark = pytest.mark.skipif(not os.path.exists(BANK), reason="Không có webcam_templates.json")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DIP_FAST_DTW", raising=False)


def _make(fast: bool, **kw):
    kw.setdefault("sign_weights_path", "")
    kw.setdefault("strict_weights", False)
    return GoldenTriadDTWMatcher(templates_path=BANK, use_fast_dtw=fast, **kw)


def _traj(s):
    raw = s.get("traj_features_9d")
    if raw is None:
        raw = s["traj_features"]
    return np.array(raw, dtype=np.float32)


def _queries(matcher, seed=1234):
    """Mẫu bank + bản nhiễu Gaussian 0.05 + biến thể chirality (None/Left/Right) => >= 100 query."""
    rng = np.random.default_rng(seed)
    out = []
    for wk, wd in matcher.vocabulary.items():
        for s in wd["samples"][:3]:
            t = _traj(s)
            noisy = (t + rng.normal(0, 0.05, t.shape)).astype(np.float32)
            for base in (t, noisy):
                for hand in (None, "Left", "Right"):
                    out.append((base, s.get("hand_shapes", ["Unknown"] * 32), hand,
                                bool(s.get("body_anchor_valid", True))))
    return out


def test_khoang_cach_tung_cap_giong_duong_goc():
    m = _make(True)
    qs = _queries(m)[::2]   # thưa 1/2 vì đường gốc chậm; vẫn >= 100 query
    assert len(qs) >= 100
    bank = _FastDTWBank(m)
    assert bank.supported
    max_diff, n_pairs = 0.0, 0
    for traj, shapes, hand, has_body in qs:
        q, _ = canonicalize_chirality(traj, handedness=hand)
        fast = bank.distances(q, has_body)
        i = 0
        for wk, wd in m.vocabulary.items():
            eff = m._get_effective_weights(wk)
            for s in wd["samples"]:
                wb = eff["w_b"] if (has_body and s.get("body_anchor_valid", True)) else 0.0
                ref = compute_golden_triad_dtw_distance(
                    q, shapes, _traj(s), s.get("hand_shapes", ["Unknown"] * 32),
                    w_p=eff["w_p"], w_a=eff["w_a"], w_m=eff["w_m"], w_n=eff["w_n"],
                    lambda_shape=0.0, lambda_shape_global=0.0, w_z=0.0, w_b=wb, w_s=0.0, w_r=0.0)
                max_diff = max(max_diff, abs(ref - float(fast[i])))
                i += 1
                n_pairs += 1
    print(f"[parity] {len(qs)} query, {n_pairs} cặp, sai khác tối đa {max_diff:.3e}")
    assert max_diff < 1e-5


def _assert_same_result(a, b):
    assert a.status == b.status
    assert a.top_word == b.top_word
    assert [c.word_key for c in a.candidates] == [c.word_key for c in b.candidates]
    assert a.best_distance == pytest.approx(b.best_distance, abs=1e-5)
    assert a.best_confidence == pytest.approx(b.best_confidence, abs=1e-5)
    assert a.match_quality == pytest.approx(b.match_quality, abs=1e-3)
    assert a.r_ratio == pytest.approx(b.r_ratio, abs=1e-5)
    for ca, cb in zip(a.candidates, b.candidates):
        assert ca.distance == pytest.approx(cb.distance, abs=1e-5)
        assert ca.display_name == cb.display_name


@pytest.mark.parametrize("cfg", [
    {"sign_weights_path": ""},
    {"sign_weights_path": "", "threshold_mode": "scale_k"},
    {},  # cấu hình mặc định (có trọng số riêng theo ASL-LEX nếu file có)
])
def test_matchresult_giong_duong_goc(cfg):
    slow, fast = _make(False, **cfg), _make(True, **cfg)
    qs = _queries(slow)
    # đường gốc chậm (~0,4-0,7 s/lần): cấu hình chính lấy 1/2 (>= 100 query), cấu hình phụ lấy 1/6
    qs = qs[::6] if cfg else qs[::2]
    for traj, shapes, hand, has_body in qs:
        ra = slow.match(traj, shapes, handedness=hand, query_has_body=has_body)
        rb = fast.match(traj, shapes, handedness=hand, query_has_body=has_body)
        _assert_same_result(ra, rb)
    assert len(qs) >= (30 if cfg else 100)
    assert fast._fast_bank is not None and fast._fast_bank.supported


def test_fallback_khi_ngoai_pham_vi():
    slow, fast = _make(False, w_z=0.5), _make(True, w_z=0.5)
    for traj, shapes, hand, has_body in _queries(slow)[:20]:
        _assert_same_result(slow.match(traj, shapes, handedness=hand, query_has_body=has_body),
                            fast.match(traj, shapes, handedness=hand, query_has_body=has_body))
    assert fast._fast_bank is not None and not fast._fast_bank.supported   # đã nhận diện ngoài phạm vi
    # lambda_shape khác 0 cũng phải quay về đường gốc
    s2, f2 = _make(False, lambda_shape=0.3), _make(True, lambda_shape=0.3)
    traj, shapes, hand, hb = _queries(s2)[7]
    _assert_same_result(s2.match(traj, shapes, handedness=hand), f2.match(traj, shapes, handedness=hand))
    assert not f2._fast_bank.supported


def test_query_sai_chieu_thi_fallback():
    fast, slow = _make(True), _make(False)
    first = next(iter(fast.vocabulary.values()))["samples"][0]
    traj = _traj(first)[:, :17]
    _assert_same_result(slow.match(traj, ["Unknown"] * 32), fast.match(traj, ["Unknown"] * 32))


def test_invalidate_sau_set_vocabulary_va_them_mau():
    fast, slow = _make(True), _make(False)
    traj, shapes, hand, hb = _queries(fast)[5]
    fast.match(traj, shapes, handedness=hand)
    bank_cu = fast._fast_bank
    assert bank_cu is not None

    with open(BANK, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = data["vocabulary"]
    # bank mới: bỏ 1 từ, thêm 1 mẫu nhiễu vào từ khác
    keys = list(vocab.keys())
    del vocab[keys[0]]
    extra = copy.deepcopy(vocab[keys[1]]["samples"][0])
    t = _traj(extra)
    extra.pop("traj_features_9d", None)
    extra["traj_features"] = (t + np.random.default_rng(7).normal(0, 0.05, t.shape)).astype(np.float32).tolist()
    vocab[keys[1]]["samples"].append(extra)
    for m in (fast, slow):
        m.set_vocabulary(vocab, data.get("metadata"))
    assert fast._fast_bank is None                      # đã bị vô hiệu
    for q in _queries(slow)[:30]:
        _assert_same_result(slow.match(q[0], q[1], handedness=q[2]), fast.match(q[0], q[1], handedness=q[2]))
    assert fast._fast_bank is not bank_cu
    assert keys[0] not in fast._fast_bank.words

    # sửa trực tiếp vocabulary (không qua set_vocabulary): chữ ký tự phát hiện
    fast.vocabulary[keys[1]]["samples"].pop()
    slow.vocabulary[keys[1]]["samples"].pop()
    q = _queries(slow)[3]
    _assert_same_result(slow.match(q[0], q[1], handedness=q[2]), fast.match(q[0], q[1], handedness=q[2]))
    assert fast._fast_bank.N == sum(len(w["samples"]) for w in fast.vocabulary.values())

    # đổi trọng số toàn cục
    fast.w_p, slow.w_p = 1.1, 1.1
    _assert_same_result(slow.match(q[0], q[1], handedness=q[2]), fast.match(q[0], q[1], handedness=q[2]))


def test_cong_tac_moi_truong(monkeypatch):
    fast = _make(True)
    q = _queries(fast)[4]
    monkeypatch.setenv("DIP_FAST_DTW", "0")
    fast.match(q[0], q[1], handedness=q[2])
    assert fast._fast_bank is None                      # tắt khẩn cấp: không dựng khối
    monkeypatch.setenv("DIP_FAST_DTW", "verify")
    r = fast.match(q[0], q[1], handedness=q[2])
    monkeypatch.delenv("DIP_FAST_DTW")
    _assert_same_result(_make(False).match(q[0], q[1], handedness=q[2]), r)
    # strict: lỗi trong đường nhanh phải nổi lên; chế độ thường thì fallback
    monkeypatch.setattr(_FastDTWBank, "distances", lambda *a, **k: 1 / 0)
    monkeypatch.setenv("DIP_FAST_DTW", "strict")
    with pytest.raises(ZeroDivisionError):
        _make(True).match(q[0], q[1], handedness=q[2])
    monkeypatch.delenv("DIP_FAST_DTW")
    _assert_same_result(_make(False).match(q[0], q[1], handedness=q[2]), _make(True).match(q[0], q[1], handedness=q[2]))
