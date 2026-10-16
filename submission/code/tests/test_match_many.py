"""Kiểm thử GoldenTriadDTWMatcher.match_many (SPEC 09 Nhiệm vụ B) trên bank TỔNG HỢP nhỏ do test tự tạo (không đọc file bank sống)."""
import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamic_dtw_engine import GoldenTriadDTWMatcher, _FastDTWBank  # noqa: E402

K, D = 32, 24


def _smooth(rng):
    t = np.cumsum(rng.normal(0, 0.05, (K, D)), axis=0).astype(np.float32)
    n = rng.normal(0, 1, (K, 3)).astype(np.float32) + np.array([0, 0, -2], dtype=np.float32)
    t[:, 9:12] = n / np.linalg.norm(n, axis=1, keepdims=True)
    return t


def _make_bank(rng, words=5, per=4):
    vocab = {}
    for w in range(words):
        base = _smooth(rng)
        samples = []
        for _ in range(per):
            t = base + rng.normal(0, 0.02, base.shape).astype(np.float32)
            t[:, 9:12] /= np.linalg.norm(t[:, 9:12], axis=1, keepdims=True)
            samples.append({"traj_features": t.tolist(), "hand_shapes": ["Unknown"] * K, "rep_count": 1,
                            "body_anchor_valid": bool(rng.random() > 0.2), "handedness": "Right"})
        vocab[f"W{w}"] = {"display_name": f"W{w}", "threshold": 1.25, "num_samples": per, "samples": samples}
    return {"metadata": {"feature_dim": 24}, "vocabulary": vocab}


def _matcher(tmp_path, bank, **kw):
    return GoldenTriadDTWMatcher(templates_path=str(tmp_path / "x.json"), sign_weights_path="", w_p=0.6, w_a=1.0, w_m=0.4,
                                 w_n=0.8, w_b=0.8, templates_dict=bank, threshold_mode="scale_k", threshold_k=1.75,
                                 uncertainty_k=2.35, cache_dir=str(tmp_path / "cache"), **kw)


def _queries(m, rng, n):
    pool = [np.array(s["traj_features"], dtype=np.float32) for wd in m.vocabulary.values() for s in wd["samples"]]
    out = []
    for i in range(n):
        t = pool[i % len(pool)] + rng.normal(0, 0.03 + 0.05 * (i % 4), (K, D)).astype(np.float32)
        t[:, 9:12] /= np.linalg.norm(t[:, 9:12], axis=1, keepdims=True)
        out.append((t, ["Unknown"] * K, ["Left", "Right", None][i % 3], bool(i % 5)))
    return out


def _same(a, b):
    assert a.status == b.status and a.top_word == b.top_word
    assert a.best_distance == b.best_distance and a.match_quality == b.match_quality and a.r_ratio == b.r_ratio
    assert [(c.word_key, c.distance, c.match_quality, c.r_ratio) for c in a.candidates] == \
        [(c.word_key, c.distance, c.match_quality, c.r_ratio) for c in b.candidates]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mm")
    rng = np.random.default_rng(7)
    m = _matcher(tmp, _make_bank(rng))
    return m, _queries(m, rng, 120), tmp


def test_parity_matchresult_identico(env):
    m, qs, _ = env
    many = m.match_many(qs)
    assert len(many) == len(qs)
    for q, r in zip(qs, many):
        _same(m.match(*q), r)


def test_parity_khoang_cach_moi_template(env):
    m, qs, _ = env
    m.match(*qs[0])
    bank = m._fast_bank
    arr = np.stack([q[0] for q in qs[:100]])
    hb = [q[3] for q in qs[:100]]
    dm = bank.distances_many(arr, hb, chunk=32)
    worst = 0.0
    for i in range(100):
        worst = max(worst, float(np.abs(dm[i] - bank.distances(arr[i], hb[i])).max()))
    assert worst < 1e-5


def test_thu_tu_va_do_dai(env):
    m, qs, _ = env
    sub = qs[:9]
    many = m.match_many(sub)
    rev = m.match_many(sub[::-1])[::-1]
    for a, b in zip(many, rev):
        _same(a, b)
    assert m.match_many([]) == []


def test_fallback_khi_tat_duong_nhanh(env, monkeypatch):
    m, qs, _ = env
    monkeypatch.setenv("DIP_FAST_DTW", "0")
    for q, r in zip(qs[:5], m.match_many(qs[:5])):
        _same(m.match(*q), r)


def test_fallback_khi_query_sai_kich_thuoc(env):
    m, qs, _ = env
    bad = (qs[0][0][:20], ["Unknown"] * 20, "Right", True)
    res = m.match_many([qs[1], bad])
    _same(m.match(*qs[1]), res[0])
    _same(m.match(*bad), res[1])


def test_fallback_lambda_shape_khac_0(tmp_path):
    rng = np.random.default_rng(3)
    m = _matcher(tmp_path, _make_bank(rng, 3, 3), lambda_shape=0.1)
    qs = _queries(m, rng, 4)
    for q, r in zip(qs, m.match_many(qs)):
        _same(m.match(*q), r)


def test_thoi_gian_khong_qua_cham_so_voi_vong_lap(env):
    """Chỉ kiểm tra không tệ hơn nhiều lần (đo thật in ra bằng -s); KHÔNG khẳng định nhanh hơn."""
    m, qs, _ = env
    m.match_many(qs[:4])
    for n in (1, 8, 16, 32):
        a = time.perf_counter(); m.match_many(qs[:n]); t_many = time.perf_counter() - a
        a = time.perf_counter(); [m.match(*x) for x in qs[:n]]; t_loop = time.perf_counter() - a
        print(f"[match_many] n={n}: many {t_many*1000:.1f} ms | loop {t_loop*1000:.1f} ms")
        assert t_many < 3.0 * t_loop + 0.05
