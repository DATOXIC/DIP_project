"""
TEST: spotter_gate (RuleSpotter + OneShotLock) - cơ chế KHÓA MỘT LẦN.
Dùng chuỗi v_norm tổng hợp, không cần webcam/cv2.
"""

import numpy as np
import pytest

from spotter_gate import OneShotLock, RuleSpotter, Spotter, make_spotter

DT = 1.0 / 30.0


def _run(spotter, v_seq, t0=0.0, hand=True):
    """Chạy spotter qua chuỗi v_norm; trả về list (t, open)."""
    out = []
    for i, v in enumerate(v_seq):
        t = t0 + i * DT
        out.append((t, spotter.update(t, hand, v)))
    return out


def _spotter():
    return RuleSpotter(open_v=0.05, close_v=0.035, open_hold_sec=0.10, close_hold_sec=0.25)


def test_rule_spotter_is_spotter():
    assert isinstance(_spotter(), Spotter)
    assert isinstance(make_spotter("rule"), RuleSpotter)
    with pytest.raises(ValueError):
        make_spotter("khong_co")


def test_rule_spotter_invalid_hysteresis():
    with pytest.raises(ValueError):
        RuleSpotter(open_v=0.02, close_v=0.05)


def test_opens_only_after_hold():
    sp = _spotter()
    res = _run(sp, [0.1] * 10)
    # 0.10 s ~ 3 khung: khung đầu chưa mở, về sau mở
    assert res[0][1] is False
    assert res[1][1] is False
    assert res[-1][1] is True
    first_open = next(t for t, o in res if o)
    assert first_open - res[0][0] >= 0.10 - 1e-9


def test_brief_spike_does_not_open():
    sp = _spotter()
    res = _run(sp, [0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0])
    assert not any(o for _, o in res)


def test_closes_after_still_hold_and_not_before():
    sp = _spotter()
    _run(sp, [0.1] * 10)
    assert sp.is_open
    # đứng yên ngắn (< 0.25 s ~ 7.5 khung): vẫn mở
    res = _run(sp, [0.0] * 5, t0=10 * DT)
    assert all(o for _, o in res)
    # cử động lại -> bộ đếm đóng bị reset
    res = _run(sp, [0.1] * 3, t0=15 * DT)
    assert all(o for _, o in res)
    # đứng yên đủ lâu -> đóng
    res = _run(sp, [0.0] * 12, t0=18 * DT)
    assert res[0][1] is True
    assert res[-1][1] is False


def test_hysteresis_band_keeps_open():
    sp = _spotter()
    _run(sp, [0.1] * 10)
    # v giữa close_v và open_v (0.04): không được coi là đứng yên => vẫn mở
    res = _run(sp, [0.04] * 30, t0=10 * DT)
    assert all(o for _, o in res)


def test_hand_lost_closes_immediately():
    sp = _spotter()
    _run(sp, [0.1] * 10)
    assert sp.is_open
    assert sp.update(1.0, False, 0.0) is False
    # tay quay lại nhưng chưa đủ open_hold -> vẫn đóng
    assert sp.update(1.01, True, 0.1) is False


def test_from_segmenter_thresholds():
    sp = RuleSpotter.from_segmenter(0.035)
    assert sp.open_v == pytest.approx(0.0525)
    assert sp.close_v == pytest.approx(0.035)
    assert sp.close_hold_sec == pytest.approx(0.25)


def test_reset():
    sp = _spotter()
    _run(sp, [0.1] * 10)
    sp.reset()
    assert sp.is_open is False


def _lock_from(spotter, segments):
    """segments: list (n_frames, v). Trả về (lock, t_cuối)."""
    lock = OneShotLock()
    t = 0.0
    for n, v in segments:
        for _ in range(n):
            lock.observe(t, spotter.update(t, True, v))
            t += DT
    return lock, t


def test_four_commits_in_one_span_one_allowed_three_suppressed():
    lock = OneShotLock()
    for i in range(10):
        lock.observe(i * DT, True)
    t0 = 0.05
    results = [lock.allow_commit(t0 + 0.05 * k) for k in range(4)]
    assert results[0] == (True, "first")
    assert all(r == (False, "dup_in_span") for r in results[1:])
    assert lock.stats == {"allowed": 1, "suppressed": 3}


def test_two_spans_two_allowed():
    lock = OneShotLock()
    for i in range(10):
        lock.observe(i * DT, True)
    for i in range(10, 25):
        lock.observe(i * DT, False)
    for i in range(25, 40):
        lock.observe(i * DT, True)
    for i in range(40, 55):
        lock.observe(i * DT, False)
    assert len(lock.spans) == 2
    assert lock.spans[0][2] is not None
    assert lock.allow_commit(2 * DT) == (True, "first")
    assert lock.allow_commit(27 * DT) == (True, "first")
    assert lock.allow_commit(30 * DT) == (False, "dup_in_span")
    assert lock.stats == {"allowed": 2, "suppressed": 1}


def test_no_span_fail_open():
    lock = OneShotLock()
    assert lock.allow_commit(1.0) == (True, "no_span")
    assert lock.allow_commit(1.0) == (True, "no_span")
    assert lock.stats == {"allowed": 2, "suppressed": 0}


def test_stroke_start_long_after_closed_span_is_no_span():
    lock = OneShotLock()
    for i in range(10):
        lock.observe(i * DT, True)
    lock.observe(10 * DT, False)
    assert lock.allow_commit(0.1) == (True, "first")
    # stroke bắt đầu 5 s sau khi span đóng: không thuộc span cũ => fail-open
    assert lock.allow_commit(5.0) == (True, "no_span")


def test_stroke_start_slightly_before_span_open_maps_to_span():
    lock = OneShotLock()
    lock.observe(1.0, False)
    lock.observe(1.1, True)   # span mở tại 1.1
    assert lock.allow_commit(1.0) == (True, "first")   # stroke bắt đầu 0.1 s trước khi spotter mở
    assert lock.allow_commit(1.2) == (False, "dup_in_span")


def test_memory_limit():
    lock = OneShotLock(max_spans=50)
    t = 0.0
    for _ in range(120):
        lock.observe(t, True)
        t += 0.2
        lock.observe(t, False)
        t += 0.2
    spans = lock.spans
    assert len(spans) == 50
    ids = [s[0] for s in spans]
    assert ids == sorted(ids) and ids[-1] == 119   # giữ các span GẦN NHẤT


def test_end_to_end_rule_spotter_with_lock():
    """Một động tác dài có 4 stroke bị cắt (không đứng yên đủ lâu) -> 1 allow, 3 suppress;
    sau khi đứng yên đủ lâu + ký tiếp -> span mới được phép."""
    sp = _spotter()
    lock = OneShotLock()
    t = 0.0
    starts = []
    # động tác 1: chuyển động liên tục, xen vài nhịp khựng ngắn (< close_hold_sec)
    for block in range(4):
        starts.append(t)
        for _ in range(8):
            lock.observe(t, sp.update(t, True, 0.1))
            t += DT
        for _ in range(4):   # khựng ~0.13 s
            lock.observe(t, sp.update(t, True, 0.0))
            t += DT
    verdicts = [lock.allow_commit(s)[0] for s in starts]
    assert verdicts == [True, False, False, False]
    # nghỉ dài rồi ký tiếp
    for _ in range(20):
        lock.observe(t, sp.update(t, True, 0.0))
        t += DT
    t_next = t
    for _ in range(10):
        lock.observe(t, sp.update(t, True, 0.1))
        t += DT
    assert lock.allow_commit(t_next) == (True, "first")


def test_known_limit_two_words_without_pause_share_span():
    """GIỚI HẠN ĐÃ BIẾT: hai từ liền nhau không có khoảng khựng đủ dài nằm chung một span -> từ thứ hai bị nuốt."""
    sp = _spotter()
    lock, _ = _lock_from(sp, [(30, 0.1)])
    assert lock.allow_commit(0.0)[0] is True
    assert lock.allow_commit(0.6)[0] is False


# ------------------------------------------------------------------ OneShotLock strict + DLSpotter
def test_lock_strict_blocks_stroke_without_span():
    lock = OneShotLock(strict=True)
    ok, why = lock.allow_commit(5.0)                       # chưa có span nào: tay vu vơ, spotter báo "không ký"
    assert (ok, why) == (False, "no_span") and lock.stats["suppressed"] == 1
    for k in range(10):
        lock.observe(10.0 + 0.1 * k, True)
    lock.observe(11.0, False)
    assert lock.allow_commit(10.2) == (True, "first")
    assert lock.allow_commit(10.6) == (False, "dup_in_span")
    assert OneShotLock().allow_commit(5.0) == (True, "no_span")   # mặc định vẫn fail-open như cũ


def _random_model(tmp_path, seed=0):
    import torch
    from spotter_dl import build_net
    from spotter_features import FEAT_DIM, WINDOW
    torch.manual_seed(seed)
    net = build_net()
    path = str(tmp_path / "m.pth")
    rng = np.random.default_rng(seed)
    torch.save({"state_dict": net.state_dict(), "mean": rng.normal(0, 0.1, FEAT_DIM).astype(np.float32),
                "std": (1.0 + rng.random(FEAT_DIM)).astype(np.float32), "window": WINDOW, "feat_dim": FEAT_DIM,
                "version": 1}, path)
    return path


def test_dl_spotter_streaming_equals_offline(tmp_path):
    import torch
    from spotter_dl import DLSpotter, load_model
    from spotter_features import session_features, windows_from_features
    path = _random_model(tmp_path)
    rng = np.random.default_rng(1)
    T = 60
    base = rng.random((21, 3)) * 0.2 + 0.4
    lm = np.stack([base + 0.01 * rng.normal(size=(21, 3)) + 0.002 * i for i in range(T)])   # (T, 21, 3)
    lm[20:26] = np.nan                                       # mất tay vài khung
    t = np.arange(T) / 30.0
    scores = np.ones((T, 2))
    lm2 = np.full((T, 2, 21, 3), np.nan)
    lm2[:, 0] = lm
    feats = session_features(lm2, scores, t, 640.0, 480.0)
    net, mean, std = load_model(path)
    w = (windows_from_features(feats) - mean) / std
    with torch.no_grad():
        offline = torch.sigmoid(net(torch.from_numpy(w.astype(np.float32)))).numpy()
    sp = DLSpotter(path)
    online = []
    for i in range(T):
        found = not np.isnan(lm[i, 0, 0])
        sp.update(float(t[i]), found, 0.0, lm[i] if found else None, (640, 480))
        online.append(sp.last_prob)
    assert np.allclose(np.array(online), offline, atol=1e-5)


def test_dl_spotter_rejects_mismatched_model(tmp_path):
    import torch
    from spotter_dl import load_model
    p = str(tmp_path / "bad.pth")
    torch.save({"state_dict": {}, "mean": np.zeros(3), "std": np.ones(3), "window": 8, "feat_dim": 3}, p)
    with pytest.raises(ValueError):
        load_model(p)
