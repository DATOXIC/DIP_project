# -*- coding: utf-8 -*-
import json
import os

import numpy as np

from collect_continuous import find_session_dirs, load_session, WANDER
from online_negatives import OnlineNegativeRecorder


def _pts(x0=0.4):
    p = np.zeros((21, 3), dtype=np.float32)
    p[:, 0] = x0 + np.linspace(0, 0.1, 21)
    p[:, 1] = 0.5
    return p


def _fill(rec, n=90, fps=30.0, t0=100.0, mirror=False, label="Left"):
    for i in range(n):
        rec.push(t0 + i / fps, label, 0.9, _pts(), mirror=mirror)
    return t0 + (n - 1) / fps


def test_save_negative_creates_loadable_wander_session(tmp_path):
    rec = OnlineNegativeRecorder(root=str(tmp_path), window_sec=2.0)
    t_end = _fill(rec)
    d = rec.save_negative(t_end, reason="user_n", extra={"removed_word": "Yes"})
    assert d is not None and d.endswith("_onlineneg") and not d.endswith("_neg")
    assert find_session_dirs(str(tmp_path)) == [d]
    s = load_session(d)
    assert s["landmarks"].shape[1:] == (2, 21, 3)
    assert int((s["labels"] == WANDER).sum()) > 30          # phần lớn cửa sổ là WANDER (trừ biên)
    assert s["video_path"] is None
    log = [json.loads(x) for x in open(os.path.join(str(tmp_path), "log.jsonl"), encoding="utf-8")]
    assert log[0]["kind"] == "negative" and log[0]["removed_word"] == "Yes"


def test_window_is_limited_and_too_few_frames_rejected(tmp_path):
    rec = OnlineNegativeRecorder(root=str(tmp_path), buffer_sec=4.0, window_sec=1.0, min_frames=20)
    t_end = _fill(rec, n=300)                                # 10 s dữ liệu, chỉ giữ 4 s trong bộ đệm
    assert len(rec) <= 4 * 30 + 2
    d = rec.save_negative(t_end)
    s = load_session(d)
    assert s["t"].shape[0] <= 1.0 * 30 + 2                   # chỉ 1 s gần nhất
    empty = OnlineNegativeRecorder(root=str(tmp_path / "x"))
    assert empty.save_negative(0.0) is None


def test_mirror_flips_x_and_swaps_handedness(tmp_path):
    rec = OnlineNegativeRecorder(root=str(tmp_path), window_sec=2.0)
    t_end = _fill(rec, mirror=True, label="Left")
    d = rec.save_negative(t_end)
    s = load_session(d)
    # nhãn Left bị đảo thành Right => khe 1; x -> 1 - x
    assert np.isnan(s["landmarks"][0, 0]).all() and not np.isnan(s["landmarks"][0, 1]).any()
    assert abs(float(s["landmarks"][0, 1, 0, 0]) - (1.0 - 0.4)) < 1e-5


def test_lost_hand_frames_are_nan(tmp_path):
    rec = OnlineNegativeRecorder(root=str(tmp_path), window_sec=2.0)
    for i in range(60):
        rec.push(10.0 + i / 30.0, None, 0.0, None)
    d = rec.save_negative(10.0 + 59 / 30.0)
    assert np.isnan(load_session(d)["landmarks"]).all()
