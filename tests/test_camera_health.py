"""Kiểm thử camera_health.CameraHealthMonitor (khung mới thật/giây, độ sáng, cảnh báo có trễ)."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_health import CameraHealthMonitor  # noqa: E402

RNG = np.random.default_rng(1)


_POOL = {}
_CNT = [0]


def _noisy(level: int) -> np.ndarray:
    """Khung 640x480 có nhiễu cảm biến quanh mức xám `level` (xoay vòng 5 khung nhiễu khác nhau để chạy nhanh;
    hai lần gọi liên tiếp luôn cho hai khung KHÁC nhau)."""
    if level not in _POOL:
        _POOL[level] = [np.clip(level + RNG.integers(-6, 7, (480, 640, 3)), 0, 255).astype(np.uint8) for _ in range(5)]
    _CNT[0] += 1
    return _POOL[level][_CNT[0] % 5].copy()


def _run(mon, seconds, read_hz=30.0, new_hz=30.0, level=140, moving=False, t0=0.0):
    """Mô phỏng luồng đọc: mỗi lần đọc trả khung mới với tần số new_hz, còn lại lặp đúng khung cũ. Trả về t cuối."""
    dt = 1.0 / read_hz
    cur = _noisy(level)
    next_new = t0
    t = t0
    events = []
    while t < t0 + seconds:
        if t >= next_new - 1e-9:
            cur = _noisy(level)
            next_new += 1.0 / new_hz
        mon.update(cur.copy(), t, moving)
        events.append((t, mon.status()))
        t += dt
    return t, events


def test_identical_frame_is_duplicate_and_noisy_is_new():
    mon = CameraHealthMonitor()
    f = _noisy(120)
    mon.update(f.copy(), 0.0, False)
    assert mon.is_new is True                     # khung đầu tiên luôn là mới
    mon.update(f.copy(), 0.03, False)
    assert mon.is_new is False                    # giống hệt từng điểm ảnh
    mon.update(_noisy(120), 0.06, False)
    assert mon.is_new is True                     # có nhiễu -> mới


def test_unique_fps_and_dup_ratio_measured():
    mon = CameraHealthMonitor()
    _run(mon, 4.0, read_hz=30.0, new_hz=6.0)
    assert mon.unique_fps == pytest.approx(6.0, abs=0.7)
    assert mon.loop_fps == pytest.approx(30.0, abs=1.0)
    assert mon.dup_ratio == pytest.approx(0.8, abs=0.05)


def test_slow_camera_warning_when_hand_moving():
    mon = CameraHealthMonitor()
    t_end, ev = _run(mon, 8.0, read_hz=30.0, new_hz=6.0, moving=True)
    first = next((t for t, s in ev if s is not None), None)
    assert first is not None, "phải phát cảnh báo slow_camera"
    assert first >= 2.0
    assert ev[-1][1][0] == "slow_camera"
    assert ev[-1][1][1] == "bad"                  # 6 < 8


def test_static_scene_with_duplicates_no_warning():
    mon = CameraHealthMonitor()
    _, ev = _run(mon, 10.0, read_hz=30.0, new_hz=6.0, moving=False)
    assert all(s is None for _, s in ev)


def test_healthy_camera_no_warning():
    mon = CameraHealthMonitor()
    _, ev = _run(mon, 8.0, read_hz=30.0, new_hz=30.0, moving=True, level=140)
    assert all(s is None for _, s in ev)


def test_low_light_and_bad_level():
    mon = CameraHealthMonitor()
    _, ev = _run(mon, 8.0, level=80)
    assert ev[-1][1] == ("low_light", "warn")
    mon2 = CameraHealthMonitor()
    _, ev2 = _run(mon2, 8.0, level=50)
    assert ev2[-1][1] == ("low_light", "bad")


def test_too_bright_by_mean_and_by_saturation():
    mon = CameraHealthMonitor()
    _, ev = _run(mon, 8.0, level=235)
    assert ev[-1][1][0] == "too_bright"
    # nửa khung cháy sáng (>= 250), phần còn lại vừa phải: trung bình ~ 190 nhưng sat_frac = 50%
    f = np.full((480, 640, 3), 130, np.uint8)
    f[:, 320:] = 255
    mon2 = CameraHealthMonitor()
    for k in range(240):
        g = np.clip(f.astype(np.int16) + RNG.integers(-2, 3, f.shape), 0, 255).astype(np.uint8)
        mon2.update(g, k / 30.0, False)
    assert mon2.status() is not None and mon2.status()[0] == "too_bright"


def test_priority_too_bright_over_slow():
    mon = CameraHealthMonitor()
    _, ev = _run(mon, 10.0, read_hz=30.0, new_hz=6.0, moving=True, level=240)
    assert ev[-1][1][0] == "too_bright"


def test_no_flicker_hysteresis_and_min_show():
    """Độ sáng dao động sát ngưỡng vào/ra: cảnh báo không được nhấp nháy; mỗi lần bật hiện >= 3 s."""
    mon = CameraHealthMonitor()
    t = 0.0
    _, ev = _run(mon, 6.0, level=80)              # bật low_light
    assert ev[-1][1] is not None
    t0 = ev[-1][0]
    # dao động 92..100 (quanh ngưỡng vào 95 nhưng dưới ngưỡng thoát 104,5): phải giữ nguyên
    seq = []
    tt = t0
    for k in range(600):
        lvl = 92 if (k // 15) % 2 == 0 else 100
        mon.update(_noisy(lvl), tt, False)
        seq.append(mon.status())
        tt += 1 / 30.0
    assert all(s is not None for s in seq)
    # trở lại sáng bình thường: chỉ tắt sau >= clear_sec liên tục
    _, ev2 = _run(mon, 6.0, level=150, t0=tt)
    off = next(t for t, s in ev2 if s is None)
    assert off - tt >= 2.0
    changes = sum(1 for a, b in zip(ev2, ev2[1:]) if a[1] != b[1])
    assert changes <= 1


def test_warnings_disabled_still_measures():
    mon = CameraHealthMonitor(warnings_enabled=False)
    _, ev = _run(mon, 6.0, level=40)
    assert all(s is None for _, s in ev)
    assert mon.luma_mean < 60


RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "camera_diag", "toi_20260920_090135", "raw.avi")


@pytest.mark.skipif(not os.path.exists(RAW), reason="không có raw.avi của phiên đo tối")
def test_real_dark_session_luma():
    cap = cv2.VideoCapture(RAW)
    if not cap.isOpened():
        pytest.skip("không mở được raw.avi")
    mon = CameraHealthMonitor()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    seen = set()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        mon.update(f, i / fps, False)
        seen.add(mon.status())
        i += 1
    assert i > 100
    assert 60 < mon.luma_mean < 90               # phiên tối: độ sáng TB ~ 72-79
    assert ("low_light", "warn") in seen or ("low_light", "bad") in seen
