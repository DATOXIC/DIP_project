"""
Unit tests for Golden Triad [P + A + JM] feature engineering and DTW matcher (v5.1).
Covers:
- Joint flexion angles computation (straight = 0.0, fully bent = 1.0)
- 3D rotation invariance of joint flexion angles
- Resampling of 9D feature sequences to K=32
- Golden Triad Dimension-Normalized Weighted DTW distance
- Relative Contrast Margin Guard safe division
- Backward compatibility with legacy 4D templates
"""

import math
import time
import numpy as np
import pytest

from predict_sentence_webcam import (
    compute_joint_flexion_angles,
    compute_palm_normal_vector,
    resample_9d_gesture_sequence,
    compute_golden_triad_dtw_distance,
    GoldenTriadDTWMatcher,
)


def test_straight_finger_angles():
    """Tat ca 5 ngon duoi thang hoan toan -> Goc uon (Flexion Score) phai bang 0.0."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Thumb: 1 -> 2 -> 3
    lm[1] = [0.0, 0.0, 0.0]
    lm[2] = [0.0, 1.0, 0.0]
    lm[3] = [0.0, 2.0, 0.0]

    # Index: 5 -> 6 -> 7
    lm[5] = [1.0, 0.0, 0.0]
    lm[6] = [1.0, 1.0, 0.0]
    lm[7] = [1.0, 2.0, 0.0]

    # Middle: 9 -> 10 -> 11
    lm[9] = [2.0, 0.0, 0.0]
    lm[10] = [2.0, 1.0, 0.0]
    lm[11] = [2.0, 2.0, 0.0]

    # Ring: 13 -> 14 -> 15
    lm[13] = [3.0, 0.0, 0.0]
    lm[14] = [3.0, 1.0, 0.0]
    lm[15] = [3.0, 2.0, 0.0]

    # Pinky: 17 -> 18 -> 19
    lm[17] = [4.0, 0.0, 0.0]
    lm[18] = [4.0, 1.0, 0.0]
    lm[19] = [4.0, 2.0, 0.0]

    angles = compute_joint_flexion_angles(lm)
    assert angles.shape == (5,)
    for k in range(5):
        assert abs(angles[k] - 0.0) < 1e-4, f"Finger {k} expected 0.0, got {angles[k]}"


def test_fully_bent_finger_angles():
    """Ngon tay gap sat 180 do -> Goc uon (Flexion Score) phai bang 1.0."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Index gap nguoc lai: LM5=[1, 0, 0], LM6=[1, 1, 0], LM7=[1, 0, 0]
    lm[5] = [1.0, 0.0, 0.0]
    lm[6] = [1.0, 1.0, 0.0]
    lm[7] = [1.0, 0.0, 0.0]

    angles = compute_joint_flexion_angles(lm)
    assert abs(angles[1] - 1.0) < 1e-4, f"Index finger expected 1.0, got {angles[1]}"


def test_angles_3d_rotation_invariance():
    """Xoay toan bo ban tay trong khong gian 3D -> Goc uon khong thay doi."""
    np.random.seed(42)
    lm = np.random.randn(21, 3).astype(np.float32)
    angles_orig = compute_joint_flexion_angles(lm)

    # Ma tran xoay Euler 3D (yaw, pitch, roll)
    alpha = np.radians(35)
    beta = np.radians(50)
    gamma = np.radians(70)

    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha),  np.cos(alpha), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ], dtype=np.float32)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma),  np.cos(gamma)]
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx
    lm_rotated = lm @ R.T

    angles_rot = compute_joint_flexion_angles(lm_rotated)
    np.testing.assert_allclose(angles_orig, angles_rot, atol=1e-4)


def test_resample_9d_shape_and_bounds():
    """Noi suy chuoi 9D tu T frames bat ky ve chuan K=32 frames."""
    np.random.seed(123)
    T = 24
    raw_frames_9d = np.random.randn(T, 9).astype(np.float32)
    shapes = ["A" if i < 12 else "B" for i in range(T)]

    res_traj, res_shapes = resample_9d_gesture_sequence(raw_frames_9d, shapes, k_target=32)

    assert res_traj.shape == (32, 9)
    assert len(res_shapes) == 32
    assert all(s in ["A", "B"] for s in res_shapes)
    assert not np.any(np.isnan(res_traj))
    assert not np.any(np.isinf(res_traj))


def test_golden_triad_dtw_distance():
    """Kiem tra khoang cach DTW 9D co rang buoc Sakoe-Chiba."""
    k = 32
    traj_q = np.ones((k, 9), dtype=np.float32)
    shapes_q = ["B"] * k

    # Trung khop hoan toan -> khoang cach = 0.0
    dist_zero = compute_golden_triad_dtw_distance(traj_q, shapes_q, traj_q, shapes_q)
    assert abs(dist_zero - 0.0) < 1e-4

    # Khac biet hoan toan -> khoang cach > 0
    traj_t = traj_q + 1.0
    shapes_t = ["V"] * k
    dist_diff = compute_golden_triad_dtw_distance(traj_q, shapes_q, traj_t, shapes_t)
    assert dist_diff > 0.5


def test_margin_guard_safe_division():
    """Margin guard voi can an toan khong bao gio gay loi ZeroDivisionError."""
    matcher = GoldenTriadDTWMatcher()
    ratio = matcher.compute_margin_ratio(d_top1=0.0, d_top2=0.5)
    assert ratio == 0.5 / 1e-4  # 5000.0


def test_feature_extraction_performance():
    """Do thoi gian tinh toan dac trung 9D < 0.15 ms tren CPU."""
    lm = np.random.randn(21, 3).astype(np.float32)
    N = 1000
    start = time.perf_counter()
    for _ in range(N):
        _ = compute_joint_flexion_angles(lm)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / N
    assert elapsed_ms < 0.15, f"Qua cham: {elapsed_ms:.4f} ms/frame"


def test_palm_normal_vector():
    """Kiem tra vector phap tuyen long ban tay 3D."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # LM0 (Wrist) = (0, 0, 0)
    # LM5 (Index MCP) = (0, 1, 0)  -> u = (0, 1, 0)
    # LM17 (Pinky MCP) = (1, 0, 0) -> v = (1, 0, 0)
    # u x v = (0*0 - 0*0, 0*1 - 0*0, 0*0 - 1*1) = (0, 0, -1)
    lm[5] = [0.0, 1.0, 0.0]
    lm[17] = [1.0, 0.0, 0.0]

    n_right = compute_palm_normal_vector(lm, handedness="Right")
    assert abs(np.linalg.norm(n_right) - 1.0) < 1e-4
    assert abs(n_right[0] - 0.0) < 1e-4
    assert abs(n_right[1] - 0.0) < 1e-4
    assert abs(n_right[2] - (-1.0)) < 1e-4

    # Handedness Left -> dao chieu n
    n_left = compute_palm_normal_vector(lm, handedness="Left")
    assert abs(n_left[2] - 1.0) < 1e-4


def test_12d_resample_and_dtw():
    """Kiem tra noi suy va so khop 12D voi khoang cach Cosine."""
    T = 20
    raw_12d = np.random.randn(T, 12).astype(np.float32)
    # 3 cot cuoi la normal vector can norm = 1
    for t in range(T):
        raw_12d[t, 9:12] /= np.linalg.norm(raw_12d[t, 9:12])
    shapes = ["B"] * T

    res_traj, res_shapes = resample_9d_gesture_sequence(raw_12d, shapes, k_target=32)
    assert res_traj.shape == (32, 12)
    assert len(res_shapes) == 32
    # Norm cua 3 cot cuoi phai luon la 1.0
    for i in range(32):
        assert abs(np.linalg.norm(res_traj[i, 9:12]) - 1.0) < 1e-3

    # So khop chinh no -> dist = 0
    d_same = compute_golden_triad_dtw_distance(res_traj, res_shapes, res_traj, res_shapes)
    assert abs(d_same - 0.0) < 1e-4
