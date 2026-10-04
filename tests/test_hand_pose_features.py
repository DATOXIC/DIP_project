# -*- coding: utf-8 -*-
"""Kiểm chứng tính bất biến của hand_pose_features (xoay 3D, tỉ lệ, dịch chuyển, chiều tay)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_pose_features import (  # noqa: E402
    FEATURE_DIM, _rotation_matrix, canonicalize, landmarks_to_features,
    simulate_view, view_angle_deg,
)


def make_hand(curl: float = 0.0, seed: int = 0) -> np.ndarray:
    """Bàn tay tổng hợp (tay phải, lòng bàn tay hướng +z): cổ tay ở gốc, ngón mọc theo +y.
    curl ∈ [0,1] gập các ngón về phía lòng bàn tay (+z)."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((21, 3))
    mcp_x = {1: -0.35, 5: -0.25, 9: 0.0, 13: 0.22, 17: 0.42}
    for base, x in mcp_x.items():
        y0 = 0.45 if base != 1 else 0.2
        pts[base] = [x, y0, 0.0]
        direction = np.array([0.0, 1.0, 0.0])
        cur = pts[base].copy()
        for k in range(1, 4):
            ang = curl * 0.9 * k / 3.0
            direction = np.array([x * 0.15, np.cos(ang), np.sin(ang)])
            direction /= np.linalg.norm(direction)
            cur = cur + 0.22 * direction
            pts[base + k] = cur
    pts += rng.normal(0, 0.002, pts.shape)
    return pts


def mirror_x(p: np.ndarray) -> np.ndarray:
    q = p.copy()
    q[:, 0] *= -1.0
    return q


def test_feature_dim():
    assert landmarks_to_features(make_hand(), "Right").shape == (FEATURE_DIM,)


def test_invariant_translation_scale_rotation():
    p = make_hand(curl=0.6)
    ref = landmarks_to_features(p, "Right")
    rng = np.random.default_rng(1)
    for _ in range(20):
        rot = _rotation_matrix(*rng.uniform(-3.1, 3.1, 3))
        q = (p @ rot.T) * rng.uniform(0.3, 4.0) + rng.normal(0, 5, 3)
        np.testing.assert_allclose(landmarks_to_features(q, "Right"), ref, atol=2e-4)


def test_left_hand_mirror_matches_right():
    p = make_hand(curl=0.5)
    right = landmarks_to_features(p, "Right")
    left = landmarks_to_features(mirror_x(p), "Left")
    np.testing.assert_allclose(left, right, atol=2e-4)


def test_different_shapes_stay_different():
    a = landmarks_to_features(make_hand(curl=0.0), "Right")
    b = landmarks_to_features(make_hand(curl=0.9), "Right")
    assert np.linalg.norm(a - b) > 1.0


def test_degenerate_returns_zeros():
    assert not np.any(landmarks_to_features(np.zeros((21, 3)), "Right"))


def test_view_angle_range():
    p = make_hand()
    assert view_angle_deg(p) < 1.0                      # lòng bàn tay hướng +z => thẳng camera
    rot = _rotation_matrix(np.radians(80), 0, 0)
    assert view_angle_deg(p @ rot.T) > 70.0             # quay gần như nhìn cạnh


def test_simulate_view_without_noise_is_invariant_with_noise_is_not():
    p = make_hand(curl=0.4)
    ref = landmarks_to_features(p, "Right")
    rng = np.random.default_rng(3)
    clean = simulate_view(p, rng, z_sigma_range=(0.0, 0.0), xy_sigma=0.0)
    np.testing.assert_allclose(landmarks_to_features(clean, "Right"), ref, atol=2e-4)
    noisy = simulate_view(p, rng, z_sigma_range=(0.2, 0.3))
    assert np.linalg.norm(landmarks_to_features(noisy, "Right") - ref) > 0.05
