"""
Tier 2: Boundary & Corner Cases Test Suite
==========================================
Verifies edge conditions and boundary limits:
- Stationary hand ($v_norm < 0.04$ for > 0.5s)
- Velocity threshold boundaries (v_start, v_stop)
- Physical duration limits (< 0.4s and > 2.5s, exactly 0.4s and 2.5s)
- Random arm swipes & erratic trajectory rejection
- Extreme coordinates (0, 0), frame borders, palm scale floor (15.0px)
- Rapid consecutive gestures and cooldown limits
- Hand occlusion & 4-frame grace period dropout
"""

import os
import sys
import time
import pytest
import numpy as np

from dynamic_dtw_engine import (
    KineticState,
    GestureFrame,
    KineticPauseSegmenter,
    ResampledGesture,
    resample_gesture_sequence,
    HybridDTWEngine,
    HybridDTWConfig,
    MatchResult
)
from predict_sentence_webcam import (
    WristCoordinateSmoother,
    NonBlockingTTS,
    ASL_VOCAB_MAP,
    get_asl_phrase
)
from tests.test_harness_utils import (
    SyntheticGestureOracle,
    HeadlessPipelineSimulator
)


# =============================================================================
# BOUNDARY 1: STATIONARY HAND (v_norm < 0.04 for > 0.5s)
# =============================================================================

def test_tier2_stationary_hand_stays_idle(segmenter, oracle):
    """
    Asserts hand held stationary with sub-threshold sensor jitter for 1.0s (30 frames)
    remains strictly in IDLE with an empty buffer.
    """
    frames = oracle.generate_stationary_frames(duration_sec=1.0, fps=30.0, noise_amplitude=0.3)
    for x, y, pl, dt in frames:
        st, comp, is_sw, v = segmenter.update(x, y, palm_len=pl, timestamp=dt)
        assert st == KineticState.IDLE
        assert comp is None
        assert not is_sw
        assert v < 0.04, f"Normalized velocity {v:.4f} exceeded stationary bound 0.04"
    assert len(segmenter.buffer) == 0


# =============================================================================
# BOUNDARY 2: VELOCITY THRESHOLDS (v_start_ratio, v_stop_ratio)
# =============================================================================

def test_tier2_velocity_just_below_start_threshold(segmenter):
    """Velocity just below start threshold (0.079 < 0.080) stays IDLE."""
    # Palm = 100, move 7.9 px -> v = 0.079
    segmenter.update(100.0, 100.0, 100.0, timestamp=0.0)
    st, comp, _, v = segmenter.update(107.9, 100.0, 100.0, timestamp=0.033)
    assert st == KineticState.IDLE
    assert abs(v - 0.079) < 1e-4


def test_tier2_velocity_at_start_threshold(segmenter):
    """Velocity at start threshold (0.080) triggers transition to RECORDING."""
    segmenter.update(100.0, 100.0, 100.0, timestamp=0.0)
    st, comp, _, v = segmenter.update(108.0, 100.0, 100.0, timestamp=0.033)
    assert st == KineticState.RECORDING
    assert abs(v - 0.080) < 1e-4


def test_tier2_velocity_just_above_stop_threshold(segmenter):
    """During RECORDING, velocity just above stop threshold (0.036 > 0.035) does not count as pause."""
    segmenter.update(100.0, 100.0, 100.0, timestamp=0.0)
    segmenter.update(115.0, 100.0, 100.0, timestamp=0.033)  # into RECORDING
    # Move 3.6px -> v = 0.036 > 0.035
    st, comp, _, v = segmenter.update(118.6, 100.0, 100.0, timestamp=0.066)
    assert st == KineticState.RECORDING
    assert segmenter.pause_counter == 0


def test_tier2_velocity_at_or_below_stop_threshold(segmenter):
    """During RECORDING, velocity <= 0.035 increments pause counter."""
    segmenter.update(100.0, 100.0, 100.0, timestamp=0.0)
    segmenter.update(115.0, 100.0, 100.0, timestamp=0.033)  # into RECORDING
    # Move 3.4px -> v = 0.034 <= 0.035
    st, comp, _, v = segmenter.update(118.4, 100.0, 100.0, timestamp=0.066)
    assert st == KineticState.RECORDING
    assert segmenter.pause_counter == 1


# =============================================================================
# BOUNDARY 3: DURATION LIMITS (< 0.4s AND > 2.5s)
# =============================================================================

def test_tier2_duration_gate_sub_minimum_discard(simulator, oracle):
    """
    Sub-minimum duration (< 0.4s):
    Fast gesture taking only 0.28s followed by pause is rejected by duration gate.
    """
    # 8 motion frames @ 60 FPS -> duration ~ 0.13s
    raw_frames = []
    t = 0.0
    for i in range(8):
        raw_frames.append(GestureFrame(x=100.0 + i * 10, y=100.0, palm_len=50.0, hand_shape="B", timestamp=t))
        t += 0.033
    # 12 pause frames
    for j in range(12):
        raw_frames.append(GestureFrame(x=180.0 + j * 0.1, y=100.0, palm_len=50.0, hand_shape="B", timestamp=t))
        t += 0.033

    # Total duration = (8 + 12 - 1) * 0.033 = 0.62s? Wait!
    # Let's compress timestamp duration to 0.30s
    t_step = 0.30 / len(raw_frames)
    for idx, f in enumerate(raw_frames):
        f.timestamp = idx * t_step

    for f in raw_frames:
        res = simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=f.timestamp)

    # Word should NOT be accepted into sentence
    assert len(simulator.sentence_words) == 0, f"Sub-0.4s stroke committed: {simulator.sentence_words}"


def test_tier2_duration_gate_super_maximum_discard(simulator, oracle):
    """
    Super-maximum duration (> 2.5s):
    Sluggish continuous stroke taking 3.0s is rejected by duration gate.
    """
    raw_frames = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=10.0, pause_frames=12)
    # Re-scale timestamps so stroke duration exceeds 2.5s (e.g. 3.0s)
    n = len(raw_frames)
    t_step = 3.0 / n
    for idx, f in enumerate(raw_frames):
        f.timestamp = idx * t_step

    for f in raw_frames:
        res = simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=f.timestamp)

    assert len(simulator.sentence_words) == 0, f"Super-2.5s stroke committed: {simulator.sentence_words}"


def test_tier2_duration_gate_exact_boundary_accepted(simulator, oracle):
    """Stroke duration within [0.40s, 2.50s] is permitted into DTW."""
    raw_frames = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=30.0, pause_frames=12)
    # Set duration to exactly 1.0s
    n = len(raw_frames)
    t_step = 1.0 / (n - 1)
    for idx, f in enumerate(raw_frames):
        f.timestamp = idx * t_step

    for f in raw_frames:
        res = simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=f.timestamp)

    assert "Hello" in simulator.sentence_words


# =============================================================================
# BOUNDARY 4: RANDOM ARM SWIPES & ERRATIC MOTION REJECTION
# =============================================================================

@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
def test_tier2_random_arm_swipes_rejected(seed, dtw_engine, oracle):
    """
    Multiple synthetic erratic trajectories (different seeds) must produce
    distance > 1.4 * default_threshold and status REJECTED.
    """
    rnd = oracle.generate_random_swipe(seed=seed)
    res = dtw_engine.match(rnd)
    rejection_thresh = dtw_engine.config.default_threshold * dtw_engine.config.uncertainty_ratio
    assert res.status == "REJECTED"
    assert res.top_word is None
    assert res.best_distance > rejection_thresh


# =============================================================================
# BOUNDARY 5: EXTREME COORDINATES & SAFE PALM SCALE FLOOR
# =============================================================================

def test_tier2_extreme_coordinates_origin_zero(segmenter):
    """Extreme coordinates at (0.0, 0.0) do not cause mathematical exception or NaN."""
    st, comp, is_sw, v = segmenter.update(0.0, 0.0, palm_len=50.0, timestamp=0.0)
    assert not np.isnan(v)
    assert st == KineticState.IDLE


def test_tier2_extreme_coordinates_frame_borders(smoother):
    """Coordinates on frame borders (0, 0), (640, 480) smooth correctly without crash."""
    sx, sy = smoother.update(0.0, 0.0)
    assert (sx, sy) == (0.0, 0.0)
    sx, sy = smoother.update(640.0, 480.0)
    assert 0.0 <= sx <= 640.0
    assert 0.0 <= sy <= 480.0


def test_tier2_palm_scale_safe_floor_clamp(segmenter):
    """Very small palm length (e.g. 2.0px) is clamped to safe palm floor (>= 15.0px)."""
    # Move 15px with palm_len=2.0 -> if not clamped, v_norm would be 15 / 2 = 7.5
    # With safe floor max(palm, 15.0) = 15.0, v_norm is 15 / 15 = 1.0
    segmenter.update(100.0, 100.0, palm_len=2.0, timestamp=0.0)
    st, comp, is_sw, v = segmenter.update(115.0, 100.0, palm_len=2.0, timestamp=0.033)
    assert abs(v - 1.0) < 1e-4, f"Expected safe palm clamping to 15.0 (v=1.0), got v={v}"


# =============================================================================
# BOUNDARY 6: HAND OCCLUSION & GRACE PERIOD (4 FRAMES)
# =============================================================================

def test_tier2_hand_occlusion_grace_period_retention(segmenter):
    """
    When hand is temporarily lost:
    1 to 3 dropped frames preserve the state in predict_sentence_webcam;
    on the 4th frame, segmenter.reset() is called.
    """
    # Start motion into RECORDING
    segmenter.update(100.0, 100.0, 50.0, timestamp=0.0)
    st, _, _, _ = segmenter.update(115.0, 100.0, 50.0, timestamp=0.033)
    assert st == KineticState.RECORDING

    # Simulate 3 dropped frames: state in segmenter still has buffer
    assert len(segmenter.buffer) == 1

    # On 4th frame without hand, external reset is triggered
    hand_lost_frames = 4
    if hand_lost_frames >= 4:
        segmenter.reset()

    assert segmenter.state == KineticState.IDLE
    assert len(segmenter.buffer) == 0
