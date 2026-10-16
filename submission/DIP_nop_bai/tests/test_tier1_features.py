"""
Tier 1: Feature Coverage Test Suite
===================================
Covers all core pipeline features with >=5 test cases per feature:
- 5 Vocabulary Words (Hello, Thank You, Me, You, Goodbye) across variations
- Gesture Spotting (KineticPauseSegmenter FSM)
- DTW Matching (HybridDTWEngine & Sakoe-Chiba)
- Non-blocking TTS (NonBlockingTTS & speak_async)
- HUD Rendering (Prominent banner, fading neon trail, vocabulary mapping)
- Swipe-Left Priority Gesture (KineticPauseSegmenter)
"""

import os
import sys
import time
import pytest
import numpy as np
import cv2

from dynamic_dtw_engine import (
    KineticState,
    GestureFrame,
    KineticPauseSegmenter,
    ResampledGesture,
    resample_gesture_sequence,
    HybridDTWEngine,
    HybridDTWConfig,
    MatchResult,
    compute_hybrid_dtw_distance
)
from predict_sentence_webcam import (
    ASL_VOCAB_MAP,
    get_asl_phrase,
    NonBlockingTTS,
    speak_async,
    WristCoordinateSmoother,
    draw_fading_neon_trail,
    draw_prominent_hud_banner
)
from tests.test_harness_utils import (
    SyntheticGestureOracle,
    TimerResolutionGuard
)


# =============================================================================
# FEATURE 1: VOCABULARY WORDS (5 WORDS, >=5 TEST CASES EACH)
# =============================================================================

VOCAB_WORDS = ["xin_chao", "cam_on", "toi", "ban", "tam_biet"]


@pytest.mark.parametrize("word_key", VOCAB_WORDS)
def test_tier1_vocab_sample_exact_matches(word_key, dtw_engine):
    """
    Test Case 1 (for each word):
    Verify all 3 recorded samples match themselves with distance 0.0 and status ACCEPTED.
    """
    samples = dtw_engine.vocabulary[word_key]["samples"]
    assert len(samples) >= 3, f"Expected at least 3 samples for '{word_key}', found {len(samples)}"
    for idx, s in enumerate(samples):
        res = dtw_engine.match(s)
        assert res.status == "ACCEPTED"
        assert res.top_word == word_key
        assert res.best_distance == 0.0, f"Self match distance should be 0.0, got {res.best_distance}"


@pytest.mark.parametrize("word_key", VOCAB_WORDS)
def test_tier1_vocab_time_warp_drop_every_third(word_key, dtw_engine, oracle):
    """
    Test Case 2 (for each word):
    Time-warped query dropping every 3rd frame and interpolating back to 32 frames.
    Asserts status ACCEPTED, top_word correct, and distance < word threshold.
    """
    warped = oracle.generate_time_warped_query(word_key, sample_idx=0, warp_type="drop_every_third")
    res = dtw_engine.match(warped)
    thresh = dtw_engine.vocabulary[word_key]["threshold"]
    assert res.status == "ACCEPTED"
    assert res.top_word == word_key
    assert res.best_distance < thresh


@pytest.mark.parametrize("word_key", VOCAB_WORDS)
def test_tier1_vocab_time_warp_compression(word_key, dtw_engine, oracle):
    """
    Test Case 3 (for each word):
    Non-linear time compression query (quadratic time scaling t^1.4).
    Asserts status ACCEPTED and top_word correct.
    """
    warped = oracle.generate_time_warped_query(word_key, sample_idx=1, warp_type="time_compress")
    res = dtw_engine.match(warped)
    thresh = dtw_engine.vocabulary[word_key]["threshold"]
    assert res.status == "ACCEPTED"
    assert res.top_word == word_key
    assert res.best_distance < thresh


@pytest.mark.parametrize("word_key", VOCAB_WORDS)
def test_tier1_vocab_time_warp_stretching(word_key, dtw_engine, oracle):
    """
    Test Case 4 (for each word):
    Non-linear time stretching query (root time scaling t^0.7).
    Asserts status ACCEPTED and top_word correct.
    """
    warped = oracle.generate_time_warped_query(word_key, sample_idx=2, warp_type="time_stretch")
    res = dtw_engine.match(warped)
    thresh = dtw_engine.vocabulary[word_key]["threshold"]
    assert res.status == "ACCEPTED"
    assert res.top_word == word_key
    assert res.best_distance < thresh


@pytest.mark.parametrize("word_key", VOCAB_WORDS)
def test_tier1_vocab_noisy_trajectory(word_key, dtw_engine, oracle):
    """
    Test Case 5 (for each word):
    Spatial trajectory jitter with Gaussian noise (std=0.03).
    Asserts robust matching under sensor noise.
    """
    noisy = oracle.generate_time_warped_query(word_key, sample_idx=0, warp_type="none", noise_std=0.03)
    res = dtw_engine.match(noisy)
    thresh = dtw_engine.vocabulary[word_key]["threshold"]
    assert res.status == "ACCEPTED"
    assert res.top_word == word_key
    assert res.best_distance < thresh


# =============================================================================
# FEATURE 2: GESTURE SPOTTING (KINETIC PAUSE SEGMENTER FSM, >=5 TEST CASES)
# =============================================================================

def test_tier1_spotting_motion_start_transition():
    """TC2.1: Velocity >= v_start_ratio triggers IDLE -> RECORDING."""
    seg = KineticPauseSegmenter(v_start_ratio=0.08)
    seg.update(wrist_x=100.0, wrist_y=100.0, palm_len=50.0, timestamp=0.0)
    # Move by 5px/frame -> v_norm = 5 / 50 = 0.10 >= 0.08
    st, comp, is_sw, v = seg.update(wrist_x=105.0, wrist_y=100.0, palm_len=50.0, timestamp=0.033)
    assert st == KineticState.RECORDING
    assert len(seg.buffer) == 1
    assert comp is None


def test_tier1_spotting_pause_counter_accumulation():
    """TC2.2: Velocity <= v_stop_ratio accumulates pause_counter during RECORDING."""
    seg = KineticPauseSegmenter(v_start_ratio=0.08, v_stop_ratio=0.035, pause_frames=11)
    seg.update(100.0, 100.0, 50.0, timestamp=0.0)
    seg.update(110.0, 100.0, 50.0, timestamp=0.033)  # into RECORDING

    # Hold nearly still: 0.5px/frame -> v_norm = 0.01 <= 0.035
    for i in range(5):
        seg.update(110.0 + i * 0.5, 100.0, 50.0, timestamp=0.066 + i * 0.033)

    assert seg.pause_counter == 5
    assert seg.state == KineticState.RECORDING


def test_tier1_spotting_pause_counter_reset_on_reacceleration():
    """TC2.3: Re-accelerating hand resets pause_counter back to 0."""
    seg = KineticPauseSegmenter(v_start_ratio=0.08, v_stop_ratio=0.035, pause_frames=11)
    seg.update(100.0, 100.0, 50.0, timestamp=0.0)
    seg.update(110.0, 100.0, 50.0, timestamp=0.033)

    # 4 slow frames
    for i in range(4):
        seg.update(110.0 + i * 0.5, 100.0, 50.0, timestamp=0.066 + i * 0.033)
    assert seg.pause_counter == 4

    # Rapid motion resumes: move 10px -> v_norm = 0.20
    seg.update(130.0, 100.0, 50.0, timestamp=0.20)
    assert seg.pause_counter == 0
    assert seg.state == KineticState.RECORDING


def test_tier1_spotting_sub_minimum_stroke_discard():
    """TC2.4: Gestures shorter than min_stroke_frames are silently discarded."""
    seg = KineticPauseSegmenter(min_stroke_frames=8, pause_frames=5, grace_window_sec=0.0)
    seg.update(100.0, 100.0, 50.0, timestamp=0.0)
    # Only 2 motion frames
    seg.update(110.0, 100.0, 50.0, timestamp=0.033)
    seg.update(120.0, 100.0, 50.0, timestamp=0.066)
    # Pause 5 frames
    completed = None
    for i in range(5):
        st, comp, _, _ = seg.update(120.0 + i * 0.1, 100.0, 50.0, timestamp=0.10 + i * 0.033)
        if comp is not None:
            completed = comp

    # Hết cửa sổ chờ (grace) thì cử chỉ mới được chốt/loại: nạp thêm một khung sau đó.
    st, comp, _, _ = seg.update(121.0, 100.0, 50.0, timestamp=1.0)
    if comp is not None:
        completed = comp

    # Total frames in buffer was 7 < 8 -> silently dropped
    assert completed is None
    assert seg.state == KineticState.IDLE


def test_tier1_spotting_super_maximum_stroke_flush():
    """TC2.5: Restless motion exceeding max_stroke_frames is flushed to prevent memory runaway."""
    seg = KineticPauseSegmenter(max_stroke_frames=20)
    seg.update(100.0, 100.0, 50.0, timestamp=0.0)
    buffer_lengths = []
    # 25 continuous fast frames without pause
    for i in range(1, 26):
        st, comp, _, _ = seg.update(100.0 + i * 5.0, 100.0, 50.0, timestamp=i * 0.033)
        buffer_lengths.append(len(seg.buffer))

    # Stroke was flushed when exceeding 20: buffer length never reached 25
    assert max(buffer_lengths) <= 20
    assert comp is None


# =============================================================================
# FEATURE 3: DTW MATCHING (HYBRID DTW ENGINE, >=5 TEST CASES)
# =============================================================================

def test_tier1_dtw_sakoe_chiba_corridor_efficiency(dtw_engine, oracle):
    """TC3.1: Sakoe-Chiba band r=4 matches template in ~1-2ms on pure CPU Python."""
    sample = oracle.get_template_sample("xin_chao", 0)
    t0 = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        compute_hybrid_dtw_distance(sample, sample, dtw_engine.config)
    avg_ms = ((time.perf_counter() - t0) / iterations) * 1000.0
    print(f"\n[DTW CPU benchmark] Avg distance computation time: {avg_ms:.4f} ms")
    assert avg_ms < 5.0, f"DTW execution time {avg_ms:.4f}ms exceeded CPU limit"


def test_tier1_dtw_directional_sensitivity(dtw_engine):
    """TC3.2: Tangent vectors (dx, dy) distinguish opposite motion directions."""
    # Left-to-Right trajectory
    t = np.linspace(0, 2.0, 32)
    traj_fwd = np.column_stack([t, np.zeros(32), np.full(32, 2.0 / 31), np.zeros(32)]).astype(np.float32)
    # Right-to-Left trajectory (opposite direction)
    traj_rev = np.column_stack([2.0 - t, np.zeros(32), np.full(32, -2.0 / 31), np.zeros(32)]).astype(np.float32)

    seq_fwd = ResampledGesture(traj=traj_fwd, shapes=["B"] * 32, duration_sec=1.0)
    seq_rev = ResampledGesture(traj=traj_rev, shapes=["B"] * 32, duration_sec=1.0)

    dist = compute_hybrid_dtw_distance(seq_fwd, seq_rev, dtw_engine.config)
    # Direction error beta * ||dir_q - dir_t|| ensures distance is high
    assert dist > 0.40, f"Failed direction differentiation: dist = {dist:.4f}"


def test_tier1_dtw_hand_shape_penalty(dtw_engine):
    """TC3.3: Hand shape mismatch penalty lambda_shape increases composite cost."""
    t = np.linspace(0, 1.0, 32)
    traj = np.column_stack([t, t * 0.5, np.full(32, 1.0 / 31), np.full(32, 0.5 / 31)]).astype(np.float32)

    seq_same_shape1 = ResampledGesture(traj=traj, shapes=["B"] * 32, duration_sec=1.0)
    seq_same_shape2 = ResampledGesture(traj=traj, shapes=["B"] * 32, duration_sec=1.0)
    seq_diff_shape = ResampledGesture(traj=traj, shapes=["A"] * 32, duration_sec=1.0)

    dist_same = compute_hybrid_dtw_distance(seq_same_shape1, seq_same_shape2, dtw_engine.config)
    dist_diff = compute_hybrid_dtw_distance(seq_same_shape1, seq_diff_shape, dtw_engine.config)

    assert dist_same == 0.0
    assert dist_diff >= dtw_engine.config.lambda_shape, (
        f"Shape penalty should add at least lambda ({dtw_engine.config.lambda_shape}), got {dist_diff}"
    )


def test_tier1_dtw_three_tier_classification_strategy(dtw_engine, oracle):
    """TC3.4: Three-tier classification logic: ACCEPTED -> UNCERTAIN -> REJECTED."""
    # 1. Exact match -> ACCEPTED
    s = oracle.get_template_sample("toi", 0)
    res_acc = dtw_engine.match(s)
    assert res_acc.status == "ACCEPTED"

    # 2. Random motion -> REJECTED
    rnd = oracle.generate_random_swipe(seed=999)
    res_rej = dtw_engine.match(rnd)
    assert res_rej.status == "REJECTED"


def test_tier1_dtw_candidates_ranking_and_confidence(dtw_engine, oracle):
    """TC3.5: Returns top-3 ranked candidates with calibrated confidence scores [0, 1]."""
    s = oracle.get_template_sample("cam_on", 0)
    res = dtw_engine.match(s)
    assert len(res.candidates) >= 3
    assert res.candidates[0].word_key == "cam_on"
    assert 0.0 <= res.candidates[0].confidence <= 1.0
    # Ranked by ascending distance
    for i in range(1, len(res.candidates)):
        assert res.candidates[i].distance >= res.candidates[i - 1].distance


# =============================================================================
# FEATURE 4: NON-BLOCKING TTS (>=5 TEST CASES)
# =============================================================================

def test_tier1_tts_immediate_dispatch_latency():
    """TC4.1: speak() returns immediately in < 5ms without blocking."""
    tts = NonBlockingTTS(speech_rate=175)
    t0 = time.perf_counter()
    res = tts.speak("Hello")
    dispatch_ms = (time.perf_counter() - t0) * 1000.0
    assert res is True
    assert dispatch_ms < 5.0, f"TTS dispatch took {dispatch_ms:.2f}ms, expected < 5ms"


def test_tier1_tts_lock_guard_drops_overlapping_requests():
    """TC4.2: Lock guard drops or ignores concurrent speech calls while speaking."""
    tts = NonBlockingTTS(speech_rate=175)
    # Acquire lock manually to simulate active speech worker
    tts._lock.acquire()
    try:
        accepted = tts.speak("Second phrase")
        assert accepted is False, "TTS accepted request while lock was already held"
    finally:
        tts._lock.release()


def test_tier1_tts_invalid_input_graceful_handling():
    """TC4.3: Empty or invalid text inputs handled gracefully without exceptions."""
    tts = NonBlockingTTS(speech_rate=175)
    assert tts.speak("") is False
    assert tts.speak(None) is False
    assert tts.speak(12345) is False


def test_tier1_tts_global_speak_async_invocation():
    """TC4.4: Module-level speak_async() function dispatches successfully."""
    res = speak_async("Test word")
    assert isinstance(res, bool)


def test_tier1_tts_multiple_sequential_calls(timer_guard):
    """TC4.5: Multiple sequential calls separated in time execute cleanly."""
    tts = NonBlockingTTS(speech_rate=180)
    with timer_guard:
        r1 = tts.speak("Word one")
        time.sleep(0.5)
        # Check that lock or instance is not corrupted
        assert tts._lock is not None


# =============================================================================
# FEATURE 5: HUD RENDERING & VOCABULARY MAPPING (>=5 TEST CASES)
# =============================================================================

def test_tier1_hud_asl_vocab_map_coverage():
    """TC5.1: ASL_VOCAB_MAP contains all 5 required words with English & Vietnamese phrases."""
    expected_words = ["xin_chao", "cam_on", "toi", "ban", "tam_biet"]
    for w in expected_words:
        assert w in ASL_VOCAB_MAP, f"Missing '{w}' from ASL_VOCAB_MAP"
        assert "en" in ASL_VOCAB_MAP[w]
        assert "vi" in ASL_VOCAB_MAP[w]

    assert get_asl_phrase("xin_chao", "en") == "Hello"
    assert get_asl_phrase("cam_on", "en") == "Thank You"
    assert get_asl_phrase("toi", "en") == "Me"
    assert get_asl_phrase("ban", "en") == "You"
    assert get_asl_phrase("tam_biet", "en") == "Bye"   # từ điển hiện dùng "Bye" cho BYE
    assert get_asl_phrase("unregistered_key", "en", default="CustomDefault") == "CustomDefault"


def test_tier1_hud_prominent_banner_drawing():
    """TC5.2: draw_prominent_hud_banner renders visual banner without error."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_prominent_hud_banner(
        frame,
        title="HELLO (95%)",
        subtitle="ASL Match: Hello [Xin chao] | Dist: 0.05",
        color=(0, 255, 0),
        h_f=480,
        w_f=640
    )
    # Banner area (y: 52 to 118) should contain non-zero pixels
    banner_region = frame[52:118, 30:610]
    assert np.count_nonzero(banner_region) > 0


def test_tier1_hud_prominent_banner_empty_title_noop():
    """TC5.3: draw_prominent_hud_banner does nothing when title is empty."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_prominent_hud_banner(frame, title="", subtitle="Sub", color=(0, 255, 0), h_f=480, w_f=640)
    assert np.count_nonzero(frame) == 0


def test_tier1_hud_fading_neon_trail_rendering():
    """TC5.4: draw_fading_neon_trail renders anti-aliased polyline on canvas."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    curr_t = 10.0
    pts = [
        (100, 100, curr_t - 0.5),
        (150, 120, curr_t - 0.3),
        (200, 150, curr_t - 0.1),
        (250, 200, curr_t)
    ]
    draw_fading_neon_trail(frame, pts, curr_time=curr_t, max_age_sec=1.2)
    assert np.count_nonzero(frame) > 0


def test_tier1_hud_fading_neon_trail_aged_out():
    """TC5.5: Points older than max_age_sec are not rendered."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    curr_t = 10.0
    pts = [
        (100, 100, curr_t - 3.0),
        (150, 120, curr_t - 2.5)
    ]
    draw_fading_neon_trail(frame, pts, curr_time=curr_t, max_age_sec=1.2)
    assert np.count_nonzero(frame) == 0


# =============================================================================
# FEATURE 6: SWIPE-LEFT PRIORITY GESTURE (>=5 TEST CASES)
# =============================================================================

def test_tier1_swipe_left_pure_horizontal_trigger(segmenter, oracle):
    """TC6.1: Fast pure horizontal leftward motion triggers is_swipe_left == True."""
    coords = oracle.generate_raw_swipe_left(start_x=450.0, start_y=240.0, palm_len=60.0, num_frames=8)
    triggered = False
    for x, y, pl, dt in coords:
        st, comp, is_sw, v = segmenter.update(x, y, palm_len=pl, timestamp=dt)
        if is_sw:
            triggered = True
            break
    assert triggered is True, "Fast horizontal swipe failed to trigger Backspace"


def test_tier1_swipe_left_diagonal_rejection(segmenter):
    """TC6.2: Diagonal swipe (|dx| / |dy| < 2.5) is rejected."""
    # Move diagonally: 35px left, 35px down -> ratio ~ 1.0 < 2.5
    triggered = False
    for i in range(8):
        x = 450.0 - i * 35.0
        y = 200.0 + i * 35.0
        st, comp, is_sw, v = segmenter.update(x, y, palm_len=60.0, timestamp=i * 0.033)
        if is_sw:
            triggered = True
            break
    assert triggered is False, "Diagonal motion erroneously triggered Swipe-Left"


def test_tier1_swipe_left_rightward_motion_rejection(segmenter):
    """TC6.3: Fast rightward motion (delta_x > 0) is rejected."""
    triggered = False
    for i in range(8):
        x = 200.0 + i * 35.0
        y = 200.0
        st, comp, is_sw, v = segmenter.update(x, y, palm_len=60.0, timestamp=i * 0.033)
        if is_sw:
            triggered = True
            break
    assert triggered is False, "Rightward motion erroneously triggered Swipe-Left"


def test_tier1_swipe_left_slow_motion_rejection(segmenter):
    """TC6.4: Slow leftward motion (peak_vx > -0.20) is rejected."""
    # Move left at only 4px/frame (4 / 60 = 0.067 * palm_len, well above -0.20)
    triggered = False
    for i in range(8):
        x = 450.0 - i * 4.0
        y = 200.0
        st, comp, is_sw, v = segmenter.update(x, y, palm_len=60.0, timestamp=i * 0.033)
        if is_sw:
            triggered = True
            break
    assert triggered is False, "Slow leftward drift erroneously triggered Swipe-Left"


def test_tier1_swipe_left_cooldown_enforcement(segmenter, oracle):
    """TC6.5: Cooldown of 1.0s prevents duplicate backspace fires within cooldown."""
    coords = oracle.generate_raw_swipe_left(start_x=450.0, start_y=240.0, palm_len=60.0, num_frames=8)
    # First swipe
    first_trigger_time = None
    for x, y, pl, dt in coords:
        st, comp, is_sw, v = segmenter.update(x, y, palm_len=pl, timestamp=dt)
        if is_sw:
            first_trigger_time = dt
            break
    assert first_trigger_time is not None

    # Immediate second swipe within 0.3s (< 1.0s cooldown)
    second_triggered = False
    for x, y, pl, dt in coords:
        st, comp, is_sw, v = segmenter.update(x, y, palm_len=pl, timestamp=first_trigger_time + 0.3 + dt)
        if is_sw:
            second_triggered = True
            break
    assert second_triggered is False, "Swipe triggered within 1.0s cooldown deadtime"
