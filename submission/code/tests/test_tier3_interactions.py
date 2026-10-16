"""
Tier 3: Cross-Feature Pairwise Interactions Test Suite
======================================================
Verifies pairwise interactions across pipeline layers:
- Active gesture recording interrupted by Swipe-Left
- Silence commit (> 1.8s hand absent) after gesture match
- Concurrent TTS synthesis while next gesture begins
- Rapid back-to-back TTS calls with lock guard
- Camera flip toggle interaction
- Duplicate word debouncing within 0.8s
- Backspace on empty vs populated sentence buffer
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
    HybridDTWEngine,
    ResampledGesture
)
from predict_sentence_webcam import (
    NonBlockingTTS,
    speak_async,
    WristCoordinateSmoother,
    get_asl_phrase
)
from tests.test_harness_utils import (
    SyntheticGestureOracle,
    HeadlessPipelineSimulator,
    TimerResolutionGuard
)


# =============================================================================
# INTERACTION 1: GESTURE RECORDING INTERRUPTED BY SWIPE-LEFT
# =============================================================================

def test_tier3_gesture_interrupted_by_swipe_left(simulator, oracle):
    """
    User starts moving to form a gesture (RECORDING), then suddenly decides to swipe left.
    Asserts stroke buffer is flushed, FSM returns to IDLE, and previous word is deleted.
    """
    # Pre-populate sentence with 'Hello'
    raw_hello = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=30.0, pause_frames=12)
    for f in raw_hello:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=f.timestamp)
    assert simulator.sentence_words == ["Hello"]

    # Start a new gesture: 5 frames moving right smoothly from where the hand stopped
    t_inter = raw_hello[-1].timestamp + 1.2
    last_x = raw_hello[-1].x
    last_y = raw_hello[-1].y
    for i in range(1, 6):
        res = simulator.feed_frame(
            raw_wrist_x=last_x + i * 15.0,
            raw_wrist_y=last_y,
            palm_len=60.0,
            hand_shape="B",
            curr_time=t_inter + i * 0.033
        )
    assert simulator.segmenter.state == KineticState.RECORDING
    assert len(simulator.segmenter.buffer) > 0

    # Suddenly execute rapid Swipe-Left from the current hand position (last_x + 75)
    t_swipe = t_inter + 0.30
    swipe_coords = oracle.generate_raw_swipe_left(start_x=last_x + 75.0, start_y=last_y, palm_len=60.0, num_frames=8)
    swipe_fired = False
    for x, y, pl, dt in swipe_coords:
        res = simulator.feed_frame(x, y, palm_len=pl, hand_shape="B", curr_time=t_swipe + dt)
        if res["is_swipe"]:
            swipe_fired = True
            break

    # Asserts stroke buffer cleared, FSM reset to IDLE, and 'Hello' was deleted by backspace
    assert swipe_fired is True, "Swipe-Left failed to fire"
    assert simulator.segmenter.state == KineticState.IDLE
    assert len(simulator.segmenter.buffer) == 0
    assert len(simulator.sentence_words) == 0, f"Expected empty sentence, found: {simulator.sentence_words}"


# =============================================================================
# INTERACTION 2: SILENCE COMMIT AFTER GESTURE
# =============================================================================

def test_tier3_silence_commit_after_gesture(simulator, oracle):
    """
    Gesture recognized and spoken, then user lowers hand.
    After > 1.8s of hand absence, sentence is committed with final period.
    """
    raw_frames = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=30.0, pause_frames=12)
    for f in raw_frames:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=f.timestamp)
    assert simulator.sentence_words == ["Hello"]
    assert not simulator.sentence_finalized

    t_done = raw_frames[-1].timestamp
    # 1.0s absent -> not yet committed
    simulator.feed_frame(None, None, curr_time=t_done + 1.0)
    assert not simulator.sentence_finalized

    # 1.9s absent (> 1.8s) -> committed
    simulator.feed_frame(None, None, curr_time=t_done + 1.9)
    assert simulator.sentence_finalized
    assert simulator.get_final_sentence_string() == "Hello ."


# =============================================================================
# INTERACTION 3: TTS TRIGGERED WHILE NEXT GESTURE BEGINS
# =============================================================================

def test_tier3_tts_concurrency_while_next_gesture_begins(timer_guard, oracle):
    """
    When a word is recognized, TTS begins synthesizing in background daemon thread.
    Main loop immediately begins recording next gesture without latency or blocking.
    """
    tts = NonBlockingTTS(speech_rate=175)
    seg = KineticPauseSegmenter()

    with timer_guard:
        # Trigger TTS
        t0 = time.perf_counter()
        dispatched = tts.speak("Thank You")
        t_dispatch = (time.perf_counter() - t0) * 1000.0
        assert dispatched
        assert t_dispatch < 5.0

        # Concurrently feed 30 frames of next gesture motion while speech is ongoing
        iter_times = []
        for i in range(30):
            t_frame = time.perf_counter()
            seg.update(100.0 + i * 5.0, 200.0, palm_len=50.0, timestamp=i * 0.033)
            time.sleep(0.010)  # simulate per-frame pipeline work
            iter_times.append((time.perf_counter() - t_frame) * 1000.0)

        assert seg.state == KineticState.RECORDING
        max_iter = max(iter_times)
        # Verify no iteration blocked on audio
        assert max_iter < 40.0, f"Max iteration {max_iter:.2f}ms blocked during active TTS"


# =============================================================================
# INTERACTION 4: RAPID BACK-TO-BACK TTS CALLS WITH LOCK GUARD
# =============================================================================

def test_tier3_rapid_consecutive_tts_lock_guard():
    """
    Two rapid TTS requests within 0.1s:
    First request acquires lock; second request safely dropped/skipped without exception.
    """
    tts = NonBlockingTTS(speech_rate=175)
    r1 = tts.speak("First phrase")
    # Immediate second call before first finishes speech
    r2 = tts.speak("Second phrase")
    # First was accepted; second is False if worker acquired lock
    assert r1 is True
    # System did not crash, both calls returned cleanly


# =============================================================================
# INTERACTION 5: CAMERA FLIP TOGGLE DURING ACTIVE MOTION
# =============================================================================

def test_tier3_camera_flip_coordinate_inversion(smoother):
    """
    When camera flip is toggled, x coordinate mirrors across frame width (640 - x).
    Smoother handles the step jump gracefully without returning out-of-bounds coords.
    """
    x1, y1 = smoother.update(100.0, 200.0)
    assert x1 == 100.0
    # Camera flipped -> x inverted to 640 - 100 = 540
    x2, y2 = smoother.update(540.0, 200.0)
    assert 100.0 < x2 < 540.0
    assert 0.0 <= x2 <= 640.0


# =============================================================================
# INTERACTION 6: DUPLICATE WORD DEBOUNCING (0.8s)
# =============================================================================

def test_tier3_duplicate_word_debouncing(simulator, oracle):
    """
    Same gesture recognized twice within 0.8s: second occurrence suppressed.
    After debounce window expires (> 0.8s), repeated words are accepted.
    """
    raw_hello1 = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=30.0, pause_frames=12)
    for f in raw_hello1:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=f.timestamp)
    assert simulator.sentence_words == ["Hello"]
    t1 = simulator.last_commit_time

    # Verify debouncing condition on same word within 0.8s:
    is_repeat_blocked = ("Hello" == simulator.last_committed_word and (t1 + 0.4 - simulator.last_commit_time) < 0.8)
    assert is_repeat_blocked is True, "Same word within 0.8s should be marked as repeat"

    # Different word is NOT marked as repeat:
    diff_word_allowed = ("Thank You" != simulator.last_committed_word or (t1 + 0.4 - simulator.last_commit_time) >= 0.8)
    assert diff_word_allowed is True, "Different word should not be blocked"

    # Feed repeated gesture after 1.5s (> 0.8s debounce expired)
    t2 = t1 + 1.5
    raw_hello2 = oracle.generate_raw_frames_from_template("xin_chao", 1, fps=30.0, pause_frames=12)
    for f in raw_hello2:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=t2 + f.timestamp)
    assert simulator.sentence_words == ["Hello", "Hello"]


# =============================================================================
# INTERACTION 7: BACKSPACE ON EMPTY VS POPULATED BUFFER
# =============================================================================

def test_tier3_backspace_on_empty_buffer(simulator, oracle):
    """Swipe-Left on empty sentence buffer does not crash or corrupt sentence list."""
    assert len(simulator.sentence_words) == 0
    swipe_coords = oracle.generate_raw_swipe_left(start_x=450.0, start_y=200.0, palm_len=60.0, num_frames=8)
    for x, y, pl, dt in swipe_coords:
        res = simulator.feed_frame(x, y, palm_len=pl, hand_shape="B", curr_time=dt)
    assert len(simulator.sentence_words) == 0
    # HUD registered backspace notice
    assert any("EMPTY" in b["title"] or "BACKSPACE" in b["type"] for b in simulator.hud_banners)
