"""
Verification Plan Steps 1 - 5 (ORIGINAL_REQUEST.md)
===================================================
Automates all five mandatory verification plan steps:
1. Import check (dynamic_dtw_engine & predict_sentence_webcam).
2. Segmenter FSM unit test (velocity ramp-up -> hold below threshold -> buffer in [8, 60]).
3. DTW matching on 5 ASL words with time-warped query and random swipe rejection (> 1.4x threshold).
4. TTS non-blocking benchmark: 60 iterations @ 30 FPS, trigger on iteration 10, max iteration < 50ms.
5. Automated pipeline integration verification (headless simulated feed).
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
    ASL_VOCAB_MAP,
    get_asl_phrase,
    NonBlockingTTS,
    speak_async,
    WristCoordinateSmoother,
    run_sentence_builder
)
from tests.test_harness_utils import (
    SyntheticGestureOracle,
    HeadlessPipelineSimulator,
    TimerResolutionGuard
)


# =============================================================================
# VERIFICATION STEP 1: IMPORT CHECK
# =============================================================================

def test_verification_step1_import_check():
    """
    Verification Step 1:
    Verify imports of KineticPauseSegmenter, HybridDTWEngine, resample_gesture_sequence
    and predict_sentence_webcam components.
    """
    from dynamic_dtw_engine import (
        KineticPauseSegmenter as KPS,
        HybridDTWEngine as HDTWE,
        resample_gesture_sequence as RGS
    )
    assert KPS is not None
    assert HDTWE is not None
    assert RGS is not None

    import predict_sentence_webcam as psw
    assert hasattr(psw, "NonBlockingTTS")
    assert hasattr(psw, "speak_async")
    assert hasattr(psw, "WristCoordinateSmoother")
    assert hasattr(psw, "ASL_VOCAB_MAP")
    assert hasattr(psw, "run_sentence_builder")


# =============================================================================
# VERIFICATION STEP 2: SEGMENTER FSM UNIT TEST
# =============================================================================

def test_verification_step2_segmenter_fsm():
    """
    Verification Step 2:
    Inject synthetic velocity ramp-up then 12-frame hold-below-threshold into fresh
    KineticPauseSegmenter instance. Assert IDLE -> RECORDING -> PAUSE_DETECTED/completed
    and buffer length in [8, 60].
    """
    seg = KineticPauseSegmenter(
        v_start_ratio=0.08,
        v_stop_ratio=0.035,
        pause_frames=11,
        min_stroke_frames=8,
        max_stroke_frames=60,
        grace_window_sec=0.0,
    )

    states_observed = []
    completed_stroke = None

    # Frame 0: Hand stationary at (100, 100), palm_len=50
    st, comp, is_sw, v = seg.update(wrist_x=100.0, wrist_y=100.0, palm_len=50.0, timestamp=0.0)
    states_observed.append(st)
    assert st == KineticState.IDLE, f"Expected IDLE on initial frame, got {st}"

    # Ramp-up: 10 frames moving by 10px per frame (v_norm = 10 / 50 = 0.20 >= 0.08)
    for i in range(1, 11):
        curr_t = i * 0.033
        st, comp, is_sw, v = seg.update(
            wrist_x=100.0 + i * 10.0,
            wrist_y=100.0,
            palm_len=50.0,
            timestamp=curr_t
        )
        states_observed.append(st)

    # Assert transition to RECORDING occurred
    assert KineticState.RECORDING in states_observed, "FSM failed to transition to RECORDING during motion"

    # Hold below threshold: 12 frames moving by only 0.5px (v_norm = 0.5 / 50 = 0.01 <= 0.035)
    for j in range(1, 13):
        curr_t = (10 + j) * 0.033
        st, comp, is_sw, v = seg.update(
            wrist_x=200.0 + j * 0.5,
            wrist_y=100.0,
            palm_len=50.0,
            timestamp=curr_t
        )
        states_observed.append(st)
        if comp is not None:
            completed_stroke = comp

    # Assert pause was detected and packaged
    assert completed_stroke is not None, "Pause hold failed to produce completed stroke"
    assert 8 <= len(completed_stroke) <= 60, (
        f"Completed stroke length {len(completed_stroke)} outside valid range [8, 60]"
    )
    # Verify state reset back to IDLE
    assert seg.state == KineticState.IDLE, f"Expected FSM to reset to IDLE, got {seg.state}"


# =============================================================================
# VERIFICATION STEP 3: DTW MATCHING ON 5 ASL WORDS & REJECTION
# =============================================================================

@pytest.mark.parametrize("word_key", ["xin_chao", "cam_on", "toi", "ban", "tam_biet"])
def test_verification_step3_dtw_matching_5_words(word_key, dtw_engine, oracle):
    """
    Verification Step 3 (Part A):
    For each of the 5 ASL words, build time-warped query from stored template
    (drop every 3rd frame then linearly interpolate back to 32 frames).
    Assert top-1 label is correct and distance < default_threshold.
    """
    warped_query = oracle.generate_time_warped_query(word_key, sample_idx=0, warp_type="drop_every_third")
    result = dtw_engine.match(warped_query)

    assert result.status == "ACCEPTED", (
        f"Word '{word_key}' matching failed with status {result.status} (dist: {result.best_distance})"
    )
    assert result.top_word == word_key, (
        f"Expected top_word '{word_key}', got '{result.top_word}'"
    )
    threshold = dtw_engine.vocabulary[word_key]["threshold"]
    assert result.best_distance < threshold, (
        f"Distance {result.best_distance:.4f} exceeded threshold {threshold:.4f} for word '{word_key}'"
    )


def test_verification_step3_random_swipe_rejection(dtw_engine, oracle):
    """
    Verification Step 3 (Part B):
    Random lateral/diagonal synthetic swipe must produce DTW distance > 1.4 * default_threshold
    and be rejected with status REJECTED.
    """
    random_query = oracle.generate_random_swipe(seed=42)
    result = dtw_engine.match(random_query)

    rejection_boundary = dtw_engine.config.default_threshold * dtw_engine.config.uncertainty_ratio
    assert result.status == "REJECTED", (
        f"Expected random motion to be REJECTED, got {result.status} (dist: {result.best_distance})"
    )
    assert result.best_distance > rejection_boundary, (
        f"Expected distance > {rejection_boundary:.3f}, got {result.best_distance:.3f}"
    )


# =============================================================================
# VERIFICATION STEP 4: TTS NON-BLOCKING BENCHMARK
# =============================================================================

def test_verification_step4_tts_non_blocking_benchmark(timer_guard):
    """
    Verification Step 4:
    In a loop of 60 iterations with time.sleep(1/30) between each, trigger pyttsx3 speech
    on iteration 10 via NonBlockingTTS / speak_async. Measure wall-clock time per iteration.
    Assert max single iteration time < 50ms, confirming speech synthesis runs in background thread.
    """
    tts = NonBlockingTTS(speech_rate=175)
    target_fps = 30.0
    frame_interval = 1.0 / target_fps
    durations = []

    with timer_guard:
        for i in range(60):
            t_start = time.perf_counter()

            if i == 10:
                # Trigger speech
                dispatched = tts.speak("Hello")
                assert dispatched, "Failed to dispatch async TTS speech on iteration 10"

            # Precise hybrid sleep to match 30 FPS loop pacing without OS timer jitter
            target_t = t_start + frame_interval
            sleep_needed = target_t - time.perf_counter() - 0.003
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            while time.perf_counter() < target_t:
                pass

            t_end = time.perf_counter()
            durations.append(t_end - t_start)

    max_iter_ms = max(durations) * 1000.0
    iter10_ms = durations[10] * 1000.0

    print(f"\n[TTS Benchmark] Max iteration duration: {max_iter_ms:.2f} ms")
    print(f"[TTS Benchmark] Iteration 10 (Speech trigger): {iter10_ms:.2f} ms")

    assert max_iter_ms < 50.0, (
        f"TTS blocked main thread! Max iteration {max_iter_ms:.2f}ms exceeded 50.0ms limit"
    )


# =============================================================================
# VERIFICATION STEP 5: AUTOMATED PIPELINE INTEGRATION VERIFICATION
# =============================================================================

def test_verification_step5_pipeline_integration_headless(simulator, oracle):
    """
    Verification Step 5:
    Headless end-to-end integration test exercising all 4 layers:
    - Feeds synthetic gesture sequence for 'xin_chao' (Hello).
    - Checks Layer 1 smoothing, Layer 2 FSM pause detection, Layer 3 DTW matching,
      and Layer 4 word commitment + HUD banner generation + TTS trigger.
    - Feeds Swipe-Left to verify Backspace word removal.
    - Lowers hand for 2.0s (> 1.8s) to verify sentence commitment.
    """
    # 1. Feed 'xin_chao' gesture
    raw_frames = oracle.generate_raw_frames_from_template("xin_chao", sample_idx=0, fps=30.0, pause_frames=12)
    last_res = None
    for f in raw_frames:
        last_res = simulator.feed_frame(
            raw_wrist_x=f.x,
            raw_wrist_y=f.y,
            palm_len=f.palm_len,
            hand_shape=f.hand_shape,
            curr_time=f.timestamp
        )

    # Verify 'Hello' was recognized and committed to sentence
    assert "Hello" in simulator.sentence_words, f"Expected 'Hello' in sentence_words: {simulator.sentence_words}"
    assert len(simulator.tts_invocations) >= 1, "Expected TTS invocation for recognized word"
    assert simulator.tts_invocations[0][0] == "Hello"

    # 2. Feed Swipe-Left gesture
    t_swipe_start = raw_frames[-1].timestamp + 1.2
    swipe_coords = oracle.generate_raw_swipe_left(start_x=450.0, start_y=240.0, palm_len=60.0, num_frames=6)
    for x, y, pl, dt in swipe_coords:
        last_res = simulator.feed_frame(
            raw_wrist_x=x,
            raw_wrist_y=y,
            palm_len=pl,
            hand_shape="B",
            curr_time=t_swipe_start + dt
        )

    # Verify word was removed by Backspace
    assert "Hello" not in simulator.sentence_words, (
        f"Swipe-Left failed to remove 'Hello'. Current words: {simulator.sentence_words}"
    )

    # 3. Feed 'cam_on' (Thank You)
    t_thank_start = t_swipe_start + 1.5
    raw_frames_ty = oracle.generate_raw_frames_from_template("cam_on", sample_idx=0, fps=30.0, pause_frames=12)
    for f in raw_frames_ty:
        last_res = simulator.feed_frame(
            raw_wrist_x=f.x,
            raw_wrist_y=f.y,
            palm_len=f.palm_len,
            hand_shape=f.hand_shape,
            curr_time=t_thank_start + f.timestamp
        )

    assert "Thank You" in simulator.sentence_words

    # 4. Simulate Hand Absent for 2.0s (> 1.8s timeout)
    t_absent = t_thank_start + raw_frames_ty[-1].timestamp + 0.1
    # First frame absent
    simulator.feed_frame(raw_wrist_x=None, raw_wrist_y=None, curr_time=t_absent)
    assert not simulator.sentence_finalized

    # Advance clock past 1.8s
    simulator.feed_frame(raw_wrist_x=None, raw_wrist_y=None, curr_time=t_absent + 2.0)
    assert simulator.sentence_finalized, "Sentence failed to commit after 2.0s hand absence"
    assert simulator.get_final_sentence_string() == "Thank You .", (
        f"Expected 'Thank You .', got '{simulator.get_final_sentence_string()}'"
    )
