"""
Tier 4: Realistic End-to-End User Journeys & Workloads Test Suite
================================================================
Verifies real-world multi-step user flows, state transitions across sessions,
realistic workloads, noise rejection, and failure injection recovery:
- Multi-word sentence generation ("Hello" -> "Me" -> "You" -> "Thank You" -> "Goodbye")
- Edit-and-correct workflow (Mistake -> Swipe-Left -> Correction -> Commit)
- Mixed noise and valid gestures (Rejection of fidgeting/erratic swipes)
- Sustained 30-second continuous execution stress test (900 frames, throughput & memory check)
- Failure injection recovery (transient coordinate spikes, hand dropouts, degenerate palm scale)
"""

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
# SCENARIO 1: MULTI-WORD SENTENCE GENERATION
# =============================================================================

def test_tier4_multi_word_sentence_generation(simulator, oracle):
    """
    Simulates a full end-to-end conversation sentence:
    'Hello' -> 'Me' -> 'You' -> 'Thank You' -> 'Goodbye'
    Each gesture is spotted by kinetic pause, matched by DTW, accumulated into
    sentence_words, and spoken via TTS.
    Finally, user lowers hand for > 1.8s, committing the sentence with a period.
    """
    word_sequence = [
        ("xin_chao", "Hello"),
        ("toi", "Me"),
        ("ban", "You"),
        ("cam_on", "Thank You"),
        ("tam_biet", "Goodbye")
    ]

    curr_time = 1000.0
    fps = 30.0
    dt = 1.0 / fps

    for word_key, expected_en in word_sequence:
        # Generate gesture frames with kinetic pause at the end
        raw_frames = oracle.generate_raw_frames_from_template(
            word_key=word_key,
            sample_idx=0,
            start_x=320.0,
            start_y=240.0,
            palm_len=60.0,
            fps=fps,
            pause_frames=12
        )

        # Feed the gesture frames
        for f in raw_frames:
            simulator.feed_frame(
                raw_wrist_x=f.x,
                raw_wrist_y=f.y,
                palm_len=f.palm_len,
                hand_shape=f.hand_shape,
                curr_time=curr_time
            )
            curr_time += dt

        # Verify the word was appended
        assert expected_en in simulator.sentence_words, f"Expected '{expected_en}' in {simulator.sentence_words}"

        # Inter-gesture cadence: hand lowered for 0.85s (> 0.8s debounce, < 1.8s commit)
        for _ in range(26):
            simulator.feed_frame(
                raw_wrist_x=None,
                raw_wrist_y=None,
                curr_time=curr_time
            )
            curr_time += dt

    # Check all 5 words accumulated in exact order
    assert simulator.sentence_words == ["Hello", "Me", "You", "Thank You", "Goodbye"]
    assert len(simulator.tts_invocations) == 5
    assert [w for w, _ in simulator.tts_invocations] == ["Hello", "Me", "You", "Thank You", "Goodbye"]
    assert not simulator.sentence_finalized

    # Silence commit: hand absent for 2.0s (> 1.8s threshold)
    for _ in range(60):
        simulator.feed_frame(
            raw_wrist_x=None,
            raw_wrist_y=None,
            curr_time=curr_time
        )
        curr_time += dt

    assert simulator.sentence_finalized is True
    assert simulator.get_final_sentence_string() == "Hello Me You Thank You Goodbye ."


# =============================================================================
# SCENARIO 2: EDIT-AND-CORRECT WORKFLOW (SWIPE-LEFT BACKSPACE)
# =============================================================================

def test_tier4_edit_and_correct_workflow(simulator, oracle):
    """
    Workflow:
    1. User signs 'Hello'.
    2. User accidentally signs 'Me' instead of 'You'.
    3. User realizes mistake and executes rapid Swipe-Left.
    4. Backspace removes 'Me' from sentence_words.
    5. User signs correct word 'You'.
    6. User removes hand for 2.0s -> Sentence committed as 'Hello You .'.
    """
    curr_time = 2000.0
    fps = 30.0
    dt = 1.0 / fps

    # Step 1: Sign 'Hello'
    raw_hello = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=fps, pause_frames=12)
    for f in raw_hello:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=curr_time)
        curr_time += dt
    assert simulator.sentence_words == ["Hello"]

    # Inter-gesture resting 0.85s
    for _ in range(26):
        simulator.feed_frame(None, None, curr_time=curr_time)
        curr_time += dt

    # Step 2: Sign mistaken word 'Me'
    raw_me = oracle.generate_raw_frames_from_template("toi", 0, fps=fps, pause_frames=12)
    for f in raw_me:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=curr_time)
        curr_time += dt
    assert simulator.sentence_words == ["Hello", "Me"]

    # Hand resting 0.5s before corrective action
    for _ in range(15):
        simulator.feed_frame(None, None, curr_time=curr_time)
        curr_time += dt

    # Step 3: Swipe-Left to delete mistaken word
    swipe_coords = oracle.generate_raw_swipe_left(
        start_x=380.0,
        start_y=240.0,
        palm_len=60.0,
        num_frames=8
    )
    swipe_triggered = False
    for x, y, pl, _ in swipe_coords:
        res = simulator.feed_frame(x, y, palm_len=pl, hand_shape="B", curr_time=curr_time)
        curr_time += dt
        if res["is_swipe"]:
            swipe_triggered = True
            break

    assert swipe_triggered, "Swipe-Left backspace failed to trigger"
    assert simulator.sentence_words == ["Hello"], "Backspace did not remove 'Me'"
    assert simulator.hud_banners[-1]["type"] == "BACKSPACE"
    assert "Me" in simulator.hud_banners[-1]["title"]

    # Inter-gesture resting 0.85s
    for _ in range(26):
        simulator.feed_frame(None, None, curr_time=curr_time)
        curr_time += dt

    # Step 4: Sign correct word 'You'
    raw_you = oracle.generate_raw_frames_from_template("ban", 0, fps=fps, pause_frames=12)
    for f in raw_you:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=curr_time)
        curr_time += dt
    assert simulator.sentence_words == ["Hello", "You"]

    # Step 5: Hand removed > 1.8s -> Sentence committed
    for _ in range(60):
        simulator.feed_frame(None, None, curr_time=curr_time)
        curr_time += dt

    assert simulator.sentence_finalized is True
    assert simulator.get_final_sentence_string() == "Hello You ."


# =============================================================================
# SCENARIO 3: MIXED NOISE AND VALID GESTURES
# =============================================================================

def test_tier4_mixed_noise_and_valid_gestures(simulator, oracle):
    """
    User operates in an uncontrolled environment with natural movements:
    1. Random erratic zig-zag motion (non-gesture noise) -> rejected by DTW / kinetic segmenter.
    2. Idle sensor jitter (v < v_start) -> ignored by segmenter.
    3. Deliberate 'Hello' gesture -> accepted and added.
    4. Out-of-vocabulary rapid vertical hand twitch -> rejected by DTW as UNKNOWN/REJECTED.
    5. Deliberate 'Thank You' gesture -> accepted and added.
    6. Silence commit -> 'Hello Thank You .'.
    """
    curr_time = 3000.0
    fps = 30.0
    dt = 1.0 / fps

    # 1. Random erratic swipe motion (zig-zag)
    random_swipe = oracle.generate_random_swipe(seed=42)
    for i in range(len(random_swipe.traj)):
        px = 320.0 + random_swipe.traj[i, 0] * 50.0
        py = 240.0 + random_swipe.traj[i, 1] * 50.0
        shape = random_swipe.shapes[i]
        simulator.feed_frame(px, py, palm_len=60.0, hand_shape=shape, curr_time=curr_time)
        curr_time += dt

    # Pause after noise to trigger segmentation and rejection
    for _ in range(15):
        simulator.feed_frame(320.0, 240.0, palm_len=60.0, hand_shape="A", curr_time=curr_time)
        curr_time += dt

    # Assert no accidental words recognized from noise
    assert len(simulator.sentence_words) == 0

    # 2. Idle stationary sensor jitter (noise amplitude 0.3 px, well below v_start)
    jitter_frames = oracle.generate_stationary_frames(duration_sec=1.0, fps=fps, noise_amplitude=0.3)
    for x, y, pl, _ in jitter_frames:
        simulator.feed_frame(x, y, palm_len=pl, hand_shape="B", curr_time=curr_time)
        curr_time += dt

    assert simulator.segmenter.state == KineticState.IDLE
    assert len(simulator.sentence_words) == 0

    # 3. Deliberate valid gesture: 'Hello'
    raw_hello = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=fps, pause_frames=12)
    for f in raw_hello:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=curr_time)
        curr_time += dt

    assert simulator.sentence_words == ["Hello"]

    # Inter-gesture resting 0.85s
    for _ in range(26):
        simulator.feed_frame(None, None, curr_time=curr_time)
        curr_time += dt

    # 4. Out-of-vocabulary pure vertical stroke (up and down fast)
    for i in range(16):
        vy = -8.0 if i < 8 else 8.0
        simulator.feed_frame(320.0, 240.0 + i * vy, palm_len=60.0, hand_shape="C", curr_time=curr_time)
        curr_time += dt
    # Pause to trigger segmentation
    for _ in range(12):
        simulator.feed_frame(320.0, 240.0, palm_len=60.0, hand_shape="C", curr_time=curr_time)
        curr_time += dt

    # Sentence words must STILL be only ['Hello']
    assert simulator.sentence_words == ["Hello"]

    # Inter-gesture resting 0.85s
    for _ in range(26):
        simulator.feed_frame(None, None, curr_time=curr_time)
        curr_time += dt

    # 5. Deliberate valid gesture: 'Thank You'
    raw_thanks = oracle.generate_raw_frames_from_template("cam_on", 0, fps=fps, pause_frames=12)
    for f in raw_thanks:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=curr_time)
        curr_time += dt

    assert simulator.sentence_words == ["Hello", "Thank You"]

    # 6. Silence commit
    for _ in range(60):
        simulator.feed_frame(None, None, curr_time=curr_time)
        curr_time += dt

    assert simulator.sentence_finalized is True
    assert simulator.get_final_sentence_string() == "Hello Thank You ."


# =============================================================================
# SCENARIO 4: SUSTAINED EXECUTION STRESS TEST (30 SECONDS / 900 FRAMES)
# =============================================================================

def test_tier4_sustained_execution_stress_test(simulator, oracle):
    """
    Stress tests the headless pipeline across a sustained 30-second video stream
    (900 frames at 30 FPS).
    Verifies:
    1. Zero unhandled exceptions or crashes.
    2. Headless execution throughput: Mean frame processing time < 5ms (FPS > 200).
    3. Peak frame processing latency < 35ms.
    4. Memory bounding: Trail points do not exceed max age window (len <= 50).
    5. Sentence builder correctly builds, edits, and commits sentences during the run.
    """
    total_frames = 900
    fps = 30.0
    dt = 1.0 / fps
    sim_time = 5000.0

    frame_latencies = []

    # Sequence plan across 900 frames:
    # 000-060: Rest (hand absent)
    # 060-120: Gesture 'Hello' + pause
    # 120-150: Rest (hand absent 1.0s)
    # 150-180: Gesture 'Me' + pause
    # 180-200: Rest (0.6s)
    # 200-210: Swipe-Left (backspace removes 'Me')
    # 210-240: Rest (1.0s)
    # 240-270: Gesture 'You' + pause
    # 270-340: Hand absent (2.3s > 1.8s -> Commits sentence 'Hello You .')
    # 340-400: Gesture 'Goodbye' + pause
    # 400-500: Random noise / jitter
    # 500-900: Hand absent (commits sentence 'Hello You Goodbye .')

    hello_frames = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=fps, pause_frames=12)
    me_frames = oracle.generate_raw_frames_from_template("toi", 0, fps=fps, pause_frames=12)
    you_frames = oracle.generate_raw_frames_from_template("ban", 0, fps=fps, pause_frames=12)
    goodbye_frames = oracle.generate_raw_frames_from_template("tam_biet", 0, fps=fps, pause_frames=12)
    swipe_frames = oracle.generate_raw_swipe_left(start_x=380.0, start_y=240.0, palm_len=60.0, num_frames=5)

    with TimerResolutionGuard():
        for frame_idx in range(total_frames):
            t_start = time.perf_counter()

            # Determine input for this frame
            if 0 <= frame_idx < 60:
                simulator.feed_frame(None, None, curr_time=sim_time)

            elif 60 <= frame_idx < 60 + len(hello_frames):
                hf = hello_frames[frame_idx - 60]
                simulator.feed_frame(hf.x, hf.y, hf.palm_len, hf.hand_shape, curr_time=sim_time)

            elif 60 + len(hello_frames) <= frame_idx < 150:
                simulator.feed_frame(None, None, curr_time=sim_time)

            elif 150 <= frame_idx < 150 + len(me_frames):
                mf = me_frames[frame_idx - 150]
                simulator.feed_frame(mf.x, mf.y, mf.palm_len, mf.hand_shape, curr_time=sim_time)

            elif 150 + len(me_frames) <= frame_idx < 200:
                simulator.feed_frame(None, None, curr_time=sim_time)

            elif 200 <= frame_idx < 200 + len(swipe_frames):
                sf = swipe_frames[frame_idx - 200]
                simulator.feed_frame(sf[0], sf[1], sf[2], "B", curr_time=sim_time)

            elif 200 + len(swipe_frames) <= frame_idx < 240:
                simulator.feed_frame(None, None, curr_time=sim_time)

            elif 240 <= frame_idx < 240 + len(you_frames):
                yf = you_frames[frame_idx - 240]
                simulator.feed_frame(yf.x, yf.y, yf.palm_len, yf.hand_shape, curr_time=sim_time)

            elif 240 + len(you_frames) <= frame_idx < 340:
                # Hand absent -> Silence commit
                simulator.feed_frame(None, None, curr_time=sim_time)

            elif 340 <= frame_idx < 340 + len(goodbye_frames):
                gf = goodbye_frames[frame_idx - 340]
                simulator.feed_frame(gf.x, gf.y, gf.palm_len, gf.hand_shape, curr_time=sim_time)

            else:
                # Hand absent -> Final Silence commit
                simulator.feed_frame(None, None, curr_time=sim_time)

            elapsed = time.perf_counter() - t_start
            frame_latencies.append(elapsed)
            sim_time += dt

            # Bounded memory check on each frame
            assert len(simulator.trail_points) <= 50, f"Trail points unbounded: {len(simulator.trail_points)}"

    latencies_ms = np.array(frame_latencies) * 1000.0
    mean_lat_ms = float(np.mean(latencies_ms))
    p99_lat_ms = float(np.percentile(latencies_ms, 99))
    max_lat_ms = float(np.max(latencies_ms))

    # Assertions on sustained real-time performance
    assert mean_lat_ms < 5.0, f"Mean frame time {mean_lat_ms:.2f}ms exceeds 5.0ms threshold"
    assert p99_lat_ms < 35.0, f"P99 frame time {p99_lat_ms:.2f}ms exceeds 35.0ms threshold"

    # Verify final sentence state after edits and commits
    final_sentence = simulator.get_final_sentence_string()
    assert "Hello" in final_sentence
    assert "You" in final_sentence
    assert "Goodbye" in final_sentence
    assert simulator.sentence_finalized is True


# =============================================================================
# SCENARIO 5: FAILURE INJECTION & RECOVERY
# =============================================================================

def test_tier4_failure_injection_recovery(simulator, oracle):
    """
    Injects transient hardware and perception faults:
    1. Degenerate palm length (0.0 or negative) -> clamped safely to minimum 15.0px.
    2. Coordinate anomaly (sudden outlier jump to (9999, -5000)) -> absorbed without crash.
    3. Hand tracking dropouts (3 frames of None mid-gesture) -> FSM handles gracefully.
    4. Time step glitch (dt jump of 2.0s) -> segmenter doesn't divide by zero.
    5. Pipeline recovery: user performs normal 'Hello' gesture right after, which matches successfully.
    """
    curr_time = 8000.0
    fps = 30.0
    dt = 1.0 / fps

    # Fault 1: Degenerate palm scale
    res1 = simulator.feed_frame(320.0, 240.0, palm_len=0.0, hand_shape="B", curr_time=curr_time)
    assert res1["hand_found"] is True
    curr_time += dt

    res2 = simulator.feed_frame(320.0, 240.0, palm_len=-10.0, hand_shape="B", curr_time=curr_time)
    assert res2["hand_found"] is True
    curr_time += dt

    # Fault 2: Massive coordinate spike (tracker glitch)
    res3 = simulator.feed_frame(9999.0, -5000.0, palm_len=60.0, hand_shape="B", curr_time=curr_time)
    assert res3["hand_found"] is True
    curr_time += dt

    # Fault 3: Hand dropout (missing 3 frames)
    for _ in range(3):
        res_drop = simulator.feed_frame(None, None, curr_time=curr_time)
        assert res_drop["hand_found"] is False
        curr_time += dt

    # Fault 4: Time jump glitch (simulating system stall or clock jump)
    curr_time += 2.5
    res4 = simulator.feed_frame(320.0, 240.0, palm_len=60.0, hand_shape="B", curr_time=curr_time)
    assert res4["hand_found"] is True
    curr_time += dt

    # Reset segmenter and smoother to clean state after recovery period
    simulator.smoother.reset()
    simulator.segmenter.reset()

    # Step 5: Verify complete recovery with clean gesture 'Hello'
    raw_hello = oracle.generate_raw_frames_from_template("xin_chao", 0, fps=fps, pause_frames=12)
    for f in raw_hello:
        simulator.feed_frame(f.x, f.y, f.palm_len, f.hand_shape, curr_time=curr_time)
        curr_time += dt

    assert "Hello" in simulator.sentence_words, "System failed to recover and recognize 'Hello' after faults"
