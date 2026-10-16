"""
Test Harness Utilities & Synthetic Oracles
==========================================
Provides deterministic synthetic gesture generation, oracles, timer resolution guards,
and headless pipeline simulation helpers for E2E testing.
"""

import os
import sys
import time
import math
import ctypes
from typing import List, Tuple, Optional, Dict, Any
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
    draw_fading_neon_trail,
    draw_prominent_hud_banner
)


class TimerResolutionGuard:
    """Context manager ensuring 1ms OS timer resolution on Windows during benchmarks."""
    def __enter__(self):
        if sys.platform == "win32":
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.platform == "win32":
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass


class SyntheticGestureOracle:
    """Authoritative generator for deterministic synthetic gesture queries and edge cases."""
    def __init__(self, templates_path: str = "dynamic_templates.json"):
        self.templates_path = templates_path
        self.dtw_engine = HybridDTWEngine(templates_path=templates_path)

    def get_template_sample(self, word_key: str, sample_idx: int = 0) -> ResampledGesture:
        """Retrieves a stored template sample for a word."""
        if word_key not in self.dtw_engine.vocabulary:
            raise KeyError(f"Word '{word_key}' not in templates vocabulary: {list(self.dtw_engine.vocabulary.keys())}")
        samples = self.dtw_engine.vocabulary[word_key]["samples"]
        if not samples:
            raise ValueError(f"No samples recorded for word '{word_key}'")
        return samples[sample_idx % len(samples)]

    def generate_time_warped_query(
        self,
        word_key: str,
        sample_idx: int = 0,
        warp_type: str = "drop_every_third",
        noise_std: float = 0.0
    ) -> ResampledGesture:
        """
        Creates a time-warped synthetic query from an authoritative template.
        Supports drop-and-interpolate, quadratic acceleration, and decel-warp.
        """
        tmpl = self.get_template_sample(word_key, sample_idx)
        orig_k = len(tmpl.traj)
        orig_t = np.linspace(0.0, 1.0, orig_k)

        if warp_type == "drop_every_third":
            keep_idx = [i for i in range(orig_k) if i % 3 != 0]
            sub_t = orig_t[keep_idx]
            sub_traj = tmpl.traj[keep_idx]
            sub_shapes = [tmpl.shapes[i] for i in keep_idx]
        elif warp_type == "drop_alternate":
            keep_idx = [i for i in range(orig_k) if i % 2 == 0]
            sub_t = orig_t[keep_idx]
            sub_traj = tmpl.traj[keep_idx]
            sub_shapes = [tmpl.shapes[i] for i in keep_idx]
        elif warp_type == "time_compress":
            # Warping time: t_warped = t^1.4 (slow start, rapid finish)
            sub_t = orig_t ** 1.4
            sub_traj = tmpl.traj
            sub_shapes = tmpl.shapes
        elif warp_type == "time_stretch":
            # Warping time: t_warped = t^0.7 (fast start, slow finish)
            sub_t = orig_t ** 0.7
            sub_traj = tmpl.traj
            sub_shapes = tmpl.shapes
        else:
            sub_t = orig_t
            sub_traj = tmpl.traj
            sub_shapes = tmpl.shapes

        target_t = np.linspace(0.0, 1.0, 32)
        new_traj = np.zeros((32, 4), dtype=np.float32)
        for c in range(4):
            new_traj[:, c] = np.interp(target_t, sub_t, sub_traj[:, c])

        if noise_std > 0.0:
            np.random.seed(42 + sample_idx)
            noise = np.random.normal(0.0, noise_std, new_traj.shape).astype(np.float32)
            new_traj[:, :2] += noise[:, :2]
            # Recalculate finite differences
            new_traj[1:, 2] = new_traj[1:, 0] - new_traj[:-1, 0]
            new_traj[1:, 3] = new_traj[1:, 1] - new_traj[:-1, 1]
            new_traj[0, 2] = new_traj[1, 2]
            new_traj[0, 3] = new_traj[1, 3]

        new_shapes = [
            sub_shapes[int(np.clip(np.round(t * (len(sub_shapes) - 1)), 0, len(sub_shapes) - 1))]
            for t in target_t
        ]

        return ResampledGesture(
            traj=new_traj,
            shapes=new_shapes,
            duration_sec=float(tmpl.duration_sec)
        )

    def generate_random_swipe(self, seed: int = 1234) -> ResampledGesture:
        """Generates an arbitrary erratic lateral/diagonal zig-zag motion to test rejection."""
        np.random.seed(seed)
        t = np.linspace(0.0, 4.0 * np.pi, 32)
        x = np.sin(t * 1.5) * 2.8 + np.cos(t * 3.0) * 1.2
        y = np.cos(t * 2.2) * 2.5 - np.sin(t * 0.8) * 1.5
        dx = np.zeros(32, dtype=np.float32)
        dy = np.zeros(32, dtype=np.float32)
        dx[1:] = x[1:] - x[:-1]
        dy[1:] = y[1:] - y[:-1]
        dx[0], dy[0] = dx[1], dy[1]
        traj = np.column_stack([x, y, dx, dy]).astype(np.float32)
        shapes = ['A' if i % 2 == 0 else 'C' for i in range(32)]
        return ResampledGesture(traj=traj, shapes=shapes, duration_sec=1.1)

    def generate_raw_frames_from_template(
        self,
        word_key: str,
        sample_idx: int = 0,
        start_x: float = 320.0,
        start_y: float = 240.0,
        palm_len: float = 60.0,
        fps: float = 30.0,
        stroke_frames: Optional[int] = None,
        pause_frames: int = 12
    ) -> List[GestureFrame]:
        """
        Converts a resampled template trajectory back into a series of raw GestureFrames
        that simulate a live user executing the gesture and then pausing to trigger segmentation.
        Dynamically scales the stroke duration to respect natural human execution velocity (v >= v_start).
        """
        tmpl = self.get_template_sample(word_key, sample_idx)
        raw_frames: List[GestureFrame] = []
        dt = 1.0 / fps
        curr_t = 0.0

        if stroke_frames is None:
            if word_key in ("xin_chao", "tam_biet"):
                stroke_frames = 32
            elif word_key == "cam_on":
                stroke_frames = 12
            else:  # "toi", "ban"
                stroke_frames = 8

        t_samples = np.linspace(0.0, 1.0, stroke_frames)
        for t in t_samples:
            idx = int(round(t * (len(tmpl.traj) - 1)))
            x_norm = tmpl.traj[idx, 0]
            y_norm = tmpl.traj[idx, 1]
            px = start_x + x_norm * palm_len
            py = start_y + y_norm * palm_len
            shape_idx = int(round(t * (len(tmpl.shapes) - 1))) if tmpl.shapes else 0
            shape = tmpl.shapes[shape_idx] if shape_idx < len(tmpl.shapes) else "B"
            raw_frames.append(GestureFrame(
                x=px, y=py, palm_len=palm_len, hand_shape=shape, timestamp=curr_t
            ))
            curr_t += dt

        # Append pause frames held below velocity threshold
        last_frame = raw_frames[-1]
        for p in range(pause_frames):
            raw_frames.append(GestureFrame(
                x=last_frame.x + 0.005 * p,
                y=last_frame.y,
                palm_len=palm_len,
                hand_shape=last_frame.hand_shape,
                timestamp=curr_t
            ))
            curr_t += dt

        return raw_frames

    def generate_raw_swipe_left(
        self,
        start_x: float = 450.0,
        start_y: float = 240.0,
        palm_len: float = 60.0,
        num_frames: int = 8,
        dt: float = 0.033
    ) -> List[Tuple[float, float, float, float]]:
        """
        Generates coordinate series satisfying Swipe-Left criteria:
        displacement <= -0.80 * palm_len, ratio >= 2.5, peak_vx <= -0.20 * palm_len.
        Returns list of (wrist_x, wrist_y, palm_len, timestamp).
        """
        frames = []
        # Rapid leftward motion: 35px (~0.58 * palm_len) shift per frame
        step_x = 35.0
        curr_t = 0.0
        for i in range(num_frames):
            x = start_x - i * step_x
            y = start_y  # Pure horizontal motion
            frames.append((x, y, palm_len, curr_t))
            curr_t += dt
        return frames

    def generate_stationary_frames(
        self,
        duration_sec: float = 1.0,
        fps: float = 30.0,
        wrist_x: float = 320.0,
        wrist_y: float = 240.0,
        palm_len: float = 50.0,
        noise_amplitude: float = 0.5
    ) -> List[Tuple[float, float, float, float]]:
        """Generates stationary hand frames with minimal sub-threshold sensor jitter."""
        num_frames = int(duration_sec * fps)
        dt = 1.0 / fps
        frames = []
        np.random.seed(999)
        for i in range(num_frames):
            jx = np.random.uniform(-noise_amplitude, noise_amplitude)
            jy = np.random.uniform(-noise_amplitude, noise_amplitude)
            frames.append((wrist_x + jx, wrist_y + jy, palm_len, i * dt))
        return frames


class HeadlessPipelineSimulator:
    """
    Simulates the full 4-layer sentence builder pipeline in memory:
    - Layer 1: Wrist smoothing
    - Layer 2: Kinetic segmentation & Swipe-Left detection
    - Layer 3: DTW matching & rejection
    - Layer 4: Sentence building, Backspace, Silence commit (1.8s), HUD, and TTS triggering
    """
    def __init__(self, templates_path: str = "dynamic_templates.json"):
        self.smoother = WristCoordinateSmoother(alpha=0.70)
        self.segmenter = KineticPauseSegmenter(
            v_start_ratio=0.08,
            v_stop_ratio=0.035,
            pause_frames=11,
            min_stroke_frames=8,
            max_stroke_frames=60,
            swipe_threshold=-0.80,
            swipe_ratio=2.5,
            swipe_cooldown_sec=1.0,
            grace_window_sec=0.0,   # test cũ mô phỏng chốt NGAY sau khi khựng (không chờ gộp đa pha)
        )
        self.dtw_engine = HybridDTWEngine(templates_path=templates_path)
        self.sentence_words: List[str] = []
        self.last_commit_time = 0.0
        self.last_committed_word: Optional[str] = None
        self.last_hand_seen_time = 0.0
        self.sentence_finalized = False
        self.tts_invocations: List[Tuple[str, float]] = []
        self.hud_banners: List[Dict[str, Any]] = []
        self.trail_points: List[Tuple[int, int, float]] = []

    def feed_frame(
        self,
        raw_wrist_x: Optional[float],
        raw_wrist_y: Optional[float],
        palm_len: float = 50.0,
        hand_shape: str = "B",
        curr_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Feeds a single time step into the pipeline simulator.
        If raw_wrist_x/y is None, simulates hand absent.
        """
        if curr_time is None:
            curr_time = time.time()

        result = {
            "hand_found": False,
            "seg_state": KineticState.IDLE,
            "is_swipe": False,
            "matched_word": None,
            "match_status": None,
            "sentence_words": list(self.sentence_words),
            "sentence_finalized": self.sentence_finalized,
            "tts_triggered": False
        }

        if raw_wrist_x is not None and raw_wrist_y is not None:
            result["hand_found"] = True
            self.last_hand_seen_time = curr_time
            self.sentence_finalized = False

            # Layer 1: EMA smoothing
            smooth_x, smooth_y = self.smoother.update(raw_wrist_x, raw_wrist_y)

            # Layer 2: KineticPauseSegmenter
            seg_state, completed_stroke, is_swipe, v_norm = self.segmenter.update(
                wrist_x=smooth_x,
                wrist_y=smooth_y,
                palm_len=palm_len,
                hand_shape=hand_shape,
                timestamp=curr_time
            )
            result["seg_state"] = seg_state
            result["is_swipe"] = is_swipe

            self.trail_points.append((int(smooth_x), int(smooth_y), curr_time))
            self.trail_points = [p for p in self.trail_points if (curr_time - p[2]) <= 1.2]

            # Layer 4: Swipe-Left Backspace
            if is_swipe:
                if len(self.sentence_words) > 0:
                    removed = self.sentence_words.pop()
                    self.sentence_finalized = False
                    self.hud_banners.append({
                        "title": f"<- BACKSPACE: -{removed}",
                        "type": "BACKSPACE",
                        "time": curr_time
                    })
                else:
                    self.hud_banners.append({
                        "title": "<- SENTENCE IS EMPTY",
                        "type": "EMPTY",
                        "time": curr_time
                    })
                self.trail_points.clear()

            # Layer 3: DTW matching
            if completed_stroke is not None:
                stroke_duration = completed_stroke[-1].timestamp - completed_stroke[0].timestamp
                # Layer 2/3: Duration Gate [0.4s, 2.5s]
                if 0.40 <= stroke_duration <= 2.50:
                    resampled = resample_gesture_sequence(completed_stroke, k_target=32)
                    if resampled is not None:
                        match_result = self.dtw_engine.match(resampled)
                        result["match_status"] = match_result.status

                        if match_result.status == "ACCEPTED" and match_result.top_word:
                            word_key = match_result.top_word
                            en_phrase = get_asl_phrase(word_key, lang="en", default=word_key)
                            result["matched_word"] = en_phrase

                            # Debounce repeat check (0.8s)
                            is_repeat = (en_phrase == self.last_committed_word and (curr_time - self.last_commit_time) < 0.8)
                            if not is_repeat:
                                self.sentence_words.append(en_phrase)
                                self.last_commit_time = curr_time
                                self.last_committed_word = en_phrase
                                self.sentence_finalized = False

                                thresh = self.dtw_engine.vocabulary[word_key]["threshold"]
                                conf = max(0.0, min(100.0, (1.0 - (match_result.best_distance / (thresh + 1e-6))) * 100.0))
                                self.hud_banners.append({
                                    "title": f"{en_phrase} ({conf:.0f}%)",
                                    "type": "ACCEPTED",
                                    "time": curr_time
                                })
                                self.tts_invocations.append((en_phrase, curr_time))
                                result["tts_triggered"] = True

                        elif match_result.status == "UNCERTAIN":
                            self.hud_banners.append({
                                "title": "? UNCERTAIN",
                                "type": "UNCERTAIN",
                                "time": curr_time
                            })
                        else:
                            self.hud_banners.append({
                                "title": "X UNKNOWN GESTURE",
                                "type": "REJECTED",
                                "time": curr_time
                            })
                else:
                    result["duration_gated"] = True

        else:
            # Hand absent
            if len(self.sentence_words) > 0 and (curr_time - self.last_hand_seen_time) > 1.8 and not self.sentence_finalized:
                self.sentence_finalized = True
                self.hud_banners.append({
                    "title": "SENTENCE COMMITTED",
                    "type": "COMMIT",
                    "time": curr_time
                })

        result["sentence_words"] = list(self.sentence_words)
        result["sentence_finalized"] = self.sentence_finalized
        return result

    def get_final_sentence_string(self) -> str:
        """Returns the assembled sentence string, adding final period if committed."""
        if not self.sentence_words:
            return ""
        s = " ".join(self.sentence_words)
        if self.sentence_finalized:
            s += " ."
        return s
