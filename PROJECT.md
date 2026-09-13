# Project: ASL Dynamic Gesture Recognition Pipeline

## Architecture
The system is a real-time 4-layer gesture recognition pipeline running on CPU:
- **Layer 1 (Perception)**: MediaPipe BlazeHand extracts wrist coordinates $(c_x, c_y)$ and palm length $\text{palm\_len}$ per frame.
- **Layer 2 (Segmentation & Anti-Trigger)**: 
  - Raw wrist coordinates filtered via Exponential Moving Average (EMA, $\alpha \approx 0.65$).
  - `KineticPauseSegmenter` (from `dynamic_dtw_engine.py`) monitors normalized velocity $v_{\text{norm}} = \Delta w / \text{palm\_len}$.
  - State machine transitions: `IDLE` $\to$ `RECORDING` $\to$ `PAUSE_DETECTED`.
  - Duration gate discards strokes outside $[0.4\text{s}, 2.5\text{s}]$.
  - Fast leftward motion triggers Swipe-Left (Backspace).
- **Layer 3 (DIP & Static Hand Shape & DTW Matching)**:
  - `extract_smart_hand_crop` + `process_lean_dip_tensor` (from `improved_dip_pipeline.py`) produce 3-channel DIP tensor.
  - `TwoStreamHandCNN` (from `train.py`) infers static hand shape class. To sustain $\ge 25$ FPS on CPU, static shape inference is interleaved (every 2nd frame during active motion).
  - Resampled trajectory ($K=32$) matched via `HybridDTWEngine` with Sakoe-Chiba band $r=4$ against `dynamic_templates.json`. Distance threshold rejection ensures non-matches output "Unknown".
- **Layer 4 (Sentence Builder & Audio-Visual Feedback)**:
  - Accumulates ASL words; silence $> 1.8\text{s}$ commits the sentence.
  - Large on-screen banner displays recognized English word and confidence percentage for $2.0\text{s}$.
  - Fading neon motion trail renders hand trajectory.
  - Background daemon thread running `pyttsx3` verbalizes the recognized English word without blocking the main webcam loop (max loop latency $< 50\text{ ms}$, maintaining $\ge 25\text{ FPS}$).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Non-blocking TTS Integration | Non-blocking daemon thread `pyttsx3` synthesis with lock guard, loop latency $< 50$ ms | M1 | R4, survey |
| 2 | Audio-Visual Feedback (Banner & Trail) | 2.0s prominent on-screen banner with confidence %; fading neon polyline motion trail | M1 | R4, survey |
| 3 | 3-Layer Anti-False-Trigger Guard | EMA wrist filter, duration gate $[0.4\text{s}, 2.5\text{s}]$, DTW distance threshold rejection | M1 | R3, survey |
| 4 | ASL English Vocabulary & Translation | Mapping Vietnamese template keys (`xin_chao`, `cam_on`, etc.) to ASL English words ("Hello", "Thank You", etc.) | M1 | R1, survey |
| 5 | Pipeline Layer 1-4 End-to-End Integration | Robust FSM handling (`completed_stroke is not None`), Swipe-Left backspace, silence sentence commit | M1 | R1, survey |
| 6 | Real-time CPU Performance ($\ge 25$ FPS) | Interleaved DIP+CNN inference to keep average CPU frame latency $< 40$ ms | M1 | Acceptance, survey |
| 7 | Template Tool Compatibility | Verification of `collect_dynamic_templates.py` schema v1.0 and template consistency | M1 | R5, survey |
| 8 | E2E Test Infra & Test Suites (Tiers 1-4) | Comprehensive test harness covering Tiers 1-4 and Verification Plan steps 1-5, publishing `TEST_READY.md` | M_TEST | Dual Track, survey |
| 9 | 100% E2E Pass & Tier 5 Hardening | Validate 100% pass on Tiers 1-4 and execute adversarial coverage hardening | M_FINAL | Dual Track, survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M_TEST | E2E Testing Track | Test harness and Tiers 1-4 test suite generation (Verification steps 1-5, synthetic oracles, publishing `TEST_READY.md`) | none | IN_PROGRESS |
| M1 | Pipeline Integration & Audio-Visual TTS | Refine `predict_sentence_webcam.py` with non-blocking TTS, anti-false-trigger guards, banner, trail, vocabulary mapping, and FPS interleaving | none | IN_PROGRESS |
| M_FINAL | Final Milestone: 100% E2E & Tier 5 Hardening | Verify 100% pass on E2E test suite, followed by Tier 5 adversarial stress testing | M_TEST, M1 | PLANNED |

## Interface Contracts

### `dynamic_dtw_engine.py` (STABLE / IMMUTABLE)
- **`KineticPauseSegmenter.update(wrist_x: float, wrist_y: float, palm_len: float, timestamp: float = None) -> (KineticState, Optional[List[Tuple[float, float, float, str]]], bool)`**:
  - `wrist_x`, `wrist_y`: raw (or smoothed) pixel coordinates.
  - Returns `(state, completed_stroke, is_swipe_left)`.
  - Note: When pause detected, `completed_stroke` contains the stroke points; `state` is immediately reset to `KineticState.IDLE`. Integrators MUST check `if completed_stroke is not None:`.
- **`resample_gesture_sequence(raw_points: List[Tuple[float, float, float, str]], target_k: int = 32) -> ResampledGesture`**:
  - Returns `ResampledGesture(trajectory: np.ndarray (32, 4), duration_sec: float, shape_sequence: List[str])`.
- **`HybridDTWEngine.match(query: ResampledGesture) -> DTWMatchResult`**:
  - Returns `DTWMatchResult(best_word: str, best_distance: float, status: 'ACCEPTED'|'UNCERTAIN'|'REJECTED', margin: float, shape_penalty: float)`.

### `improved_dip_pipeline.py` (STABLE / IMMUTABLE)
- **`extract_smart_hand_crop(frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool, bool]`**:
  - Returns `(hand_crop, (x1, y1, x2, y2), touching_border, is_flipped)`.
- **`process_lean_dip_tensor(hand_crop: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]`**:
  - Returns `(tensor_np (128, 128, 3) uint8 [CLAHE, AutoCanny, Sobel], canvas, rot_angle_deg)`.

### Non-blocking TTS Interface in `predict_sentence_webcam.py`
- Function / Class: `TTSAnnouncer` or `speak_async(text: str)`:
  - Thread-local SAPI5 engine initialization inside worker thread (`pythoncom.CoInitialize()` if needed, `pyttsx3.init()`, `say()`, `runAndWait()`, `del`).
  - Guarded by non-blocking lock (`lock.acquire(blocking=False)`).
  - Main thread dispatch latency $< 1\text{ ms}$, ensuring no webcam frame drops.

## Code Layout
- Immutable / Read-only:
  - `c:\DIP\DIP_project\dynamic_dtw_engine.py` (Core DTW & Segmenter)
  - `c:\DIP\DIP_project\improved_dip_pipeline.py` (DIP v4 engine)
  - `c:\DIP\DIP_project\dynamic_templates.json` (ASL template database)
  - `c:\DIP\DIP_project\train.py` (CNN model architecture)
  - `c:\DIP\DIP_project\vsl_dataset\models\best_model_v2.pth` (Model weights)
- Work File (Owned by M1 Worker):
  - `c:\DIP\DIP_project\predict_sentence_webcam.py`
- Test Files (Owned by M_TEST & M_FINAL Workers):
  - `c:\DIP\DIP_project\tests\`
  - `c:\DIP\DIP_project\run_e2e_tests.py`
  - `c:\DIP\DIP_project\TEST_INFRA.md`
  - `c:\DIP\DIP_project\TEST_READY.md`
