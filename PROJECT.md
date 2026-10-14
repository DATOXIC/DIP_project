# Project: ASL Dynamic Gesture Recognition Pipeline

## Architecture
> **Cập nhật 2026-10-14.** Mô tả dưới đây đã được đối chiếu lại với code. Hệ thống nhận diện từ bằng landmark MediaPipe + DTW 24D; CNN ảnh (DIP) **không** chạy trong vòng lặp từ động (`enable_static_shape_cnn = False`). Xem `docs/DE_CUONG_DO_AN_DIP_ASL.md` cho kiến trúc đầy đủ, `docs/TEST_READY.md` cho kiểm thử.
>
> **Từ vựng.** Bank sản xuất `webcam_templates.json` (chủ bank: Đạt, đóng băng) có **13 từ**: HELLO, BYE, ME, YOU, THANK_YOU, EAT, DRINK, WATER, SORRY, PLEASE, YES, BAD, HUNGRY. Năm từ quy ước 1 tay thu thêm trong hồ sơ `profiles/Phi.json`: **LIKE, NO, WANT, MY, NAME** (NICE, MEET đã có tên hiển thị và luật ghép câu nhưng **chưa có mẫu**). Bank demo `demo_templates.json` = bank Đạt (giữ nguyên) + mẫu của Phi cho 5 từ mới = **18 từ**; chạy bằng `py dip.py nhan-dien -- --cam 0 --spotter rule --templates demo_templates.json`. Các từ mới là ký hiệu **quy ước đơn giản hóa từ ASL** (nhiều từ vốn dùng 2 tay), không phải ASL chuẩn.
>
> **Đánh vần chữ cái.** Phím `s` trong cửa sổ nhận diện bật chế độ đánh vần (`fingerspell.py`) bằng bộ phân loại landmark `letter_landmark_model.pt`: **14 chữ** `A B C D E F G H I K L O S W` (model còn 2 lớp hình tay `1`, `5`). Tên ký xong vào câu dưới dạng token `SPELL:<TÊN>` và `sentence_composer.py` ghép thành "My name is …".

The system is a real-time 4-layer gesture recognition pipeline running on CPU:
- **Layer 1 (Perception)**: MediaPipe BlazeHand extracts wrist coordinates $(c_x, c_y)$ and palm length $\text{palm\_len}$ per frame.
- **Layer 2 (Segmentation & Anti-Trigger)**: 
  - Raw wrist coordinates filtered via Exponential Moving Average (EMA, $\alpha \approx 0.65$).
  - `KineticPauseSegmenter` (from `dynamic_dtw_engine.py`) monitors normalized velocity $v_{\text{norm}} = \Delta w / \text{palm\_len}$.
  - State machine transitions: `IDLE` $\to$ `RECORDING` $\to$ `PAUSE_DETECTED`.
  - Duration gate discards strokes outside $[0.4\text{s}, 2.5\text{s}]$.
  - Fast leftward motion triggers Swipe-Left (Backspace).
- **Layer 3 (DIP & Static Hand Shape & DTW Matching)**:
  - Trajectory features are 24D (`dynamic_features.py`), resampled to $K=32$ and matched via `GoldenTriadDTWMatcher` (Sakoe-Chiba band) against the active bank (`dynamic_templates.json`, `webcam_templates.json` or a per-person file in `profiles/`). Config: `configs/G1p90.json` (`global_k = 1.9`). Non-matches output `UNCERTAIN`/`REJECTED`.
  - Static hand shape is handled by landmarks (`handshape_gate.py`, `letter_landmark_model_image.pt`), not by the image CNN. `extract_smart_hand_crop` + `process_lean_dip_tensor` (`improved_dip_pipeline.py`) and `TwoStreamHandCNN` (`train.py`) are used for data collection / letter-training, and are disabled in the live word loop.
  - Legacy description (kept for history): CNN inference interleaved every 2nd frame with `HybridDTWEngine` 4D. No longer the production path.
- **Layer 4 (Sentence Builder & Audio-Visual Feedback)**:
  - Accumulates ASL words; silence $> 1.8\text{s}$ or 8 words commits the sentence; `sentence_composer.py` builds a bilingual natural sentence (SPEC 14), including `ME LIKE YOU`, `ME WANT WATER`, `ME NO LIKE YOU`, `MY NAME SPELL:<NAME>`, `NICE MEET YOU`.
  - Fingerspelling mode (key `s`, `fingerspell.py`): a letter is committed when the landmark classifier holds it stable for ~0.7 s; repeated letters need a hand drop or a shape change of ~0.35 s.
  - One-shot lock + spotter: with `--spotter dl` (default when `spotter_model.pth` exists) a stroke outside a "signing" span is blocked. The DL spotter was trained on the 13 original words, so new words can be blocked (`[OneShot] Chặn ... no_span`); use `--spotter rule` until the spotter is retrained on sessions containing the new words.
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
| 6 | Real-time CPU Performance ($\ge 25$ FPS) | Target. Landmark-only live loop (CNN off); FPS per machine in `results/camera_diag`, `results/fps_invariance_*` | M1 | Acceptance, survey |
| 7 | Template Tool Compatibility | Verification of `collect_dynamic_templates.py` schema v1.0 and template consistency | M1 | R5, survey |
| 8 | E2E Test Infra & Test Suites (Tiers 1-4) | Comprehensive test harness covering Tiers 1-4 and Verification Plan steps 1-5, publishing `TEST_READY.md` | M_TEST | Dual Track, survey |
| 9 | 100% E2E Pass & Tier 5 Hardening | Validate 100% pass on Tiers 1-4 and execute adversarial coverage hardening | M_FINAL | Dual Track, survey |
| 10 | Quick word teaching (`hieu-chinh`) | `py dip.py hieu-chinh --ho-so <tên> --tu <TỪ...> --so-mau N` teaches a new or existing word from N signed samples and saves it to `profiles/<tên>.json` (no training). The owner bank is never written. | post-M1 | `session_enroll.py`, `dip.py` |
| 11 | Fingerspelling mode | Key `s`: letters (14 supported) -> `SPELL:<NAME>` token -> composed into the sentence. Pure logic in `fingerspell.py`, tests in `tests/test_fingerspell.py`. | post-M1 | SPEC 15 (landmark fallback) |
| 12 | Extended gloss -> sentence rules | `LIKE`, `WANT` (+ negation with `NO`), `MY NAME <name>`, `NICE MEET YOU`, `NO` as an answer. | post-M1 | SPEC 14 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M_TEST | E2E Testing Track | Test harness and Tiers 1-4 test suite generation (Verification steps 1-5, synthetic oracles, publishing `TEST_READY.md`) | none | DONE (2026-10-14: ~455 pass, see `docs/TEST_READY.md`) |
| M1 | Pipeline Integration & Audio-Visual TTS | Refine `predict_sentence_webcam.py` with non-blocking TTS, anti-false-trigger guards, banner, trail, vocabulary mapping, and FPS interleaving | none | DONE (shipped; live test on strangers tracked in `KIEM_TRA_TRUC_TIEP.md`) |
| M_FINAL | Final Milestone: 100% E2E & Tier 5 Hardening | Verify 100% pass on E2E test suite, followed by Tier 5 adversarial stress testing | M_TEST, M1 | IN_PROGRESS (Tier 5 chưa làm; 1 test SAPI thật còn chập chờn) |

## Interface Contracts

### `dynamic_dtw_engine.py` (contains BOTH the legacy 4D engine and the real production 12D engine)
- **This file now houses two DTW engines side by side:**
  - **`HybridDTWEngine` (4D, DEPRECATED / LEGACY)**: the original engine, historically documented here as "STABLE/IMMUTABLE", but it is **not** what actually runs in production anymore. It is kept fully intact, unmodified in behavior, solely because `tests/` (`conftest.py` + `test_tier1-4*.py` + `test_verification_steps.py` + `test_harness_utils.py`) still imports and exercises it directly. Migrating those tests to the real engine is deferred to Phase 2 — do not delete this class.
  - **`GoldenTriadDTWMatcher` (12D "Golden Triad", REAL ENGINE IN PRODUCTION)**: this is the engine actually instantiated by `predict_sentence_webcam.py` and `collect_dynamic_templates.py`. It was moved here from `predict_sentence_webcam.py` so that all DTW matching logic lives in one place.
- **`KineticPauseSegmenter.update(wrist_x: float, wrist_y: float, palm_len: float, hand_shape: str = "Unknown", timestamp: float = None) -> Tuple[KineticState, Optional[List[GestureFrame]], bool, float]`**:
  - `wrist_x`, `wrist_y`: raw (or smoothed) pixel coordinates.
  - Returns `(state, completed_stroke, is_swipe_left, v_norm)`.
  - FSM is now 4-state: `IDLE -> RECORDING -> PENDING_COMMIT -> (RECORDING merge | IDLE commit)`. On a detected pause, the segmenter no longer commits immediately — it enters `PENDING_COMMIT` for a `grace_window_sec` (default 0.20s) to allow multi-phase signs to merge back into `RECORDING` before the stroke is actually committed and `completed_stroke` is populated. Integrators MUST check `if completed_stroke is not None:`.
  - Swipe-Left (Backspace) detection is now additionally gated by `swipe_max_buffer_frames` (default 10) so it cannot fire mid-way through a long, already-`RECORDING` real word.
- **`resample_gesture_sequence(raw_frames: List[GestureFrame], k_target: int = 32) -> Optional[ResampledGesture]`** *(legacy, 4D only)*:
  - Returns `ResampledGesture(traj: np.ndarray (32, 4), shapes: List[str], duration_sec: float)`.
- **`resample_9d_gesture_sequence(raw_frames_9d: np.ndarray, shapes: List[str], k_target: int = 32) -> Tuple[Optional[np.ndarray], Optional[List[str]]]`** *(real engine, 4D/9D/12D)*:
  - Temporal-normalizes a raw feature trajectory of D channels (D=4, 9, or 12) to K=32 frames.
- **`GoldenTriadDTWMatcher.match(query_traj_9d: np.ndarray, query_shapes: List[str]) -> MatchResult`** *(the contract actually in use)*:
  - Returns `MatchResult(status: 'ACCEPTED'|'UNCERTAIN'|'REJECTED', top_word, top_display, best_distance, best_confidence, candidates: List[MatchCandidate])`.
  - Decision is 3-tier (`ACCEPTED` if `distance <= threshold`; `REJECTED` if `distance > threshold * uncertainty_ratio`; `UNCERTAIN` otherwise), plus an **adaptive margin guard**: if the top-2 candidates are too close together (`margin_ratio < margin_threshold`, tightened to 8% automatically for legacy 4D templates, or to 4% when the top candidate is already very close to its own threshold), the result is downgraded to `UNCERTAIN` even if the distance alone would have passed.
- **`HybridDTWEngine.match(query: ResampledGesture) -> MatchResult`** *(legacy, 4D only — see deprecation note above)*:
  - Same `MatchResult` shape, computed via the legacy `compute_hybrid_dtw_distance`.

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
  - `dynamic_dtw_engine.py` (Core DTW & Segmenter — contains BOTH the legacy 4D `HybridDTWEngine` (deprecated, kept only for `tests/` compatibility pending Phase 2 migration) AND the real production 12D `GoldenTriadDTWMatcher`)
  - `improved_dip_pipeline.py` (DIP v4 engine)
  - `webcam_templates.json` (production bank of the project owner; frozen — never written by the recognizer)
  - `dynamic_templates.json`, `asl_citizen_templates.json` (older / ASL-Citizen template banks)
  - `train.py` (CNN model architecture)
  - `vsl_dataset/models/best_model_v2.pth` (Model weights)
- Work Files:
  - `predict_sentence_webcam.py` (live loop, sentence builder, fingerspelling key `s`)
  - `sentence_composer.py` (gloss -> natural sentence, SPEC 14), `fingerspell.py` (letter stability / word assembly)
  - `dip.py` + `dip_app.py` (single entry point: `nhan-dien`, `hieu-chinh`, `thu`, `kiem-tra`, ...)
  - `demo_templates.json` (demo bank, generated from the owner bank + Phi's new-word samples)
- Data of volunteers: `profiles/<tên>.json` (templates), `raw_volunteers/` (raw strokes), `letter_landmarks/` (letter samples).
- Tooling moved out of the root: `tools/` (evaluation, audit, e2e runner, letter live demo) and `legacy/` (superseded scripts such as `webcam.py`, `predict_webcam.py`, `collect_webcam.py`).
- Test Files:
  - `tests/`
  - `tools/run_e2e_tests.py`
  - `docs/TEST_INFRA.md`
  - `docs/TEST_READY.md`
