"""Kiểm thử collect_phrases.py (SPEC 09 nhiệm vụ C): kế hoạch, lược đồ, tiếp tục phiên, .discarded, đầu-cuối bằng video,
cờ khung trùng, và so khớp từng khung với collect_dynamic_templates.py."""
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import collect_phrases as cp  # noqa: E402

VOCAB13 = ["BAD", "BYE", "DRINK", "EAT", "HELLO", "HUNGRY", "ME", "PLEASE", "SORRY", "THANK_YOU", "WATER", "YES", "YOU"]
HAVE_MODEL = os.path.exists(os.path.join(ROOT, "hand_landmarker.task"))


@pytest.fixture(scope="module")
def plan():
    return cp.make_plan(VOCAB13, seed=cp.DEFAULT_SEED)


def _phrases(plan, s):
    return [c for c in plan["sessions"][s]["clips"] if c["kind"] == "phrase"]


def _idles(plan, s):
    return [c for c in plan["sessions"][s]["clips"] if c["kind"] == "idle"]


# ------------------------------------------------------------------ kế hoạch
def test_plan_sizes_and_constraints(plan):
    assert len(_phrases(plan, "S1")) == 40 and len(_idles(plan, "S1")) == 5
    assert len(_phrases(plan, "S2")) == 30 and len(_idles(plan, "S2")) == 5
    for s in ("S1", "S2"):
        for c in _phrases(plan, s):
            t = c["truth"]
            assert 2 <= len(t) <= 3
            assert all(w in VOCAB13 for w in t)
            assert all(t[i] != t[i + 1] for i in range(len(t) - 1)), f"lặp từ liền nhau: {t}"
        ids = [c["phrase_id"] for c in plan["sessions"][s]["clips"]]
        assert len(ids) == len(set(ids))
        for c in _idles(plan, s):
            assert c["truth"] == [] and c["duration_sec"] == 20.0 and c["activity"]
        # câu trong một phiên không trùng nhau
        sents = [tuple(c["truth"]) for c in _phrases(plan, s)]
        assert len(sents) == len(set(sents))


def test_plan_coverage_s1(plan):
    cov = plan["coverage"]["S1"]["per_word"]
    for w in VOCAB13:
        assert cov[w]["total"] >= 8, (w, cov[w])
        assert cov[w]["first"] >= 1 and cov[w]["last"] >= 1, (w, cov[w])
    assert plan["coverage"]["S1"]["min_met"]
    assert plan["coverage"]["S1"]["distinct_adjacent_pairs"] >= 40
    # đếm lại độc lập với bảng phủ đã lưu
    recount = {w: 0 for w in VOCAB13}
    for c in _phrases(plan, "S1"):
        for w in c["truth"]:
            recount[w] += 1
    assert all(recount[w] == cov[w]["total"] for w in VOCAB13)


def test_plan_sessions_disjoint(plan):
    a = {tuple(c["truth"]) for c in _phrases(plan, "S1")}
    b = {tuple(c["truth"]) for c in _phrases(plan, "S2")}
    assert not (a & b)
    assert plan["overlap_sentences_S1_S2"] == []


def test_plan_reproducible_and_seed_sensitive(plan):
    strip = lambda p: {k: v for k, v in p.items() if k != "created_at"}  # noqa: E731
    again = cp.make_plan(VOCAB13, seed=cp.DEFAULT_SEED)
    assert strip(again) == strip(plan)
    other = cp.make_plan(VOCAB13, seed=cp.DEFAULT_SEED + 7)
    assert strip(other) != strip(plan)


def test_make_plan_cli_and_backup(tmp_path, capsys):
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps({"vocabulary": {w: {} for w in VOCAB13}}), encoding="utf-8")
    out = tmp_path / "plan.json"
    assert cp.main(["--make-plan", "--templates", str(tpl), "--plan", str(out)]) == 0
    assert out.exists()
    txt = capsys.readouterr().out
    assert "BẢNG PHỦ S1" in txt and "BẢNG PHỦ S2" in txt
    # chạy lại cùng seed: không ghi lại, không backup
    cp.main(["--make-plan", "--templates", str(tpl), "--plan", str(out)])
    assert not glob.glob(str(tmp_path / "plan.backup_*"))
    # đổi seed: file cũ được sao lưu, không mất
    cp.main(["--make-plan", "--templates", str(tpl), "--plan", str(out), "--seed", "5"])
    assert len(glob.glob(str(tmp_path / "plan.backup_replan_*.json"))) == 1
    loaded = cp.load_plan(str(out))
    assert loaded["seed"] == 5


def test_load_plan_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        cp.load_plan(str(tmp_path / "khong_co.json"))


# ------------------------------------------------------------------ lược đồ, tiếp tục, .discarded
def _fake_frames(n=12, hand=True):
    fr = []
    for i in range(n):
        fr.append({
            "timestamp": 1000.0 + i / 30.0, "hand_present": hand and i % 5 != 4, "lk_fallback": False,
            "landmarks_norm": None, "landmarks_px": None, "palm_len": 50.0, "wrist_raw": [1.0, 2.0],
            "wrist_smoothed": [1.0, 2.0], "flexion_angles": [0.1] * 5, "palm_normal": [0.0, 0.0, -1.0],
            "hand_shape": "Unknown", "hand_shape_conf": None, "handedness": "Right", "face_box": None,
            "frame_is_new": i % 2 == 0, "luma_mean": 120.5, "frame_idx": i,
        })
    return fr


_META = {"session_id": "S1_test", "set": "S1", "camera_index": 0, "source": "camera", "frame_width": 640,
         "frame_height": 480, "flip_cam": False, "capture_fourcc": "MJPG", "capture_fps_reported": 30.0}
_CLIP = {"phrase_id": "S1_p01", "set": "S1", "kind": "phrase", "truth": ["HELLO", "YOU"], "text": "HELLO → YOU"}


def test_payload_roundtrip_schema(tmp_path):
    payload = cp.build_payload(_CLIP, _fake_frames(12), dict(_META))
    path = cp.write_clip(cp.clip_path(str(tmp_path), "S1_test", "S1_p01"), payload)
    assert path.endswith(os.path.join("S1_test", "S1_p01.json"))
    assert not os.path.exists(path + ".tmp")
    back = cp.read_clip(path)
    for k in ("schema_version", "session_id", "set", "phrase_id", "kind", "truth", "duration_sec", "num_frames",
              "frames", "capture_fourcc", "capture_fps_reported", "frame_width", "frame_height", "hand_frac",
              "new_frame_frac", "unique_fps_est"):
        assert k in back, k
    assert back["schema_version"] == cp.SCHEMA_VERSION == "phrase_raw_v1.0"
    assert back["kind"] == "phrase" and back["truth"] == ["HELLO", "YOU"] and back["set"] == "S1"
    assert back["num_frames"] == len(back["frames"]) == 12
    assert back["duration_sec"] == pytest.approx(11 / 30.0, abs=1e-5)
    assert back["num_new_frames"] == 6
    for f in back["frames"]:
        assert isinstance(f["frame_is_new"], bool) and isinstance(f["luma_mean"], float)
        assert isinstance(f["hand_present"], bool)


def test_resume_and_discard(tmp_path):
    clips = [dict(_CLIP, phrase_id=f"S1_p{i:02d}") for i in range(1, 6)]
    out = str(tmp_path)
    assert cp.pending_indices(clips, out, "S1_x") == [0, 1, 2, 3, 4]
    for i in (0, 1, 3):
        cp.write_clip(cp.clip_path(out, "S1_x", clips[i]["phrase_id"]),
                      cp.build_payload(clips[i], _fake_frames(4), dict(_META, session_id="S1_x")))
    assert cp.pending_indices(clips, out, "S1_x") == [2, 4]
    assert cp._next_pending(clips, out, "S1_x") == 2
    assert cp._next_pending(clips, out, "S1_x", start=3) == 4
    p = cp.clip_path(out, "S1_x", "S1_p04")
    before = open(p, "rb").read()
    tgt = cp.discard_clip(p)
    assert tgt == p + ".discarded" and not os.path.exists(p) and open(tgt, "rb").read() == before
    assert cp.pending_indices(clips, out, "S1_x") == [2, 3, 4]
    # thu lại rồi bỏ lần nữa: không ghi đè bản .discarded cũ
    cp.write_clip(p, cp.build_payload(clips[3], _fake_frames(5), dict(_META, session_id="S1_x")))
    tgt2 = cp.discard_clip(p)
    assert tgt2 == p + ".discarded2" and os.path.exists(tgt)
    assert cp.discard_clip(p) is None


def test_sealed_s2_no_replay_feature():
    assert "NIÊM PHONG" in cp.instructions_text("S2") and "KHÔNG mở xem" in cp.instructions_text("S2")
    assert "NIÊM PHONG" not in cp.instructions_text("S1")
    assert "LIỀN MẠCH" in cp.instructions_text("S1")
    names = [n.lower() for n in dir(cp)]
    assert not any(k in n for n in names for k in ("replay", "playback", "review_session", "view_clip", "show_clip"))


def test_cli_help_mentions_all_options():
    h = cp.build_parser().format_help()
    for opt in ("--make-plan", "--set", "--session", "--out-dir", "--cam", "--video", "--dry-run", "--max-frames",
                "--headless", "--seed"):
        assert opt in h


def test_hud_renders_all_states():
    hud = cp.Hud()
    base = {"set": "S1", "pos": 3, "total": 45, "n_done": 2, "kind": "phrase", "sealed": False,
            "text": "HELLO → YOU", "sub": "Xin chào · Bạn", "hint": "Ký liền mạch", "elapsed": 1.2, "count": 2,
            "landmarks_px": [[10 + i, 20 + i, 0.0] for i in range(21)], "hand_present": True, "new_pct": 55.0,
            "luma": 120.0, "warn": "low_light", "flash": "Đã lưu", "flash_color": (0, 255, 0), "flash_until": 1e12}
    for state in ("WAIT", "COUNTDOWN", "RECORDING", "DONE"):
        img = np.full((480, 640, 3), 90, np.uint8)
        out = hud.render(img, dict(base, state=state))
        assert out.shape == (480, 640, 3) and out.std() > 1.0


# ------------------------------------------------------------------ đầu-cuối bằng video (cần MediaPipe)
pytestmark_video = pytest.mark.skipif(not HAVE_MODEL, reason="thiếu hand_landmarker.task")


def _a_video():
    vids = sorted(glob.glob(os.path.join(ROOT, "ASL_Citizen_videos", "HELLO", "*.mp4")))
    if not vids:
        pytest.skip("không có video ASL_Citizen_videos/HELLO")
    return vids[0]


@pytestmark_video
def test_end_to_end_video_dry_and_real_and_resume(tmp_path):
    video = _a_video()
    plan_path = str(tmp_path / "plan.json")
    cp.write_plan(cp.make_plan(VOCAB13), plan_path)
    out = str(tmp_path / "raw")

    # 1) dry-run: không ghi file nào
    res = cp.run_collection(set_name="S1", video_path=video, dry_run=True, headless=True, plan_path=plan_path,
                            out_dir=out, max_frames=40)
    assert not os.path.exists(out) and len(res["payloads"]) == 1

    # 2) thật: ghi clip S1_p01 rồi (cùng session) clip kế S1_p02 — chứng minh tiếp tục phiên
    r1 = cp.run_collection(set_name="S1", session_id="S1_e2e", video_path=video, headless=True, plan_path=plan_path,
                           out_dir=out, max_frames=40)
    r2 = cp.run_collection(set_name="S1", session_id="S1_e2e", video_path=video, headless=True, plan_path=plan_path,
                           out_dir=out, max_frames=40)
    assert len(r1["saved"]) == 1 and len(r2["saved"]) == 1 and r1["saved"] != r2["saved"]
    files = sorted(os.listdir(os.path.join(out, "S1_e2e")))
    assert len(files) == 2 and files[0] != files[1], files
    assert not any(f.endswith(".tmp") for f in files)
    plan = cp.load_plan(plan_path)
    first_two = [c["phrase_id"] + ".json" for c in plan["sessions"]["S1"]["clips"][:2]]
    assert files == sorted(first_two)

    d = cp.read_clip(os.path.join(out, "S1_e2e", first_two[0]))
    assert d["schema_version"] == "phrase_raw_v1.0" and d["session_id"] == "S1_e2e" and d["set"] == "S1"
    assert d["num_frames"] == len(d["frames"]) == 40
    ts = [f["timestamp"] for f in d["frames"]]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    need = {"timestamp", "hand_present", "landmarks_norm", "landmarks_px", "palm_len", "wrist_raw", "wrist_smoothed",
            "flexion_angles", "palm_normal", "hand_shape", "handedness", "face_box", "frame_is_new", "luma_mean"}
    for f in d["frames"]:
        assert need <= set(f)
        assert isinstance(f["frame_is_new"], bool) and isinstance(f["luma_mean"], float)
        if f["hand_present"]:
            assert len(f["landmarks_px"]) == 21 and len(f["landmarks_px"][0]) == 3
            assert len(f["flexion_angles"]) == 5 and len(f["palm_normal"]) == 3 and len(f["wrist_smoothed"]) == 2
            assert f["handedness"] in ("Left", "Right")
        else:
            assert f["landmarks_px"] is None
    assert d["kind"] in ("phrase", "idle")
    assert d["truth"] == plan["sessions"]["S1"]["clips"][0]["truth"]


def _write_dup_video(path, pattern, w=320, h=240):
    import cv2
    rng = np.random.default_rng(0)
    uniq = {}
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
    for k in pattern:
        if k not in uniq:
            uniq[k] = np.clip(120 + rng.integers(-40, 41, (h, w, 3)), 0, 255).astype(np.uint8)
        vw.write(uniq[k])
    vw.release()


@pytestmark_video
def test_duplicate_frames_flagged_not_dropped(tmp_path):
    pattern = list("AABCCCDEEFGGHH")            # khung lặp lại đúng bộ đệm = khung trùng
    expected = [True] + [pattern[i] != pattern[i - 1] for i in range(1, len(pattern))]
    vpath = str(tmp_path / "dup.avi")
    _write_dup_video(vpath, pattern)
    res = cp.run_collection(set_name="S1", video_path=vpath, dry_run=True, headless=True, truth_override="HELLO",
                            plan_path=str(tmp_path / "khong_co_plan.json"), out_dir=str(tmp_path / "o"))
    fr = res["payloads"][0]["frames"]
    assert len(fr) == len(pattern)                    # khung trùng KHÔNG bị loại
    assert [f["frame_is_new"] for f in fr] == expected
    assert res["payloads"][0]["num_new_frames"] == sum(expected)
    assert all(f["luma_mean"] > 50 for f in fr)


# ------------------------------------------------------------------ so khớp từng khung với collector
@pytestmark_video
def test_frame_parity_with_collector(tmp_path):
    """Chạy collector (bản gốc, không sửa) và collect_phrases trên CÙNG video với dấu thời gian giả cố định;
    mọi trường từng khung phải giống hệt (sai khác 0). Chỉ bộ phân đoạn của collector bị thay bằng bộ giả chốt
    stroke đều đặn để nó chịu ghi khung thô — phần này không ảnh hưởng giá trị từng khung."""
    import cv2
    try:
        import collect_dynamic_templates as cdt
        from dynamic_dtw_engine import GestureFrame, KineticState
    except Exception as e:  # collector/engine đang bị sửa dở
        pytest.skip(f"không import được collector: {e}")
    video = _a_video()
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS) or 30.0
    T0 = 1_700_000_000.0
    captured = []

    class CapWriter:
        def enqueue(self, word_key, display_name, frame_records, meta, session_id=None):
            captured.append(list(frame_records))

        def flush_and_stop(self):
            pass

    class ForcedSeg:
        v_start_ratio = v_stop_ratio = pause_frames = min_stroke_frames = max_stroke_frames = 0
        grace_window_sec = swipe_threshold = swipe_ratio = swipe_cooldown_sec = swipe_max_buffer_frames = 0

        def __init__(self):
            self.buf, self.n = [], 0

        def reset(self):
            self.buf = []

        def update(self, wrist_x, wrist_y, palm_len, hand_shape, timestamp):
            self.buf.append(GestureFrame(wrist_x, wrist_y, palm_len, hand_shape, timestamp))
            self.n += 1
            if self.n % 20 == 0:
                return KineticState.RECORDING, list(self.buf[-20:]), False, 0.0
            return KineticState.RECORDING, None, False, 0.0

    Orig = cdt.ParallelHandDetector

    class FakeTs(Orig):
        def submit(self, seq, image_rgb, meta):
            meta["capture_ts"] = T0 + seq / fps
            return super().submit(seq, image_rgb, meta)

    saved = (cdt.ParallelHandDetector, cdt.BackgroundRawArchiveWriter, cdt.make_segmenter,
             cdt.is_movement_epenthesis, cdt.is_stroke_duration_valid)
    try:
        cdt.ParallelHandDetector, cdt.BackgroundRawArchiveWriter = FakeTs, CapWriter
        cdt.make_segmenter = lambda cfg: ForcedSeg()
        cdt.is_movement_epenthesis = lambda s: False
        cdt.is_stroke_duration_valid = lambda d: True
        tpl = str(tmp_path / "t.json")
        shutil.copy(os.path.join(ROOT, "webcam_templates.json"), tpl)
        cdt.run_dynamic_collector(video_path=video, templates_path=tpl, negative_path=str(tmp_path / "neg.json"),
                                  headless=True, session_id="parity")
    finally:
        (cdt.ParallelHandDetector, cdt.BackgroundRawArchiveWriter, cdt.make_segmenter,
         cdt.is_movement_epenthesis, cdt.is_stroke_duration_valid) = saved

    coll = [r for frames in captured for r in frames]
    assert len(coll) >= 10, "collector không ghi khung nào để so"
    res = cp.run_collection(set_name="S1", video_path=video, dry_run=True, headless=True, video_t0=T0,
                            truth_override="HELLO", plan_path=str(tmp_path / "np.json"), out_dir=str(tmp_path / "o"))
    mine = {round(f["timestamp"], 6): f for f in res["payloads"][0]["frames"]}
    for r in coll:
        m = mine[round(r["timestamp"], 6)]
        for k in ("landmarks_px", "landmarks_norm", "palm_len", "wrist_raw", "wrist_smoothed", "flexion_angles",
                  "palm_normal"):
            assert np.max(np.abs(np.asarray(r[k], float) - np.asarray(m[k], float))) == 0.0, k
        assert r["handedness"] == m["handedness"] and r["face_box"] == m["face_box"] and r["hand_shape"] == m["hand_shape"]


# ------------------------------------------------------------------ máy trạng thái phím (camera giả + phím theo kịch bản)
class _FakeCap:
    """Giống cv2.VideoCapture: khung nhiễu 320x240 phát ~30 khung/s theo giờ tường (không có tay)."""

    def __init__(self):
        import time
        self._t = time.time()
        self._n = 0
        self._rng = np.random.default_rng(3)

    def isOpened(self):
        return True

    def get(self, prop):
        return 30.0

    def release(self):
        pass

    def read(self):
        import time
        self._n += 1
        target = self._t + self._n / 30.0
        time.sleep(max(0.0, target - time.time()))
        return True, np.clip(100 + self._rng.integers(-30, 31, (240, 320, 3)), 0, 255).astype(np.uint8)


@pytestmark_video
def test_keyboard_state_machine_countdown_record_discard(tmp_path, monkeypatch):
    """Kịch bản: SPACE (đếm ngược 3 s, ghi) -> SPACE (kết thúc) => S1_p01.json; r => S1_p01 thành .discarded;
    SPACE ... SPACE thu lại S1_p01.json; SPACE ... SPACE thu S1_p02.json; q. Cửa sổ/phím được giả lập."""
    import time
    import cv2
    plan = {"schema_version": cp.PLAN_SCHEMA_VERSION, "sessions": {"S1": {"clips": [
        {"phrase_id": "S1_p01", "set": "S1", "kind": "phrase", "truth": ["HELLO", "YOU"], "text": "HELLO → YOU"},
        {"phrase_id": "S1_p02", "set": "S1", "kind": "phrase", "truth": ["ME", "HUNGRY"], "text": "ME → HUNGRY"},
    ]}}}
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(plan), encoding="utf-8")
    t0 = time.time()
    script = [(0.5, 32), (4.2, 32), (5.2, ord("r")), (5.8, 32), (9.5, 32), (10.3, 32), (14.0, 32), (15.0, ord("q"))]
    fired = set()

    def fake_waitkey(_ms):
        el = time.time() - t0
        for i, (t, k) in enumerate(script):
            if i not in fired and el >= t:
                fired.add(i)
                return k
        return 255

    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", fake_waitkey)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    out = str(tmp_path / "raw")
    res = cp.run_collection(set_name="S1", session_id="S1_keys", plan_path=str(pp), out_dir=out, capture=_FakeCap())
    files = sorted(os.listdir(os.path.join(out, "S1_keys")))
    assert files == ["S1_p01.json", "S1_p01.json.discarded", "S1_p02.json"], files
    d = cp.read_clip(os.path.join(out, "S1_keys", "S1_p01.json"))
    assert d["truth"] == ["HELLO", "YOU"] and d["num_frames"] >= 10
    # bản đầu (đã bỏ) ghi ~1 s sau đếm ngược (bấm 0.5 -> ghi từ 3.5 -> dừng 4.2): còn nguyên nội dung, không bị xóa
    old = cp.read_clip(os.path.join(out, "S1_keys", "S1_p01.json.discarded"))
    assert old["phrase_id"] == "S1_p01" and old["num_frames"] >= 2
    assert all(isinstance(f["frame_is_new"], bool) for f in d["frames"])
