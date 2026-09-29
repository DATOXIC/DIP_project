"""
TEST cho cụm cải tiến "không cần thu thêm dữ liệu":
  - augment_templates (SPEC 06A, chế độ A)
  - dynamic_dtw_engine: độ rộng nội lớp chỉ tính trên mẫu GỐC; width_blend (mặc định 0 = hành vi cũ)
  - KineticPauseSegmenter: chuẩn hóa theo thời gian (mặc định TẮT)
Chạy nhanh, không cần webcam, không đụng dữ liệu thật.
"""

import copy
import os

import numpy as np
import pytest

from dynamic_dtw_engine import GoldenTriadDTWMatcher, KineticPauseSegmenter, KineticState
from pipeline_common import SHARED_CONFIG, make_segmenter


# ============================================================================ helpers
def _hand_template() -> np.ndarray:
    """21 landmark (px) có hình bàn tay đủ để tính palm_len/góc: cổ tay gốc, các ngón tỏa lên."""
    lm = np.zeros((21, 3), dtype=np.float64)
    for f in range(5):
        base_x = (f - 2) * 22.0
        for j in range(4):
            idx = 1 + f * 4 + j
            lm[idx] = [base_x * (1 + 0.15 * j), -(60.0 + 28.0 * j), 3.0 * j]
    lm[9] = [0.0, -100.0, 0.0]        # gốc ngón giữa: palm_2d ~ 100 px
    return lm


def _synthetic_raw(session="phien_test", word="HELLO", n=24):
    """Stroke thô giả: cổ tay đi cung tròn (không thẳng, không bị loại bởi epenthesis), tay cố định hình dạng."""
    tpl = _hand_template()
    frames = []
    for i in range(n):
        a = 1.2 * np.pi * i / (n - 1)
        wx, wy = 300 + 120 * np.cos(a), 300 + 120 * np.sin(a)
        lm = tpl.copy()
        lm[:, 0] += wx
        lm[:, 1] += wy
        frames.append({
            "timestamp": 1000.0 + i * 0.05,
            "landmarks_px": lm.tolist(),
            "wrist_smoothed": [float(wx), float(wy)],
            "hand_shape": "Unknown",
            "handedness": "Left",
            "face_box": [300.0, 150.0, 90.0, 90.0],
        })
    return {"schema_version": "raw_stroke_v1.1", "session_id": session, "word_key": word,
            "stroke_duration_sec": 1.15, "frames": frames}


def _bank_from_raw(raw):
    from rebuild_templates_from_raw import rebuild_one_stroke
    s = rebuild_one_stroke(raw)
    assert s is not None
    keep = {k: s[k] for k in ("traj_features", "hand_shapes", "duration_sec", "handedness", "chirality_canonical",
                              "session_id", "rep_count", "body_anchor_valid")}
    return {"metadata": {"k_resample_frames": 32, "feature_dim": 24}, "vocabulary": {raw["word_key"]: {"display_name": "Hello", "samples": [keep]}}}


# ============================================================================ augment_templates
class TestAugment:
    def test_bien_the_thua_huong_va_khong_lat_guong(self):
        from augment_templates import build_augmented_bank
        raw = _synthetic_raw()
        bank = _bank_from_raw(raw)
        out, st = build_augmented_bank(bank, [("x.json", raw)], seed=7, strength=1.0)
        samples = out["vocabulary"]["HELLO"]["samples"]
        orig = [s for s in samples if s["is_original"]]
        vars_ = [s for s in samples if not s["is_original"]]
        assert len(orig) == 1 and st["originals_with_raw"] == 1 and len(vars_) >= 5
        oid = orig[0]["sample_id"]
        for v in vars_:
            assert v["augment_of"] == oid and v["session_id"] == "phien_test"
            assert v["handedness"] == "Left" and v["chirality_canonical"] is True     # không lật gương
            assert np.array(v["traj_features"]).shape == (32, 24)
            assert 0.40 <= v["duration_sec"] <= 2.50

    def test_tai_lap_theo_seed(self):
        from augment_templates import build_augmented_bank
        raw = _synthetic_raw()
        bank = _bank_from_raw(raw)
        a, _ = build_augmented_bank(bank, [("x.json", raw)], seed=3)
        b, _ = build_augmented_bank(bank, [("x.json", raw)], seed=3)
        c, _ = build_augmented_bank(bank, [("x.json", raw)], seed=4)
        ta = [s["traj_features"] for s in a["vocabulary"]["HELLO"]["samples"]]
        tb = [s["traj_features"] for s in b["vocabulary"]["HELLO"]["samples"]]
        tc = [s["traj_features"] for s in c["vocabulary"]["HELLO"]["samples"]]
        assert ta == tb and ta != tc

    def test_strength_0_la_phep_dong_nhat(self):
        from augment_templates import build_augmented_bank
        raw = _synthetic_raw()
        bank = _bank_from_raw(raw)
        out, _ = build_augmented_bank(bank, [("x.json", raw)], seed=1, strength=0.0)
        samples = out["vocabulary"]["HELLO"]["samples"]
        base = np.array(samples[0]["traj_features"])
        for v in samples[1:]:
            assert np.allclose(np.array(v["traj_features"]), base, atol=2e-3), v["augment_kind"]

    def test_chi_sinh_bien_the_cho_mau_co_stroke_tho(self):
        from augment_templates import build_augmented_bank
        raw = _synthetic_raw()
        bank = _bank_from_raw(raw)
        other = copy.deepcopy(bank["vocabulary"]["HELLO"]["samples"][0])
        other["session_id"] = "phien_khac"                      # mẫu bank KHÔNG có stroke thô tương ứng
        bank["vocabulary"]["HELLO"]["samples"].append(other)
        out, st = build_augmented_bank(bank, [("x.json", raw)], seed=1)
        assert st["originals"] == 2 and st["originals_with_raw"] == 1
        assert all(s.get("augment_of") != "HELLO__phien_khac__b1" for s in out["vocabulary"]["HELLO"]["samples"])

    def test_tu_choi_ghi_de_file_du_lieu_that(self, monkeypatch):
        import augment_templates
        monkeypatch.setattr("sys.argv", ["augment_templates.py", "--out", "webcam_templates.json"])
        with pytest.raises(SystemExit) as e:
            augment_templates.main()
        assert e.value.code == 2


# ============================================================================ engine: độ rộng nội lớp + width_blend
def _traj(seed):
    rng = np.random.default_rng(seed)
    t = np.zeros((32, 24), dtype=np.float32)
    t[:, 0] = np.linspace(0, 1, 32) + rng.normal(0, 0.02, 32)
    t[:, 1] = np.sin(np.linspace(0, 3, 32)) + rng.normal(0, 0.02, 32)
    t[:, 4:9] = rng.normal(0.5, 0.05, (32, 5))
    return t


def _mk_bank(with_variants: bool):
    vocab = {}
    for w, seeds in {"A": [1, 2, 3, 4], "B": [11, 12, 13, 14, 15]}.items():
        samples = [{"traj_features": _traj(s).tolist(), "hand_shapes": ["Unknown"] * 32, "session_id": f"s{s}",
                    "chirality_canonical": True, "handedness": "Right", "rep_count": 1, "body_anchor_valid": True} for s in seeds]
        if with_variants:
            for i, s in enumerate(list(samples)):
                v = copy.deepcopy(s)
                v["traj_features"] = (np.array(v["traj_features"]) + 0.001).tolist()
                v["is_original"] = False
                v["augment_of"] = f"{w}{i}"
                samples.append(v)
        vocab[w] = {"display_name": w, "samples": samples}
    return {"metadata": {"k_resample_frames": 32, "feature_dim": 24}, "vocabulary": vocab}


def _matcher(bank, tmp_path, **kw):
    return GoldenTriadDTWMatcher(sign_weights_path="", w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8, w_b=0.0, templates_dict=bank,
                                 threshold_mode="scale_k", threshold_k=1.75, uncertainty_k=2.35,
                                 cache_dir=str(tmp_path / "cache"), **kw)


class TestEngine:
    def test_do_rong_noi_lop_bo_qua_bien_the(self, tmp_path):
        a = _matcher(_mk_bank(False), tmp_path / "a")
        b = _matcher(_mk_bank(True), tmp_path / "b")
        assert a.intra_widths == b.intra_widths and a.r_typical == b.r_typical

    def test_width_blend_0_giu_nguyen_ket_qua(self, tmp_path):
        bank = _mk_bank(False)
        m0 = _matcher(bank, tmp_path / "m0")
        m1 = _matcher(bank, tmp_path / "m1", width_blend=0.0)
        q = _traj(2)
        r0 = m0.match(q, ["Unknown"] * 32, handedness="Right", query_has_body=False)
        r1 = m1.match(q, ["Unknown"] * 32, handedness="Right", query_has_body=False)
        assert (r0.status, r0.top_word, r0.best_distance, r0.r_ratio, r0.match_quality) == (r1.status, r1.top_word, r1.best_distance, r1.r_ratio, r1.match_quality)

    def test_width_blend_cong_thuc(self, tmp_path):
        bank = _mk_bank(False)
        m = _matcher(bank, tmp_path / "x", width_blend=1.0)
        med = float(np.median(list(m.intra_widths.values())))
        assert all(abs(m.effective_width(w) - med) < 1e-9 for w in m.intra_widths)
        m5 = _matcher(bank, tmp_path / "y", width_blend=0.5)
        for w, v in m5.intra_widths.items():
            assert abs(m5.effective_width(w) - np.sqrt(v * med)) < 1e-9
        assert _matcher(bank, tmp_path / "z", width_blend=7.0).width_blend == 1.0     # kẹp vào [0, 1]


# ============================================================================ segmenter theo thời gian
def _run(seg, xs, fps, dup=False, y=300.0, palm=100.0):
    strokes, swipes = 0, 0
    last = None
    for i, x in enumerate(xs):
        cur = x if not (dup and i % 2 == 1 and last is not None) else last
        last = cur
        _, done, sw, _ = seg.update(wrist_x=cur, wrist_y=y, palm_len=palm, hand_shape="x", timestamp=1000.0 + i / fps)
        strokes += done is not None
        swipes += bool(sw)
    return strokes, swipes


def _time_norm_seg(**tn):
    cfg = dict(SHARED_CONFIG)
    cfg["time_norm"] = dict({"enabled": True, "ref_fps": 30.0, "v_scale_min": 1.0}, **tn)
    return make_segmenter(cfg)


class TestSegmenterTimeNorm:
    def test_mac_dinh_tat(self):
        seg = make_segmenter(SHARED_CONFIG)
        assert seg.time_normalized is False and seg.recent_wrists.maxlen == 6
        assert KineticPauseSegmenter().time_normalized is False

    def test_bat_qua_config(self):
        seg = _time_norm_seg(swipe_window_sec=0.2)
        assert seg.time_normalized is True and seg.swipe_window_sec == 0.2 and seg.v_scale_min == 1.0 and seg.ref_fps == 30.0

    @pytest.mark.parametrize("fps", [8, 10, 14, 20, 30, 45, 60])
    def test_swipe_that_van_bat_duoc_o_moi_fps(self, fps):
        n = int(1.9 * fps)
        xs, x = [], 500.0
        for i in range(n):
            t = i / fps
            if 0.6 <= t < 0.9:
                x -= 9.0 * 100.0 / fps          # 9 palm/s
            xs.append(x)
        assert _run(_time_norm_seg(), xs, fps)[1] == 1

    @pytest.mark.parametrize("fps", [8, 15, 60])
    def test_troi_cham_khong_gay_swipe_nham(self, fps):
        n = int(2.0 * fps)
        xs = [500.0 - 2.0 * 100.0 * (i / fps) for i in range(n)]     # trôi trái 2 palm/s (chậm hơn nhiều ngưỡng 4.7)
        assert _run(_time_norm_seg(), xs, fps)[1] == 0

    @pytest.mark.parametrize("fps", [30, 60])
    def test_mot_stroke_van_chot_duoc_o_30_va_60fps(self, fps):
        """Cùng một chuyển động vật lý (1.5 palm trong 0.5 s, rồi đứng yên) phải ra đúng 1 stroke ở cả 30 và 60 fps."""
        xs = []
        for i in range(int(0.8 * fps)):
            xs.append(500.0)
        m = int(0.5 * fps)
        for i in range(m):
            xs.append(500.0 + 150.0 * 0.5 * (1 - np.cos(np.pi * (i + 1) / m)))
        for i in range(int(1.6 * fps)):
            xs.append(650.0)
        assert _run(_time_norm_seg(), xs, fps)[0] == 1


class TestSegmenterKhongDoiKhiTat:
    def test_swipe_cu_van_hoat_dong(self):
        seg = make_segmenter(SHARED_CONFIG)
        xs, x = [], 500.0
        for i in range(60):
            t = i / 30.0
            if 0.6 <= t < 0.9:
                x -= 9.0 * 100.0 / 30.0
            xs.append(x)
        assert _run(seg, xs, 30)[1] == 1
