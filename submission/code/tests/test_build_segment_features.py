"""
TEST: dynamic_features.build_segment_features (SPEC 08 / SPEC 09 Nhiệm vụ A)
============================================================================
Hàm dùng chung dựng đặc trưng 24D cho MỘT đoạn khung phải cho kết quả BIT-GIỐNG HỆT (np.array_equal) đoạn code inline
cũ trong predict_sentence_webcam.py (cả nhánh chốt thật lẫn nhánh "dự đoán tạm"). Hai hàm tham chiếu dưới đây là bản sao
NGUYÊN VĂN logic inline cũ (trước khi tách). Thử trên >= 100 stroke thật từ raw_strokes/, cả khi lịch sử góc/pháp tuyến
DÀI hơn stroke và khi NGẮN hơn (nhánh đệm đầu).
"""

import glob
import json
import os

import numpy as np
import pytest

from dynamic_dtw_engine import GestureFrame, resample_9d_gesture_sequence
from dynamic_features import (
    append_extended_columns,
    build_segment_features,
    compute_extended_columns,
    extend_still_tail,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------
# Tham chiếu: logic inline cũ (nhánh CHỐT THẬT)
# ---------------------------------------------------------------------------------
def legacy_commit(completed_stroke, stroke_angles_history, stroke_normals_history, stroke_ext_history):
    n_stroke = len(completed_stroke)
    if len(stroke_angles_history) >= n_stroke:
        angles_for_stroke = stroke_angles_history[-n_stroke:]
    else:
        pad_count = n_stroke - len(stroke_angles_history)
        first_ang = stroke_angles_history[0] if stroke_angles_history else np.zeros(5, dtype=np.float32)
        angles_for_stroke = [first_ang] * pad_count + stroke_angles_history

    if len(stroke_normals_history) >= n_stroke:
        normals_for_stroke = stroke_normals_history[-n_stroke:]
    else:
        pad_norm = n_stroke - len(stroke_normals_history)
        first_norm = stroke_normals_history[0] if stroke_normals_history else np.array([0, 0, -1], dtype=np.float32)
        normals_for_stroke = [first_norm] * pad_norm + stroke_normals_history

    if len(stroke_ext_history) >= n_stroke:
        ext_for_stroke = stroke_ext_history[-n_stroke:]
    else:
        ext_for_stroke = [stroke_ext_history[0] if stroke_ext_history else (None, None)] * (n_stroke - len(stroke_ext_history)) + stroke_ext_history

    x0, y0 = completed_stroke[0].x, completed_stroke[0].y
    ref_palm = max(completed_stroke[0].palm_len, 15.0)

    raw_xs = np.array([(f.x - x0) / ref_palm for f in completed_stroke], dtype=np.float32)
    raw_ys = np.array([(f.y - y0) / ref_palm for f in completed_stroke], dtype=np.float32)

    raw_dxs = np.zeros(n_stroke, dtype=np.float32)
    raw_dys = np.zeros(n_stroke, dtype=np.float32)
    raw_dxs[1:] = raw_xs[1:] - raw_xs[:-1]
    raw_dys[1:] = raw_ys[1:] - raw_ys[:-1]
    raw_dxs[0] = raw_dxs[1] if n_stroke > 1 else 0.0
    raw_dys[0] = raw_dys[1] if n_stroke > 1 else 0.0

    raw_angles_mat = np.array(angles_for_stroke, dtype=np.float32)
    raw_dangles_mat = np.zeros_like(raw_angles_mat)
    raw_dangles_mat[1:] = raw_angles_mat[1:] - raw_angles_mat[:-1]
    raw_dangles_mat[0] = raw_dangles_mat[1] if n_stroke > 1 else 0.0

    raw_normals_mat = np.array(normals_for_stroke, dtype=np.float32)
    raw_17d = np.column_stack([raw_xs, raw_ys, raw_dxs, raw_dys, raw_angles_mat, raw_normals_mat, raw_dangles_mat]).astype(np.float32)
    raw_ext, body_ok = compute_extended_columns(
        [f.palm_len for f in completed_stroke],
        [(f.x, f.y) for f in completed_stroke],
        [e[0] for e in ext_for_stroke],
        [e[1] for e in ext_for_stroke]
    )
    raw_17d = append_extended_columns(raw_17d, raw_ext)
    stroke_shapes = [f.hand_shape for f in completed_stroke]
    res_traj_17d, res_shapes = resample_9d_gesture_sequence(raw_17d, stroke_shapes, k_target=32)
    return res_traj_17d, res_shapes, body_ok


# ---------------------------------------------------------------------------------
# Tham chiếu: logic inline cũ (nhánh DỰ ĐOÁN TẠM, biến đặt tên khác nhưng công thức y hệt)
# ---------------------------------------------------------------------------------
def legacy_temp(temp_stroke, stroke_angles_history, stroke_normals_history, stroke_ext_history):
    n_ts = len(temp_stroke)
    if len(stroke_angles_history) >= n_ts:
        t_angs = stroke_angles_history[-n_ts:]
    else:
        pad_a = n_ts - len(stroke_angles_history)
        fa = stroke_angles_history[0] if stroke_angles_history else np.zeros(5, dtype=np.float32)
        t_angs = [fa] * pad_a + stroke_angles_history
    if len(stroke_normals_history) >= n_ts:
        t_norms = stroke_normals_history[-n_ts:]
    else:
        pad_n = n_ts - len(stroke_normals_history)
        fn = stroke_normals_history[0] if stroke_normals_history else np.array([0, 0, -1], dtype=np.float32)
        t_norms = [fn] * pad_n + stroke_normals_history
    if len(stroke_ext_history) >= n_ts:
        t_exts = stroke_ext_history[-n_ts:]
    else:
        t_exts = [stroke_ext_history[0] if stroke_ext_history else (None, None)] * (n_ts - len(stroke_ext_history)) + stroke_ext_history

    tx0, ty0 = temp_stroke[0].x, temp_stroke[0].y
    tref = max(temp_stroke[0].palm_len, 15.0)
    txs = np.array([(f.x - tx0) / tref for f in temp_stroke], dtype=np.float32)
    tys = np.array([(f.y - ty0) / tref for f in temp_stroke], dtype=np.float32)
    tdxs = np.zeros(n_ts, dtype=np.float32)
    tdys = np.zeros(n_ts, dtype=np.float32)
    tdxs[1:] = txs[1:] - txs[:-1]
    tdys[1:] = tys[1:] - tys[:-1]
    tdxs[0] = tdxs[1] if n_ts > 1 else 0.0
    tdys[0] = tdys[1] if n_ts > 1 else 0.0
    t_ang_mat = np.array(t_angs, dtype=np.float32)
    t_dang_mat = np.zeros_like(t_ang_mat)
    t_dang_mat[1:] = t_ang_mat[1:] - t_ang_mat[:-1]
    t_dang_mat[0] = t_dang_mat[1] if n_ts > 1 else 0.0
    t_norm_mat = np.array(t_norms, dtype=np.float32)
    t_17d = np.column_stack([txs, tys, tdxs, tdys, t_ang_mat, t_norm_mat, t_dang_mat]).astype(np.float32)
    t_ext, t_body_ok = compute_extended_columns(
        [f.palm_len for f in temp_stroke],
        [(f.x, f.y) for f in temp_stroke],
        [e[0] for e in t_exts],
        [e[1] for e in t_exts]
    )
    t_17d = append_extended_columns(t_17d, t_ext)
    t_shapes = [f.hand_shape for f in temp_stroke]
    rt_17d, rt_shapes = resample_9d_gesture_sequence(t_17d, t_shapes, k_target=32)
    return rt_17d, rt_shapes, t_body_ok


def _strokes(min_count=100):
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, "raw_strokes", "*", "*.json")))
             if os.path.basename(os.path.dirname(f)) != "NEGATIVE"]
    out = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        fr = r["frames"]
        if len(fr) < 8:
            continue
        gf = [GestureFrame(x=float(x["wrist_smoothed"][0]), y=float(x["wrist_smoothed"][1]), palm_len=max(float(x["palm_len"]), 15.0),
                           hand_shape=x.get("hand_shape", "Unknown"), timestamp=float(x["timestamp"])) for x in fr]
        angs = [np.array(x["flexion_angles"], dtype=np.float32) for x in fr]
        norms = [np.array(x["palm_normal"], dtype=np.float32) for x in fr]
        exts = [(np.array(x["landmarks_px"], dtype=np.float32), tuple(x["face_box"]) if x.get("face_box") else None) for x in fr]
        out.append((gf, angs, norms, exts))
    if len(out) < min_count:
        pytest.skip(f"chỉ có {len(out)} stroke thô (< {min_count})")
    return out


def _records(gf, angs, norms, exts):
    """Bản ghi khung đúng như predict_sentence_webcam._segment_records (đã đủ độ dài = số khung)."""
    return [{"x": f.x, "y": f.y, "palm_len": f.palm_len, "hand_shape": f.hand_shape, "t": f.timestamp,
             "angles": angs[i], "normal": norms[i], "landmarks": exts[i][0], "face_box": exts[i][1]}
            for i, f in enumerate(gf)]


class TestBuildSegmentFeaturesParity:
    def test_giong_het_tren_stroke_that_lich_su_dai(self):
        strokes = _strokes()
        assert len(strokes) >= 100
        for gf, angs, norms, exts in strokes:
            ref_c = legacy_commit(gf, list(angs), list(norms), list(exts))
            ref_t = legacy_temp(gf, list(angs), list(norms), list(exts))
            got = build_segment_features(_records(gf, angs, norms, exts))
            assert ref_c[0] is not None and got[0] is not None
            assert np.array_equal(ref_c[0], got[0]) and np.array_equal(ref_t[0], got[0])
            assert ref_c[1] == got[1] and ref_t[1] == got[1]
            assert ref_c[2] == got[2] and ref_t[2] == got[2]
            assert got[0].dtype == np.float32 and got[0].shape == (32, 24)

    def test_dominant_handedness(self):
        gf, angs, norms, exts = _strokes()[0]
        recs = _records(gf, angs, norms, exts)
        for i, r in enumerate(recs):
            r["handedness"] = "Left" if i % 3 else "Right"
        assert build_segment_features(recs)[3] == "Left"
        assert build_segment_features(_records(gf, angs, norms, exts))[3] is None   # khóa vắng => None

    def test_qua_ngan_tra_none(self):
        gf, angs, norms, exts = _strokes()[0]
        assert build_segment_features(_records(gf[:5], angs[:5], norms[:5], exts[:5]))[0] is None


class TestPredictHelpersParity:
    """_segment_records của predict (đệm lịch sử) + build_segment_features == logic inline cũ, cả khi lịch sử ngắn hơn stroke."""

    def test_lich_su_dai_ngan_va_rong(self):
        import predict_sentence_webcam as psw
        strokes = _strokes()
        rng = np.random.default_rng(3)
        n_checked = 0
        for gf, angs, norms, exts in strokes[:120]:
            n = len(gf)
            junk_a = [rng.normal(size=5).astype(np.float32) for _ in range(7)]
            junk_n = [rng.normal(size=3).astype(np.float32) for _ in range(7)]
            junk_e = [(None, None)] * 7
            for keep in (n, max(1, n // 2), 0):           # lịch sử: đủ, thiếu một nửa, rỗng
                ha = junk_a + list(angs[n - keep:]) if keep else []
                hn = junk_n + list(norms[n - keep:]) if keep else []
                he = junk_e + list(exts[n - keep:]) if keep else []
                for legacy in (legacy_commit, legacy_temp):
                    ref = legacy(gf, list(ha), list(hn), list(he))
                    got = build_segment_features(psw._segment_records(gf, list(ha), list(hn), list(he)))
                    assert ref[0] is not None and got[0] is not None
                    assert np.array_equal(ref[0], got[0])
                    assert ref[1] == got[1] and ref[2] == got[2]
                    n_checked += 1
        assert n_checked >= 300


class TestExtendStillTail:
    def _recs(self, n_move, n_still, dt=1 / 30.0):
        recs = []
        x = 100.0
        for i in range(n_move):
            x += 12.0
            recs.append({"x": x, "y": 50.0, "palm_len": 100.0, "t": i * dt, "hand_shape": "U", "tag": i})
        for j in range(n_still):
            recs.append({"x": x, "y": 50.0, "palm_len": 100.0, "t": (n_move + j) * dt, "hand_shape": "U", "tag": n_move + j})
        return recs

    def test_keo_dai_dung_thoi_gian(self):
        recs = self._recs(10, 6)                                   # đuôi yên 6 khung = 0.2 s (tính từ khung cuối chuyển động)
        out, n_add = extend_still_tail(recs, 0.45)
        assert n_add == round((0.45 - 0.2) * 30)                   # 7.5 -> làm tròn 8 khung
        assert len(out) == len(recs) + n_add
        assert out[-1]["t"] - out[9]["t"] == pytest.approx(0.45, abs=0.5 / 30 + 1e-6)   # sai số làm tròn <= nửa khung

    def test_da_du_thi_khong_them_va_khong_doi_goc(self):
        recs = self._recs(10, 16)
        out, n_add = extend_still_tail(recs, 0.45)
        assert n_add == 0 and len(out) == len(recs)
        assert len(recs) == 26 and recs is not out

    def test_nhan_ban_khung_cuoi_va_du_min_frames(self):
        recs = self._recs(2, 1)
        out, n_add = extend_still_tail(recs, 0.05, min_frames=8)
        assert len(out) >= 8
        assert all(o["x"] == recs[-1]["x"] and o["tag"] == recs[-1]["tag"] for o in out[len(recs):])

    def test_tail_pad_dua_ve_gan_stroke_chot_cu(self):
        """Stroke thật cắt đuôi còn 0.10 s rồi kéo lại 0.45 s cho đặc trưng rất gần bản nguyên (cả hai đều đuôi yên)."""
        strokes = _strokes()
        diffs = []
        for gf, angs, norms, exts in strokes[:60]:
            recs = _records(gf, angs, norms, exts)
            full = build_segment_features(recs)[0]
            last_motion = 0
            for i in range(1, len(recs)):
                if np.hypot(recs[i]["x"] - recs[i - 1]["x"], recs[i]["y"] - recs[i - 1]["y"]) / recs[i]["palm_len"] > 0.04:
                    last_motion = i
            cut = [r for r in recs if r["t"] <= recs[last_motion]["t"] + 0.10 + 1e-9]
            if len(cut) < 4:
                continue
            padded, _ = extend_still_tail(cut, 0.45)
            got = build_segment_features(padded)[0]
            if got is None or full is None:
                continue
            diffs.append(float(np.mean(np.abs(got[:, :2] - full[:, :2]))))
        assert len(diffs) >= 30
        assert np.median(diffs) < 0.15      # quỹ đạo (x, y) sau resample lệch nhau rất nhỏ (đơn vị palm)


class TestReplayJitter:
    """extend_still_tail(replay_jitter=True): khung thêm lặp lại nhiễu của đuôi thật => độ dài đường đi giữ như đuôi thật."""

    def _recs(self):
        rng = np.random.default_rng(1)
        dt = 1 / 30.0
        recs, x, y = [], 100.0, 50.0
        for i in range(10):
            x += 12.0
            recs.append({"x": x, "y": y, "palm_len": 100.0, "t": i * dt, "hand_shape": "U"})
        for j in range(6):                                   # đuôi yên có nhiễu ~0.5 px
            x += float(rng.normal(0, 0.5)); y += float(rng.normal(0, 0.5))
            recs.append({"x": x, "y": y, "palm_len": 100.0, "t": (10 + j) * dt, "hand_shape": "U"})
        return recs

    def test_mac_dinh_van_bat_dong(self):
        recs = self._recs()
        out, n_add = extend_still_tail(recs, 0.45)
        assert n_add > 0
        assert all(o["x"] == recs[-1]["x"] and o["y"] == recs[-1]["y"] for o in out[len(recs):])

    def test_co_nhieu_giu_do_dai_duong_di_va_khong_troi(self):
        recs = self._recs()
        out, n_add = extend_still_tail(recs, 0.45, replay_jitter=True)
        assert len(out) == len(recs) + n_add and n_add > 4
        xs = np.array([o["x"] for o in out]); ys = np.array([o["y"] for o in out])
        added_path = float(np.sum(np.hypot(np.diff(xs[len(recs) - 1:]), np.diff(ys[len(recs) - 1:]))))
        assert added_path > 0.5                              # có chuyển động nhiễu thật
        assert abs(xs[-1] - recs[-1]["x"]) < 2.0 and abs(ys[-1] - recs[-1]["y"]) < 2.0   # không trôi khỏi tư thế cuối
        assert all(o["palm_len"] == recs[-1]["palm_len"] and o["hand_shape"] == "U" for o in out)   # trường khác nhân bản

    def test_khong_co_duoi_thi_khong_nhieu(self):
        recs = self._recs()[:10]                             # kết thúc ngay khi còn chuyển động: không có bước nhiễu để lặp
        out, n_add = extend_still_tail(recs, 0.45, replay_jitter=True)
        assert n_add > 0 and all(o["x"] == recs[-1]["x"] for o in out[len(recs):])

    def test_records_to_gesture_frames(self):
        from dynamic_features import records_to_gesture_frames
        gf = records_to_gesture_frames(self._recs())
        assert len(gf) == 16 and gf[0].timestamp == 0.0 and gf[3].x == pytest.approx(148.0)
