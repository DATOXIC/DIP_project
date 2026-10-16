"""
TEST: palm_len CÓ MỘT NGUỒN SỰ THẬT DUY NHẤT (dynamic_features.compute_palm_len)
==============================================================================
Bối cảnh: collect_dynamic_templates.py từng tính thành phần y của palm_3d bằng w_f (bề rộng) thay vì h_f
(bề cao) như predict_sentence_webcam.py => palm_len khi thu lớn hơn lúc chạy tới 640/480 = 1.33 lần,
quỹ đạo trong bank bị co lại. Các test dưới đây khóa lại 3 điều:
  (a) hàm mới == công thức của predict (bit-giống hệt) trên landmark ngẫu nhiên và landmark thật;
  (b) công thức LỖI cũ của collector cho kết quả khác (test không rỗng);
  (c) collector / predict không còn tự viết công thức, mà GỌI hàm dùng chung.
"""

import glob
import json
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

from dynamic_features import compute_palm_len

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------
# Công thức tham chiếu (sao nguyên văn từ predict_sentence_webcam.py TRƯỚC khi sửa) và công thức LỖI cũ
# ---------------------------------------------------------------------------------
def palm_len_predict_cu(lm0, lm5, lm9, lm17, w_f, h_f):
    palm_2d = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)
    palm_3d = math.sqrt(((lm9.x - lm0.x) * w_f) ** 2 + ((lm9.y - lm0.y) * h_f) ** 2 + ((lm9.z - lm0.z) * w_f) ** 2)
    knuckle_span = math.hypot((lm17.x - lm5.x) * w_f, (lm17.y - lm5.y) * h_f)
    return max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)


def palm_len_collector_loi(lm0, lm5, lm9, lm17, w_f, h_f):
    palm_2d = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)
    palm_3d = math.sqrt(((lm9.x - lm0.x) * w_f) ** 2 + ((lm9.y - lm0.y) * w_f) ** 2 + ((lm9.z - lm0.z) * w_f) ** 2)
    knuckle_span = math.hypot((lm17.x - lm5.x) * w_f, (lm17.y - lm5.y) * h_f)
    return max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)


def _lm(x, y, z):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def _bo_landmark_ngau_nhien(rng, huong_len):
    """4 mốc (0, 5, 9, 17) chuẩn hóa. huong_len=True: lòng bàn tay dọc (thành phần y trội), tay to để vượt sàn 30 px."""
    lm0 = _lm(rng.uniform(0.2, 0.8), rng.uniform(0.5, 0.95), rng.uniform(-0.05, 0.05))
    if huong_len:
        dx, dy = rng.uniform(-0.03, 0.03), -rng.uniform(0.12, 0.30)
    else:
        dx, dy = rng.uniform(-0.30, 0.30), rng.uniform(-0.05, 0.05)
    lm9 = _lm(lm0.x + dx, lm0.y + dy, lm0.z + rng.uniform(-0.05, 0.05))
    lm5 = _lm(lm0.x + rng.uniform(-0.1, 0.1), lm0.y + rng.uniform(-0.2, 0.0), lm0.z + rng.uniform(-0.05, 0.05))
    lm17 = _lm(lm0.x + rng.uniform(-0.1, 0.1), lm0.y + rng.uniform(-0.2, 0.0), lm0.z + rng.uniform(-0.05, 0.05))
    return lm0, lm5, lm9, lm17


# ---------------------------------------------------------------------------------
# (a) hàm mới == công thức cũ của predict
# ---------------------------------------------------------------------------------
def test_a_ngau_nhien_bit_giong_predict_cu():
    rng = np.random.RandomState(20260920)
    n_kiem = 0
    for i in range(1500):
        w_f, h_f = ((640, 480), (1280, 720), (1920, 1080), (480, 640))[i % 4]
        lm = _bo_landmark_ngau_nhien(rng, huong_len=(i % 2 == 0))
        moi = compute_palm_len(*lm, w_f, h_f)
        cu = palm_len_predict_cu(*lm, w_f, h_f)
        assert moi == cu, f"lệch ở bộ #{i}: moi={moi!r} cu={cu!r}"
        n_kiem += 1
    assert n_kiem >= 1000


def test_a_landmark_that_tu_raw_strokes():
    files = sorted(glob.glob(os.path.join(ROOT, "raw_strokes", "*", "*.json")))
    if not files:
        pytest.skip("không có raw_strokes/ để lấy landmark thật")
    n_khung = 0
    for fp in files[:: max(1, len(files) // 60)]:
        with open(fp, "r", encoding="utf-8") as f:
            raw = json.load(f)
        w_f, h_f = float(raw["frame_width"]), float(raw["frame_height"])
        for fr in raw["frames"]:
            px = fr["landmarks_px"]   # (x*w, y*h, z*w) => chia lại về chuẩn hóa
            lm = [_lm(px[i][0] / w_f, px[i][1] / h_f, px[i][2] / w_f) for i in (0, 5, 9, 17)]
            moi = compute_palm_len(*lm, w_f, h_f)
            cu = palm_len_predict_cu(*lm, w_f, h_f)
            assert moi == cu
            n_khung += 1
    assert n_khung >= 500


# ---------------------------------------------------------------------------------
# (b) công thức lỗi cũ của collector cho kết quả KHÁC (để test (a) không rỗng)
# ---------------------------------------------------------------------------------
def test_b_cong_thuc_loi_cu_cho_ket_qua_khac():
    rng = np.random.RandomState(7)
    w_f, h_f = 640, 480
    n_khac, ti_le_max = 0, 0.0
    for _ in range(500):
        lm = _bo_landmark_ngau_nhien(rng, huong_len=True)
        dung = compute_palm_len(*lm, w_f, h_f)
        loi = palm_len_collector_loi(*lm, w_f, h_f)
        assert loi >= dung - 1e-9                       # nhân w_f >= h_f nên lỗi không bao giờ NHỎ hơn
        ti_le = loi / dung
        assert ti_le <= w_f / h_f + 1e-9                # trần lý thuyết 640/480
        ti_le_max = max(ti_le_max, ti_le)
        n_khac += int(loi != dung)
    assert n_khac >= 400, "công thức lỗi phải cho kết quả khác ở đa số bộ landmark tay hướng lên"
    assert ti_le_max > 1.2, f"tỉ lệ lỗi lớn nhất {ti_le_max:.3f} quá nhỏ: bộ ngẫu nhiên không bắt được lỗi"


def test_b_landmark_that_cong_thuc_loi_khop_palm_len_da_luu():
    """Chứng minh lỗi có thật trong dữ liệu: palm_len LƯU trong raw_strokes == công thức lỗi, != công thức đúng.
    Chỉ kiểm các phiên thu TRƯỚC bản sửa palm_len: phien1, phien2.
    Các phiên 20260920_* và mới hơn đã lưu giá trị đúng từ compute_palm_len => bỏ qua.
    """
    # Tập phiên thu trước bản sửa palm_len (collector còn dùng công thức sai)
    PRE_FIX_SESSIONS = {"phien1", "phien2"}

    files = sorted(glob.glob(os.path.join(ROOT, "raw_strokes", "*", "*.json")))
    if not files:
        pytest.skip("không có raw_strokes/")
    n_khac_dung = n_khop_loi = n = 0
    for fp in files[:: max(1, len(files) // 40)]:
        with open(fp, "r", encoding="utf-8") as f:
            raw = json.load(f)
        sid = raw.get("session_id", "")
        # Chỉ kiểm các phiên thu trước bản sửa; bỏ qua thu_nghiem và các phiên sau
        if sid not in PRE_FIX_SESSIONS:
            continue
        w_f, h_f = float(raw["frame_width"]), float(raw["frame_height"])
        for fr in raw["frames"]:
            px = fr["landmarks_px"]
            lm = [_lm(px[i][0] / w_f, px[i][1] / h_f, px[i][2] / w_f) for i in (0, 5, 9, 17)]
            luu = float(fr["palm_len"])
            n += 1
            n_khop_loi += int(abs(luu - palm_len_collector_loi(*lm, w_f, h_f)) < 1e-3)
            n_khac_dung += int(abs(luu - compute_palm_len(*lm, w_f, h_f)) > 1e-3)
    if n == 0:
        pytest.skip("không có khung nào từ phiên phien1/phien2 trong mẫu lấy")
    assert n > 50, f"quá ít khung ({n}) để kiểm tra"
    assert n_khop_loi == n, f"palm_len đã lưu phải khớp công thức lỗi ({n_khop_loi}/{n})"
    assert n_khac_dung > 0.3 * n, f"palm_len đã lưu phải lệch công thức đúng ở nhiều khung ({n_khac_dung}/{n})"


# ---------------------------------------------------------------------------------
# (c) grep trong source: collector không còn công thức lỗi; collector và predict GỌI hàm dùng chung
# ---------------------------------------------------------------------------------
def _doc(ten):
    with open(os.path.join(ROOT, ten), "r", encoding="utf-8") as f:
        return f.read()


def test_c_collector_khong_con_cong_thuc_loi():
    src = _doc("collect_dynamic_templates.py")
    # thành phần y nhân w_f: ((lm9.y - lm0.y) * w_f)
    assert not re.search(r"lm9\.y\s*-\s*lm0\.y\)\s*\*\s*w_f", src), "collector còn công thức lỗi (y nhân w_f)"
    assert "palm_3d" not in src and "palm_2d" not in src and "knuckle_span" not in src, "collector còn tự tính palm_len"
    assert "compute_palm_len(lm0, lm5, lm9, lm17, w_f, h_f)" in src


def test_c_predict_goi_ham_dung_chung():
    src = _doc("predict_sentence_webcam.py")
    assert "palm_3d" not in src and "palm_2d" not in src and "knuckle_span" not in src, "predict còn tự tính palm_len"
    assert "compute_palm_len(lm0, lm5, lm9, lm17, w_f, h_f)" in src
    assert re.search(r"from dynamic_features import \([^)]*compute_palm_len", src, re.S)
