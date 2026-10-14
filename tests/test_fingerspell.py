# -*- coding: utf-8 -*-
"""Test cho fingerspell.py: ghép chữ cái đánh vần từ chuỗi nhãn theo thời gian. Thuần Python."""
import os
import subprocess
import sys

from fingerspell import Fingerspeller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30.0


def _feed(sp, labels, t0=0.0, fps=FPS):
    """Nạp lần lượt các nhãn, mỗi nhãn một khung. Trả (danh sách chữ được chốt, thời điểm kế tiếp)."""
    out, t = [], t0
    for lab in labels:
        r = sp.update(t, lab)
        if r:
            out.append(r)
        t += 1.0 / fps
    return out, t


def _hold(label, sec, fps=FPS):
    return [label] * int(round(sec * fps))


def test_giu_on_dinh_du_lau_thi_chot_mot_chu():
    sp = Fingerspeller(hold_sec=0.7)
    out, _ = _feed(sp, _hold("a", 1.0))
    assert out == ["A"]
    assert sp.word == "A"


def test_giu_khong_du_lau_khong_chot():
    sp = Fingerspeller(hold_sec=0.7)
    out, _ = _feed(sp, _hold("a", 0.4))
    assert out == [] and sp.word == ""


def test_nhan_nhap_nhay_khong_chot_chu_nao():
    sp = Fingerspeller(hold_sec=0.7)
    out, _ = _feed(sp, ["a", "b"] * 60)
    assert out == [] and sp.word == ""


def test_giu_mai_mot_chu_chi_chot_mot_lan():
    sp = Fingerspeller(hold_sec=0.7)
    out, _ = _feed(sp, _hold("d", 3.0))
    assert out == ["D"]


def test_chu_doi_can_ha_tay_hoac_doi_hinh_roi_giu_lai():
    sp = Fingerspeller(hold_sec=0.5, rearm_sec=0.3)
    labels = _hold("n", 0.8) + _hold(None, 0.5) + _hold("n", 0.8)
    out, _ = _feed(sp, labels)
    assert out == ["N", "N"]


def test_ha_tay_qua_ngan_khong_mo_khoa_chu_doi():
    sp = Fingerspeller(hold_sec=0.5, rearm_sec=0.4)
    labels = _hold("n", 0.8) + _hold(None, 0.1) + _hold("n", 0.8)
    out, _ = _feed(sp, labels)
    assert out == ["N"]


def test_ghep_thanh_ten_an_na():
    sp = Fingerspeller(hold_sec=0.5, rearm_sec=0.3)
    labels = []
    for ch in "annd":
        labels += _hold(ch, 0.7) + _hold(None, 0.4)
    out, _ = _feed(sp, labels)
    assert "".join(out) == "ANND"
    assert sp.token == "SPELL:ANND"


def test_nhan_khong_phai_chu_cai_bi_bo_qua():
    sp = Fingerspeller(hold_sec=0.5)
    out, _ = _feed(sp, _hold("1", 1.0) + _hold("5", 1.0) + _hold(None, 1.0))
    assert out == [] and sp.token == ""


def test_backspace_va_reset():
    sp = Fingerspeller(hold_sec=0.3, rearm_sec=0.2)
    _feed(sp, _hold("a", 0.5) + _hold(None, 0.3) + _hold("b", 0.5))
    assert sp.word == "AB"
    assert sp.backspace() == "B"
    assert sp.word == "A"
    sp.reset()
    assert sp.word == "" and sp.token == "" and sp.backspace() is None


def test_gioi_han_do_dai_tu():
    sp = Fingerspeller(hold_sec=0.2, rearm_sec=0.1, max_len=2)
    labels = []
    for ch in "abc":
        labels += _hold(ch, 0.4) + _hold(None, 0.2)
    _feed(sp, labels)
    assert sp.word == "AB"


def test_tien_do_giu_chu():
    sp = Fingerspeller(hold_sec=1.0)
    sp.update(0.0, "w")
    assert sp.current == "W"
    assert abs(sp.progress(0.5) - 0.5) < 1e-6
    assert sp.progress(2.0) == 1.0


def test_module_khong_keo_theo_cv2_mediapipe_torch():
    code = ("import sys, fingerspell; "
            "bad=[m for m in ('cv2','mediapipe','torch') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert r.returncode == 0
