# -*- coding: utf-8 -*-
"""
Test hiển thị câu tự nhiên ở panel dưới (SPEC 14). Vẽ headless lên mảng numpy, không cần webcam.
"""
import numpy as np

import ui_side_panel
from ui_render import strip_diacritics
from ui_side_panel import BOTTOM_H, SidePanelState, draw_bottom_panel

W = 640


def _draw(**kw):
    canvas = np.zeros((BOTTOM_H, W, 3), np.uint8)
    st = SidePanelState(sentence_words=["Me", "Hungry"], help_line="[q]:Thoat", **kw)
    draw_bottom_panel(canvas, 0, 0, W, BOTTOM_H, st)
    return canvas


def test_trang_thai_cu_tao_duoc_khong_can_composed_sentence():
    assert SidePanelState().composed_sentence is None


def test_co_cau_tu_nhien_khi_da_chot_thi_ve_them_dong():
    plain = _draw(sentence_finalized=True)
    with_sentence = _draw(sentence_finalized=True, composed_sentence="I am hungry.")
    assert not np.array_equal(plain, with_sentence)


def test_khong_co_cau_tu_nhien_thi_giong_het_ban_cu():
    # composed_sentence=None (mặc định) phải cho đúng ảnh như trước khi có tính năng.
    a = _draw(sentence_finalized=True)
    b = _draw(sentence_finalized=True, composed_sentence=None)
    assert np.array_equal(a, b)


def test_chua_chot_cau_thi_khong_ve_cau_tu_nhien():
    plain = _draw(sentence_finalized=False)
    with_sentence = _draw(sentence_finalized=False, composed_sentence="I am hungry.")
    assert np.array_equal(plain, with_sentence)


def test_dong_gloss_goc_khong_bi_che():
    plain = _draw(sentence_finalized=True)
    with_sentence = _draw(sentence_finalized=True, composed_sentence="I am hungry.")
    # Vùng dòng gloss (y từ 12 đến 52) phải giống hệt: dòng mới chỉ nằm bên dưới.
    assert np.array_equal(plain[12:52], with_sentence[12:52])


def test_cau_tu_nhien_nam_giua_dong_gloss_va_dong_phim_tat():
    plain = _draw(sentence_finalized=True)
    with_sentence = _draw(sentence_finalized=True, composed_sentence="I am hungry.")
    diff_rows = np.where(np.any(plain != with_sentence, axis=(1, 2)))[0]
    assert diff_rows.size > 0
    help_top = BOTTOM_H - 16 - 12              # dòng phím tắt vẽ tại y0 + h - 16
    assert diff_rows.min() > 52 and diff_rows.max() < help_top


def test_cau_rat_dai_khong_loi_va_khong_tran_khung():
    long_text = "Could I have some water, please? " * 8
    img = _draw(sentence_finalized=True, composed_sentence=long_text)
    assert img.shape == (BOTTOM_H, W, 3)
    plain = _draw(sentence_finalized=True)
    # Lề phải (14 px, bằng `pad` của panel) phải giống hệt bản không có câu tự nhiên: chữ không tràn ra mép.
    assert np.array_equal(img[:, W - 14:], plain[:, W - 14:])


def test_strip_diacritics_tra_ve_ascii_cho_cau_tieng_viet():
    for s in ("Làm ơn cho tôi xin nước uống.", "Tôi cảm thấy tệ.", "Xin chào, Đạt."):
        out = strip_diacritics(s)
        assert out.isascii() and out
    assert strip_diacritics("Làm ơn cho tôi xin nước uống.") == "Lam on cho toi xin nuoc uong."


def test_event_log_van_dung_duoc_voi_cau_khong_dau():
    log = ui_side_panel.EventLog(maxlen=5)
    log.push("Cau tu nhien: " + strip_diacritics("Tôi đói."), ui_side_panel.COL_OK)
    assert len(log.items()) == 1
