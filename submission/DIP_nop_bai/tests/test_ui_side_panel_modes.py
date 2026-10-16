# -*- coding: utf-8 -*-
"""
Test giao diện panel bên/dưới: log tiếng Việt có dấu không ra '???', chế độ demo gọn khác chế độ debug,
chip trạng thái đúng ngữ cảnh. Vẽ headless lên mảng numpy, không cần webcam.
"""
import types

import numpy as np

import ui_side_panel as u

PANEL_H = 620


def _cand(name, conf):
    return types.SimpleNamespace(display_name=name, word_key=name, confidence=conf, distance=0.5)


def _mr(status="ACCEPTED", conf=0.9):
    return types.SimpleNamespace(
        top_word="HELLO", top_display="HELLO", status=status, result_source="DTW", best_distance=0.3,
        best_confidence=conf, match_quality=80.0, fusion_note="", dtw_status=status, dtw_word="HELLO",
        dl_word=None, dl_confidence=-1.0, candidates=[_cand("HELLO", 0.5), _cand("HI", 0.3), _cand("BYE", 0.1)])


def _right(mode, **kw):
    canvas = np.zeros((PANEL_H, u.PANEL_W, 3), np.uint8)
    st = u.SidePanelState(ui_mode=mode, **kw)
    u.draw_right_panel(canvas, 0, 0, u.PANEL_W, PANEL_H, st)
    return canvas


def _log(*texts):
    log = u.EventLog(maxlen=10)
    for t in texts:
        log.push(t)
    return log


def test_ascii_bo_dau_tieng_viet():
    assert u._ascii("Nhận từ: Đạt được") == "Nhan tu: Dat duoc"
    assert u._ascii("Chốt câu → xong") == "Chot cau  xong"


def test_log_co_dau_giong_het_log_khong_dau():
    # Trước khi sửa, 'ậ' 'ừ' vẽ thành '???' nên ảnh khác bản không dấu; giờ phải y hệt từng điểm ảnh.
    for mode in ("demo", "debug"):
        with_marks = _right(mode, event_log=_log("Nhận từ: HELLO"))
        # thời gian trong log debug khác nhau giữa hai lần gọi -> dùng cùng một EventLog cho cả hai.
        log = _log("Nhận từ: HELLO")
        a = _right(mode, event_log=log)
        log_plain = u.EventLog(maxlen=10)
        log_plain._items = type(log._items)(
            [(t, "Nhan tu: HELLO", c) for (t, _, c) in log.items()], maxlen=10)
        b = _right(mode, event_log=log_plain)
        assert np.array_equal(a, b), mode
        assert with_marks.shape == a.shape


def test_demo_va_debug_khac_nhau_o_panel_phai():
    kw = dict(match_result=_mr(), seg_text="SAN SANG")
    assert not np.array_equal(_right("demo", **kw), _right("debug", **kw))


def test_demo_khong_hien_bang_ky_thuat():
    # Debug có tiêu đề 'TOP-K' ở vùng giữa panel; demo không có -> hai vùng giữa phải khác nhau và demo trống hơn.
    kw = dict(match_result=_mr("ACCEPTED"), seg_text="SAN SANG")
    demo = _right("demo", **kw)
    debug = _right("debug", **kw)
    assert (demo[300:].sum()) < (debug[300:].sum())


def test_che_do_khac_demo_dung_giao_dien_debug():
    kw = dict(match_result=_mr(), seg_text="SAN SANG")
    # Dòng đầu in nhãn chế độ ('debug/vi' hay 'legacy/vi') nên chỉ so phần thân panel, từ dòng 60 trở xuống.
    assert np.array_equal(_right("debug", **kw)[60:], _right("legacy", **kw)[60:])


def test_chip_trang_thai_theo_ngu_canh():
    base = dict(sentence_words=[], hand_found=False, seg_text="CHO TAY")
    assert u._chip_state(u.SidePanelState(**base))[0] == "DUA TAY VAO KHUNG HINH"
    assert u._chip_state(u.SidePanelState(**{**base, "hand_found": True, "seg_text": "KHUNG TAY"}))[0] == "DA THAY TAY"
    assert u._chip_state(u.SidePanelState(**{**base, "seg_text": "SAN SANG"}))[0] == "SAN SANG"
    assert u._chip_state(u.SidePanelState(**{**base, "seg_text": "DANG GHI"}))[0] == "DANG GHI CU CHI"
    done = u.SidePanelState(sentence_words=["A"], sentence_finalized=True)
    assert u._chip_state(done)[0] == "DA CHOT CAU"
    empty = u.SidePanelState(vocabulary_empty=True, seg_text="DANG GHI")
    assert u._chip_state(empty)[0] == "CHUA CO TU DIEN"   # lỗi dữ liệu phải thắng mọi trạng thái khác


def test_demo_khong_loi_khi_chua_co_ket_qua_hoac_khong_co_log():
    img = _right("demo", match_result=None, event_log=None)
    assert img.shape == (PANEL_H, u.PANEL_W, 3)


def test_demo_ung_vien_chi_hien_khi_chua_chac():
    sure = _right("demo", match_result=_mr("ACCEPTED"))
    unsure = _right("demo", match_result=_mr("UNCERTAIN"))
    assert not np.array_equal(sure, unsure)


def test_demo_panel_duoi_cau_dai_va_co_dau_khong_loi_va_khong_tran():
    w, h = 640, u.BOTTOM_H
    words = ["Xin chào"] * 12
    canvas = np.zeros((h, w, 3), np.uint8)
    st = u.SidePanelState(ui_mode="demo", sentence_words=words, sentence_finalized=True,
                          composed_sentence="Làm ơn cho tôi xin nước uống. " * 6, help_line="[q]:Thoat")
    u.draw_bottom_panel(canvas, 0, 0, w, h, st)
    plain = np.zeros((h, w, 3), np.uint8)
    u.draw_bottom_panel(plain, 0, 0, w, h, u.SidePanelState(ui_mode="demo", sentence_finalized=True))
    # lề phải 18px (pad của panel demo) bên trong viền 2px không bị chữ tràn vào
    assert np.array_equal(canvas[4:h - 4, w - 17:w - 3], plain[4:h - 4, w - 17:w - 3])
