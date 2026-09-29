"""Kiểm thử dip.py và people.py: dựng lệnh đúng, hồ sơ theo tên, không đụng dữ liệu thật."""
import csv
import os

import pytest

import dip
import people


def test_slugify_bo_dau_va_khoang_trang():
    assert people.slugify("Nguyễn An") == "NguyenAn"
    assert people.slugify("Đạt") == "Dat"
    assert people.slugify("  mai ") == "Mai"
    with pytest.raises(ValueError):
        people.slugify("123")
    with pytest.raises(ValueError):
        people.slugify("???")


def test_people_mac_dinh_va_them_moi(tmp_path):
    path = str(tmp_path / "profiles" / "people.json")
    d = people.load_people(path)
    assert set(d) == {"DAT", "Mai", "Phi"}
    assert d["DAT"]["ten"] == "Đạt"
    assert people.volunteer_ids(path) == ["Mai", "Phi"]
    assert people.add_person("Nguyễn An", path=path) == "NguyenAn"
    assert "NguyenAn" in people.volunteer_ids(path)
    with pytest.raises(ValueError):
        people.add_person("nguyen an", path=path)          # trùng mã (không phân biệt hoa/thường)


def test_resolve_theo_ten_hoac_ma(tmp_path):
    path = str(tmp_path / "people.json")
    assert people.resolve("đạt", path) == "DAT"
    assert people.resolve("Dat", path) == "DAT"
    assert people.resolve("mai", path) == "Mai"
    assert people.resolve("PHI", path) == "Phi"
    assert people.resolve("ai_do", path) is None
    assert people.resolve("", path) is None


def test_lenh_nhan_dien():
    assert dip.cmd_nhan_dien("DAT") == ["predict_sentence_webcam.py"]
    assert dip.cmd_nhan_dien("Mai", ["--time-norm"]) == ["predict_sentence_webcam.py", "--templates", "profiles/Mai.json", "--time-norm"]


def test_lenh_thu_tinh_nguyen_vien_ghi_vao_raw_volunteers():
    a = dip.cmd_thu("Phi", ngay="20260929")
    assert a[0] == "collect_dynamic_templates.py"
    assert a[a.index("--templates") + 1] == "profiles/Phi.json"
    assert a[a.index("--session") + 1] == "Phi_20260929"
    assert a[a.index("--raw-dir") + 1] == "raw_volunteers"           # KHÔNG bao giờ lẫn vào raw_strokes
    assert a[a.index("--negative-file") + 1] == "profiles/Phi_neg.json"


def test_lenh_thu_cua_dat():
    assert dip.cmd_thu("DAT") == ["collect_dynamic_templates.py"]    # bank sản xuất (cần xác nhận ở tầng trên)
    a = dip.cmd_thu("DAT", dieu_kien="dark", ngay="20260929")
    assert a[a.index("--templates") + 1] == "profiles/DAT_cond.json"  # KHÔNG đụng webcam_templates.json
    assert a[a.index("--session") + 1] == "C_dark_20260929"
    assert a[a.index("--raw-dir") + 1] == "raw_volunteers"
    with pytest.raises(ValueError):
        dip.cmd_thu("DAT", dieu_kien="khong_co")


def test_lenh_kiem_tra_va_danh_gia():
    assert dip.cmd_kiem_tra("Mai") == ["check_volunteer_data.py", "--signer", "Mai"]
    assert dip.cmd_danh_gia("open", "Phi")[-2:] == ["--signer", "Phi"]
    assert dip.cmd_danh_gia("loso", None)[0] == "scoreboard.py"
    assert dip.cmd_danh_gia("hybrid", "Mai")[0] == "eval_hybrid.py"
    with pytest.raises(ValueError):
        dip.cmd_danh_gia("khong_co", None)


def test_ghi_chu_buoi_thu_khong_trung(tmp_path, monkeypatch):
    monkeypatch.setattr(dip, "NOTES_PATH", str(tmp_path / "notes.csv"))
    dip.ensure_notes_row("Mai", "Mai_20260930", hoi=False)
    dip.ensure_notes_row("Mai", "Mai_20260930", hoi=False)            # gọi lại: không thêm dòng
    dip.ensure_notes_row("DAT", "C_dark_20260930", hoi=False, condition="dark")
    rows = list(csv.DictReader(open(tmp_path / "notes.csv", encoding="utf-8", newline="")))
    assert [r["session_id"] for r in rows] == ["Mai_20260930", "C_dark_20260930"]
    assert rows[1]["condition"] == "dark" and rows[1]["signer_id"] == "DAT"


def test_tach_net_tho_nhan_ra_tinh_nguyen_vien_theo_ten(tmp_path):
    import json
    import separate_volunteer_raw as sv
    raw, dest = tmp_path / "raw", tmp_path / "dest"
    for sess, w in (("Mai_20260930", "HELLO"), ("phien1", "HELLO")):
        d = raw / w
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sess}.json").write_text(json.dumps({"session_id": sess}), encoding="utf-8")
    res = sv.scan_and_separate(raw_dir=str(raw), dest_dir=str(dest), apply=False)
    assert res[1] == 1        # chỉ nét của Mai khớp, nét của Đạt (phien1) thì không


def test_mien_tu_cho_mot_nguoi(tmp_path):
    path = str(tmp_path / "people.json")
    assert people.skipped_words("Mai", path) == []
    assert people.set_skipped_words("Mai", ["you", " yes ", "BAD", "you"], path) == ["BAD", "YES", "YOU"]
    assert people.skipped_words("Mai", path) == ["BAD", "YES", "YOU"]
    assert people.skipped_words("Phi", path) == []             # người khác không bị ảnh hưởng
    assert people.set_skipped_words("Mai", [], path) == []      # xoá miễn
    assert people.skipped_words("Mai", path) == []
    with pytest.raises(ValueError):
        people.set_skipped_words("KhongCo", ["YOU"], path)


def test_lenh_kiem_tra_kem_tu_duoc_mien():
    assert dip.cmd_kiem_tra("Mai", ["YOU", "YES"]) == ["check_volunteer_data.py", "--signer", "Mai", "--bo-qua", "YOU", "YES"]
    assert dip.cmd_kiem_tra("Mai") == ["check_volunteer_data.py", "--signer", "Mai"]


def test_kiem_tra_bo_qua_tu_khong_bao_thieu(tmp_path):
    import json
    import check_volunteer_data as cv

    prof = tmp_path / "profiles"
    prof.mkdir()
    samples = {w: {"samples": [{"session_id": "Zed_1", "handedness": "Right"}] * 3} for w in cv.REQUIRED_WORDS if w != "YOU"}
    (prof / "Zed.json").write_text(json.dumps({"vocabulary": samples}), encoding="utf-8")
    kw = dict(signer_id="Zed", profiles_dir=str(prof), raw_volunteers_dir=str(tmp_path / "rv"),
              raw_strokes_dir=str(tmp_path / "rs"), notes_file=str(tmp_path / "notes.csv"))
    _, _, thieu, _, _ = cv.check_signer_data(**kw)
    assert any(t.startswith("YOU") for t in thieu)
    _, _, thieu2, counts2, _ = cv.check_signer_data(**kw, skip_words=["you"])
    assert not any(t.startswith("YOU") for t in thieu2) and "YOU" not in counts2


def test_merge_banks_concatenates_samples(tmp_path, monkeypatch):
    import json
    import dip
    monkeypatch.setattr(dip, "ROOT", str(tmp_path))
    a = {"metadata": {"m": 1}, "vocabulary": {"HI": {"threshold": 5, "samples": [1, 2]}}}
    b = {"metadata": {}, "vocabulary": {"HI": {"threshold": 9, "samples": [3]}, "NO": {"samples": [4]}}}
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    m = dip.merge_banks(["a.json", "b.json"])
    assert m["vocabulary"]["HI"]["samples"] == [1, 2, 3]
    assert m["vocabulary"]["HI"]["threshold"] == 5 and m["vocabulary"]["HI"]["num_samples"] == 3
    assert m["vocabulary"]["NO"]["samples"] == [4]


def test_cmd_nhan_dien_templates_override():
    import dip
    assert dip.cmd_nhan_dien("Mai", None, "profiles/_gop_Mai.json") == ["predict_sentence_webcam.py", "--templates", "profiles/_gop_Mai.json"]


def test_cmd_nhan_dien_confirm_flag():
    import dip
    assert dip.cmd_nhan_dien("DAT", None, None, True) == ["predict_sentence_webcam.py", "--confirm-enroll"]


def test_session_enroller_adds_and_undoes():
    import numpy as np
    from dynamic_dtw_engine import GoldenTriadDTWMatcher
    from session_enroll import SessionEnroller
    import json
    bank = json.load(open("webcam_templates.json", encoding="utf-8"))
    m = GoldenTriadDTWMatcher(templates_dict=bank, threshold_mode="scale_k", threshold_k=1.9, uncertainty_k=2.5, width_blend=1.0)
    word = next(iter(m.vocabulary))
    s0 = m.vocabulary[word]["samples"][0]
    n0 = len(m.vocabulary[word]["samples"])
    e = SessionEnroller(m, cap_per_word=2)
    assert e.confirm(0, 0.0) is None                          # chưa có câu hỏi
    assert e.offer(np.array(s0["traj_features"]), s0["hand_shapes"], s0.get("handedness"), True, 1.0, [word, "X"], now=0.0)
    assert e.confirm(0, 100.0) is None                        # hết hạn
    assert e.offer(np.array(s0["traj_features"]), s0["hand_shapes"], s0.get("handedness"), True, 1.0, [word], now=0.0)
    assert e.confirm(0, 1.0) == word
    assert len(m.vocabulary[word]["samples"]) == n0 + 1
    r = m.match(np.array(s0["traj_features"], dtype=np.float32), s0["hand_shapes"], handedness=s0.get("handedness"))
    assert r.top_word == word
    for i in range(2):                                        # cap = 2
        e.offer(np.array(s0["traj_features"]), s0["hand_shapes"], None, True, 1.0, [word], now=0.0)
        got = e.confirm(0, 1.0)
        assert (got == word) == (i == 0)
    assert e.undo() == word and len(m.vocabulary[word]["samples"]) == n0 + 1
    assert e.undo() == word and len(m.vocabulary[word]["samples"]) == n0
    assert e.undo() is None


def test_tien_do_runs_and_reports_counts(capsys):
    import dip
    assert dip.main(["tien-do"]) == 0
    out = capsys.readouterr().out
    assert "TIẾN ĐỘ" in out and f"/{dip.TARGET_NGUOI} người đạt" in out
    t = dip.tinh_trang("Mai")
    assert set(t) == {"dat", "thieu", "tu_du", "tu_can", "mau", "neg"}


def test_app_window_builds_and_streams_subprocess():
    import time
    import pytest
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("không có màn hình")
    root.withdraw()
    try:
        import dip_app
        dip_app.messagebox.showwarning = lambda *a, **k: None      # đừng chặn test bằng hộp thoại
        a = dip_app.App(root)
        got = []
        assert a.run(["-c", "print('xin chào'); raise SystemExit(3)"], "thử", on_done=got.append)
        for _ in range(100):
            root.update()
            time.sleep(0.05)
            if got:
                break
        assert got == [3] and "xin chào" in a.log.get("1.0", "end")
        assert a.run(["-c", "import time; time.sleep(30)"], "ngủ")
        assert not a.run(["-c", "pass"], "đụng độ")          # đang bận thì từ chối việc thứ hai
        a.stop()
    finally:
        root.destroy()
