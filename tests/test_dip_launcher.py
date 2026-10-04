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
    assert dip.cmd_nhan_dien("DAT", spotter=False) == ["predict_sentence_webcam.py", "--no-dl"]
    assert dip.cmd_nhan_dien("Mai", ["--time-norm"], spotter=False) == ["predict_sentence_webcam.py", "--templates", "profiles/Mai.json", "--time-norm"]


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
    # bank gộp là file TẠM => mẫu xác nhận phải được lưu vào hồ sơ cá nhân thật (--luu-ho-so), không vào file gộp
    assert dip.cmd_nhan_dien("Mai", None, "profiles/_gop_Mai.json", spotter=False) == [
        "predict_sentence_webcam.py", "--luu-ho-so", "profiles/Mai.json", "--templates", "profiles/_gop_Mai.json"]
    assert "--luu-ho-so" not in dip.cmd_nhan_dien("Mai", None, "profiles/khac.json", spotter=False)


def test_cmd_nhan_dien_day_tu():
    a = dip.cmd_nhan_dien("Mai", None, None, True, None, None, False, ["HUNGRY", "SORRY"], 2, spotter=False)
    assert a[:6] == ["predict_sentence_webcam.py", "--day-tu", "HUNGRY", "SORRY", "--day-so-mau", "2"]


def test_build_merged_bank_ho_so_chua_co_file(tmp_path, monkeypatch):
    """Hồ sơ mới (chưa có file) vẫn dựng được bank gộp = chỉ bank Đạt."""
    import json
    (tmp_path / "profiles").mkdir()
    (tmp_path / "webcam_templates.json").write_text(json.dumps({"metadata": {}, "vocabulary": {"HI": {"samples": [1]}}}))
    monkeypatch.setattr(dip, "ROOT", str(tmp_path))
    out = dip.build_merged_bank("Moi", True)
    assert json.load(open(tmp_path / out))["vocabulary"]["HI"]["samples"] == [1]


def test_cmd_nhan_dien_confirm_flag():
    import dip
    # xác nhận đã BẬT MẶC ĐỊNH ở predict_sentence_webcam.py: bật = không thêm cờ, tắt = --no-confirm-enroll
    assert dip.cmd_nhan_dien("DAT", None, None, True, spotter=False) == ["predict_sentence_webcam.py", "--no-dl"]
    assert dip.cmd_nhan_dien("DAT", None, None, False, spotter=False) == ["predict_sentence_webcam.py", "--no-confirm-enroll", "--no-dl"]


def test_cmd_nhan_dien_dl_fusion_off_for_owner_only():
    import dip
    assert "--no-dl" in dip.cmd_nhan_dien("DAT", None, None, True, spotter=False)
    assert "--no-dl" not in dip.cmd_nhan_dien("V01", None, None, True, spotter=False)
    assert "--no-dl" not in dip.cmd_nhan_dien("DAT", None, None, True, "m.pth", None, spotter=False)
    assert "--no-dl" not in dip.cmd_nhan_dien("DAT", None, None, True, None, 0.5, spotter=False)


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


def _teach_engine():
    import json
    from dynamic_dtw_engine import GoldenTriadDTWMatcher
    bank = json.load(open("webcam_templates.json", encoding="utf-8"))
    # sign_weights_path="" = trọng số toàn cục như cấu hình thật của predict_sentence_webcam.py (ASL-LEX sẽ từ chối từ mới)
    return GoldenTriadDTWMatcher(templates_dict=bank, threshold_mode="scale_k", threshold_k=1.9, uncertainty_k=2.5,
                                 width_blend=1.0, sign_weights_path="")


def test_teach_plan_progress_follows_undo():
    from session_enroll import SessionEnroller, TeachPlan
    m = _teach_engine()
    e = SessionEnroller(m)
    plan = TeachPlan(["newword", "YOU", "newword"], per_word=2)
    assert plan.words == ["NEWWORD", "YOU"]
    s = m.vocabulary["EAT"]["samples"][0]
    args = (s["traj_features"], s["hand_shapes"], "Left", True, 1.0)
    assert e.add_taught("NEWWORD", *args) == 1
    assert plan.current(e) == ("NEWWORD", 1, 2)
    e.add_taught("NEWWORD", *args)
    assert plan.current(e) == ("YOU", 0, 2)
    assert e.undo() == "NEWWORD"                       # hoàn tác lùi tiến độ đúng
    assert plan.current(e) == ("NEWWORD", 1, 2)
    assert "NEWWORD" in m.vocabulary and len(m.vocabulary["NEWWORD"]["samples"]) == 1


def test_teach_new_word_is_recognised_immediately_and_saved_to_profile(tmp_path):
    import json
    import numpy as np
    from session_enroll import SessionEnroller
    m = _teach_engine()
    e = SessionEnroller(m)
    src = m.vocabulary["EAT"]["samples"]
    for s in src[:3]:
        e.add_taught("NEWWORD", s["traj_features"], s["hand_shapes"], s.get("handedness"), True, 1.0)
    q = src[5]
    r = m.match(np.asarray(q["traj_features"], dtype=np.float32), q["hand_shapes"], handedness=q.get("handedness"), query_has_body=True)
    assert "NEWWORD" in [c.word_key for c in r.candidates[:3]]          # dùng được ngay, không train
    path = str(tmp_path / "Moi.json")
    assert e.merge_into_file(path, enrolled_cap=6) == 3
    saved = json.load(open(path, encoding="utf-8"))["vocabulary"]["NEWWORD"]
    assert len(saved["samples"]) == 3 and saved["display_name"] == "NEWWORD" and all(x["enrolled"] for x in saved["samples"])
    assert e.merge_into_file(path, enrolled_cap=6) == 3                  # mẫu dạy luôn được ghi (không bị cap)


def test_merge_into_full_profile_counts_only_enrolled_samples(tmp_path):
    """Hồ sơ đã có nhiều mẫu thu thường: cap xác nhận chỉ đếm mẫu 'enrolled', nên vẫn nhận thêm mẫu xác nhận."""
    import json
    from session_enroll import SessionEnroller
    m = _teach_engine()
    s = m.vocabulary["YOU"]["samples"][0]
    path = tmp_path / "Mai.json"
    path.write_text(json.dumps({"metadata": {}, "vocabulary": {"YOU": {"samples": [dict(s) for _ in range(10)]}}}))
    e = SessionEnroller(m, cap_per_word=3)
    sid = e._next_id
    s2 = dict(s, session_added=sid)
    m.vocabulary["YOU"]["samples"].append(s2)
    e.added.append({"word": "YOU", "id": sid})
    e._next_id += 1
    assert e.merge_into_file(str(path)) == 0                             # cap kiểu cũ: 10 >= 3 nên bỏ
    assert e.merge_into_file(str(path), enrolled_cap=6) == 1


def test_nhan_dien_mac_dinh_bat_spotter_neu_co_model(monkeypatch):
    monkeypatch.setattr(dip.os.path, "exists", lambda p: True)
    a = dip.cmd_nhan_dien("DAT", None, None, True)
    assert a[:4] == ["predict_sentence_webcam.py", "--one-shot", "--spotter", "dl"]
    monkeypatch.setattr(dip.os.path, "exists", lambda p: False)
    assert "--one-shot" not in dip.cmd_nhan_dien("DAT", None, None, True)           # không có model => không bật
    assert "--one-shot" not in dip.cmd_nhan_dien("DAT", None, None, True, spotter=False)


def test_lenh_hieu_chinh_tu_choi_chu_du_an_va_danh_sach_rong():
    with pytest.raises(ValueError):
        dip.lenh_hieu_chinh(people.OWNER_ID, ["HUNGRY"])
    with pytest.raises(ValueError):
        dip.lenh_hieu_chinh("Mai", ["  "])


def test_lenh_mau_am():
    assert dip.cmd_quay_mau_am(people.OWNER_ID, 4.0) == ["collect_continuous.py", "--signer", "Dat", "--negative", "--minutes", "4.0"]
    assert dip.cmd_quay_mau_am("Mai", 3)[:3] == ["collect_continuous.py", "--signer", "Mai"]
    a = dip.cmd_huan_luyen_spotter(["_neg"], "x.pth")
    assert a == ["train_spotter.py", "--models", "CNN", "--save", "--out", "x.pth", "--exclude", "_neg"]
    assert "spotter_model.pth" not in dip.cmd_huan_luyen_spotter()                  # mặc định KHÔNG đè model đang dùng
    assert dip.cmd_cham_mau_am(None, ["a.pth", "b.pth"]) == ["eval_negative_session.py", "--latest", "--model", "a.pth", "--model", "b.pth"]
