"""Kiểm thử threshold_trial_report: ghép truth/sự kiện, đếm, Wilson (so với giá trị tính tay)."""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threshold_trial_report as ttr  # noqa: E402

HEADER = ["timestamp", "event", "predicted_word", "best_distance", "second_word", "second_distance",
          "margin_pct", "duration_sec", "num_frames", "body_anchor_valid", "fps",
          "best_confidence", "match_quality", "r_ratio"]


def _row(event, word=""):
    return ["2026-09-20 10:00:00", event, word, 0.5, "", "", 10.0, 1.0, 30, True, 30.0, 0.5, 50.0, 1.0]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def _write_truth(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_wilson_gia_tri_tinh_tay():
    # 1/10: tâm = (0.1 + 3.8415/20)/1.38415 = 0.21101 ; nửa rộng = 0.19315 -> [0.0179; 0.4042]
    lo, hi = ttr.wilson_interval(1, 10)
    assert lo == pytest.approx(0.0179, abs=5e-4)
    assert hi == pytest.approx(0.4042, abs=5e-4)
    # 0/8: cận dưới 0, cận trên = z^2/(n+z^2) = 3.8415/11.8415 = 0.3244
    lo, hi = ttr.wilson_interval(0, 8)
    assert lo == 0.0
    assert hi == pytest.approx(0.3244, abs=5e-4)
    # 5/10: đối xứng quanh 0.5 -> [0.2366; 0.7634]
    lo, hi = ttr.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=5e-4)
    assert hi == pytest.approx(0.7634, abs=5e-4)
    # n = 0 -> nan
    lo, hi = ttr.wilson_interval(0, 0)
    assert lo != lo and hi != hi


def test_parse_truth_bo_chu_thich(tmp_path):
    p = tmp_path / "truth.txt"
    _write_truth(p, ["# tiêu đề", "hello", "", "NEG   # gãi mặt", "  THANK_YOU  "])
    assert ttr.parse_truth(str(p)) == ["HELLO", "NEG", "THANK_YOU"]


def test_ghep_dem_dung_mot_muc_bo_sot_mot_thua_mot_neg_nhan_nham(tmp_path):
    truth = ["HELLO", "BYE", "ME", "NEG", "YOU", "THANK_YOU", "EAT", "NEG"]
    truth_path = tmp_path / "truth.txt"
    _write_truth(truth_path, truth)

    rows = [
        _row("ACCEPTED", "HELLO"),          # HELLO: ACCEPTED_đúng
        _row("OUT_OF_DURATION"),            # không tham gia ghép
        # BYE: KHÔNG có sự kiện -> BỎ_SÓT
        _row("ACCEPTED", "ME"),             # ME: ACCEPTED_đúng
        _row("ACCEPTED", "DRINK"),          # NEG bị nhận nhầm
        _row("UNCERTAIN", "YOU"),           # YOU: UNCERTAIN_top1_đúng
        _row("ACCEPTED", "HELLO"),          # SỰ KIỆN THỪA -> nhận ma
        _row("REJECTED", "THANK_YOU"),      # THANK_YOU: REJECTED
        _row("ACCEPTED", "WATER"),          # EAT: ACCEPTED_sai
        _row("OUT_OF_DURATION"),
        # NEG cuối: không sinh sự kiện -> ĐÚNG_KHI_KHÔNG_NHẬN
    ]
    csv_path = tmp_path / "a.csv"
    _write_csv(csv_path, rows)

    events, ood = ttr.load_events(str(csv_path))
    assert len(events) == 7 and len(ood) == 2
    s = ttr.summarize(ttr.parse_truth(str(truth_path)), events, ood)

    c = s["counts"]
    assert c[ttr.C_ACC_OK] == 2
    assert c[ttr.C_ACC_BAD] == 1
    assert c[ttr.C_UNC_OK] == 1
    assert c[ttr.C_UNC_BAD] == 0
    assert c[ttr.C_REJ] == 1
    assert c[ttr.C_MISS] == 1
    assert s["n_word"] == 6 and s["n_neg"] == 2 and s["n_truth"] == 8
    assert s["neg_acc"] == 1 and s["neg_ok"] == 1 and s["neg_no_event"] == 1
    assert s["ghost_acc"] == 1 and s["ghost_other"] == 0
    assert s["out_of_duration"] == 2
    # chi phí = ACCEPTED_sai (1) + NEG bị nhận (1) + nhận ma (1)
    assert s["bad_accepts"] == 3


def test_ghep_tat_dinh_va_khong_co_su_kien(tmp_path):
    # Không có sự kiện nào: mọi từ thật là BỎ_SÓT, mọi NEG là đúng khi không nhận.
    s = ttr.summarize(["HELLO", "NEG", "BYE"], [], [])
    assert s["counts"][ttr.C_MISS] == 2
    assert s["neg_ok"] == 1 and s["neg_acc"] == 0 and s["bad_accepts"] == 0
    pairs = ttr.align(["A", "B"], [])
    assert pairs == [(0, None), (1, None)]


def test_csv_khong_hop_le_bao_loi(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ttr.load_events(str(p))


def test_main_compare_chay_duoc(tmp_path, capsys):
    truth_path = tmp_path / "truth.txt"
    _write_truth(truth_path, ["HELLO", "NEG"])
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, [_row("ACCEPTED", "HELLO"), _row("REJECTED", "ME")])
    _write_csv(b, [_row("UNCERTAIN", "HELLO"), _row("ACCEPTED", "ME")])
    rc = ttr.main(["--compare", str(a), str(b), "--truth", str(truth_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "KHÔNG ghép cặp" in out and "CHI PHÍ" in out
