"""
Unit tests cho SPEC 12 - Buoc 0 (Ve sinh du lieu va cac cong T0-a ... T0-d).
Chay nhanh, khong can webcam, tu don dep thu muc tam sau khi test.
"""

import csv
import json
import os
import shutil
import tempfile
import pytest

from collect_dynamic_templates import (
    write_raw_stroke_archive,
    BackgroundRawArchiveWriter,
    save_negative_strokes,
    RAW_STROKES_DIR
)
from separate_volunteer_raw import scan_and_separate
from check_volunteer_data import (
    check_signer_data,
    REQUIRED_WORDS,
    MIN_SAMPLES_PER_WORD,
    MIN_NEGATIVE_SAMPLES
)
from freeze_volunteer_data import signer_of


def test_t0_a_collector_raw_dir_and_parent_dirs():
    """T0-a: collect_dynamic_templates ghi dung raw_dir va tu dong tao thu muc cha."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_raw = os.path.join(tmpdir, "custom_raw")
        
        # 1. Ghi voi raw_dir chi dinh
        dummy_frames = [{"timestamp": 0.0, "wrist_smoothed": [10.0, 20.0], "face_box": [0.5, 0.5, 0.2, 0.2]}]
        dummy_meta = {"session_id": "test_session_123"}
        out_path = write_raw_stroke_archive(
            word_key="HELLO",
            display_name="Hello",
            frame_records=dummy_frames,
            meta=dummy_meta,
            session_id="test_session_123",
            raw_dir=custom_raw
        )
        assert out_path is not None
        assert os.path.exists(out_path)
        assert custom_raw in os.path.abspath(out_path)
        
        # 2. BackgroundRawArchiveWriter voi raw_dir
        bg_writer = BackgroundRawArchiveWriter(raw_dir=custom_raw)
        bg_writer.enqueue(
            word_key="BYE",
            display_name="Bye",
            frame_records=dummy_frames,
            meta=dummy_meta,
            session_id="test_session_123"
        )
        bg_writer.flush_and_stop()
        bye_dir = os.path.join(custom_raw, "BYE")
        assert os.path.isdir(bye_dir)
        assert len(os.listdir(bye_dir)) >= 1
        
        # 3. Tu tao thu muc cha cho negative_file
        nested_neg = os.path.join(tmpdir, "nested_dir", "test_neg.json")
        assert not os.path.exists(os.path.dirname(nested_neg))
        saved = save_negative_strokes(nested_neg, [{"sample_id": 1}])
        assert saved is True
        assert os.path.exists(nested_neg)


def test_t0_b_separate_volunteer_raw():
    """T0-b: separate_volunteer_raw dry-run tren kho that va apply tren thu muc gia lap."""
    # 1. Tren raw_strokes that
    scanned, matched, moved, skipped = scan_and_separate(
        raw_dir="raw_strokes",
        dest_dir="raw_volunteers",
        apply=False
    )
    assert scanned >= 0
    assert matched == 0, "Hien tai chua co net tinh nguyen vien nao trong raw_strokes that"
    assert moved == 0
    
    # 2. Tren thu muc gia lap
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = os.path.join(tmpdir, "fake_raw")
        dst_dir = os.path.join(tmpdir, "fake_dest")
        word_dir = os.path.join(src_dir, "HELLO")
        os.makedirs(word_dir, exist_ok=True)
        
        # Tao 3 file khop (2 tinh nguyen vien V01, V02; 1 dieu kien C_dark)
        files_matched = [
            ("V01_file1.json", {"session_id": "V01_20260922_090000"}),
            ("V02_file2.json", {"session_id": "V02_20260922_100000"}),
            ("C_dark_file3.json", {"session_id": "C_dark_20260922_110000"}),
        ]
        # Tao 2 file thuong cua Dat
        files_dat = [
            ("dat_file1.json", {"session_id": "20260920_094150"}),
            ("dat_file2.json", {"session_id": "phien1"}),
        ]
        
        for fname, content in files_matched + files_dat:
            fpath = os.path.join(word_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(content, f)
                
        # Dry-run
        s1, m1, mov1, skip1 = scan_and_separate(src_dir, dst_dir, apply=False)
        assert s1 == 5
        assert m1 == 3
        assert mov1 == 0
        assert len(os.listdir(word_dir)) == 5
        
        # Apply lan 1: phai chuyen dung 3 file
        s2, m2, mov2, skip2 = scan_and_separate(src_dir, dst_dir, apply=True)
        assert s2 == 5
        assert m2 == 3
        assert mov2 == 3
        assert len(os.listdir(word_dir)) == 2  # Con lai 2 file cua Dat
        assert os.path.isdir(os.path.join(dst_dir, "HELLO"))
        assert len(os.listdir(os.path.join(dst_dir, "HELLO"))) == 3
        
        # Apply lan 2: idempotent, khong chuyen them file nao
        s3, m3, mov3, skip3 = scan_and_separate(src_dir, dst_dir, apply=True)
        assert s3 == 2
        assert m3 == 0
        assert mov3 == 0


def test_t0_c_check_volunteer_data():
    """T0-c: check_volunteer_data danh gia dung ho so dat, thieu, va co ro ri."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_dir = os.path.join(tmpdir, "profiles")
        raw_vol_dir = os.path.join(tmpdir, "raw_volunteers")
        raw_str_dir = os.path.join(tmpdir, "raw_strokes")
        os.makedirs(prof_dir, exist_ok=True)
        os.makedirs(raw_vol_dir, exist_ok=True)
        os.makedirs(raw_str_dir, exist_ok=True)
        
        notes_file = os.path.join(prof_dir, "notes.csv")
        
        # --- Ca 1: Ho so V01 day du -> DAT ---
        with open(notes_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "signer_id", "condition", "dominant_hand", "height_cm_est", "distance_cm_est", "lighting", "background", "camera", "luma", "new_fps", "imitated", "variant_words", "notes"])
            writer.writerow(["V01_20260922", "V01", "normal", "Right", "170", "80", "bright", "wall", "webcam", "120", "30", "yes", "", "Tot"])
            writer.writerow(["V02_20260922", "V02", "normal", "Right", "165", "80", "bright", "wall", "webcam", "120", "30", "yes", "", "Thieu"])
        
        v01_vocab = {}
        for w in REQUIRED_WORDS:
            v01_vocab[w] = {
                "samples": [
                    {"session_id": "V01_20260922", "traj_features": [0.0] * 24}
                    for _ in range(MIN_SAMPLES_PER_WORD)
                ]
            }
        with open(os.path.join(prof_dir, "V01.json"), "w", encoding="utf-8") as f:
            json.dump({"vocabulary": v01_vocab}, f)
            
        with open(os.path.join(prof_dir, "V01_neg.json"), "w", encoding="utf-8") as f:
            json.dump({"samples": [{"session_id": "V01_20260922"} for _ in range(MIN_NEGATIVE_SAMPLES)]}, f)
            
        # Tao net tho cho V01 trong raw_volunteers co face_box
        v01_raw_dir = os.path.join(raw_vol_dir, "HELLO")
        os.makedirs(v01_raw_dir, exist_ok=True)
        with open(os.path.join(v01_raw_dir, "V01_raw_1.json"), "w", encoding="utf-8") as f:
            json.dump({
                "session_id": "V01_20260922",
                "dominant_handedness": "Right",
                "frames": [{"wrist_smoothed": [0, 0], "face_box": [100, 100, 50, 50]}]
            }, f)
            
        passed_v01, fails_v01, missing_v01, _, _ = check_signer_data(
            signer_id="V01",
            profiles_dir=prof_dir,
            raw_volunteers_dir=raw_vol_dir,
            raw_strokes_dir=raw_str_dir,
            notes_file=notes_file
        )
        assert passed_v01 is True, f"V01 phai DAT nhung lai fail vi: {fails_v01}"
        assert len(missing_v01) == 0
        
        # --- Ca 2: Ho so V02 thieu tu va thieu net am -> CHUA DAT ---
        v02_vocab = {
            "HELLO": {"samples": [{"session_id": "V02_20260922"}]},
            "BYE": {"samples": [{"session_id": "V02_20260922"}]}
        }
        with open(os.path.join(prof_dir, "V02.json"), "w", encoding="utf-8") as f:
            json.dump({"vocabulary": v02_vocab}, f)
        with open(os.path.join(prof_dir, "V02_neg.json"), "w", encoding="utf-8") as f:
            json.dump({"samples": [{"session_id": "V02_20260922"}]}, f)  # chi co 1 mau am (< 8)
            
        passed_v02, fails_v02, missing_v02, _, _ = check_signer_data(
            signer_id="V02",
            profiles_dir=prof_dir,
            raw_volunteers_dir=raw_vol_dir,
            raw_strokes_dir=raw_str_dir,
            notes_file=notes_file
        )
        assert passed_v02 is False
        assert len(missing_v02) > 0  # Missing 11 tu con lai va thieu so luong HELLO, BYE
        assert any("am" in r.lower() for r in fails_v02)
        
        # --- Ca 3: Ho so V01 bi ro ri vao raw_strokes -> Bao loi ro ri ---
        leak_dir = os.path.join(raw_str_dir, "HELLO")
        os.makedirs(leak_dir, exist_ok=True)
        leak_file = os.path.join(leak_dir, "leak_v01.json")
        with open(leak_file, "w", encoding="utf-8") as f:
            json.dump({"session_id": "V01_20260922"}, f)
            
        passed_leak, fails_leak, _, _, _ = check_signer_data(
            signer_id="V01",
            profiles_dir=prof_dir,
            raw_volunteers_dir=raw_vol_dir,
            raw_strokes_dir=raw_str_dir,
            notes_file=notes_file
        )
        assert passed_leak is False
        assert any("ro ri" in r.lower() or "separate_volunteer_raw.py" in r for r in fails_leak)


def test_t0_d_rebuild_exclude_session_prefix():
    """T0-d: rebuild_templates_from_raw loc bo dung net tho khop exclude_session_prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, "raw_strokes")
        hello_dir = os.path.join(raw_dir, "HELLO")
        os.makedirs(hello_dir, exist_ok=True)
        
        # Tao dummy stroke thuan tuy 12 frame hop le
        def make_dummy_raw(sess_id):
            return {
                "schema_version": "raw_stroke_v1.1",
                "session_id": sess_id,
                "word_key": "HELLO",
                "stroke_duration_sec": 1.0,
                "frames": [
                    {
                        "timestamp": i * 0.033,
                        "wrist_smoothed": [float(i), float(i)],
                        "landmarks_px": [[0.0, 0.0, 0.0]] * 21,
                        "palm_len": 50.0,
                        "face_box": [100.0, 100.0, 40.0, 40.0]
                    }
                    for i in range(15)
                ]
            }
            
        # 3 net cua Dat (phien1), 2 net cua V01 (V01_phien)
        with open(os.path.join(hello_dir, "dat_1.json"), "w", encoding="utf-8") as f:
            json.dump(make_dummy_raw("phien1"), f)
        with open(os.path.join(hello_dir, "dat_2.json"), "w", encoding="utf-8") as f:
            json.dump(make_dummy_raw("phien1"), f)
        with open(os.path.join(hello_dir, "dat_3.json"), "w", encoding="utf-8") as f:
            json.dump(make_dummy_raw("phien2"), f)
        with open(os.path.join(hello_dir, "v01_1.json"), "w", encoding="utf-8") as f:
            json.dump(make_dummy_raw("V01_20260922"), f)
        with open(os.path.join(hello_dir, "v01_2.json"), "w", encoding="utf-8") as f:
            json.dump(make_dummy_raw("V01_20260922"), f)
            
        from rebuild_templates_from_raw import main as rebuild_main
        import sys
        
        # 1. Chay rebuild co exclude V
        out_excl = os.path.join(tmpdir, "out_excl.json")
        old_argv = sys.argv
        try:
            sys.argv = [
                "rebuild_templates_from_raw.py",
                "--raw-dir", raw_dir,
                "--out", out_excl,
                "--exclude-session-prefix", "V"
            ]
            rebuild_main()
        finally:
            sys.argv = old_argv
            
        assert os.path.exists(out_excl)
        with open(out_excl, "r", encoding="utf-8") as f:
            data_excl = json.load(f)
        samples_excl = data_excl["vocabulary"]["HELLO"]["samples"]
        assert len(samples_excl) == 3, f"Phai loai bo 2 mau V01, con lai 3 mau nhung duoc {len(samples_excl)}"
        for s in samples_excl:
            assert not s.get("source_raw_file", "").startswith("V")
            
        # 2. Chay rebuild KHONG exclude
        out_all = os.path.join(tmpdir, "out_all.json")
        try:
            sys.argv = [
                "rebuild_templates_from_raw.py",
                "--raw-dir", raw_dir,
                "--out", out_all
            ]
            rebuild_main()
        finally:
            sys.argv = old_argv
            
        assert os.path.exists(out_all)
        with open(out_all, "r", encoding="utf-8") as f:
            data_all = json.load(f)
        samples_all = data_all["vocabulary"]["HELLO"]["samples"]
        assert len(samples_all) == 5, f"Khong exclude thi phai co du 5 mau nhung duoc {len(samples_all)}"


def test_signer_of_function():
    """Kiem tra ham signer_of quy doi session_id thanh signer_id."""
    notes = [
        {"session_id": "phien_dac_biet", "signer_id": "V03"},
        {"session_id": "phien_dat_cond", "signer_id": "DAT"}
    ]
    # 1. Tra tu notes
    assert signer_of("phien_dac_biet", notes) == "V03"
    assert signer_of("phien_dat_cond", notes) == "DAT"
    
    # 2. Regex ^V\d{2}
    assert signer_of("V01_20260922_100000", notes) == "V01"
    assert signer_of("V05_abc", None) == "V05"
    
    # 3. Mac dinh DAT
    assert signer_of("phien1", notes) == "DAT"
    assert signer_of("quan_coffee", None) == "DAT"
    assert signer_of("", None) == "DAT"
