# -*- coding: utf-8 -*-
"""Kiểm thử nhanh lớp web (web_app.py): không chạy pipeline thật, chỉ kiểm tra kiểm soát đầu vào, bảo vệ localhost và lọc sự kiện."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import web_app  # noqa: E402

SAMPLE = "BAD/1583686602856138-BAD.mp4"


@pytest.fixture()
def client():
    web_app.MEDIA.clear()
    web_app.JOBS.clear()
    web_app._active_job = None
    with TestClient(web_app.app, base_url="http://127.0.0.1:8000") as c:
        yield c


def test_vocab_names_loaded_from_pipeline_source():
    assert web_app._names("BAD") == {"en": "Bad", "vi": "Tệ"}
    assert web_app._names("KHONG_CO") == {"en": "KHONG_CO", "vi": ""}


def test_index_and_profiles(client):
    assert client.get("/").status_code == 200
    profiles = client.get("/api/profiles").json()
    assert profiles and profiles[0]["id"] == "DAT" and profiles[0]["samples"] > 0


def test_rejects_non_local_host_and_foreign_origin(client):
    assert client.get("/api/profiles", headers={"Host": "evil.example"}).status_code == 403
    r = client.post("/api/sample", json={"path": SAMPLE}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_sample_path_traversal_is_rejected(client):
    for bad in ("../web_app.py", "..\\web_app.py", "BAD/../../dip.py", "/etc/passwd", "BAD/khong_ton_tai.mp4"):
        assert client.post("/api/sample", json={"path": bad}).status_code == 400


def test_upload_rejects_wrong_extension_and_unreadable_video(client):
    r = client.post("/api/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
    r = client.post("/api/upload", files={"file": ("x.mp4", b"khong phai video", "video/mp4")})
    assert r.status_code == 400
    leftovers = [p for p in web_app.UPLOAD_DIR.glob("*.mp4") if p.stat().st_size == len(b"khong phai video")]
    assert leftovers == []


def test_fetch_rejects_non_http_scheme(client):
    assert client.post("/api/fetch", json={"url": "file:///C:/Windows/win.ini"}).status_code == 400
    assert client.post("/api/fetch", json={"url": "ftp://example.com/a.mp4"}).status_code == 400


def test_analyze_validation(client):
    assert client.post("/api/analyze", json={"media_id": "khong_co", "start": 0}).status_code == 404
    if not (web_app.SAMPLES_DIR / SAMPLE).is_file():
        pytest.skip("thiếu video mẫu ASL Citizen")
    media = client.post("/api/sample", json={"path": SAMPLE}).json()
    assert media["duration"] > 1.0
    base = {"media_id": media["id"]}
    assert client.post("/api/analyze", json={**base, "start": 999}).status_code == 400
    assert client.post("/api/analyze", json={**base, "start": 1.0, "end": 1.1}).status_code == 400
    assert client.post("/api/analyze", json={**base, "start": 0.0, "profile": "../x"}).status_code == 400
    assert client.post("/api/analyze", json={**base, "start": 0.0, "segmenter": "--evil"}).status_code == 400
    assert client.post("/api/analyze", json={**base, "start": 0.0, "predictor": "x; rm"}).status_code == 400
    assert client.post("/api/analyze", json={**base, "start": 0.0, "tune": {"pause_sec": 99}}).status_code == 400
    assert client.post("/api/analyze", json={**base, "start": 0.0, "tune": {"pause_sec": "abc"}}).status_code == 400
    assert client.post("/api/analyze", json={**base, "start": 0.0, "tune": {"--evil": 1}}).status_code == 400
    assert client.post("/api/analyze", json={**base, "start": 0.0, "tune": {"early_commit": "no"}}).status_code == 400
    assert web_app._active_job is None


def test_read_events_filters_before_start_and_adds_names(tmp_path):
    path = tmp_path / "ev.jsonl"
    rows = [
        {"event": "ACCEPTED", "word": "BAD", "t_start": 1.0, "t_end": 1.4, "duration": 0.4, "candidates": [{"word": "BAD"}], "dtw_word": "BAD", "dl_word": None},
        {"event": "ACCEPTED", "word": "GO", "t_start": 5.0, "t_end": 5.5, "duration": 0.5, "candidates": [], "dtw_word": None, "dl_word": None},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write('{"event": "ACCEPTED", "word": "dang_ghi_do')   # dòng dở dang (chưa xuống dòng) phải bị bỏ qua
    kept = web_app._read_events(path, start=2.0)
    assert [e["word"] for e in kept] == ["GO"]
    assert kept[0]["name"]["en"] == "Go"
    both = web_app._read_events(path, start=0.0)
    assert [e["word"] for e in both] == ["BAD", "GO"]
    assert both[0]["dtw_word_name"]["vi"] == "Tệ" and both[0]["dl_word_name"] is None


def test_read_events_keeps_file_index_and_strips_sample(tmp_path):
    path = tmp_path / "ev.jsonl"
    rows = [
        {"event": "ACCEPTED", "word": "BAD", "t_start": 1.0, "t_end": 1.4, "duration": 0.4, "candidates": [], "sample": {"traj_features": [[0.0]]}},
        {"event": "OUT_OF_DURATION", "word": "", "t_start": 0.2, "t_end": 0.3, "duration": 0.1, "candidates": []},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    shown = web_app._read_events(path, start=0.0)
    assert [e["idx"] for e in shown] == [1, 0]            # sắp theo thời gian, nhưng idx = thứ tự dòng trong file
    assert all("sample" not in e for e in shown)           # mẫu thô không gửi cho trình duyệt
    assert {e["idx"]: e["has_sample"] for e in shown} == {0: True, 1: False}
    full = web_app._read_events(path, start=0.0, keep_sample=True)
    assert [e for e in full if e["idx"] == 0][0]["sample"]["traj_features"] == [[0.0]]


def test_force_mode_requires_end_and_save_sample_validation(client):
    if not (web_app.SAMPLES_DIR / SAMPLE).is_file():
        pytest.skip("thiếu video mẫu ASL Citizen")
    media = client.post("/api/sample", json={"path": SAMPLE}).json()
    assert client.post("/api/analyze", json={"media_id": media["id"], "start": 0.2, "segmenter": "force"}).status_code == 400
    assert client.post("/api/jobs/khong_co/sample", json={"idx": 0, "word": "BAD", "person": "A"}).status_code == 404
    assert client.get("/api/jobs/khong_co/trace").status_code == 404
    # job giả đã xong, không có sự kiện: kiểm tra thứ tự và nội dung các lỗi đầu vào
    web_app.JOBS["fake"] = {"id": "fake", "status": "done", "events_path": web_app.RUN_DIR / "khong_ton_tai.jsonl", "start": 0.0, "media_id": "m"}
    url = "/api/jobs/fake/sample"
    assert client.post(url, json={"idx": 0, "word": "KHONG_CO", "person": "A"}).status_code == 400
    assert client.post(url, json={"idx": 0, "word": "BAD", "person": "  "}).status_code == 400
    assert client.post(url, json={"idx": 0, "word": "BAD", "person": "Đạt"}).status_code == 400     # không ghi vào bank sản xuất
    assert client.post(url, json={"idx": 0, "word": "BAD", "person": "NguoiLaThu"}).status_code == 404   # không có sự kiện nào
    web_app.JOBS["fake"]["status"] = "running"
    assert client.post(url, json={"idx": 0, "word": "BAD", "person": "NguoiLaThu"}).status_code == 409
