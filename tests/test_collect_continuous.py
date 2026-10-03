"""Kiểm thử collect_continuous.py: kịch bản, FSM phím giả, nhãn từng khung, lưu/đọc phiên (không cần webcam)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import collect_continuous as cc  # noqa: E402

VOCAB13 = ["BAD", "BYE", "DRINK", "EAT", "HELLO", "HUNGRY", "ME", "PLEASE", "SORRY", "THANK_YOU", "WATER", "YES", "YOU"]


# ------------------------------------------------------------------ build_script
def test_build_script_deterministic_and_complete():
    a = cc.build_script(VOCAB13, pairs=8, seed=5)
    b = cc.build_script(list(reversed(VOCAB13)), pairs=8, seed=5)
    c = cc.build_script(VOCAB13, pairs=8, seed=6)
    assert a == b                      # xác định theo seed, không phụ thuộc thứ tự vocab
    assert a != c
    kinds = [p["type"] for p in a]
    assert kinds.count("SIGN") == 13 and kinds.count("PAIR") == 8 and kinds.count("WANDER") == 2
    assert kinds[0] == "REST" and kinds[-1] == "WANDER"
    assert [p["index"] for p in a] == list(range(len(a)))
    assert {p["words"][0] for p in a if p["type"] == "SIGN"} == set(VOCAB13)
    for p in a:
        if p["type"] == "PAIR":
            assert len(p["words"]) == 2 and p["words"][0] != p["words"][1]
        if p["type"] == "WANDER":
            assert p["duration_sec"] == 30.0 and p["hints"]
    # SIGN/PAIR xen kẽ REST: sau mỗi SIGN/PAIR là REST
    for i, p in enumerate(a):
        if p["type"] in ("SIGN", "PAIR"):
            assert a[i + 1]["type"] == "REST"
    assert 4 * 60 <= cc.estimate_duration_sec(a) <= 9 * 60
    assert len({p["words"] for p in [dict(words=tuple(q["words"])) for q in a if q["type"] == "PAIR"]}) == 8


def _mini_script():
    return [
        {"index": 0, "type": "SIGN", "words": ["HELLO"], "duration_sec": None, "hints": []},
        {"index": 1, "type": "PAIR", "words": ["ME", "YOU"], "duration_sec": None, "hints": []},
        {"index": 2, "type": "REST", "words": [], "duration_sec": 2.0, "hints": []},
        {"index": 3, "type": "SIGN", "words": ["BYE"], "duration_sec": None, "hints": []},
    ]


def _types(runner, aid=None):
    return [e["type"] for e in runner.events if aid is None or e.get("aid") == aid]


# ------------------------------------------------------------------ ScriptRunner
def test_runner_sign_pair_and_timed_flow():
    r = cc.ScriptRunner(_mini_script(), rest_prep_sec=1.0)
    assert r.key(" ", 0.0) == "ignored"            # chưa start
    r.start(0.0)
    assert r.state == "READY" and r.prompt["type"] == "SIGN"
    assert r.key("m", 0.5) == "ignored"            # 'm' chỉ trong PAIR
    assert r.key(" ", 1.0) == "sign_start" and r.state == "SIGNING"
    assert r.key(" ", 2.0) == "sign_end" and r.state == "SIGNED"
    assert r.key(" ", 2.1) == "ignored"
    assert r.key("\r", 2.5) == "next"              # Enter = n
    assert r.prompt["type"] == "PAIR"
    r.key(" ", 3.0)
    assert r.key("m", 4.0) == "boundary"
    assert r.key("m", 4.1) == "boundary"           # lần 2: đã ổn định ở vị trí từ 2
    assert r.key("m", 4.2) == "ignored"            # tối đa 2 lần bấm m cho cặp 2 từ
    r.key(" ", 5.0)
    r.key("n", 5.5)
    # REST: chuẩn bị 1 s rồi 2 s
    assert r.state == "PREP" and r.prompt["type"] == "REST"
    r.tick(6.4)
    assert r.state == "PREP"
    r.tick(6.6)
    assert r.state == "TIMED"
    ev_begin = [e for e in r.events if e["type"] == "prompt_begin" and e["kind"] == "REST"][0]
    assert abs(ev_begin["t"] - 6.5) < 1e-9
    r.tick(8.6)                                    # hết 2 s -> tự sang prompt kế
    assert r.prompt["type"] == "SIGN" and r.prompt["words"] == ["BYE"] and r.state == "READY"
    ev_end = [e for e in r.events if e["type"] == "prompt_end" and e["prompt_index"] == 2][0]
    assert abs(ev_end["t"] - 8.5) < 1e-9
    r.key(" ", 9.0)
    r.key("q", 9.5)                                # q khi đang ký: sign_end tự động + prompt_end + session_end
    assert r.finished
    tail = [e["type"] for e in r.events][-3:]
    assert tail == ["sign_end", "prompt_end", "session_end"]
    assert r.events[-3].get("auto") is True
    assert [e["t"] for e in r.events] == sorted(e["t"] for e in r.events)


def test_runner_bad_and_back():
    r = cc.ScriptRunner(_mini_script(), rest_prep_sec=0.5)
    r.start(0.0)
    r.key(" ", 1.0)
    r.key(" ", 2.0)
    assert r.key("x", 2.2) == "bad"                # đánh dấu attempt vừa ký
    bad = [e for e in r.events if e["type"] == "bad"][0]
    assert bad["aid"] == 1 and bad["prompt_index"] == 0
    assert r.key("x", 2.3) == "ignored"            # đã bad rồi, không lặp
    r.key("n", 2.5)                                # sang PAIR (chưa có nội dung)
    assert r.state == "READY"
    r.key("b", 3.0)                                # làm lại PAIR hiện tại (aid 2, chưa ký), KHÔNG động tới attempt SIGN aid 1
    assert sorted(e["aid"] for e in r.events if e["type"] == "attempt_discard") == [2]
    assert r.index == 1 and r.state == "READY" and 1 not in r.discarded_aids
    r.key(" ", 4.0)
    r.key(" ", 5.0)                                # ký xong lần mới (aid 3)
    assert r.cur_aid == 3
    r.key("b", 5.5)                                # làm lại: chỉ bỏ aid 3
    assert sorted(e["aid"] for e in r.events if e["type"] == "attempt_discard") == [2, 3]
    assert r.index == 1 and r.state == "READY" and 1 not in r.discarded_aids
    r.key(" ", 6.0)
    r.key(" ", 7.0)
    assert r.cur_aid == 4 and 4 not in r.bad_aids
    r.key("n", 7.2)                                # sang prompt kế; x nhắm vào attempt vừa xong (aid 4)
    assert r.key("x", 7.3) == "bad"
    assert [e["aid"] for e in r.events if e["type"] == "bad"][-1] == 4


def test_runner_b_at_first_prompt_and_wander_skip():
    script = [{"index": 0, "type": "WANDER", "words": [], "duration_sec": 30.0, "hints": ["wave"]},
              {"index": 1, "type": "SIGN", "words": ["YES"], "duration_sec": None, "hints": []}]
    r = cc.ScriptRunner(script, wander_prep_sec=1.0)
    r.start(0.0)
    r.tick(1.0)
    assert r.state == "TIMED"
    assert r.key("n", 5.0) == "next"               # bỏ qua WANDER sớm: vẫn có prompt_end
    assert any(e["type"] == "prompt_end" and e["t"] == 5.0 for e in r.events)
    assert r.prompt["type"] == "SIGN"
    r.key("b", 6.0)
    assert r.index == 1 and r.state == "READY"     # làm lại đúng prompt SIGN hiện tại, không lùi về WANDER
    assert r.snapshot(6.5)["remaining"] == 0


# ------------------------------------------------------------------ derive_frame_labels
def _ev(t, typ, aid, idx=0, **kw):
    e = {"t": t, "type": typ, "aid": aid, "prompt_index": idx}
    e.update(kw)
    return e


def test_derive_labels_margin_transition_bad():
    t = np.arange(0.0, 40.0, 0.05)
    evs = [
        _ev(0.0, "prompt_begin", 1, 0, kind="REST", words=[]), _ev(4.0, "prompt_end", 1, 0),
        _ev(5.0, "prompt_begin", 2, 1, kind="SIGN", words=["HELLO"]),
        _ev(6.0, "sign_start", 2, 1), _ev(8.0, "sign_end", 2, 1), _ev(9.0, "prompt_end", 2, 1),
        _ev(10.0, "prompt_begin", 3, 2, kind="PAIR", words=["ME", "YOU"]),
        _ev(11.0, "sign_start", 3, 2), _ev(12.0, "boundary", 3, 2, index=0), _ev(14.0, "sign_end", 3, 2),
        _ev(15.0, "prompt_end", 3, 2),
        _ev(16.0, "prompt_begin", 4, 3, kind="WANDER", words=[]), _ev(26.0, "prompt_end", 4, 3),
        _ev(27.0, "prompt_begin", 5, 4, kind="SIGN", words=["BAD"]),
        _ev(28.0, "sign_start", 5, 4), _ev(30.0, "sign_end", 5, 4), _ev(31.0, "bad", 5, 4),
        _ev(31.5, "prompt_end", 5, 4),
        _ev(32.0, "prompt_begin", 6, 5, kind="SIGN", words=["EAT"]),
        _ev(33.0, "sign_start", 6, 5), _ev(35.0, "sign_end", 6, 5), _ev(36.0, "attempt_discard", 6, 5),
    ]
    m = 0.15
    lab = cc.derive_frame_labels(t, {"events": evs}, margin_sec=m)
    assert lab.dtype == np.uint8 and lab.shape == t.shape

    def at(x):
        return lab[int(round(x / 0.05))]

    # REST co margin ở hai đầu
    assert at(0.05) == cc.IGNORE and at(0.30) == cc.REST and at(2.0) == cc.REST and at(3.95) == cc.IGNORE
    # SIGN: ngoài khoảng bấm -> IGNORE; quanh sign_start/sign_end ±margin -> IGNORE; giữa -> SIGN
    assert at(5.5) == cc.IGNORE and at(8.5) == cc.IGNORE
    assert at(6.0) == cc.IGNORE and at(6.10) == cc.IGNORE and at(6.2) == cc.SIGN
    assert at(7.0) == cc.SIGN and at(7.8) == cc.SIGN and at(7.95) == cc.IGNORE and at(8.1) == cc.IGNORE
    # PAIR: TRANSITION trong ±margin quanh boundary, hai bên là SIGN
    assert at(11.5) == cc.SIGN and at(12.0) == cc.TRANSITION and at(11.9) == cc.TRANSITION and at(12.1) == cc.TRANSITION
    assert at(11.7) == cc.SIGN and at(12.3) == cc.SIGN and at(13.0) == cc.SIGN and at(14.0) == cc.IGNORE
    # WANDER
    assert at(20.0) == cc.WANDER and at(16.05) == cc.IGNORE
    # bad -> IGNORE toàn attempt; discard -> IGNORE
    assert set(np.unique(lab[(t >= 27.0) & (t <= 31.5)])) == {cc.IGNORE}
    assert set(np.unique(lab[(t >= 32.0) & (t <= 36.0)])) == {cc.IGNORE}
    # danh sách trần cũng được chấp nhận
    assert np.array_equal(lab, cc.derive_frame_labels(t, evs, margin_sec=m))
    # margin 0: SIGN phủ trọn khoảng bấm
    lab0 = cc.derive_frame_labels(t, evs, margin_sec=0.0)
    assert lab0[int(round(6.0 / 0.05))] == cc.SIGN and lab0[int(round(7.0 / 0.05))] == cc.SIGN


def test_derive_labels_unterminated_and_empty():
    t = np.arange(0.0, 10.0, 0.1)
    evs = [_ev(1.0, "prompt_begin", 1, 0, kind="SIGN", words=["A"]), _ev(2.0, "sign_start", 1, 0)]
    lab = cc.derive_frame_labels(t, evs, 0.15)       # phiên bị cắt giữa chừng: sign_end mặc định = hết dữ liệu
    assert lab[int(round(5.0 / 0.1))] == cc.SIGN
    assert cc.derive_frame_labels(np.array([]), evs).shape == (0,)
    assert (cc.derive_frame_labels(t, []) == cc.IGNORE).all()


# ------------------------------------------------------------------ lưu / đọc phiên
def test_save_load_roundtrip(tmp_path):
    script = _mini_script()
    runner, rec = cc.simulate_session(script, fps=30.0, seed=1, rest_prep_sec=0.5, bad_every=2)
    assert runner.finished and len(rec) > 100
    sdir = str(tmp_path / "V99" / "s1")
    meta = {"signer": "V99", "lang": "vi", "seed": 1}
    paths = cc.save_session(sdir, rec, runner.events, meta, script)
    assert os.path.isfile(paths["frames"]) and os.path.isfile(paths["events"])
    s = cc.load_session(sdir)
    arr = rec.arrays()
    assert s["landmarks"].shape == (len(rec), 2, 21, 3) and s["landmarks"].dtype == np.float32
    assert np.array_equal(s["t"], arr["t"])
    assert np.array_equal(np.isnan(s["landmarks"]), np.isnan(arr["landmarks"]))
    assert np.allclose(np.nan_to_num(s["landmarks"]), np.nan_to_num(arr["landmarks"]))
    assert s["handedness"].shape == (len(rec), 2) and s["face_box"].shape == (len(rec), 4)
    assert abs(s["fps"] - 30.0) < 0.5
    assert s["events"]["events"] == runner.events
    assert s["meta"]["signer"] == "V99" and s["meta"]["schema_version"] == cc.SCHEMA_VERSION
    assert s["script"] == script and s["video_path"] is None
    assert np.array_equal(s["labels"], cc.derive_frame_labels(arr["t"], runner.events))
    # tay chỉ xuất hiện khi đang ký (mô phỏng): khung SIGN có tay, khung REST không
    has_hand = ~np.isnan(s["landmarks"][:, 1, 0, 0])
    assert has_hand[s["labels"] == cc.SIGN].all()
    assert not has_hand[s["labels"] == cc.REST].any()
    # attempt bị x ở prompt SIGN thứ 2 (PAIR) -> không có nhãn trong khoảng đó
    bad = [e for e in runner.events if e["type"] == "bad"]
    assert len(bad) == 1
    txt = cc.summarize(str(tmp_path))
    assert "SIGN" in txt and "TỔNG 1 phiên" in txt
    assert cc.main(["--summary", str(tmp_path)]) == 0


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    rc = cc.main(["--signer", "V98", "--dry-run", "--out", str(tmp_path), "--pairs", "2"])
    assert rc == 0
    assert not any(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert "SIGN" in out and "WANDER" in out


def test_derive_labels_pair_two_marks_and_missing_marks():
    t = np.arange(0.0, 30.0, 0.05)
    evs = [
        _ev(0.0, "prompt_begin", 1, 0, kind="PAIR", words=["ME", "YOU"]),
        _ev(1.0, "sign_start", 1, 0), _ev(3.0, "boundary", 1, 0, index=0), _ev(4.0, "boundary", 1, 0, index=1),
        _ev(6.0, "sign_end", 1, 0), _ev(7.0, "prompt_end", 1, 0),
        _ev(10.0, "prompt_begin", 2, 1, kind="PAIR", words=["EAT", "YES"]),
        _ev(11.0, "sign_start", 2, 1), _ev(14.0, "sign_end", 2, 1), _ev(15.0, "prompt_end", 2, 1),
    ]
    m = 0.15
    lab = cc.derive_frame_labels(t, {"events": evs}, margin_sec=m)

    def at(x):
        return lab[int(round(x / 0.05))]

    assert at(2.0) == cc.SIGN                                   # từ 1
    assert at(3.0) == cc.IGNORE and at(3.1) == cc.IGNORE        # quanh điểm bấm b0
    assert at(3.5) == cc.TRANSITION and at(3.3) == cc.TRANSITION  # cả đoạn b0..b1 là chuyển tiếp
    assert at(4.0) == cc.IGNORE and at(4.1) == cc.IGNORE        # quanh điểm bấm b1
    assert at(5.0) == cc.SIGN                                   # từ 2
    # PAIR không bấm m: bỏ cả đoạn (IGNORE) thay vì gán nhầm SIGN
    assert set(np.unique(lab[(t >= 10.0) & (t <= 15.0)])) == {cc.IGNORE}
