"""
TEST SPEC 08 - cấu hình "ký nhanh": make_segmenter, tập từ nhiều pha, điều kiện chốt sớm, giao diện.
Mọi thứ mới phải TẮT theo mặc định và không đổi hành vi cũ.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from dynamic_dtw_engine import KineticPauseSegmenter
from pipeline_common import SHARED_CONFIG, early_commit_ok, find_multi_phase_words, make_segmenter


# ------------------------------------------------------------------ make_segmenter
class TestMakeSegmenter:
    def test_mac_dinh_giu_nguyen_tham_so_cu(self):
        seg = make_segmenter(SHARED_CONFIG)
        assert isinstance(seg, KineticPauseSegmenter)
        assert (seg.v_start_ratio, seg.v_stop_ratio, seg.pause_frames) == (0.08, 0.040, 9)
        assert (seg.min_stroke_frames, seg.max_stroke_frames, seg.max_phase_merges) == (8, 60, 1)
        assert seg.grace_window_sec == 0.15 and seg.require_ready_pose is True
        assert seg.pause_sec is None and seg.ready_stable_sec is None and seg.min_stroke_sec is None and seg.max_stroke_sec is None
        assert seg.chain_window_sec == 0.0

    def test_fast_none_hoac_rong_khong_doi(self):
        a, b, c = make_segmenter(SHARED_CONFIG), make_segmenter(SHARED_CONFIG, None), make_segmenter(SHARED_CONFIG, {})
        for s in (b, c):
            assert vars(s).keys() == vars(a).keys()
            assert all(getattr(s, k) == getattr(a, k) for k in ("pause_frames", "grace_window_sec", "pause_sec", "chain_window_sec"))

    def test_fast_dat_tham_so_giay(self):
        cfg = {"pause_sec": 0.12, "grace_window_sec": 0.05, "chain_window_sec": 1.2, "min_stroke_sec": 0.10}
        seg = make_segmenter(SHARED_CONFIG, cfg)
        assert seg.pause_sec == 0.12 and seg.grace_window_sec == 0.05 and seg.chain_window_sec == 1.2 and seg.min_stroke_sec == 0.10
        assert seg.ready_stable_sec is None and seg.max_stroke_sec is None
        assert seg.pause_frames == 9            # số khung cũ vẫn còn đó, chỉ bị pause_sec che khi quy đổi

    def test_config_predict_mac_dinh_bat(self):
        """Từ 2026-09-21 chế độ ký nhanh + cắt dẫn đầu là bản chuẩn (BẬT mặc định); --no-fast-chain quay về hành vi cũ."""
        import predict_sentence_webcam as psw
        fc = psw.CONFIG["fast_chain"]
        assert fc["enabled"] is True
        assert fc["lead_trim"]["enabled"] is True
        assert fc["transition_lockout_sec"] == 0.0     # khoảng cấm cố định vẫn TẮT (chưa được thử trực tiếp)
        for k in ("chain_window_sec", "pause_sec", "grace_window_sec", "early_commit", "early_min_quality", "early_min_margin",
                  "multi_phase_words", "tail_pad_sec", "min_stroke_sec"):
            assert k in fc


# ------------------------------------------------------------------ từ nhiều pha
def _traj(step_profile):
    """Chuỗi K=32 khung với tốc độ (đơn vị palm/bước) cho trước (31 bước)."""
    step = np.asarray(step_profile, dtype=np.float32)
    x = np.concatenate([[0.0], np.cumsum(step)])
    t = np.zeros((32, 24), dtype=np.float32)
    t[:, 0] = x
    return t.tolist()


class TestMultiPhaseWords:
    def test_tu_mot_pha_khong_bi_gan_co(self):
        moving = [0.10] * 20 + [0.0] * 11                     # chuyển động rồi đứng yên tới hết (đuôi yên) -> KHÔNG phải nhiều pha
        vocab = {"ONE": {"samples": [{"traj_features": _traj(moving)}]}}
        assert find_multi_phase_words(vocab) == {}

    def test_nhip_dung_o_giua_bi_phat_hien(self):
        prof = [0.10] * 8 + [0.0] * 6 + [0.10] * 6 + [0.0] * 11   # đi - dừng 6 bước - đi tiếp - đuôi yên
        vocab = {"TWO": {"samples": [{"traj_features": _traj(prof)}, {"traj_features": _traj([0.1] * 20 + [0.0] * 11)}]}}
        assert find_multi_phase_words(vocab) == {"TWO": 1}

    def test_dung_qua_ngan_hoac_ngoai_cua_so_khong_tinh(self):
        short = [0.10] * 8 + [0.0] * 3 + [0.10] * 9 + [0.0] * 11      # dừng chỉ 3 bước < 4
        early = [0.0] * 5 + [0.10] * 14 + [0.0] * 12                  # dừng ở đầu (trước 20%): không phải nhịp giữa
        vocab = {"S": {"samples": [{"traj_features": _traj(short)}]}, "E": {"samples": [{"traj_features": _traj(early)}]}}
        assert find_multi_phase_words(vocab) == {}

    def test_bank_that_chay_duoc(self):
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webcam_templates.json")
        if not os.path.exists(path):
            pytest.skip("không có webcam_templates.json")
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)["vocabulary"]
        out = find_multi_phase_words(vocab)
        assert isinstance(out, dict) and set(out) <= set(vocab)


# ------------------------------------------------------------------ điều kiện chốt sớm
def _match(status="ACCEPTED", word="HELLO", quality=90.0, d1=0.20, d2=0.40):
    cands = [SimpleNamespace(word_key=word, distance=d1), SimpleNamespace(word_key="ME", distance=d2)]
    return SimpleNamespace(status=status, top_word=word, match_quality=quality, candidates=cands)


class TestEarlyCommitOk:
    cfg = {"early_commit": True, "early_min_quality": 80, "early_min_margin": 0.25}

    def test_du_dieu_kien(self):
        assert early_commit_ok(_match(), self.cfg, ["EAT"]) is True

    def test_tat_hoac_thieu_cau_hinh(self):
        assert early_commit_ok(_match(), dict(self.cfg, early_commit=False), []) is False
        assert early_commit_ok(_match(), {}, []) is False
        assert early_commit_ok(_match(), None, []) is False

    @pytest.mark.parametrize("status", ["UNCERTAIN", "REJECTED"])
    def test_khong_phai_accepted(self, status):
        assert early_commit_ok(_match(status=status), self.cfg, []) is False

    def test_chat_luong_thap(self):
        assert early_commit_ok(_match(quality=79.9), self.cfg, []) is False
        assert early_commit_ok(_match(quality=80.0), self.cfg, []) is True

    def test_margin_thap(self):
        assert early_commit_ok(_match(d1=0.20, d2=0.24), self.cfg, []) is False          # (0.24-0.20)/0.20 = 0.2 < 0.25
        assert early_commit_ok(_match(d1=0.20, d2=0.26), self.cfg, []) is True           # 0.30 >= 0.25 (tránh biên số thực)

    def test_tu_nhieu_pha_khong_chot_som(self):
        assert early_commit_ok(_match(word="THANK_YOU"), self.cfg, ["THANK_YOU"]) is False

    def test_mot_ung_vien_duy_nhat(self):
        m = _match()
        m.candidates = m.candidates[:1]
        assert early_commit_ok(m, self.cfg, []) is True


# ------------------------------------------------------------------ giao diện
class TestUiChainReady:
    def test_chuoi_song_ngu(self):
        import ui_render
        assert ui_render.tr("vi", "chip_chain_ready") == "Ký tiếp được rồi"
        assert ui_render.tr("en", "chip_chain_ready") == "Ready — keep signing"
        assert ui_render.tr("vi", "key_j") and ui_render.tr("en", "key_j")
