"""
TEST HỒI QUY SPEC 08 - BỘ PHÂN ĐOẠN "KÝ NHANH" (KineticPauseSegmenter)
=====================================================================
1. Với MỌI tham số mới ở mặc định (pause_sec/ready_stable_sec/min_stroke_sec/max_stroke_sec = None,
   chain_window_sec = 0), KineticPauseSegmenter.update cho kết quả Y HỆT bản trước SPEC 08 trên >= 200 luồng
   (luồng thật ghép từ raw_strokes + luồng ngẫu nhiên có seed): so state, completed_stroke (từng trường của khung),
   is_swipe và v_norm của TỪNG khung. "Bản vàng" là bản sao nguyên văn của lớp cũ (_OldKineticPauseSegmenter)
   nằm ngay trong file này (sao từ dynamic_dtw_engine.py trước khi sửa).
2. Test riêng cho quy đổi giây <-> khung, chain_window_sec và force_commit.
"""

import glob
import json
import math
import os
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import pytest

from dynamic_dtw_engine import (
    KineticPauseSegmenter,
    KineticState,
    GestureFrame,
    STROKE_DURATION_MAX_SEC,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# BẢN SAO NGUYÊN VĂN LỚP CŨ (trước SPEC 08) - KHÔNG SỬA
# =============================================================================

class _OldKineticPauseSegmenter:
    """
    TẦNG 2: Bộ Dò Điểm Dừng & Phân Đoạn theo Tỷ Lệ Lòng Bàn Tay (Palm Scale).
    - Đo vận tốc chuẩn hóa không thứ nguyên: v_norm = Δw / palm_len.
    - Quản lý FSM 4 trạng thái:
        [IDLE] -> [RECORDING] -> [PENDING_COMMIT] -> (RECORDING gộp lại | IDLE chốt cử chỉ)
      Khi tay khựng đủ lâu (pause_frames), FSM KHÔNG chốt cử chỉ ngay mà chuyển sang
      PENDING_COMMIT — một cửa sổ chờ xác nhận (grace_window_sec, mặc định 0.20s).
      Trong lúc chờ, nếu tay cử động lại rõ rệt (v_norm >= v_start_ratio) thì GỘP pha
      mới vào cùng 1 cử chỉ liên tục (quay lại RECORDING, không mất buffer) — phục vụ
      các từ vựng đa pha (VD: "Thank You" chạm cằm rồi dừng nhẹ trước khi đẩy tay ra).
      Nếu hết grace window mà tay vẫn đứng yên, cử chỉ mới thực sự được CHỐT và FSM về IDLE.
    - Bắt lệnh ưu tiên Swipe-Left (Backspace) tức thời mà không cần điểm dừng, nhưng chỉ
      kích hoạt khi buffer cử chỉ hiện tại còn NGẮN (<= swipe_max_buffer_frames, mặc định
      10 khung ~0.33s) — tránh cắt ngang từ vựng thật đã RECORDING lâu (VD: Goodbye vẫy tay
      có thành phần chuyển động ngang-trái mạnh).
    """
    def __init__(
        self,
        v_start_ratio: float = 0.08,      # Bắt đầu cử chỉ khi v >= 8% lòng bàn tay/frame
        v_stop_ratio: float = 0.035,      # Khựng tay khi v <= 3.5% lòng bàn tay/frame
        pause_frames: int = 11,           # Khựng liên tục 11 frames (~0.35s ở 30fps)
        min_stroke_frames: int = 8,       # Cử chỉ tối thiểu 8 frames (~0.27s)
        max_stroke_frames: int = 60,      # Cử chỉ tối đa 60 frames (~2.0s)
        swipe_threshold: float = -0.80,   # Quét sang trái >= 80% lòng bàn tay trong 6 frames
        swipe_ratio: float = 2.5,         # |Δx| / (|Δy| + ε) >= 2.5
        swipe_cooldown_sec: float = 1.0,  # Khóa lệnh xóa 1.0s sau mỗi lần kích hoạt
        grace_window_sec: float = 0.20,     # Cửa sổ chờ gộp cử chỉ đa pha (VD: "Thank You" có 2 pha)
        swipe_max_buffer_frames: int = 10,  # Chỉ cho Swipe-Left kích hoạt khi buffer còn ngắn (≤ ~0.33s)
        require_ready_pose: bool = False,     # Mac dinh False = giu nguyen FSM cu cho tests
        ready_stable_frames: int = 8,         # Tay phai dung yen 8 frame (~0.25s @30fps) moi coi la san sang
        max_phase_merges: int = 999,   # Mac dinh 999 = gan nhu vo han = giu hanh vi cu
    ):
        self.v_start_ratio = v_start_ratio
        self.v_stop_ratio = v_stop_ratio
        self.pause_frames = pause_frames
        self.min_stroke_frames = min_stroke_frames
        self.max_stroke_frames = max_stroke_frames
        self.swipe_threshold = swipe_threshold
        self.swipe_ratio = swipe_ratio
        self.swipe_cooldown_sec = swipe_cooldown_sec
        self.grace_window_sec = grace_window_sec
        self.swipe_max_buffer_frames = swipe_max_buffer_frames
        self.require_ready_pose = require_ready_pose
        self.ready_stable_frames = ready_stable_frames
        self.max_phase_merges = max_phase_merges

        self.state = KineticState.IDLE
        self.buffer: List[GestureFrame] = []
        self.pause_counter = 0
        self.prev_wrist: Optional[Tuple[float, float]] = None
        self.last_swipe_time = -999.0
        self.recent_wrists = deque(maxlen=6)   # Cửa sổ trượt 6 frames phát hiện Swipe
        self._pending_deadline: float = 0.0
        self.ready_counter = 0
        self.merge_count = 0

    def update(
        self,
        wrist_x: float,
        wrist_y: float,
        palm_len: float,
        hand_shape: str = "Unknown",
        timestamp: Optional[float] = None
    ) -> Tuple[KineticState, Optional[List[GestureFrame]], bool, float]:
        """
        Cập nhật từng khung hình camera.

        Returns:
            current_state: KineticState (IDLE, RECORDING, PENDING_COMMIT, PAUSE_DETECTED)
            completed_stroke: List[GestureFrame] nếu cử chỉ vừa hoàn tất, ngược lại None
            is_swipe_left: bool (True nếu phát hiện lệnh Backspace tức thời)
            v_norm: float (Vận tốc chuẩn hóa hiện tại)
        """
        if timestamp is None:
            timestamp = time.time()

        safe_palm = max(palm_len, 15.0)
        curr_frame = GestureFrame(
            x=wrist_x, y=wrist_y,
            palm_len=safe_palm,
            hand_shape=hand_shape,
            timestamp=timestamp
        )

        # 1. Tính vận tốc chuẩn hóa theo cơ thể
        if self.prev_wrist is not None:
            dx = wrist_x - self.prev_wrist[0]
            dy = wrist_y - self.prev_wrist[1]
            dist_px = math.hypot(dx, dy)
            v_norm = dist_px / safe_palm
        else:
            v_norm = 0.0
            dx, dy = 0.0, 0.0

        self.prev_wrist = (wrist_x, wrist_y)
        self.recent_wrists.append((wrist_x, wrist_y, dx, dy, safe_palm, timestamp))

        # 2. KIỂM TRA LỆNH ƯU TIÊN: SWIPE-LEFT (BACKSPACE) TỨC THỜI
        #    R6 FIX: Chỉ cho phép kích hoạt khi buffer cử chỉ hiện tại còn NGẮN
        #    (<= swipe_max_buffer_frames, mặc định 10 khung ~0.33s). Một cú quẹt
        #    thật luôn hoàn tất đủ tiêu chí hình học trong 5-6 khung đầu tiên (vì
        #    peak_vx phải >= 2.5x v_start_ratio), nên điều kiện này KHÔNG làm mất
        #    khả năng bắt Swipe thật. Nó chỉ tắt Swipe sau khi một từ vựng thật đã
        #    được RECORDING lâu (buffer dài), để tránh cắt ngang từ có thành phần
        #    chuyển động ngang-trái mạnh (VD: Goodbye vẫy tay).
        #    Lưu ý: chặn cứng theo state==IDLE (thử nghiệm ban đầu) đã bị loại bỏ vì
        #    nó vô hiệu hóa Swipe hoàn toàn — state luôn rời IDLE ngay khung đầu của
        #    chính cú quẹt do v_start_ratio (0.08) thấp hơn nhiều peak_vx của Swipe (0.20).
        is_swipe_left = False
        swipe_buffer_ok = len(self.buffer) <= self.swipe_max_buffer_frames
        if swipe_buffer_ok and len(self.recent_wrists) >= 5 and (timestamp - self.last_swipe_time) >= self.swipe_cooldown_sec:
            old_x, old_y, _, _, ref_palm, _ = self.recent_wrists[0]
            delta_x_total = (wrist_x - old_x) / ref_palm
            delta_y_total = (wrist_y - old_y) / ref_palm

            # Kiểm tra 3 tiêu chí: Quãng đường âm mạnh + Áp đảo ngang + Tốc độ bứt phá
            peak_vx = min(item[2] / item[4] for item in self.recent_wrists)
            if (
                delta_x_total <= self.swipe_threshold
                and (abs(delta_x_total) / (abs(delta_y_total) + 1e-4)) >= self.swipe_ratio
                and peak_vx <= -0.20
            ):
                is_swipe_left = True
                self.last_swipe_time = timestamp
                self.reset()  # Reset bộ đệm để không lưu cú quẹt vào từ vựng
                return KineticState.IDLE, None, True, v_norm

        # 3. MÁY TRẠNG THÁI HỮU HẠN (FSM) CHO CỬ CHỈ TỪ VỰNG
        completed_stroke: Optional[List[GestureFrame]] = None

        if self.state == KineticState.IDLE:
            if not self.require_ready_pose:
                if v_norm >= self.v_start_ratio:
                    self.state = KineticState.RECORDING
                    self.buffer = [curr_frame]
                    self.pause_counter = 0
            else:
                if v_norm <= self.v_stop_ratio:
                    self.ready_counter += 1
                    if self.ready_counter >= self.ready_stable_frames:
                        self.state = KineticState.READY
                else:
                    self.ready_counter = 0

        elif self.state == KineticState.READY:
            if v_norm >= self.v_start_ratio:
                self.state = KineticState.RECORDING
                self.buffer = [curr_frame]
                self.pause_counter = 0

        elif self.state == KineticState.RECORDING:
            self.buffer.append(curr_frame)

            if v_norm <= self.v_stop_ratio:
                self.pause_counter += 1
            else:
                self.pause_counter = 0

            if self.pause_counter >= self.pause_frames:
                # R7 FIX: Vừa khựng đủ lâu — KHÔNG chốt ngay. Chuyển sang chờ
                # xác nhận (grace window) để tránh cắt đôi cử chỉ đa pha (VD:
                # "Thank You" chạm cằm rồi dừng nhẹ trước khi đẩy tay ra).
                self.state = KineticState.PENDING_COMMIT
                self._pending_deadline = timestamp + self.grace_window_sec
                self.pause_counter = 0
            elif len(self.buffer) > self.max_stroke_frames:
                self.reset()

        elif self.state == KineticState.PENDING_COMMIT:
            # Vẫn tiếp tục ghi khung hình trong lúc chờ, để không mất dữ liệu
            # nếu tay cử động lại (gộp pha 2 vào cùng 1 cử chỉ).
            self.buffer.append(curr_frame)

            if v_norm >= self.v_start_ratio:
                if self.merge_count < self.max_phase_merges and (timestamp - self.buffer[0].timestamp) <= STROKE_DURATION_MAX_SEC:
                    # Tay cử động lại rõ rệt trong lúc chờ -> GỘP làm 1 cử chỉ liên tục
                    self.state = KineticState.RECORDING
                    self.pause_counter = 0
                    self.merge_count += 1
                else:
                    total_n = len(self.buffer)
                    if self.min_stroke_frames <= total_n <= self.max_stroke_frames:
                        completed_stroke = list(self.buffer)
                    self.buffer.clear()
                    self.pause_counter = 0
                    self.state = KineticState.IDLE
                    self.merge_count = 0
            elif timestamp >= self._pending_deadline:
                # Hết thời gian chờ, tay vẫn đứng yên -> CHỐT CỬ CHỈ THẬT SỰ
                total_n = len(self.buffer)
                if self.min_stroke_frames <= total_n <= self.max_stroke_frames:
                    completed_stroke = list(self.buffer)
                self.buffer.clear()
                self.pause_counter = 0
                self.state = KineticState.IDLE
                self.merge_count = 0
            elif len(self.buffer) > self.max_stroke_frames:
                self.reset()

        if completed_stroke is not None:
            self.merge_count = 0

        return self.state, completed_stroke, is_swipe_left, v_norm

    def reset(self):
        """Khởi tạo lại toàn bộ trạng thái bộ đệm."""
        self.state = KineticState.IDLE
        self.buffer.clear()
        self.pause_counter = 0
        self.recent_wrists.clear()
        self._pending_deadline = 0.0
        self.prev_wrist = None
        self.ready_counter = 0
        self.merge_count = 0



# =============================================================================
# LUỒNG THỬ
# =============================================================================
# Một "khung" thử: (x, y, palm_len, hand_shape, timestamp)
Frame = Tuple[float, float, float, str, float]


def _real_streams(max_streams: int = 120, seed: int = 20260920) -> List[List[Frame]]:
    """Luồng THẬT: ghép 1-3 stroke thô trong raw_strokes (cổ tay đã làm mượt, palm_len lưu sẵn), xen kẽ đoạn đứng yên."""
    rng = np.random.default_rng(seed)
    files = [f for f in glob.glob(os.path.join(ROOT, "raw_strokes", "*", "*.json"))
             if os.path.basename(os.path.dirname(f)) != "NEGATIVE"]
    files.sort()
    if not files:
        pytest.skip("không có raw_strokes")
    cache = {}

    def load(f):
        if f not in cache:
            with open(f, "r", encoding="utf-8") as fh:
                r = json.load(fh)
            cache[f] = [(float(fr["wrist_smoothed"][0]), float(fr["wrist_smoothed"][1]),
                         float(fr["palm_len"]), fr.get("hand_shape", "Unknown")) for fr in r["frames"]]
        return cache[f]

    streams = []
    for _ in range(max_streams):
        dt = float(rng.choice([1 / 30.0, 1 / 15.0, 1 / 24.0]))
        t = 1000.0 + float(rng.random())
        out: List[Frame] = []
        cx, cy = 300.0, 250.0
        for _w in range(int(rng.integers(1, 4))):
            still = int(rng.integers(3, 30))
            for _ in range(still):
                out.append((cx + float(rng.normal(0, 0.2)), cy + float(rng.normal(0, 0.2)), 100.0, "Unknown", t))
                t += dt
            fr = load(files[int(rng.integers(0, len(files)))])
            ox, oy = fr[0][0], fr[0][1]
            for (x, y, p, sh) in fr:
                out.append((cx + (x - ox), cy + (y - oy), p, sh, t))
                t += dt
            cx, cy = out[-1][0], out[-1][1]
        for _ in range(int(rng.integers(5, 40))):
            out.append((cx + float(rng.normal(0, 0.2)), cy + float(rng.normal(0, 0.2)), 100.0, "Unknown", t))
            t += dt
        streams.append(out)
    return streams


def _random_streams(n: int = 100, seed: int = 7) -> List[List[Frame]]:
    """Luồng NGẪU NHIÊN có seed: các đoạn đứng yên / chuyển động chậm / nhanh / giật (kể cả quẹt trái), nhịp khung không đều."""
    rng = np.random.default_rng(seed)
    streams = []
    for _ in range(n):
        t = 500.0
        x, y = 320.0, 240.0
        out: List[Frame] = []
        palm = float(rng.uniform(40, 160))
        for _seg in range(int(rng.integers(3, 12))):
            kind = int(rng.integers(0, 5))
            length = int(rng.integers(3, 40))
            vx, vy = float(rng.normal(0, 1)), float(rng.normal(0, 1))
            if kind == 4:                      # quẹt trái mạnh
                vx, vy = -2.5, 0.0
            speed = {0: 0.0, 1: 0.01, 2: 0.06, 3: 0.15, 4: 0.30}[kind] * palm
            nrm = math.hypot(vx, vy) or 1.0
            for _ in range(length):
                x += vx / nrm * speed + float(rng.normal(0, 0.1))
                y += vy / nrm * speed + float(rng.normal(0, 0.1))
                if rng.random() < 0.02:
                    palm = float(rng.uniform(10, 160))
                out.append((x, y, palm, "B" if rng.random() < 0.5 else "Unknown", t))
                t += float(rng.choice([1 / 30.0, 1 / 30.0, 1 / 15.0, 1 / 60.0, 0.0]))   # cả khung trùng dấu thời gian
        streams.append(out)
    return streams


def _frame_key(f: GestureFrame):
    return (f.x, f.y, f.palm_len, f.hand_shape, f.timestamp)


def _run(seg, stream: List[Frame]):
    out = []
    for (x, y, p, sh, t) in stream:
        state, stroke, swipe, v = seg.update(x, y, p, sh, t)
        out.append((state, None if stroke is None else [_frame_key(f) for f in stroke], swipe, v))
    return out


def _production_kwargs():
    """Tham số của make_segmenter(CONFIG) hiện tại (bộ phân đoạn production)."""
    return dict(v_start_ratio=0.08, v_stop_ratio=0.040, pause_frames=9, min_stroke_frames=8, max_stroke_frames=60,
                swipe_threshold=-0.80, swipe_ratio=2.5, swipe_cooldown_sec=1.0, require_ready_pose=True,
                max_phase_merges=1, grace_window_sec=0.15)


# =============================================================================
# 1. HỒI QUY BIT-GIỐNG-HỆT Ở MẶC ĐỊNH
# =============================================================================
class TestDefaultsBitIdentical:
    @pytest.mark.parametrize("kw_name", ["production", "class_defaults", "legacy_no_ready"])
    def test_ba_bo_tham_so_giong_het_ban_cu(self, kw_name):
        kw = {"production": _production_kwargs(), "class_defaults": {},
              "legacy_no_ready": dict(require_ready_pose=False)}[kw_name]
        streams = _real_streams() + _random_streams()
        assert len(streams) >= 200
        n_strokes = n_swipes = n_frames = 0
        for i, st in enumerate(streams):
            ref = _run(_OldKineticPauseSegmenter(**kw), st)
            new = _run(KineticPauseSegmenter(**kw), st)
            assert len(ref) == len(new)
            for k, (a, b) in enumerate(zip(ref, new)):
                assert a == b, f"luồng {i} khung {k} lệch: cũ={a[0], a[2], a[3]} mới={b[0], b[2], b[3]}"
            n_strokes += sum(1 for r in ref if r[1] is not None)
            n_swipes += sum(1 for r in ref if r[2])
            n_frames += len(ref)
        # Bộ luồng phải THẬT SỰ có việc: có stroke hoàn tất và có quẹt trái (nếu không thì so sánh rỗng ý nghĩa).
        if kw_name == "production":
            assert n_strokes >= 100, n_strokes
            assert n_swipes >= 5, n_swipes
            assert n_frames > 10000

    def test_reset_giong_het(self):
        st = _real_streams(10)[3]
        a, b = _OldKineticPauseSegmenter(**_production_kwargs()), KineticPauseSegmenter(**_production_kwargs())
        for k, (x, y, p, sh, t) in enumerate(st):
            ra, rb = a.update(x, y, p, sh, t), b.update(x, y, p, sh, t)
            assert (ra[0], ra[2], ra[3]) == (rb[0], rb[2], rb[3])
            if k % 17 == 16:
                a.reset()
                b.reset()


# =============================================================================
# 2. QUY ĐỔI GIÂY <-> KHUNG
# =============================================================================
def _feed_still(seg, n, t0, dt, x=100.0, y=100.0):
    t = t0
    for _ in range(n):
        seg.update(x, y, 100.0, "Unknown", t)
        t += dt
    return t


class TestSecToFrames:
    def test_mac_dinh_tra_ve_so_khung_cu(self):
        seg = KineticPauseSegmenter(**_production_kwargs())
        assert seg.effective_pause_frames() == 9
        assert seg.effective_ready_stable_frames() == 8
        assert seg.effective_min_stroke_frames() == 8
        assert seg.effective_max_stroke_frames() == 60

    @pytest.mark.parametrize("hz,expect", [(30.0, 6), (15.0, 3), (60.0, 12), (10.0, 2)])
    def test_pause_sec_theo_nhip_khung_that(self, hz, expect):
        seg = KineticPauseSegmenter(pause_sec=0.20)
        assert seg.effective_pause_frames() == 6            # chưa có dữ liệu -> giả định 30 khung/giây
        _feed_still(seg, 40, 10.0, 1.0 / hz)
        assert seg.effective_pause_frames() == expect

    def test_kep_trong_khoang(self):
        seg = KineticPauseSegmenter(pause_sec=0.20)
        _feed_still(seg, 40, 10.0, 1.0 / 200.0)             # 200 Hz -> kẹp dt >= 1/60
        assert seg.effective_pause_frames() == 12
        seg2 = KineticPauseSegmenter(pause_sec=0.20)
        _feed_still(seg2, 10, 10.0, 0.2)                    # 5 Hz -> khoảng > 0.25? không, 0.2 <= 0.25 nên tính, kẹp dt <= 1/8
        assert seg2.effective_pause_frames() == 2

    def test_khoang_trong_mat_tay_khong_lam_lech_dt(self):
        seg = KineticPauseSegmenter(pause_sec=0.20)
        t = _feed_still(seg, 30, 10.0, 1 / 30.0)
        before = seg.effective_pause_frames()
        seg.update(100.0, 100.0, 100.0, "Unknown", t + 2.0)   # tay biến mất 2 giây rồi quay lại
        assert seg.effective_pause_frames() == before

    def test_khung_trung_dau_thoi_gian_khong_lam_lech_dt(self):
        seg = KineticPauseSegmenter(pause_sec=0.20)
        t = 10.0
        for _ in range(40):
            seg.update(100.0, 100.0, 100.0, "Unknown", t)
            seg.update(100.0, 100.0, 100.0, "Unknown", t)     # cùng dấu thời gian
            t += 1 / 30.0
        assert seg.effective_pause_frames() == 6

    def test_chot_dung_thoi_gian_o_15hz_va_30hz(self):
        """pause_sec=0.20 => PENDING sau ~0.2 s dù nhịp khung là 30 hay 15."""
        for hz in (30.0, 15.0):
            dt = 1.0 / hz
            seg = KineticPauseSegmenter(**dict(_production_kwargs(), pause_frames=9), pause_sec=0.20, ready_stable_sec=0.20)
            t = _feed_still(seg, 30, 10.0, dt)
            assert seg.state == KineticState.READY
            for k in range(int(0.6 * hz)):                  # 0.6 s chuyển động
                seg.update(100.0 + 12.0 * (k + 1), 100.0, 100.0, "Unknown", t)
                t += dt
            assert seg.state == KineticState.RECORDING
            x_end = 100.0 + 12.0 * int(0.6 * hz)
            t_stop = t - dt
            while seg.state == KineticState.RECORDING:
                seg.update(x_end, 100.0, 100.0, "Unknown", t)
                t += dt
            waited = (t - dt) - t_stop
            assert 0.2 - 1e-6 <= waited <= 0.2 + dt + 1e-6, (hz, waited)


# =============================================================================
# 3. CHAIN WINDOW
# =============================================================================
def _sign_then_still(seg, t0, dt, moving=14, still=40):
    """READY -> ký (chuyển động rõ) -> đứng yên tới khi chốt. Trả về (t, stroke_or_None, t_commit)."""
    t = t0
    x = 100.0
    stroke, t_commit = None, None
    for _ in range(moving):
        x += 15.0
        seg.update(x, 100.0, 100.0, "Unknown", t)
        t += dt
    for _ in range(still):
        s, st, _sw, _v = seg.update(x, 100.0, 100.0, "Unknown", t)
        if st is not None and stroke is None:
            stroke, t_commit = st, t
        t += dt
        if stroke is not None:
            break
    return t, x, stroke, t_commit


class TestChainWindow:
    def _armed(self, **kw):
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), **kw))
        t = _feed_still(seg, 20, 10.0, 1 / 30.0)
        assert seg.state == KineticState.READY
        return seg, t

    def test_tat_thi_chuyen_dong_ngay_sau_chot_bi_bo_qua(self):
        seg, t = self._armed()
        t, x, stroke, _ = _sign_then_still(seg, t, 1 / 30.0)
        assert stroke is not None
        # Tay ký tiếp NGAY (không đứng yên): không được RECORDING vì chưa ổn định lại
        for _ in range(4):
            x += 15.0
            state, st, _s, _v = seg.update(x, 100.0, 100.0, "Unknown", t)
            t += 1 / 30.0
        assert state == KineticState.IDLE

    def test_bat_thi_chuyen_dong_ngay_sau_chot_bat_dau_ghi(self):
        seg, t = self._armed(chain_window_sec=1.2)
        t, x, stroke, _ = _sign_then_still(seg, t, 1 / 30.0)
        assert stroke is not None
        x += 15.0
        state, _st, _s, v = seg.update(x, 100.0, 100.0, "Unknown", t)
        assert v >= 0.08
        assert state == KineticState.RECORDING
        assert len(seg.buffer) == 1

    def test_het_cua_so_ve_hanh_vi_cu(self):
        seg, t = self._armed(chain_window_sec=0.5)
        t, x, stroke, t_commit = _sign_then_still(seg, t, 1 / 30.0)
        assert stroke is not None
        # Tay chuyển động NHẸ (giữa v_stop và v_start) suốt cửa sổ, rồi hết hạn
        t = t_commit + 0.6
        x += 15.0
        # ngoài cửa sổ: chuyển động rõ rệt ở trạng thái IDLE bị bỏ qua như cũ
        state, _st, _s, _v = seg.update(x, 100.0, 100.0, "Unknown", t)
        assert state == KineticState.IDLE

    def test_reset_dong_cua_so(self):
        seg, t = self._armed(chain_window_sec=1.2)
        t, x, stroke, _ = _sign_then_still(seg, t, 1 / 30.0)
        assert stroke is not None
        seg.reset()
        x += 15.0
        state, *_ = seg.update(x, 100.0, 100.0, "Unknown", t)
        x += 15.0
        state, *_ = seg.update(x, 100.0, 100.0, "Unknown", t + 1 / 30.0)
        assert state == KineticState.IDLE

    def test_chuyen_dong_nhe_khong_lam_mat_san_sang(self):
        seg, t = self._armed(chain_window_sec=1.2)
        t, x, stroke, _ = _sign_then_still(seg, t, 1 / 30.0)
        assert stroke is not None
        x += 5.0                                              # v = 0.05: giữa v_stop (0.04) và v_start (0.08)
        seg.update(x, 100.0, 100.0, "Unknown", t)
        assert seg.ready_counter >= seg.ready_stable_frames
        x += 15.0
        state, *_ = seg.update(x, 100.0, 100.0, "Unknown", t + 1 / 30.0)
        assert state == KineticState.RECORDING


# =============================================================================
# 4. FORCE COMMIT
# =============================================================================
def _drive_to_pending(seg, dt=1 / 30.0):
    t = _feed_still(seg, 20, 10.0, dt)
    x = 100.0
    for _ in range(14):
        x += 15.0
        seg.update(x, 100.0, 100.0, "Unknown", t)
        t += dt
    while seg.state != KineticState.PENDING_COMMIT:
        seg.update(x, 100.0, 100.0, "Unknown", t)
        t += dt
    return t, x


class TestForceCommit:
    def test_ngoai_pending_tra_none_khong_doi_trang_thai(self):
        seg = KineticPauseSegmenter(**_production_kwargs())
        assert seg.force_commit(1.0) is None
        assert seg.state == KineticState.IDLE
        _feed_still(seg, 20, 10.0, 1 / 30.0)
        assert seg.state == KineticState.READY
        assert seg.force_commit(11.0) is None
        assert seg.state == KineticState.READY
        seg.update(115.0, 100.0, 100.0, "Unknown", 11.0)
        assert seg.state == KineticState.RECORDING
        n = len(seg.buffer)
        assert seg.force_commit(11.1) is None
        assert seg.state == KineticState.RECORDING and len(seg.buffer) == n

    def test_trong_pending_tra_ve_stroke_va_ve_idle(self):
        seg = KineticPauseSegmenter(**_production_kwargs())
        t, x = _drive_to_pending(seg)
        buf = [_frame_key(f) for f in seg.buffer]
        stroke = seg.force_commit(t)
        assert stroke is not None
        assert [_frame_key(f) for f in stroke] == buf
        assert seg.state == KineticState.IDLE
        assert seg.buffer == []

    def test_khong_chot_trung_cung_mot_stroke(self):
        seg = KineticPauseSegmenter(**_production_kwargs())
        t, x = _drive_to_pending(seg)
        assert seg.force_commit(t) is not None
        assert seg.force_commit(t) is None                    # gọi lần hai: đã IDLE
        for _ in range(60):                                   # hết grace cũng không phát lại stroke
            _s, st, _sw, _v = seg.update(x, 100.0, 100.0, "Unknown", t)
            assert st is None
            t += 1 / 30.0

    def test_buffer_ngoai_khoang_tra_none(self):
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), min_stroke_frames=200, max_stroke_frames=300))
        t, x = _drive_to_pending(seg)
        assert seg.state == KineticState.PENDING_COMMIT
        assert seg.force_commit(t) is None
        assert seg.state == KineticState.PENDING_COMMIT

    def test_ap_dung_cua_so_noi(self):
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), chain_window_sec=1.2))
        t, x = _drive_to_pending(seg)
        assert seg.force_commit(t) is not None
        x += 15.0
        state, *_ = seg.update(x, 100.0, 100.0, "Unknown", t + 1 / 30.0)
        assert state == KineticState.RECORDING

    def test_chot_som_nhanh_hon_chot_thuong(self):
        """Chốt sớm xong thì từ kế ghi được ngay; chốt thường phải đợi hết grace."""
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), pause_frames=6, grace_window_sec=0.10))
        t, x = _drive_to_pending(seg)
        t_pending = t - 1 / 30.0        # dấu thời gian của khung ĐẦU TIÊN ở PENDING_COMMIT
        stroke = None
        while stroke is None:
            _s, stroke, _sw, _v = seg.update(x, 100.0, 100.0, "Unknown", t)
            t += 1 / 30.0
        assert (t - 1 / 30.0) - t_pending >= 0.10 - 1e-6   # chốt thường phải đợi hết grace 0.10 s


# =============================================================================
# 4. SPEC 08B: TEST KHOẢNG CẤM CHUYỂN TIẾP (transition_lockout_sec)
# =============================================================================
class TestTransitionLockout:
    def test_chuyen_dong_trong_lockout_bi_bo_qua(self):
        """Trong thời gian T lockout sau chốt: chuyển động mạnh bị bỏ qua, state giữ IDLE, không RECORDING."""
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), transition_lockout_sec=0.20, chain_window_sec=1.2))
        t, x = _drive_to_pending(seg)
        # Chốt stroke tại thời điểm t
        committed = seg.force_commit(t)
        assert committed is not None
        assert seg.state == KineticState.IDLE
        assert seg.in_lockout(t + 0.05)
        assert not seg.chain_window_open(t + 0.05)

        # Trong khoảng [t, t + 0.20): tay cử động mạnh (v_norm >= 0.08)
        # 3 khung ở 30Hz: dt = 0.033, t_curr in [t + 0.033, t + 0.10]
        cur_t = t + 0.033
        x += 20.0
        state, stroke, is_sw, v = seg.update(x, 100.0, 100.0, "Unknown", cur_t)
        assert state == KineticState.IDLE
        assert stroke is None
        assert not is_sw

        cur_t = t + 0.10
        x += 20.0
        state, stroke, is_sw, v = seg.update(x, 100.0, 100.0, "Unknown", cur_t)
        assert state == KineticState.IDLE
        assert stroke is None

    def test_het_lockout_cap_ready_ngay_va_ghi_khi_co_chuyen_dong(self):
        """Hết lockout theo dấu thời gian thật: cấp READY ngay; nếu v >= v_start thì ghi ngay khung đó."""
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), transition_lockout_sec=0.20, chain_window_sec=1.2))
        t, x = _drive_to_pending(seg)
        seg.force_commit(t)

        # Khung tại t + 0.21 (đã qua 0.20s): chuyển động mạnh => bắt đầu RECORDING ngay
        cur_t = t + 0.21
        x += 25.0
        state, stroke, is_sw, v = seg.update(x, 100.0, 100.0, "Unknown", cur_t)
        assert state == KineticState.RECORDING
        assert len(seg.buffer) == 1
        assert not seg.in_lockout(cur_t)

    def test_het_lockout_dung_yen_thi_cap_ready(self):
        """Hết lockout nếu tay đứng yên (v <= v_stop): cấp READY ngay không bắt đợi thêm."""
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), transition_lockout_sec=0.20, chain_window_sec=1.2))
        t, x = _drive_to_pending(seg)
        seg.force_commit(t)

        # Khung tại t + 0.21: đứng yên (x không đổi) => vào READY
        cur_t = t + 0.21
        state, stroke, is_sw, v = seg.update(x, 100.0, 100.0, "Unknown", cur_t)
        assert state == KineticState.READY

    def test_reset_xoa_lockout(self):
        """Mất tay hoặc reset() xóa lockout ngay lập tức."""
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), transition_lockout_sec=0.30))
        t, x = _drive_to_pending(seg)
        seg.force_commit(t)
        assert seg.in_lockout(t + 0.10)

        seg.reset()
        assert not seg.in_lockout(t + 0.10)
        assert seg.state == KineticState.IDLE

    def test_swipe_trong_lockout_hoat_dong_binh_thuong(self):
        """Swipe-Left trong lockout vẫn nhận diện và xóa stroke cũ, không bị nuốt."""
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), transition_lockout_sec=0.30))
        t, x = _drive_to_pending(seg)
        seg.force_commit(t)
        assert seg.in_lockout(t + 0.05)

        # Thực hiện quẹt trái nhanh: >= 5 frames với dx âm mạnh
        swiped = False
        cur_t = t + 0.05
        # Feed vài khung quẹt trái
        for step in range(6):
            cur_t += 0.03
            x -= 40.0
            _s, _st, is_sw, _v = seg.update(x, 100.0, 100.0, "Unknown", cur_t)
            if is_sw:
                swiped = True
                break
        assert swiped, "Swipe-Left phải được phát hiện kể cả khi đang trong lockout"
        assert not seg.in_lockout(cur_t), "Swipe thành công phải reset() và xóa lockout"

    def test_quy_doi_thoi_gian_theo_dau_thoi_gian_khong_deu(self):
        """Lockout đo bằng giây thực, không đếm số khung."""
        seg = KineticPauseSegmenter(**dict(_production_kwargs(), transition_lockout_sec=0.15))
        t, x = _drive_to_pending(seg)
        seg.force_commit(t)

        # Camera giật: chỉ 1 khung sau 0.08s -> vẫn lockout
        cur_t = t + 0.08
        x += 20.0
        state, *_ = seg.update(x, 100.0, 100.0, "Unknown", cur_t)
        assert state == KineticState.IDLE

        # Khung thứ hai nhảy vọt sau 0.16s (tổng 0.24s) -> đã hết lockout
        cur_t = t + 0.24
        x += 20.0
        state, *_ = seg.update(x, 100.0, 100.0, "Unknown", cur_t)
        assert state == KineticState.RECORDING

