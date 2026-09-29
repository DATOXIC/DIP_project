"""
GIÁM SÁT SỨC KHỎE CAMERA (CAMERA HEALTH) — KHUNG MỚI THẬT / ĐỘ SÁNG
====================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

Vấn đề: FPS của vòng lặp (1/thời-gian-mỗi-vòng) KHÔNG phải tốc độ camera. Khi thiếu sáng, webcam tự kéo dài thời gian
phơi sáng nên chỉ ~6-14 khung MỚI mỗi giây, còn `cap.read()` vẫn trả về khung cũ ở những lần đọc còn lại, làm
vòng lặp báo ~27-60 FPS trong khi nhận diện kém. Module này đo hai thứ khác nhau và cảnh báo người dùng:
  * số khung MỚI THẬT mỗi giây (`unique_fps`) so với số lần đọc mỗi giây (`loop_fps`);
  * độ sáng khung hình (trung bình, phân vị 95, tỉ lệ điểm cháy sáng).

Thuần numpy/cv2, KHÔNG phụ thuộc MediaPipe. Chỉ ĐO và CẢNH BÁO — không bao giờ bỏ khung hay đổi nhận diện.

Nhận biết khung trùng (quan trọng):
  * `is_new` = False khi mẫu con 1/4 (mỗi 4x4 điểm lấy 1, xem update()) GIỐNG HỆT từng điểm ảnh (np.array_equal) với khung trước, tức driver
    lặp lại đúng bộ đệm cũ. Cảm biến thật luôn có nhiễu nên khung mới hầu như không bao giờ giống hệt.
    GIẢ ĐỊNH "driver lặp lại đúng bộ đệm" CHƯA được kiểm chứng trên webcam thật: nếu driver trả khung đã nội suy /
    làm mịn thì khung "cũ" vẫn khác vài mức xám và bị tính là MỚI -> `unique_fps` bị ước lượng CAO hơn thực tế.
  * `near_dup_ratio` dùng ngưỡng cũ của camera_diag.py (sai khác tuyệt đối trung bình < 0.5 mức xám trên ảnh xám 64x48).
    Ngưỡng này lẫn với cảnh tĩnh hợp lệ nên CHỈ để hiển thị chẩn đoán (HUD debug), KHÔNG dùng cho cảnh báo.
    Chủ dự án nên đối chiếu hai số trên HUD debug để biết giả định trên đúng hay không.

Cảnh báo (có trễ + giữ tối thiểu, tránh nhấp nháy). Ưu tiên: too_bright > low_light > slow_camera.
  * slow_camera: unique_fps (chỉ tính trên các khung lúc TAY ĐANG CHUYỂN ĐỘNG) < 12 (bad < 8) liên tục >= 2 s;
    cảnh tĩnh không gây báo động giả.
  * low_light: luma_mean (EMA) < 95 (bad < 70) liên tục >= 2 s.
  * too_bright: luma_mean > 215 hoặc sat_frac > ngưỡng, liên tục >= 2 s.
  * Mỗi cảnh báo hiện tối thiểu 3 s; tắt khi trở lại bình thường (ngưỡng thoát nới ~10%) liên tục >= 2 s.
"""

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

# Cấu hình mặc định (bị CONFIG["camera_health"] của predict_sentence_webcam.py ghi đè)
DEFAULT_CAMERA_HEALTH_CONFIG: Dict[str, object] = {
    "enabled": True,
    "window_sec": 2.0,           # cửa sổ trượt tính unique_fps / loop_fps / tỉ lệ trùng
    "unique_fps_warn": 12.0,     # khung mới/giây dưới mức này (lúc tay chuyển động) -> slow_camera
    "unique_fps_bad": 8.0,       # dưới mức này -> mức "bad"
    "luma_low_warn": 95.0,       # độ sáng TB dưới mức này -> low_light
    "luma_low_bad": 70.0,        # dưới mức này -> mức "bad"
    "luma_high": 215.0,          # độ sáng TB trên mức này -> too_bright
    "sat_frac_high": 0.25,       # tỉ lệ điểm >= 250 trên mức này -> too_bright (xem bảng đối chiếu trong báo cáo)
    "persist_sec": 2.0,          # điều kiện phải liên tục ít nhất ngần này giây mới bật cảnh báo
    "show_min_sec": 3.0,         # cảnh báo đã bật hiện tối thiểu ngần này giây
    "clear_sec": 2.0,            # phải bình thường liên tục ngần này giây mới tắt
    "hysteresis": 0.10,          # ngưỡng thoát nới hơn ngưỡng vào ~10%
    "hand_moving_v": 0.03,       # v_norm >= mức này coi là tay đang chuyển động (dùng ở predict_sentence_webcam)
    "moving_hold_sec": 0.35,     # giữ cờ "đang chuyển động" thêm ngần này giây (khung trùng làm v_norm tụt về 0)
    "min_moving_frac": 0.6,      # cần tay chuyển động trong >= 60% cửa sổ mới đánh giá slow_camera
    "luma_tau_sec": 0.5,         # hằng số thời gian của EMA độ sáng
    "warmup_sec": 2.0,           # bỏ qua vài giây đầu: camera đang tự chỉnh phơi sáng (khung đầu rất tối)
    "warn_on_video": False,      # video file: khung trùng là bình thường -> mặc định tắt cảnh báo
}

_LEVELS = np.arange(256, dtype=np.float64)
WARNING_KEYS = ("too_bright", "low_light", "slow_camera")   # theo thứ tự ưu tiên hiển thị


class CameraHealthMonitor:
    """Theo dõi khung mới thật/giây, tỉ lệ trùng và độ sáng; phát cảnh báo có trễ. Xem docstring module."""

    def __init__(self, cfg: Optional[Dict[str, object]] = None, warnings_enabled: bool = True):
        c = dict(DEFAULT_CAMERA_HEALTH_CONFIG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.warnings_enabled = bool(warnings_enabled)
        self.window_sec = float(c["window_sec"])

        self.is_new = True                 # của lần update gần nhất
        self.luma_mean = 0.0               # đã làm mượt EMA
        self.luma_raw = 0.0                # của khung gần nhất
        self.luma_p95 = 0.0
        self.sat_frac = 0.0

        self._prev_sub: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._t_start: Optional[float] = None
        self._t_prev: Optional[float] = None
        self._moving_until = -1e9
        # mẫu: (t, dt, is_new, near_dup, moving)
        self._win: Deque[Tuple[float, float, bool, bool, bool]] = deque()
        self._n_frames = 0
        self._tot_new = 0                  # tích lũy cả phiên (khác cửa sổ trượt): khung mới thật / gần-trùng
        self._tot_near = 0

        # trạng thái từng loại cảnh báo
        self._st: Dict[str, Dict[str, object]] = {
            k: {"active": False, "cond_since": None, "ok_since": None, "active_since": 0.0, "bad": False}
            for k in WARNING_KEYS
        }
        self._status: Optional[Tuple[str, str]] = None
        # số liệu cửa sổ (cập nhật mỗi update)
        self.unique_fps = 0.0
        self.loop_fps = 0.0
        self.dup_ratio = 0.0
        self.near_dup_ratio = 0.0
        self.unique_fps_moving: Optional[float] = None

    # ------------------------------------------------------------------ vào dữ liệu
    def update(self, frame_bgr: np.ndarray, t: float, hand_moving: bool = False) -> None:
        """Gọi MỖI lần đọc được khung (kể cả khung trùng). `t` = dấu thời gian (giây, tăng dần)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        # Mẫu con liên tục trong bộ nhớ (tương đương lấy 1 điểm ảnh mỗi 4x4). cv2.resize NEAREST nhanh hơn ~4 lần
        # so với frame[::4, ::4] rồi so sánh/đổi xám (truy cập bước nhảy chậm); chỉ khác ở độ lệch lấy mẫu (2 thay vì 0).
        fh, fw = frame_bgr.shape[:2]
        sub = cv2.resize(frame_bgr, (max(1, fw // 4), max(1, fh // 4)), interpolation=cv2.INTER_NEAREST)
        if self._prev_sub is None:
            is_new = True
        else:
            is_new = not (sub.shape == self._prev_sub.shape and np.array_equal(sub, self._prev_sub))
        self._prev_sub = sub                 # sub là mảng mới (không dùng chung bộ đệm của driver)

        if sub.ndim == 3:
            gray_full = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        else:
            gray_full = sub
        gray = cv2.resize(gray_full, (64, 48), interpolation=cv2.INTER_AREA)
        if self._prev_gray is None:
            near_dup = False
        else:
            near_dup = float(cv2.mean(cv2.absdiff(gray, self._prev_gray))[0]) < 0.5
        self._prev_gray = gray

        hist = np.bincount(gray.ravel(), minlength=256)
        n = gray.size
        luma_raw = float(np.dot(hist, _LEVELS)) / n
        cum = np.cumsum(hist)
        luma_p95 = float(np.searchsorted(cum, 0.95 * n))
        sat_frac = float(hist[250:].sum()) / n

        self.ingest(t, is_new, luma_raw, luma_p95, sat_frac, hand_moving, near_dup)

    def ingest(self, t: float, is_new: bool, luma: float, luma_p95: float = 0.0, sat_frac: float = 0.0,
               hand_moving: bool = False, near_dup: bool = False) -> None:
        """Đưa số đo đã tính sẵn vào máy trạng thái (dùng trong update() và để phát lại dữ liệu chẩn đoán đã ghi)."""
        c = self.cfg
        if self._t_start is None:
            self._t_start = t
            dt = 0.0
            self.luma_mean = luma
        else:
            dt = max(0.0, t - (self._t_prev if self._t_prev is not None else t))
            alpha = 1.0 - math.exp(-dt / max(1e-3, float(c["luma_tau_sec"])))
            self.luma_mean += alpha * (luma - self.luma_mean)
        self._t_prev = t
        self._n_frames += 1
        self._tot_new += 1 if is_new else 0
        self._tot_near += 1 if near_dup else 0
        self.is_new = bool(is_new)
        self.luma_raw = float(luma)
        self.luma_p95 = float(luma_p95)
        self.sat_frac = float(sat_frac)

        if hand_moving:
            self._moving_until = t + float(c["moving_hold_sec"])
        moving = t <= self._moving_until
        self._win.append((t, dt, self.is_new, bool(near_dup), moving))
        cutoff = t - self.window_sec
        while self._win and self._win[0][0] <= cutoff:
            self._win.popleft()

        self._recompute_window(t)
        if self.warnings_enabled and bool(c.get("enabled", True)):
            self._advance_states(t)
        else:
            self._status = None

    def session_summary(self) -> Optional[str]:
        """Tóm tắt CẢ PHIÊN (không phải cửa sổ trượt): vòng lặp/s, khung mới thật/s, % khung trùng y hệt, % gần-trùng.
        Dùng để quyết định có cần --dedupe-frames / --time-norm hay không. None nếu chưa đủ dữ liệu."""
        if self._t_start is None or self._t_prev is None or self._n_frames < 2:
            return None
        span = self._t_prev - self._t_start
        if span <= 1e-6:
            return None
        n = self._n_frames
        return (f"{n} khung / {span:.1f}s | vòng lặp {n / span:.1f} khung/s | khung mới thật {self._tot_new / span:.1f}/s | "
                f"trùng y hệt {100.0 * (1.0 - self._tot_new / n):.1f}% | gần-trùng {100.0 * self._tot_near / n:.1f}%")

    # ------------------------------------------------------------------ số liệu cửa sổ
    def _recompute_window(self, t: float) -> None:
        span = min(self.window_sec, t - (self._t_start if self._t_start is not None else t))
        n = len(self._win)
        if span <= 1e-6 or n == 0:
            self.unique_fps = self.loop_fps = 0.0
            self.dup_ratio = self.near_dup_ratio = 0.0
            self.unique_fps_moving = None
            return
        n_new = n_near = 0
        mv_time = 0.0
        mv_new = 0
        for (_, dt, new, near, mv) in self._win:
            n_new += 1 if new else 0
            n_near += 1 if near else 0
            if mv:
                mv_time += dt
                mv_new += 1 if new else 0
        self.unique_fps = n_new / span
        self.loop_fps = n / span
        self.dup_ratio = 1.0 - n_new / n
        self.near_dup_ratio = n_near / n
        need = float(self.cfg["min_moving_frac"]) * self.window_sec
        self.unique_fps_moving = (mv_new / mv_time) if mv_time >= need else None

    # ------------------------------------------------------------------ máy trạng thái cảnh báo
    def _conditions(self, key: str) -> Tuple[bool, bool, bool, bool]:
        """Trả (vào, bình_thường_để_thoát, bad_vào, bad_thoát) theo ngưỡng hiện tại và ngưỡng nới (hysteresis)."""
        c = self.cfg
        h = 1.0 + float(c["hysteresis"])
        if key == "low_light":
            v = self.luma_mean
            lw, lb = float(c["luma_low_warn"]), float(c["luma_low_bad"])
            return v < lw, v > lw * h, v < lb, v > lb * h
        if key == "too_bright":
            v, s = self.luma_mean, self.sat_frac
            hi, sh = float(c["luma_high"]), float(c["sat_frac_high"])
            return (v > hi or s > sh), (v < hi / h and s < sh / h), False, True
        r = self.unique_fps_moving       # slow_camera
        fw, fb = float(c["unique_fps_warn"]), float(c["unique_fps_bad"])
        if r is None:                    # tay không chuyển động đủ lâu: không đánh giá, coi là bình thường
            return False, True, False, True
        return r < fw, r >= fw * h, r < fb, r >= fb * h

    def _advance_states(self, t: float) -> None:
        c = self.cfg
        persist, show_min, clear = float(c["persist_sec"]), float(c["show_min_sec"]), float(c["clear_sec"])
        for key in WARNING_KEYS:
            st = self._st[key]
            enter, ok, bad_in, bad_ok = self._conditions(key)
            if t - (self._t_start if self._t_start is not None else t) < float(c["warmup_sec"]):
                enter = False    # khởi động: chưa cho phép BẬT cảnh báo mới
            if not st["active"]:
                if enter:
                    if st["cond_since"] is None:
                        st["cond_since"] = t
                    if t - st["cond_since"] >= persist:
                        st.update(active=True, active_since=t, ok_since=None, bad=bad_in)
                else:
                    st["cond_since"] = None
            else:
                if bad_in:
                    st["bad"] = True
                elif bad_ok:
                    st["bad"] = False
                if ok:
                    if st["ok_since"] is None:
                        st["ok_since"] = t
                    if t - st["ok_since"] >= clear and t - st["active_since"] >= show_min:
                        st.update(active=False, cond_since=None, ok_since=None, bad=False)
                else:
                    st["ok_since"] = None
        self._status = None
        for key in WARNING_KEYS:
            st = self._st[key]
            if st["active"]:
                self._status = (key, "bad" if st["bad"] else "warn")
                break

    # ------------------------------------------------------------------ đầu ra
    def status(self) -> Optional[Tuple[str, str]]:
        """None hoặc (khóa cảnh báo, mức "warn"|"bad"); khóa: too_bright | low_light | slow_camera."""
        return self._status

    @property
    def warning(self) -> Optional[str]:
        return self._status[0] if self._status else None

    @property
    def warning_level(self) -> str:
        return self._status[1] if self._status else "warn"
