"""
DAC TRUNG TUNG KHUNG CHO BINARY SPOTTER (ky / khong ky)
=======================================================
Mot nguon su that duy nhat cho ca HUAN LUYEN (train_spotter.py) lan CHAY THAT (DLSpotter trong spotter_gate.py),
de hai ben khong the lech nhau. Chi dung QUA KHU (nhan qua), nen chay thoi gian thuc duoc.

Moi khung -> vector FEAT_DIM = 66 chieu:
    [0]        co tay (1/0)
    [1:61]     20 khop (bo co tay) toa do TUONG DOI so voi co tay, chia cho palm_len (x, y, z)
    [61:63]    vi tri co tay tren khung hinh (x, y chuan hoa 0..1)   [chieu cao tay co y nghia: tay gio len hay buong]
    [63:66]    van toc co tay / palm_len / giay (vx, vy) va toc do |v|
Mat tay -> toan 0 (co tay = 0).

Cua so: WINDOW khung gan nhat, dem 0 o dau neu chua du.
"""

from collections import deque
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from dynamic_features import compute_palm_len

FEAT_DIM = 66
WINDOW = 16                      # ~0.53 s o 30 fps
_VEL_CLIP = 20.0                 # chan nhieu: van toc chuan hoa toi da (palm_len / giay)


class FrameFeaturizer:
    """Trich dac trung tung khung kieu streaming. Goi push() moi khung theo thu tu thoi gian."""

    def __init__(self, frame_w: float = 640.0, frame_h: float = 480.0):
        self.w = float(frame_w)
        self.h = float(frame_h)
        self.reset()

    def reset(self) -> None:
        self._prev_wrist: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None

    def push(self, t: float, pts: Optional[np.ndarray]) -> np.ndarray:
        """pts: mang (21, 3) toa do chuan hoa cua MOT tay (hoac None / toan NaN neu mat tay)."""
        f = np.zeros(FEAT_DIM, dtype=np.float32)
        if pts is None or np.isnan(pts[0, 0]):
            self.reset()
            return f
        pts = np.asarray(pts, dtype=np.float64).reshape(21, 3)
        p = lambda k: SimpleNamespace(x=float(pts[k, 0]), y=float(pts[k, 1]), z=float(pts[k, 2]))   # noqa: E731
        palm = max(float(compute_palm_len(p(0), p(5), p(9), p(17), self.w, self.h)), 15.0)
        wrist_px = np.array([pts[0, 0] * self.w, pts[0, 1] * self.h], dtype=np.float64)
        rel = np.empty((20, 3), dtype=np.float64)
        rel[:, 0] = (pts[1:, 0] - pts[0, 0]) * self.w / palm
        rel[:, 1] = (pts[1:, 1] - pts[0, 1]) * self.h / palm
        rel[:, 2] = (pts[1:, 2] - pts[0, 2]) * self.w / palm
        f[0] = 1.0
        f[1:61] = rel.reshape(-1).astype(np.float32)
        f[61] = float(pts[0, 0])
        f[62] = float(pts[0, 1])
        if self._prev_wrist is not None and self._prev_t is not None:
            dt = max(float(t) - self._prev_t, 1e-3)
            v = (wrist_px - self._prev_wrist) / palm / dt
            v = np.clip(v, -_VEL_CLIP, _VEL_CLIP)
            f[63], f[64] = float(v[0]), float(v[1])
            f[65] = float(np.hypot(v[0], v[1]))
        self._prev_wrist = wrist_px
        self._prev_t = float(t)
        return f


class WindowBuffer:
    """Giu WINDOW vector dac trung gan nhat; window() tra mang (WINDOW, FEAT_DIM), dem 0 o dau neu thieu."""

    def __init__(self, window: int = WINDOW):
        self.window = int(window)
        self._q: deque = deque(maxlen=self.window)

    def reset(self) -> None:
        self._q.clear()

    def push(self, f: np.ndarray) -> None:
        self._q.append(f)

    def window_array(self) -> np.ndarray:
        out = np.zeros((self.window, FEAT_DIM), dtype=np.float32)
        n = len(self._q)
        if n:
            out[self.window - n:] = np.stack(self._q)
        return out


def pick_hand_slot(landmarks_t: np.ndarray, scores_t: np.ndarray) -> Optional[np.ndarray]:
    """Tu (2, 21, 3) va diem (2,), chon tay co diem cao nhat. None neu khong thay tay."""
    vis = ~np.isnan(landmarks_t[:, 0, 0])
    if not vis.any():
        return None
    sc = np.where(vis, np.nan_to_num(scores_t, nan=0.0) + 1e-6, -1.0)
    return landmarks_t[int(np.argmax(sc))]


def session_features(landmarks: np.ndarray, scores: np.ndarray, t: np.ndarray,
                     frame_w: float = 640.0, frame_h: float = 480.0) -> np.ndarray:
    """Dac trung tung khung cua ca phien -> (T, FEAT_DIM). Dung chung duong code voi chay that."""
    fz = FrameFeaturizer(frame_w, frame_h)
    out = np.zeros((landmarks.shape[0], FEAT_DIM), dtype=np.float32)
    for i in range(landmarks.shape[0]):
        out[i] = fz.push(float(t[i]), pick_hand_slot(landmarks[i], scores[i]))
    return out


def windows_from_features(feats: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """(T, D) -> (T, window, D); cua so ket thuc tai khung i (chi nhin qua khu, dem 0 o dau)."""
    T, D = feats.shape
    pad = np.zeros((window - 1, D), dtype=feats.dtype)
    ext = np.concatenate([pad, feats], axis=0)
    idx = np.arange(T)[:, None] + np.arange(window)[None, :]
    return ext[idx]


def window_stats(win: np.ndarray) -> np.ndarray:
    """(N, window, D) -> (N, K) thong ke cua so cho mo hinh don gian: trung binh, do lech, khung cuoi, + toc do."""
    mean = win.mean(axis=1)
    std = win.std(axis=1)
    last = win[:, -1, :]
    speed = win[:, :, 65]
    extra = np.stack([speed.mean(1), speed.max(1), speed.min(1), win[:, :, 0].mean(1)], axis=1)
    return np.concatenate([mean, std, last, extra], axis=1).astype(np.float32)
