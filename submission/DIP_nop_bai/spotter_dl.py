"""
SPOTTER DL: bo bao hieu "dang ky / khong ky" bang mang 1D-CNN nho (spotter_model.pth).

Chay tung khung, thoi gian thuc, CHI dung qua khu:
    landmark (21,3) -> FrameFeaturizer (spotter_features.py) -> cua so 16 khung -> CNN -> xac suat p
    -> ProbGate (cong nguong kep, spotter_gate.py) -> True/False "dang ky".
Dung chung spotter_features.py voi train_spotter.py nen khong the lech dac trung giua train va chay that.

Giao dien khop spotter_gate.Spotter: update(t, hand_found, v_norm, landmarks=None, frame_wh=None) -> bool.
v_norm khong dung (de giu chung giao dien voi RuleSpotter).
"""

from typing import Any, Optional, Tuple

import numpy as np

from spotter_features import FEAT_DIM, WINDOW, FrameFeaturizer, WindowBuffer
from spotter_gate import ProbGate, Spotter


def build_net():
    """Kien truc CNN cua Spotter. train_spotter.py va DLSpotter cung dung ham nay de state_dict luon khop."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(FEAT_DIM, 32, 3, padding=1)
            self.conv2 = nn.Conv1d(32, 32, 3, padding=1, stride=2)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(64, 1)
            self.act = nn.ReLU()

        def forward(self, x):                      # x: (N, WINDOW, FEAT_DIM)
            h = x.transpose(1, 2)
            h = self.act(self.conv1(h))
            h = self.act(self.conv2(h))
            feat = h.mean(dim=2)
            last = h[:, :, -1]
            return self.fc(self.drop(torch.cat([feat, last], dim=1))).squeeze(1)

    return Net()


def load_model(path: str):
    """Doc spotter_model.pth -> (net da eval, mean, std). Bao loi ro rang neu khac phien ban dac trung."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck.get("window", -1)) != WINDOW or int(ck.get("feat_dim", -1)) != FEAT_DIM:
        raise ValueError(f"Model '{path}' dung cua so/dac trung khac (window={ck.get('window')}, feat_dim={ck.get('feat_dim')}); "
                         f"code hien tai la window={WINDOW}, feat_dim={FEAT_DIM}. Hay huan luyen lai bang train_spotter.py --save.")
    net = build_net()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, np.asarray(ck["mean"], dtype=np.float32), np.asarray(ck["std"], dtype=np.float32)


class DLSpotter(Spotter):
    def __init__(self, model_path: str = "spotter_model.pth", open_thr: float = 0.5, close_thr: float = 0.35,
                 open_hold_sec: float = 0.10, close_hold_sec: float = 0.20):
        self.net, self.mean, self.std = load_model(model_path)
        self.gate = ProbGate(open_thr, close_thr, open_hold_sec, close_hold_sec)
        self.feat = FrameFeaturizer()
        self.buf = WindowBuffer(WINDOW)
        self.last_prob = 0.0

    def reset(self) -> None:
        self.gate.reset()
        self.feat.reset()
        self.buf.reset()
        self.last_prob = 0.0

    @staticmethod
    def landmarks_to_array(landmarks: Any) -> Optional[np.ndarray]:
        """Danh sach 21 landmark MediaPipe (.x .y .z chuan hoa) hoac mang (21,3) -> mang (21,3)."""
        if landmarks is None:
            return None
        if isinstance(landmarks, np.ndarray):
            return landmarks.reshape(21, 3)
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)

    def predict_prob(self) -> float:
        import torch
        w = (self.buf.window_array() - self.mean) / self.std
        with torch.no_grad():
            return float(torch.sigmoid(self.net(torch.from_numpy(w[None].astype(np.float32))))[0])

    def update(self, t: float, hand_found: bool, v_norm: float = 0.0, landmarks: Any = None,
               frame_wh: Optional[Tuple[float, float]] = None) -> bool:
        if frame_wh is not None:
            self.feat.w, self.feat.h = float(frame_wh[0]), float(frame_wh[1])
        pts = self.landmarks_to_array(landmarks) if hand_found else None
        self.buf.push(self.feat.push(float(t), pts))
        self.last_prob = self.predict_prob()
        return self.gate.update(float(t), self.last_prob)
