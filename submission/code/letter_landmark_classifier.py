# -*- coding: utf-8 -*-
"""
Bộ phân loại chữ cái từ landmark 3D chuẩn hóa (MLP nhỏ, chạy CPU < 1 ms/khung).

Dùng:
    clf = LetterLandmarkClassifier("letter_landmark_model.pt")
    label, conf, probs = clf.predict(world_landmarks, "Right")      # label=None nếu conf < min_conf
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from hand_pose_features import FEATURE_DIM, landmarks_to_features

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "letter_landmark_model.pt")


class LandmarkMLP(nn.Module):
    def __init__(self, num_classes: int, in_dim: int = FEATURE_DIM, hidden: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LetterLandmarkClassifier:
    def __init__(self, path: str = DEFAULT_MODEL_PATH, min_conf: float = 0.5) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if int(ck["feature_dim"]) != FEATURE_DIM:
            raise ValueError(f"Model {path} dùng {ck['feature_dim']} đặc trưng, code hiện tại là {FEATURE_DIM}: cần train lại.")
        self.classes: List[str] = list(ck["classes"])
        self.mean = np.asarray(ck["mean"], dtype=np.float32)
        self.std = np.asarray(ck["std"], dtype=np.float32)
        self.space: str = str(ck.get("space", "world"))   # "world" hoặc "image": loại landmark model đã học
        self.min_conf = float(min_conf)
        self.model = LandmarkMLP(len(self.classes))
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    def predict_features(self, feats: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(((np.asarray(feats, dtype=np.float32) - self.mean) / self.std).reshape(-1, FEATURE_DIM))
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=1).numpy()

    def predict(self, landmarks_3d: np.ndarray, handedness: Optional[str]) -> Tuple[Optional[str], float, Dict[str, float]]:
        probs = self.predict_features(landmarks_to_features(landmarks_3d, handedness))[0]
        k = int(np.argmax(probs))
        conf = float(probs[k])
        return (self.classes[k] if conf >= self.min_conf else None), conf, {c: float(p) for c, p in zip(self.classes, probs)}
