# -*- coding: utf-8 -*-
"""
MẪU ÂM TRỰC TUYẾN: khi đang nhận diện, người dùng bấm [n] ("vừa rồi là cử chỉ vu vơ, không phải ký") thì vài giây
landmark gần nhất được ghi thành một phiên mini cùng định dạng với phiên `*_neg` của collect_continuous.py
(frames.npz + events.json có một đoạn WANDER). `py dip.py huan-luyen-spotter` tự gộp các phiên này khi huấn luyện lại.

Không học ngay trong lúc chạy (chỉ ghi dữ liệu): spotter là mạng nhỏ, cập nhật từ vài mẫu của một người dễ lệch, và sẽ
không còn phiên độc lập để đo. Huấn luyện lại ngoại tuyến, có phiên kiểm tra riêng, rồi mới thay model.

Quy ước lưu:
  * Thư mục:  <root>/<YYYYMMDD_HHMMSS>_onlineneg   (KHÔNG kết thúc bằng `_neg` để `cham-mau-am` mặc định không chọn nhầm,
    vì phiên mini không có session.mp4 để phát lại).
  * Landmark ở đúng chiều gương mà spotter đã học (xem `mirror` của push()); nhãn tay là nhãn MediaPipe thô của chiều đó.
  * Mỗi lần ghi kèm một dòng trong <root>/log.jsonl để truy vết (thời điểm, lý do, từ bị xóa nếu có).

Thuần Python + numpy: không import OpenCV / MediaPipe / PyTorch.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from collect_continuous import (
    DEFAULT_MARGIN_SEC,
    P_WANDER,
    FrameRecorder,
    SCHEMA_VERSION,
    save_session,
)

DEFAULT_ROOT = os.path.join("continuous_data", "_online")
_SWAP = {"left": "Right", "right": "Left"}


class OnlineNegativeRecorder:
    def __init__(self, root: str = DEFAULT_ROOT, buffer_sec: float = 4.0, window_sec: float = 3.0,
                 min_frames: int = 20, signer: str = "online") -> None:
        self.root = root
        self.buffer_sec = float(buffer_sec)
        self.window_sec = float(window_sec)
        self.min_frames = int(min_frames)
        self.signer = signer
        self._buf: Deque[Tuple[float, str, float, Optional[np.ndarray]]] = deque()
        self.saved = 0

    # ------------------------------------------------------------------ nạp khung
    def push(self, t: float, label: Optional[str], score: float, pts: Any, mirror: bool = False) -> None:
        """
        Ghi một khung. pts: (21,3) toạ độ chuẩn hóa [0,1] hoặc None (mất tay).
        mirror=True: khung hiện tại KHÔNG ở chiều gương như dữ liệu huấn luyện -> x -> 1-x và đảo nhãn tay Left<->Right.
        """
        arr: Optional[np.ndarray] = None
        lab = ""
        if pts is not None:
            arr = np.asarray(pts, dtype=np.float32).reshape(21, 3).copy()
            lab = str(label or "Right")
            if mirror:
                arr[:, 0] = 1.0 - arr[:, 0]
                lab = _SWAP.get(lab.lower(), lab)
        t = float(t)
        self._buf.append((t, lab, float(score), arr))
        while self._buf and t - self._buf[0][0] > self.buffer_sec:
            self._buf.popleft()

    def __len__(self) -> int:
        return len(self._buf)

    # ------------------------------------------------------------------ ghi mẫu âm
    def save_negative(self, t_now: float, reason: str = "user_n", extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Ghi `window_sec` giây gần nhất thành phiên mini. Trả đường dẫn thư mục, hoặc None nếu quá ít khung."""
        t_now = float(t_now)
        frames = [f for f in self._buf if t_now - f[0] <= self.window_sec]
        if len(frames) < self.min_frames:
            return None
        rec = FrameRecorder()
        for t, lab, sc, arr in frames:
            rec.add(t, [(lab, sc, arr)] if arr is not None else [])
        t0, t1 = frames[0][0], frames[-1][0]
        # nhãn WANDER co lại DEFAULT_MARGIN_SEC ở hai đầu: kéo giãn một chút để không mất khung biên
        pad = float(DEFAULT_MARGIN_SEC)
        events: List[Dict[str, Any]] = [
            {"t": t0 - pad - 0.01, "type": "session_start", "prompt_index": -1, "aid": None},
            {"t": t0 - pad, "type": "prompt_begin", "prompt_index": 0, "aid": 1, "kind": P_WANDER,
             "words": [], "word": None, "duration_sec": round(t1 - t0 + 2 * pad, 3)},
            {"t": t1 + pad, "type": "prompt_end", "prompt_index": 0, "aid": 1},
        ]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.root, f"{stamp}_onlineneg")
        k = 1
        while os.path.exists(session_dir):          # hai lần bấm trong cùng một giây
            k += 1
            session_dir = os.path.join(self.root, f"{stamp}_{k}_onlineneg")
        meta = {"signer": self.signer, "session_id": os.path.basename(session_dir), "negative_only": True,
                "source": "online_negative", "reason": reason,
                "handedness_note": "nhãn MediaPipe thô của chiều gương mà spotter đã học (khe 0='Left', khe 1='Right')",
                "schema_version": SCHEMA_VERSION}
        save_session(session_dir, rec, events, meta, [P_WANDER])
        self.saved += 1
        self._log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "negative", "dir": session_dir,
                   "frames": len(frames), "reason": reason, **(extra or {})})
        return session_dir

    def log_wrong_word(self, predicted: str, info: Optional[Dict[str, Any]] = None) -> None:
        """Ghi một dòng 'sai từ' (từ nhận ra là X nhưng người dùng nói sai) để dùng làm bằng chứng / mẫu khó sau này."""
        self._log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "wrong_word", "predicted": predicted,
                   **(info or {})})

    def _log(self, row: Dict[str, Any]) -> None:
        try:
            os.makedirs(self.root, exist_ok=True)
            with open(os.path.join(self.root, "log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass
