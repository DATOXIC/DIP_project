# -*- coding: utf-8 -*-
"""
GHÉP CHỮ CÁI ĐÁNH VẦN (chế độ phím `s` của predict_sentence_webcam.py)
=====================================================================

Nhận từng nhãn chữ cái theo thời gian (từ LetterLandmarkClassifier, mỗi khung một nhãn hoặc None) và quyết định
khi nào một chữ được CHỐT: nhãn phải giữ ổn định `hold_sec` giây. Chữ vừa chốt bị khóa; muốn ký lại CÙNG chữ
(ví dụ "NN" trong "ANNA") phải đổi hình tay hoặc hạ tay ít nhất `rearm_sec` giây rồi giữ lại.

Kết quả của một lần đánh vần là token `SPELL:<CHỮ HOA>` mà sentence_composer.py đã biết nhận.

Thuần Python: KHÔNG import OpenCV / MediaPipe / PyTorch / giao diện.
"""
from __future__ import annotations

from typing import Optional


class Fingerspeller:
    def __init__(self, hold_sec: float = 0.7, rearm_sec: float = 0.35, max_len: int = 12) -> None:
        self.hold_sec = float(hold_sec)
        self.rearm_sec = float(rearm_sec)
        self.max_len = int(max_len)
        self.letters: list = []
        self._cand: Optional[str] = None        # nhãn đang được giữ
        self._cand_since: float = 0.0
        self._blocked: Optional[str] = None     # chữ vừa chốt, chưa được chốt lại
        self._gap_since: Optional[float] = None  # từ khi nhãn khác chữ đang khóa

    # ------------------------------------------------------------------ trạng thái
    @property
    def word(self) -> str:
        return "".join(self.letters)

    @property
    def token(self) -> str:
        """Token cho bộ ghép câu: 'SPELL:ANNA' (chuỗi rỗng nếu chưa có chữ nào)."""
        return f"SPELL:{self.word}" if self.letters else ""

    @property
    def current(self) -> Optional[str]:
        """Chữ đang được giữ (chưa chốt), chữ HOA; None nếu chưa có."""
        return self._cand.upper() if self._cand else None

    def progress(self, t: float) -> float:
        """Tiến độ giữ chữ hiện tại, 0..1 (đủ 1 thì chữ được chốt)."""
        if not self._cand or self._cand == self._blocked:
            return 0.0
        return max(0.0, min(1.0, (float(t) - self._cand_since) / self.hold_sec))

    # ------------------------------------------------------------------ cập nhật
    def update(self, t: float, label: Optional[str]) -> Optional[str]:
        """Nạp nhãn của khung tại thời điểm t (giây). Trả chữ HOA vừa được chốt, hoặc None."""
        t = float(t)
        label = label.lower() if isinstance(label, str) and label.isalpha() else None

        # mở khóa chữ vừa chốt khi nhãn đã khác nó đủ lâu
        if self._blocked is not None:
            if label != self._blocked:
                if self._gap_since is None:
                    self._gap_since = t
                if t - self._gap_since >= self.rearm_sec:
                    self._blocked = None
                    self._gap_since = None
            else:
                self._gap_since = None

        if label != self._cand:
            self._cand = label
            self._cand_since = t
            return None
        if label is None or label == self._blocked:
            return None
        if t - self._cand_since >= self.hold_sec and len(self.letters) < self.max_len:
            self.letters.append(label.upper())
            self._blocked = label
            self._gap_since = None
            return label.upper()
        return None

    # ------------------------------------------------------------------ chỉnh sửa
    def backspace(self) -> Optional[str]:
        """Xóa chữ cuối. Trả chữ vừa xóa (HOA) hoặc None nếu chưa có chữ nào."""
        if not self.letters:
            return None
        return self.letters.pop()

    def reset(self) -> None:
        self.letters.clear()
        self._cand = None
        self._cand_since = 0.0
        self._blocked = None
        self._gap_since = None
