"""
Chế độ HỎI XÁC NHẬN: khi hệ thống không chắc (UNCERTAIN) hoặc từ chối (REJECTED) một cử chỉ, nó hỏi "có phải từ X không?".
Người dùng trả lời bằng phím -> mẫu đó (nhãn do NGƯỜI xác nhận) được thêm vào bank CỦA PHIÊN (trong bộ nhớ, file trên đĩa
CHƯA đổi ngay) rồi bank được dựng lại (~0,1 s). Đây là "enroll có nhãn" mà eval_enroll.py đã đo (1-3 mẫu/từ: 58-65% -> 92-100%
nhận đúng, xem thêm HUNGRY: 0% -> 83-100% chỉ với 1 mẫu).

Phím: [y] = đúng, là ứng viên 1   [2] = đúng, là ứng viên 2   [n] = bỏ qua   [z] = hoàn tác mẫu vừa thêm.
Mỗi từ tối đa `cap_per_word` mẫu (mặc định 3) — TÍCH LŨY qua nhiều lần dùng khi ghi bằng merge_into_file()
(khác save(): save() ghi một file mới mỗi lần, không cộng dồn; merge_into_file() cộng dồn vào đúng 1 hồ sơ).
Cuối phiên, predict_sentence_webcam.py tự gọi merge_into_file(profiles/<tên>.json) NẾU đang chạy với hồ sơ cá
nhân (--templates profiles/<tên>.json, tức `py dip.py nhan-dien --ho-so <tên>`) — KHÔNG BAO GIỜ tự ghi vào bank
sản xuất chính (webcam_templates.json).
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np


class SessionEnroller:
    def __init__(self, engine: Any, cap_per_word: int = 3, ttl_sec: float = 8.0):
        self.engine = engine
        self.cap = int(cap_per_word)
        self.ttl = float(ttl_sec)
        self.pending: Optional[Dict[str, Any]] = None
        self.added: List[Dict[str, Any]] = []    # {"word", "id"} theo thứ tự thêm
        self._next_id = 1

    # ------------------------------------------------------------------ chờ xác nhận
    def offer(self, traj: Any, shapes: List[str], handedness: Optional[str], body_ok: bool,
              duration: float, candidates: List[str], now: float) -> bool:
        """Ghi nhớ cử chỉ vừa rồi để hỏi. Trả True nếu có gì để hỏi."""
        cands = [c for c in candidates if c][:2]
        if traj is None or not cands:
            self.pending = None
            return False
        self.pending = {"traj": np.asarray(traj, dtype=np.float32).tolist(), "shapes": list(shapes),
                        "handedness": handedness, "body_ok": bool(body_ok), "duration": float(duration),
                        "cands": cands, "expires": now + self.ttl}
        return True

    def clear(self) -> None:
        self.pending = None

    def pending_valid(self, now: float) -> bool:
        return self.pending is not None and now <= self.pending["expires"]

    # ------------------------------------------------------------------ xác nhận / hoàn tác
    def confirm(self, index: int, now: float) -> Optional[str]:
        """index 0 = ứng viên 1, 1 = ứng viên 2. Trả từ đã thêm, hoặc None (hết hạn / chưa có / đủ mẫu)."""
        if not self.pending_valid(now):
            self.pending = None
            return None
        cands = self.pending["cands"]
        if index >= len(cands):
            return None
        word = cands[index]
        if sum(1 for a in self.added if a["word"] == word) >= self.cap:
            return None
        vocab = self.engine.vocabulary
        if word not in vocab:
            return None
        p = self.pending
        sid = self._next_id
        sample = {"traj_features": p["traj"], "hand_shapes": p["shapes"], "handedness": p["handedness"],
                  "chirality_canonical": False, "body_anchor_valid": p["body_ok"], "rep_count": None,
                  "duration_sec": p["duration"], "is_original": True, "session_added": sid,
                  "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        vocab[word].setdefault("samples", []).append(sample)
        self._rebuild()
        self._next_id += 1
        self.added.append({"word": word, "id": sid})
        self.pending = None
        return word

    def undo(self) -> Optional[str]:
        if not self.added:
            return None
        last = self.added.pop()
        for d in self.engine.vocabulary.values():
            d["samples"] = [s for s in d.get("samples", []) if s.get("session_added") != last["id"]]
        self._rebuild()
        return last["word"]

    def _rebuild(self) -> None:
        """Dựng lại chuẩn hóa chirality, độ rộng nội lớp và khối DTW nhanh sau khi bank đổi."""
        e = self.engine
        e.set_vocabulary(e.vocabulary, {"feature_dim": e.feature_dim})

    # ------------------------------------------------------------------ tổng kết / lưu
    def count(self) -> int:
        return len(self.added)

    def summary(self) -> str:
        per: Dict[str, int] = {}
        for a in self.added:
            per[a["word"]] = per.get(a["word"], 0) + 1
        return ", ".join(f"{w} x{n}" for w, n in sorted(per.items()))

    def save(self, path: str) -> int:
        """Ghi các mẫu đã thêm ra file (định dạng bank, dùng lại được với --templates)."""
        voc: Dict[str, Any] = {}
        for w, d in self.engine.vocabulary.items():
            ss = [s for s in d.get("samples", []) if s.get("session_added")]
            if ss:
                voc[w] = {k: v for k, v in d.items() if k != "samples"}
                voc[w]["samples"] = ss
                voc[w]["num_samples"] = len(ss)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"metadata": {"feature_dim": self.engine.feature_dim, "source": "session_enroll"},
                       "vocabulary": voc}, f, ensure_ascii=False)
        return sum(len(d["samples"]) for d in voc.values())

    def merge_into_file(self, path: str) -> int:
        """
        Cộng dồn các mẫu đã xác nhận TRONG PHIÊN NÀY vào file hồ sơ <path> (tạo mới nếu
        chưa có), để lợi ích tích lũy qua nhiều lần dùng thay vì mất khi tắt chương trình
        (khác save(): save() ghi ra một file RIÊNG mỗi lần, không cộng dồn).

        An toàn: gọi hàm này CHỈ khi <path> là hồ sơ cá nhân (profiles/<tên>.json), KHÔNG
        BAO GIỜ gọi với bank sản xuất chính (webcam_templates.json) — việc kiểm tra đó do
        nơi gọi (predict_sentence_webcam.py) đảm nhiệm.

        `self.cap` được diễn giải lại thành cap TÍCH LŨY (đếm cả mẫu đã có sẵn trong file
        từ các phiên trước), không phải cap riêng phiên này — để hồ sơ không phình to vô
        hạn qua nhiều lần dùng.
        """
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("metadata", {})
            voc = data.setdefault("vocabulary", {})
        else:
            data = {"metadata": {"feature_dim": self.engine.feature_dim, "source": "session_enroll_merged"},
                    "vocabulary": {}}
            voc = data["vocabulary"]

        added_ids = {a["id"] for a in self.added}
        by_id: Dict[int, Dict[str, Any]] = {}
        for d in self.engine.vocabulary.values():
            for s in d.get("samples", []):
                sid = s.get("session_added")
                if sid in added_ids:
                    by_id[sid] = s

        n_merged = 0
        for a in self.added:
            s = by_id.get(a["id"])
            if s is None:
                continue
            w = a["word"]
            entry = voc.setdefault(w, {"samples": []})
            existing = entry.setdefault("samples", [])
            if len(existing) >= self.cap:
                continue  # da du cap TICH LUY cho tu nay (tinh ca cac phien truoc), bo qua
            existing.append({k: v for k, v in s.items() if k != "session_added"})
            entry["num_samples"] = len(existing)
            n_merged += 1

        data["metadata"]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return n_merged
