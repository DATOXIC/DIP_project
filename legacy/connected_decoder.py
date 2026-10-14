# -*- coding: utf-8 -*-
"""
BỘ GIẢI MÃ NỐI TỪ (CONNECTED SIGN DECODER) - SPEC 09 Nhiệm vụ D
================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

Người ký các từ LIỀN NHAU (không dừng giữa từ). Bộ giải mã nhận từng khung, tự tìm điểm cắt ứng viên trên đường vận tốc,
chấm điểm từng đoạn giữa hai điểm cắt bằng ĐÚNG bộ DTW hiện có (GoldenTriadDTWMatcher.match_many), rồi dùng quy hoạch động
(weighted interval scheduling) chọn tập đoạn-từ không chồng lấn có tổng phần thưởng lớn nhất. Đoạn không đủ tốt là "chuyển tiếp".

Thuần Python/numpy: KHÔNG phụ thuộc OpenCV/MediaPipe => chạy được trực tiếp lẫn phát lại ngoại tuyến (eval_connected.py).

Hợp đồng đầu vào push(rec): rec là dict
    t (giây, bắt buộc), hand_present (bool, mặc định True)
    và khi hand_present: x, y (cổ tay đã làm mượt), palm_len, hand_shape, angles (5,), normal (3,), landmarks (21,3) hoặc None,
    face_box hoặc None, handedness hoặc None.
NGƯỜI GỌI phải bỏ các khung TRÙNG (frame_is_new == False) trước khi đẩy vào; dấu thời gian là của khung mới.

Swipe-Left: bộ giải mã KHÔNG tự phát hiện (bộ dò của KineticPauseSegmenter tính theo số khung, sai ở tần số thấp). Nếu người gọi
có bộ dò riêng thì gọi hard_boundary() để chốt/flush và xóa bộ đệm.

DEFAULT_CONNECTED_CONFIG nằm ở đây; Nhiệm vụ F sẽ đưa vào CONFIG["connected"] của predict_sentence_webcam.py.
Mọi ngưỡng thời gian tính bằng GIÂY; số khung suy ra từ dt trung vị của bộ đệm.
"""
from __future__ import annotations

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_CONNECTED_CONFIG: Dict[str, Any] = {
    "enabled": False,            # mặc định TẮT (chế độ thử nghiệm có cổng kiểm tra, SPEC 09 G)
    "buffer_sec": 4.0,           # bộ đệm cuộn tối đa
    "valley_win_sec": 0.13,      # cửa sổ +-: khung là cực tiểu cục bộ của v trong cửa sổ này
    "v_cut": 0.06,               # trần v (đơn vị palm/khung@30fps) để một trũng thành ứng viên
    "valley_prom": 0.04,         # độ nổi tối thiểu ở ít nhất một phía
    "still_v": 0.04,             # v <= still_v coi là đứng yên (= v_stop_ratio)
    "still_sec": 0.30,           # đứng yên >= ngần này giây: sinh 2 ứng viên (đầu, cuối), vùng yên bị bỏ qua
    "still_skip": True,          # False = tắt "vùng yên là bỏ qua" (dùng cho ablation)
    "confirm_sec": 0.10,         # ứng viên chỉ được xác nhận sau ngần này giây
    "max_candidates": 10,
    "seg_min_sec": 0.20,       # DRINK/ME/HUNGRY chuyển động thật chỉ ~0,23-0,30 s (đo trên raw_strokes)
    "seg_max_sec": 2.5,
    "seg_gap_sec": 0.50,         # khoảng cách dấu thời gian lớn nhất giữa hai khung liền nhau TRONG một đoạn (mất tay ngắn)
    "min_seg_frames": 4,         # số khung THÔ tối thiểu; resample cần >= 8 khung nên phần thiếu do kéo đuôi yên bù (tail_pad_mode != off)
    "max_starts_per_end": 6,     # cắt tỉa: mỗi điểm kết thúc chỉ thử tối đa ngần này điểm bắt đầu gần nhất
    "margin_threshold": 0.12,
    "margin_gate": True,         # False = tắt cổng margin (ablation)
    "r_accept": 1.6,
    "word_penalty": 0.30,
    "lookahead_sec": 0.25,
    "stable_runs": 2,            # đoạn phải nằm trên đường đi tốt nhất ở >= ngần này lần chạy DP liên tiếp (1 = tắt kiểm tra)
    "stable_tol_sec": 0.10,
    "repeat_guard_sec": 0.5,
    "poll_sec": 0.20,
    "lost_frames": 4,            # mất tay >= ngần này khung liên tiếp = ranh giới cứng
    "hard_gap_sec": 1.0,         # khoảng trống dấu thời gian >= ngần này giây cũng là ranh giới cứng
    "v_stop": 0.040,             # cho extend_still_tail
    "tail_pad_sec": 0.45,        # độ dài đuôi yên kéo tới (dùng khi tail_pad_mode != "off")
    "tail_pad_mode": "off",      # "off" | "all" | "still" (chỉ đoạn kết thúc ở đầu vùng yên). Mặc định chọn theo số đo, xem báo cáo
    "background": False,         # True = chấm điểm ở luồng nền (D6)
}


@dataclass
class DecodedWord:
    word_key: str
    t_start: float
    t_end: float
    r_ratio: float
    match_quality: float
    margin_ratio: float
    t_commit: float


@dataclass
class Candidate:
    idx: int          # chỉ số khung trong mảng đầu vào
    kind: str         # "valley" | "still_start" | "still_end"
    prom: float


@dataclass
class SegScore:
    word: Optional[str]
    r_ratio: float
    margin: float
    status: str
    quality: float
    distance: float

    def is_word(self, cfg: Dict[str, Any]) -> bool:
        if self.status != "ACCEPTED" or not self.word:
            return False
        if cfg.get("margin_gate", True) and self.margin < float(cfg["margin_threshold"]):
            return False
        return (float(cfg["r_accept"]) - self.r_ratio - float(cfg["word_penalty"])) > 0.0

    def reward(self, cfg: Dict[str, Any]) -> float:
        return float(cfg["r_accept"]) - self.r_ratio - float(cfg["word_penalty"])


# =============================================================================
# D2. ỨNG VIÊN ĐIỂM CẮT
# =============================================================================
def _median3(v: np.ndarray) -> np.ndarray:
    if len(v) < 3:
        return v.copy()
    out = v.copy()
    out[1:-1] = np.median(np.stack([v[:-2], v[1:-1], v[2:]]), axis=0)
    return out


def still_runs(t: np.ndarray, v: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Các đoạn liên tục v <= still_v kéo dài >= still_sec giây: [(chỉ số đầu, chỉ số cuối)]."""
    n = len(v)
    out: List[Tuple[int, int]] = []
    i = 0
    still = v <= float(cfg["still_v"])
    while i < n:
        if still[i]:
            j = i
            while j + 1 < n and still[j + 1]:
                j += 1
            if t[j] - t[i] >= float(cfg["still_sec"]):
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def find_boundary_candidates(t: np.ndarray, v: np.ndarray, cfg: Dict[str, Any], final: bool = False
                             ) -> Tuple[List[Candidate], List[Tuple[int, int]]]:
    """
    Ứng viên điểm cắt trên đường vận tốc v (đã lọc trung vị 3 khung; đơn vị palm/khung@30fps), dấu thời gian t (giây).
    - Trũng: cực tiểu cục bộ trong cửa sổ +-valley_win_sec, v <= v_cut, độ nổi >= valley_prom ở ít nhất một phía, không nằm trong vùng yên.
    - Vùng yên >= still_sec: ứng viên "still_start" (khung đầu) và "still_end" (khung cuối, chỉ khi vùng đã kết thúc: thấy v tăng lại).
    - Xác nhận: trũng cần cửa sổ phải đã quan sát đủ và t[-1]-t[k] >= confirm_sec; still_end cần t[-1]-t[end] >= confirm_sec.
      final=True (flush/ranh giới cứng): bỏ mọi điều kiện chờ, cửa sổ bị cắt ở mép.
    Giới hạn max_candidates: giữ các ứng viên có độ nổi cao nhất (ứng viên vùng yên coi như độ nổi vô hạn), trả về theo thứ tự thời gian.
    Trả (danh sách ứng viên, danh sách vùng yên).
    """
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n == 0:
        return [], []
    dts = np.diff(t)
    dts = dts[dts > 1e-6]
    dt = float(np.clip(np.median(dts), 1.0 / 60.0, 0.25)) if dts.size else 1.0 / 30.0
    w = max(1, int(round(float(cfg["valley_win_sec"]) / dt)))
    runs = still_runs(t, v, cfg)
    in_still = np.zeros(n, dtype=bool)
    for a, b in runs:
        in_still[a:b + 1] = True
    cands: List[Candidate] = []
    t_last = float(t[-1])
    confirm = float(cfg["confirm_sec"])
    if bool(cfg.get("still_skip", True)):
        for a, b in runs:
            cands.append(Candidate(a, "still_start", 1e9))
            if b < n - 1 and (final or t_last - t[b] >= confirm):
                cands.append(Candidate(b, "still_end", 1e9))
    v_cut, v_prom = float(cfg["v_cut"]), float(cfg["valley_prom"])
    for k in range(n):
        if v[k] > v_cut or (in_still[k] and bool(cfg.get("still_skip", True))):
            continue
        hi = min(n - 1, k + w)
        if not final and (k + w > n - 1 or t_last - t[k] < confirm):
            continue
        lo = max(0, k - w)
        win = v[lo:hi + 1]
        if lo + int(np.argmin(win)) != k:
            continue
        prom = max(float(v[lo:k + 1].max() - v[k]), float(v[k:hi + 1].max() - v[k]))
        if prom >= v_prom:
            cands.append(Candidate(k, "valley", prom))
    mx = int(cfg["max_candidates"])
    if len(cands) > mx:
        cands = sorted(sorted(cands, key=lambda c: -c.prom)[:mx], key=lambda c: c.idx)
    else:
        cands.sort(key=lambda c: (c.idx, c.kind))
    return cands, runs


# =============================================================================
# D4. QUY HOẠCH ĐỘNG (weighted interval scheduling)
# =============================================================================
def best_path(items: Sequence[Tuple[float, float, float]]) -> Tuple[float, List[int]]:
    """
    items: (start, end, reward) (start <= end). Chọn tập không chồng lấn (chạm đầu mút được: start_j >= end_i) có tổng reward lớn nhất
    (chỉ dùng reward > 0). Trả (tổng, chỉ số các phần tử đã chọn theo thời gian). DP O(M^2) trên các phần tử sắp theo (end, start).
    """
    order = sorted((i for i, it in enumerate(items) if it[2] > 0.0), key=lambda i: (items[i][1], items[i][0], i))
    m = len(order)
    best = [0.0] * (m + 1)
    take = [False] * (m + 1)
    prev = [0] * (m + 1)
    for a in range(1, m + 1):
        i = order[a - 1]
        p = 0
        for b in range(a - 1, 0, -1):
            if items[order[b - 1]][1] <= items[i][0]:
                p = b
                break
        prev[a] = p
        with_i = items[i][2] + best[p]
        if with_i > best[a - 1] + 1e-12:
            best[a], take[a] = with_i, True
        else:
            best[a] = best[a - 1]
    chosen: List[int] = []
    a = m
    while a > 0:
        if take[a]:
            chosen.append(order[a - 1])
            a = prev[a]
        else:
            a -= 1
    chosen.reverse()
    return best[m], chosen


# =============================================================================
# D1, D3, D5. BỘ GIẢI MÃ
# =============================================================================
class ConnectedSignDecoder:
    """push(rec) -> List[DecodedWord]; flush() -> List[DecodedWord]. Xem docstring module."""

    def __init__(self, matcher: Any, cfg: Optional[Dict[str, Any]] = None):
        self.matcher = matcher
        self.cfg: Dict[str, Any] = dict(DEFAULT_CONNECTED_CONFIG)
        if cfg:
            self.cfg.update(cfg)
        self.stats = {"n_scored": 0, "n_cache_hit": 0, "n_runs": 0, "n_lost_flush": 0}
        self._lock = threading.Lock()
        self.reset()

    # ---------------------------------------------------------------- trạng thái
    def reset(self) -> None:
        self.base = 0                    # chỉ số tuyệt đối của khung đầu bộ đệm
        self.recs: List[Dict[str, Any]] = []
        self.v: List[float] = []
        self.anchor = 0                  # chỉ số tuyệt đối: đoạn phải bắt đầu >= anchor
        self.missing = 0
        self.cache: Dict[Tuple[int, int], SegScore] = {}
        self.seen: set = set()
        self.last_poll_t = -1e9
        self.prev_path: List[Tuple[str, float, float, int]] = []   # (từ, t0, t1, số lần liên tiếp)
        self.cur_path: List[Tuple[str, float, float]] = []
        self.last_commit: Optional[DecodedWord] = None
        self.log_uncertain: List[Dict[str, Any]] = []
        self.last_alignment: Dict[str, Any] = {}

    def hard_boundary(self, t: Optional[float] = None) -> List[DecodedWord]:
        """Ranh giới cứng do người gọi báo (vd Swipe-Left): chốt mọi thứ trước đó rồi xóa bộ đệm."""
        return self.flush(t)

    # ---------------------------------------------------------------- D1: push
    def push(self, rec: Dict[str, Any]) -> List[DecodedWord]:
        cfg = self.cfg
        t = float(rec["t"])
        if not rec.get("hand_present", True):
            self.missing += 1
            if self.missing >= int(cfg["lost_frames"]) and self.recs:
                self.stats["n_lost_flush"] += 1
                return self.flush(t)
            return []
        self.missing = 0
        if self.recs and t - self.recs[-1]["t"] >= float(cfg["hard_gap_sec"]):
            out = self.flush(self.recs[-1]["t"])
        else:
            out = []
        r = dict(rec)
        r["palm_len"] = max(float(r["palm_len"]), 15.0)
        if self.recs:
            p = self.recs[-1]
            dt = float(np.clip(t - p["t"], 1.0 / 60.0, 0.25))
            vk = math.hypot(r["x"] - p["x"], r["y"] - p["y"]) / r["palm_len"] / dt / 30.0
        else:
            vk = 1.0                      # khung đầu: coi như đang chuyển động (không phải trũng/yên giả)
        self.recs.append(r)
        self.v.append(vk)
        self._trim(t)
        out += self._maybe_run(t)
        return out

    def _trim(self, t: float) -> None:
        cut = 0
        while cut < len(self.recs) - 1 and t - self.recs[cut]["t"] > float(self.cfg["buffer_sec"]):
            cut += 1
        if cut:
            del self.recs[:cut]
            del self.v[:cut]
            self.base += cut
            self.anchor = max(self.anchor, self.base)
            self.cache = {k: s for k, s in self.cache.items() if k[0] >= self.base}

    def _arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        i0 = self.anchor - self.base
        t = np.array([r["t"] for r in self.recs[i0:]], dtype=np.float64)
        v = _median3(np.array(self.v[i0:], dtype=np.float64))
        return t, v

    def _maybe_run(self, t: float) -> List[DecodedWord]:
        i0 = self.anchor - self.base
        if len(self.recs) - i0 < 3:
            return []
        ts, vf = self._arrays()
        cands, runs = find_boundary_candidates(ts, vf, self.cfg)
        keys = {(self.anchor + c.idx, c.kind) for c in cands}
        new = keys - self.seen
        self.seen |= keys
        if not new and t - self.last_poll_t < float(self.cfg["poll_sec"]):
            return []
        self.last_poll_t = t
        return self._run(t, ts, vf, cands, runs, final=False)

    # ---------------------------------------------------------------- D3: chấm điểm đoạn
    def _segments(self, ts: np.ndarray, cands: List[Candidate], final: bool, runs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Danh sách đoạn (a, b) chỉ số TUYỆT ĐỐI, hợp lệ theo thời lượng/khoảng trống."""
        cfg = self.cfg
        A = self.anchor
        n = len(ts)
        starts = [0] if n else []
        for c in cands:
            if c.kind in ("valley", "still_end"):
                starts.append(c.idx)
        ends = [c.idx for c in cands if c.kind in ("valley", "still_start")]
        if final and n:
            ends.append(n - 1)
        # bỏ điểm bắt đầu nằm trong vùng yên (khung anchor nằm giữa vùng yên)
        if bool(cfg.get("still_skip", True)):
            starts = [s for s in starts if not any(ra <= s <= rb for ra, rb in runs)]
            if 0 in starts and any(ra <= 1 for ra, _rb in runs[:1]):
                starts.remove(0)
        starts = sorted(set(starts))
        ends = sorted(set(ends))
        gaps = np.diff(ts) if n > 1 else np.zeros(0)
        out: List[Tuple[int, int]] = []
        for e in ends:
            cnt = 0
            for s in reversed([s for s in starts if s < e]):
                dur = ts[e] - ts[s]
                if dur > float(cfg["seg_max_sec"]):
                    break
                if dur < float(cfg["seg_min_sec"]) or (e - s + 1) < int(cfg["min_seg_frames"]):
                    continue
                if gaps[s:e].size and float(gaps[s:e].max()) > float(cfg["seg_gap_sec"]):
                    continue
                out.append((A + s, A + e))
                cnt += 1
                if cnt >= int(cfg["max_starts_per_end"]):
                    break
        return out

    def _seg_records(self, a: int, b: int, still_end: bool) -> List[Dict[str, Any]]:
        recs = self.recs[a - self.base:b - self.base + 1]
        cfg = self.cfg
        mode = cfg.get("tail_pad_mode", "off")
        if mode == "off" or not cfg.get("tail_pad_sec"):
            return recs
        if mode == "still" and not still_end:
            return recs
        from dynamic_features import extend_still_tail
        out, _ = extend_still_tail(recs, float(cfg["tail_pad_sec"]), float(cfg["v_stop"]), replay_jitter=True)
        return out

    def _score(self, segs: List[Tuple[int, int]], still_ends: set) -> None:
        from dynamic_features import build_segment_features
        todo, queries, keys = [], [], []
        for (a, b) in segs:
            if (a, b) in self.cache:
                self.stats["n_cache_hit"] += 1
                continue
            recs = self._seg_records(a, b, b in still_ends)
            traj, shapes, body_ok, dom = build_segment_features(recs)
            if traj is None:
                self.cache[(a, b)] = SegScore(None, 99.0, 0.0, "REJECTED", 0.0, 999.0)
                continue
            queries.append((traj, shapes, dom, body_ok))
            keys.append((a, b))
        if not queries:
            return
        res = self.matcher.match_many(queries) if hasattr(self.matcher, "match_many") else [self.matcher.match(*q) for q in queries]
        self.stats["n_scored"] += len(queries)
        for k, m in zip(keys, res):
            if m.candidates and len(m.candidates) >= 2:
                margin = float(self.matcher.compute_margin_ratio(m.candidates[0].distance, m.candidates[1].distance))
            else:
                margin = 9.9
            self.cache[k] = SegScore(m.top_word, float(m.r_ratio), margin, m.status, float(m.match_quality), float(m.best_distance))

    # ---------------------------------------------------------------- D4 + D5: chạy DP và chốt
    def _run(self, t_now: float, ts: np.ndarray, vf: np.ndarray, cands: List[Candidate], runs: List[Tuple[int, int]], final: bool) -> List[DecodedWord]:
        cfg = self.cfg
        self.stats["n_runs"] += 1
        segs = self._segments(ts, cands, final, runs)
        still_ends = {self.anchor + c.idx for c in cands if c.kind == "still_start"}
        self._score(segs, still_ends)
        items, meta = [], []
        for (a, b) in segs:
            sc = self.cache.get((a, b))
            if sc is None:
                continue
            ta = self.recs[a - self.base]["t"]
            tb = self.recs[b - self.base]["t"]
            if sc.is_word(cfg):
                items.append((ta, tb, sc.reward(cfg)))
                meta.append((a, b, sc))
            elif sc.status == "UNCERTAIN" or (sc.status == "ACCEPTED"):
                if len(self.log_uncertain) < 2000:
                    self.log_uncertain.append({"t0": ta, "t1": tb, "word": sc.word, "r": sc.r_ratio, "margin": sc.margin, "status": sc.status})
        total, chosen = best_path(items)
        path = [meta[i] for i in chosen]
        self.cur_path = [(sc.word, self.recs[a - self.base]["t"], self.recs[b - self.base]["t"]) for a, b, sc in path]
        self.last_alignment = {"t_now": t_now, "final": final, "candidates": [(self.anchor + c.idx, c.kind, round(c.prom, 3)) for c in cands],
                               "n_segments": len(segs), "path": [(sc.word, a, b, round(sc.r_ratio, 3), round(sc.margin, 3)) for a, b, sc in path], "total": total}
        tol = float(cfg["stable_tol_sec"])
        new_prev: List[Tuple[str, float, float, int]] = []
        for (a, b, sc) in path:
            ta, tb = self.recs[a - self.base]["t"], self.recs[b - self.base]["t"]
            streak = 1
            for (w0, t0, t1, s0) in self.prev_path:
                if w0 == sc.word and abs(t0 - ta) <= tol and abs(t1 - tb) <= tol:
                    streak = s0 + 1
                    break
            new_prev.append((sc.word, ta, tb, streak))
        self.prev_path = new_prev
        out: List[DecodedWord] = []
        for (a, b, sc), (_w, ta, tb, streak) in zip(path, new_prev):
            if not final:
                if t_now - tb < float(cfg["lookahead_sec"]):
                    break
                if streak < int(cfg["stable_runs"]):
                    break
            dw = DecodedWord(sc.word, ta, tb, sc.r_ratio, sc.quality, sc.margin, t_now)
            if self._is_repeat(dw):
                self._advance(b)
                continue
            out.append(dw)
            self.last_commit = dw
            self._advance(b)
        done = {(w.word_key, w.t_start) for w in out}
        self.cur_path = [p for p in self.cur_path if (p[0], p[1]) not in done]
        return out

    def _is_repeat(self, w: DecodedWord) -> bool:
        lc = self.last_commit
        if lc is None or lc.word_key != w.word_key or w.t_commit - lc.t_commit >= float(self.cfg["repeat_guard_sec"]):
            return False
        inter = min(lc.t_end, w.t_end) - max(lc.t_start, w.t_start)
        return inter > 0.5 * max(1e-6, min(lc.t_end - lc.t_start, w.t_end - w.t_start))

    def _advance(self, end_abs: int) -> None:
        """Sau khi chốt: anchor = cuối đoạn; bỏ cache/ứng viên đã cũ; xóa lịch sử ổn định."""
        self.anchor = max(self.anchor, end_abs)
        self.cache = {k: s for k, s in self.cache.items() if k[0] >= self.anchor}
        self.seen = {k for k in self.seen if k[0] >= self.anchor}
        self.prev_path = []

    # ---------------------------------------------------------------- flush
    def flush(self, t: Optional[float] = None) -> List[DecodedWord]:
        """Chốt mọi đoạn-từ trên đường đi tốt nhất (bỏ điều kiện nhìn trước và ổn định) rồi xóa bộ đệm."""
        out: List[DecodedWord] = []
        if len(self.recs) - (self.anchor - self.base) >= 3:
            ts, vf = self._arrays()
            cands, runs = find_boundary_candidates(ts, vf, self.cfg, final=True)
            t_now = float(t) if t is not None else float(ts[-1])
            out = self._run(t_now, ts, vf, cands, runs, final=True)
        # xóa bộ đệm nhưng giữ last_commit để chống bắn đôi xuyên ranh giới
        keep = self.last_commit
        log = self.log_uncertain
        align = self.last_alignment
        st = self.stats
        n_abs = self.base + len(self.recs)
        self.reset()
        self.base = self.anchor = n_abs
        self.last_commit, self.log_uncertain, self.last_alignment, self.stats = keep, log, align, st
        return out

    def pending_words(self) -> List[Tuple[str, float, float]]:
        """Từ CHƯA chốt trên đường đi tốt nhất lần chạy DP gần nhất (cho HUD: dấu chấm mờ)."""
        return list(self.cur_path)
