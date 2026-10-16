"""
SPOTTER GATE: KHOA MOT LAN (one-shot lock) cho moi lan ky
=========================================================
Van de: KineticPauseSegmenter co the cat MOT dong tac thanh nhieu stroke, moi stroke tu commit
=> nguoi dung ky 2 tu nhung may doc 4 lan.

Y tuong: mot bo "Spotter" bao hieu "dang ky / khong ky" theo thoi gian. Moi khoang lien tuc "dang ky"
(span) chi cho phep commit DUNG MOT tu. Spotter DL (binary spotter) CHUA co, nen file nay cung cap:
  - giao dien `Spotter` (sau nay thay bang model ma khong sua luong chinh),
  - `RuleSpotter`: bo spotter dung luat van toc co lich tre (hysteresis),
  - `OneShotLock`: ghi span tu chuoi gate_open va cap phep commit mot lan tren moi span.

File nay KHONG phu thuoc cv2/mediapipe/numpy de import nhe.

GIOI HAN da biet: RuleSpotter chi dua vao van toc nen KHONG phan biet duoc hai tu lien nhau khong co
khoang dung (tay khong cham nguong dong du lau) -> chung nam trong cung mot span va tu thu hai bi nuot.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class Spotter(ABC):
    """Giao dien bo bao hieu "dang ky / khong ky". Cac cai dat (luat, model DL) deu phai theo giao dien nay."""

    @abstractmethod
    def update(self, t: float, hand_found: bool, v_norm: float, landmarks: Any = None,
               frame_wh: Optional[Tuple[float, float]] = None) -> bool:
        """Nhan 1 khung (thoi gian giay, co thay tay khong, van toc chuan hoa theo long ban tay/khung;
        tuy chon: 21 landmark chuan hoa cua tay va kich thuoc khung cho spotter hoc may).
        Tra ve True neu dang ky (gate mo)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Dua ve trang thai ban dau (mac dinh: khong lam gi)."""
        return None


class RuleSpotter(Spotter):
    """
    Spotter dung luat voi lich tre (hysteresis):
      - MO khi hand_found va v_norm >= open_v lien tuc >= open_hold_sec.
      - DONG khi mat tay HOAC v_norm < close_v lien tuc >= close_hold_sec.
    v_norm cung don vi voi KineticPauseSegmenter.update (khoang dich chuyen co tay / palm_len moi khung).
    Mac dinh: open_v = 1.5 * v_stop_ratio (0.035 -> 0.0525), close_v = v_stop_ratio.
    """

    DEFAULT_V_STOP_RATIO = 0.035

    def __init__(
        self,
        open_v: float = DEFAULT_V_STOP_RATIO * 1.5,
        close_v: float = DEFAULT_V_STOP_RATIO,
        open_hold_sec: float = 0.10,
        close_hold_sec: float = 0.25,
    ):
        if close_v > open_v:
            raise ValueError("close_v phai <= open_v de co lich tre (hysteresis).")
        self.open_v = float(open_v)
        self.close_v = float(close_v)
        self.open_hold_sec = float(open_hold_sec)
        self.close_hold_sec = float(close_hold_sec)
        self.is_open = False
        self._open_since: Optional[float] = None    # bat dau chuoi v >= open_v lien tuc
        self._close_since: Optional[float] = None   # bat dau chuoi v < close_v lien tuc (hoac mat tay)

    @classmethod
    def from_segmenter(cls, v_stop_ratio: float, open_hold_sec: float = 0.10, close_hold_sec: float = 0.25) -> "RuleSpotter":
        """Dung nguong theo v_stop_ratio cua segmenter dang chay (cung don vi v_norm)."""
        return cls(open_v=float(v_stop_ratio) * 1.5, close_v=float(v_stop_ratio),
                   open_hold_sec=open_hold_sec, close_hold_sec=close_hold_sec)

    def reset(self) -> None:
        self.is_open = False
        self._open_since = None
        self._close_since = None

    def update(self, t: float, hand_found: bool, v_norm: float, landmarks: Any = None,
               frame_wh: Optional[Tuple[float, float]] = None) -> bool:
        t = float(t)
        if not self.is_open:
            self._close_since = None
            if hand_found and float(v_norm) >= self.open_v:
                if self._open_since is None:
                    self._open_since = t
                if t - self._open_since >= self.open_hold_sec:
                    self.is_open = True
                    self._open_since = None
            else:
                self._open_since = None
            return self.is_open

        # Dang mo: xet dieu kien dong
        self._open_since = None
        if (not hand_found) or float(v_norm) < self.close_v:
            if not hand_found:
                # Mat tay: dong ngay (theo dac ta "mat tay HOAC ...")
                self.is_open = False
                self._close_since = None
                return False
            if self._close_since is None:
                self._close_since = t
            if t - self._close_since >= self.close_hold_sec:
                self.is_open = False
                self._close_since = None
        else:
            self._close_since = None
        return self.is_open


class ProbGate:
    """
    Cong nguong kep (hysteresis) tren XAC SUAT "dang ky" p cua mot mo hinh (Spotter DL):
      - MO khi p >= open_thr lien tuc >= open_hold_sec.
      - DONG khi p < close_thr lien tuc >= close_hold_sec.
    Dung chung cho do offline (train_spotter.py) va chay that, de hai ben cung mot luat.
    """

    def __init__(self, open_thr: float = 0.5, close_thr: float = 0.35,
                 open_hold_sec: float = 0.10, close_hold_sec: float = 0.20):
        if close_thr > open_thr:
            raise ValueError("close_thr phai <= open_thr de co lich tre (hysteresis).")
        self.open_thr = float(open_thr)
        self.close_thr = float(close_thr)
        self.open_hold_sec = float(open_hold_sec)
        self.close_hold_sec = float(close_hold_sec)
        self.is_open = False
        self._open_since: Optional[float] = None
        self._close_since: Optional[float] = None

    def reset(self) -> None:
        self.is_open = False
        self._open_since = None
        self._close_since = None

    def update(self, t: float, p: float) -> bool:
        t = float(t)
        if not self.is_open:
            self._close_since = None
            if p >= self.open_thr:
                if self._open_since is None:
                    self._open_since = t
                if t - self._open_since >= self.open_hold_sec:
                    self.is_open = True
                    self._open_since = None
            else:
                self._open_since = None
            return self.is_open
        self._open_since = None
        if p < self.close_thr:
            if self._close_since is None:
                self._close_since = t
            if t - self._close_since >= self.close_hold_sec:
                self.is_open = False
                self._close_since = None
        else:
            self._close_since = None
        return self.is_open


def make_spotter(kind: str = "rule", v_stop_ratio: float = RuleSpotter.DEFAULT_V_STOP_RATIO,
                 model_path: str = "spotter_model.pth") -> Spotter:
    """Nha may tao spotter theo ten: 'rule' (luat van toc) hoac 'dl' (CNN nho, doc model_path)."""
    if kind == "rule":
        return RuleSpotter.from_segmenter(v_stop_ratio)
    if kind == "dl":
        from spotter_dl import DLSpotter   # import tre: torch chi nap khi that su dung
        return DLSpotter(model_path)
    raise ValueError(f"Spotter khong ho tro: {kind!r} (co 'rule', 'dl').")


class OneShotLock:
    """
    Ghi nhan cac span (span_id, t_open, t_close|None) tu chuoi gate_open theo thoi gian, va cho phep
    toi da MOT commit tren moi span.

    allow_commit(t_stroke_start): tim span chua t_stroke_start, hoac span gan nhat truoc do (co cho phep
    lech `tolerance_sec` vi spotter mo tre hon luc stroke bat dau mot chut):
      - span chua dung  -> danh dau da dung, (True, "first")
      - span da dung    -> (False, "dup_in_span")
      - khong co span   -> (True, "no_span") [fail-open, hanh vi cu]; strict=True -> (False, "no_span") [chan tu vu vo]
    """

    def __init__(self, max_spans: int = 50, tolerance_sec: float = 0.5, max_gap_sec: float = 1.0, strict: bool = False,
                 min_gap_sec: float = 1.2, soft_prob: float = 0.25):
        # min_gap_sec = 1.2 chọn theo phát lại (eval_replay_log.py): 0.5 và 0.8 làm tụt từ đúng 25 -> 23 và sinh chốt lặp; 1.2 giữ nguyên kết quả khóa nghiêm.
        self.soft_prob = float(soft_prob)           # strict: stroke khong co span nhung p >= nguong nay van duoc cho qua
        self._probs: List[Tuple[float, float]] = [] # (t, p) gan day, de xem p trong luc stroke
        self.min_gap_sec = float(min_gap_sec)       # khoang toi thieu giua hai lan chot (hai tu khac nhau) trong cung span
        # strict=True: stroke KHONG thuoc span nao (spotter bao 'khong ky') bi CHAN thay vi cho qua -> loai tu vu vo.
        self.strict = bool(strict)
        self.max_spans = int(max_spans)
        self.tolerance_sec = float(tolerance_sec)   # cho phep stroke bat dau truoc t_open cua span toi toi da chung nay
        self.max_gap_sec = float(max_gap_sec)       # span da dong qua lau (> chung nay) thi khong con tinh la "gan nhat truoc"
        # Moi phan tu: [span_id, t_open, t_close|None, used]
        self._spans: List[list] = []
        self._next_id = 0
        self._prev_open = False
        self.last_pmax = 0.0
        self.stats: Dict[str, int] = {"allowed": 0, "suppressed": 0}

    @property
    def spans(self) -> List[Tuple[int, float, Optional[float]]]:
        """Danh sach (span_id, t_open, t_close|None) dang giu trong bo nho."""
        return [(s[0], s[1], s[2]) for s in self._spans]

    def observe(self, t: float, gate_open: bool, prob: Optional[float] = None) -> None:
        """Nhan gate_open cua khung tai thoi diem t; mo/dong span theo canh len/xuong."""
        t = float(t)
        if prob is not None:
            self._probs.append((t, float(prob)))
            if len(self._probs) > 300:
                del self._probs[:100]
        gate_open = bool(gate_open)
        if gate_open and not self._prev_open:
            self._spans.append([self._next_id, t, None, False, None, 0.0])
            self._next_id += 1
            if len(self._spans) > self.max_spans:
                del self._spans[: len(self._spans) - self.max_spans]
        elif (not gate_open) and self._prev_open:
            if self._spans and self._spans[-1][2] is None:
                self._spans[-1][2] = t
        self._prev_open = gate_open

    def _find_span(self, t_start: float) -> Optional[list]:
        """Span chua t_start, neu khong thi span gan nhat bat dau truoc (hoac sat) t_start."""
        containing = None
        before = None
        after = None
        for s in self._spans:   # theo thu tu thoi gian
            t_open, t_close = s[1], s[2]
            if t_open <= t_start:
                before = s
                if t_close is None or t_start <= t_close:
                    containing = s
            elif after is None and t_open - self.tolerance_sec <= t_start:
                after = s   # stroke bat dau ngay truoc khi spotter mo (spotter mo tre open_hold_sec)
        if containing is not None:
            return containing
        if after is not None:
            return after   # stroke bat dau trong khoang nghi, ngay truoc khi span moi mo => thuoc span moi
        if before is not None and before[2] is not None and t_start - before[2] <= self.max_gap_sec:
            return before
        return None

    def allow_commit(self, t_stroke_start: float, word: Optional[str] = None) -> Tuple[bool, str]:
        """
        word: tu vua doc ra. Neu co, ky nhanh (gate khong dong giua hai tu) van duoc chot them MOT TU KHAC
        trong cung span khi stroke bat dau cach lan chot truoc >= min_gap_sec. Cung tu => van la doc lap.
        """
        t_start = float(t_stroke_start)
        span = self._find_span(t_start)
        if span is None:
            if self.strict:
                pmax = max((p for tt, p in self._probs if tt >= t_start - 0.3), default=0.0)
                self.last_pmax = pmax
                if pmax >= self.soft_prob:
                    self.stats["allowed"] += 1
                    return True, "soft"
                self.stats["suppressed"] += 1
                return False, "no_span"
            self.stats["allowed"] += 1
            return True, "no_span"
        if span[3]:
            last_word, last_t = span[4], span[5]
            if word is not None and last_word is not None and word != last_word \
                    and t_start - last_t >= self.min_gap_sec:
                span[4], span[5] = word, t_start
                self.stats["allowed"] += 1
                return True, "next_word"
            self.stats["suppressed"] += 1
            return False, "dup_in_span"
        span[3] = True
        span[4], span[5] = word, t_start
        self.stats["allowed"] += 1
        return True, "first"

    def reset(self) -> None:
        self._spans.clear()
        self._prev_open = False
        self.stats = {"allowed": 0, "suppressed": 0}
