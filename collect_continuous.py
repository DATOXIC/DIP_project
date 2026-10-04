"""
THU PHIÊN LIÊN TỤC CÓ PHÍM GÁN NHÃN (CONTINUOUS LABELLED SESSION COLLECTOR)
==========================================================================
Mục đích: lấy dữ liệu huấn luyện cho Binary Spotter (đang ký / không ký). Khác với kho mẫu âm cũ
(negative_strokes.json: nét cắt sẵn, không có landmark thô), script này ghi LUỒNG LIÊN TỤC:
  * landmark thô của TỪNG khung (2 tay, NaN khi không thấy tay) + hộp mặt (mỏ neo thân) + khung mới/độ sáng;
  * danh sách sự kiện có dấu thời gian cùng trục t (prompt_begin/end, sign_start/end, boundary, bad, ...);
  * video session.mp4 (khung i của video == chỉ số i trong frames.npz) để gán nhãn lại sau này.

Phiên đi theo KỊCH BẢN cố định (build_script, xác định theo seed) gồm 4 loại prompt:
  REST   : hạ tay nghỉ vài giây (tự đếm giờ, không cần phím);
  SIGN   : ký 1 từ rồi hạ tay (SPACE lúc bắt đầu ký, SPACE lần nữa lúc kết thúc);
  PAIR   : ký 2 từ liền nhau không hạ tay (SPACE bắt đầu, 'm' ở điểm ranh giới giữa 2 từ, SPACE kết thúc);
  WANDER : 30 giây cử động vu vơ theo gợi ý cố định (tự đếm giờ).
Protocol A (settle-then-sign): trước mỗi lần bấm SPACE bắt đầu, người ký hạ tay/đứng yên ~1 giây.

Tái sử dụng: ParallelHandDetector / open_camera / decode_fourcc (pipeline_common), CachedFaceDetector
(dynamic_features), CameraHealthMonitor (camera_health), TextRenderer (ui_render), từ vựng + chú giải + cặp câu
(collect_phrases). Phần logic (build_script, ScriptRunner, derive_frame_labels, lưu/đọc phiên) là hàm thuần,
KHÔNG cần webcam; chỉ run_session mới import cv2/mediapipe.

Cách dùng:
  py collect_continuous.py --signer V03 [--out continuous_data] [--camera 0] [--lang vi] [--no-video] [--pairs 8]
  py collect_continuous.py --signer V03 --synthetic            # người ký giả lập, không webcam, vẫn ghi file
  py collect_continuous.py --signer V03 --dry-run              # như --synthetic nhưng KHÔNG ghi file, chỉ in thống kê
  py collect_continuous.py --summary continuous_data/V03       # thống kê frame/giây mỗi lớp của các phiên
Phím: SPACE (bắt đầu/kết thúc ký), m (ranh giới trong PAIR), x (lần thử vừa rồi hỏng -> bị loại),
      n/Enter (prompt tiếp), b (làm lại prompt hiện tại), q (kết thúc & lưu).
"""

import os
import sys
import json
import time
import random
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import collect_phrases as cp

# ---------------------------------------------------------------------------
# HẰNG SỐ
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "continuous_v1.0"
DEFAULT_OUT_DIR = "continuous_data"
DEFAULT_SEED = 20261003
DEFAULT_PAIRS = 8
REST_SEC = 4.0                 # thời lượng một prompt REST (sau phần chuẩn bị)
REST_PREP_SEC = 1.5            # chuẩn bị trước REST: người ký kịp hạ tay (KHÔNG nằm trong prompt)
WANDER_SEC = 30.0
WANDER_PREP_SEC = 3.0
DEFAULT_MARGIN_SEC = 0.15
HAND_MODEL_PATH = "hand_landmarker.task"
DETECT_FACE_EVERY_N = 8

# Nhãn mỗi khung
IGNORE, SIGN, REST, WANDER, TRANSITION = 0, 1, 2, 3, 4
LABEL_NAMES = {IGNORE: "IGNORE", SIGN: "SIGN", REST: "REST", WANDER: "WANDER", TRANSITION: "TRANSITION"}

# Loại prompt
P_REST, P_SIGN, P_PAIR, P_WANDER = "REST", "SIGN", "PAIR", "WANDER"
TIMED_KINDS = (P_REST, P_WANDER)
SIGN_KINDS = (P_SIGN, P_PAIR)

# Gợi ý WANDER cố định: mỗi khối một danh sách khóa, đổi gợi ý đều nhau trong 30 s
WANDER_HINT_SETS = [
    ["scratch_face", "fix_hair", "wave", "hold_cup", "type_keyboard", "hands_up_down"],
    ["rest_chin", "rub_hands", "point_around", "hold_phone", "stretch", "light_clap"],
]

# Chuỗi giao diện vi/en (riêng của script này; ui_render.UI_STRINGS không có các khóa này)
STRINGS = {
    "vi": {
        "title_rest": "Hạ tay xuống nghỉ",
        "title_sign": "KÝ: {w}",
        "title_pair": "KÝ LIỀN: {w}",
        "title_wander": "CỬ ĐỘNG VU VƠ (không ký)",
        "prep": "Chuẩn bị... {n:.0f}",
        "idle": "Bấm SPACE để bắt đầu phiên",
        "st_wait_sign": "CHƯA ghi nhãn ký — đứng yên ~1 s rồi bấm SPACE khi bắt đầu ký",
        "st_signing": "ĐANG GHI NHÃN KÝ {s:4.1f}s — ký xong bấm SPACE NGAY, vẫn GIỮ tay",
        "st_pair_mark": "Xong từ 1: bấm m (giữ tay), rồi chuyển sang từ 2",
        "st_pair_mark2": "Vào vị trí ổn định của từ 2 thì bấm m lần nữa, rồi ký",
        "st_signed": "Đã ghi nhãn — GIỮ tay. Hỏng bấm x; Enter khi sẵn sàng (sau đó mới hạ tay)",
        "st_rest": "NGHỈ — tay buông tự nhiên, còn {s:3.1f}s",
        "st_wander": "VU VƠ — còn {s:4.1f}s",
        "st_done": "HOÀN TẤT — đang lưu...",
        "head": "Prompt {i}/{n}  ·  còn {r}",
        "keys": "SPACE ký/xong  ·  m (x2 trong cặp)  ·  x hỏng  ·  n/Enter tiếp  ·  b lại  ·  q lưu & thoát",
        "hand_yes": "tay: có", "hand_no": "tay: không",
        "bad_flash": "Đã đánh dấu HỎNG (sẽ bị loại)",
        "back_flash": "Làm lại prompt này",
        "hint_scratch_face": "gãi mặt", "hint_fix_hair": "chỉnh tóc", "hint_wave": "vẫy tay lơ đãng",
        "hint_hold_cup": "cầm cốc uống nước", "hint_type_keyboard": "gõ bàn phím",
        "hint_hands_up_down": "đưa tay lên xuống", "hint_rest_chin": "chống cằm",
        "hint_rub_hands": "xoa hai tay", "hint_point_around": "chỉ trỏ vu vơ",
        "hint_hold_phone": "cầm điện thoại", "hint_stretch": "vươn vai", "hint_light_clap": "vỗ tay nhẹ",
        "hint_talk_gesture": "vừa nói vừa múa tay (KHÔNG ký)", "hint_reach_item": "vươn tay lấy đồ trên bàn",
        "hint_hands_in_out": "đưa tay vào rồi ra khỏi khung hình", "hint_cross_arms": "khoanh tay rồi buông",
        "hint_adjust_glasses": "chỉnh kính / sờ cổ áo", "hint_tidy_desk": "dọn / xếp đồ trên bàn",
        "hint_scratch_arm": "gãi tay, xoa cổ", "hint_lean_on_hand": "tì tay lên má", "hint_open_bottle": "mở nắp chai, rót nước",
        "hint_hold_pen": "cầm bút viết", "hint_pat_knee": "gõ tay lên đùi", "hint_phone_scroll": "lướt điện thoại",
    },
    "en": {
        "title_rest": "Lower your hands and rest",
        "title_sign": "SIGN: {w}",
        "title_pair": "SIGN BACK-TO-BACK: {w}",
        "title_wander": "FREE MOVEMENT (no signing)",
        "prep": "Get ready... {n:.0f}",
        "idle": "Press SPACE to start the session",
        "st_wait_sign": "NOT labelling — settle ~1 s, then press SPACE when you start signing",
        "st_signing": "LABELLING SIGN {s:4.1f}s — press SPACE right when done, KEEP hand up",
        "st_pair_mark": "Finished word 1: press m (keep hand up), then move to word 2",
        "st_pair_mark2": "Settled at word 2 start: press m again, then sign",
        "st_signed": "Labelled - KEEP hand up. Bad: x; Enter when ready (then lower hand)",
        "st_rest": "REST — hands relaxed, {s:3.1f}s left",
        "st_wander": "WANDER — {s:4.1f}s left",
        "st_done": "DONE — saving...",
        "head": "Prompt {i}/{n}  ·  {r} left",
        "keys": "SPACE sign/done  ·  m (x2 in pair)  ·  x bad  ·  n/Enter next  ·  b redo  ·  q save & quit",
        "hand_yes": "hand: yes", "hand_no": "hand: no",
        "bad_flash": "Marked BAD (will be excluded)",
        "back_flash": "Redoing this prompt",
        "hint_scratch_face": "scratch your face", "hint_fix_hair": "fix your hair", "hint_wave": "wave absent-mindedly",
        "hint_hold_cup": "hold a cup, drink", "hint_type_keyboard": "type on a keyboard",
        "hint_hands_up_down": "raise and lower hands", "hint_rest_chin": "rest your chin on your hand",
        "hint_rub_hands": "rub your hands", "hint_point_around": "point at things idly",
        "hint_hold_phone": "hold a phone", "hint_stretch": "stretch", "hint_light_clap": "clap lightly",
        "hint_talk_gesture": "talk while gesturing (NOT signing)", "hint_reach_item": "reach for something on the desk",
        "hint_hands_in_out": "move hands into then out of the frame", "hint_cross_arms": "cross arms then drop them",
        "hint_adjust_glasses": "adjust glasses / touch your collar", "hint_tidy_desk": "tidy things on the desk",
        "hint_scratch_arm": "scratch your arm, rub your neck", "hint_lean_on_hand": "rest your cheek on your hand",
        "hint_open_bottle": "open a bottle, pour water", "hint_hold_pen": "hold a pen and write",
        "hint_pat_knee": "tap your hand on your knee", "hint_phone_scroll": "scroll a phone",
    },
}


def T(lang: str, key: str, **fmt) -> str:
    """Tra chuỗi giao diện; thiếu khóa -> rơi về 'en' -> rơi về tên khóa."""
    text = STRINGS.get(lang, {}).get(key)
    if text is None:
        text = STRINGS["en"].get(key, key)
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return text
    return text


# ===========================================================================
# 1. KỊCH BẢN
# ===========================================================================

def _pick_pairs(vocab: Sequence[str], n_pairs: int, rng: random.Random) -> List[Tuple[str, str]]:
    """Chọn n_pairs cặp 2 từ khác nhau (không lặp từ liền nhau). Ưu tiên cặp tự nhiên của collect_phrases.CORE_2,
    thiếu thì bù bằng cặp ngẫu nhiên từ vocab (có seed)."""
    vs = sorted(set(vocab))
    if n_pairs <= 0 or len(vs) < 2:
        return []
    vset = set(vs)
    pool = [s for s in cp._parse_pool(cp.CORE_2) if len(s) == 2 and cp._valid_sentence(s, vset)]
    pool = sorted(set(pool))
    rng.shuffle(pool)
    chosen: List[Tuple[str, str]] = list(pool[:n_pairs])
    seen = set(chosen)
    max_possible = len(vs) * (len(vs) - 1)
    guard = 0
    while len(chosen) < min(n_pairs, max_possible) and guard < 10000:
        guard += 1
        a, b = rng.sample(vs, 2)
        if (a, b) not in seen:
            seen.add((a, b))
            chosen.append((a, b))
    return chosen


def build_script(vocab: Sequence[str], pairs: int = DEFAULT_PAIRS, seed: int = DEFAULT_SEED,
                 rest_sec: float = REST_SEC, wander_sec: float = WANDER_SEC) -> List[Dict[str, Any]]:
    """
    Sinh kịch bản phiên (xác định theo seed, độc lập thứ tự đầu vào của vocab):
      REST mở đầu -> các "đơn vị" (SIGN từ / PAIR cặp, mỗi đơn vị theo sau là REST) xáo ngẫu nhiên
      -> khối WANDER thứ nhất ở giữa -> ... -> khối WANDER thứ hai ở cuối.
    Mỗi prompt: {"index", "type", "words", "duration_sec" (None nếu do người bấm), "hints" (chỉ WANDER)}.
    """
    rng = random.Random(seed)
    vs = sorted(set(vocab))

    def rest() -> Dict[str, Any]:
        return {"type": P_REST, "words": [], "duration_sec": float(rest_sec), "hints": []}

    units: List[List[Dict[str, Any]]] = []
    words = list(vs)
    rng.shuffle(words)
    for w in words:
        units.append([{"type": P_SIGN, "words": [w], "duration_sec": None, "hints": []}, rest()])
    for a, b in _pick_pairs(vs, pairs, rng):
        units.append([{"type": P_PAIR, "words": [a, b], "duration_sec": None, "hints": []}, rest()])
    rng.shuffle(units)

    def wander(k: int) -> Dict[str, Any]:
        return {"type": P_WANDER, "words": [], "duration_sec": float(wander_sec),
                "hints": list(WANDER_HINT_SETS[k % len(WANDER_HINT_SETS)])}

    script: List[Dict[str, Any]] = [rest()]
    mid = len(units) // 2
    for i, unit in enumerate(units):
        if i == mid:
            script.append(wander(0))
        script.extend(unit)
    if not units:
        script.append(wander(0))
    script.append(wander(1))
    for i, p in enumerate(script):
        p["index"] = i
    return script


# Phiên CHỈ MẪU ÂM: mọi giây đều KHÔNG ký -> cả phiên là nhãn âm (WANDER/REST), không cần gán nhãn từng đoạn.
NEG_HINT_SETS = [
    ["scratch_face", "fix_hair", "hold_cup", "type_keyboard", "rest_chin", "stretch"],
    ["talk_gesture", "reach_item", "hands_in_out", "cross_arms", "adjust_glasses", "tidy_desk"],
    ["scratch_arm", "lean_on_hand", "open_bottle", "hold_pen", "pat_knee", "phone_scroll"],
    ["hold_phone", "rub_hands", "hands_up_down", "point_around", "light_clap", "fix_hair"],
]


def build_negative_script(minutes: float = 4.0, seed: int = DEFAULT_SEED, wander_sec: float = WANDER_SEC,
                          rest_sec: float = REST_SEC) -> List[Dict[str, Any]]:
    """
    Kịch bản chỉ gồm WANDER (30 s, đổi gợi ý đều) xen REST ngắn, xáo thứ tự bộ gợi ý theo seed. Thời gian ~ `minutes`
    (mỗi đơn vị WANDER+REST ~ 38,5 s kể cả chuẩn bị). Tối thiểu 2 đơn vị.
    """
    rng = random.Random(seed)
    unit = wander_sec + WANDER_PREP_SEC + rest_sec + REST_PREP_SEC
    n = max(2, int(round(minutes * 60.0 / unit)))
    order: List[int] = []
    while len(order) < n:
        block = list(range(len(NEG_HINT_SETS)))
        rng.shuffle(block)
        order.extend(block)
    script: List[Dict[str, Any]] = []
    for k in order[:n]:
        hints = list(NEG_HINT_SETS[k])
        rng.shuffle(hints)
        script.append({"type": P_WANDER, "words": [], "duration_sec": float(wander_sec), "hints": hints})
        script.append({"type": P_REST, "words": [], "duration_sec": float(rest_sec), "hints": []})
    for i, p in enumerate(script):
        p["index"] = i
    return script


def estimate_duration_sec(script: Sequence[Dict[str, Any]], sign_sec: float = 9.0) -> float:
    """Ước lượng tổng thời gian (giây): prompt tự đếm giờ cộng chuẩn bị; SIGN/PAIR ước tính ~sign_sec mỗi cái."""
    total = 0.0
    for p in script:
        if p["type"] == P_REST:
            total += REST_PREP_SEC + float(p["duration_sec"])
        elif p["type"] == P_WANDER:
            total += WANDER_PREP_SEC + float(p["duration_sec"])
        else:
            total += sign_sec * (1.0 + 0.5 * (len(p["words"]) - 1))
    return total


# ===========================================================================
# 2. MÁY TRẠNG THÁI PHÍM (THUẦN, KHÔNG CV2)
# ===========================================================================

class ScriptRunner:
    """
    FSM chạy kịch bản. Mọi thời gian truyền vào tường minh (t, giây trên trục phiên) nên kiểm thử được bằng phím giả.

    Trạng thái: IDLE (chưa start) / PREP (chuẩn bị trước prompt tự đếm giờ) / TIMED (REST|WANDER đang đếm) /
                READY (SIGN|PAIR chờ SPACE) / SIGNING / SIGNED (đã SPACE kết thúc, chờ n) / FINISHED.
    Mỗi lần thực hiện một prompt là một "attempt" (aid); thu lại (b) sẽ phát sự kiện attempt_discard cho aid cũ,
    'x' phát sự kiện bad cho aid. Mọi sự kiện đều mang prompt_index và aid.
    """

    def __init__(self, script: Sequence[Dict[str, Any]], rest_prep_sec: float = REST_PREP_SEC,
                 wander_prep_sec: float = WANDER_PREP_SEC):
        self.script = list(script)
        self.rest_prep_sec = float(rest_prep_sec)
        self.wander_prep_sec = float(wander_prep_sec)
        self.events: List[Dict[str, Any]] = []
        self.state = "IDLE"
        self.index = -1
        self.cur_aid: Optional[int] = None
        self._next_aid = 1
        self.history: List[int] = []                 # aid theo thứ tự tạo, chỉ các attempt chưa bị bỏ
        self._aid_index: Dict[int, int] = {}         # aid -> prompt_index
        self._last_aid_by_index: Dict[int, int] = {}
        self.bad_aids: set = set()
        self.discarded_aids: set = set()
        self._prep_end = 0.0
        self._timed_start = 0.0
        self._timed_end = 0.0
        self._sign_open = False
        self._sign_start_t = 0.0
        self._n_boundaries = 0
        self.last_msg = ""                           # mã thông báo gần nhất cho HUD
        self.last_msg_t = -1e9

    # ---- tiện ích ----
    @property
    def finished(self) -> bool:
        return self.state == "FINISHED"

    @property
    def prompt(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.index < len(self.script):
            return self.script[self.index]
        return None

    @property
    def remaining(self) -> int:
        return max(0, len(self.script) - self.index - 1)

    def _emit(self, etype: str, t: float, aid: Optional[int] = None, prompt_index: Optional[int] = None, **kw) -> None:
        ev: Dict[str, Any] = {"t": round(float(t), 6), "type": etype,
                              "prompt_index": self.index if prompt_index is None else prompt_index,
                              "aid": self.cur_aid if aid is None else aid}
        ev.update(kw)
        self.events.append(ev)

    def _msg(self, code: str, t: float) -> None:
        self.last_msg, self.last_msg_t = code, float(t)

    # ---- chuyển prompt ----
    def start(self, t: float) -> None:
        """Bắt đầu kịch bản (vào prompt đầu tiên)."""
        if self.state != "IDLE":
            return
        self._emit("session_start", t, aid=None, prompt_index=-1)
        self._enter(0, t)

    def _enter(self, idx: int, t: float) -> None:
        if idx >= len(self.script):
            self.finish(t)
            return
        self.index = idx
        self.cur_aid = None
        self._sign_open = False
        self._n_boundaries = 0
        kind = self.script[idx]["type"]
        if kind in TIMED_KINDS:
            prep = self.rest_prep_sec if kind == P_REST else self.wander_prep_sec
            self._prep_end = t + prep
            self.state = "PREP"
            if prep <= 0:
                self._begin_prompt(t)
        else:
            self._begin_prompt(t)

    def _begin_prompt(self, t: float) -> None:
        p = self.script[self.index]
        aid = self._next_aid
        self._next_aid += 1
        self.cur_aid = aid
        self.history.append(aid)
        self._aid_index[aid] = self.index
        self._last_aid_by_index[self.index] = aid
        words = list(p["words"])
        self._emit("prompt_begin", t, kind=p["type"], words=words,
                   word=(words[0] if len(words) == 1 else None), duration_sec=p.get("duration_sec"))
        if p["type"] in TIMED_KINDS:
            self.state = "TIMED"
            self._timed_start = t
            self._timed_end = t + float(p["duration_sec"])
        else:
            self.state = "READY"

    def _end_prompt(self, t: float) -> None:
        """Đóng prompt đang mở (nếu đã bắt đầu): sign_end tự động nếu quên, rồi prompt_end."""
        if self.cur_aid is None or self.state in ("PREP", "IDLE", "FINISHED"):
            return
        if self._sign_open:
            self._emit("sign_end", t, auto=True)
            self._sign_open = False
        self._emit("prompt_end", t)

    def tick(self, t: float) -> None:
        """Gọi mỗi khung: xử lý chuyển prompt tự đếm giờ (PREP -> TIMED -> prompt kế)."""
        for _ in range(10000):
            if self.state == "PREP" and t >= self._prep_end:
                self._begin_prompt(self._prep_end)
                continue
            if self.state == "TIMED" and t >= self._timed_end:
                te = self._timed_end
                self._end_prompt(te)
                self._enter(self.index + 1, te)
                continue
            break

    def finish(self, t: float) -> None:
        if self.state == "FINISHED":
            return
        self._end_prompt(t)
        self._emit("session_end", t, aid=None)
        self.state = "FINISHED"

    # ---- phím ----
    def key(self, k: str, t: float) -> str:
        """
        Xử lý một phím: ' ' (SPACE), 'm', 'x', 'n' (cũng nhận '\\r', '\\n'), 'b', 'q'.
        Trả mã hành động ('sign_start','sign_end','boundary','bad','next','back','quit', hoặc 'ignored').
        """
        if k in ("\r", "\n", "enter", "Enter"):
            k = "n"
        k = k.lower() if len(k) == 1 else k
        if self.state in ("IDLE", "FINISHED"):
            return "ignored"
        self.tick(t)
        if self.state == "FINISHED":
            return "ignored"
        kind = self.prompt["type"] if self.prompt else None
        out = "ignored"
        if k == " ":
            if kind in SIGN_KINDS and self.state == "READY":
                self._emit("sign_start", t)
                self._sign_open = True
                self._sign_start_t = t
                self.state = "SIGNING"
                out = "sign_start"
            elif kind in SIGN_KINDS and self.state == "SIGNING":
                self._emit("sign_end", t)
                self._sign_open = False
                self.state = "SIGNED"
                out = "sign_end"
        elif k == "m":
            if kind == P_PAIR and self.state == "SIGNING" and self._n_boundaries < 2 * (len(self.prompt["words"]) - 1):
                self._emit("boundary", t, index=self._n_boundaries)
                self._n_boundaries += 1
                out = "boundary"
        elif k == "x":
            target = self._bad_target()
            if target is not None and target not in self.bad_aids:
                self.bad_aids.add(target)
                self._emit("bad", t, aid=target, prompt_index=self._aid_index[target])
                out = "bad"
        elif k == "n":
            self._end_prompt(t)
            self._enter(self.index + 1, t)
            out = "next"
        elif k == "b":
            self._back(t)
            out = "back"
        elif k == "q":
            self.finish(t)
            out = "quit"
        if out != "ignored":
            self._msg(out, t)
        return out

    def _bad_target(self) -> Optional[int]:
        """Attempt mà 'x' nhắm tới: attempt hiện tại nếu đã có nội dung (đang/đã ký hoặc đang đếm giờ),
        ngược lại là attempt trước đó."""
        if self.cur_aid is not None and self.state in ("SIGNING", "SIGNED", "TIMED"):
            return self.cur_aid
        cands = [a for a in self.history if a != self.cur_aid]
        return cands[-1] if cands else None

    def _discard(self, aid: int, t: float) -> None:
        if aid in self.discarded_aids:
            return
        self.discarded_aids.add(aid)
        if aid in self.history:
            self.history.remove(aid)
        self._emit("attempt_discard", t, aid=aid, prompt_index=self._aid_index[aid])

    def _back(self, t: float) -> None:
        """b: LÀM LẠI prompt hiện tại - chỉ bỏ attempt đang làm, KHÔNG đụng attempt của prompt trước
        (attempt trước muốn loại thì dùng 'x')."""
        i = self.index
        cur = self.cur_aid
        if cur is not None:
            self._end_prompt(t)
            self._discard(cur, t)
        self._enter(max(0, i), t)

    # ---- ảnh chụp cho HUD ----
    def snapshot(self, t: float) -> Dict[str, Any]:
        p = self.prompt
        snap: Dict[str, Any] = {
            "state": self.state, "index": self.index, "total": len(self.script), "remaining": self.remaining,
            "prompt": p, "kind": p["type"] if p else None, "labelling": self.state == "SIGNING",
            "sign_elapsed": (t - self._sign_start_t) if self.state == "SIGNING" else 0.0,
            "prep_left": max(0.0, self._prep_end - t) if self.state == "PREP" else 0.0,
            "timed_left": max(0.0, self._timed_end - t) if self.state == "TIMED" else 0.0,
            "timed_elapsed": (t - self._timed_start) if self.state == "TIMED" else 0.0,
            "boundaries": self._n_boundaries, "msg": self.last_msg, "msg_t": self.last_msg_t,
            "cur_is_bad": self.cur_aid in self.bad_aids if self.cur_aid is not None else False,
        }
        return snap


# ===========================================================================
# 3. NHÃN TỪNG KHUNG (THUẦN)
# ===========================================================================

def _events_list(events: Any) -> List[Dict[str, Any]]:
    if isinstance(events, dict):
        return list(events.get("events", []))
    return list(events or [])


def _collect_attempts(evs: List[Dict[str, Any]], t_end: float = 0.0) -> Dict[int, Dict[str, Any]]:
    att: Dict[int, Dict[str, Any]] = {}
    # prompt/ký chưa đóng (phiên bị cắt giữa chừng) kéo dài tới hết dữ liệu
    t_last = max([float(e["t"]) for e in evs] + [float(t_end)])
    for e in evs:
        aid = e.get("aid")
        et = e.get("type")
        if aid is None:
            continue
        if et == "prompt_begin":
            att[aid] = {"kind": e.get("kind"), "tb": float(e["t"]), "te": None, "signs": [], "boundaries": [],
                        "bad": False, "discarded": False}
            continue
        a = att.get(aid)
        if a is None:
            continue
        if et == "prompt_end":
            a["te"] = float(e["t"])
        elif et == "sign_start":
            a["signs"].append([float(e["t"]), None])
        elif et == "sign_end":
            if a["signs"] and a["signs"][-1][1] is None:
                a["signs"][-1][1] = float(e["t"])
        elif et == "boundary":
            a["boundaries"].append(float(e["t"]))
        elif et == "bad":
            a["bad"] = True
        elif et == "attempt_discard":
            a["discarded"] = True
    for a in att.values():
        if a["te"] is None:
            a["te"] = t_last
        for s in a["signs"]:
            if s[1] is None:
                s[1] = a["te"]
    return att


def derive_frame_labels(t: np.ndarray, events: Any, margin_sec: float = DEFAULT_MARGIN_SEC) -> np.ndarray:
    """
    Nhãn mỗi khung theo thời gian t (giây): 0 IGNORE, 1 SIGN, 2 REST, 3 WANDER, 4 TRANSITION.
      * REST / WANDER: toàn prompt, co lại margin_sec ở hai đầu (vùng đệm ranh giới nhãn -> IGNORE).
      * SIGN / PAIR: chỉ khoảng sign_start..sign_end, co margin_sec ở hai đầu (phản xạ người bấm trễ/sớm);
        PAIR: vùng ±margin quanh mỗi 'boundary' -> TRANSITION (nằm trong khoảng SIGN đã co).
      * Ngoài khoảng bấm trong prompt SIGN/PAIR -> IGNORE; attempt bị 'x' (bad) hoặc bị bỏ (attempt_discard) -> IGNORE.
    `events` là dict (khóa "events") hoặc thẳng danh sách sự kiện.
    """
    t = np.asarray(t, dtype=np.float64)
    labels = np.zeros(t.shape[0], dtype=np.uint8)
    m = float(margin_sec)
    t_end = float(t.max()) if t.size else 0.0
    for a in _collect_attempts(_events_list(events), t_end).values():
        if a["bad"] or a["discarded"]:
            continue
        kind = a["kind"]
        if kind in (P_REST, P_WANDER):
            sel = (t >= a["tb"] + m) & (t <= a["te"] - m)
            labels[sel] = REST if kind == P_REST else WANDER
        elif kind in SIGN_KINDS:
            for ss, se in a["signs"]:
                if se <= ss:
                    continue
                sel = (t >= ss + m) & (t <= se - m)
                labels[sel] = SIGN
                if kind == P_PAIR:
                    bs = sorted(b for b in a["boundaries"] if ss <= b <= se)
                    if len(bs) >= 2:
                        # 2 lần bấm m: b0 = vừa XONG từ 1, b1 = đã ổn định ở vị trí từ 2.
                        # Toàn bộ b0..b1 là đoạn chuyển tiếp (epenthesis); co margin vào trong ở hai đầu.
                        b0, b1 = bs[0], bs[1]
                        labels[sel] = SIGN
                        labels[(t > b0 + m) & (t < b1 - m)] = TRANSITION
                        labels[(t >= b0 - m) & (t <= b0 + m)] = IGNORE   # phản xạ người bấm trễ
                        labels[(t >= b1 - m) & (t <= b1 + m)] = IGNORE
                    elif len(bs) == 1:
                        # tương thích dữ liệu cũ (1 lần bấm m): TRANSITION chỉ là cửa sổ ±margin quanh điểm bấm.
                        labels[sel & (t >= bs[0] - m) & (t <= bs[0] + m)] = TRANSITION
                    else:
                        # PAIR không có điểm ranh giới: đoạn chuyển tiếp bị lẫn vào SIGN -> bỏ cả đoạn.
                        labels[sel] = IGNORE
    return labels


def label_durations(t: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Số khung và số giây mỗi lớp (giây = tổng khoảng cách tới khung kế; khung cuối dùng dt trung vị)."""
    t = np.asarray(t, dtype=np.float64)
    out: Dict[str, Dict[str, float]] = {}
    if t.size == 0:
        return {n: {"frames": 0, "seconds": 0.0} for n in LABEL_NAMES.values()}
    dt = np.diff(t)
    dt = np.append(dt, np.median(dt) if dt.size else 0.0)
    for k, name in LABEL_NAMES.items():
        sel = labels == k
        out[name] = {"frames": int(sel.sum()), "seconds": float(dt[sel].sum())}
    return out


# ===========================================================================
# 4. GHI KHUNG, LƯU & ĐỌC PHIÊN
# ===========================================================================

class FrameRecorder:
    """Gom dữ liệu thô từng khung. Khe tay: 0 = nhãn 'Left', 1 = nhãn 'Right' (nhãn MediaPipe thô, KHÔNG đảo)."""

    def __init__(self):
        self.t: List[float] = []
        self.landmarks: List[np.ndarray] = []
        self.hand_label: List[Tuple[str, str]] = []
        self.hand_score: List[Tuple[float, float]] = []
        self.face: List[Tuple[float, float, float, float]] = []
        self.is_new: List[bool] = []
        self.luma: List[float] = []

    def __len__(self) -> int:
        return len(self.t)

    def add(self, t: float, hands: Sequence[Tuple[str, float, np.ndarray]], face: Optional[Sequence[float]] = None,
            is_new: bool = True, luma: float = 0.0) -> None:
        """hands: danh sách (nhãn 'Left'/'Right', điểm, mảng (21,3) toạ độ chuẩn hoá). Tối đa 2 tay."""
        lm = np.full((2, 21, 3), np.nan, dtype=np.float32)
        lab = ["", ""]
        sc = [np.nan, np.nan]
        for label, score, pts in hands[:2]:
            slot = 0 if str(label).lower().startswith("l") else 1
            if lab[slot]:                       # khe đã có (hai tay cùng nhãn): dùng khe còn lại
                slot = 1 - slot
                if lab[slot]:
                    continue
            lm[slot] = np.asarray(pts, dtype=np.float32).reshape(21, 3)
            lab[slot] = str(label)
            sc[slot] = float(score)
        self.t.append(float(t))
        self.landmarks.append(lm)
        self.hand_label.append((lab[0], lab[1]))
        self.hand_score.append((sc[0], sc[1]))
        self.face.append(tuple(float(v) for v in face) if face is not None else (np.nan,) * 4)
        self.is_new.append(bool(is_new))
        self.luma.append(float(luma))

    def arrays(self) -> Dict[str, np.ndarray]:
        n = len(self.t)
        t = np.asarray(self.t, dtype=np.float64)
        fps = float((n - 1) / (t[-1] - t[0])) if n >= 2 and t[-1] > t[0] else 0.0
        return {
            "t": t,
            "landmarks": (np.stack(self.landmarks) if n else np.zeros((0, 2, 21, 3), np.float32)).astype(np.float32),
            "handedness": (np.array(self.hand_label, dtype="<U5") if n else np.zeros((0, 2), dtype="<U5")),
            "hand_score": (np.array(self.hand_score, dtype=np.float32) if n else np.zeros((0, 2), np.float32)),
            "face_box": (np.array(self.face, dtype=np.float32) if n else np.zeros((0, 4), np.float32)),
            "frame_is_new": np.array(self.is_new, dtype=bool),
            "luma_mean": np.array(self.luma, dtype=np.float32),
            "fps": np.array(fps, dtype=np.float64),
        }


def _atomic_write_bytes(path: str, writer) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def save_session(session_dir: str, recorder: FrameRecorder, runner_events: List[Dict[str, Any]],
                 meta: Dict[str, Any], script: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Ghi frames.npz + events.json (nguyên tử: file tạm rồi đổi tên). Trả đường dẫn các file."""
    os.makedirs(session_dir, exist_ok=True)
    arrs = recorder.arrays()
    meta = dict(meta)
    meta["fps"] = float(arrs["fps"])
    meta["num_frames"] = int(arrs["t"].shape[0])
    meta["duration_sec"] = float(arrs["t"][-1] - arrs["t"][0]) if arrs["t"].shape[0] >= 2 else 0.0
    meta["schema_version"] = SCHEMA_VERSION
    npz_path = os.path.join(session_dir, "frames.npz")
    ev_path = os.path.join(session_dir, "events.json")
    _atomic_write_bytes(npz_path, lambda f: np.savez_compressed(f, **arrs))
    payload = {"schema_version": SCHEMA_VERSION, "meta": meta, "script": list(script), "events": list(runner_events)}
    _atomic_write_bytes(ev_path, lambda f: f.write(json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")))
    return {"frames": npz_path, "events": ev_path}


def load_session(session_dir: str, margin_sec: float = DEFAULT_MARGIN_SEC) -> Dict[str, Any]:
    """
    Đọc một phiên. Trả dict: t, landmarks (T,2,21,3), handedness (T,2), hand_score, face_box, frame_is_new,
    luma_mean, fps, events (dict của events.json), meta, script, labels (derive_frame_labels), video_path (hoặc None).
    """
    with np.load(os.path.join(session_dir, "frames.npz"), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["fps"] = float(d["fps"])
    with open(os.path.join(session_dir, "events.json"), "r", encoding="utf-8") as f:
        events = json.load(f)
    d["events"] = events
    d["meta"] = events.get("meta", {})
    d["script"] = events.get("script", [])
    d["labels"] = derive_frame_labels(d["t"], events, margin_sec)
    vp = os.path.join(session_dir, "session.mp4")
    d["video_path"] = vp if os.path.isfile(vp) else None
    d["session_dir"] = session_dir
    return d


def find_session_dirs(root: str) -> List[str]:
    """Các thư mục chứa events.json + frames.npz dưới root (kể cả root)."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        if "events.json" in files and "frames.npz" in files:
            out.append(dirpath)
    return sorted(out)


def summarize(root: str, margin_sec: float = DEFAULT_MARGIN_SEC) -> str:
    """Bảng thống kê số khung/giây mỗi lớp cho từng phiên tìm thấy dưới root và tổng cộng."""
    dirs = find_session_dirs(root)
    if not dirs:
        return f"Không thấy phiên nào dưới '{root}' (cần events.json + frames.npz)."
    names = list(LABEL_NAMES.values())
    tot = {n: {"frames": 0, "seconds": 0.0} for n in names}
    lines: List[str] = []
    n_all = 0
    for d in dirs:
        s = load_session(d, margin_sec)
        dur = label_durations(s["t"], s["labels"])
        n_all += int(s["t"].shape[0])
        lines.append(f"== {d}  (signer={s['meta'].get('signer')}, {s['t'].shape[0]} khung, fps~{s['fps']:.1f}, "
                     f"video={'có' if s['video_path'] else 'không'})")
        bad = sum(1 for e in s["events"]["events"] if e["type"] == "bad")
        disc = sum(1 for e in s["events"]["events"] if e["type"] == "attempt_discard")
        lines.append(f"   {'LỚP':<12}{'KHUNG':>8}{'GIÂY':>9}")
        for n in names:
            lines.append(f"   {n:<12}{dur[n]['frames']:>8}{dur[n]['seconds']:>9.1f}")
            tot[n]["frames"] += dur[n]["frames"]
            tot[n]["seconds"] += dur[n]["seconds"]
        lines.append(f"   attempt bị 'x' (bad): {bad}; bị bỏ khi làm lại (b): {disc}")
    lines.append(f"== TỔNG {len(dirs)} phiên, {n_all} khung")
    lines.append(f"   {'LỚP':<12}{'KHUNG':>8}{'GIÂY':>9}")
    for n in names:
        lines.append(f"   {n:<12}{tot[n]['frames']:>8}{tot[n]['seconds']:>9.1f}")
    return "\n".join(lines)


# ===========================================================================
# 5. NGƯỜI KÝ GIẢ LẬP (--synthetic / --dry-run / kiểm thử): chạy ScriptRunner + FrameRecorder không webcam
# ===========================================================================

def _fake_hand(phase: float, label: str, rng: np.random.Generator) -> np.ndarray:
    """Bàn tay giả (21,3) chuẩn hoá, chuyển động theo phase."""
    base = np.linspace(0.0, 1.0, 21)[:, None] * np.array([0.05, -0.12, 0.0])[None, :]
    cx = 0.5 + 0.15 * np.sin(phase) + (0.15 if label == "Right" else -0.15)
    cy = 0.5 + 0.10 * np.cos(1.7 * phase)
    return (np.array([cx, cy, 0.0])[None, :] + base + rng.normal(0, 0.002, (21, 3))).astype(np.float32)


def simulate_session(script: Sequence[Dict[str, Any]], fps: float = 30.0, seed: int = 0,
                     rest_prep_sec: float = REST_PREP_SEC, wander_prep_sec: float = WANDER_PREP_SEC,
                     sign_hold_sec: float = 2.0, settle_sec: float = 1.0,
                     bad_every: int = 0) -> Tuple[ScriptRunner, FrameRecorder]:
    """
    Robot người ký: với SIGN/PAIR đứng yên settle_sec, SPACE, ký sign_hold_sec (PAIR: 'm' ở giữa), SPACE, chờ 0.5 s, 'n'.
    Prompt tự đếm giờ: chỉ chờ. Nếu bad_every>0, mỗi bad_every prompt SIGN/PAIR gửi thêm 'x' ngay sau khi ký xong.
    Có tay trong khung khi robot đang "ký" hoặc đang WANDER.
    """
    rng = np.random.default_rng(seed)
    runner = ScriptRunner(script, rest_prep_sec=rest_prep_sec, wander_prep_sec=wander_prep_sec)
    rec = FrameRecorder()
    dt = 1.0 / fps
    t = 0.0
    runner.start(t)
    plan_keys: List[Tuple[float, str]] = []   # (thời điểm, phím) đang chờ cho prompt hiện tại
    cur_idx = -1
    n_sign_prompts = 0
    guard = 0
    while not runner.finished and guard < 10_000_000:
        guard += 1
        runner.tick(t)
        if runner.finished:
            break
        if runner.index != cur_idx:
            cur_idx = runner.index
            plan_keys = []
            p = runner.prompt
            if p and p["type"] in SIGN_KINDS:
                n_sign_prompts += 1
                t1 = t + settle_sec
                plan_keys.append((t1, " "))
                if p["type"] == P_PAIR:
                    plan_keys.append((t1 + sign_hold_sec * 0.4, "m"))
                    plan_keys.append((t1 + sign_hold_sec * 0.6, "m"))
                t2 = t1 + sign_hold_sec
                plan_keys.append((t2, " "))
                tn = t2 + 0.5
                if bad_every > 0 and n_sign_prompts % bad_every == 0:
                    plan_keys.append((t2 + 0.2, "x"))
                plan_keys.append((tn, "n"))
        while plan_keys and t >= plan_keys[0][0]:
            _, k = plan_keys.pop(0)
            runner.key(k, t)
            if runner.finished or runner.index != cur_idx:
                break
        hands: List[Tuple[str, float, np.ndarray]] = []
        if runner.state == "SIGNING" or (runner.state == "TIMED" and runner.prompt["type"] == P_WANDER):
            hands.append(("Right", 0.95, _fake_hand(t * 4.0, "Right", rng)))
        rec.add(t, hands, face=(320.0, 160.0, 90.0, 110.0), is_new=True, luma=120.0)
        t += dt
    return runner, rec


# ===========================================================================
# 6. HUD + VÒNG THU THẬT (cv2/mediapipe chỉ import ở đây)
# ===========================================================================

class ContHud(cp.Hud):
    """HUD của phiên liên tục: dùng lại TextRenderer + _band của collect_phrases.Hud."""

    def render_cont(self, frame: np.ndarray, snap: Dict[str, Any], lang: str, hand_present: bool,
                    landmarks_px: Optional[List[Any]], now: float) -> np.ndarray:
        import cv2
        h, w = frame.shape[:2]
        for pts in (landmarks_px or []):
            col = (0, 0, 255) if snap["labelling"] else (0, 220, 0)
            for p in pts:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, col, -1, cv2.LINE_AA)
        self._band(frame, 0, 140)
        self._band(frame, h - 84, h)
        state, p = snap["state"], snap["prompt"]
        head = T(lang, "head", i=max(1, snap["index"] + 1), n=snap["total"], r=snap["remaining"])
        self._txt(frame, head, 12, 6, 17, (200, 200, 200))
        title, sub, status, status_col = "", "", "", (255, 255, 255)
        if state == "IDLE":
            title = T(lang, "idle")
        elif state == "FINISHED":
            title = T(lang, "st_done")
        elif p is not None:
            gloss = lambda wd: (cp.VI_GLOSS.get(wd, wd) if lang == "vi" else wd)  # noqa: E731
            kind = p["type"]
            if kind == P_REST:
                title = T(lang, "title_rest")
            elif kind == P_SIGN:
                title = T(lang, "title_sign", w=p["words"][0])
                sub = gloss(p["words"][0])
            elif kind == P_PAIR:
                title = T(lang, "title_pair", w=" → ".join(p["words"]))
                sub = " → ".join(gloss(x) for x in p["words"])
            else:
                title = T(lang, "title_wander")
                hints = p.get("hints") or []
                if hints:
                    k = min(len(hints) - 1, int(snap["timed_elapsed"] / (float(p["duration_sec"]) / len(hints))))
                    sub = T(lang, "hint_" + hints[k])
            if state == "PREP":
                status, status_col = T(lang, "prep", n=snap["prep_left"] + 0.999), (0, 255, 255)
            elif state == "TIMED":
                status = T(lang, "st_rest" if kind == P_REST else "st_wander", s=snap["timed_left"])
                status_col = (0, 255, 255)
            elif state == "READY":
                status, status_col = T(lang, "st_wait_sign"), (0, 255, 0)
            elif state == "SIGNING":
                status = T(lang, "st_signing", s=snap["sign_elapsed"])
                status_col = (60, 60, 255)
                cv2.circle(frame, (24, h - 58), 9, (0, 0, 255), -1, cv2.LINE_AA)
                if kind == P_PAIR and snap["boundaries"] < 2:
                    self._txt(frame, T(lang, "st_pair_mark" if snap["boundaries"] == 0 else "st_pair_mark2"),
                              w // 2, 112, 20, (0, 220, 255), anchor="c")
            elif state == "SIGNED":
                status, status_col = T(lang, "st_signed"), (0, 255, 0)
        self._txt(frame, title, w // 2, 28, self._fit(title, 48, w - 24, True), (255, 255, 255), bold=True, anchor="c")
        if sub:
            self._txt(frame, sub, w // 2, 80, self._fit(sub, 26, w - 24), (0, 220, 255), anchor="c")
        if status:
            self._txt(frame, status, 42, h - 74, self._fit(status, 20, w - 54, True), status_col, bold=True)
        self._txt(frame, T(lang, "keys"), 12, h - 36, 15, (170, 170, 170))
        self._txt(frame, T(lang, "hand_yes" if hand_present else "hand_no"), w - 12, 6, 17, (170, 170, 170), anchor="r")
        if snap["msg"] in ("bad", "back") and now - snap["msg_t"] < 1.5:
            txt = T(lang, "bad_flash" if snap["msg"] == "bad" else "back_flash")
            self._band(frame, h // 2 - 26, h // 2 + 26, 0.75)
            self._txt(frame, txt, w // 2, h // 2 - 16, self._fit(txt, 30, w - 24, True), (0, 165, 255), bold=True,
                      anchor="c")
        if self.tr is not None:
            self.tr.flush(frame)
        return frame


def _hands_from_result(det_result: Any, w: int, h: int) -> Tuple[List[Tuple[str, float, np.ndarray]], List[np.ndarray]]:
    """MediaPipe -> [(nhãn, điểm, (21,3) chuẩn hoá)] + danh sách điểm px để vẽ."""
    hands, px = [], []
    if det_result is not None and det_result.hand_landmarks:
        for i, lms in enumerate(det_result.hand_landmarks):
            label, score = "Right", 0.0
            if det_result.handedness and i < len(det_result.handedness) and det_result.handedness[i]:
                cat = det_result.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            arr = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
            hands.append((label, score, arr))
            px.append(np.stack([arr[:, 0] * w, arr[:, 1] * h], axis=1))
    return hands, px


_KEYMAP = {32: " ", 13: "n", 10: "n", ord("n"): "n", ord("m"): "m", ord("x"): "x", ord("b"): "b", ord("q"): "q"}


def run_session(signer: str, out_dir: str, camera_index: int, lang: str, no_video: bool, pairs: int,
                seed: int, script: Optional[List[Dict[str, Any]]] = None, suffix: str = "",
                extra_meta: Optional[Dict[str, Any]] = None) -> int:
    """Vòng thu webcam thật. Mỗi khung: đọc -> gửi detector 2 tay -> (theo thứ tự) ghi landmark/mặt/độ sáng/video."""
    import cv2
    import mediapipe as mp
    from pipeline_common import ParallelHandDetector, decode_fourcc, open_camera
    from dynamic_features import CachedFaceDetector
    from camera_health import CameraHealthMonitor

    if script is None:
        vocab = cp.load_vocab_keys(cp.DEFAULT_TEMPLATES)
        script = build_script(vocab, pairs=pairs, seed=seed)
    print(f"[Script] {len(script)} prompt, ước tính ~{estimate_duration_sec(script) / 60:.1f} phút.")
    if not os.path.exists(HAND_MODEL_PATH):
        print(f"[!] Thiếu {HAND_MODEL_PATH} (chạy collect_dynamic_templates.py một lần để tự tải).")
        return 1
    session_id = time.strftime("%Y%m%d_%H%M%S") + suffix
    session_dir = os.path.join(out_dir, signer, session_id)
    os.makedirs(session_dir, exist_ok=True)

    cap = open_camera(camera_index, prefer_mjpg=True)
    if cap is None or not cap.isOpened():
        print(f"[!] KHÔNG THỂ mở camera [{camera_index}]")
        return 1
    fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = ParallelHandDetector(model_path=HAND_MODEL_PATH, workers=None, num_hands=2)
    try:
        face = CachedFaceDetector(detect_every_n_frames=DETECT_FACE_EVERY_N)
    except Exception as e:
        print(f"[!] Không nạp được bộ dò mặt ({e}) -> face_box = NaN.")
        face = None
    monitor = CameraHealthMonitor(cfg=None, warnings_enabled=False)
    hud = ContHud()
    window = "Thu Phien Lien Tuc (Continuous Collector)"

    runner = ScriptRunner(script)
    recorder = FrameRecorder()
    writer = None
    video_path = os.path.join(session_dir, "session.mp4")
    t0 = time.perf_counter()
    clock = lambda: time.perf_counter() - t0   # noqa: E731
    from collections import deque
    ready_q: deque = deque()
    seq = 0
    exhausted = False
    frame_wh = (640, 480)
    print("\n  Bấm SPACE trong cửa sổ video để bắt đầu phiên. q = lưu & thoát.\n")
    try:
        while True:
            now = clock()
            if not exhausted and not runner.finished:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                if not ret:
                    exhausted = True
                    ready_q.extend(detector.flush())
                else:
                    seq += 1
                    ts = clock()
                    monitor.update(frame, ts, False)
                    meta = {"capture_ts": ts, "frame": frame, "is_new": bool(monitor.is_new),
                            "luma_mean": float(monitor.luma_mean)}
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    meta["rgb"] = rgb
                    detector.submit(seq, rgb, meta)
                    ready_q.extend(detector.poll(block_oldest=(detector.in_flight >= detector.max_in_flight)))
            if not ready_q:
                if exhausted or runner.finished:
                    break
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    runner.finish(clock())
                    break
                continue
            _seq, meta, det = ready_q.popleft()
            frame = meta["frame"]
            h_f, w_f = frame.shape[:2]
            frame_wh = (w_f, h_f)
            hands, px = _hands_from_result(det, w_f, h_f)
            fb = None
            if face is not None:
                try:
                    fb = face.detect(meta["rgb"], out_size=(w_f, h_f))
                except Exception:
                    fb = None
            recorder.add(meta["capture_ts"], hands, face=fb, is_new=meta["is_new"], luma=meta["luma_mean"])
            if not no_video:
                if writer is None:
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(cam_fps), (w_f, h_f))
                writer.write(frame)       # khung CHƯA vẽ HUD; chỉ số khung video == chỉ số trong frames.npz
            now = clock()
            runner.tick(now)
            snap = runner.snapshot(now)
            disp = hud.render_cont(frame.copy(), snap, lang, bool(hands), px, now)
            cv2.imshow(window, disp)
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                key = _KEYMAP.get(k)
                if key is not None:
                    if runner.state == "IDLE" and key == " ":
                        runner.start(clock())
                    elif runner.state == "IDLE" and key == "q":
                        runner.finish(clock())
                    else:
                        runner.key(key, clock())
            if runner.finished:
                break
    finally:
        if not runner.finished:
            runner.finish(clock())
        try:
            cap.release()
        except Exception:
            pass
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        detector.close()
        meta_out = {
            "signer": signer, "lang": lang, "session_id": session_id, "seed": seed, "camera_index": camera_index,
            "capture_fourcc": fourcc, "capture_fps_reported": float(cam_fps),
            "frame_width": frame_wh[0], "frame_height": frame_wh[1],
            "mediapipe_version": getattr(mp, "__version__", "?"), "hand_model": HAND_MODEL_PATH,
            "num_hands": 2, "face_model": "blaze_face_short_range.tflite" if face is not None else None,
            "handedness_note": "nhãn MediaPipe thô, khe 0='Left', khe 1='Right'; ảnh KHÔNG lật gương",
            "timestamp_clock": "time.perf_counter() - t0 (giây, cùng trục với events.json)",
            "video": None if no_video else "session.mp4 (khung i == chỉ số i trong frames.npz)",
        }
        meta_out.update(extra_meta or {})
        paths = save_session(session_dir, recorder, runner.events, meta_out, script)
        print(f"[Lưu] {paths['frames']}\n[Lưu] {paths['events']}")
    print(summarize(session_dir))
    return 0


# ===========================================================================
# 7. CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Thu phiên liên tục có phím gán nhãn cho Binary Spotter (ký / không ký).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Phím: SPACE bắt đầu/kết thúc ký, m ranh giới trong PAIR, x lần thử hỏng, n/Enter prompt tiếp, "
                "b làm lại prompt, q lưu & thoát."))
    p.add_argument("--signer", type=str, default=None, help="Mã người ký (vd V03). Bắt buộc khi thu.")
    p.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help=f"Thư mục gốc (mặc định {DEFAULT_OUT_DIR}).")
    p.add_argument("--camera", type=int, default=0, help="Chỉ số camera (mặc định 0).")
    p.add_argument("--lang", choices=["vi", "en"], default="vi", help="Ngôn ngữ giao diện (mặc định vi).")
    p.add_argument("--no-video", action="store_true", help="Không ghi session.mp4.")
    p.add_argument("--pairs", type=int, default=DEFAULT_PAIRS, help=f"Số prompt PAIR (mặc định {DEFAULT_PAIRS}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed xáo kịch bản.")
    p.add_argument("--negative", action="store_true",
                   help="Phiên CHỈ MẪU ÂM: làm việc thường ngày, TUYỆT ĐỐI không ký (cả phiên là nhãn âm, thư mục có hậu tố _neg).")
    p.add_argument("--minutes", type=float, default=4.0, help="Thời lượng phiên mẫu âm (phút, mặc định 4).")
    p.add_argument("--synthetic", action="store_true", help="Người ký giả lập, không webcam, vẫn ghi file.")
    p.add_argument("--dry-run", action="store_true", help="Như --synthetic nhưng KHÔNG ghi file, chỉ in thống kê.")
    p.add_argument("--summary", type=str, default=None, help="In thống kê khung/giây mỗi lớp của thư mục phiên rồi thoát.")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN_SEC, help="margin_sec cho derive_frame_labels.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summary:
        print(summarize(args.summary, args.margin))
        return 0
    if not args.signer:
        print("[!] Thiếu --signer <mã>.")
        return 2
    if args.negative and not (args.synthetic or args.dry_run):
        script = build_negative_script(args.minutes, args.seed)
        return run_session(args.signer, args.out, args.camera, args.lang, args.no_video, args.pairs, args.seed,
                           script=script, suffix="_neg", extra_meta={"negative_only": True})
    if args.synthetic or args.dry_run:
        if args.negative:
            script = build_negative_script(args.minutes, args.seed)
        else:
            vocab = cp.load_vocab_keys(cp.DEFAULT_TEMPLATES)
            script = build_script(vocab, pairs=args.pairs, seed=args.seed)
        runner, rec = simulate_session(script, fps=30.0, seed=args.seed, bad_every=7)
        arrs = rec.arrays()
        labels = derive_frame_labels(arrs["t"], runner.events, args.margin)
        print(f"[Synthetic] {len(script)} prompt, {len(rec)} khung, {arrs['t'][-1]:.0f}s")
        for name, v in label_durations(arrs["t"], labels).items():
            print(f"   {name:<12}{v['frames']:>8} khung {v['seconds']:>8.1f}s")
        if args.dry_run:
            print("[Dry-run] KHÔNG ghi file.")
            return 0
        sdir = os.path.join(args.out, args.signer, time.strftime("%Y%m%d_%H%M%S") + "_synthetic")
        meta = {"signer": args.signer, "lang": args.lang, "seed": args.seed, "synthetic": True}
        paths = save_session(sdir, rec, runner.events, meta, script)
        print(f"[Lưu] {paths['frames']}\n[Lưu] {paths['events']}")
        return 0
    return run_session(args.signer, args.out, args.camera, args.lang, args.no_video, args.pairs, args.seed)


if __name__ == "__main__":
    sys.exit(main())
