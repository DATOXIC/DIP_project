"""
THU DỮ LIỆU CÂU NỐI (CONTINUOUS PHRASE COLLECTOR) — SPEC 09 NHIỆM VỤ C
======================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

Mục đích:
  Thu LUỒNG KHUNG LIÊN TỤC (không cắt theo KineticPauseSegmenter) của người ký một câu 2-3 từ LIỀN MẠCH,
  để sau đó đo bộ nhận dạng nối từ (SPEC 09 D/E/G) bằng phát lại ngoại tuyến.

Đường xử lý từng khung GIỐNG HỆT collect_dynamic_templates.py:
  ParallelHandDetector (MediaPipe song song) -> làm mượt cổ tay 1€ (make_wrist_smoother) -> compute_palm_len ->
  góc gập ngón + pháp tuyến lòng bàn tay -> dò mặt có cache (CachedFaceDetector, mỗi 8 khung) -> LK Inertial Fallback.
  Các hàm đều IMPORT từ module gốc; chỉ phần "dán keo" (ghép trường của một khung) là viết lại ở FramePipeline,
  vì trong collector đoạn này nằm inline trong vòng lặp chính (không tách được mà không sửa collector).
  Khác biệt DUY NHẤT (theo yêu cầu): ghi cả luồng khung, kể cả khung mất tay (cờ hand_present=False).

Mỗi khung ghi thêm (camera_health.CameraHealthMonitor, gọi ở MỖI lần đọc khung, kể cả khung trùng):
  frame_is_new (bool)  : False nếu driver camera trả lại đúng khung cũ (khung trùng);
  luma_mean (float)    : độ sáng trung bình đã làm mượt EMA (thang 0..255).
Khung trùng KHÔNG bị loại (dữ liệu thô nguyên vẹn) nhưng có cờ để phát lại ngoại tuyến bỏ khung trùng và tính đúng thời gian.

Lược đồ: phrases_raw/<session_id>/<phrase_id>.json (schema_version = "phrase_raw_v1.0").

Cách dùng:
  py collect_phrases.py --make-plan                          # sinh phrases_plan.json (seed cố định) + in bảng phủ
  py collect_phrases.py --set S1                             # thu phiên S1 (40 câu + 5 clip idle)
  py collect_phrases.py --set S1 --session <id đã có>        # tiếp tục phiên dở dang
  py collect_phrases.py --set S2                             # thu phiên S2 (NIÊM PHONG: không mở xem, không chỉnh tham số)
  py collect_phrases.py --set S1 --video x.mp4 --headless --dry-run   # kiểm thử không cần webcam
"""

import os
import sys
import json
import math
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

from backup_utils import backup_path_for

# ---------------------------------------------------------------------------
# HẰNG SỐ
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "phrase_raw_v1.0"
PLAN_SCHEMA_VERSION = "phrases_plan_v1"
DEFAULT_PLAN_PATH = "phrases_plan.json"
DEFAULT_OUT_DIR = "phrases_raw"
DEFAULT_TEMPLATES = "webcam_templates.json"
DEFAULT_SEED = 20260920
COUNTDOWN_SEC = 3.0            # đếm ngược 3-2-1 trước khi ghi
GO_FLASH_SEC = 0.7             # chữ "KÝ!" hiện thêm sau khi bắt đầu ghi
IDLE_DURATION_SEC = 20.0       # clip idle: 20 s, KHÔNG ký từ nào
MAX_PHRASE_SEC = 30.0          # chốt an toàn: quên bấm SPACE thì tự kết thúc
DETECT_EVERY_N_FRAMES = 8      # giống collector (CachedFaceDetector)

SET_SIZES = {"S1": (40, 5), "S2": (30, 5)}   # (số câu, số clip idle)

SEALED_SETS = ("S2",)

# Chú giải tiếng Việt hiển thị dưới câu cần ký (chỉ để người ký dễ đọc; không dùng để chấm)
VI_GLOSS = {
    "HELLO": "Xin chào", "BYE": "Tạm biệt", "ME": "Tôi", "YOU": "Bạn", "THANK_YOU": "Cảm ơn",
    "EAT": "Ăn", "DRINK": "Uống", "WATER": "Nước", "SORRY": "Xin lỗi", "PLEASE": "Làm ơn",
    "YES": "Vâng / Có", "BAD": "Tệ / Xấu", "HUNGRY": "Đói",
}

IDLE_ACTIVITIES = [
    "Vuốt tóc", "Gãi mặt", "Đưa tay lên xuống", "Để tay nghỉ (buông thõng / đặt lên bàn)",
    "Cầm ly nước, uống nước", "Vẫy tay lơ đãng",
]

# --- Kho câu tự nhiên (ký hiệu theo thứ tự ASL/VSL rút gọn). "CORE" = tự nhiên nhất, "EXTRA" = chấp nhận được. ---
CORE_2 = """
HELLO YOU; HELLO ME; YOU HUNGRY; ME HUNGRY; PLEASE EAT; PLEASE DRINK; PLEASE WATER; WATER PLEASE; EAT PLEASE;
DRINK PLEASE; YOU EAT; ME EAT; YOU DRINK; ME DRINK; DRINK WATER; YES PLEASE; YES ME; YES YOU; THANK_YOU YOU;
THANK_YOU BYE; BYE YOU; ME SORRY; YOU SORRY; SORRY YOU; SORRY ME; YOU BAD; ME BAD; BAD WATER; HUNGRY YES;
EAT YES; YES EAT; YES HELLO; SORRY BYE; PLEASE THANK_YOU; YES THANK_YOU; HELLO BYE; BYE ME; HUNGRY YOU; HUNGRY ME
"""
EXTRA_2 = """
YOU YES; ME YES; PLEASE YES; DRINK YES; YES DRINK; ME BYE; YOU BYE; BYE THANK_YOU; SORRY YES; SORRY PLEASE;
BAD YOU; BAD ME; BAD HUNGRY; HUNGRY BAD; WATER BAD; BAD EAT; HELLO YES; HELLO PLEASE; WATER YES; WATER ME;
ME PLEASE; YOU PLEASE; HUNGRY PLEASE; EAT YOU; DRINK YOU; DRINK ME; WATER YOU; THANK_YOU ME; THANK_YOU PLEASE;
THANK_YOU YES; BYE YES; ME WATER; YOU WATER; EAT ME; BAD SORRY; YES BAD; YES SORRY; YES BYE; HELLO HUNGRY
"""
CORE_3 = """
HELLO YOU HUNGRY; HELLO ME HUNGRY; YOU DRINK WATER; ME DRINK WATER; PLEASE DRINK WATER; YOU EAT YES; ME EAT YES;
YOU HUNGRY YES; ME HUNGRY YES; ME HUNGRY PLEASE; ME EAT PLEASE; YOU EAT PLEASE; PLEASE YOU EAT; PLEASE ME EAT;
PLEASE YOU DRINK; WATER PLEASE THANK_YOU; YOU WATER PLEASE; ME WATER PLEASE; YES ME HUNGRY; YES YOU HUNGRY;
YES PLEASE THANK_YOU; THANK_YOU YOU BYE; THANK_YOU ME BYE; HELLO YOU THANK_YOU; HELLO ME THANK_YOU;
YES THANK_YOU BYE; SORRY ME BAD; ME BAD SORRY; SORRY YOU BYE; ME SORRY BYE; ME SORRY YOU; YOU SORRY ME;
HUNGRY ME EAT; HUNGRY YOU EAT; HUNGRY ME DRINK; YES ME EAT; YES YOU EAT; YES ME DRINK; YES YOU DRINK;
DRINK WATER PLEASE; EAT PLEASE YES; DRINK WATER YES; DRINK WATER THANK_YOU; PLEASE EAT THANK_YOU;
PLEASE DRINK THANK_YOU; HELLO ME SORRY; BYE YOU THANK_YOU; YOU DRINK YES; YES WATER PLEASE; WATER ME PLEASE;
SORRY ME HUNGRY; HELLO YES YOU; YES HELLO YOU; YES HELLO ME; HELLO YOU PLEASE
"""
EXTRA_3 = """
BAD ME SORRY; YOU BAD SORRY; BAD YOU SORRY; ME HUNGRY BAD; YOU EAT BAD; HUNGRY ME PLEASE; HELLO YOU SORRY;
HELLO PLEASE YOU; BYE ME SORRY; ME WATER YES; WATER YES THANK_YOU; THANK_YOU PLEASE YES; THANK_YOU YOU PLEASE;
ME PLEASE BYE; YOU BAD BYE; SORRY BAD BYE; YES ME BAD; YES BAD SORRY; HELLO HUNGRY YOU; BYE YOU SORRY;
THANK_YOU YES BYE; HUNGRY YES PLEASE; EAT YES THANK_YOU; DRINK YES THANK_YOU; YOU HUNGRY BAD; ME BAD HUNGRY;
HELLO ME BYE; THANK_YOU ME YOU; YOU THANK_YOU BYE; PLEASE YES BYE; WATER BAD YOU; BAD WATER SORRY; YES EAT WATER
"""


def _parse_pool(text: str) -> List[Tuple[str, ...]]:
    out = []
    for chunk in text.replace("\n", " ").split(";"):
        w = tuple(chunk.split())
        if w:
            out.append(w)
    return out


# ===========================================================================
# 1. KẾ HOẠCH THU (make-plan)
# ===========================================================================

def load_vocab_keys(templates_path: str = DEFAULT_TEMPLATES) -> List[str]:
    """Đọc khóa `vocabulary` của bank (sắp xếp theo alphabet để kế hoạch không phụ thuộc thứ tự trong file)."""
    with open(templates_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data.get("vocabulary", {}).keys())


def _valid_sentence(words: Sequence[str], vocab: set) -> bool:
    """Câu hợp lệ: mọi từ thuộc vocab, KHÔNG lặp cùng một từ liền nhau (giới hạn v1 của SPEC 09)."""
    if any(w not in vocab for w in words):
        return False
    return all(words[i] != words[i + 1] for i in range(len(words) - 1))


def build_pools(vocab: Sequence[str], rng: random.Random) -> Tuple[Dict[int, List[Tuple[str, ...]]], set]:
    """
    Trả (pool theo độ dài {2:[...],3:[...]}, tập câu "core"). Từ trong vocab mà kho câu tay không phủ
    (bank thêm từ mới) được bổ sung bằng câu tổng hợp ghép với từ khác (không thuộc core).
    """
    vs = set(vocab)
    pools: Dict[int, List[Tuple[str, ...]]] = {2: [], 3: []}
    core = set()
    seen = set()
    for txt, is_core in ((CORE_2, True), (EXTRA_2, False), (CORE_3, True), (EXTRA_3, False)):
        for s in _parse_pool(txt):
            if s in seen or not _valid_sentence(s, vs):
                continue
            seen.add(s)
            pools[len(s)].append(s)
            if is_core:
                core.add(s)
    covered = {w for L in pools.values() for s in L for w in s}
    for w in sorted(vs - covered):
        others = sorted(vs - {w})
        for _ in range(8):
            a, b = rng.sample(others, 2)
            for s in ((w, a), (a, w), (w, a, b), (a, w, b), (a, b, w)):
                if s not in seen and _valid_sentence(s, vs):
                    seen.add(s)
                    pools[len(s)].append(s)
    return pools, core


def _score_selection(sel: Sequence[Tuple[str, ...]], vocab: Sequence[str], min_per_word: int,
                     core: set, start_ok: set, end_ok: set) -> float:
    counts = {w: 0 for w in vocab}
    starts = {w: 0 for w in vocab}
    ends = {w: 0 for w in vocab}
    pairs = set()
    for s in sel:
        for i, w in enumerate(s):
            counts[w] += 1
            if i == 0:
                starts[w] += 1
            if i == len(s) - 1:
                ends[w] += 1
            if i:
                pairs.add((s[i - 1], w))
    mean = sum(counts.values()) / max(1, len(vocab))
    sc = 0.0
    for w in vocab:
        sc -= 100.0 * max(0, min_per_word - counts[w])
        sc -= 0.6 * (counts[w] - mean) ** 2
        if w in start_ok and starts[w] == 0:
            sc -= 20.0
        if w in end_ok and ends[w] == 0:
            sc -= 20.0
    sc += 1.0 * len(pairs)
    sc += 2.5 * sum(1 for s in sel if s in core)
    return sc


def select_sentences(pools: Dict[int, List[Tuple[str, ...]]], core: set, vocab: Sequence[str],
                     n2: int, n3: int, min_per_word: int, rng: random.Random,
                     iters: int = 12000) -> List[Tuple[str, ...]]:
    """
    Chọn n2 câu 2 từ + n3 câu 3 từ (không trùng) tối đa hóa hàm mục tiêu (đủ số lần mỗi từ, đủ vị trí đầu/cuối,
    nhiều cặp kề khác nhau, ưu tiên câu tự nhiên) bằng tôi luyện mô phỏng có seed (xác định).
    """
    start_ok = {s[0] for L in pools.values() for s in L}
    end_ok = {s[-1] for L in pools.values() for s in L}
    p2, p3 = list(pools[2]), list(pools[3])
    if len(p2) < n2 or len(p3) < n3:
        raise ValueError(f"Kho câu không đủ: cần {n2} câu 2 từ / {n3} câu 3 từ, có {len(p2)} / {len(p3)}.")
    best, best_sc = None, -1e18
    for _restart in range(6):  # nhiều lần khởi động lại, giữ phương án tốt nhất (vẫn xác định nhờ seed)
        sel = rng.sample(p2, n2) + rng.sample(p3, n3)
        cur = _score_selection(sel, vocab, min_per_word, core, start_ok, end_ok)
        if cur > best_sc:
            best, best_sc = list(sel), cur
        for it in range(iters):
            temp = 3.0 * (1.0 - it / iters) + 0.02
            i = rng.randrange(len(sel))
            pool = p2 if len(sel[i]) == 2 else p3
            cand = pool[rng.randrange(len(pool))]
            if cand in sel:
                continue
            old = sel[i]
            sel[i] = cand
            sc = _score_selection(sel, vocab, min_per_word, core, start_ok, end_ok)
            if sc >= cur or rng.random() < math.exp((sc - cur) / temp):
                cur = sc
                if cur > best_sc:
                    best, best_sc = list(sel), cur
            else:
                sel[i] = old
    return best


def _order_sentences(sel: List[Tuple[str, ...]], rng: random.Random) -> List[Tuple[str, ...]]:
    """Xáo trộn có seed, tránh hai câu liền nhau bắt đầu bằng cùng một từ (đỡ nhàm cho người ký)."""
    order = list(sel)
    for _ in range(300):
        rng.shuffle(order)
        if all(order[i][0] != order[i + 1][0] for i in range(len(order) - 1)):
            break
    return order


def _mins_met(sel, vocab, min_per_word) -> bool:
    c = {w: 0 for w in vocab}
    for s in sel:
        for w in s:
            c[w] += 1
    return all(v >= min_per_word for v in c.values())


def build_session_clips(set_name: str, pools, core, vocab: Sequence[str], seed: int, min_per_word: int,
                        n_phrases: int, n_idle: int, n_two: int, exclude: set) -> List[Dict[str, Any]]:
    """Sinh danh sách clip (câu + idle xen kẽ đều) cho một phiên."""
    rng = random.Random(seed)
    pl = {L: [s for s in pools[L] if s not in exclude] for L in (2, 3)}
    n3 = n_phrases - n_two
    sel = None
    # Nếu không đạt số lần tối thiểu mỗi từ, tăng dần số câu 3 từ (nhiều từ hơn mỗi câu) — báo lại trong plan.
    for extra in range(0, 12):
        n2_try, n3_try = n_two - extra, n3 + extra
        if n2_try < 0:
            break
        for _restart in range(3):
            cand = select_sentences(pl, core, vocab, n2_try, n3_try, min_per_word, rng)
            if _mins_met(cand, vocab, min_per_word):
                sel = cand
                n_two, n3 = n2_try, n3_try
                break
        if sel is not None:
            break
    if sel is None:
        # Không thể đạt tối thiểu: lấy phương án tốt nhất của lần cuối (báo rõ trong coverage.min_met=False)
        sel = select_sentences(pl, core, vocab, n_two, n3, min_per_word, rng)
    order = _order_sentences(sel, rng)

    acts = list(IDLE_ACTIVITIES)
    rng.shuffle(acts)
    idle_acts = [acts[i % len(acts)] for i in range(n_idle)]

    clips: List[Dict[str, Any]] = []
    idle_at = {int((k + 1) * len(order) / (n_idle + 1)) for k in range(n_idle)}
    if len(idle_at) < n_idle:  # phòng trường hợp trùng chỉ số khi kế hoạch rất ngắn
        idle_at = set(range(n_idle))
    pi = ii = 0
    for pos in range(len(order) + 1):
        if pos in idle_at and ii < n_idle:
            ii += 1
            clips.append({"phrase_id": f"{set_name}_i{ii:02d}", "set": set_name, "kind": "idle", "truth": [],
                          "text": "(KHÔNG ký từ nào)", "activity": idle_acts[ii - 1],
                          "duration_sec": IDLE_DURATION_SEC})
        if pos < len(order):
            pi += 1
            s = order[pos]
            clips.append({"phrase_id": f"{set_name}_p{pi:02d}", "set": set_name, "kind": "phrase",
                          "truth": list(s), "text": " → ".join(s), "natural": s in core})
    while ii < n_idle:  # idle còn dư (hiếm) đưa xuống cuối
        ii += 1
        clips.append({"phrase_id": f"{set_name}_i{ii:02d}", "set": set_name, "kind": "idle", "truth": [],
                      "text": "(KHÔNG ký từ nào)", "activity": idle_acts[ii - 1], "duration_sec": IDLE_DURATION_SEC})
    return clips


def coverage_table(clips: Sequence[Dict[str, Any]], vocab: Sequence[str], min_per_word: int) -> Dict[str, Any]:
    """Bảng phủ: mỗi từ xuất hiện bao nhiêu lần, ở đầu / giữa / cuối câu; số cặp kề khác nhau; tỉ lệ 2/3 từ."""
    per = {w: {"total": 0, "first": 0, "middle": 0, "last": 0} for w in vocab}
    pairs = set()
    n2 = n3 = 0
    for c in clips:
        t = c["truth"]
        if c["kind"] != "phrase":
            continue
        n2 += (len(t) == 2)
        n3 += (len(t) == 3)
        for i, w in enumerate(t):
            per[w]["total"] += 1
            per[w]["first" if i == 0 else ("last" if i == len(t) - 1 else "middle")] += 1
            if i:
                pairs.add((t[i - 1], w))
    return {
        "per_word": per, "n_two_word": n2, "n_three_word": n3,
        "distinct_adjacent_pairs": len(pairs),
        "n_core_natural": sum(1 for c in clips if c["kind"] == "phrase" and c.get("natural")),
        "min_per_word_required": min_per_word,
        "min_met": all(v["total"] >= min_per_word for v in per.values()),
    }


def format_coverage(name: str, cov: Dict[str, Any]) -> str:
    lines = [f"--- BẢNG PHỦ {name}: {cov['n_two_word']} câu 2 từ + {cov['n_three_word']} câu 3 từ, "
             f"{cov['distinct_adjacent_pairs']} cặp kề khác nhau, {cov['n_core_natural']} câu 'tự nhiên nhất' ---",
             f"{'TỪ':<10}{'TỔNG':>6}{'ĐẦU':>6}{'GIỮA':>6}{'CUỐI':>6}"]
    for w, v in cov["per_word"].items():
        lines.append(f"{w:<10}{v['total']:>6}{v['first']:>6}{v['middle']:>6}{v['last']:>6}")
    lines.append(f"Tối thiểu mỗi từ >= {cov['min_per_word_required']}: {'ĐẠT' if cov['min_met'] else 'KHÔNG ĐẠT'}")
    return "\n".join(lines)


def make_plan(vocab: Sequence[str], seed: int = DEFAULT_SEED, min_per_word_s1: int = 8,
              min_per_word_s2: int = 4) -> Dict[str, Any]:
    """
    Sinh kế hoạch hai phiên. S1: 40 câu + 5 idle (mỗi từ >= 8 lần); S2: 30 câu + 5 idle (câu KHÔNG trùng S1 khi có thể).

    GHI CHÚ MÂU THUẪN TRONG SPEC: S1 có 40 câu, ~60% câu 2 từ (24 câu) + 40% câu 3 từ (16 câu) chỉ cho 96 vị trí từ,
    nhưng 13 từ x >= 8 lần cần >= 104 vị trí. Hai yêu cầu không thể cùng đạt; ở S1 ưu tiên "mỗi từ >= 8 lần"
    (cứng) nên tỉ lệ 2/3 từ dịch về ~40%/60%. S2 (không có yêu cầu tối thiểu 8) giữ ~60%/40%.
    """
    vocab = sorted(vocab)
    rng_pool = random.Random(seed)
    pools, core = build_pools(vocab, rng_pool)
    p1, i1 = SET_SIZES["S1"]
    p2, i2 = SET_SIZES["S2"]
    s1 = build_session_clips("S1", pools, core, vocab, seed, min_per_word_s1, p1, i1,
                             n_two=int(round(p1 * 0.4)), exclude=set())
    used = {tuple(c["truth"]) for c in s1 if c["kind"] == "phrase"}
    s2 = build_session_clips("S2", pools, core, vocab, seed + 1, min_per_word_s2, p2, i2,
                             n_two=int(round(p2 * 0.6)), exclude=used)
    cov = {"S1": coverage_table(s1, vocab, min_per_word_s1), "S2": coverage_table(s2, vocab, min_per_word_s2)}
    overlap = sorted({tuple(c["truth"]) for c in s1 if c["kind"] == "phrase"} &
                     {tuple(c["truth"]) for c in s2 if c["kind"] == "phrase"})
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocabulary": vocab,
        "vi_gloss": {w: VI_GLOSS.get(w, "") for w in vocab},
        "params": {"S1": {"n_phrases": p1, "n_idle": i1, "min_per_word": min_per_word_s1},
                   "S2": {"n_phrases": p2, "n_idle": i2, "min_per_word": min_per_word_s2},
                   "no_repeat_adjacent_word": True, "idle_duration_sec": IDLE_DURATION_SEC},
        "sessions": {"S1": {"clips": s1}, "S2": {"clips": s2}},
        "coverage": cov,
        "overlap_sentences_S1_S2": [" ".join(t) for t in overlap],
    }


def write_plan(plan: Dict[str, Any], path: str) -> Optional[str]:
    """Ghi kế hoạch; nếu đã có file KHÁC nội dung thì sao lưu trước (theo PREAMBLE). Trả đường dẫn backup (nếu có)."""
    backup = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = None
        strip = lambda p: {k: v for k, v in (p or {}).items() if k != "created_at"}  # noqa: E731
        if strip(old) == strip(plan):
            print(f"[Plan] {path} đã đúng nội dung (cùng seed/từ vựng) — không ghi lại.")
            return None
        backup = backup_path_for(path, "backup_replan")
        os.replace(path, backup)
        print(f"[Plan] Đã sao lưu kế hoạch cũ -> {backup}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    return backup


def load_plan(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa có kế hoạch '{path}'. Chạy trước:  py collect_phrases.py --make-plan")
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"Kế hoạch có schema_version lạ: {plan.get('schema_version')!r}")
    return plan


# ===========================================================================
# 2. ĐỌC/GHI CLIP, TIẾP TỤC PHIÊN, ĐÁNH DẤU BỎ
# ===========================================================================

def clip_path(out_dir: str, session_id: str, phrase_id: str) -> str:
    return os.path.join(out_dir, session_id, f"{phrase_id}.json")


def clip_done(out_dir: str, session_id: str, phrase_id: str) -> bool:
    return os.path.isfile(clip_path(out_dir, session_id, phrase_id))


def write_clip(path: str, payload: Dict[str, Any]) -> str:
    """Ghi nguyên tử (file tạm rồi đổi tên) để mất điện/ngắt giữa chừng không để lại clip nửa vời."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def read_clip(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discard_clip(path: str) -> Optional[str]:
    """Đánh dấu clip cũ là BỎ: đổi tên thêm hậu tố .discarded (.discarded2, ... nếu đã có). KHÔNG xóa."""
    if not os.path.isfile(path):
        return None
    target = path + ".discarded"
    n = 1
    while os.path.exists(target):
        n += 1
        target = f"{path}.discarded{n}"
    os.replace(path, target)
    return target


def pending_indices(clips: Sequence[Dict[str, Any]], out_dir: str, session_id: str) -> List[int]:
    """Chỉ số các clip CHƯA thu (chưa có file <phrase_id>.json) — dùng để tiếp tục phiên dở dang."""
    return [i for i, c in enumerate(clips) if not clip_done(out_dir, session_id, c["phrase_id"])]


def build_payload(clip: Dict[str, Any], frames: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ghép payload clip theo lược đồ phrase_raw_v1.0 (khung + thống kê + metadata camera/fps)."""
    n = len(frames)
    dur = float(frames[-1]["timestamp"] - frames[0]["timestamp"]) if n >= 2 else 0.0
    n_hand = sum(1 for f in frames if f["hand_present"])
    n_new = sum(1 for f in frames if f["frame_is_new"])
    lumas = [f["luma_mean"] for f in frames]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "session_id": meta["session_id"],
        "set": clip.get("set", meta.get("set")),
        "phrase_id": clip["phrase_id"],
        "kind": clip["kind"],
        "truth": list(clip["truth"]),
        "text": clip.get("text", ""),
        "activity": clip.get("activity"),
        "duration_sec": round(dur, 6),
        "num_frames": n,
        "num_hand_frames": n_hand,
        "num_new_frames": n_new,
        "hand_frac": round(n_hand / n, 4) if n else 0.0,
        "new_frame_frac": round(n_new / n, 4) if n else 0.0,
        "unique_fps_est": round(n_new / dur, 3) if dur > 0 else 0.0,
        "loop_fps_est": round((n - 1) / dur, 3) if dur > 0 else 0.0,
        "luma_mean_avg": round(float(np.mean(lumas)), 3) if lumas else 0.0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update({k: v for k, v in meta.items() if k not in payload})
    payload["frames"] = frames
    return payload


# ===========================================================================
# 3. ĐƯỜNG XỬ LÝ TỪNG KHUNG (GIỐNG collector)
# ===========================================================================

def _round6(v: float) -> float:
    return round(float(v), 6)


class FramePipeline:
    """
    Phần "dán keo" của đường xử lý từng khung. Mọi phép tính lấy từ hàm import (không chép công thức):
    make_wrist_smoother, compute_palm_len, compute_joint_flexion_angles, compute_palm_normal_vector,
    CachedFaceDetector, LKInertialFallback. Thứ tự các bước và làm tròn (6 chữ số) y hệt vòng lặp chính của
    collect_dynamic_templates.run_dynamic_collector.
    """

    def __init__(self, P: Dict[str, Any]):
        self.P = P
        self.wrist_smoother = P["make_wrist_smoother"]()
        self.lk_fallback = P["LKInertialFallback"]()
        try:
            self.face_detector = P["CachedFaceDetector"](detect_every_n_frames=DETECT_EVERY_N_FRAMES)
        except Exception as e:  # giống collector: không có bộ dò mặt thì face_box = None
            print(f"[!] KHÔNG nạp được bộ dò mặt ({e}) -> face_box = null.")
            self.face_detector = None
        self.hand_lost_frames = 0
        self.cached_palm_len = 50.0
        self.cached_handedness = "Right"
        self.prev_wrist: Optional[Tuple[float, float]] = None
        self.last_v_norm = 0.0

    def process(self, meta_item: Dict[str, Any], det_result: Any) -> Dict[str, Any]:
        import cv2
        P = self.P
        frame = meta_item["frame"]
        curr_time = meta_item["capture_ts"]
        rgb_frame = meta_item.get("rgb_frame")
        if rgb_frame is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_f, w_f = frame.shape[:2]

        if self.face_detector is not None:
            face_box = self.face_detector.detect(rgb_frame, out_size=(w_f, h_f))
        else:
            face_box = None
        face_json = [round(float(v), 3) for v in face_box] if face_box is not None else None

        rec: Dict[str, Any] = {
            "timestamp": round(float(curr_time), 6),
            "hand_present": False,
            "lk_fallback": False,
            "landmarks_norm": None, "landmarks_px": None, "palm_len": None,
            "wrist_raw": None, "wrist_smoothed": None,
            "flexion_angles": None, "palm_normal": None,
            "hand_shape": "Unknown", "hand_shape_conf": None,
            "handedness": None, "face_box": face_json,
            "frame_is_new": bool(meta_item.get("is_new", True)),
            "luma_mean": round(float(meta_item.get("luma_mean", 0.0)), 3),
        }

        if det_result is not None and det_result.hand_landmarks and len(det_result.hand_landmarks) > 0:
            self.hand_lost_frames = 0
            landmarks = det_result.hand_landmarks[0]
            handedness_label = det_result.handedness[0][0].category_name if det_result.handedness else "Right"
            lm0, lm5, lm9, lm17 = landmarks[0], landmarks[5], landmarks[9], landmarks[17]
            raw_wrist_x = lm0.x * w_f
            raw_wrist_y = lm0.y * h_f
            wrist_x, wrist_y = self.wrist_smoother.update(raw_wrist_x, raw_wrist_y, timestamp=curr_time)
            palm_len = P["compute_palm_len"](lm0, lm5, lm9, lm17, w_f, h_f)
            self.cached_palm_len = palm_len
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.lk_fallback.update_reference(gray_frame, landmarks)
            lm_3d = np.array([[lm.x * w_f, lm.y * h_f, lm.z * w_f] for lm in landmarks], dtype=np.float32)
            angles = P["compute_joint_flexion_angles"](lm_3d)
            normal = P["compute_palm_normal_vector"](lm_3d, handedness=handedness_label)
            self.cached_handedness = handedness_label
            rec.update({
                "hand_present": True,
                "landmarks_norm": [[_round6(lm.x), _round6(lm.y), _round6(lm.z)] for lm in landmarks],
                "landmarks_px": [[_round6(lm.x * w_f), _round6(lm.y * h_f), _round6(lm.z * w_f)] for lm in landmarks],
                "palm_len": _round6(palm_len),
                "wrist_raw": [_round6(raw_wrist_x), _round6(raw_wrist_y)],
                "wrist_smoothed": [_round6(wrist_x), _round6(wrist_y)],
                "flexion_angles": [_round6(a) for a in angles],
                "palm_normal": [_round6(n) for n in normal],
                "handedness": handedness_label,
            })
            cur_w = (wrist_x, wrist_y)
        else:
            cur_w = None
            self.hand_lost_frames += 1
            if 1 <= self.hand_lost_frames <= 2:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                predicted = self.lk_fallback.predict_wrist(gray_frame)
                if predicted is not None:
                    wrist_x, wrist_y = self.wrist_smoother.update(*predicted, timestamp=curr_time)
                    # Khung mất tay nhưng LK dự đoán được cổ tay: vẫn hand_present=False (không có landmark thật).
                    rec.update({"lk_fallback": True, "palm_len": _round6(self.cached_palm_len),
                                "wrist_smoothed": [_round6(wrist_x), _round6(wrist_y)],
                                "handedness": self.cached_handedness})
                    cur_w = (wrist_x, wrist_y)
            if self.hand_lost_frames >= 4:
                self.wrist_smoother.reset()
                self.lk_fallback.reset()

        # vận tốc chuẩn hóa thô (chỉ để báo "tay đang chuyển động" cho CameraHealthMonitor, KHÔNG ghi vào file)
        if cur_w is not None and self.prev_wrist is not None and rec["frame_is_new"]:
            self.last_v_norm = math.hypot(cur_w[0] - self.prev_wrist[0], cur_w[1] - self.prev_wrist[1]) / \
                max(self.cached_palm_len, 15.0)
        elif not rec["frame_is_new"]:
            pass  # khung trùng: giữ v cũ (đừng để tụt về 0)
        else:
            self.last_v_norm = 0.0
        self.prev_wrist = cur_w
        return rec


def load_pipeline_modules() -> Dict[str, Any]:
    """IMPORT chậm (torch/mediapipe nặng) các hàm dùng chung; --help và --make-plan không cần tới."""
    from dynamic_features import CachedFaceDetector, compute_palm_len
    from predict_sentence_webcam import CONFIG, compute_joint_flexion_angles, compute_palm_normal_vector
    from pipeline_common import ParallelHandDetector, decode_fourcc, open_camera, make_wrist_smoother, LKInertialFallback
    from camera_health import CameraHealthMonitor
    return {
        "CachedFaceDetector": CachedFaceDetector, "compute_palm_len": compute_palm_len, "CONFIG": CONFIG,
        "compute_joint_flexion_angles": compute_joint_flexion_angles,
        "compute_palm_normal_vector": compute_palm_normal_vector,
        "ParallelHandDetector": ParallelHandDetector, "decode_fourcc": decode_fourcc, "open_camera": open_camera,
        "make_wrist_smoother": make_wrist_smoother, "LKInertialFallback": LKInertialFallback,
        "CameraHealthMonitor": CameraHealthMonitor,
    }


# ===========================================================================
# 4. GIAO DIỆN
# ===========================================================================

_WARN_TEXT = {
    "too_bright": "Quá chói / ngược sáng — giảm sáng, tránh cửa sổ sau lưng",
    "low_light": "Thiếu sáng — bật đèn như lúc demo",
    "slow_camera": "Camera ra ít khung mới — thêm ánh sáng",
}


class Hud:
    """Vẽ HUD lên khung webcam. Dùng ui_render.TextRenderer (chữ Unicode) nếu import được, không thì chữ không dấu."""

    def __init__(self):
        self.tr = None
        self.strip = None
        try:
            from ui_render import TextRenderer, strip_diacritics
            self.tr = TextRenderer()
            self.strip = strip_diacritics
        except Exception as e:
            print(f"[HUD] Không dùng được ui_render.TextRenderer ({e}) -> chữ không dấu.")

    def _txt(self, frame, text, x, y, size, color, bold=False, anchor="l"):
        import cv2
        if self.tr is not None:
            self.tr.queue(text, x, y, size, color, bold=bold, anchor=anchor)
            return
        plain = _strip_vn(text)
        scale = size / 34.0
        thick = 2 if bold else 1
        (tw, th), _ = cv2.getTextSize(plain, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if anchor == "c":
            x -= tw // 2
        elif anchor == "r":
            x -= tw
        cv2.putText(frame, plain, (int(x), int(y + th)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _fit(self, text, size, max_w, bold=False):
        if self.tr is not None:
            return self.tr.fit_size(text, size, max_w, bold=bold)
        return size

    @staticmethod
    def _band(frame, y0, y1, alpha=0.62):
        import cv2
        roi = frame[y0:y1]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, alpha, roi, 1 - alpha, 0, roi)

    def render(self, frame: np.ndarray, st: Dict[str, Any]) -> np.ndarray:
        import cv2
        h, w = frame.shape[:2]
        # khung xương thô (chấm) để người ký thấy hệ thống đang thấy tay
        lm = st.get("landmarks_px")
        if lm:
            col = (0, 220, 0) if st["state"] != "RECORDING" else (0, 0, 255)
            for p in lm:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, col, -1, cv2.LINE_AA)
        elif st["state"] in ("WAIT", "COUNTDOWN", "RECORDING"):
            cv2.putText(frame, "", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 0), 1)

        # dải trên: câu cần ký (cỡ lớn)
        self._band(frame, 0, 158)
        head = f"{st['set']}  ·  clip {st['pos']}/{st['total']}  ·  đã thu {st['n_done']}"
        self._txt(frame, head, 12, 6, 17, (200, 200, 200))
        if st.get("sealed"):
            self._txt(frame, "PHIÊN NIÊM PHONG", w - 12, 6, 17, (0, 165, 255), bold=True, anchor="r")
        text = st["text"]
        size = self._fit(text, 54, w - 24, bold=True)
        self._txt(frame, text, w // 2, 32, size, (255, 255, 255), bold=True, anchor="c")
        sub = st.get("sub", "")
        if sub:
            self._txt(frame, sub, w // 2, 100, self._fit(sub, 26, w - 24), (0, 220, 255), anchor="c")
        hint = st.get("hint", "")
        if hint:
            self._txt(frame, hint, w // 2, 134, self._fit(hint, 17, w - 24), (180, 180, 180), anchor="c")

        # dải dưới: trạng thái + phím
        self._band(frame, h - 78, h)
        if st.get("warn"):
            self._txt(frame, "(!) " + _WARN_TEXT.get(st["warn"], st["warn"]), 12, h - 96, 18, (0, 165, 255), bold=True)
        state = st["state"]
        if state == "RECORDING":
            cv2.circle(frame, (24, h - 58), 9, (0, 0, 255), -1, cv2.LINE_AA)
            msg = f"ĐANG GHI  {st['elapsed']:4.1f}s"
            if st["kind"] == "phrase":
                msg += "  —  xong: đứng yên ~1 s, bấm SPACE"
            self._txt(frame, msg, 42, h - 70, self._fit(msg, 22, w - 54, True), (60, 60, 255), bold=True)
        elif state == "COUNTDOWN":
            self._txt(frame, "Chuẩn bị — đứng yên, tay buông tự nhiên", 12, h - 70, 22, (0, 255, 255), bold=True)
        elif state == "WAIT":
            self._txt(frame, "Bấm SPACE để bắt đầu (đếm ngược 3-2-1)", 12, h - 70, 22, (0, 255, 0), bold=True)
        elif state == "DONE":
            self._txt(frame, "HOÀN TẤT PHIÊN — bấm q để thoát", 12, h - 70, 22, (0, 255, 0), bold=True)
        keys = "SPACE: bắt đầu/kết thúc  ·  f: lật gương  ·  r: thu lại  ·  q: thoát"
        self._txt(frame, keys, 12, h - 34, 16, (170, 170, 170))
        st_line = f"tay:{'có' if st['hand_present'] else 'không'}   khung mới:{st['new_pct']:.0f}%   sáng:{st['luma']:.0f}"
        self._txt(frame, st_line, w - 12, h - 34, 16, (170, 170, 170), anchor="r")
        if st.get("flash") and time.time() < st.get("flash_until", 0):
            self._band(frame, h // 2 - 26, h // 2 + 26, 0.75)
            self._txt(frame, st["flash"], w // 2, h // 2 - 16, self._fit(st["flash"], 30, w - 24, True),
                      st.get("flash_color", (0, 255, 0)), bold=True, anchor="c")
        # số đếm ngược khổng lồ
        if state == "COUNTDOWN" and st.get("count") is not None:
            self._txt(frame, str(st["count"]), w // 2, h // 2 - 100, 200, (0, 255, 255), bold=True, anchor="c")
        elif state == "RECORDING" and st["elapsed"] < GO_FLASH_SEC:
            self._txt(frame, "KÝ!", w // 2, h // 2 - 70, 140, (0, 255, 0), bold=True, anchor="c")
        if self.tr is not None:
            self.tr.flush(frame)
        return frame


_VN_MAP = str.maketrans(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD")


def _strip_vn(text: str) -> str:
    return (text.translate(_VN_MAP).replace("→", "->").replace("·", "|").replace("—", "-").replace("⚠", "!")
            .replace("…", "..."))


def instructions_text(set_name: str) -> str:
    lines = [
        "=" * 78,
        "  THU CÂU NỐI (SPEC 09-C) — HƯỚNG DẪN CHO NGƯỜI KÝ",
        "  * Ký LIỀN MẠCH, TỰ NHIÊN như nói chuyện. KHÔNG dừng giữa các từ, KHÔNG dừng chờ máy.",
        "  * Chỉ đứng yên ~1 giây TRƯỚC câu (tay buông tự nhiên) và ~1 giây SAU cả câu, rồi mới bấm SPACE.",
        "  * Đèn sáng như lúc demo; tránh ngược sáng (cửa sổ sau lưng) và tránh quá chói.",
        "  * Phím: SPACE bắt đầu (đếm ngược 3-2-1) / kết thúc clip;  f lật gương camera;",
        "          r thu lại clip vừa rồi (clip cũ đổi hậu tố .discarded);",
        "          q thoát (phiên dở dang tiếp tục được bằng --session <id>).",
        "  * Clip IDLE (20 s): KHÔNG ký từ nào; làm đúng hoạt động được ghi trên màn hình (vuốt tóc, gãi mặt, ...).",
    ]
    if set_name in SEALED_SETS:
        lines += [
            "-" * 78,
            f"  !!! CẢNH BÁO: {set_name} LÀ TẬP NIÊM PHONG (kiểm tra cuối) !!!",
            "  KHÔNG mở xem dữ liệu, KHÔNG dùng để chỉnh tham số; chỉ chạy đúng MỘT lần ở bước G.3 của SPEC 09.",
            "  Chương trình này KHÔNG có chức năng xem lại/phát lại " + set_name + ".",
        ]
    lines.append("=" * 78)
    return "\n".join(lines)


# ===========================================================================
# 5. VÒNG THU CHÍNH
# ===========================================================================

def _next_pending(clips, out_dir, session_id, start=0) -> Optional[int]:
    n = len(clips)
    for k in range(n):
        i = (start + k) % n
        if not clip_done(out_dir, session_id, clips[i]["phrase_id"]):
            return i
    return None


def run_collection(
    set_name: str = "S1",
    session_id: Optional[str] = None,
    plan_path: str = DEFAULT_PLAN_PATH,
    out_dir: str = DEFAULT_OUT_DIR,
    camera_index: Optional[int] = None,
    video_path: Optional[str] = None,
    dry_run: bool = False,
    max_frames: Optional[int] = None,
    headless: bool = False,
    detector_workers: Optional[int] = None,
    detect_scale: Optional[float] = None,
    truth_override: Optional[str] = None,
    phrase_id: Optional[str] = None,
    video_t0: Optional[float] = None,
    clock: Optional[Any] = None,
    pipeline_modules: Optional[Dict[str, Any]] = None,
    capture: Optional[Any] = None,
    flip_cam: bool = False,
) -> Dict[str, Any]:
    """
    Chạy phiên thu. Trả về {"session_id", "saved": [đường dẫn], "clips_recorded": n, "payloads": [...] (chỉ khi dry_run)}.
    `clock(seq) -> giây` (tùy chọn) thay đồng hồ dấu thời gian khung (dùng cho kiểm thử so khớp collector).
    `capture` (tùy chọn, chỉ để kiểm thử): đối tượng giống cv2.VideoCapture thay cho camera (chạy đủ máy trạng thái phím).
    """
    import cv2
    P = pipeline_modules or load_pipeline_modules()
    CONFIG = P["CONFIG"]
    if camera_index is None:
        camera_index = int(CONFIG.get("camera_source", 0))
    if detect_scale is None:
        detect_scale = float(CONFIG.get("detect_scale", 1.0))
    if detector_workers is None:
        detector_workers = CONFIG.get("detector_workers", None)
    auto_mode = headless or (video_path is not None)   # không có phím: bắt đầu clip ngay từ khung đầu

    # ---- danh sách clip ----
    clips: List[Dict[str, Any]]
    if video_path is not None and not os.path.exists(plan_path):
        clips = [{"phrase_id": phrase_id or f"{set_name}_video_test", "set": set_name, "kind": "phrase",
                  "truth": (truth_override or "").split(), "text": truth_override or "(video thử)"}]
    else:
        plan = load_plan(plan_path)
        if set_name not in plan["sessions"]:
            raise ValueError(f"Kế hoạch không có phiên '{set_name}' (có: {list(plan['sessions'])}).")
        clips = plan["sessions"][set_name]["clips"]
    if not session_id:
        session_id = f"{set_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    sealed = set_name in SEALED_SETS
    print(instructions_text(set_name))
    print(f"  Phiên (session_id) : {session_id}\n  Thư mục ghi        : {os.path.join(out_dir, session_id)}"
          f"{'  [DRY-RUN: KHÔNG ghi file]' if dry_run else ''}")

    idx = _next_pending(clips, out_dir, session_id) if not dry_run else 0
    n_done0 = sum(1 for c in clips if clip_done(out_dir, session_id, c["phrase_id"]))
    if n_done0:
        print(f"  Tiếp tục phiên dở dang: đã có {n_done0}/{len(clips)} clip, bỏ qua các clip đó.")
    if idx is None:
        print("  Phiên này đã đủ clip — không còn gì để thu.")
        return {"session_id": session_id, "saved": [], "clips_recorded": 0, "payloads": []}
    if video_path is not None and phrase_id:
        matches = [i for i, c in enumerate(clips) if c["phrase_id"] == phrase_id]
        if matches:
            idx = matches[0]
    if sealed:
        print(f"\n  !!! {set_name} NIÊM PHONG: không mở xem, không dùng để chỉnh tham số. !!!\n")

    # ---- khởi tạo bộ dò + nguồn hình ----
    model_asset_path = "hand_landmarker.task"
    if not os.path.exists(model_asset_path):
        raise FileNotFoundError(f"Thiếu {model_asset_path} (chạy collect_dynamic_templates.py một lần để tự tải).")
    detector_pool = P["ParallelHandDetector"](model_path=model_asset_path, workers=detector_workers, num_hands=1)
    pipe = FramePipeline(P)
    monitor = P["CameraHealthMonitor"](cfg=(CONFIG.get("camera_health") or None), warnings_enabled=(video_path is None))

    cap = None
    actual_fourcc, actual_fps = "UNKNOWN", 30.0
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            detector_pool.close()
            raise RuntimeError(f"KHÔNG THỂ mở video: {video_path}")
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        actual_fourcc = P["decode_fourcc"](cap.get(cv2.CAP_PROP_FOURCC))
        t0 = video_t0 if video_t0 is not None else time.time()
        if clock is None:
            clock = lambda seq: t0 + seq / actual_fps      # noqa: E731  (giờ video, không phải giờ tường)
        print(f"[Video] {video_path} (FPS {actual_fps:.1f}, {actual_fourcc}) — dấu thời gian = giờ video.")
        source = f"video:{os.path.basename(video_path)}"
    elif capture is not None:
        cap = capture
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if clock is None:
            clock = lambda seq: time.time()                # noqa: E731
        source = "camera"
    else:
        cap = P["open_camera"](camera_index, prefer_mjpg=True)
        if cap is None or not cap.isOpened():
            detector_pool.close()
            raise RuntimeError(f"KHÔNG THỂ mở camera [{camera_index}]")
        actual_fourcc = P["decode_fourcc"](cap.get(cv2.CAP_PROP_FOURCC))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if clock is None:
            clock = lambda seq: time.time()                # noqa: E731
        source = "camera"

    hud = Hud() if not headless else None
    window = "Thu Cau Noi (VSL Phrase Collector)"

    # ---- trạng thái ----
    state = "RECORDING" if auto_mode else "WAIT"
    rec_start_ts = 0.0
    stop_after_ts: Optional[float] = None
    rec_frames: List[Dict[str, Any]] = []
    saved: List[str] = []
    payloads: List[Dict[str, Any]] = []
    last_saved_idx: Optional[int] = None
    if not dry_run:
        done_idx = [i for i, c in enumerate(clips) if clip_done(out_dir, session_id, c["phrase_id"])]
        last_saved_idx = done_idx[-1] if done_idx else None
    flash, flash_color, flash_until = "", (0, 255, 0), 0.0
    seq = 0
    processed = 0
    from collections import deque
    ready_q: deque = deque()
    exhausted = False
    hand_moving = False
    last_rec: Dict[str, Any] = {}
    new_hist: deque = deque(maxlen=60)
    quit_now = False

    def _meta_for_clip(w_f, h_f):
        return {
            "session_id": session_id, "set": set_name, "camera_index": (-1 if video_path else int(camera_index)),
            "source": source, "frame_width": int(w_f), "frame_height": int(h_f), "flip_cam": bool(flip_cam),
            "capture_fourcc": actual_fourcc, "capture_fps_reported": round(float(actual_fps), 2),
            "detect_scale": float(detect_scale), "detector_workers": int(detector_pool.num_workers),
            "pipeline": {"wrist_smoother": "make_wrist_smoother() f_c_min=1.0,beta=0.007",
                         "face_detect_every_n_frames": DETECT_EVERY_N_FRAMES, "enable_static_shape_cnn": False,
                         "timestamp_clock": "video_time" if video_path else "time.time()",
                         "note": "frame_is_new/luma_mean từ camera_health.CameraHealthMonitor; khung trùng KHÔNG bị loại"},
        }

    def finalize(reason: str, wh: Tuple[int, int]) -> None:
        nonlocal state, idx, last_saved_idx, flash, flash_color, flash_until, rec_frames, stop_after_ts, quit_now
        clip = clips[idx]
        frames = rec_frames
        rec_frames = []
        stop_after_ts = None
        if len(frames) < 2:
            flash, flash_color, flash_until = "Clip quá ngắn — không lưu", (0, 0, 255), time.time() + 2.5
            print(f"[Thu] BỎ clip {clip['phrase_id']}: chỉ {len(frames)} khung ({reason}).")
            state = "WAIT"
            return
        payload = build_payload(clip, frames, _meta_for_clip(*wh))
        note = ""
        if clip["kind"] == "phrase" and payload["hand_frac"] < 0.5:
            note = f" CẢNH BÁO: chỉ {payload['hand_frac']*100:.0f}% khung thấy tay — nên bấm r thu lại."
        if dry_run:
            payloads.append(payload)
            path = "(dry-run: không ghi)"
        else:
            path = write_clip(clip_path(out_dir, session_id, clip["phrase_id"]), payload)
            saved.append(path)
            last_saved_idx = idx
        print(f"[Thu] {clip['phrase_id']}: {payload['num_frames']} khung, {payload['duration_sec']:.2f}s, "
              f"tay {payload['hand_frac']*100:.0f}%, khung mới {payload['new_frame_frac']*100:.0f}% "
              f"({payload['unique_fps_est']:.1f}/s), sáng {payload['luma_mean_avg']:.0f} -> {path}.{note}")
        flash = ("Đã lưu " + clip["phrase_id"]) + (" — tay ít, nên thu lại" if note else "")
        flash_color = (0, 165, 255) if note else (0, 255, 0)
        flash_until = time.time() + 2.0
        if auto_mode:
            state = "DONE"
            quit_now = True
            return
        nxt = _next_pending(clips, out_dir, session_id, idx + 1)
        if dry_run:
            nxt = idx + 1 if idx + 1 < len(clips) else None
        if nxt is None:
            state = "DONE"
        else:
            idx = nxt
            state = "WAIT"

    print("\n  Bắt đầu. " + ("(chế độ tự động: clip = toàn bộ nguồn video)" if auto_mode else "Bấm SPACE để bắt đầu clip đầu tiên.") + "\n")
    frame_wh = (640, 480)
    while True:
        if not exhausted:
            ret, frame = cap.read()
            if not ret:
                exhausted = True
                ready_q.extend(detector_pool.flush())
            else:
                seq += 1
                ts = float(clock(seq))
                if flip_cam:
                    frame = cv2.flip(frame, 1)
                # CameraHealth gọi MỖI lần đọc (kể cả khung trùng), đúng thứ tự đọc
                monitor.update(frame, ts, hand_moving)
                meta = {"capture_ts": ts, "frame": frame, "is_new": bool(monitor.is_new),
                        "luma_mean": float(monitor.luma_mean), "warn": monitor.warning}
                if detect_scale < 0.999:
                    det_frame = cv2.resize(frame, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
                else:
                    det_frame = frame
                meta["rgb_frame"] = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
                detector_pool.submit(seq, meta["rgb_frame"], meta)
                ready_q.extend(detector_pool.poll(block_oldest=(detector_pool.in_flight >= detector_pool.max_in_flight)))

        if not ready_q:
            if exhausted:
                break
            if not headless:
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
            continue

        seq_out, meta_item, det_result = ready_q.popleft()
        processed += 1
        frame = meta_item["frame"]
        frame_wh = (frame.shape[1], frame.shape[0])
        ts = meta_item["capture_ts"]
        rec = pipe.process(meta_item, det_result)
        last_rec = rec
        hand_moving = pipe.last_v_norm >= float((CONFIG.get("camera_health") or {}).get("hand_moving_v", 0.03))
        new_hist.append(1.0 if rec["frame_is_new"] else 0.0)

        # ---- máy trạng thái clip ----
        if state == "COUNTDOWN" and ts >= rec_start_ts:
            state = "RECORDING"
            rec_frames = []
            stop_after_ts = None
        if state == "RECORDING":
            if stop_after_ts is not None and ts > stop_after_ts:
                finalize("người dùng kết thúc", frame_wh)
            else:
                rec["frame_idx"] = len(rec_frames)
                rec_frames.append(rec)
                clip = clips[idx]
                t_el = ts - rec_frames[0]["timestamp"]
                if clip["kind"] == "idle" and t_el >= float(clip.get("duration_sec", IDLE_DURATION_SEC)):
                    finalize("hết 20 s idle", frame_wh)
                elif t_el >= MAX_PHRASE_SEC and clip["kind"] == "phrase":
                    finalize("quá 30 s", frame_wh)
                elif max_frames is not None and len(rec_frames) >= max_frames:
                    finalize("đủ --max-frames", frame_wh)
        if quit_now:
            break

        # ---- hiển thị + phím ----
        key = 255
        if not headless:
            now_w = time.time()
            clip = clips[idx]
            count = None
            if state == "COUNTDOWN":
                count = max(1, int(math.ceil(rec_start_ts - now_w)))
            st = {
                "state": state, "set": set_name, "pos": idx + 1, "total": len(clips),
                "n_done": sum(1 for c in clips if clip_done(out_dir, session_id, c["phrase_id"])) if not dry_run else len(saved) + len(payloads),
                "kind": clip["kind"], "sealed": sealed,
                "text": clip["text"] if clip["kind"] == "phrase" else "IDLE — KHÔNG KÝ TỪ NÀO",
                "sub": (" · ".join(VI_GLOSS.get(w, w) for w in clip["truth"]) if clip["kind"] == "phrase"
                        else clip.get("activity", "")),
                "hint": ("Ký liền mạch, không dừng giữa các từ" if clip["kind"] == "phrase"
                         else f"Làm tự nhiên trong {IDLE_DURATION_SEC:.0f} s (tự kết thúc)"),
                "elapsed": (ts - rec_frames[0]["timestamp"]) if (state == "RECORDING" and rec_frames) else 0.0,
                "count": count, "landmarks_px": rec["landmarks_px"], "hand_present": rec["hand_present"],
                "new_pct": 100.0 * (sum(new_hist) / max(1, len(new_hist))), "luma": rec["luma_mean"],
                "warn": meta_item.get("warn") if video_path is None else None,
                "flash": flash, "flash_color": flash_color, "flash_until": flash_until,
            }
            disp = hud.render(frame.copy(), st)
            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f") and not auto_mode:
            flip_cam = not flip_cam
            status_txt = "BẬT" if flip_cam else "TẮT"
            flash, flash_color, flash_until = f"Lật gương: {status_txt}", (0, 255, 255), time.time() + 1.5
            print(f"[Thu] Lật gương camera: {status_txt}.")
        elif key == 32 and not auto_mode:      # SPACE
            if state == "WAIT":
                rec_start_ts = time.time() + COUNTDOWN_SEC
                state = "COUNTDOWN"
            elif state == "RECORDING" and stop_after_ts is None:
                stop_after_ts = time.time()
        elif key == ord("r") and not auto_mode:
            if state in ("COUNTDOWN", "RECORDING"):
                state = "WAIT"
                rec_frames = []
                stop_after_ts = None
                flash, flash_color, flash_until = "Đã hủy clip đang ghi — bấm SPACE để ghi lại", (0, 255, 255), time.time() + 2.5
                print(f"[Thu] Hủy clip đang ghi {clips[idx]['phrase_id']}.")
            elif state in ("WAIT", "DONE") and last_saved_idx is not None:
                old = clip_path(out_dir, session_id, clips[last_saved_idx]["phrase_id"])
                tgt = discard_clip(old)
                if tgt:
                    print(f"[Thu] Đánh dấu BỎ: {old} -> {tgt}")
                    idx = last_saved_idx
                    state = "WAIT"
                    flash, flash_color, flash_until = f"Bỏ clip {clips[idx]['phrase_id']} — thu lại", (0, 255, 255), time.time() + 2.5
                    remaining = [i for i, c in enumerate(clips[:idx]) if clip_done(out_dir, session_id, c["phrase_id"])]
                    last_saved_idx = remaining[-1] if remaining else None
                else:
                    flash, flash_color, flash_until = "Không tìm thấy clip để thu lại", (0, 0, 255), time.time() + 2.0
            else:
                flash, flash_color, flash_until = "Chưa có clip nào để thu lại", (0, 0, 255), time.time() + 2.0
        if max_frames is not None and processed >= max_frames and auto_mode:
            if state == "RECORDING":
                finalize("đủ --max-frames", frame_wh)
            break

    # video cạn / thoát khi đang ghi (chế độ tự động): chốt clip đang dở
    if auto_mode and state == "RECORDING" and rec_frames:
        finalize("nguồn hết khung", frame_wh)

    try:
        cap.release()
    except Exception:
        pass
    if not headless:
        cv2.destroyAllWindows()
    detector_pool.close()
    print(f"-> Đã đóng. Đã ghi {len(saved)} clip mới vào {os.path.join(out_dir, session_id)}.")
    return {"session_id": session_id, "saved": saved, "clips_recorded": len(saved) + len(payloads), "payloads": payloads}


# ===========================================================================
# 6. CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Thu luồng khung câu nối (SPEC 09-C). Ký liền mạch; mỗi clip lưu phrases_raw/<session_id>/<phrase_id>.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Thứ tự làm việc: (1) --make-plan  (2) --set S1  (3) [sau khi chỉnh xong tham số trên S1] --set S2.\n"
            "S2 là tập NIÊM PHONG: không mở xem, không dùng chỉnh tham số.\n"
            "Phím: SPACE bắt đầu(đếm ngược 3-2-1)/kết thúc clip, r thu lại clip vừa rồi (.discarded), q thoát."))
    p.add_argument("--make-plan", action="store_true", help="Sinh phrases_plan.json (seed cố định) + in bảng phủ rồi thoát.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed sinh kế hoạch (mặc định {DEFAULT_SEED}).")
    p.add_argument("--min-per-word", type=int, default=8, help="Số lần tối thiểu mỗi từ trong S1 (mặc định 8).")
    p.add_argument("--templates", type=str, default=DEFAULT_TEMPLATES,
                   help="Bank để lấy khóa 'vocabulary' khi tạo kế hoạch (mặc định webcam_templates.json).")
    p.add_argument("--plan", type=str, default=DEFAULT_PLAN_PATH, help="Đường dẫn kế hoạch (mặc định phrases_plan.json).")
    p.add_argument("--set", dest="set_name", choices=["S1", "S2"], default="S1",
                   help="Phiên cần thu: S1 (40 câu + 5 idle, chỉnh tham số) hoặc S2 (30 câu + 5 idle, NIÊM PHONG).")
    p.add_argument("--session", type=str, default=None,
                   help="session_id; đặt lại đúng id cũ để TIẾP TỤC phiên dở dang (bỏ qua clip đã thu). Mặc định <SET>_<thời gian>.")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Thư mục gốc lưu clip (mặc định phrases_raw).")
    p.add_argument("--cam", type=int, default=None, help="Chỉ số camera. Mặc định lấy CONFIG['camera_source'] (predict_sentence_webcam.py).")
    p.add_argument("--video", type=str, default=None,
                   help="Chạy trên file video thay webcam để kiểm thử: cả video = MỘT clip mô phỏng (không cần phím).")
    p.add_argument("--truth", type=str, default=None, help="Với --video và chưa có kế hoạch: đáp án mô phỏng, vd 'HELLO YOU'.")
    p.add_argument("--phrase-id", type=str, default=None, help="Với --video: chọn clip của kế hoạch để gán (mặc định clip chưa thu đầu tiên).")
    p.add_argument("--dry-run", action="store_true", help="Chạy đủ đường xử lý nhưng KHÔNG ghi file nào (kể cả .discarded).")
    p.add_argument("--max-frames", type=int, default=None, help="Dừng sau số khung này (kiểm thử).")
    p.add_argument("--headless", action="store_true", help="Không mở cửa sổ (kiểm thử): clip tự bắt đầu ngay, dừng khi đủ --max-frames/hết video.")
    p.add_argument("--detect-scale", type=float, default=None, help="Thu nhỏ khung trước khi dò (mặc định lấy CONFIG, 1.0).")
    p.add_argument("--detector-workers", type=int, default=None, help="Số luồng MediaPipe (mặc định tự chọn như collector).")
    p.add_argument("--flip", action="store_true", help="Bật lật gương màn hình camera ngay khi khởi động (có thể bấm 'f' để bật/tắt).")
    return p


def cmd_make_plan(args) -> int:
    vocab = load_vocab_keys(args.templates)
    print(f"[Plan] Từ vựng ({len(vocab)}): {' '.join(vocab)}  (từ {args.templates})")
    plan = make_plan(vocab, seed=args.seed, min_per_word_s1=args.min_per_word)
    write_plan(plan, args.plan)
    print(f"[Plan] Đã ghi {args.plan} (seed {args.seed}).\n")
    for name in ("S1", "S2"):
        print(format_coverage(name, plan["coverage"][name]))
        print()
    ov = plan["overlap_sentences_S1_S2"]
    print(f"Câu trùng giữa S1 và S2: {len(ov)} {ov if ov else ''}")
    for name in ("S1", "S2"):
        c = plan["sessions"][name]["clips"]
        print(f"\n[{name}] thứ tự clip:")
        print("  " + "\n  ".join(f"{x['phrase_id']}: {x['text']}" + (f"  [{x['activity']}]" if x['kind'] == 'idle' else "") for x in c))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.make_plan:
        return cmd_make_plan(args)
    if args.headless and args.video is None and args.max_frames is None:
        print("[!] --headless với camera cần --max-frames (không có phím để kết thúc clip).")
        return 2
    try:
        res = run_collection(
            set_name=args.set_name, session_id=args.session, plan_path=args.plan, out_dir=args.out_dir,
            camera_index=args.cam, video_path=args.video, dry_run=args.dry_run, max_frames=args.max_frames,
            headless=args.headless, detector_workers=args.detector_workers, detect_scale=args.detect_scale,
            truth_override=args.truth, phrase_id=args.phrase_id, flip_cam=args.flip)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[!] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
