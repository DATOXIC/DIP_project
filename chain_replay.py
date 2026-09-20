# -*- coding: utf-8 -*-
"""
CHAIN REPLAY - MÔ PHỎNG CÂU NỐI TỪ KHÔNG CẦN CAMERA (SPEC 08 Nhiệm vụ A / D / F)
================================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

MỤC ĐÍCH
    Ghép các stroke thô trong raw_strokes/<TỪ>/*.json thành luồng khung giả lập (30 Hz hoặc 15 Hz) của một CÂU 2-3 từ,
    với khoảng nghỉ g giữa hai từ kiểm soát được, rồi cho luồng chạy qua ĐÚNG chuỗi production:
        make_segmenter(CONFIG) -> dựng đặc trưng (dynamic_features.build_segment_features) -> GoldenTriadDTWMatcher.match
    (bank = webcam_templates.json hoặc bản sao truyền qua --bank; leave-one-out: mẫu bank trùng stroke nguồn bị loại).
    Đo: tỉ lệ câu ra ĐÚNG chuỗi, từ mất, từ chèn thừa, độ trễ chốt, "thời gian chờ tối thiểu" (g nhỏ nhất cho >= 90% câu đúng),
    phần đầu từ kế tiếp bị mất, và lỗi "chốt sớm sai trên từ nhiều pha".

CẢNH BÁO TRUNG THỰC
    Đây là dữ liệu TỔNG HỢP: các stroke được thu theo kiểu "dừng rồi ký", không có ảnh hưởng nối tiếp thật giữa hai từ
    (movement epenthesis, tay đã ở sẵn tư thế bắt đầu...). Kết quả chỉ dùng để SO SÁNH TƯƠNG ĐỐI giữa cấu hình và để
    tìm lỗi logic, KHÔNG phải độ chính xác thật khi người dùng ký liền. Vòng lặp quyết định ở đây MÔ PHỎNG vòng lặp của
    predict_sentence_webcam.py (cùng hàm dựng đặc trưng, cùng hàm early_commit_ok / find_multi_phase_words); không chạy
    được chính vòng lặp đó vì nó gắn với camera.

CÁCH DÙNG
    # Mốc "trước" (production hiện tại), 30 Hz, đủ các mức g:
    py chain_replay.py --bank <bank.json> --raw-dir <raw_strokes> --exclude-session-prefix 20260920_ --fps 30
    # Chế độ ký nhanh với cấu hình CONFIG["fast_chain"]:
    py chain_replay.py ... --fast-chain
    # Ghi đè tham số ký nhanh:  --pause-sec 0.25 --grace 0.10 --chain-window 1.2 --early-quality 80 --no-early
    # Kiểu nghỉ "carry" (nội suy từ tư thế cuối sang tư thế đầu từ kế, g <= 0.10): --mode carry --gaps 0,0.1
    # Nghiên cứu ảnh hưởng của việc cắt đuôi yên lên nhận diện từ đơn:   --study tail
    # Quét lưới tham số (Nhiệm vụ D bước 3):                            --study grid
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import difflib
import glob
import hashlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dynamic_dtw_engine import (  # noqa: E402
    GoldenTriadDTWMatcher,
    KineticState,
    is_stroke_duration_valid,
    is_movement_epenthesis,
    _FastDTWBank,
    _FAST_K,
    _FAST_D,
)
from dynamic_features import (  # noqa: E402
    build_segment_features, compute_palm_len, extend_still_tail, records_to_gesture_frames,
)
from pipeline_common import make_segmenter, find_multi_phase_words, early_commit_ok  # noqa: E402
from lead_trim import select_best_lead_trim, DEFAULT_LEAD_TRIM_CONFIG  # noqa: E402
import predict_sentence_webcam as psw  # noqa: E402
from predict_sentence_webcam import (  # noqa: E402
    CONFIG,
    compute_joint_flexion_angles,
    compute_palm_normal_vector,
    _segment_records,
)
from rebuild_templates_from_raw import rebuild_one_stroke  # noqa: E402

V_STOP = 0.040          # đúng CONFIG["v_stop_ratio"]
V_START = 0.08
HOLD_NOISE_PX = 0.56    # độ lệch chuẩn bước cổ tay (px) đo trên đuôi yên thật của raw_strokes (0.561)
T0 = 1000.0
NEWLINE = chr(10)


# =============================================================================
# 1. NẠP STROKE THÔ
# =============================================================================
@dataclass
class WordStroke:
    word: str
    path: str
    session: str
    t: np.ndarray            # (n,) giây, tăng dần
    wrist: np.ndarray        # (n, 2) cổ tay ĐÃ làm mượt (đúng thứ đưa vào bộ phân đoạn)
    palm: np.ndarray         # (n,) palm_len TÍNH LẠI bằng compute_palm_len (trường palm_len lưu sẵn là giá trị LỖI cũ)
    angles: np.ndarray       # (n, 5)
    normals: np.ndarray      # (n, 3)
    lm: np.ndarray           # (n, 21, 3) landmark pixel (z nhân bề rộng khung)
    face: List[Optional[Tuple[float, float, float, float]]]
    shapes: List[str]
    hand: List[str]
    last_motion: int         # chỉ số khung CUỐI còn chuyển động (v > v_stop); phần sau là đuôi đứng yên
    traj_ref: Optional[np.ndarray] = None    # (K, 24) đặc trưng tái dựng từ TOÀN BỘ stroke (để nhận mẫu bank tương ứng)
    bank_key: Optional[Tuple[str, int]] = None


def load_words(raw_dir: str, exclude_prefixes: Sequence[str] = ()) -> List[WordStroke]:
    """Nạp mọi stroke thô (bỏ NEGATIVE và các phiên có tiền tố trong exclude_prefixes)."""
    out: List[WordStroke] = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "*", "*.json"))):
        word = os.path.basename(os.path.dirname(f))
        if word == "NEGATIVE":
            continue
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        sess = str(r.get("session_id", ""))
        if any(sess.startswith(p) for p in exclude_prefixes if p):
            continue
        fr = r["frames"]
        n = len(fr)
        if n < 8:
            continue
        w_f, h_f = float(r["frame_width"]), float(r["frame_height"])
        t = np.array([x["timestamp"] for x in fr], dtype=np.float64)
        t = t - t[0]
        if not np.all(np.diff(t) > 0):
            continue
        lm_px = np.array([x["landmarks_px"] for x in fr], dtype=np.float32)
        palm = []
        for x in fr:
            l = x["landmarks_norm"]
            objs = [SimpleNamespace(x=l[i][0], y=l[i][1], z=l[i][2]) for i in (0, 5, 9, 17)]
            palm.append(compute_palm_len(objs[0], objs[1], objs[2], objs[3], w_f, h_f))
        hand = [x.get("handedness", "Right") for x in fr]
        angles = np.array([compute_joint_flexion_angles(lm_px[i]) for i in range(n)], dtype=np.float32)
        normals = np.array([compute_palm_normal_vector(lm_px[i], handedness=hand[i]) for i in range(n)], dtype=np.float32)
        wrist = np.array([x["wrist_smoothed"] for x in fr], dtype=np.float64)
        palm_a = np.array(palm, dtype=np.float64)
        v = np.r_[0.0, np.hypot(*np.diff(wrist, axis=0).T)] / np.maximum(palm_a, 15.0)
        mov = np.where(v > V_STOP)[0]
        last_motion = int(mov[-1]) if mov.size else 0
        face = [tuple(float(c) for c in x["face_box"]) if x.get("face_box") else None for x in fr]
        rb = rebuild_one_stroke(r, use_stored_features=False)
        out.append(WordStroke(
            word=word, path=f, session=sess, t=t, wrist=wrist, palm=palm_a, angles=angles, normals=normals,
            lm=lm_px, face=face, shapes=[x.get("hand_shape", "Unknown") for x in fr], hand=hand,
            last_motion=last_motion, traj_ref=(None if rb is None else np.asarray(rb["_traj_full"], dtype=np.float32)),
        ))
    return out


def map_bank_samples(words: List[WordStroke], vocab: Dict[str, Any], tol: float = 6e-4) -> int:
    """Gán bank_key = (từ, chỉ số mẫu) cho stroke thô nào có mẫu bank trùng (so đặc trưng tái dựng với traj_features)."""
    used: Dict[Tuple[str, int], float] = {}
    n_mapped = 0
    for w in words:
        if w.traj_ref is None or w.word not in vocab:
            continue
        best, best_d = None, tol
        for j, s in enumerate(vocab[w.word].get("samples", [])):
            tr = np.asarray(s.get("traj_features_9d") or s.get("traj_features"), dtype=np.float32)
            if tr.shape != w.traj_ref.shape:
                continue
            d = float(np.max(np.abs(tr - w.traj_ref)))
            if d < best_d and (w.word, j) not in used:
                best, best_d = j, d
        if best is not None:
            w.bank_key = (w.word, best)
            used[(w.word, best)] = best_d
            n_mapped += 1
    return n_mapped


# =============================================================================
# 2. DỰNG LUỒNG KHUNG GIẢ LẬP
# =============================================================================
def _resample_moving_part(w: WordStroke, fps: float) -> Dict[str, Any]:
    """Nội suy phần CHUYỂN ĐỘNG (khung 0..last_motion) lên lưới đều `fps` Hz."""
    n = w.last_motion + 1
    t = w.t[:n]
    total = float(t[-1] - t[0])
    m = max(3, int(math.floor(total * fps + 1e-9)) + 1)
    grid = np.arange(m) / fps
    idx = np.clip(np.searchsorted(t, grid, side="left"), 0, n - 1)
    near = np.where(np.abs(t[np.clip(idx - 1, 0, n - 1)] - grid) < np.abs(t[idx] - grid), np.clip(idx - 1, 0, n - 1), idx)

    def interp(a: np.ndarray) -> np.ndarray:
        a = a[:n].reshape(n, -1)
        return np.stack([np.interp(grid, t, a[:, c]) for c in range(a.shape[1])], axis=1)

    nrm = interp(w.normals)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    lm = interp(w.lm.reshape(len(w.lm), -1)).reshape(m, 21, 3).astype(np.float32)
    return dict(
        wrist=interp(w.wrist), palm=interp(w.palm)[:, 0], angles=interp(w.angles).astype(np.float32),
        normals=nrm.astype(np.float32), lm=lm, face=[w.face[int(i)] for i in near],
        shape=[w.shapes[int(i)] for i in near], hand=[w.hand[int(i)] for i in near], m=m,
    )


@dataclass
class Stream:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    palm: np.ndarray
    shape: List[str]
    angles: np.ndarray
    normals: np.ndarray
    lm: np.ndarray
    face: List[Optional[Tuple[float, float, float, float]]]
    hand: List[str]
    spans: List[Dict[str, Any]]      # mỗi từ: word, i0 (khung đầu), i1 (khung cuối còn chuyển động), key


def build_stream(words: Sequence[WordStroke], gaps: Sequence[float], fps: float, mode: str, rng: np.random.Generator,
                 lead: float = 1.0, trail: float = 1.0, cache: Optional[Dict[Any, Any]] = None,
                 hold_dest: float = 0.0, dup_frac: float = 0.0) -> Stream:
    """
    Ghép các từ thành một luồng liên tục.
    mode='hold'     : sau từ i tay ĐỨNG YÊN ở tư thế cuối (nhiễu nhỏ) đúng g giây, rồi từ i+1 bắt đầu. Từ i+1 được TỊNH TIẾN
                      (cổ tay, hộp mặt, landmark cùng một độ dời => mốc cơ thể giữ nguyên) sao cho khung ngay trước nó trùng
                      tư thế cuối của từ i => không có bước nhảy giả (một bước nhảy sẽ bị hiểu nhầm là quẹt trái).
    mode='carry'    : không tịnh tiến; trong g giây nội suy tuyến tính (cổ tay, palm, góc ngón, pháp tuyến, landmark) từ khung
                      cuối của từ i sang khung đầu của từ i+1 (tay đi thẳng sang tư thế mới, không đứng yên).
    mode='realistic': mô phỏng chuyển tiếp thực tế (SPEC 08B Nhiệm vụ A):
                      - g giây chuyển tiếp với profile min-jerk mượt mà s(u) = 10u^3 - 15u^4 + 6u^5.
                      - Thêm nhiễu gaussian nhẹ lên cổ tay và landmarks.
                      - Tùy chọn hold_dest (h giây) dừng nhẹ tại tư thế đích trước khi từ kế tiếp bắt đầu chuyển động.
                      - Ground truth i0 ghi nhận CHÍNH XÁC khung bắt đầu của từ kế tiếp (sau chuyển tiếp và sau hold_dest).
                      - Hỗ trợ dup_frac (lặp khung trước với xác suất dup_frac, dấu thời gian vẫn đều).
    g được đo từ khung cuối còn chuyển động của từ i tới khung đầu của từ i+1: thời gian giữa hai khung = ~g.
    """
    fr: Dict[str, List[Any]] = {k: [] for k in ("x", "y", "palm", "shape", "angles", "normals", "lm", "face", "hand")}
    spans: List[Dict[str, Any]] = []

    def push(x, y, palm, shape, ang, nrm, lm, face, hand):
        fr["x"].append(float(x)); fr["y"].append(float(y)); fr["palm"].append(float(palm)); fr["shape"].append(shape)
        fr["angles"].append(ang); fr["normals"].append(nrm); fr["lm"].append(lm); fr["face"].append(face); fr["hand"].append(hand)

    def hold(n, last):
        x, y = last["x"], last["y"]
        for _ in range(n):
            x += float(rng.normal(0.0, HOLD_NOISE_PX))
            y += float(rng.normal(0.0, HOLD_NOISE_PX))
            push(x, y, last["palm"], last["shape"], last["angles"], last["normals"], last["lm"], last["face"], last["hand"])
        return dict(last, x=x, y=y)

    last: Optional[Dict[str, Any]] = None
    for i, w in enumerate(words):
        key = (id(w), float(fps))
        if cache is not None and key in cache:
            G = cache[key]
        else:
            G = _resample_moving_part(w, fps)
            if cache is not None:
                cache[key] = G
        step = G["wrist"][1] - G["wrist"][0]
        need = 0.09 * max(float(G["palm"][0]), 15.0)
        nrm_s = float(np.hypot(step[0], step[1]))
        if nrm_s < 1e-6:
            step = np.array([need, 0.0])
        elif nrm_s < need:
            step = step * (need / nrm_s)
        pre = G["wrist"][0] - step                       # vị trí "khung ngay trước" khung đầu của từ (v khung đầu >= 0.09 palm)
        g = lead if i == 0 else float(gaps[i - 1])
        if i == 0:
            # Đoạn đứng yên đầu câu (mặc định 1.0 s) tại tư thế "ngay trước khung đầu" của từ 1, rồi từ 1 nối tiếp liền mạch.
            off0 = np.array([320.0, 240.0]) - pre
            first = dict(x=pre[0] + off0[0], y=pre[1] + off0[1], palm=G["palm"][0], shape=G["shape"][0], angles=G["angles"][0],
                         normals=G["normals"][0], lm=G["lm"][0] + np.array([off0[0], off0[1], 0.0], dtype=np.float32),
                         face=_shift_face(G["face"][0], off0), hand=G["hand"][0])
            last = hold(int(round(lead * fps)), first)
            off = np.array([last["x"], last["y"]]) - pre
        elif mode == "hold":
            n_h = max(0, int(round(g * fps)) - 1)
            last = hold(n_h, last)
            off = np.array([last["x"], last["y"]]) - pre
        elif mode == "realistic":
            off = np.zeros(2)
            n_trans = int(round(g * fps))
            tgt = dict(x=G["wrist"][0][0], y=G["wrist"][0][1], palm=G["palm"][0], angles=G["angles"][0],
                       normals=G["normals"][0], lm=G["lm"][0], shape=G["shape"][0], face=G["face"][0], hand=G["hand"][0])
            for j in range(n_trans):
                u = (j + 1) / (n_trans + 1)
                # Min-jerk polynomial: s(u) = 10*u^3 - 15*u^4 + 6*u^5
                s = 10.0 * (u ** 3) - 15.0 * (u ** 4) + 6.0 * (u ** 5)
                # Nội suy chuẩn hóa pháp tuyến
                nn = (1.0 - s) * last["normals"] + s * tgt["normals"]
                nn = nn / max(float(np.linalg.norm(nn)), 1e-6)
                # Thêm nhiễu gaussian nhỏ
                jitter_x = float(rng.normal(0.0, HOLD_NOISE_PX * 0.5))
                jitter_y = float(rng.normal(0.0, HOLD_NOISE_PX * 0.5))
                cur_x = (1.0 - s) * last["x"] + s * tgt["x"] + jitter_x
                cur_y = (1.0 - s) * last["y"] + s * tgt["y"] + jitter_y
                cur_palm = (1.0 - s) * last["palm"] + s * tgt["palm"]
                cur_ang = (1.0 - s) * last["angles"] + s * tgt["angles"]
                cur_lm = (1.0 - s) * last["lm"] + s * tgt["lm"]
                cur_lm[:, :2] += np.array([jitter_x, jitter_y], dtype=np.float32)
                cur_shape = tgt["shape"] if s >= 0.5 else last["shape"]
                cur_face = tgt["face"] if s >= 0.5 else last["face"]
                push(cur_x, cur_y, cur_palm, cur_shape, cur_ang.astype(np.float32),
                     nn.astype(np.float32), cur_lm.astype(np.float32), cur_face, tgt["hand"])
            if n_trans > 0:
                last = dict(x=fr["x"][-1], y=fr["y"][-1], palm=fr["palm"][-1], shape=fr["shape"][-1],
                            angles=fr["angles"][-1], normals=fr["normals"][-1], lm=fr["lm"][-1],
                            face=fr["face"][-1], hand=fr["hand"][-1])
            # Dừng nhẹ tại đích (hold_dest) nếu có
            if hold_dest > 0.0:
                n_hd = int(round(hold_dest * fps))
                if n_hd > 0:
                    last = hold(n_hd, last)
        else:  # carry
            off = np.zeros(2)
            n_c = int(round(g * fps))
            tgt = dict(x=G["wrist"][0][0], y=G["wrist"][0][1], palm=G["palm"][0], angles=G["angles"][0],
                       normals=G["normals"][0], lm=G["lm"][0])
            for j in range(n_c):
                a = (j + 1) / (n_c + 1)
                nn = (1 - a) * last["normals"] + a * tgt["normals"]
                nn = nn / max(float(np.linalg.norm(nn)), 1e-6)
                push((1 - a) * last["x"] + a * tgt["x"], (1 - a) * last["y"] + a * tgt["y"],
                     (1 - a) * last["palm"] + a * tgt["palm"], last["shape"], (1 - a) * last["angles"] + a * tgt["angles"],
                     nn.astype(np.float32), ((1 - a) * last["lm"] + a * tgt["lm"]).astype(np.float32), last["face"], last["hand"])
        i0 = len(fr["x"])
        shift3 = np.array([off[0], off[1], 0.0], dtype=np.float32)
        for k in range(G["m"]):
            push(G["wrist"][k][0] + off[0], G["wrist"][k][1] + off[1], G["palm"][k], G["shape"][k], G["angles"][k],
                 G["normals"][k], G["lm"][k] + shift3, _shift_face(G["face"][k], off), G["hand"][k])
        i1 = len(fr["x"]) - 1
        spans.append(dict(word=w.word, i0=i0, i1=i1, key=w.bank_key))
        last = dict(x=fr["x"][-1], y=fr["y"][-1], palm=fr["palm"][-1], shape=fr["shape"][-1], angles=fr["angles"][-1],
                    normals=fr["normals"][-1], lm=fr["lm"][-1], face=fr["face"][-1], hand=fr["hand"][-1])
    hold(int(round(trail * fps)), last)
    n = len(fr["x"])

    # Mô phỏng khung trùng (dup_frac) nếu được chỉ định
    if dup_frac > 0.0 and n > 2:
        new_fr: Dict[str, List[Any]] = {k: [] for k in ("x", "y", "palm", "shape", "angles", "normals", "lm", "face", "hand")}
        orig_to_new = np.zeros(n, dtype=np.int32)
        for idx in range(n):
            orig_to_new[idx] = len(new_fr["x"])
            # Xác suất lặp khung trước: thay thế khung hiện tại bằng khung trước
            if idx > 0 and rng.random() < dup_frac:
                for k in new_fr:
                    new_fr[k].append(new_fr[k][-1])
            else:
                for k in new_fr:
                    new_fr[k].append(fr[k][idx])
        # Cập nhật spans theo chỉ số khung mới
        for sp in spans:
            sp["i0"] = int(orig_to_new[sp["i0"]])
            sp["i1"] = int(orig_to_new[sp["i1"]])
        fr = new_fr
        n = len(fr["x"])

    return Stream(
        t=T0 + np.arange(n) / fps, x=np.array(fr["x"]), y=np.array(fr["y"]), palm=np.array(fr["palm"]), shape=fr["shape"],
        angles=np.array(fr["angles"], dtype=np.float32), normals=np.array(fr["normals"], dtype=np.float32),
        lm=np.array(fr["lm"], dtype=np.float32), face=fr["face"], hand=fr["hand"], spans=spans,
    )


def _shift_face(face, off):
    if face is None:
        return None
    return (face[0] + float(off[0]), face[1] + float(off[1]), face[2], face[3])


# =============================================================================
# 3. MATCHER LEAVE-ONE-OUT
# =============================================================================
class LooMatcher(GoldenTriadDTWMatcher):
    """
    GoldenTriadDTWMatcher có thể LOẠI một số mẫu bank (chỉ số toàn cục) khỏi phép so khớp, không dựng lại bank.
    LƯU Ý: độ rộng nội lớp (intra_widths, dùng cho ngưỡng scale_k) vẫn tính từ bank ĐẦY ĐỦ (gồm cả mẫu bị loại) =>
    hơi lạc quan so với mẫu chưa từng thấy; không dựng lại vì mỗi lần tính lại mất hàng phút. Dùng cho so sánh tương đối.
    """
    excluded: Tuple[int, ...] = ()

    def _fast_min_distances(self, query_traj_9d: Any, query_has_body: bool) -> Optional[Dict[str, float]]:
        if not self.excluded:
            return super()._fast_min_distances(query_traj_9d, query_has_body)
        q = np.asarray(query_traj_9d, dtype=np.float32)
        if q.ndim != 2 or q.shape != (_FAST_K, _FAST_D):
            raise RuntimeError("LooMatcher: query sai kích thước")
        sig = self._fast_signature()
        if self._fast_bank is None or sig != self._fast_sig:
            self._fast_bank = _FastDTWBank(self)
            self._fast_sig = sig
        bank = self._fast_bank
        if not bank.supported:
            raise RuntimeError(f"LooMatcher cần đường DTW nhanh: {bank.reason}")
        d = bank.distances(q, query_has_body).copy()
        d[list(self.excluded)] = np.inf
        return {wk: float(np.min(d[a:b])) for wk, (a, b) in zip(bank.words, bank.slices)}

    def global_index(self, key: Tuple[str, int]) -> int:
        """Chỉ số toàn cục của mẫu (từ, j) trong khối nhanh."""
        if self._fast_bank is None:
            self._fast_bank = _FastDTWBank(self)
            self._fast_sig = self._fast_signature()
        bank = self._fast_bank
        a, b = bank.slices[bank.words.index(key[0])]
        if b - a != len(self.vocabulary[key[0]]["samples"]):
            raise RuntimeError(f"Từ {key[0]} có mẫu bị bỏ qua (sai độ dài) - không ánh xạ được chỉ số")
        return a + key[1]


def build_matcher(bank_path: str, cache_dir: str, threshold_k: float, uncertainty_k: float) -> LooMatcher:
    """Dựng matcher đúng như run_sentence_builder (trọng số toàn cục CONFIG, ngưỡng scale_k)."""
    return LooMatcher(
        templates_path=bank_path,
        sign_weights_path=CONFIG.get("sign_weights_path") or "",
        w_p=CONFIG["w_wrist_pos"], w_a=CONFIG["w_finger_angles"], w_m=CONFIG["w_wrist_dir"], w_n=CONFIG["w_palm_normal"],
        lambda_shape=CONFIG["lambda_shape"], margin_threshold=CONFIG["margin_threshold"],
        uncertainty_ratio=CONFIG["uncertainty_ratio"], threshold_override=None,
        asllex_blend_alpha=CONFIG.get("asllex_blend_alpha", 1.0), asllex_max_deviation=CONFIG.get("asllex_max_deviation"),
        w_z=CONFIG.get("w_z", 0.0), w_b=CONFIG.get("w_b", 0.0), w_s=CONFIG.get("w_s", 0.0), w_r=CONFIG.get("w_r", 0.0),
        threshold_mode=CONFIG.get("threshold_mode", "scale_k"), threshold_k=threshold_k, uncertainty_k=uncertainty_k,
        cache_dir=cache_dir,
    )


# =============================================================================
# 4. CHẠY MỘT LUỒNG QUA CHUỖI PRODUCTION
# =============================================================================
def run_stream(stream: Stream, matcher: LooMatcher, fast_cfg: Optional[Dict[str, Any]], multi_phase: Sequence[str]) -> Dict[str, Any]:
    """
    Mô phỏng vòng lặp predict_sentence_webcam.py. fast_cfg=None => production hiện tại (bộ phân đoạn make_segmenter(CONFIG),
    không chốt sớm, không kéo đuôi). fast_cfg khác None => chế độ ký nhanh với các tham số trong dict.
    Trả về dict: words [(từ, t_chốt, cách)], strokes [...], swipes, early_commits [...].
    """
    seg = make_segmenter(CONFIG, fast_cfg)
    fast = fast_cfg is not None
    tail_pad = fast_cfg.get("tail_pad_sec") if fast else None
    early_on = bool(fast and fast_cfg.get("early_commit"))
    ang_h: List[Any] = []
    nrm_h: List[Any] = []
    ext_h: List[Any] = []
    words: List[Tuple[str, float, str]] = []
    strokes: List[Dict[str, Any]] = []
    swipes = 0
    early_list: List[Dict[str, Any]] = []
    has_pending = False
    last_commit_time = 0.0
    last_committed: Optional[str] = None
    n_match = 0
    hitch_times: List[float] = []
    lead_trim_cfg = fast_cfg.get("lead_trim") if fast else None
    lead_trim_on = bool(fast and lead_trim_cfg and lead_trim_cfg.get("enabled"))
    n = len(stream.t)
    for k in range(n):
        t = float(stream.t[k])
        state, completed, is_swipe, _v = seg.update(
            wrist_x=float(stream.x[k]), wrist_y=float(stream.y[k]), palm_len=float(stream.palm[k]),
            hand_shape=stream.shape[k], timestamp=t)
        ang_h.append(stream.angles[k]); nrm_h.append(stream.normals[k]); ext_h.append((stream.lm[k], stream.face[k]))
        if len(ang_h) > 120:
            ang_h.pop(0); nrm_h.pop(0); ext_h.pop(0)
        handed = stream.hand[k]
        if state == KineticState.RECORDING:
            has_pending = False
        if state in (KineticState.IDLE, KineticState.READY):
            has_pending = False

        pre_match = None      # (traj, shapes, body_ok, match) của dự đoán tạm khi chốt sớm

        # ---- dự đoán tạm khi PENDING_COMMIT (production luôn chạy; ở đây chỉ chạy khi cần để chốt sớm) ----
        if state == KineticState.PENDING_COMMIT and not has_pending:
            has_pending = True
            if early_on:
                buf = seg.buffer
                if buf and len(buf) >= seg.effective_min_stroke_frames():
                    recs = _segment_records(buf, ang_h, nrm_h, ext_h)
                    epi = buf
                    if tail_pad:
                        recs, _ = extend_still_tail(recs, float(tail_pad), V_STOP, replay_jitter=True)
                        epi = records_to_gesture_frames(recs)
                    dur = recs[-1]["t"] - recs[0]["t"]
                    if is_stroke_duration_valid(dur) and not is_movement_epenthesis(epi):
                        if lead_trim_on:
                            _t_trim_start = time.perf_counter()
                            res_trim = select_best_lead_trim(recs, matcher, cfg=lead_trim_cfg, handedness=handed,
                                                            query_has_body=True, tail_pad_sec=tail_pad, v_stop=V_STOP)
                            hitch_times.append((time.perf_counter() - _t_trim_start) * 1000.0)
                            m = res_trim.match_result
                            n_match += res_trim.n_tried
                            if early_commit_ok(m, fast_cfg, multi_phase):
                                forced = seg.force_commit(t)
                                if forced is not None:
                                    completed = forced
                                    pre_match = (None, None, True, m, float(res_trim.trim_sec), int(res_trim.n_tried))
                                    early_list.append(dict(t=t, word=m.top_word))
                        else:
                            traj, shp, body_ok, _dom = build_segment_features(recs)
                            if traj is not None:
                                m = matcher.match(traj, shp, handedness=handed, query_has_body=body_ok)
                                n_match += 1
                                if early_commit_ok(m, fast_cfg, multi_phase):
                                    forced = seg.force_commit(t)
                                    if forced is not None:
                                        completed = forced
                                        pre_match = (traj, shp, body_ok, m, 0.0, 1)
                                        early_list.append(dict(t=t, word=m.top_word))
        # ---- Swipe-Left = Backspace ----
        if is_swipe:
            swipes += 1
            if words:
                words.pop()
            ang_h.clear(); ext_h.clear()
        # ---- cử chỉ hoàn thành ----
        if completed is not None:
            info = dict(t0=completed[0].timestamp, t1=completed[-1].timestamp, n=len(completed), outcome="", top=None,
                        via="early" if pre_match is not None else "normal", t_emit=t, hitch_ms=0.0, lead_trim_sec=0.0)
            recs = _segment_records(completed, ang_h, nrm_h, ext_h)
            epi = completed
            if tail_pad:
                recs, _ = extend_still_tail(recs, float(tail_pad), V_STOP, replay_jitter=True)
                epi = records_to_gesture_frames(recs)
            dur = (recs[-1]["t"] - recs[0]["t"]) if tail_pad else (completed[-1].timestamp - completed[0].timestamp)
            if not is_stroke_duration_valid(dur):
                info["outcome"] = "duration_gate"
            elif is_movement_epenthesis(epi):
                info["outcome"] = "epenthesis"
            else:
                ang_h.clear(); nrm_h.clear(); ext_h.clear()
                if pre_match is not None:
                    _traj, _shp, _body, m, _tau, _nt = pre_match
                    info["lead_trim_sec"] = _tau
                elif lead_trim_on:
                    _t_trim_start = time.perf_counter()
                    res_trim = select_best_lead_trim(recs, matcher, cfg=lead_trim_cfg, handedness=handed,
                                                    query_has_body=True, tail_pad_sec=tail_pad, v_stop=V_STOP)
                    hitch_ms = (time.perf_counter() - _t_trim_start) * 1000.0
                    hitch_times.append(hitch_ms)
                    info["hitch_ms"] = hitch_ms
                    info["lead_trim_sec"] = float(res_trim.trim_sec)
                    m = res_trim.match_result
                    n_match += res_trim.n_tried
                else:
                    traj, shp, body_ok, _dom = build_segment_features(recs)
                    if traj is None:
                        m = None
                    else:
                        m = matcher.match(traj, shp, handedness=handed, query_has_body=body_ok)
                        n_match += 1
                if m is None:
                    info["outcome"] = "too_short"
                else:
                    info["outcome"] = m.status
                    info["top"] = m.top_word
                    if m.status == "ACCEPTED" and m.top_word:
                        rep = (m.top_word == last_committed and (t - last_commit_time) < 0.8)
                        if not rep:
                            words.append((m.top_word, t, info["via"]))
                            last_commit_time = t
                            last_committed = m.top_word
            strokes.append(info)
    return dict(words=words, strokes=strokes, swipes=swipes, early=early_list, n_match=n_match,
                hitch_times=hitch_times)


# =============================================================================
# 5. ĐÁNH GIÁ MỘT CÂU
# =============================================================================
def evaluate_sentence(stream: Stream, res: Dict[str, Any], fps: float, multi_phase: Sequence[str]) -> Dict[str, Any]:
    truth = [s["word"] for s in stream.spans]
    pred = [w[0] for w in res["words"]]
    sm = difflib.SequenceMatcher(a=truth, b=pred, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    lcs = sum(b.size for b in blocks)
    pairs = [(b.a + j, b.b + j) for b in blocks for j in range(b.size)]
    lat = []
    for ti, pi in pairs:
        t_stop = float(stream.t[stream.spans[ti]["i1"]])
        lat.append(res["words"][pi][1] - t_stop)
    # phần đầu từ bị mất + từ bị gộp vào stroke của từ khác + leadin contamination & head loss > 0.10s
    head_lost, no_stroke, merged = [], 0, 0
    leadin_contam_words = 0
    head_loss_words = 0
    start_errors: List[float] = []

    for i, sp in enumerate(stream.spans):
        t_i0 = float(stream.t[sp["i0"]])
        t_stop = float(stream.t[sp["i1"]])
        cover = [s for s in res["strokes"] if s["t0"] - 1e-9 <= t_stop <= s["t1"] + 1e-9]
        if not cover:
            no_stroke += 1
            continue
        s0 = cover[0]
        # s0_start có thể được tinh chỉnh bởi lead_trim_sec nếu stroke đó áp dụng lead_trim
        s0_start = s0["t0"] + float(s0.get("lead_trim_sec", 0.0))
        err = s0_start - t_i0
        start_errors.append(err)
        if err < -0.10:
            leadin_contam_words += 1
        elif err > 0.10:
            head_loss_words += 1

        head_lost.append(max(0, int(round((s0["t0"] - t_i0) * fps))))
        n_words_in = sum(1 for sp2 in stream.spans if s0["t0"] - 1e-9 <= float(stream.t[sp2["i1"]]) <= s0["t1"] + 1e-9)
        if n_words_in > 1:
            merged += 1
    # chốt sớm sai trên từ nhiều pha: chốt sớm xảy ra TRONG khoảng của từ (trước khi từ thật sự dừng)
    mp_words = 0
    mp_bad = 0
    prem_any = 0          # số từ bị cắt giữa chừng bởi BẤT KỲ kiểu chốt nào (chốt sớm hoặc chốt thường)
    for i, sp in enumerate(stream.spans):
        t_i0 = float(stream.t[sp["i0"]])
        t_stop = float(stream.t[sp["i1"]])
        if any(t_i0 <= s["t1"] < t_stop - 1e-9 for s in res["strokes"]):
            prem_any += 1
        if sp["word"] not in multi_phase:
            continue
        mp_words += 1
        bad = any(t_i0 <= e["t"] < t_stop - 1e-9 for e in res["early"])   # chốt sớm khi từ CHƯA xong
        mp_bad += int(bad)
    return dict(
        truth=truth, pred=pred, correct=(truth == pred), lost=len(truth) - lcs, extra=len(pred) - lcs,
        latency=lat, head_lost=head_lost, no_stroke=no_stroke, merged=merged, n_words=len(truth), swipes=res["swipes"],
        early_n=len(res["early"]), mp_words=mp_words, mp_bad=mp_bad, prem_any=prem_any,
        early_wrong=sum(1 for e in res["early"] if e["word"] not in truth),
        outcomes=[s["outcome"] for s in res["strokes"]],
        leadin_contam=leadin_contam_words, head_loss_words=head_loss_words,
        start_errors=start_errors, hitch_times=res.get("hitch_times", []),
    )


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    lat = [x for r in rows for x in r["latency"]]
    hl = [x for r in rows for x in r["head_lost"]]
    nw = sum(r["n_words"] for r in rows)
    outc: Dict[str, int] = {}
    for r in rows:
        for o in r["outcomes"]:
            outc[o] = outc.get(o, 0) + 1
    mpw = sum(r["mp_words"] for r in rows)
    st_errs = [x for r in rows for x in r.get("start_errors", [])]
    hitch_all = [x for r in rows for x in r.get("hitch_times", [])]
    tot_words = max(nw, 1)

    return dict(
        sentences=n, words=nw, acc=sum(r["correct"] for r in rows) / max(n, 1),
        word_acc=1.0 - (sum(r["lost"] for r in rows) + sum(r["extra"] for r in rows)) / tot_words,
        lost=sum(r["lost"] for r in rows), extra=sum(r["extra"] for r in rows),
        lat_median=float(np.median(lat)) if lat else float("nan"), lat_p90=float(np.percentile(lat, 90)) if lat else float("nan"),
        lat_p95=float(np.percentile(lat, 95)) if lat else float("nan"),
        head_mean=float(np.mean(hl)) if hl else float("nan"), head_frac=float(np.mean([h >= 1 for h in hl])) if hl else float("nan"),
        no_stroke=sum(r["no_stroke"] for r in rows), merged=sum(r["merged"] for r in rows), swipes=sum(r["swipes"] for r in rows),
        early_n=sum(r["early_n"] for r in rows), early_wrong=sum(r["early_wrong"] for r in rows),
        mp_words=mpw, mp_bad=sum(r["mp_bad"] for r in rows), mp_rate=(sum(r["mp_bad"] for r in rows) / mpw) if mpw else 0.0,
        prem_any=sum(r["prem_any"] for r in rows), prem_any_rate=sum(r["prem_any"] for r in rows) / tot_words,
        leadin_contam=sum(r.get("leadin_contam", 0) for r in rows),
        leadin_contam_rate=sum(r.get("leadin_contam", 0) for r in rows) / tot_words,
        head_loss_words=sum(r.get("head_loss_words", 0) for r in rows),
        head_loss_rate=sum(r.get("head_loss_words", 0) for r in rows) / tot_words,
        start_err_median=float(np.median(st_errs)) if st_errs else float("nan"),
        start_err_p90=float(np.percentile(st_errs, 90)) if st_errs else float("nan"),
        hitch_mean_ms=float(np.mean(hitch_all)) if hitch_all else 0.0,
        hitch_p95_ms=float(np.percentile(hitch_all, 95)) if hitch_all else 0.0,
        hitch_max_ms=float(np.max(hitch_all)) if hitch_all else 0.0,
        outcomes=outc,
    )


# =============================================================================
# 6. SINH CÂU, CHẠY MỘT MỨC g
# =============================================================================
def isolated_ok_words(words: List[WordStroke], matcher: "LooMatcher") -> List[WordStroke]:
    """
    Các stroke mà khi nhận diện ĐƠN LẺ (cả stroke thô như lúc thu, leave-one-out) ra ACCEPTED-đúng.
    Dùng làm nguyên liệu câu để đo riêng ảnh hưởng của việc NỐI TỪ, tách khỏi giới hạn nhận diện từ đơn của bank.
    """
    keep = []
    for w in words:
        frames = [dict(x=float(w.wrist[i, 0]), y=float(w.wrist[i, 1]), palm_len=max(float(w.palm[i]), 15.0), hand_shape=w.shapes[i],
                       t=float(w.t[i]), angles=w.angles[i], normal=w.normals[i], landmarks=w.lm[i], face_box=w.face[i])
                  for i in range(len(w.t))]
        traj, shp, body_ok, _ = build_segment_features(frames)
        if traj is None:
            continue
        matcher.excluded = (matcher.global_index(w.bank_key),) if w.bank_key is not None else ()
        m = matcher.match(traj, shp, handedness=w.hand[-1], query_has_body=body_ok)
        if m.status == "ACCEPTED" and m.top_word == w.word:
            keep.append(w)
    matcher.excluded = ()
    return keep


def make_sentences(words: List[WordStroke], n: int, seed: int) -> List[List[WordStroke]]:
    """n câu 2-3 từ KHÁC NHAU (seed cố định => cùng bộ câu cho mọi g và mọi cấu hình, so sánh ghép cặp được)."""
    rng = np.random.default_rng(seed)
    by_word: Dict[str, List[WordStroke]] = {}
    for w in words:
        by_word.setdefault(w.word, []).append(w)
    vocab = sorted(by_word)
    out = []
    for _ in range(n):
        k = 2 if rng.random() < 0.6 else 3
        chosen = rng.choice(len(vocab), size=k, replace=False)
        out.append([by_word[vocab[int(c)]][int(rng.integers(0, len(by_word[vocab[int(c)]])))] for c in chosen])
    return out


def sample_transition_gaps(n_gaps: int, mode: str, dist_str: Optional[str], fixed_g: float, rng: np.random.Generator) -> List[float]:
    if mode != "realistic":
        return [fixed_g] * n_gaps
    if dist_str:
        # Hỗ trợ tùy chọn chuỗi phân phối tùy biến, e.g. "0.2:0.0,0.8:0.10-0.45"
        pass
    # Mặc định SPEC 08B: 20% d = 0.0, 80% d ~ Uniform[0.10, 0.45]
    gaps = []
    for _ in range(n_gaps):
        if rng.random() < 0.20:
            gaps.append(0.0)
        else:
            gaps.append(float(rng.uniform(0.10, 0.45)))
    return gaps


def run_level(sentences: List[List[WordStroke]], g: float, fps: float, mode: str, matcher: LooMatcher,
              fast_cfg: Optional[Dict[str, Any]], multi_phase: Sequence[str], seed: int,
              cache: Dict[Any, Any], keep_rows: bool = False,
              hold_dest: float = 0.0, dup_frac: float = 0.0, dist_str: Optional[str] = None) -> Dict[str, Any]:
    rows = []
    for si, ws in enumerate(sentences):
        rng = np.random.default_rng([seed, int(round(g * 1000)), si])
        gaps = sample_transition_gaps(len(ws) - 1, mode, dist_str, g, rng)
        stream = build_stream(ws, gaps, fps, mode, rng, cache=cache, hold_dest=hold_dest, dup_frac=dup_frac)
        matcher.excluded = tuple(sorted(matcher.global_index(w.bank_key) for w in ws if w.bank_key is not None))
        res = run_stream(stream, matcher, fast_cfg, multi_phase)
        rows.append(evaluate_sentence(stream, res, fps, multi_phase))
    agg = aggregate(rows)
    agg["g"] = g
    if keep_rows:
        agg["rows"] = rows
    return agg


def fast_cfg_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if not args.fast_chain:
        return None
    cfg = dict(CONFIG["fast_chain"])
    cfg["enabled"] = True
    if args.pause_sec is not None:
        cfg["pause_sec"] = args.pause_sec
    if args.grace is not None:
        cfg["grace_window_sec"] = args.grace
    if args.chain_window is not None:
        cfg["chain_window_sec"] = args.chain_window
    if args.early_quality is not None:
        cfg["early_min_quality"] = args.early_quality
    if args.early_margin is not None:
        cfg["early_min_margin"] = args.early_margin
    if args.no_early:
        cfg["early_commit"] = False
    if args.tail_pad is not None:
        cfg["tail_pad_sec"] = None if args.tail_pad <= 0 else args.tail_pad
    if getattr(args, "transition_lockout", None) is not None:
        cfg["transition_lockout_sec"] = float(args.transition_lockout)
    if getattr(args, "lead_trim", None) is not None:
        cfg.setdefault("lead_trim", dict(DEFAULT_LEAD_TRIM_CONFIG))
        cfg["lead_trim"]["enabled"] = bool(args.lead_trim)
    return cfg


def min_wait(rows: List[Dict[str, Any]], target: float = 0.90) -> Optional[float]:
    """g nhỏ nhất (giây) mà từ đó trở lên (mọi g lớn hơn trong lưới) đạt >= target; None nếu không có."""
    ok = None
    for r in sorted(rows, key=lambda x: x["g"], reverse=True):
        if r["acc"] >= target:
            ok = r["g"]
        else:
            break
    return ok


def multi_phase_for(matcher: GoldenTriadDTWMatcher, fast_cfg: Optional[Dict[str, Any]]) -> List[str]:
    if fast_cfg and fast_cfg.get("multi_phase_words") is not None:
        return list(fast_cfg["multi_phase_words"])
    return sorted(find_multi_phase_words(matcher.vocabulary))


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# =============================================================================
# 7. NGHIÊN CỨU CẮT ĐUÔI (Nhiệm vụ D bước 1) - nhận diện từ ĐƠN
# =============================================================================
def study_tail(words: List[WordStroke], matcher: LooMatcher, keeps: Sequence[float], pad_to: float) -> List[Dict[str, Any]]:
    """
    Với mỗi stroke thô (leave-one-out mẫu bank tương ứng), cắt đuôi yên còn `keep` giây ở phía TRUY VẤN (bank giữ nguyên),
    dựng đặc trưng bằng build_segment_features rồi match. Báo top-1, d_correct, ACCEPTED-đúng, wrong-accept.
    Biến thể 'pad' = cắt còn keep rồi kéo dài lại về pad_to giây (extend_still_tail).
    """
    results: List[Dict[str, Any]] = []
    variants = ([("native", None, None, False)] + [(f"trim{k:.2f}", k, None, False) for k in keeps]
                + [(f"trim{k:.2f}+pad", k, pad_to, False) for k in keeps if k < pad_to]
                + [(f"trim{k:.2f}+pad+jitter", k, pad_to, True) for k in keeps if k < pad_to])
    for name, keep, pad, jit in variants:
        top1 = acc_ok = wrong_acc = n = rej = unc = epi_n = 0
        dcor: List[float] = []
        for w in words:
            n_all = len(w.t)
            # đuôi thật: khung sau last_motion. Giữ các khung có t <= t_last_motion + keep
            end = n_all if keep is None else int(np.searchsorted(w.t, w.t[w.last_motion] + keep + 1e-9, side="right"))
            end = max(end, w.last_motion + 1)
            frames = []
            for i in range(end):
                frames.append(dict(x=float(w.wrist[i, 0]), y=float(w.wrist[i, 1]), palm_len=max(float(w.palm[i]), 15.0),
                                   hand_shape=w.shapes[i], t=float(w.t[i]), angles=w.angles[i], normal=w.normals[i],
                                   landmarks=w.lm[i], face_box=w.face[i]))
            if pad is not None:
                frames, _ = extend_still_tail(frames, pad, V_STOP, replay_jitter=jit)
            traj, shp, body_ok, _ = build_segment_features(frames)
            if traj is None:
                continue
            epi_n += int(is_movement_epenthesis(records_to_gesture_frames(frames)))
            matcher.excluded = (matcher.global_index(w.bank_key),) if w.bank_key is not None else ()
            m = matcher.match(traj, shp, handedness=w.hand[end - 1], query_has_body=body_ok)
            n += 1
            ok = (m.top_word == w.word)
            top1 += ok
            for c in m.candidates:
                if c.word_key == w.word:
                    dcor.append(c.distance)
            if m.status == "ACCEPTED":
                acc_ok += ok
                wrong_acc += (not ok)
            elif m.status == "UNCERTAIN":
                unc += 1
            else:
                rej += 1
        results.append(dict(variant=name, n=n, top1=top1, top1_pct=100.0 * top1 / max(n, 1), accepted_correct=acc_ok,
                            accepted_correct_pct=100.0 * acc_ok / max(n, 1), wrong_accept=wrong_acc,
                            uncertain=unc, rejected=rej, epenthesis=epi_n, d_correct_mean=float(np.mean(dcor)), d_correct_median=float(np.median(dcor))))
    return results


# =============================================================================
# 8. GIAO DIỆN DÒNG LỆNH
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mô phỏng câu nối từ không cần camera (SPEC 08). Xem docstring đầu file.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default=os.path.join(HERE, "webcam_templates.json"), help="Bank template (nên là bản sao đóng băng).")
    p.add_argument("--raw-dir", default=os.path.join(HERE, "raw_strokes"), help="Thư mục raw_strokes.")
    p.add_argument("--exclude-session-prefix", action="append", default=[], help="Bỏ stroke thô thuộc phiên có tiền tố này (lặp được).")
    p.add_argument("--fps", type=float, default=30.0, help="Nhịp khung của luồng giả lập (30 hoặc 15).")
    p.add_argument("--gaps", default="0.0,0.10,0.20,0.30,0.50,0.80", help="Các mức nghỉ g (giây), cách nhau bằng dấu phẩy.")
    p.add_argument("--n", type=int, default=150, help="Số câu mỗi mức g (>= 150).")
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--mode", choices=["hold", "carry", "realistic"], default="hold")
    p.add_argument("--transition-model", dest="mode", choices=["hold", "carry", "realistic"], default=None,
                   help="Mô hình chuyển tiếp giữa hai từ: hold (đứng yên), carry (nội suy), realistic (min-jerk).")
    p.add_argument("--transition-dist", default=None,
                   help="Phân phối thời gian chuyển tiếp d (giây) cho realistic, ví dụ '0.2:0.0,0.8:0.10-0.45' (mặc định 20%% d=0, 80%% d in [0.10, 0.45]).")
    p.add_argument("--hold-dest", type=float, default=0.0,
                   help="Thời gian dừng nhẹ tại đích (giây) sau chuyển tiếp trong realistic (0, 0.05, 0.10).")
    p.add_argument("--dup-frac", type=float, default=0.0,
                   help="Tỉ lệ khung trùng lặp ngẫu nhiên mô phỏng camera lag (0.0 đến 0.5).")
    p.add_argument("--fast-chain", action="store_true", help="Dùng chế độ ký nhanh (CONFIG['fast_chain']).")
    p.add_argument("--pause-sec", type=float, default=None)
    p.add_argument("--grace", type=float, default=None)
    p.add_argument("--chain-window", type=float, default=None)
    p.add_argument("--early-quality", type=float, default=None)
    p.add_argument("--early-margin", type=float, default=None)
    p.add_argument("--no-early", action="store_true", help="Tắt chốt sớm (giữ pause/grace/chain_window).")
    p.add_argument("--tail-pad", type=float, default=None, help="tail_pad_sec (<=0 = tắt kéo đuôi).")
    p.add_argument("--transition-lockout", type=float, default=None,
                   help="Khoảng cấm chuyển tiếp sau khi chốt từ (giây), ví dụ 0.15.")
    p.add_argument("--lead-trim", dest="lead_trim", action="store_true", default=None,
                   help="Bật cắt phần dẫn đầu thích nghi (lead_trim).")
    p.add_argument("--no-lead-trim", dest="lead_trim", action="store_false", default=None,
                   help="Tắt cắt phần dẫn đầu thích nghi.")
    p.add_argument("--threshold-k", type=float, default=None)
    p.add_argument("--uncertainty-k", type=float, default=None)
    p.add_argument("--all-strokes", action="store_true",
                   help="Dùng MỌI stroke thô làm nguyên liệu câu (mặc định chỉ dùng stroke mà nhận diện đơn lẻ ra ACCEPTED-đúng, "
                        "để tách ảnh hưởng của nối từ khỏi giới hạn nhận diện từ đơn).")
    p.add_argument("--emit-csv", default=None,
                   help="Ghi một phiên GIẢ LẬP dạng CSV --log-session (kèm <đường dẫn>.truth.txt) để thử chain_trial_report.py.")
    p.add_argument("--emit-g", type=float, default=0.6, help="Mức nghỉ g dùng khi ghi phiên giả lập (giây).")
    p.add_argument("--emit-n", type=int, default=20, help="Số câu trong phiên giả lập.")
    p.add_argument("--study", choices=["tail", "grid"], default=None)
    p.add_argument("--workers", type=int, default=6, help="Số tiến trình cho --study grid.")
    p.add_argument("--out", default=None, help="Thư mục kết quả (mặc định results/chain_replay_<mốc thời gian>).")
    p.add_argument("--label", default="", help="Nhãn ghép vào tên thư mục kết quả.")
    return p.parse_args()


def setup(args: argparse.Namespace):
    cache_dir = os.path.join(os.environ.get("TEMP", HERE), "chain_replay_cache")
    tk = args.threshold_k if args.threshold_k is not None else float(CONFIG.get("threshold_k", 1.75))
    uk = args.uncertainty_k if args.uncertainty_k is not None else float(CONFIG.get("uncertainty_k", 2.35))
    matcher = build_matcher(args.bank, cache_dir, tk, uk)
    words = load_words(args.raw_dir, args.exclude_session_prefix)
    n_map = map_bank_samples(words, matcher.vocabulary)
    if not getattr(args, "all_strokes", False) and getattr(args, "study", None) != "tail":
        n_all = len(words)
        words = isolated_ok_words(words, matcher)
        print(f"[chain_replay] nguyên liệu câu: {len(words)}/{n_all} stroke nhận đúng khi đứng một mình (dùng --all-strokes để lấy hết)")
    return matcher, words, n_map, tk, uk


def fmt_table(aggs: List[Dict[str, Any]], title: str) -> str:
    lines = [f"### {title}", "",
             "| g (s) | câu đúng | từ đúng | mất | thừa | nhiễm dẫn đầu (>0.1s) | mất đầu từ (>0.1s) | trễ chốt TB/p95 (s) | hitch p95 (ms) | quẹt | chốt sớm | nhiều pha cắt |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in aggs:
        lines.append(f"| {a['g']:.2f} | {100*a['acc']:.1f}% ({round(a['acc']*a['sentences'])}/{a['sentences']}) | {100*a.get('word_acc', 0.0):.1f}% | "
                     f"{a['lost']} | {a['extra']} | {a.get('leadin_contam', 0)} ({100*a.get('leadin_contam_rate', 0.0):.1f}%) | "
                     f"{a.get('head_loss_words', 0)} ({100*a.get('head_loss_rate', 0.0):.1f}%) | "
                     f"{a['lat_median']:.3f}/{a.get('lat_p95', a['lat_p90']):.3f} | {a.get('hitch_p95_ms', 0.0):.1f} | "
                     f"{a['swipes']} | {a['early_n']} | {a['mp_bad']}/{a['mp_words']} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    t_start = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(HERE, "results", f"chain_replay_{ts}" + (f"_{args.label}" if args.label else ""))
    os.makedirs(out_dir, exist_ok=True)
    matcher, words, n_map, tk, uk = setup(args)
    fast_cfg = fast_cfg_from_args(args)
    mp = multi_phase_for(matcher, fast_cfg)
    print(f"[chain_replay] bank={args.bank} sha256={file_sha256(args.bank)[:16]} | từ={len(matcher.vocabulary)} "
          f"mẫu={sum(len(v['samples']) for v in matcher.vocabulary.values())} | stroke thô={len(words)} (ánh xạ bank: {n_map}) | k={tk}, unc_k={uk}")
    print(f"[chain_replay] chế độ={'FAST ' + json.dumps(fast_cfg, ensure_ascii=False) if fast_cfg else 'PRODUCTION (mốc)'} | từ nhiều pha: {mp}")
    meta = dict(bank=args.bank, bank_sha256=file_sha256(args.bank), n_words=len(words), n_mapped=n_map, fps=args.fps, seed=args.seed,
                n=args.n, mode=args.mode, fast_cfg=fast_cfg, multi_phase=mp, threshold_k=tk, uncertainty_k=uk,
                exclude_prefix=args.exclude_session_prefix, hold_dest=args.hold_dest, dup_frac=args.dup_frac)

    if args.emit_csv:
        emit_session(args, matcher, words, fast_cfg, mp)
        return

    if args.study == "tail":
        res = study_tail(words, matcher, [0.45, 0.30, 0.20, 0.10], 0.45)
        with open(os.path.join(out_dir, "tail_study.json"), "w", encoding="utf-8") as f:
            json.dump(dict(meta=meta, results=res), f, ensure_ascii=False, indent=1)
        print("\n| biến thể | n | top-1 | ACCEPTED-đúng | wrong-accept | UNCERTAIN | REJECTED | loại bởi heuristic chuyển động chêm | d_correct TB | d_correct trung vị |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in res:
            print(f"| {r['variant']} | {r['n']} | {r['top1']} ({r['top1_pct']:.1f}%) | {r['accepted_correct']} ({r['accepted_correct_pct']:.1f}%) | "
                  f"{r['wrong_accept']} | {r['uncertain']} | {r['rejected']} | {r['epenthesis']} | {r['d_correct_mean']:.3f} | {r['d_correct_median']:.3f} |")
        return

    if args.study == "grid":
        run_grid(args, matcher, words, meta, out_dir)
        return

    gaps = [float(x) for x in args.gaps.split(",")]
    sentences = make_sentences(words, args.n, args.seed)
    cache: Dict[Any, Any] = {}
    aggs = []
    all_rows: Dict[float, Any] = {}
    for g in gaps:
        a = run_level(sentences, g, args.fps, args.mode, matcher, fast_cfg, mp, args.seed, cache, keep_rows=True,
                      hold_dest=args.hold_dest, dup_frac=args.dup_frac, dist_str=args.transition_dist)
        all_rows[g] = a.pop("rows")
        aggs.append(a)
        print(f"  g={g:.2f}: câu đúng {100*a['acc']:.1f}% | từ đúng {100*a['word_acc']:.1f}% | mất {a['lost']} thừa {a['extra']} | "
              f"nhiễm dẫn đầu {a['leadin_contam']} ({100*a['leadin_contam_rate']:.1f}%) | mất đầu từ {a['head_loss_words']} ({100*a['head_loss_rate']:.1f}%) | "
              f"trễ TB {a['lat_median']:.3f}s / p95 {a['lat_p95']:.3f}s | hitch p95 {a['hitch_p95_ms']:.1f}ms")
    mw = min_wait(aggs)
    title = f"{'FAST' if fast_cfg else 'MỐC production'} | {args.fps:.0f} Hz | {args.mode} | {args.n} câu/mức"
    md = fmt_table(aggs, title) + f"\n\nThời gian chờ tối thiểu (g nhỏ nhất cho >= 90% câu đúng, mọi g lớn hơn cũng đạt): {mw if mw is not None else 'KHÔNG đạt ở mức nào'}\n"
    print("\n" + md)
    tag = f"{'fast' if fast_cfg else 'base'}_{args.fps:.0f}hz_{args.mode}" + ("_allstrokes" if args.all_strokes else "") + (f"_{args.label}" if args.label else "")
    with open(os.path.join(out_dir, f"table_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, f"result_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, aggregates=aggs, min_wait_sec=mw, elapsed_sec=time.time() - t_start), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, f"sentences_{tag}.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["g", "truth", "pred", "correct", "lost", "extra", "latency_s", "head_lost_frames", "outcomes",
                     "leadin_contam", "head_loss_words", "start_errors"])
        for g, rows in all_rows.items():
            for r in rows:
                wr.writerow([g, " ".join(r["truth"]), " ".join(r["pred"]), int(r["correct"]), r["lost"], r["extra"],
                             ";".join(f"{x:.3f}" for x in r["latency"]), ";".join(map(str, r["head_lost"])), ";".join(r["outcomes"]),
                             r.get("leadin_contam", 0), r.get("head_loss_words", 0),
                             ";".join(f"{x:.3f}" for x in r.get("start_errors", []))])
    print(f"[chain_replay] đã ghi {out_dir} ({time.time() - t_start:.0f} s)")


# =============================================================================
# 8b. GHI PHIÊN GIẢ LẬP DẠNG CSV --log-session
# =============================================================================
SESSION_HEADER = ["timestamp", "event", "predicted_word", "best_distance", "second_word", "second_distance", "margin_pct",
                  "duration_sec", "num_frames", "body_anchor_valid", "fps", "best_confidence", "match_quality", "r_ratio",
                  "t_unix", "commit_mode", "t_start"]


def emit_session(args: argparse.Namespace, matcher: LooMatcher, words: List[WordStroke], fast_cfg: Optional[Dict[str, Any]],
                 mp: Sequence[str]) -> None:
    """Chạy args.emit_n câu giả lập ở mức nghỉ args.emit_g rồi ghi CSV cùng định dạng predict_sentence_webcam --log-session."""
    sentences = make_sentences(words, args.emit_n, args.seed + 7)
    cache: Dict[Any, Any] = {}
    t_off = 1_790_000_000.0
    truth_lines: List[str] = []
    ev_map = {"ACCEPTED": "ACCEPTED", "UNCERTAIN": "UNCERTAIN", "REJECTED": "REJECTED", "duration_gate": "OUT_OF_DURATION"}
    with open(args.emit_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(SESSION_HEADER)
        for si, ws in enumerate(sentences):
            rng = np.random.default_rng([args.seed, 77, si])
            stream = build_stream(ws, [args.emit_g] * (len(ws) - 1), args.fps, args.mode, rng, cache=cache)
            matcher.excluded = tuple(sorted(matcher.global_index(w.bank_key) for w in ws if w.bank_key is not None))
            res = run_stream(stream, matcher, fast_cfg, mp)
            truth_lines.append(" ".join(w.word for w in ws))
            for st in res["strokes"]:
                ev = ev_map.get(st["outcome"])
                if ev is None:
                    continue
                t_unix = t_off + (st["t_emit"] - T0)
                wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_unix)), ev, st["top"] or "", 0.0, "", "", 0.0,
                             round(st["t1"] - st["t0"], 3), st["n"], True, args.fps, 0.0, 0.0, 0.0, round(t_unix, 3), st["via"], round(t_off + (st["t0"] - T0), 3)])
            t_off += float(stream.t[-1] - stream.t[0]) + 3.0     # nghỉ 3 s giữa hai câu (hạ tay > 1.8 s)
    with open(args.emit_csv + ".truth.txt", "w", encoding="utf-8") as f:
        f.write(NEWLINE.join(truth_lines) + NEWLINE)
    print(f"[chain_replay] đã ghi phiên giả lập {args.emit_csv} ({len(sentences)} câu) và {args.emit_csv}.truth.txt")


# =============================================================================
# 9. QUÉT LƯỚI (Nhiệm vụ D bước 3)
# =============================================================================
_W: Dict[str, Any] = {}


def _grid_init(bank: str, raw_dir: str, excl: List[str], tk: float, uk: float, n: int, seed: int, all_strokes: bool) -> None:
    ns = argparse.Namespace(bank=bank, raw_dir=raw_dir, exclude_session_prefix=excl, threshold_k=tk, uncertainty_k=uk,
                            all_strokes=bool(all_strokes), study=None)
    matcher, words, _n_map, _tk, _uk = setup(ns)
    _W["matcher"], _W["words"] = matcher, words
    _W["sentences"] = make_sentences(words, n, seed)
    _W["cache"] = {}
    _W["seed"] = seed


def _grid_task(job: Dict[str, Any]) -> Dict[str, Any]:
    cfg = job["cfg"]
    mp = multi_phase_for(_W["matcher"], cfg)
    a = run_level(_W["sentences"], job["g"], job["fps"], "hold", _W["matcher"], cfg, mp, _W["seed"], _W["cache"])
    a["cfg"] = cfg
    a["job"] = job["name"]
    return a


def run_grid(args: argparse.Namespace, matcher: LooMatcher, words: List[WordStroke], meta: Dict[str, Any], out_dir: str) -> None:
    base = dict(CONFIG["fast_chain"])
    base["enabled"] = True
    jobs = []
    for pause, grace, chain, q in itertools.product([0.30, 0.25, 0.20, 0.15, 0.12], [0.15, 0.10, 0.05], [0.0, 0.8, 1.2, 2.0], [70, 80, 90]):
        cfg = dict(base, pause_sec=pause, grace_window_sec=grace, chain_window_sec=chain, early_min_quality=q)
        jobs.append(dict(name=f"p{pause}_g{grace}_c{chain}_q{q}", cfg=cfg, g=0.20, fps=args.fps))
    print(f"[grid] {len(jobs)} cấu hình x g=0.20 x {args.fps:.0f} Hz x {args.n} câu, {args.workers} tiến trình")
    tk = args.threshold_k if args.threshold_k is not None else float(CONFIG.get("threshold_k", 1.6))
    uk = args.uncertainty_k if args.uncertainty_k is not None else float(CONFIG.get("uncertainty_k", 2.2))
    results = []
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_grid_init,
                                initargs=(args.bank, args.raw_dir, args.exclude_session_prefix, tk, uk, args.n, args.seed, args.all_strokes)) as ex:
        for i, r in enumerate(ex.map(_grid_task, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s")
    with open(os.path.join(out_dir, f"grid_{args.fps:.0f}hz.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, results=results), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, f"grid_{args.fps:.0f}hz.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["job", "pause", "grace", "chain", "early_q", "acc_pct", "lost", "extra", "lat_median", "early_n", "early_wrong", "mp_bad", "mp_words", "mp_rate_pct"])
        for r in results:
            c = r["cfg"]
            wr.writerow([r["job"], c["pause_sec"], c["grace_window_sec"], c["chain_window_sec"], c["early_min_quality"], f"{100*r['acc']:.1f}",
                         r["lost"], r["extra"], f"{r['lat_median']:.3f}", r["early_n"], r["early_wrong"], r["mp_bad"], r["mp_words"], f"{100*r['mp_rate']:.1f}"])
    print(f"[grid] xong {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
