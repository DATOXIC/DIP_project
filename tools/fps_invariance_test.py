"""
THÍ NGHIỆM NGOẠI TUYẾN: FPS LỆCH CÓ LÀM HỎNG NHẬN DIỆN KHÔNG? (FPS INVARIANCE TEST)
===================================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

MỤC ĐÍCH
--------
Nghi vấn: pipeline phụ thuộc FPS vì (i) các cột vận tốc dxs, dys, dangles là hiệu SỐ KHUNG
(per-frame) trước khi resample K=32, (ii) các ngưỡng phân đoạn (pause_frames, v_start_ratio,
max_stroke_frames...) tính bằng số khung. Script này ĐO điều đó bằng cách lấy các stroke thô
trong raw_strokes/, tạo các biến thể "cùng chuyển động vật lý, khác fps" rồi chạy lại đúng
pipeline dựng đặc trưng + so khớp + phân đoạn của dự án. KHÔNG cần camera. KHÔNG sửa dự án.

CÁC BIẾN THỂ (theo nhãn)
------------------------
  a_goc      : stroke gốc (chuẩn so sánh; kiểm chứng đường dựng khớp bank).
  b_60fps    : chèn khung nội suy tuyến tính giữa mọi cặp khung liền nhau (x2).
  c_15fps    : bỏ xen kẽ một khung (x0.5).
  d_trung    : mô phỏng driver lặp khung: khung thứ hai của mỗi cặp bị thay bằng bản sao khung liền trước
               (số lần đọc/giây vẫn như gốc, timestamp vẫn tăng đều, nhưng khung MỚI chỉ còn một nửa).
  e_45fps    : nội suy tỉ lệ 1.5.
  f_bo_trung : (PHỤ, ngoài yêu cầu) bỏ các khung trùng y hệt sẵn có trong stroke gốc.
  g_palm_predict : (PHỤ, ngoài yêu cầu) CÙNG khung, CÙNG fps nhưng palm_len tính lại bằng công thức của
               predict_sentence_webcam (recompute_palm_len) thay vì palm_len đã lưu bởi collect_dynamic_templates.
               Lý do: collect_dynamic_templates tính palm_3d với (lm9.y-lm0.y)*w_f (nhân bề RỘNG) còn predict nhân
               bề CAO => palm_len của bank lớn hơn ~1.0-1.33 lần; biến thể này đo tác động của độ lệch đó.

CÁCH LÀM ĐÚNG (không sao chép công thức)
-----------------------------------------
- Dựng đặc trưng 24D + resample K=32 + canonicalize: GỌI NGUYÊN VĂN rebuild_one_stroke(raw, use_stored_features=True)
  của rebuild_templates_from_raw.py (chính hàm dựng bank), qua adapter: biến thể => dict "raw" cùng schema.
  (Chế độ mặc định use_stored_features=False KHÔNG tái tạo được bank: palm_len tính lại khác palm_len collector đã lưu.)
  rebuild_one_stroke trả query ĐÃ canonical (một lần), nên match() được gọi với handedness="Right"
  (không lật lần hai).
- So khớp: GoldenTriadDTWMatcher dựng như run_sentence_builder (CONFIG hiện hành, sign_weights_path="",
  threshold_mode="scale_k"). LEAVE-ONE-OUT: với mỗi stroke có mẫu tương ứng trong bank, bank BỎ đúng mẫu đó
  (set_vocabulary), độ rộng nội lớp được tính lại theo bank đã bỏ. Khoảng cách DTW cặp-mẫu được
  memo hóa (chỉ để tăng tốc, không đổi kết quả).
- Phân đoạn: luồng khung của từng biến thể (có đệm tĩnh trước/sau) đi qua make_segmenter(CONFIG).
- Điều chỉnh chuẩn hóa thời gian (CHỈ THỬ NGHIỆM): nhân cột đạo hàm với (fps_thực/30). Do resample là nội suy
  tuyến tính theo cột nên nhân trước hay sau resample là như nhau => nhân trên chuỗi K=32.

CÁCH DÙNG
---------
    py fps_invariance_test.py                       # toàn bộ stroke
    py fps_invariance_test.py --limit 40 --seed 1   # tập con ngẫu nhiên có seed (chạy thử)
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import os
import sys
import io
import csv
import json
import glob
import time
import shutil
import argparse
import tempfile
import contextlib
import copy
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import dynamic_dtw_engine as dde
from dynamic_dtw_engine import (
    GoldenTriadDTWMatcher,
    is_stroke_duration_valid,
    is_movement_epenthesis,
)
from pipeline_common import make_segmenter
import rebuild_templates_from_raw as rtr
from predict_sentence_webcam import CONFIG

# ---------------------------------------------------------------------------------
# HẰNG SỐ
# ---------------------------------------------------------------------------------
BANK_PATH = "webcam_templates.json"
RAW_DIR = "raw_strokes"
FPS_REF = 30.0                  # fps chuẩn để chuẩn hóa đạo hàm (theo yêu cầu: nhân fps_thực/30)
COL_VEL = slice(2, 4)           # dxs, dys
COL_DANG = slice(12, 17)        # dangles (5 cột)
MAP_TOL = 2e-3                  # khớp chính xác stroke thô <-> mẫu bank (bank làm tròn 4 chữ số)
MAP_LOOSE = 0.30                # khớp gần đúng (vẫn coi là cùng stroke, chỉ để LOO)
PRE_ROLL_SEC = 0.8              # đệm tĩnh trước stroke khi dựng luồng phân đoạn
POST_ROLL_SEC = 1.5             # đệm tĩnh sau stroke (đủ để chốt cử chỉ)

VARIANTS = ["a_goc", "b_60fps", "c_15fps", "d_trung", "e_45fps", "f_bo_trung", "g_palm_predict"]
MODES = ["asis", "scale_vel", "scale_dang", "scale_both", "orig_deriv"]
E2E_MODES = ["asis", "scale_both"]
SEG_MODES = ["asis", "scaled"]   # asis = segmenter y hệt production; scaled = ngưỡng theo khung nhân/chia fps/30 (thử nghiệm)

MODE_DESC = {
    "asis": "nguyên trạng (pipeline hiện tại)",
    "scale_vel": "nhân dxs,dys x fps/30",
    "scale_dang": "nhân dangles x fps/30",
    "scale_both": "nhân cả hai nhóm x fps/30",
    "orig_deriv": "cận trên: thay đạo hàm bằng của bản gốc (a)",
}


# ---------------------------------------------------------------------------------
# TIỆN ÍCH
# ---------------------------------------------------------------------------------
@contextlib.contextmanager
def im_lang():
    """Chặn print của các hàm dự án (cảnh báo schema, [Chirality]...) để log gọn."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Khoảng tin cậy Wilson 95% cho tỉ lệ k/n (đơn vị %)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def stroke_rate(frames: List[Dict[str, Any]]) -> float:
    """Tốc độ lấy mẫu (khung/giây) tính từ timestamp của stroke: (n-1)/(t_cuối - t_đầu)."""
    if len(frames) < 2:
        return FPS_REF
    dur = float(frames[-1]["timestamp"] - frames[0]["timestamp"])
    if dur <= 1e-6:
        return FPS_REF
    return (len(frames) - 1) / dur


# ---------------------------------------------------------------------------------
# SINH BIẾN THỂ FPS (adapter: trả về dict "raw" cùng schema để đưa vào rebuild_one_stroke)
# ---------------------------------------------------------------------------------
_NUMERIC_KEYS = ("landmarks_px", "landmarks_norm", "palm_len", "wrist_raw", "wrist_smoothed",
                 "flexion_angles", "palm_normal")


def _lerp_frame(fa: Dict[str, Any], fb: Dict[str, Any], frac: float) -> Dict[str, Any]:
    """Khung nội suy giữa fa và fb (frac in (0,1)); trường số nội suy, nhãn rời rạc lấy khung gần nhất."""
    out = copy.deepcopy(fa if frac < 0.5 else fb)
    out["timestamp"] = float(fa["timestamp"] + frac * (fb["timestamp"] - fa["timestamp"]))
    for k in _NUMERIC_KEYS:
        if k in fa and k in fb and fa[k] is not None and fb[k] is not None:
            va = np.asarray(fa[k], dtype=np.float64)
            vb = np.asarray(fb[k], dtype=np.float64)
            out[k] = (va + frac * (vb - va)).tolist() if va.ndim > 0 else float(va + frac * (vb - va))
    if "palm_normal" in out and out["palm_normal"] is not None:
        nv = np.asarray(out["palm_normal"], dtype=np.float64)
        nn = float(np.linalg.norm(nv))
        if nn > 1e-9:
            out["palm_normal"] = (nv / nn).tolist()          # pháp tuyến chuẩn hóa lại
    fba, fbb = fa.get("face_box"), fb.get("face_box")
    if fba is not None and fbb is not None:
        a = np.asarray(fba, dtype=np.float64)
        b = np.asarray(fbb, dtype=np.float64)
        out["face_box"] = (a + frac * (b - a)).tolist()
    else:
        out["face_box"] = copy.deepcopy(fba if frac < 0.5 else fbb)
    return out


def resample_by_index(frames: List[Dict[str, Any]], ratio: float) -> List[Dict[str, Any]]:
    """
    Lấy mẫu lại chuỗi khung theo chỉ số: khung mới j nằm ở chỉ số gốc u = j / ratio.
    ratio=2 => chèn khung giữa mọi cặp (60fps); ratio=0.5 => bỏ xen kẽ (15fps); ratio=1.5 => 45fps.
    Tại chỉ số nguyên lấy nguyên khung gốc (không nội suy).
    """
    n = len(frames)
    m = int(np.floor((n - 1) * ratio + 1e-9)) + 1
    out: List[Dict[str, Any]] = []
    for j in range(m):
        u = j / ratio
        i0 = int(np.floor(u + 1e-9))
        frac = u - i0
        if frac < 1e-6 or i0 + 1 >= n:
            out.append(copy.deepcopy(frames[min(i0, n - 1)]))
        else:
            out.append(_lerp_frame(frames[i0], frames[i0 + 1], frac))
    return out


def variant_duplicate(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Khung chỉ số lẻ = bản sao NỘI DUNG khung liền trước (giữ timestamp riêng, tăng đều như cũ)."""
    out = []
    for i, f in enumerate(frames):
        if i % 2 == 1:
            g = copy.deepcopy(out[i - 1])
            g["timestamp"] = f["timestamp"]
            out.append(g)
        else:
            out.append(copy.deepcopy(f))
    return out


def variant_dedup(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bỏ các khung có landmarks_px y hệt khung liền trước (khung trùng sẵn có do đọc lặp)."""
    out = [copy.deepcopy(frames[0])]
    for prev, cur in zip(frames[:-1], frames[1:]):
        same = np.abs(np.asarray(prev["landmarks_px"]) - np.asarray(cur["landmarks_px"])).max() < 1e-3
        if not same:
            out.append(copy.deepcopy(cur))
    return out


def make_variant(raw: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Dựng dict raw cho biến thể `name`; giữ nguyên metadata, thay frames."""
    fr = raw["frames"]
    if name == "a_goc":
        nf = copy.deepcopy(fr)
    elif name == "b_60fps":
        nf = resample_by_index(fr, 2.0)
    elif name == "c_15fps":
        nf = resample_by_index(fr, 0.5)
    elif name == "d_trung":
        nf = variant_duplicate(fr)
    elif name == "e_45fps":
        nf = resample_by_index(fr, 1.5)
    elif name == "f_bo_trung":
        nf = variant_dedup(fr)
    elif name == "g_palm_predict":
        nf = copy.deepcopy(fr)
        for f in nf:
            f["palm_len"] = float(rtr.recompute_palm_len(np.asarray(f["landmarks_px"], dtype=np.float32)))
    else:
        raise ValueError(name)
    v = {k: val for k, val in raw.items() if k != "frames"}
    v["frames"] = nf
    v["stroke_duration_sec"] = float(nf[-1]["timestamp"] - nf[0]["timestamp"]) if len(nf) > 1 else 0.0
    return v


# ---------------------------------------------------------------------------------
# DỰNG ĐẶC TRƯNG: dùng NGUYÊN VĂN rebuild_one_stroke
# ---------------------------------------------------------------------------------
def build_traj(raw_like: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Trả về dict mẫu của rebuild_one_stroke (có _traj_full canonical) hoặc None nếu quá ngắn/lỗi."""
    if len(raw_like.get("frames", [])) < 8:
        return None
    with im_lang():
        try:
            # use_stored_features=True: dùng palm_len / góc / pháp tuyến ĐÃ LƯU (đúng thứ collector đã dùng để
            # ghi bank). Chế độ mặc định của rebuild (tính lại palm_len bằng recompute_palm_len) KHÔNG tái tạo được
            # bank vì công thức palm_len của collector khác của predict (xem biến thể g_palm_predict).
            return rtr.rebuild_one_stroke(raw_like, use_stored_features=True)
        except Exception:
            return None


def apply_mode(traj: np.ndarray, traj_a: Optional[np.ndarray], factor: float, mode: str) -> Optional[np.ndarray]:
    t = traj.copy()
    if mode == "asis":
        return t
    if mode == "scale_vel":
        t[:, COL_VEL] *= factor
    elif mode == "scale_dang":
        t[:, COL_DANG] *= factor
    elif mode == "scale_both":
        t[:, COL_VEL] *= factor
        t[:, COL_DANG] *= factor
    elif mode == "orig_deriv":
        if traj_a is None:
            return None
        t[:, COL_VEL] = traj_a[:, COL_VEL]
        t[:, COL_DANG] = traj_a[:, COL_DANG]
    else:
        raise ValueError(mode)
    return t


# ---------------------------------------------------------------------------------
# MATCHER (LEAVE-ONE-OUT) — dựng giống run_sentence_builder
# ---------------------------------------------------------------------------------
_MEMO: Dict[Any, float] = {}
_ORIG_DTW = dde.compute_golden_triad_dtw_distance


def _memo_dtw(**kw):
    """Bọc compute_golden_triad_dtw_distance: nhớ kết quả theo (2 chuỗi, trọng số) — chỉ tăng tốc."""
    key = (kw["q_traj"].tobytes(), kw["t_traj"].tobytes(),
           kw["w_p"], kw["w_a"], kw["w_m"], kw["w_n"], kw.get("w_b", 0.0), kw.get("w_z", 0.0),
           kw.get("w_s", 0.0), kw.get("w_r", 0.0), kw.get("lambda_shape", 0.0), kw.get("lambda_shape_global", 0.0))
    v = _MEMO.get(key)
    if v is None:
        v = _ORIG_DTW(**kw)
        _MEMO[key] = v
    return v


def build_matcher(bank: Dict[str, Any], cache_dir: str) -> GoldenTriadDTWMatcher:
    """GoldenTriadDTWMatcher với đúng tham số CONFIG mà predict_sentence_webcam.run_sentence_builder dùng."""
    with im_lang():
        m = GoldenTriadDTWMatcher(
            templates_path=BANK_PATH,
            sign_weights_path="",
            w_p=CONFIG["w_wrist_pos"],
            w_a=CONFIG["w_finger_angles"],
            w_m=CONFIG["w_wrist_dir"],
            w_n=CONFIG["w_palm_normal"],
            lambda_shape=CONFIG["lambda_shape"],
            margin_threshold=CONFIG["margin_threshold"],
            uncertainty_ratio=CONFIG["uncertainty_ratio"],
            threshold_override=None,               # scale_k => run_sentence_builder truyền None
            asllex_blend_alpha=CONFIG.get("asllex_blend_alpha", 1.0),
            asllex_max_deviation=CONFIG.get("asllex_max_deviation", None),
            w_z=CONFIG.get("w_z", 0.0),
            w_b=CONFIG.get("w_b", 0.0),
            w_s=CONFIG.get("w_s", 0.0),
            w_r=CONFIG.get("w_r", 0.0),
            threshold_mode=CONFIG.get("threshold_mode", "scale_k"),
            threshold_k=float(CONFIG.get("threshold_k", 1.6)),
            uncertainty_k=float(CONFIG.get("uncertainty_k", 2.2)),
            cache_dir=cache_dir,
        )
    return m


def vocab_without(bank_vocab: Dict[str, Any], word: str, idx: int) -> Dict[str, Any]:
    """Bản bank bỏ đúng mẫu (word, idx) — các từ khác giữ nguyên tham chiếu (set_vocabulary sẽ deepcopy)."""
    v = dict(bank_vocab)
    entry = dict(bank_vocab[word])
    entry["samples"] = [s for i, s in enumerate(bank_vocab[word]["samples"]) if i != idx]
    v[word] = entry
    return v


def do_match(matcher: GoldenTriadDTWMatcher, traj: np.ndarray, body_ok: bool, truth: str) -> Dict[str, Any]:
    """So khớp một query canonical; trả về các số đo."""
    with im_lang():
        r = matcher.match(traj, ["Unknown"] * traj.shape[0], handedness="Right", query_has_body=bool(body_ok))
    dist = {c.word_key: c.distance for c in r.candidates}
    d_true = dist.get(truth, float("nan"))
    others = [d for w, d in dist.items() if w != truth]
    d_wrong = min(others) if others else float("nan")
    cands = sorted(r.candidates, key=lambda c: c.distance)
    marg_ratio = ((cands[1].distance - cands[0].distance) / max(cands[0].distance, 1e-4)) if len(cands) >= 2 else float("nan")
    return {
        "status": r.status, "top_word": r.top_word, "ok": bool(r.top_word == truth),
        "d_true": d_true, "d_wrong": d_wrong, "margin": d_wrong - d_true, "margin_ratio": marg_ratio,
        "d_top1": r.best_distance,
    }


# ---------------------------------------------------------------------------------
# PHÂN ĐOẠN
# ---------------------------------------------------------------------------------
def build_stream(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Luồng khung liên tục cho segmenter: đệm tĩnh trước (tay đứng yên đủ để vào READY) + stroke + đệm tĩnh sau.
    Điểm đệm trước = lùi một bước theo vận tốc khung đầu, để khung 0 của stroke là khung đầu tiên có chuyển động
    (mô phỏng cách stroke gốc bắt đầu). Đệm sau đứng yên tại khung cuối để segmenter chốt cử chỉ.
    """
    n = len(frames)
    t0, t1 = float(frames[0]["timestamp"]), float(frames[-1]["timestamp"])
    dt = (t1 - t0) / max(n - 1, 1)
    dt = max(dt, 1e-3)
    w = np.array([f["wrist_smoothed"] for f in frames], dtype=np.float64)
    p_prev = w[0] - (w[1] - w[0]) if n > 1 else w[0]
    n_pre = int(np.ceil(PRE_ROLL_SEC / dt))
    n_post = int(np.ceil(POST_ROLL_SEC / dt))
    stream: List[Dict[str, Any]] = []
    for k in range(n_pre):
        g = copy.deepcopy(frames[0])
        g["wrist_smoothed"] = p_prev.tolist()
        g["timestamp"] = t0 - dt * (n_pre - k)
        g["_synthetic"] = True
        stream.append(g)
    n_pre_real = len(stream)
    for f in frames:
        g = copy.deepcopy(f)
        g["_synthetic"] = False
        stream.append(g)
    for k in range(n_post):
        g = copy.deepcopy(frames[-1])
        g["timestamp"] = t1 + dt * (k + 1)
        g["_synthetic"] = True
        stream.append(g)
    return stream


def run_segmenter(stream: List[Dict[str, Any]], scale: Optional[float] = None) -> Dict[str, Any]:
    """
    Đưa luồng qua make_segmenter(CONFIG) đúng như vòng lặp production (segmenter.update mỗi khung).
    scale=None: cấu hình y hệt production. scale=r (r = fps_thực/30): THỬ NGHIỆM chuẩn hóa thời gian cho các
    ngưỡng tính bằng số khung — v_start/v_stop chia r (vận tốc mỗi khung tỉ lệ 1/fps), pause_frames,
    ready_stable_frames, max_stroke_frames nhân r; min_stroke_frames nhân r nhưng sàn 8 (resample_9d_gesture_sequence
    cũng đòi >= 8 khung nên không thể thấp hơn). Chỉ đổi thuộc tính của thực thể segmenter trong script này.
    Ngưỡng swipe KHÔNG chuẩn hóa (cửa sổ 6 khung cố định trong mã segmenter).
    """
    seg = make_segmenter(CONFIG)
    if scale is not None:
        r = float(scale)
        seg.v_start_ratio = seg.v_start_ratio / r
        seg.v_stop_ratio = seg.v_stop_ratio / r
        seg.pause_frames = max(1, int(round(seg.pause_frames * r)))
        seg.ready_stable_frames = max(2, int(round(seg.ready_stable_frames * r)))
        seg.max_stroke_frames = max(8, int(round(seg.max_stroke_frames * r)))
        seg.min_stroke_frames = max(8, int(round(seg.min_stroke_frames * r)))
    emitted: List[List[Any]] = []
    n_swipe = 0
    for i, f in enumerate(stream):
        wx, wy = f["wrist_smoothed"]
        _, done, is_swipe, _ = seg.update(
            wrist_x=float(wx), wrist_y=float(wy), palm_len=float(f["palm_len"]),
            hand_shape=str(i), timestamp=float(f["timestamp"]),
        )
        if is_swipe:
            n_swipe += 1
        if done is not None:
            emitted.append(done)
    return {"emitted": emitted, "n_swipe": n_swipe}


# ---------------------------------------------------------------------------------
# NẠP DỮ LIỆU
# ---------------------------------------------------------------------------------
def load_raw_strokes(raw_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for fp in sorted(glob.glob(os.path.join(raw_dir, "*", "*.json"))):
        if os.sep + "NEGATIVE" + os.sep in fp or "/NEGATIVE/" in fp.replace("\\", "/"):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue
        wk = raw.get("word_key")
        if not wk or wk == "__NEG__":
            continue
        if len(raw.get("frames", [])) < 8:
            continue
        out.append((fp, raw))
    return out


def map_to_bank(traj: np.ndarray, word: str, dur: float, bank_vocab: Dict[str, Any]) -> Tuple[Optional[int], float]:
    """
    Ghép stroke thô với mẫu bank: ứng viên = mẫu cùng từ có duration_sec khớp (|chênh| < 1.5e-3, bank làm tròn 3 số);
    chọn mẫu có sai số quỹ đạo max nhỏ nhất. Trả (chỉ số, sai số) hoặc (None, inf) nếu không có ứng viên.
    Sai số < MAP_TOL = khớp chính xác; nhỏ hơn MAP_LOOSE = khớp gần đúng (stroke có khung LK-fallback làm lệch chỉ số
    lịch sử) — cả hai đều bị LOO để không rò rỉ mẫu tự nó.
    """
    best_i, best_e = None, float("inf")
    for i, s in enumerate(bank_vocab.get(word, {}).get("samples", [])):
        if abs(float(s.get("duration_sec", -1)) - dur) >= 1.5e-3:
            continue
        b = np.asarray(s.get("traj_features_9d") or s.get("traj_features"), dtype=np.float64)
        if b.shape != traj.shape:
            continue
        e = float(np.max(np.abs(b - traj)))
        if e < best_e:
            best_i, best_e = i, e
    if best_i is not None and best_e < MAP_LOOSE:
        return best_i, best_e
    return None, best_e


def raw_timing_stats(strokes: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Thống kê nhịp thời gian của KHO THÔ theo phiên thu (session_id), để soi giả thuyết "khung trùng / đọc dồn":
      - tốc độ đọc   = 1 / mean(dt)            (số lần cap.read() xử lý mỗi giây, đúng nghĩa)
      - fps_hiển_thị = trung bình của 1/dt     (cái mà EMA fps = 0.9*fps + 0.1*(1/dt) hội tụ tới; lệch CAO khi dt dồn cục)
      - tỉ lệ khung trùng = cặp khung liền nhau có landmarks_px y hệt (cùng một ảnh camera bị đọc/xử lý hai lần)
      - fps khung MỚI = tốc độ đọc x (1 - tỉ lệ trùng)
    """
    by: Dict[str, Dict[str, Any]] = {}
    for _, raw in strokes:
        sid = str(raw.get("session_id", "unknown"))
        fr = raw["frames"]
        ts = np.array([f["timestamp"] for f in fr], dtype=np.float64)
        dts = np.diff(ts)
        d = by.setdefault(sid, {"dts": [], "dup": 0, "pairs": 0, "n_strokes": 0, "sum_dt": 0.0, "n_dt": 0})
        d["n_strokes"] += 1
        d["dts"].extend(dts.tolist())
        d["sum_dt"] += float(dts.sum())
        d["n_dt"] += len(dts)
        for a, b in zip(fr[:-1], fr[1:]):
            d["pairs"] += 1
            d["dup"] += int(np.abs(np.asarray(a["landmarks_px"]) - np.asarray(b["landmarks_px"])).max() < 1e-3)
    out = []
    for sid, d in sorted(by.items()):
        dts = np.array(d["dts"])
        dts = dts[dts > 1e-4]
        read_rate = d["n_dt"] / d["sum_dt"] if d["sum_dt"] > 0 else float("nan")
        dup = d["dup"] / max(d["pairs"], 1)
        out.append({
            "session": sid, "n_strokes": d["n_strokes"], "n_pairs": d["pairs"],
            "read_rate_fps": read_rate, "mean_inv_dt_fps": float(np.mean(1.0 / dts)),
            "dup_frac": dup, "new_frame_fps": read_rate * (1 - dup),
            "dt_p05_ms": 1000 * float(np.percentile(dts, 5)), "dt_p50_ms": 1000 * float(np.percentile(dts, 50)),
            "dt_p95_ms": 1000 * float(np.percentile(dts, 95)), "frac_dt_lt_15ms": float(np.mean(dts < 0.015)),
        })
    return out


# ---------------------------------------------------------------------------------
# THỐNG KÊ & BÁO CÁO
# ---------------------------------------------------------------------------------
def agg_match(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    valid = [r for r in rows if r["status"] != "NOFEAT"]
    dt = np.array([r["d_true"] for r in valid], dtype=float)
    mg = np.array([r["margin"] for r in valid], dtype=float)
    st = {s: sum(1 for r in rows if r["status"] == s) for s in ("ACCEPTED", "UNCERTAIN", "REJECTED", "NOFEAT")}
    lo, hi = wilson(ok, n)
    acc_ok = sum(1 for r in rows if r["ok"] and r["status"] == "ACCEPTED")
    return {
        "n": n, "top1": 100.0 * ok / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi,
        "d_mean": float(np.nanmean(dt)) if len(dt) else float("nan"),
        "d_med": float(np.nanmedian(dt)) if len(dt) else float("nan"),
        "m_mean": float(np.nanmean(mg)) if len(mg) else float("nan"),
        "m_med": float(np.nanmedian(mg)) if len(mg) else float("nan"),
        "acc": 100.0 * st["ACCEPTED"] / n if n else float("nan"),
        "unc": 100.0 * st["UNCERTAIN"] / n if n else float("nan"),
        "rej": 100.0 * st["REJECTED"] / n if n else float("nan"),
        "nofeat": st["NOFEAT"],
        "acc_correct": 100.0 * acc_ok / n if n else float("nan"),
    }


def fmt_row_match(label: str, a: Dict[str, Any]) -> str:
    return (f"| {label} | {a['n']} | {a['top1']:.1f}% [{a['ci_lo']:.0f}-{a['ci_hi']:.0f}] | "
            f"{a['d_mean']:.3f} / {a['d_med']:.3f} | {a['m_mean']:+.3f} / {a['m_med']:+.3f} | "
            f"{a['acc']:.0f} / {a['unc']:.0f} / {a['rej']:.0f} | {a['acc_correct']:.1f}% | {a['nofeat']} |")


MATCH_HEADER = ("| Nhóm | N | top-1 [CI95 Wilson] | d_đúng mean/med | margin (d_sai_min - d_đúng) mean/med | "
                "ACC/UNC/REJ (%) | ACCEPTED&đúng | không dựng được |\n"
                "|---|---|---|---|---|---|---|---|")


def main() -> None:
    ap = argparse.ArgumentParser(description="Thi nghiem ngoai tuyen: FPS lech co lam hong nhan dien khong.")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--bank", default=BANK_PATH)
    ap.add_argument("--limit", type=int, default=0, help="Chi chay tap con ngau nhien co seed (0 = tat ca stroke).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("results", f"fps_invariance_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = tempfile.mkdtemp(prefix="fpsinv_cache_")

    print("=" * 90)
    print("  THÍ NGHIỆM ĐỘ BẤT BIẾN THEO FPS (ngoại tuyến, không cần camera)")
    print("=" * 90)

    with open(args.bank, "r", encoding="utf-8") as f:
        bank = json.load(f)
    bank_vocab = bank["vocabulary"]
    bank_meta = bank.get("metadata", {})
    n_bank = sum(len(w["samples"]) for w in bank_vocab.values())
    print(f"  Bank: {args.bank} — {len(bank_vocab)} từ, {n_bank} mẫu, feature_dim={bank_meta.get('feature_dim')}")

    strokes = load_raw_strokes(args.raw_dir)
    print(f"  Stroke thô (bỏ NEGATIVE): {len(strokes)}")

    # ---- Bước 1: dựng bản gốc (a) bằng rebuild_one_stroke, ghép với bank ----
    items: List[Dict[str, Any]] = []
    for fp, raw in strokes:
        s = build_traj(raw)
        if s is None:
            continue
        word = raw["word_key"]
        dur = float(raw.get("stroke_duration_sec") or 0.0)
        idx, err = map_to_bank(s["_traj_full"], word, dur, bank_vocab)
        items.append({"path": fp, "raw": raw, "word": word, "bank_idx": idx, "map_err": err, "sample_a": s})
    n_in = sum(1 for it in items if it["bank_idx"] is not None)
    n_exact = sum(1 for it in items if it["bank_idx"] is not None and it["map_err"] < MAP_TOL)
    print(f"  Dựng được {len(items)} stroke; ghép được với mẫu trong bank: {n_in} "
          f"(chính xác {n_exact}, gần đúng {n_in - n_exact}); ngoài bank: {len(items) - n_in}")
    if n_in == 0:
        print("[!] DỪNG: không stroke thô nào khớp mẫu trong bank => công thức dựng đặc trưng KHÔNG khớp bank.")
        sys.exit(3)

    # ---- Bước 2: kiểm chứng (a) trên CHÍNH bank (không LOO): phải top-1 gần tuyệt đối ----
    dde.compute_golden_triad_dtw_distance = _memo_dtw
    matcher = build_matcher(bank, cache_dir)
    with im_lang():
        matcher.set_vocabulary(bank_vocab, bank_meta)
    chk = []
    for it in items:
        if it["bank_idx"] is None or it["map_err"] >= MAP_TOL:
            continue
        chk.append(do_match(matcher, it["sample_a"]["_traj_full"], it["sample_a"]["body_anchor_valid"], it["word"]))
    if not chk:
        print("[!] DỪNG: không có stroke nào khớp chính xác bank để kiểm chứng biến thể (a).")
        sys.exit(3)
    chk_ok = 100.0 * sum(1 for r in chk if r["ok"]) / len(chk)
    chk_d = float(np.median([r["d_true"] for r in chk]))
    print(f"  KIỂM CHỨNG (a) trên chính bank, KHÔNG LOO: top-1 = {chk_ok:.1f}% | d_đúng trung vị = {chk_d:.5f} (kỳ vọng ~100% và ~0)")
    if chk_ok < 95.0 or chk_d > 1e-2:
        print("[!] DỪNG: biến thể (a) không tái tạo bank (top-1 < 95% hoặc d_đúng lớn). "
              "Nghĩa là dựng query từ raw KHÔNG cùng không gian với bank (chirality/công thức) — kết quả sau đó vô nghĩa.")
        sys.exit(3)

    # ---- Bước 3: chọn tập chạy ----
    rng = np.random.RandomState(args.seed)
    sel = list(range(len(items)))
    if args.limit and args.limit < len(items):
        sel = sorted(rng.choice(len(items), size=args.limit, replace=False).tolist())
        print(f"  CHẠY TẬP CON NGẪU NHIÊN: {len(sel)}/{len(items)} stroke (seed={args.seed})")
    words_cov = sorted({items[i]["word"] for i in sel})
    print(f"  Số từ được phủ: {len(words_cov)}")

    timing = raw_timing_stats(strokes)
    match_rows: List[Dict[str, Any]] = []
    seg_rows: List[Dict[str, Any]] = []
    e2e_rows: List[Dict[str, Any]] = []

    # Tách: stroke ngoài bank dùng bank đầy đủ (một lần set_vocabulary); stroke trong bank dùng LOO.
    order = [i for i in sel if items[i]["bank_idx"] is not None] + [i for i in sel if items[i]["bank_idx"] is None]
    full_state_set = False
    t_start = time.time()
    for pos, i in enumerate(order):
        it = items[i]
        word, raw = it["word"], it["raw"]
        in_bank = it["bank_idx"] is not None
        with im_lang():
            if in_bank:
                matcher.set_vocabulary(vocab_without(bank_vocab, word, it["bank_idx"]), bank_meta)
                full_state_set = False
            elif not full_state_set:
                matcher.set_vocabulary(bank_vocab, bank_meta)
                full_state_set = True
        if pos % 10 == 0:
            el = time.time() - t_start
            print(f"  [{pos + 1}/{len(order)}] {os.path.basename(os.path.dirname(it['path']))}/{os.path.basename(it['path'])} "
                  f"(trong bank={in_bank}) — {el:.0f}s")

        traj_a = it["sample_a"]["_traj_full"]
        seg_a_dur = None
        for vname in VARIANTS:
            vraw = make_variant(raw, vname)
            vfr = vraw["frames"]
            rate = stroke_rate(vfr)
            factor = rate / FPS_REF
            samp = it["sample_a"] if vname == "a_goc" else build_traj(vraw)

            # ---- (1) so khớp theo từng chế độ cột ----
            for mode in MODES:
                base = {"stroke": os.path.relpath(it["path"]).replace("\\", "/"), "word": word, "in_bank": in_bank,
                        "variant": vname, "mode": mode, "n_frames": len(vfr), "rate_fps": round(rate, 2),
                        "factor": round(factor, 4)}
                if samp is None:
                    match_rows.append({**base, "status": "NOFEAT", "top_word": "", "ok": False, "d_true": np.nan,
                                       "d_wrong": np.nan, "margin": np.nan, "margin_ratio": np.nan, "d_top1": np.nan})
                    continue
                tq = apply_mode(samp["_traj_full"], traj_a, factor, mode)
                if tq is None:
                    continue
                m = do_match(matcher, tq, samp["body_anchor_valid"], word)
                match_rows.append({**base, **m})

            # ---- (2)+(3) phân đoạn (nguyên trạng và có chuẩn hóa thời gian) -> đầu-cuối ----
            stream = build_stream(vfr)
            t_orig0, t_orig1 = float(vfr[0]["timestamp"]), float(vfr[-1]["timestamp"])
            orig_dur = t_orig1 - t_orig0
            rel = os.path.relpath(it["path"]).replace(os.sep, "/")
            for seg_mode in SEG_MODES:
                sr = run_segmenter(stream, scale=(None if seg_mode == "asis" else factor))
                em = sr["emitted"]
                durs = [float(e[-1].timestamp - e[0].timestamp) for e in em]
                gate_ok = [bool(is_stroke_duration_valid(d) and not is_movement_epenthesis(e)) for d, e in zip(durs, em)]
                main_i = int(np.argmax(durs)) if durs else None
                main_dur = durs[main_i] if durs else float("nan")
                if vname == "a_goc" and seg_mode == "asis":
                    seg_a_dur = main_dur
                cat = "nhỡ" if len(em) == 0 else ("đúng 1" if len(em) == 1 else "cắt đôi")
                start_delay = float(em[main_i][0].timestamp - t_orig0) if durs else float("nan")
                end_delta = float(em[main_i][-1].timestamp - t_orig1) if durs else float("nan")
                seg_rows.append({
                    "stroke": rel, "word": word, "in_bank": in_bank,
                    "variant": vname, "seg_mode": seg_mode, "n_frames_in": len(vfr), "orig_dur_sec": round(orig_dur, 3),
                    "n_emitted": len(em), "cat": cat, "n_swipe": sr["n_swipe"],
                    "main_dur_sec": round(main_dur, 3) if durs else np.nan,
                    "main_frames": len(em[main_i]) if durs else 0,
                    "dur_vs_a": (main_dur / seg_a_dur) if (durs and seg_a_dur and np.isfinite(seg_a_dur) and seg_a_dur > 0) else np.nan,
                    "start_delay_sec": round(start_delay, 3) if durs else np.nan,
                    "end_delta_sec": round(end_delta, 3) if durs else np.nan,
                    "n_pass_gate": int(sum(gate_ok)),
                    "orig_frames_over_cap": int(len(vfr) > CONFIG_MAX_STROKE_FRAMES),
                })

                for mode in E2E_MODES:
                    words_committed: List[str] = []
                    n_built = 0
                    for e, g in zip(em, gate_ok):
                        if not g:
                            continue
                        idxs = [int(fr.hand_shape) for fr in e]
                        sub = {k: val for k, val in raw.items() if k != "frames"}
                        sub["frames"] = [stream[j] for j in idxs]
                        sub["stroke_duration_sec"] = float(sub["frames"][-1]["timestamp"] - sub["frames"][0]["timestamp"])
                        ss = build_traj(sub)
                        if ss is None:
                            continue
                        n_built += 1
                        fct = stroke_rate(sub["frames"]) / FPS_REF
                        tq = apply_mode(ss["_traj_full"], traj_a, fct, mode)
                        mm = do_match(matcher, tq, ss["body_anchor_valid"], word)
                        if mm["status"] == "ACCEPTED" and mm["top_word"]:
                            words_committed.append(mm["top_word"])
                    e2e_rows.append({
                        "stroke": rel, "word": word, "in_bank": in_bank,
                        "variant": vname, "seg_mode": seg_mode, "mode": mode, "n_emitted": len(em),
                        "n_pass_gate": int(sum(gate_ok)),
                        "n_matched": n_built, "committed": "|".join(words_committed),
                        "n_committed": len(words_committed),
                        "n_wrong_words": int(sum(1 for w in words_committed if w != word)),
                        "success": bool(words_committed == [word]),
                        "any_correct": bool(word in words_committed),
                    })

    print(f"\n  Xong {len(order)} stroke trong {time.time() - t_start:.0f}s.")
    dde.compute_golden_triad_dtw_distance = _ORIG_DTW
    shutil.rmtree(cache_dir, ignore_errors=True)

    # ---------------------------------------------------------------- CSV
    def write_csv(name: str, rows: List[Dict[str, Any]]) -> str:
        p = os.path.join(out_dir, name)
        if not rows:
            return p
        keys = list(rows[0].keys())
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return p

    write_csv("match_results.csv", match_rows)
    write_csv("segmentation.csv", seg_rows)
    write_csv("end_to_end.csv", e2e_rows)

    # ---------------------------------------------------------------- BÁO CÁO
    def sub(rows, **kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    L: List[str] = []
    L.append(f"# Báo cáo thí nghiệm bất biến FPS — {ts}")
    L.append("")
    L.append(f"- Bank: `{args.bank}` ({len(bank_vocab)} từ, {n_bank} mẫu). Stroke thô dùng: {len(order)} "
             f"(trong bank/LOO: {sum(1 for i in order if items[i]['bank_idx'] is not None)}, "
             f"ngoài bank: {sum(1 for i in order if items[i]['bank_idx'] is None)}). Seed={args.seed}, limit={args.limit}.")
    L.append(f"- Kiểm chứng (a) trên chính bank, không LOO: top-1 = {chk_ok:.1f}%, d_đúng trung vị = {chk_d:.5f}.")
    L.append(f"- Ngưỡng/khoảng cách: matcher như run_sentence_builder (threshold_mode={CONFIG.get('threshold_mode')}, "
             f"k={CONFIG.get('threshold_k')}, unc_k={CONFIG.get('uncertainty_k')}, w_b={CONFIG.get('w_b')}).")
    L.append("- Mọi số ở bảng chính là trên stroke CÓ mẫu trong bank, bank đã loại đúng mẫu đó (leave-one-out).")
    L.append("")
    L.append("## 0. Nhịp thời gian của kho thô (toàn bộ stroke thô, bỏ NEGATIVE)")
    L.append("")
    L.append("| Phiên | Số stroke | Số cặp khung | Tốc độ đọc 1/mean(dt) | \"FPS hiển thị\" mean(1/dt) | Tỉ lệ khung TRÙNG | FPS khung MỚI | dt p05/p50/p95 (ms) | dt<15ms |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in timing:
        L.append(f"| {r['session']} | {r['n_strokes']} | {r['n_pairs']} | {r['read_rate_fps']:.1f} | {r['mean_inv_dt_fps']:.1f} | "
                 f"{100*r['dup_frac']:.1f}% | {r['new_frame_fps']:.1f} | {r['dt_p05_ms']:.0f}/{r['dt_p50_ms']:.0f}/{r['dt_p95_ms']:.0f} | {100*r['frac_dt_lt_15ms']:.0f}% |")
    L.append("")
    L.append("`FPS hiển thị` mô phỏng đại lượng mà EMA `fps = 0.9*fps + 0.1*(1/dt)` của predict hội tụ tới: khi các lần đọc dồn cục "
             "(dt ngắn xen dt dài) nó LỚN HƠN tốc độ đọc thật, và khi khung bị đọc trùng thì FPS khung MỚI còn nhỏ hơn nữa.")
    L.append("")

    for sect, flt, title in (("in", True, "Stroke trong bank (LOO) — BẢNG CHÍNH"),
                              ("out", False, "Stroke ngoài bank (bank đầy đủ, không cần LOO) — tham khảo")):
        rows_s = [r for r in match_rows if r["in_bank"] == flt]
        if not rows_s:
            continue
        L.append(f"## 1. So khớp theo fps — {title}")
        L.append("")
        L.append("### 1a. Chế độ `asis` (pipeline hiện tại, KHÔNG chỉnh gì)")
        L.append("")
        L.append(MATCH_HEADER)
        for v in VARIANTS:
            L.append(fmt_row_match(v, agg_match(sub(rows_s, variant=v, mode="asis"))))
        L.append("")
        L.append("### 1b. Ảnh hưởng theo nhóm cột (top-1 / d_đúng mean / ACCEPTED&đúng)")
        L.append("")
        L.append("| Biến thể | " + " | ".join(MODES) + " |")
        L.append("|---|" + "---|" * len(MODES))
        for v in VARIANTS:
            cells = []
            for mo in MODES:
                a = agg_match(sub(rows_s, variant=v, mode=mo))
                cells.append(f"{a['top1']:.1f}% / {a['d_mean']:.3f} / {a['acc_correct']:.0f}%" if a["n"] else "-")
            L.append(f"| {v} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Chú giải chế độ: " + "; ".join(f"`{k}` = {d}" for k, d in MODE_DESC.items()))
        L.append("")
        L.append("Hệ số nhân trung bình (fps_thực/30): " + ", ".join(
            f"{v}={np.mean([r['factor'] for r in sub(rows_s, variant=v, mode='asis')]):.2f}" for v in VARIANTS))
        L.append("")
        # so cặp với (a)
        L.append("### 1c. So cặp với (a), chế độ `asis`: số stroke đúng->sai / sai->đúng")
        L.append("")
        L.append("| Biến thể | đúng->sai | sai->đúng |")
        L.append("|---|---|---|")
        base_map = {r["stroke"]: r["ok"] for r in sub(rows_s, variant="a_goc", mode="asis")}
        for v in VARIANTS[1:]:
            d2w = w2d = 0
            for r in sub(rows_s, variant=v, mode="asis"):
                b = base_map.get(r["stroke"])
                if b is None:
                    continue
                d2w += int(b and not r["ok"])
                w2d += int((not b) and r["ok"])
            L.append(f"| {v} | {d2w} | {w2d} |")
        L.append("")
        L.append("### 1d. Chênh lệch có cặp so với (a): Δd_đúng và Δmargin (mean, CI95 bootstrap 2000 lần, seed 0)")
        L.append("")
        L.append("Top-1 thô (rời rạc) ít nhạy nên dùng khoảng cách làm thước đo. Δ>0 ở d_đúng = xấu đi; Δ<0 ở margin = xấu đi.")
        L.append("")
        L.append("| Biến thể | Chế độ | Δd_đúng mean [CI95] | Δd_đúng % so với d(a) | Δmargin mean [CI95] |")
        L.append("|---|---|---|---|---|")
        rng_b = np.random.RandomState(0)
        a_d = {r["stroke"]: (r["d_true"], r["margin"]) for r in sub(rows_s, variant="a_goc", mode="asis")}
        for v in VARIANTS[1:]:
            for mo in ("asis", "scale_both"):
                dd, dm = [], []
                for r in sub(rows_s, variant=v, mode=mo):
                    b0 = a_d.get(r["stroke"])
                    if b0 is None or not np.isfinite(r["d_true"]):
                        continue
                    dd.append(r["d_true"] - b0[0])
                    dm.append(r["margin"] - b0[1])
                if not dd:
                    continue
                dd, dm = np.array(dd), np.array(dm)
                idx_b = rng_b.randint(0, len(dd), size=(2000, len(dd)))
                cd = np.percentile(dd[idx_b].mean(axis=1), [2.5, 97.5])
                cm = np.percentile(dm[idx_b].mean(axis=1), [2.5, 97.5])
                base_mean = float(np.mean([b0[0] for b0 in a_d.values()]))
                L.append(f"| {v} | {mo} | {dd.mean():+.4f} [{cd[0]:+.4f}, {cd[1]:+.4f}] | {100*dd.mean()/base_mean:+.1f}% | "
                         f"{dm.mean():+.4f} [{cm[0]:+.4f}, {cm[1]:+.4f}] |")
        L.append("")

    # ---- phân đoạn ----
    L.append("## 2. Phân đoạn (KineticPauseSegmenter, cấu hình CONFIG hiện hành)")
    L.append("")
    L.append(f"Cấu hình: v_start={CONFIG['v_start_ratio']}, v_stop={CONFIG['v_stop_ratio']}, pause_frames={CONFIG['pause_frames']}, "
             f"min/max_stroke_frames=8/{CONFIG_MAX_STROKE_FRAMES}, require_ready_pose=True, grace=0.15s, max_phase_merges=1 (từ make_segmenter). "
             f"Luồng = đệm tĩnh {PRE_ROLL_SEC}s + stroke + đệm tĩnh {POST_ROLL_SEC}s. Số liệu tính trên TẤT CẢ stroke (phân đoạn không phụ thuộc bank).")
    L.append("")
    L.append("`asis` = segmenter y hệt production. `scaled` = THỬ NGHIỆM: v_start/v_stop chia r, pause_frames/ready_stable/max_stroke_frames/min_stroke_frames "
             "nhân r (sàn min=8), r = fps_thực/30 của biến thể; cho biết việc chuẩn hóa ngưỡng theo thời gian có cứu được phân đoạn không.")
    L.append("")
    L.append("| Biến thể | Segmenter | N | đúng 1 stroke | cắt đôi (>=2) | nhỡ (0) | qua cổng production | swipe giả | "
             "thời lượng chốt (s) median | tỉ lệ so với (a) median | trễ bắt đầu (s) median | lệch kết thúc (s) median |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    def _med(rr, key):
        vals = [r[key] for r in rr if np.isfinite(r[key])]
        return float(np.median(vals)) if vals else float("nan")
    for v in VARIANTS:
        for sm in SEG_MODES:
            rr = sub(seg_rows, variant=v, seg_mode=sm)
            n = len(rr)
            if not n:
                continue
            one = sum(1 for r in rr if r["cat"] == "đúng 1")
            spl = sum(1 for r in rr if r["cat"] == "cắt đôi")
            mis = sum(1 for r in rr if r["cat"] == "nhỡ")
            pg = sum(1 for r in rr if r["n_pass_gate"] >= 1)
            sw = sum(1 for r in rr if r["n_swipe"] > 0)
            L.append(f"| {v} | {sm} | {n} | {100*one/n:.1f}% | {100*spl/n:.1f}% | {100*mis/n:.1f}% | {100*pg/n:.1f}% | {100*sw/n:.1f}% | "
                     f"{_med(rr,'main_dur_sec'):.2f} | {_med(rr,'dur_vs_a'):.2f} | {_med(rr,'start_delay_sec'):+.2f} | {_med(rr,'end_delta_sec'):+.2f} |")
    L.append("")
    L.append(f"Số stroke có số khung vượt trần max_stroke_frames={CONFIG_MAX_STROKE_FRAMES} (segmenter asis) theo biến thể: " + ", ".join(
        f"{v}={sum(r['orig_frames_over_cap'] for r in sub(seg_rows, variant=v, seg_mode='asis'))}" for v in VARIANTS))
    L.append("")

    # ---- đầu-cuối ----
    for flt, title in ((True, "stroke TRONG bank (LOO) — bảng chính"), (False, "stroke NGOÀI bank (bank đầy đủ) — tham khảo")):
        rows_e = [r for r in e2e_rows if r["in_bank"] == flt]
        if not rows_e:
            continue
        L.append(f"## 3. Đầu-cuối: phân đoạn -> đặc trưng -> so khớp — {title}")
        L.append("")
        L.append("`thành công` = đúng MỘT từ được chốt (ACCEPTED) và đó là từ đúng; `từ sai` = số từ ACCEPTED khác từ đúng. "
                 "`Đặc trưng scale_both` = nhân đạo hàm x fps/30 (fps đo từ chính stroke được chốt).")
        L.append("")
        L.append("| Biến thể | Segmenter | Đặc trưng | N | thành công | tổng từ sai chốt | stroke có từ sai |")
        L.append("|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            for sm in SEG_MODES:
                for mo in E2E_MODES:
                    rr = sub(rows_e, variant=v, seg_mode=sm, mode=mo)
                    n = len(rr)
                    if not n:
                        continue
                    L.append(f"| {v} | {sm} | {mo} | {n} | {100*sum(r['success'] for r in rr)/n:.1f}% | "
                             f"{sum(r['n_wrong_words'] for r in rr)} | {sum(1 for r in rr if r['n_wrong_words']>0)} |")
        L.append("")

    # ---- theo từ (asis top-1) ----
    L.append("## 4. Top-1 theo từ (stroke trong bank, chế độ asis)")
    L.append("")
    L.append("| Từ | n | " + " | ".join(VARIANTS) + " |")
    L.append("|---|---|" + "---|" * len(VARIANTS))
    for w in sorted({r["word"] for r in match_rows}):
        rr0 = [r for r in match_rows if r["word"] == w and r["in_bank"] and r["mode"] == "asis" and r["variant"] == "a_goc"]
        if not rr0:
            continue
        cells = []
        for v in VARIANTS:
            rr = [r for r in match_rows if r["word"] == w and r["in_bank"] and r["mode"] == "asis" and r["variant"] == v]
            cells.append(f"{100*sum(r['ok'] for r in rr)/len(rr):.0f}%")
        L.append(f"| {w} | {len(rr0)} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Tệp CSV")
    L.append("`match_results.csv`, `segmentation.csv`, `end_to_end.csv` (cùng thư mục).")

    rp = os.path.join(out_dir, "report.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts, "n_strokes": len(order), "seed": args.seed, "limit": args.limit,
            "verify_a_top1_pct": chk_ok, "verify_a_d_true_median": chk_d, "raw_timing": timing,
            "asis_in_bank": {v: agg_match(sub([r for r in match_rows if r["in_bank"]], variant=v, mode="asis")) for v in VARIANTS},
        }, f, ensure_ascii=False, indent=2, default=float)

    print("\n" + "\n".join(L))
    print(f"\n[OK] Đã ghi báo cáo: {rp}")


# Trần số khung của segmenter (lấy từ chính make_segmenter để không lệch cấu hình).
CONFIG_MAX_STROKE_FRAMES = make_segmenter(CONFIG).max_stroke_frames

if __name__ == "__main__":
    main()
