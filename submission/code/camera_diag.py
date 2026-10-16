"""
CÔNG CỤ CHẨN ĐOÁN CAMERA (CAMERA DIAGNOSTIC) — soi khung trùng, nhịp đọc, độ sáng, độ ổn định dò tay
==============================================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

VÌ SAO CÓ CÔNG CỤ NÀY?
----------------------
Con số "FPS" trên màn hình của predict_sentence_webcam.py là fps = 0.9*fps + 0.1*(1/dt): đó là tốc độ
VÒNG LẶP (trung bình của 1/dt), KHÔNG phải số khung MỚI thật của camera. Khi các lần đọc dồn cục (dt ngắn xen
dt dài) hoặc driver trả lại cùng một khung nhiều lần (thường gặp trong phòng tối: cảm biến tăng thời gian phơi
sáng, giảm tốc độ thật), chỉ số này vẫn có thể hiện 50-60 trong khi số khung mới thật chỉ ~15-20.
Công cụ này ĐO trực tiếp, trong cùng điều kiện với ứng dụng thật (dùng đúng open_camera của pipeline_common:
cùng backend, MJPG, 640x480, yêu cầu 30 FPS), rồi in ra và lưu số đo để so sánh phòng tối với phòng sáng.

CÁCH DÙNG
---------
    py camera_diag.py --label toi  --cam 1 --seconds 25          # đo phòng tối
    py camera_diag.py --label sang --cam 1 --seconds 25          # đo phòng sáng
    py camera_diag.py --compare results/camera_diag/toi_XXXX results/camera_diag/sang_XXXX
    py camera_diag.py --label thu --video ASL_Citizen_videos/HELLO/xxx.mp4 --no-window   # thử bằng video, không cần webcam
    py camera_diag.py --label toi --record                       # kèm lưu video thô raw.avi (để chạy A/B tiền xử lý ánh sáng ngoại tuyến)

Trong lúc đo: đếm ngược 3 giây; pha 1 "Giữ yên tay trong khung" (20% thời gian); pha 2 "Vẫy tay NHANH qua lại"
(48%); pha 3 "Hạ tay ra khỏi khung" (32%). Với --seconds 25 đúng 0-5 s / 5-17 s / 17-25 s.
Công cụ KHÔNG thử SET bất kỳ thuộc tính camera nào (chỉ ĐỌC), không thay đổi cấu hình camera của bạn.

KẾT QUẢ: results/camera_diag/<label>_<YYYYMMDD_HHMMSS>/summary.json + frames.csv (+ raw.avi nếu --record).
"""

import os
import re
import sys
import csv
import json
import math
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import cv2

# ---------------------------------------------------------------------------------
# HẰNG SỐ
# ---------------------------------------------------------------------------------
COUNTDOWN_SEC = 3.0
DEFAULT_SECONDS = 25.0
PHASE_FRACTIONS = (5.0 / 25.0, 17.0 / 25.0)      # ranh giới pha 1|2 và pha 2|3 (tỉ lệ thời gian đo)
PHASE_NAMES = ("giu_yen", "vay_nhanh", "ha_tay")
PHASE_TEXT = {
    "giu_yen": "Giu yen tay trong khung",
    "vay_nhanh": "Vay tay NHANH qua lai",
    "ha_tay": "Ha tay ra khoi khung",
}
DUP_MAD_THRESHOLD = 0.5          # sai khác tuyệt đối trung bình (mức xám) dưới ngưỡng này => khung TRÙNG
DUP_SIZE = (64, 48)              # kích thước thu nhỏ để so khung trùng
V_START_RATIO = 0.08             # ngưỡng v_start_ratio của segmenter (SHARED_CONFIG)
HAND_MODEL = "hand_landmarker.task"
LOSS_RESET_FRAMES = 4            # như production: mất tay >= 4 khung => reset bộ làm mượt
FRAMES_CSV_FIELDS = ["idx", "t_read", "t_rel", "pos_msec", "phase", "dup", "gap_ms", "read_ms", "luma_mean",
                     "luma_p5", "luma_p95", "hand", "det_ms", "palm_len", "v_norm", "v_norm_raw"]


def default_camera_index() -> int:
    """Lấy CONFIG['camera_source'] từ predict_sentence_webcam.py bằng đọc văn bản (không import torch nặng); mặc định 1."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "predict_sentence_webcam.py"), "r", encoding="utf-8") as f:
            txt = f.read()
        m = re.search(r'"camera_source"\s*:\s*(\d+)', txt)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 1


# ---------------------------------------------------------------------------------
# TIỆN ÍCH THỐNG KÊ
# ---------------------------------------------------------------------------------
def pct(values: List[float], q: float) -> float:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(np.percentile(a, q)) if a.size else float("nan")


def stat_block(values: List[float]) -> Dict[str, float]:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def phase_of(t_rel: float, total: float) -> str:
    b1, b2 = PHASE_FRACTIONS[0] * total, PHASE_FRACTIONS[1] * total
    if t_rel < b1:
        return PHASE_NAMES[0]
    if t_rel < b2:
        return PHASE_NAMES[1]
    return PHASE_NAMES[2]


# ---------------------------------------------------------------------------------
# BỘ DÒ TAY: dựng đúng model/tham số của ParallelHandDetector (IMAGE mode, num_hands=1)
# ---------------------------------------------------------------------------------
class SyncHandDetector:
    """
    Một HandLandmarker chạy ĐỒNG BỘ, cùng tùy chọn với ParallelHandDetector._worker_loop của pipeline_common:
    BaseOptions(model_asset_path=hand_landmarker.task), RunningMode.IMAGE, num_hands=1; ảnh đưa vào là RGB (SRGB).
    Có warm-up 1 khung giả như bản gốc.
    """
    def __init__(self, model_path: str = HAND_MODEL, num_hands: int = 1):
        import mediapipe as mp
        if not os.path.exists(model_path):     # cho phép chạy từ thư mục khác: tìm cạnh file script
            alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(model_path))
            if os.path.exists(alt):
                model_path = alt
        self._mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=int(num_hands),
        )
        self._det = mp.tasks.vision.HandLandmarker.create_from_options(options)
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        self._det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy))

    def detect(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._det.detect(img)

    def close(self) -> None:
        try:
            self._det.close()
        except Exception:
            pass


def palm_len_from_landmarks(landmarks, w_f: int, h_f: int) -> float:
    """palm_len đúng công thức predict_sentence_webcam (palm_2d, palm_3d, knuckle_span*0.85, sàn 30)."""
    lm0, lm5, lm9, lm17 = landmarks[0], landmarks[5], landmarks[9], landmarks[17]
    palm_2d = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)
    palm_3d = math.sqrt(((lm9.x - lm0.x) * w_f) ** 2 + ((lm9.y - lm0.y) * h_f) ** 2 + ((lm9.z - lm0.z) * w_f) ** 2)
    knuckle_span = math.hypot((lm17.x - lm5.x) * w_f, (lm17.y - lm5.y) * h_f)
    return max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)


# ---------------------------------------------------------------------------------
# VẼ HƯỚNG DẪN TRÊN CỬA SỔ
# ---------------------------------------------------------------------------------
class Overlay:
    """Vẽ hướng dẫn to. Dùng ui_render.TextRenderer nếu import được (chữ có dấu), không thì cv2.putText chữ không dấu."""
    def __init__(self):
        self.tr = None
        try:
            from ui_render import TextRenderer
            self.tr = TextRenderer()
        except Exception:
            self.tr = None

    def draw(self, frame: np.ndarray, big: str, small: str, color: Tuple[int, int, int]) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
        if self.tr is not None:
            try:
                self.tr.queue(big, w // 2, 8, max(24, int(w / 16)), color, bold=True, anchor="c")
                self.tr.queue(small, w // 2, 70, max(14, int(w / 34)), (255, 255, 255), anchor="c")
                self.tr.flush(frame)
                return
            except Exception:
                pass
        # Dự phòng: cv2.putText (chữ không dấu)
        (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(frame, big, (max(5, (w - tw) // 2), 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
        (tw2, _), _ = cv2.getTextSize(small, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, small, (max(5, (w - tw2) // 2), 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------------
# ĐỌC THUỘC TÍNH CAMERA (chỉ GET, không SET)
# ---------------------------------------------------------------------------------
def read_camera_props(cap: cv2.VideoCapture) -> Dict[str, Any]:
    from pipeline_common import decode_fourcc
    props: Dict[str, Any] = {}
    names = [
        ("CAP_PROP_FPS", cv2.CAP_PROP_FPS), ("CAP_PROP_FOURCC", cv2.CAP_PROP_FOURCC),
        ("CAP_PROP_AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE), ("CAP_PROP_EXPOSURE", cv2.CAP_PROP_EXPOSURE),
        ("CAP_PROP_GAIN", cv2.CAP_PROP_GAIN), ("CAP_PROP_BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
        ("CAP_PROP_FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH), ("CAP_PROP_FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT),
    ]
    for name, pid in names:
        try:
            v = cap.get(pid)
        except Exception:
            v = None
        props[name] = None if v is None else float(v)
    if props.get("CAP_PROP_FOURCC") is not None:
        props["fourcc_text"] = decode_fourcc(props["CAP_PROP_FOURCC"])
    try:
        props["backend"] = cap.getBackendName()
    except Exception:
        props["backend"] = None
    return props


# ---------------------------------------------------------------------------------
# PHIÊN ĐO
# ---------------------------------------------------------------------------------
def run_session(args: argparse.Namespace) -> Optional[str]:
    video_mode = args.video is not None
    ts_name = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", "camera_diag", f"{args.label}_{ts_name}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Mở nguồn ----
    if video_mode:
        if not os.path.exists(args.video):
            print(f"[!] Không thấy video: {args.video}")
            return None
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[!] Không mở được video: {args.video}")
            return None
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        media_dur = (vframes / vfps) if vfps > 0 and vframes > 0 else args.seconds
        total = float(min(args.seconds, media_dur))
        print(f"[Video] Nguồn giả: {args.video} | {vfps:.1f} fps | ~{media_dur:.1f}s -> đo {total:.1f}s "
              f"(KHÔNG phải camera thật: FPS vòng lặp ở đây là tốc độ xử lý, không phản ánh camera).")
    else:
        from pipeline_common import open_camera
        cap = open_camera(args.cam)                   # ĐÚNG hàm mở camera của ứng dụng thật
        if cap is None:
            print(f"[!] Không mở được camera {args.cam}.")
            return None
        total = float(args.seconds)

    props = read_camera_props(cap)
    print("[Camera] Thuộc tính (chỉ đọc):")
    for k, v in props.items():
        print(f"    {k:24s} = {v}")

    # ---- Bộ dò tay + bộ làm mượt cổ tay giống production ----
    detector = SyncHandDetector()
    from pipeline_common import make_wrist_smoother
    smoother = make_wrist_smoother()
    overlay = Overlay()

    window = not args.no_window
    win_name = "camera_diag (q = dung)"
    if window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # ---- Đếm ngược 3 s (đọc và bỏ khung để camera chạy liên tục, không ghi số đo) ----
    if not video_mode:
        t_cd0 = time.perf_counter()
        while True:
            left = COUNTDOWN_SEC - (time.perf_counter() - t_cd0)
            if left <= 0:
                break
            ok, fr = cap.read()
            if not ok:
                continue
            if window:
                disp = fr.copy()
                overlay.draw(disp, f"Chuan bi... {int(math.ceil(left))}", "Dat tay san sang, sap bat dau do", (0, 255, 255))
                cv2.imshow(win_name, disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    cap.release()
                    detector.close()
                    cv2.destroyAllWindows()
                    return None

    # ---- Vòng đo ----
    rows: List[Dict[str, Any]] = []
    jpeg_buf: List[bytes] = []
    prev_small: Optional[np.ndarray] = None
    prev_hand = False
    prev_wrist_s: Optional[Tuple[float, float]] = None
    prev_wrist_r: Optional[Tuple[float, float]] = None
    lost_run = 0
    t_first_clock: Optional[float] = None
    t_prev_clock: Optional[float] = None
    aborted = False
    idx = 0
    w_f = h_f = 0
    wall0 = time.perf_counter()

    while True:
        t_r0 = time.perf_counter()
        ok, frame = cap.read()
        t_r1 = time.perf_counter()
        if not ok:
            if video_mode:
                break
            continue
        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        # Đồng hồ đo: camera thật = perf_counter; video = thời gian phương tiện (POS_MSEC) để pha đúng theo nội dung video.
        if video_mode:
            t_clock = (pos_msec / 1000.0) if pos_msec > 0 else (idx / max(vfps, 1.0))
        else:
            t_clock = t_r1
        if t_first_clock is None:
            t_first_clock = t_clock
        t_rel = t_clock - t_first_clock
        if t_rel >= total:
            break

        h_f, w_f = frame.shape[:2]
        gap_ms = (t_clock - t_prev_clock) * 1000.0 if t_prev_clock is not None else float("nan")
        t_prev_clock = t_clock

        # Khung trùng: so khung xám 64x48 với khung liền trước
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, DUP_SIZE, interpolation=cv2.INTER_AREA)
        dup = False
        if prev_small is not None:
            mad = float(np.mean(np.abs(small.astype(np.int16) - prev_small.astype(np.int16))))
            dup = mad < DUP_MAD_THRESHOLD
        prev_small = small

        # Độ sáng
        luma_mean = float(cv2.mean(gray)[0])
        g_small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        luma_p5, luma_p95 = float(np.percentile(g_small, 5)), float(np.percentile(g_small, 95))

        # Dò tay
        t_d0 = time.perf_counter()
        det = detector.detect(frame)
        det_ms = (time.perf_counter() - t_d0) * 1000.0
        hand = bool(det.hand_landmarks and len(det.hand_landmarks) > 0)
        palm_len = float("nan")
        v_s = v_r = float("nan")
        if hand:
            lm = det.hand_landmarks[0]
            palm_len = palm_len_from_landmarks(lm, w_f, h_f)
            raw_x, raw_y = lm[0].x * w_f, lm[0].y * h_f
            sx, sy = smoother.update(raw_x, raw_y, timestamp=t_clock)
            lost_run = 0
            if prev_hand and prev_wrist_s is not None and prev_wrist_r is not None:
                safe_palm = max(palm_len, 15.0)
                v_s = math.hypot(sx - prev_wrist_s[0], sy - prev_wrist_s[1]) / safe_palm      # kiểu segmenter (đã làm mượt)
                v_r = math.hypot(raw_x - prev_wrist_r[0], raw_y - prev_wrist_r[1]) / safe_palm
            prev_wrist_s, prev_wrist_r = (sx, sy), (raw_x, raw_y)
        else:
            lost_run += 1
            if lost_run >= LOSS_RESET_FRAMES:
                smoother.reset()
                prev_wrist_s = prev_wrist_r = None
        prev_hand = hand

        phase = phase_of(t_rel, total)
        rows.append({
            "idx": idx, "t_read": round(t_r1, 6), "t_rel": round(t_rel, 6), "pos_msec": round(pos_msec, 3),
            "phase": phase, "dup": int(dup), "gap_ms": round(gap_ms, 3) if np.isfinite(gap_ms) else "",
            "read_ms": round((t_r1 - t_r0) * 1000.0, 3), "luma_mean": round(luma_mean, 3),
            "luma_p5": round(luma_p5, 2), "luma_p95": round(luma_p95, 2), "hand": int(hand),
            "det_ms": round(det_ms, 3), "palm_len": round(palm_len, 3) if hand else "",
            "v_norm": round(v_s, 5) if np.isfinite(v_s) else "", "v_norm_raw": round(v_r, 5) if np.isfinite(v_r) else "",
        })

        if args.record:
            okj, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            if okj:
                jpeg_buf.append(buf.tobytes())

        idx += 1

        if window:
            disp = frame.copy()
            left = max(0.0, total - t_rel)
            live = idx / max(time.perf_counter() - wall0, 1e-6)
            overlay.draw(disp, PHASE_TEXT[phase], f"con {left:4.1f}s | tay: {'CO' if hand else 'khong'} | vong lap {live:.0f} lan/s",
                         (0, 255, 0) if phase != "vay_nhanh" else (0, 165, 255))
            cv2.imshow(win_name, disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                aborted = True
                break

    cap.release()
    detector.close()
    if window:
        cv2.destroyAllWindows()

    if len(rows) < 10:
        print(f"[!] Quá ít khung ({len(rows)}) — không đủ để tổng kết.")
        return None

    summary = summarize(rows, total, props, args, video_mode, aborted, w_f, h_f)
    summary["out_dir"] = out_dir

    # ---- Ghi frames.csv, summary.json ----
    with open(os.path.join(out_dir, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=FRAMES_CSV_FIELDS)
        wri.writeheader()
        wri.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else float(o))

    # ---- raw.avi (tùy chọn): fps = số lần đọc/giây đo được, để thời lượng phát lại đúng thời gian thực ----
    if args.record and jpeg_buf:
        rec_fps = float(summary["loop"]["fps"]) if np.isfinite(summary["loop"]["fps"]) and summary["loop"]["fps"] > 1 else 30.0
        avi_path = os.path.join(out_dir, "raw.avi")
        vw = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"MJPG"), rec_fps, (w_f, h_f))
        for b in jpeg_buf:
            im = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                vw.write(im)
        vw.release()
        summary["record"] = {"path": avi_path, "fps": rec_fps, "n_frames": len(jpeg_buf),
                             "note": "Mọi khung camera trả về (kể cả khung trùng), nén JPEG q98 rồi ghi MJPG; fps = số lần đọc/giây đo được."}
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else float(o))
        print(f"[Record] Đã lưu video thô: {avi_path} ({len(jpeg_buf)} khung, {rec_fps:.1f} fps)")

    print_summary(summary)
    print(f"\n[OK] Đã lưu: {out_dir}{os.sep}summary.json và frames.csv")
    return out_dir


# ---------------------------------------------------------------------------------
# TỔNG KẾT
# ---------------------------------------------------------------------------------
def runs_of_false(flags: List[bool]) -> List[int]:
    """Độ dài các chuỗi liên tiếp giá trị False (mất tay)."""
    runs, cur = [], 0
    for f in flags:
        if not f:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def summarize(rows: List[Dict[str, Any]], total: float, props: Dict[str, Any], args: argparse.Namespace,
              video_mode: bool, aborted: bool, w_f: int, h_f: int) -> Dict[str, Any]:
    n = len(rows)
    t_rel = np.array([r["t_rel"] for r in rows], dtype=np.float64)
    span = float(t_rel[-1] - t_rel[0]) if n > 1 else float("nan")
    gaps = [float(r["gap_ms"]) for r in rows if r["gap_ms"] != ""]
    dup = np.array([r["dup"] for r in rows], dtype=bool)

    # Khoảng cách giữa hai khung MỚI liên tiếp (bỏ khung trùng) — theo đồng hồ đo
    new_idx = [i for i in range(n) if (i == 0 or not dup[i])]
    new_gaps = [1000.0 * (t_rel[b] - t_rel[a]) for a, b in zip(new_idx[:-1], new_idx[1:])]
    new_span = float(t_rel[new_idx[-1]] - t_rel[new_idx[0]]) if len(new_idx) > 1 else float("nan")

    loop_fps = (n - 1) / span if span > 0 else float("nan")
    new_fps = (len(new_idx) - 1) / new_span if new_span and new_span > 0 else float("nan")
    inv_dt = [1000.0 / g for g in gaps if g > 1e-3]

    # POS_MSEC: có ý nghĩa nếu tăng đơn điệu và không toàn 0
    pos = np.array([r["pos_msec"] for r in rows], dtype=np.float64)
    pos_ok = bool(np.count_nonzero(np.diff(pos) > 0) > 0.5 * (n - 1))
    pos_info: Dict[str, Any] = {"meaningful": pos_ok}
    if pos_ok:
        pos_span = float(pos[-1] - pos[0]) / 1000.0
        pos_new = [pos[b] - pos[a] for a, b in zip(new_idx[:-1], new_idx[1:])]
        pos_info.update({
            "span_sec": pos_span, "clock_span_sec": span,
            "drift_ratio": pos_span / span if span > 0 else float("nan"),
            "fps_by_posmsec": (n - 1) / pos_span if pos_span > 0 else float("nan"),
            "new_gap_ms": stat_block(pos_new),
        })
    else:
        pos_info["note"] = "POS_MSEC không tăng (driver không cung cấp) — bỏ qua"

    luma_mean = [r["luma_mean"] for r in rows]
    hand = np.array([r["hand"] for r in rows], dtype=bool)
    phase = np.array([r["phase"] for r in rows])
    v = np.array([float(r["v_norm"]) if r["v_norm"] != "" else np.nan for r in rows])

    def rate(mask: np.ndarray) -> float:
        return float(hand[mask].mean()) if mask.any() else float("nan")

    m_hold, m_wave, m_off = (phase == PHASE_NAMES[0]), (phase == PHASE_NAMES[1]), (phase == PHASE_NAMES[2])
    wave_runs = runs_of_false(hand[m_wave].tolist())
    v_wave = v[m_wave & np.isfinite(v)]
    v_hold = v[m_hold & np.isfinite(v)]
    det_ms = [r["det_ms"] for r in rows]

    return {
        "label": args.label, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "video" if video_mode else "camera", "video": args.video, "cam_index": None if video_mode else args.cam,
        "note_source": ("NGUỒN GIẢ (video): không phản ánh camera thật; FPS vòng lặp = tốc độ đồng hồ phương tiện." if video_mode else ""),
        "aborted": aborted, "frame_size": [w_f, h_f], "seconds_requested": args.seconds, "seconds_measured": span,
        "phase_boundaries_sec": [PHASE_FRACTIONS[0] * total, PHASE_FRACTIONS[1] * total, total],
        "camera_props": props, "n_frames": n,
        "loop": {"fps": loop_fps, "mean_inv_dt_fps": float(np.mean(inv_dt)) if inv_dt else float("nan"),
                 "gap_ms": stat_block(gaps),
                 "note": "fps = số lần read/giây; mean_inv_dt_fps = trung bình 1/dt (cái mà chỉ số FPS trên màn hình của predict hội tụ tới)"},
        "new_frames": {"fps": new_fps, "dup_rate": float(dup.mean()), "n_dup": int(dup.sum()), "n_new": len(new_idx),
                       "gap_ms": stat_block(new_gaps)},
        "pos_msec": pos_info,
        "luma": {"mean": float(np.mean(luma_mean)), "p5": float(np.mean([r["luma_p5"] for r in rows])),
                 "p95": float(np.mean([r["luma_p95"] for r in rows])),
                 "note": "mean = trung bình luma toàn khung qua các khung; p5/p95 = trung bình phân vị 5/95 của từng khung"},
        "hand": {
            "rate_total": float(hand.mean()), "rate_hold": rate(m_hold), "rate_wave": rate(m_wave), "rate_off": rate(m_off),
            "wave_loss_events": len(wave_runs),
            "wave_loss_run_len": {"mean": float(np.mean(wave_runs)) if wave_runs else 0.0,
                                  "max": int(max(wave_runs)) if wave_runs else 0, "runs": wave_runs},
        },
        "v_norm": {
            "wave": {"n": int(v_wave.size), "median": float(np.median(v_wave)) if v_wave.size else float("nan"),
                     "p95": float(np.percentile(v_wave, 95)) if v_wave.size else float("nan"),
                     "max": float(v_wave.max()) if v_wave.size else float("nan")},
            "hold_over_start": {"threshold": V_START_RATIO, "n_valid": int(v_hold.size),
                                "count": int(np.sum(v_hold > V_START_RATIO)),
                                "rate": float(np.mean(v_hold > V_START_RATIO)) if v_hold.size else float("nan"),
                                "p95": float(np.percentile(v_hold, 95)) if v_hold.size else float("nan")},
            "note": "v_norm = |Δcổ tay đã làm mượt (1€ filter như production)| / palm_len giữa hai khung LIÊN TIẾP đều có tay",
        },
        "detect_ms": stat_block(det_ms),
    }


def _f(x: Any, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def print_summary(s: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"  TỔNG KẾT CHẨN ĐOÁN CAMERA — nhãn: {s['label']}  ({s['source']})")
    print("=" * 78)
    if s.get("note_source"):
        print(f"  ({s['note_source']})")
    print(f"  Số khung ghi: {s['n_frames']} trong {_f(s['seconds_measured'])} s | kích thước {s['frame_size']}")
    lp, nf = s["loop"], s["new_frames"]
    print("\n  [NHỊP ĐỌC]")
    print(f"    FPS vòng lặp (số read/giây)      : {_f(lp['fps'])}")
    print(f"    'FPS hiển thị' (TB của 1/dt)     : {_f(lp['mean_inv_dt_fps'])}   <- con số kiểu màn hình predict; lệch cao khi đọc dồn cục")
    print(f"    FPS KHUNG MỚI THẬT (bỏ trùng)    : {_f(nf['fps'])}")
    print(f"    Tỉ lệ khung TRÙNG                : {100 * nf['dup_rate']:.1f}%  ({nf['n_dup']}/{s['n_frames']})")
    g = lp["gap_ms"]
    print(f"    Khoảng cách giữa 2 lần read (ms) : p50={_f(g['p50'],1)}  p95={_f(g['p95'],1)}  max={_f(g['max'],1)}")
    g2 = nf["gap_ms"]
    print(f"    Khoảng cách giữa 2 khung mới (ms): p50={_f(g2['p50'],1)}  p95={_f(g2['p95'],1)}  max={_f(g2['max'],1)}")
    pm = s["pos_msec"]
    if pm.get("meaningful"):
        print(f"    POS_MSEC vs đồng hồ hệ thống     : tỉ lệ khoảng = {_f(pm['drift_ratio'],3)} | fps theo POS_MSEC = {_f(pm['fps_by_posmsec'])}")
    else:
        print(f"    POS_MSEC                         : {pm.get('note')}")
    lu = s["luma"]
    print("\n  [ĐỘ SÁNG]")
    print(f"    luma trung bình={_f(lu['mean'],1)} | p5={_f(lu['p5'],1)} | p95={_f(lu['p95'],1)}   (thang 0-255)")
    h, v = s["hand"], s["v_norm"]
    print("\n  [DÒ TAY (MediaPipe đồng bộ)]")
    print(f"    Tỉ lệ khung có tay: tổng={100*h['rate_total']:.1f}% | giữ yên={100*h['rate_hold']:.1f}% | vẫy nhanh={100*h['rate_wave']:.1f}% | hạ tay={100*h['rate_off']:.1f}%")
    print(f"    Pha vẫy nhanh: {h['wave_loss_events']} lần mất tay; độ dài (khung) TB={_f(h['wave_loss_run_len']['mean'],1)} max={h['wave_loss_run_len']['max']}")
    print(f"    v_norm pha vẫy nhanh: median={_f(v['wave']['median'],3)} p95={_f(v['wave']['p95'],3)} max={_f(v['wave']['max'],3)} (n={v['wave']['n']})")
    ho = v["hold_over_start"]
    print(f"    Giữ yên: v_norm > {ho['threshold']} (v_start_ratio) {ho['count']}/{ho['n_valid']} khung ({_f(100*ho['rate'] if np.isfinite(ho['rate']) else float('nan'),1)}%) | p95={_f(ho['p95'],3)}  <- >0 nghĩa là nhiễu có thể tự kích hoạt phân đoạn")
    d = s["detect_ms"]
    print(f"    Thời gian dò MediaPipe (ms): mean={_f(d['mean'],1)} p95={_f(d['p95'],1)} max={_f(d['max'],1)}")
    print("=" * 78)


# ---------------------------------------------------------------------------------
# SO SÁNH HAI PHIÊN
# ---------------------------------------------------------------------------------
def _get(s: Dict[str, Any], path: str) -> Optional[float]:
    cur: Any = s
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if cur is None or not isinstance(cur, (int, float)) or not np.isfinite(cur):
        return None
    return float(cur)


# (đường dẫn, nhãn, kiểu đánh dấu: "rel" = lệch tương đối > 25%, "pp" = lệch tuyệt đối > 10 điểm % (tỉ lệ 0-1), hệ số hiển thị)
COMPARE_ROWS: List[Tuple[str, str, str, float]] = [
    ("loop.fps", "FPS vòng lặp (read/giây)", "rel", 1.0),
    ("loop.mean_inv_dt_fps", "'FPS hiển thị' (TB 1/dt)", "rel", 1.0),
    ("new_frames.fps", "FPS khung MỚI thật", "rel", 1.0),
    ("new_frames.dup_rate", "Tỉ lệ khung trùng (%)", "pp", 100.0),
    ("loop.gap_ms.p50", "Khoảng 2 lần read p50 (ms)", "rel", 1.0),
    ("loop.gap_ms.p95", "Khoảng 2 lần read p95 (ms)", "rel", 1.0),
    ("loop.gap_ms.max", "Khoảng 2 lần read max (ms)", "rel", 1.0),
    ("new_frames.gap_ms.p50", "Khoảng 2 khung mới p50 (ms)", "rel", 1.0),
    ("new_frames.gap_ms.p95", "Khoảng 2 khung mới p95 (ms)", "rel", 1.0),
    ("new_frames.gap_ms.max", "Khoảng 2 khung mới max (ms)", "rel", 1.0),
    ("pos_msec.drift_ratio", "POS_MSEC / đồng hồ hệ thống", "rel", 1.0),
    ("luma.mean", "Luma trung bình", "rel", 1.0),
    ("luma.p5", "Luma p5", "rel", 1.0),
    ("luma.p95", "Luma p95", "rel", 1.0),
    ("hand.rate_total", "Có tay: tổng (%)", "pp", 100.0),
    ("hand.rate_hold", "Có tay: giữ yên (%)", "pp", 100.0),
    ("hand.rate_wave", "Có tay: vẫy nhanh (%)", "pp", 100.0),
    ("hand.wave_loss_events", "Số lần mất tay khi vẫy", "rel", 1.0),
    ("hand.wave_loss_run_len.max", "Lần mất tay dài nhất (khung)", "rel", 1.0),
    ("v_norm.wave.median", "v_norm vẫy: median", "rel", 1.0),
    ("v_norm.wave.p95", "v_norm vẫy: p95", "rel", 1.0),
    ("v_norm.wave.max", "v_norm vẫy: max", "rel", 1.0),
    ("v_norm.hold_over_start.count", "Giữ yên: số lần v_norm > 0.08", "rel", 1.0),
    ("v_norm.hold_over_start.p95", "Giữ yên: v_norm p95", "rel", 1.0),
    ("detect_ms.mean", "Thời gian dò MediaPipe mean (ms)", "rel", 1.0),
    ("detect_ms.p95", "Thời gian dò MediaPipe p95 (ms)", "rel", 1.0),
]


def compare(dir_a: str, dir_b: str) -> None:
    sums = []
    for d in (dir_a, dir_b):
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            print(f"[!] Không thấy {p}")
            return
        with open(p, "r", encoding="utf-8") as f:
            sums.append(json.load(f))
    a, b = sums
    print("\n" + "=" * 100)
    print(f"  SO SÁNH HAI PHIÊN CHẨN ĐOÁN CAMERA:  A = {a['label']} ({os.path.basename(dir_a.rstrip('/\\'))})   |   B = {b['label']} ({os.path.basename(dir_b.rstrip('/\\'))})")
    print("=" * 100)
    for tag, s in (("A", a), ("B", b)):
        if s.get("source") != "camera":
            print(f"  [Lưu ý] Phiên {tag} là nguồn '{s.get('source')}' (không phải camera thật).")
    print(f"  {'Chỉ số':38s} {'A':>12s} {'B':>12s} {'B-A':>12s} {'B/A':>8s}   Ghi chú")
    print("  " + "-" * 96)
    n_flag = 0
    for path, label, kind, scale in COMPARE_ROWS:
        va, vb = _get(a, path), _get(b, path)
        if va is None or vb is None:
            print(f"  {label:38s} {'n/a' if va is None else f'{va*scale:.2f}':>12s} {'n/a' if vb is None else f'{vb*scale:.2f}':>12s}")
            continue
        va_s, vb_s = va * scale, vb * scale
        diff = vb_s - va_s
        ratio = (vb / va) if abs(va) > 1e-9 else float("inf") if abs(vb) > 1e-9 else 1.0
        flag = ""
        if kind == "rel":
            if abs(va) < 1e-9 and abs(vb) < 1e-9:
                flag = ""
            elif (abs(va) < 1e-9) or abs(ratio - 1.0) > 0.25:
                flag = "<< KHÁC NHIỀU"
        else:  # pp
            if abs(vb - va) > 0.10:
                flag = "<< KHÁC NHIỀU"
        if flag:
            n_flag += 1
        rs = "inf" if not np.isfinite(ratio) else f"{ratio:.2f}"
        print(f"  {label:38s} {va_s:12.2f} {vb_s:12.2f} {diff:+12.2f} {rs:>8s}   {flag}")
    print("  " + "-" * 96)
    print(f"  Số chỉ số khác nhiều: {n_flag}/{len(COMPARE_ROWS)}  (ngưỡng: lệch tương đối > 25%, hoặc lệch > 10 điểm % với các tỉ lệ)")
    print("=" * 100)


# ---------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Chan doan camera: khung trung, nhip doc, do sang, on dinh do tay. Dung dung open_camera cua pipeline_common.")
    ap.add_argument("--label", type=str, default="phien", help="Nhan phien do (vd: toi, sang).")
    ap.add_argument("--cam", type=int, default=None,
                    help="Chi so camera (mac dinh: CONFIG['camera_source'] trong predict_sentence_webcam.py, khong thay duoc thi 1).")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help=f"Thoi luong do (mac dinh {DEFAULT_SECONDS:.0f}s).")
    ap.add_argument("--record", action="store_true", help="Luu video tho raw.avi (mac dinh TAT).")
    ap.add_argument("--video", type=str, default=None, help="Dung video thay camera (de thu cong cu khi khong co webcam).")
    ap.add_argument("--no-window", action="store_true", help="Khong mo cua so hien thi (chay ngam).")
    ap.add_argument("--compare", nargs=2, metavar=("THU_MUC_A", "THU_MUC_B"),
                    help="So sanh hai thu muc phien (vd toi vs sang) roi thoat.")
    args = ap.parse_args()
    if args.cam is None:
        args.cam = default_camera_index()
    return args


def main() -> None:
    args = parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return
    if args.seconds <= 0:
        print("[!] --seconds phải > 0.")
        sys.exit(2)
    run_session(args)


if __name__ == "__main__":
    main()
