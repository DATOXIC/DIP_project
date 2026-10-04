"""
Chấm điểm BÁO NHẦM trên một phiên CHỈ MẪU ÂM (collect_continuous.py --negative): cả phiên không ký, nên MỌI từ mà hệ thống
chốt (ACCEPTED) trong lúc đó đều là báo nhầm. Phát lại video phiên qua predict_sentence_webcam.py (đồng hồ video) với:
  - KHÔNG spotter                       (hệ thống như chưa có spotter)
  - mỗi --model: khóa một lần + spotter DL với model đó (so sánh được model cũ và model mới huấn luyện lại)

Đếm trên các khung có nhãn WANDER/REST (bỏ IGNORE: vùng đệm ranh giới và attempt đã đánh dấu hỏng 'x').
Kết quả: số lần chốt nhầm, số lần mỗi phút, và từ bị chốt nhầm.

Cách dùng:
    py eval_negative_session.py continuous_data/Dat/<phiên>_neg
    py eval_negative_session.py --latest --model spotter_model.pth --model spotter_model_new.pth
Muốn số có giá trị: model phải được huấn luyện KHÔNG chứa phiên này (train_spotter.py --exclude <phiên>).
"""

import argparse
import collections
import csv
import glob
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from collect_continuous import REST, WANDER, label_durations, load_session

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))


def latest_negative_session(root: str = "continuous_data") -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(root, "*", "*_neg")), key=os.path.getmtime)
    return cands[-1] if cands else None


def replay_command(video: str, log_csv: str, model: Optional[str], templates: Optional[str] = None) -> List[str]:
    cmd = [sys.executable, "predict_sentence_webcam.py", "--video", video, "--video-clock", "--headless",
           "--no-confirm-enroll", "--log-session", log_csv]
    if templates:
        cmd += ["--templates", templates]
    if model:
        cmd += ["--one-shot", "--spotter", "dl", "--spotter-model", model]
    return cmd


def parse_t0(run_txt: str) -> Optional[float]:
    with open(run_txt, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"VIDEO_CLOCK_T0=([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None


def score_rows(rows: Sequence[Dict[str, Any]], t0: float, t: np.ndarray, labels: np.ndarray, fps: float = 30.0) -> Dict[str, Any]:
    """Đếm kết quả theo nhãn khung tại giữa stroke (thuần, dùng được trong kiểm thử)."""
    t_first = float(t[0])
    c: collections.Counter = collections.Counter()
    words: collections.Counter = collections.Counter()
    for r in rows:
        ev = r["event"]
        ts = float(r["t_start"]) - t0 + t_first - 1.0 / fps
        mid = ts + 0.5 * float(r["duration_sec"])
        j = int(np.clip(np.searchsorted(t, mid), 0, len(t) - 1))
        if int(labels[j]) not in (WANDER, REST):
            c["ngoai_nhan"] += 1
            continue
        c[ev] += 1
        if ev == "ACCEPTED":
            words[r["predicted_word"]] += 1
    return {"count": c, "words": words}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Chấm báo nhầm trên phiên chỉ mẫu âm.")
    ap.add_argument("session", nargs="?", default=None, help="Thư mục phiên (có session.mp4, frames.npz, events.json)")
    ap.add_argument("--latest", action="store_true", help="Dùng phiên *_neg mới nhất trong continuous_data/")
    ap.add_argument("--model", action="append", default=None, help="Model spotter để so sánh (lặp được). Mặc định spotter_model.pth")
    ap.add_argument("--templates", default=None, help="Bank template (mặc định theo CONFIG)")
    ap.add_argument("--out-dir", default=os.path.join("scratch", "negeval"))
    args = ap.parse_args(argv)

    sdir = args.session or (latest_negative_session() if args.latest else None)
    if not sdir or not os.path.isdir(sdir):
        print("[!] Cần thư mục phiên hoặc --latest (chưa có phiên *_neg nào trong continuous_data/).")
        return 2
    video = os.path.join(sdir, "session.mp4")
    if not os.path.isfile(video):
        print(f"[!] Phiên không có session.mp4 ({video}); thu lại không dùng --no-video.")
        return 2
    s = load_session(sdir)
    t, labels, fps = np.asarray(s["t"], dtype=np.float64), np.asarray(s["labels"]), float(s["fps"]) or 30.0
    dur = label_durations(t, labels)
    neg_sec = dur["WANDER"]["seconds"] + dur["REST"]["seconds"]
    neg_min = max(neg_sec / 60.0, 1e-6)
    models = args.model or (["spotter_model.pth"] if os.path.exists(os.path.join(ROOT, "spotter_model.pth")) else [])
    configs: List[Tuple[str, Optional[str]]] = [("khong spotter", None)] + [(os.path.basename(m), m) for m in models]
    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    tag = os.path.basename(os.path.normpath(sdir))
    print(f"Phiên {tag}: {neg_sec:.0f}s không ký có nhãn (WANDER {dur['WANDER']['seconds']:.0f}s, REST {dur['REST']['seconds']:.0f}s). "
          f"Phát lại {len(configs)} cấu hình song song (vài phút)...")

    procs = []
    for i, (name, model) in enumerate(configs):
        log_csv = os.path.join(args.out_dir, f"{tag}_{i}.csv")
        run_txt = os.path.join(args.out_dir, f"{tag}_{i}.txt")
        if os.path.exists(os.path.join(ROOT, log_csv)):
            os.remove(os.path.join(ROOT, log_csv))
        fh = open(os.path.join(ROOT, run_txt), "w", encoding="utf-8")
        p = subprocess.Popen(replay_command(video, log_csv, model, args.templates), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((name, p, fh, log_csv, run_txt))
    for _, p, fh, _, _ in procs:
        p.wait()
        fh.close()

    print(f"\n{'cau hinh':<28}{'chot NHAM':>10}{'/phut':>8}{'khong chac':>12}{'tu choi':>9}{'bi chan':>9}   tu bi chot nham")
    for name, p, _, log_csv, run_txt in procs:
        t0 = parse_t0(os.path.join(ROOT, run_txt))
        if p.returncode != 0 or t0 is None or not os.path.exists(os.path.join(ROOT, log_csv)):
            print(f"{name:<28}  [loi] xem {run_txt}")
            continue
        with open(os.path.join(ROOT, log_csv), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        r = score_rows(rows, t0, t, labels, fps)
        c = r["count"]
        w = ", ".join(f"{k}x{v}" for k, v in r["words"].most_common()) or "-"
        print(f"{name:<28}{c['ACCEPTED']:>10}{c['ACCEPTED'] / neg_min:>8.1f}{c['UNCERTAIN']:>12}{c['REJECTED']:>9}{c['SUPPRESSED_DUP']:>9}   {w}")
    print("\n'chot NHAM' thap la tot (cả phiên không ký). 'bi chan' = spotter chan, khong vao cau.")
    print("Luu y: so chi co gia tri neu model khong duoc huan luyen tren chinh phien nay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
