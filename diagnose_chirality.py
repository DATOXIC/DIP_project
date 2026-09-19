"""
CHẨN ĐOÁN CHUẨN HÓA TAY THUẬN (CHIRALITY DIAGNOSTIC) — CHỈ ĐỌC, KHÔNG GHI GÌ
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

MỤC ĐÍCH
--------
MediaPipe gán nhãn handedness với giả định ảnh đầu vào ĐÃ lật gương (selfie view),
nhưng cả bộ thu mẫu lẫn bộ suy luận đều để flip_cam = False. Hậu quả là quy ước dấu
của vector pháp tuyến lòng bàn tay (kênh nx, cột 9) KHÁC nhau giữa dữ liệu webcam và
dữ liệu trích từ video ASL Citizen. Vì w_n = 0.8 mà ngưỡng mỗi từ chỉ ~0.6, một vector
pháp tuyến bị lật dấu tự nó đã đủ để TỪ CHỐI một cử chỉ hoàn toàn đúng.

Script này in ra, cho TỪNG file template và TỪNG từ:
  - Số mẫu.
  - Bao nhiêu mẫu có ghi nhãn tay thuận ("handedness") và phân bố các nhãn đó.
  - mean(nx) TRƯỚC và SAU khi chuẩn hóa chirality.
  - Số mẫu rơi vào dải MẬP MỜ max|nx| < 0.05 (heuristic không đáng tin ở dải này).

Đây là công cụ để CHỐT quy ước dấu và phát hiện hồi quy về sau. Nó TUYỆT ĐỐI KHÔNG
ghi, sửa hay xóa bất cứ file nào.

CÁCH DÙNG
---------
    py diagnose_chirality.py                     # quét mọi file template tìm thấy
    py diagnose_chirality.py --files a.json b.json
    py diagnose_chirality.py --include-backups   # quét cả các file *.backup*.json
"""

import os
import sys
import glob
import json
import argparse
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from dynamic_dtw_engine import (
    canonicalize_chirality,
    CHIRALITY_AMBIGUITY_EPS,
    CHIRALITY_FLIP_COLUMNS,
)

# Các file template "chính" (không phải backup) mà project dùng thật.
PRIMARY_TEMPLATE_FILES = [
    "dynamic_templates.json",
    "asl_citizen_templates.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chan doan quy uoc dau chirality trong cac file template (CHI DOC)."
    )
    p.add_argument("--files", nargs="*", default=None,
                   help="Danh sach file template cu the can quet (mac dinh: cac file chinh).")
    p.add_argument("--include-backups", action="store_true",
                   help="Quet ca cac file *.backup*.json (chi doc, khong sua).")
    p.add_argument("--eps", type=float, default=CHIRALITY_AMBIGUITY_EPS,
                   help=f"Nua do rong dai map mo cua heuristic max|nx| (mac dinh {CHIRALITY_AMBIGUITY_EPS}).")
    return p.parse_args()


def discover_files(args: argparse.Namespace) -> List[str]:
    if args.files:
        return [f for f in args.files if os.path.exists(f)]

    found = [f for f in PRIMARY_TEMPLATE_FILES if os.path.exists(f)]
    if args.include_backups:
        backups = sorted(set(glob.glob("*.backup*.json")))
        found.extend(b for b in backups if b not in found)
    return found


def get_sample_traj(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    raw = sample.get("traj_features_9d")
    if raw is None:
        raw = sample.get("traj_features")
    if not raw:
        return None
    arr = np.array(raw, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return arr


def diagnose_file(path: str, eps: float) -> Dict[str, int]:
    """In bao cao cho 1 file template. Tra ve tong hop cap file."""
    print()
    print("=" * 100)
    print(f"  FILE: {path}")
    print("=" * 100)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [!] Khong doc duoc file: {e}")
        return {}

    meta = data.get("metadata", {})
    print(f"  metadata: feature_dim={meta.get('feature_dim', '?')}, "
          f"k_resample_frames={meta.get('k_resample_frames', '?')}, "
          f"schema_version={data.get('schema_version', '?')}")

    vocab = data.get("vocabulary", {})
    if not vocab:
        print("  [!] File khong co 'vocabulary'.")
        return {}

    header = (f"  {'TU VUNG':<16} {'#MAU':>5} {'#CO_NHAN':>9} {'PHAN BO NHAN':<24} "
              f"{'mean(nx) TRUOC':>15} {'mean(nx) SAU':>13} {'#LAT':>5} {'#MAP MO':>8} {'#CHUAN TAC':>11}")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    tot = {"samples": 0, "labeled": 0, "flipped": 0, "ambiguous": 0,
           "canonical": 0, "no_normal": 0}

    for word_key in sorted(vocab.keys()):
        samples = vocab[word_key].get("samples", [])
        n = len(samples)
        if n == 0:
            print(f"  {word_key:<16} {0:>5}  (khong co mau)")
            continue

        nx_before: List[float] = []
        nx_after: List[float] = []
        label_counts: Dict[str, int] = {}
        n_labeled = 0
        n_flipped = 0
        n_ambiguous = 0
        n_canonical = 0
        n_no_normal = 0

        for s in samples:
            traj = get_sample_traj(s)
            if traj is None:
                continue
            tot["samples"] += 1

            if traj.shape[1] < 12:
                # Du lieu 4D/9D cu: khong co kenh phap tuyen -> khong the chan doan.
                n_no_normal += 1
                tot["no_normal"] += 1
                continue

            nx_series = traj[:, 9]
            m_before = float(np.mean(nx_series))
            nx_before.append(m_before)
            max_abs_nx = float(np.max(np.abs(nx_series))) if len(nx_series) > 0 else 0.0
            if max_abs_nx < eps:
                n_ambiguous += 1
                tot["ambiguous"] += 1

            handed = s.get("handedness")
            if handed is not None:
                n_labeled += 1
                tot["labeled"] += 1
                key = str(handed)
                label_counts[key] = label_counts.get(key, 0) + 1

            if s.get("chirality_canonical") is True:
                n_canonical += 1
                tot["canonical"] += 1

            canon, was_flipped = canonicalize_chirality(traj, handedness=handed, ambiguity_eps=eps)
            nx_after.append(float(np.mean(canon[:, 9])))
            if was_flipped:
                n_flipped += 1
                tot["flipped"] += 1

        if label_counts:
            dist = ", ".join(f"{k}:{v}" for k, v in sorted(label_counts.items()))
        else:
            dist = "(khong co nhan)"

        mb = f"{np.mean(nx_before):+.4f}" if nx_before else "n/a"
        ma = f"{np.mean(nx_after):+.4f}" if nx_after else "n/a"

        suffix = f"  [!] {n_no_normal} mau <12D" if n_no_normal else ""
        print(f"  {word_key:<16} {n:>5} {n_labeled:>9} {dist:<24} "
              f"{mb:>15} {ma:>13} {n_flipped:>5} {n_ambiguous:>8} {n_canonical:>11}{suffix}")

    print()
    print(f"  TONG FILE: {tot['samples']} mau | co nhan tay thuan: {tot['labeled']} "
          f"| da danh dau chuan tac: {tot['canonical']} | bi lat: {tot['flipped']} "
          f"| map mo max|nx|<{eps}: {tot['ambiguous']} | thieu phap tuyen (<12D): {tot['no_normal']}")

    if tot["samples"] > 0 and tot["labeled"] == 0:
        print("  [!] CANH BAO: file nay KHONG co mau nao ghi nhan tay thuan -> toan bo phai dung")
        print("      nhanh heuristic max|nx|, vua khong chac chan vua co the pha thong tin dau nx.")
        print("      Moi mau thu MOI da tu dong ghi truong 'handedness'.")

    return tot


def main() -> None:
    args = parse_args()
    files = discover_files(args)

    print("=" * 100)
    print("  CHAN DOAN CHUAN HOA TAY THUAN (CHIRALITY) — CHE DO CHI DOC")
    print("=" * 100)
    print(f"  Dai map mo (ambiguity_eps)     : max|nx| < {args.eps}")
    print(f"  Cac cot bi dao dau khi lat     : {list(CHIRALITY_FLIP_COLUMNS)}  (x, dx, nx)")
    print(f"  So file template se quet       : {len(files)}")
    if not files:
        print("\n  [!] Khong tim thay file template nao. Thu --files <path> hoac --include-backups.")
        return

    grand = {"samples": 0, "labeled": 0, "flipped": 0, "ambiguous": 0,
             "canonical": 0, "no_normal": 0}
    for path in files:
        tot = diagnose_file(path, args.eps)
        for k in grand:
            grand[k] += tot.get(k, 0)

    print()
    print("=" * 100)
    print("  TONG HOP TOAN BO")
    print("=" * 100)
    print(f"  Tong so mau           : {grand['samples']}")
    print(f"  Co nhan tay thuan     : {grand['labeled']}")
    print(f"  Da danh dau chuan tac : {grand['canonical']}")
    print(f"  Se bi lat khi chuan hoa: {grand['flipped']}")
    print(f"  Nam trong dai map mo  : {grand['ambiguous']}")
    print(f"  Thieu phap tuyen (<12D): {grand['no_normal']}")
    print()
    print("  (Script nay khong ghi gi ca — chay lai bao nhieu lan cung an toan.)")


if __name__ == "__main__":
    main()
