"""
TRÍCH XUẤT LẠI BANK ASL CITIZEN SANG 24D — KHÔNG CẦN CẮT TAY LẠI (SPEC 03, NHIỆM VỤ 4)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

VÌ SAO CÓ SCRIPT NÀY?
---------------------
Mỗi mẫu trong file template Citizen đã lưu sẵn `source_video` + `trim_range`, và video gốc vẫn
còn trong ASL_Citizen_videos/<TỪ>/. Nên để thêm 4 nhóm đặc trưng mới (z_norm, mốc cơ thể, độ xòe
ngón, số nhịp lặp) ta chỉ việc MỞ LẠI đúng video, chạy lại MediaPipe (tay) + bộ dò mặt trên đúng
đoạn `trim_range` — không phải cắt tay lại 139 video.

CÁCH LÀM (đảm bảo đúng CÙNG công thức với đường trích xuất tương tác)
---------------------------------------------------------------------
Script dùng lại ASLCitizenTrimmer.analyze_video(frame_range=...) và build_sample_dict() của
inspect_and_trim_asl_citizen.py, nên vector 24D sinh ra ĐÚNG bằng thứ mà công cụ cắt tương tác sẽ
sinh ra nếu cắt lại cùng đoạn. Chỉ nạp đoạn [start-4, end+4] (đệm 4 frame để cơ chế nội suy khi mất
dấu tay 1-3 frame ở biên đoạn cắt hoạt động y như lúc cắt gốc).

  - File cũ được GIỮ NGUYÊN (mặc định ghi ra asl_citizen_templates_24d.json; --force sẽ tự backup).
  - Giữ toàn bộ metadata gốc: source_video, trim_range, quality_flag, dropout_ratio,
    max_gap_frames, created_at, và cả hand_shapes (nhãn CNN K=32; không chạy lại CNN vì
    lambda_shape = 0.0 — CNN chậm và không ảnh hưởng đặc trưng mới).
  - Áp canonicalize_chirality bằng nhãn tay thuận MediaPipe (nhánh chính thống), như đường tương tác.

CẢNH BÁO: nếu tỉ lệ mẫu KHÔNG tìm thấy mặt > 20%, nhóm mốc cơ thể không đo được trên bank này —
script vẫn ghi file nhưng thoát với mã 3 và in cảnh báo DỪNG (đừng tự ý bỏ nhóm hay điền số giả).

CÁCH DÙNG
---------
    py reextract_citizen_batch.py --src archive/backups/asl_citizen_templates.backup_before_bye_delete_20260918_183236.json
    py reextract_citizen_batch.py --limit 5            # thử nhanh 5 mẫu đầu, ghi ra file tạm
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import os
import sys
import json
import time
import shutil
import argparse
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from backup_utils import backup_path_for
from inspect_and_trim_asl_citizen import ASLCitizenTrimmer

PAD_FRAMES = 4                 # đệm quanh trim_range cho cơ chế nội suy 1-3 frame
NO_FACE_STOP_RATIO = 0.20      # > 20% mẫu không thấy mặt => nhóm mốc cơ thể không đo được
KEEP_KEYS = ("source_video", "trim_range", "quality_flag", "dropout_ratio", "max_gap_frames",
             "created_at", "duration_sec")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trich xuat lai bank ASL Citizen sang 24D (khong cat tay lai).")
    p.add_argument("--src", type=str, default="asl_citizen_templates.json",
                   help="File template Citizen 17D nguon (co source_video + trim_range).")
    p.add_argument("--out", type=str, default="asl_citizen_templates_24d.json",
                   help="File 24D dau ra (file nguon KHONG bi dong toi).")
    p.add_argument("--video-root", type=str, default="ASL_Citizen_videos", help="Thu muc video goc.")
    p.add_argument("--force", action="store_true", help="Cho phep ghi de --out (tu dong backup truoc).")
    p.add_argument("--limit", type=int, default=None, help="Chi xu ly N mau dau (de thu nhanh).")
    return p.parse_args()


def find_video(video_root: str, word: str, fname: str) -> Optional[str]:
    cand = os.path.join(video_root, word, fname)
    if os.path.exists(cand):
        return cand
    for root, _dirs, files in os.walk(video_root):
        if fname in files:
            return os.path.join(root, fname)
    return None


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.src):
        print(f"[!] Khong tim thay file nguon: {args.src}")
        sys.exit(1)
    if os.path.exists(args.out) and not args.force:
        print(f"[!] TU CHOI GHI DE: '{args.out}' da ton tai. Them --force (se tu backup) neu that su muon.")
        sys.exit(2)

    with open(args.src, "r", encoding="utf-8") as f:
        src_db = json.load(f)
    src_vocab: Dict[str, Any] = src_db.get("vocabulary", {})
    n_total_src = sum(len(e.get("samples", [])) for e in src_vocab.values())

    print("=" * 90)
    print("  TRICH XUAT LAI BANK ASL CITIZEN -> 24D")
    print("=" * 90)
    print(f"  File nguon : {args.src}  ({n_total_src} mau / {len(src_vocab)} tu)")
    print(f"  File dau ra: {args.out}")

    trimmer = ASLCitizenTrimmer(input_dir=args.video_root, output_json=args.out, load_cnn=False)
    if trimmer.face_detector is None:
        print("\n[!] CHAN DUONG: khong nap duoc bo do mat (thieu blaze_face_short_range.tflite va khong tai duoc).")
        print("    DUNG o day, khong dien du lieu gia. Hay tai model thu cong roi chay lai.")
        sys.exit(4)

    out_vocab: Dict[str, Any] = {}
    n_done = n_fail = n_noface = n_flip_disagree = 0
    face_frame_ratios: List[float] = []
    diffs_17: List[float] = []
    fail_list: List[str] = []
    t0 = time.time()
    idx = 0

    for word, entry in src_vocab.items():
        new_entry = {k: v for k, v in entry.items() if k != "samples"}
        new_entry["samples"] = []
        for smp in entry.get("samples", []):
            if args.limit is not None and idx >= args.limit:
                break
            idx += 1
            fname = smp.get("source_video")
            tr = smp.get("trim_range")
            tag = f"[{idx}/{n_total_src if args.limit is None else min(args.limit, n_total_src)}] {word:<10} {fname}"
            if not fname or not tr:
                print(f"{tag}  -> THIEU source_video/trim_range, bo qua")
                n_fail += 1
                fail_list.append(f"{word}/{fname}: thieu metadata")
                continue
            vpath = find_video(args.video_root, word, fname)
            if vpath is None:
                print(f"{tag}  -> KHONG THAY VIDEO, bo qua")
                n_fail += 1
                fail_list.append(f"{word}/{fname}: khong thay video")
                continue

            start, end = int(tr[0]), int(tr[1])
            lo = max(0, start - PAD_FRAMES)
            hi = end + PAD_FRAMES
            data = trimmer.analyze_video(vpath, frame_range=(lo, hi), run_cnn=False)
            if not data or data.get("total_frames", 0) == 0:
                print(f"{tag}  -> KHONG DOC DUOC VIDEO, bo qua")
                n_fail += 1
                fail_list.append(f"{word}/{fname}: khong doc duoc video")
                continue
            s_rel = start - lo
            e_rel = min(end - lo, data["total_frames"] - 1)
            new = trimmer.build_sample_dict(fname, data, s_rel, e_rel)
            if new is None:
                print(f"{tag}  -> build_sample_dict that bai, bo qua")
                n_fail += 1
                fail_list.append(f"{word}/{fname}: build that bai")
                continue

            # Giu metadata goc + nhan dang tay CNN goc (K=32).
            for k in KEEP_KEYS:
                if k in smp:
                    new[k] = smp[k]
            if smp.get("hand_shapes") and len(smp["hand_shapes"]) == len(new["hand_shapes"]):
                new["hand_shapes"] = smp["hand_shapes"]
            new["reextracted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            new["reextract_pad_frames"] = PAD_FRAMES

            # --- Chan doan ---
            face_ratio = float(np.mean([data["face_boxes"][i] is not None for i in range(s_rel, e_rel + 1)]))
            face_frame_ratios.append(face_ratio)
            if not new["body_anchor_valid"]:
                n_noface += 1

            # Doi chieu 17 cot dau voi ban cu (ban cu chua chuan hoa chirality, nen thu ca hai dau).
            old = np.array(smp["traj_features"], dtype=np.float32)
            newt = np.array(new["traj_features"], dtype=np.float32)
            if old.shape[0] == newt.shape[0]:
                old_f = old.copy()
                for c in (0, 2, 9):
                    old_f[:, c] = -old_f[:, c]
                d_same = float(np.max(np.abs(newt[:, :17] - old[:, :17])))
                d_flip = float(np.max(np.abs(newt[:, :17] - old_f[:, :17])))
                diffs_17.append(min(d_same, d_flip))
            print(f"{tag}  frames[{lo}..{min(hi, lo + data['total_frames'] - 1)}]  mat={face_ratio*100:5.1f}%  "
                  f"rep={new['rep_count']}  body={'OK' if new['body_anchor_valid'] else 'KHONG MAT'}  "
                  f"flip={new['chirality_flipped_on_save']}")
            new_entry["samples"].append(new)
            n_done += 1
        new_entry["num_samples"] = len(new_entry["samples"])
        out_vocab[word] = new_entry
        if args.limit is not None and idx >= args.limit:
            break

    out_db = {
        "metadata": {
            "k_resample_frames": 32,
            "feature_dim": 24,
            "description": "ASL Citizen Extracted 24D Templates (re-extracted, khong cat tay lai)",
            "created_at": src_db.get("metadata", {}).get("created_at"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reextracted_from": args.src,
            "reextract_pad_frames": PAD_FRAMES,
            "chirality_canonical": True,
        },
        "vocabulary": out_vocab,
    }
    if os.path.exists(args.out):
        bak = backup_path_for(args.out, "backup_before_reextract")
        shutil.copy2(args.out, bak)
        print(f"[Backup] {args.out} -> {bak}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_db, f, indent=2, ensure_ascii=False)

    n_all = n_done + n_fail
    noface_ratio = (n_noface / n_done) if n_done else 1.0
    print("\n" + "=" * 90)
    print("  TONG KET")
    print("=" * 90)
    print(f"  Thanh cong                      : {n_done}/{n_all} mau ({time.time() - t0:.0f}s)")
    print(f"  That bai                        : {n_fail}")
    for msg in fail_list:
        print(f"      - {msg}")
    print(f"  KHONG TIM THAY MAT (moi frame)  : {n_noface}/{n_done} mau = {noface_ratio*100:.1f}%")
    if face_frame_ratios:
        arr = np.array(face_frame_ratios)
        print(f"  Ti le frame co hop mat/stroke   : trung binh {arr.mean()*100:.1f}% | nho nhat {arr.min()*100:.1f}% | "
              f"stroke <50% frame co mat: {(arr < 0.5).sum()}")
    if diffs_17:
        d = np.array(diffs_17)
        print(f"  Doi chieu 17 cot dau voi ban cu : max|delta| trung vi {np.median(d):.4f} | lon nhat {d.max():.4f} "
              f"| so mau lech > 0.05: {(d > 0.05).sum()}/{len(d)}")
    print(f"  Da ghi: {args.out}")
    if noface_ratio > NO_FACE_STOP_RATIO:
        print(f"\n[!] DUNG VA BAO CAO: ti le khong tim thay mat {noface_ratio*100:.1f}% > {NO_FACE_STOP_RATIO*100:.0f}% "
              f"=> nhom moc co the KHONG do duoc tren bank nay.")
        sys.exit(3)


if __name__ == "__main__":
    main()
