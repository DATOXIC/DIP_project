"""
Cong cu tach net tho cua tinh nguyen vien va dieu kien ra khoi kho raw_strokes/.
(SPEC 12 - Buoc 0.2)

Mac dinh:
  CHI IN danh sach cac file khop tien to (dry-run).
Khi co --apply:
  DI CHUYEN (shutil.move) cac file khop tien to sang raw_volunteers/<TU>/ giu nguyen ten.
  Idempotent: chay nhieu lan khong thay doi them, khong ghi de file da ton tai o thu muc dich.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
from typing import List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_RAW_DIR = "raw_strokes"
DEFAULT_DEST_DIR = "raw_volunteers"
DEFAULT_REGEX_PATTERN = r"^(V\d{2}_|C_)"


def scan_and_separate(
    raw_dir: str = DEFAULT_RAW_DIR,
    dest_dir: str = DEFAULT_DEST_DIR,
    custom_prefixes: List[str] = None,
    apply: bool = False
) -> Tuple[int, int, int, int]:
    """
    Quet raw_dir va tach cac net tho tinh nguyen vien.
    
    Tra ve tuple:
      (tong_file_quet, so_file_khop, so_file_da_chuyen, so_file_bo_qua_vi_trung)
    """
    if custom_prefixes is None:
        custom_prefixes = []

    # Xay dung bieu thuc chinh quy
    patterns = [r"V\d{2}_", r"C_"]
    for p in custom_prefixes:
        if p:
            patterns.append(re.escape(p))
    regex = re.compile(r"^(" + "|".join(patterns) + ")")

    if not os.path.isdir(raw_dir):
        print(f"[separate_volunteer_raw] Thu muc '{raw_dir}' khong ton tai. Khong co file nao.")
        return 0, 0, 0, 0

    all_files = sorted(glob.glob(os.path.join(raw_dir, "*", "*.json")))
    total_scanned = len(all_files)
    matched_files = []

    for f_path in all_files:
        session_id = None
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                session_id = data.get("session_id")
        except Exception as e:
            # Neu khong doc duoc json, thu kiem tra ten file
            pass

        if not session_id:
            # Du phong kiem tra ten file neu khong co session_id trong metadata
            base_name = os.path.basename(f_path)
            session_id = base_name

        if regex.search(str(session_id)):
            matched_files.append((f_path, session_id))

    num_matched = len(matched_files)
    num_moved = 0
    num_skipped_existing = 0

    print("=" * 80)
    print("  TACH NET THO TINH NGUYEN VIEN / DIEU KIEN (SPEC 12)")
    print("=" * 80)
    print(f"  Thu muc nguon : {raw_dir} (tong so: {total_scanned} file)")
    print(f"  Thu muc dich  : {dest_dir}")
    print(f"  Tien to regex : {regex.pattern}")
    print(f"  Che do        : {'APPLY (THUC THI DI CHUYEN)' if apply else 'DRY-RUN (CHI IN DANH SACH)'}")
    print("-" * 80)

    if not matched_files:
        print("  [OK] Khong tim thay file net tho tinh nguyen vien nao can di chuyen.")
    else:
        print(f"  Tim thay {num_matched} file khop tien to tinh nguyen vien/dieu kien:")
        for idx, (src_path, sess) in enumerate(matched_files, 1):
            rel_path = os.path.relpath(src_path, raw_dir)
            word_dir = os.path.basename(os.path.dirname(src_path))
            fname = os.path.basename(src_path)
            target_path = os.path.join(dest_dir, word_dir, fname)

            if not apply:
                print(f"    [{idx:3d}] DRY-RUN: {rel_path} (session={sess}) -> {target_path}")
            else:
                if os.path.exists(target_path):
                    print(f"    [{idx:3d}] BO QUA (da ton tai): {target_path}")
                    num_skipped_existing += 1
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.move(src_path, target_path)
                    print(f"    [{idx:3d}] DA DI CHUYEN: {rel_path} -> {target_path}")
                    num_moved += 1

    print("-" * 80)
    print(f"  Tong ket: {total_scanned} file da quet, {num_matched} file khop.")
    if apply:
        print(f"  Da di chuyen thanh cong: {num_moved} file | Bo qua trung lap: {num_skipped_existing} file.")
    else:
        print("  Luu y: Day la che do dry-run. Dung them co '--apply' de thuc su di chuyen file.")
    print("=" * 80)

    return total_scanned, num_matched, num_moved, num_skipped_existing


def main():
    parser = argparse.ArgumentParser(
        description="Tach net tho cua tinh nguyen vien va dieu kien tu raw_strokes/ sang raw_volunteers/."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=DEFAULT_RAW_DIR,
        help=f"Thu muc kho tho goc (mac dinh: {DEFAULT_RAW_DIR})."
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=DEFAULT_DEST_DIR,
        help=f"Thu muc dich chua net tinh nguyen vien (mac dinh: {DEFAULT_DEST_DIR})."
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Tien to session_id bo sung can tach (lap duoc, vi du: --prefix V01_ --prefix C_)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Thuc thi di chuyen file. Mac dinh chi quet va in danh sach (dry-run)."
    )
    args = parser.parse_args()

    scan_and_separate(
        raw_dir=args.raw_dir,
        dest_dir=args.dest_dir,
        custom_prefixes=args.prefix,
        apply=args.apply
    )


if __name__ == "__main__":
    main()
