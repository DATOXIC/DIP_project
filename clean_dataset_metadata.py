"""
DON DEP metadata.json CUA vsl_dataset: loai bo cac muc "samples" tro toi file .npy
da bi xoa/mat tren dia (phat hien 2026-09-30: 16 muc mo côi khi doi chieu voi
best_model_v3.pth qua toan bo anh -- KHONG anh huong train.py vi train.py tu quet
dia truc tiep, nhung anh huong bat ky script nao doc thang metadata.json (vd doi
chieu confusion matrix, process_external_images.py) va gay FileNotFoundError.

Cach dung:  py clean_dataset_metadata.py [--dataset-dir vsl_dataset]
"""

import argparse
import os
import sys
import time
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dataset_builder import DIPDatasetBuilder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="vsl_dataset")
    p.add_argument("--dry-run", action="store_true", help="Chi in ra, khong ghi file.")
    args = p.parse_args()

    builder = DIPDatasetBuilder(output_dir=args.dataset_dir)

    before = len(builder.samples)
    kept, orphans = [], []
    for s in builder.samples:
        full_path = os.path.join(args.dataset_dir, s.get("path", ""))
        if s.get("path") and os.path.exists(full_path):
            kept.append(s)
        else:
            orphans.append(s)

    print(f"Tong muc trong metadata.json: {before}")
    print(f"Muc con file that tren dia  : {len(kept)}")
    print(f"Muc mo coi (file da mat)    : {len(orphans)}")
    if orphans:
        by_class = {}
        for s in orphans:
            by_class[s.get("class", "?")] = by_class.get(s.get("class", "?"), 0) + 1
        print("  Theo lop:", by_class)
        for s in orphans[:20]:
            print(f"   - {s.get('path')}")
        if len(orphans) > 20:
            print(f"   ... va {len(orphans) - 20} muc khac")

    if args.dry_run:
        print("\n[DRY-RUN] Khong ghi file. Chay lai bo --dry-run de ap dung.")
        return

    builder.samples = kept
    for c_name in list(builder.classes.keys()):
        builder.sample_counts[c_name] = builder._scan_disk_for_class(c_name)
    builder.save_metadata()
    print(f"\nDa don xong: {before} -> {len(kept)} muc. Sample_counts (theo dia that): {builder.sample_counts}")


if __name__ == "__main__":
    main()
