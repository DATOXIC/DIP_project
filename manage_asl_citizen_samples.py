"""
QUẢN LÝ MẪU ASL CITIZEN ĐÃ TRÍCH XUẤT
======================================
Xem tổng quan + xóa mẫu trong asl_citizen_templates.json mà KHÔNG cần mở lại video gốc.

Cách dùng:
  python manage_asl_citizen_samples.py                        # Tổng quan tất cả từ
  python manage_asl_citizen_samples.py --word BYE              # Chi tiết mẫu của từ BYE
  python manage_asl_citizen_samples.py --word BYE --delete 2   # Xóa mẫu số 2 của từ BYE
"""
import argparse
import json
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_PATH = "asl_citizen_templates.json"


def load_db(path):
    if not os.path.exists(path):
        print(f"[!] Không tìm thấy file: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(path, db):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def print_overview(db):
    vocab = db.get("vocabulary", {})
    if not vocab:
        print("[!] Chưa có từ nào trong file.")
        return
    header = f"{'word':<14}{'num_samples':>12}{'GOOD':>8}{'WARNING':>10}{'BAD':>6}{'no_flag':>10}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for word, data in vocab.items():
        samples = data.get("samples", [])
        counts = {"GOOD": 0, "WARNING": 0, "BAD": 0, "no_flag": 0}
        for s in samples:
            flag = s.get("quality_flag")
            counts[flag if flag in counts else "no_flag"] += 1
        print(f"{word:<14}{len(samples):>12}{counts['GOOD']:>8}{counts['WARNING']:>10}{counts['BAD']:>6}{counts['no_flag']:>10}")
    print("=" * len(header))


def print_word_detail(db, word):
    vocab = db.get("vocabulary", {})
    word = word.upper()
    if word not in vocab:
        print(f"[!] Không có từ '{word}' trong file.")
        return
    samples = vocab[word].get("samples", [])
    if not samples:
        print(f"[!] Từ '{word}' chưa có mẫu nào.")
        return
    header = f"{'idx':<5}{'source_video':<40}{'quality':<10}{'dropout%':>10}{'max_gap':>9}{'duration':>10}{'created_at':<20}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, s in enumerate(samples):
        dropout = s.get("dropout_ratio")
        dropout_str = f"{dropout*100:.1f}" if dropout is not None else "N/A"
        max_gap = s.get("max_gap_frames", "N/A")
        flag = s.get("quality_flag", "N/A")
        print(f"{i:<5}{str(s.get('source_video','?'))[:38]:<40}{flag:<10}{dropout_str:>10}{str(max_gap):>9}{str(s.get('duration_sec','?')):>10}{str(s.get('created_at','?')):<20}")
    print("=" * len(header))


def delete_sample(db, path, word, idx):
    vocab = db.get("vocabulary", {})
    word = word.upper()
    if word not in vocab:
        print(f"[!] Không có từ '{word}'.")
        return
    samples = vocab[word].get("samples", [])
    if idx < 0 or idx >= len(samples):
        print(f"[!] Chỉ số {idx} không hợp lệ (từ '{word}' có {len(samples)} mẫu, hợp lệ 0-{len(samples)-1}).")
        return
    removed = samples.pop(idx)
    vocab[word]["num_samples"] = len(samples)
    save_db(path, db)
    print(f"✓ Đã xóa mẫu #{idx} (video: {removed.get('source_video','?')}) khỏi từ '{word}'. Còn lại {len(samples)} mẫu.")


def main():
    parser = argparse.ArgumentParser(description="Quản lý mẫu ASL Citizen đã trích xuất")
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--word", default=None, help="Xem chi tiết mẫu của 1 từ cụ thể")
    parser.add_argument("--delete", type=int, default=None, help="Xóa mẫu theo chỉ số (yêu cầu --word)")
    args = parser.parse_args()

    db = load_db(args.path)

    if args.delete is not None:
        if not args.word:
            print("[!] Phải chỉ định --word khi dùng --delete.")
            sys.exit(1)
        delete_sample(db, args.path, args.word, args.delete)
        return

    if args.word:
        print_word_detail(db, args.word)
    else:
        print_overview(db)


if __name__ == "__main__":
    main()
