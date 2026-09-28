r"""
Kiem tra chat luong va tinh toan ven cua ho so du lieu tinh nguyen vien (SPEC 12 - Buoc 0.5).

Tieu chuan danh gia:
  - Tong so net hop le >= 45 (duong + am)
  - Moi tu trong 13 tu chuan >= 3 net
  - So net am (negative strokes) >= 8 net
  - Moi net tho trong raw_volunteers co thong tin hop mat (face_box)
  - session_id dung quy uoc (^V\d{2}_...)
  - Dung 1 dong ghi chu tuong ung trong profiles/notes.csv
  - KHONG co net nao cua signer do nam lan trong raw_strokes/
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


REQUIRED_WORDS = [
    "HELLO", "BYE", "ME", "YOU", "EAT",
    "DRINK", "WATER", "SORRY", "YES", "BAD",
    "THANK_YOU", "PLEASE", "HUNGRY"
]

MIN_SAMPLES_PER_WORD = 3
MIN_NEGATIVE_SAMPLES = 8
MIN_TOTAL_SAMPLES = 45


def check_signer_data(
    signer_id: str,
    profiles_dir: str = "profiles",
    raw_volunteers_dir: str = "raw_volunteers",
    raw_strokes_dir: str = "raw_strokes",
    notes_file: str = os.path.join("profiles", "notes.csv")
) -> Tuple[bool, List[str], List[str], Dict[str, int], Dict[str, int]]:
    """
    Kiem tra toan bo ho so cua signer_id.
    
    Tra ve:
      (is_passed, fail_reasons, words_to_supplement, word_counts, handedness_counts)
    """
    fail_reasons: List[str] = []
    words_to_supplement: List[str] = []
    word_counts: Dict[str, int] = {w: 0 for w in REQUIRED_WORDS}
    handedness_counts: Dict[str, int] = {}

    signer_prefix = f"{signer_id}_"

    # 1. Kiem tra file template duong: profiles/<signer_id>.json
    pos_file = os.path.join(profiles_dir, f"{signer_id}.json")
    total_pos_samples = 0
    if not os.path.exists(pos_file):
        fail_reasons.append(f"Khong tim thay file template duong: '{pos_file}'.")
    else:
        try:
            with open(pos_file, "r", encoding="utf-8") as f:
                pos_data = json.load(f)
            vocab = pos_data.get("vocabulary", {})
            for w in REQUIRED_WORDS:
                samples = vocab.get(w, {}).get("samples", [])
                cnt = len(samples)
                word_counts[w] = cnt
                total_pos_samples += cnt
                if cnt < MIN_SAMPLES_PER_WORD:
                    words_to_supplement.append(f"{w} (co {cnt}/{MIN_SAMPLES_PER_WORD}, can thu bu {MIN_SAMPLES_PER_WORD - cnt})")

                # Kiem tra quy uoc session_id trong tung sample
                for s in samples:
                    sess = s.get("session_id", "")
                    if sess and not sess.startswith(signer_prefix):
                        fail_reasons.append(f"Mau tu '{w}' co session_id '{sess}' khong bat dau bang '{signer_prefix}'.")
        except Exception as e:
            fail_reasons.append(f"Loi doc file template duong '{pos_file}': {e}")

    # 2. Kiem tra file mau am: profiles/<signer_id>_neg.json
    neg_file = os.path.join(profiles_dir, f"{signer_id}_neg.json")
    total_neg_samples = 0
    if not os.path.exists(neg_file):
        fail_reasons.append(f"Khong tim thay file mau am: '{neg_file}'.")
    else:
        try:
            with open(neg_file, "r", encoding="utf-8") as f:
                neg_data = json.load(f)
            neg_samples = neg_data.get("samples", [])
            total_neg_samples = len(neg_samples)
            if total_neg_samples < MIN_NEGATIVE_SAMPLES:
                fail_reasons.append(
                    f"So net am chua dat ({total_neg_samples}/{MIN_NEGATIVE_SAMPLES}, can thu bu {MIN_NEGATIVE_SAMPLES - total_neg_samples} net am)."
                )
            for s in neg_samples:
                sess = s.get("session_id", "")
                if sess and not sess.startswith(signer_prefix):
                    fail_reasons.append(f"Mau am co session_id '{sess}' khong bat dau bang '{signer_prefix}'.")
        except Exception as e:
            fail_reasons.append(f"Loi doc file mau am '{neg_file}': {e}")

    # 3. Kiem tra tong so net hop le
    total_samples = total_pos_samples + total_neg_samples
    if total_samples < MIN_TOTAL_SAMPLES:
        fail_reasons.append(
            f"Tong so net hop le chua dat ({total_samples}/{MIN_TOTAL_SAMPLES} net, gom {total_pos_samples} duong + {total_neg_samples} am)."
        )

    # 4. Kiem tra net tho trong raw_volunteers
    volunteer_raw_files = []
    if os.path.isdir(raw_volunteers_dir):
        for rf in glob.glob(os.path.join(raw_volunteers_dir, "*", "*.json")):
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    rdata = json.load(f)
                sess = str(rdata.get("session_id", ""))
                if sess.startswith(signer_prefix):
                    volunteer_raw_files.append((rf, rdata))
            except Exception:
                pass

    if not volunteer_raw_files:
        fail_reasons.append(
            f"Khong tim thay bat ky net tho nao cua '{signer_id}' trong '{raw_volunteers_dir}/'."
        )
    else:
        raw_missing_face_box = 0
        for rf, rdata in volunteer_raw_files:
            frames = rdata.get("frames", [])
            has_face_box_info = any(f.get("face_box") is not None for f in frames) if frames else False
            if not has_face_box_info:
                raw_missing_face_box += 1

            # Thong ke tay thuan
            handed = rdata.get("dominant_handedness")
            if not handed and frames:
                labels = [f.get("handedness", "Right") for f in frames if f.get("handedness")]
                if labels:
                    handed = max(set(labels), key=labels.count)
            handed = handed or "Unknown"
            handedness_counts[handed] = handedness_counts.get(handed, 0) + 1

        if raw_missing_face_box > 0:
            fail_reasons.append(
                f"Co {raw_missing_face_box}/{len(volunteer_raw_files)} net tho thieu thong tin hop mat (face_box)."
            )

    # 5. Kiem tra notes.csv
    if not os.path.exists(notes_file):
        fail_reasons.append(f"Khong tim thay file ghi chu: '{notes_file}'.")
    else:
        notes_rows = []
        try:
            with open(notes_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = (row.get("signer_id") or "").strip()
                    sess = (row.get("session_id") or "").strip()
                    if sid == signer_id or sess.startswith(signer_prefix):
                        notes_rows.append(row)
            if len(notes_rows) == 0:
                fail_reasons.append(f"Chua co dong ghi chu nao cho '{signer_id}' trong '{notes_file}'.")
            elif len(notes_rows) > 1:
                fail_reasons.append(
                    f"Co {len(notes_rows)} dong ghi chu cho '{signer_id}' trong '{notes_file}' (yeu cau dung 1 dong)."
                )
        except Exception as e:
            fail_reasons.append(f"Loi doc file ghi chu '{notes_file}': {e}")

    # 6. Kiem tra ro ri trong raw_strokes/
    leak_files = []
    if os.path.isdir(raw_strokes_dir):
        for rf in glob.glob(os.path.join(raw_strokes_dir, "*", "*.json")):
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    rdata = json.load(f)
                sess = str(rdata.get("session_id", ""))
                if sess.startswith(signer_prefix):
                    leak_files.append(rf)
            except Exception:
                pass

    if leak_files:
        fail_reasons.append(
            f"PHAT HIEN RO RI: Co {len(leak_files)} net tho cua '{signer_id}' nam nham trong '{raw_strokes_dir}/'!\n"
            f"         -> Goi y: Hay chay ngay lenh 'py separate_volunteer_raw.py --apply' de di chuyen."
        )

    is_passed = (len(fail_reasons) == 0) and (len(words_to_supplement) == 0)
    return is_passed, fail_reasons, words_to_supplement, word_counts, handedness_counts


def print_report(
    signer_id: str,
    is_passed: bool,
    fail_reasons: List[str],
    words_to_supplement: List[str],
    word_counts: Dict[str, int],
    handedness_counts: Dict[str, int]
):
    print("=" * 80)
    print(f"  KIEM TRA HO SO TINH NGUYEN VIEN: {signer_id}")
    print("=" * 80)

    # 1. Ket qua chung
    if is_passed:
        print(f"  KET LUAN CHUNG : [DAT] (HO SO HOP LE, SAN SANG CHO DO KIEM)")
    else:
        print(f"  KET LUAN CHUNG : [CHUA DAT] (CAN BO SUNG / XU LY CAC VAN DE DUOI DAY)")

    print("-" * 80)

    # 2. Bang so net 13 tu
    print("  1. Phan bo so net theo tu vung (toi thieu 3 net/tu):")
    total_pos = sum(word_counts.values())
    for w in REQUIRED_WORDS:
        cnt = word_counts.get(w, 0)
        status = "[OK]" if cnt >= MIN_SAMPLES_PER_WORD else "[THIEU]"
        print(f"     - {w:<12}: {cnt:2d} net  {status}")
    print(f"     -> Tong so net duong: {total_pos}")

    # 3. Phan bo tay thuan
    print("\n  2. Phan bo tay thuan (dominant_handedness):")
    total_hand = sum(handedness_counts.values())
    if total_hand == 0:
        print("     - Khong co du lieu tay thuan.")
    else:
        for hand, count in sorted(handedness_counts.items()):
            pct = (count / total_hand) * 100.0
            print(f"     - {hand:<10}: {count:2d} net ({pct:5.1f}%)")

    # 4. Danh sach can thu bu
    print("-" * 80)
    if words_to_supplement:
        print("  3. Danh sach tu can THU BU:")
        for idx, item in enumerate(words_to_supplement, 1):
            print(f"     [{idx}] {item}")
    else:
        print("  3. Danh sach tu can thu bu: [OK] Khong co tu nao thieu net.")

    # 5. Chi tiet ly do khong dat
    if fail_reasons:
        print("\n  4. Chi tiet cac ly do CHUA DAT:")
        for idx, reason in enumerate(fail_reasons, 1):
            print(f"     ({idx}) {reason}")
    else:
        print("\n  4. Kiem tra cac tieu chuan khac: [OK] Moi tieu chuan deu dat.")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Kiem tra tinh hop le va toan ven cua du lieu tinh nguyen vien (SPEC 12)."
    )
    parser.add_argument(
        "--signer",
        type=str,
        required=True,
        help="Ma dinh danh tinh nguyen vien (vi du: V01, V02)."
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="profiles",
        help="Thu muc chua file json ho so (mac dinh: profiles)."
    )
    parser.add_argument(
        "--raw-volunteers-dir",
        type=str,
        default="raw_volunteers",
        help="Thu muc chua net tho tinh nguyen vien (mac dinh: raw_volunteers)."
    )
    parser.add_argument(
        "--raw-strokes-dir",
        type=str,
        default="raw_strokes",
        help="Thu muc kho net tho goc cua Dat (mac dinh: raw_strokes)."
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=os.path.join("profiles", "notes.csv"),
        help="Duong dan file notes.csv (mac dinh: profiles/notes.csv)."
    )
    args = parser.parse_args()

    is_passed, fail_reasons, words_to_supplement, word_counts, handedness_counts = check_signer_data(
        signer_id=args.signer,
        profiles_dir=args.profiles_dir,
        raw_volunteers_dir=args.raw_volunteers_dir,
        raw_strokes_dir=args.raw_strokes_dir,
        notes_file=args.notes
    )

    print_report(
        signer_id=args.signer,
        is_passed=is_passed,
        fail_reasons=fail_reasons,
        words_to_supplement=words_to_supplement,
        word_counts=word_counts,
        handedness_counts=handedness_counts
    )

    sys.exit(0 if is_passed else 1)


if __name__ == "__main__":
    main()
