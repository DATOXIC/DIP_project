"""
CÔNG CỤ KIỂM TRA TÍNH NHẤT QUÁN DỰ ÁN (PRE-FLIGHT CONSISTENCY GATE)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

Mục đích:
  Bắt sớm lỗi "từ vựng có trong dynamic_templates.json nhưng KHÔNG có trọng số
  tương ứng trong ASL_ALEX_2.0/sign_weights_asllex.json" — hiện tại lỗi này chỉ
  được phát hiện lúc khởi động app thật (predict_sentence_webcam.py) qua một
  `raise KeyError` cứng bên trong
  GoldenTriadDTWMatcher.validate_weights_against_vocabulary() khi strict_weights=True.

  Script này chạy ĐỘC LẬP, không cần khởi động webcam/CNN, nên có thể chạy ngay
  sau khi thêm từ mới bằng collect_dynamic_templates.py, trước khi mở
  predict_sentence_webcam.py — dùng như một pre-flight gate (CI-friendly, trả về
  exit code khác 0 nếu có lỗi nghiêm trọng).

Cách dùng:
  python validate_project_consistency.py
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import sys
import json
from typing import Any, Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dynamic_dtw_engine import GoldenTriadDTWMatcher

# Hai đường dẫn mặc định giống hệt CONFIG["templates_path"] / CONFIG["sign_weights_path"]
# trong predict_sentence_webcam.py.
TEMPLATES_PATH = "dynamic_templates.json"
SIGN_WEIGHTS_PATH = "ASL_ALEX_2.0/sign_weights_asllex.json"

WEIGHT_SUM_TOLERANCE = 1e-4


def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[!] Không tìm thấy file: {path}")
        return {}
    except Exception as e:
        print(f"[!] Lỗi đọc {path}: {e}")
        return {}


def canonicalize(word_key: str) -> str:
    """Áp dụng cùng bảng ánh xạ bí danh GoldenTriadDTWMatcher.SIGN_KEY_ALIASES."""
    raw_k = str(word_key).strip().upper()
    return GoldenTriadDTWMatcher.SIGN_KEY_ALIASES.get(raw_k, raw_k)


def check_weight_sum(word_key: str, w_entry: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Mô phỏng lại logic kiểm tra tổng trọng số P+A+JM+N ~= 1.0 giống hệt
    GoldenTriadDTWMatcher.load_sign_weights() (dung sai 1e-4).
    Trả về (is_valid, message).
    """
    w_dict = w_entry.get("weights", w_entry) if isinstance(w_entry, dict) else {}
    for required_field in ["P", "A", "JM", "N"]:
        if required_field not in w_dict:
            return False, f"thiếu trường trọng số '{required_field}'"
    try:
        p = float(w_dict["P"])
        a = float(w_dict["A"])
        jm = float(w_dict["JM"])
        n = float(w_dict["N"])
    except (TypeError, ValueError):
        return False, "trọng số không phải số hợp lệ"

    weight_sum = p + a + jm + n
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        return False, f"tổng trọng số != 1.0 (Tổng={weight_sum:.6f})"
    return True, ""


def main() -> None:
    print("=" * 78)
    print("  KIỂM TRA TÍNH NHẤT QUÁN: TEMPLATES <-> TRỌNG SỐ ASL-LEX 2.0")
    print("=" * 78)
    print(f"  Templates:    {TEMPLATES_PATH}")
    print(f"  Sign Weights: {SIGN_WEIGHTS_PATH}")
    print("-" * 78)

    templates_data = load_json(TEMPLATES_PATH)
    weights_data = load_json(SIGN_WEIGHTS_PATH)

    vocabulary: Dict[str, Any] = templates_data.get("vocabulary", {})
    has_critical_error = False

    if not vocabulary:
        print("[!] CẢNH BÁO: Không nạp được từ vựng nào từ templates (file rỗng hoặc không tồn tại).")
    if not weights_data:
        print("[!] CẢNH BÁO: Không nạp được trọng số nào từ sign_weights (file rỗng hoặc không tồn tại).")

    weights_keys_upper = {str(k).strip().upper() for k in weights_data.keys()}

    # ------------------------------------------------------------------
    # 1. Từ có trong templates nhưng THIẾU trọng số -> LỖI NGHIÊM TRỌNG
    # ------------------------------------------------------------------
    missing_weights: List[Tuple[str, str, str]] = []  # (word_key, display_name, canonical_key)
    canonical_words_seen = set()
    for word_key, word_data in vocabulary.items():
        display_name = word_data.get("display_name", word_key) if isinstance(word_data, dict) else word_key
        canonical_k = canonicalize(word_key)
        canonical_words_seen.add(canonical_k)
        if canonical_k not in weights_keys_upper:
            missing_weights.append((word_key, display_name, canonical_k))

    print(f"\n[1] Từ vựng trong templates: {len(vocabulary)} | Trọng số ASL-LEX: {len(weights_data)}")

    if missing_weights:
        has_critical_error = True
        print(f"\n[CRITICAL] {len(missing_weights)} từ trong '{TEMPLATES_PATH}' THIẾU trọng số trong "
              f"'{SIGN_WEIGHTS_PATH}' — SẼ CRASH ứng dụng thật khi chạy với strict_weights=True:")
        for word_key, display_name, canonical_k in missing_weights:
            print(f"    ✗ {word_key} (display: '{display_name}', canonical: '{canonical_k}') -> "
                  f"KHÔNG có trong sign_weights_asllex.json")
    else:
        print("    ✓ Mọi từ vựng trong templates đều có trọng số Per-Sign tương ứng.")

    # ------------------------------------------------------------------
    # 2. Trọng số không có từ vựng tương ứng trong templates -> CHỈ LÀ THÔNG TIN
    # ------------------------------------------------------------------
    orphan_weights = sorted(weights_keys_upper - canonical_words_seen)
    print(f"\n[2] Trọng số ASL-LEX KHÔNG có từ vựng tương ứng trong templates (thông tin, KHÔNG phải lỗi):")
    if orphan_weights:
        for w in orphan_weights:
            print(f"    - {w} (có thể là từ vựng dự kiến thu thập sau, chưa chạy collect_dynamic_templates.py)")
    else:
        print("    (không có)")

    # ------------------------------------------------------------------
    # 3. Kiểm tra tổng trọng số P+A+JM+N của mỗi entry ~= 1.0
    # ------------------------------------------------------------------
    print(f"\n[3] Kiểm tra tổng trọng số P + A + JM + N ≈ 1.0 (dung sai {WEIGHT_SUM_TOLERANCE}):")
    bad_weight_entries = []
    for raw_key, w_entry in weights_data.items():
        is_valid, msg = check_weight_sum(raw_key, w_entry)
        if not is_valid:
            bad_weight_entries.append((raw_key, msg))

    if bad_weight_entries:
        has_critical_error = True
        for raw_key, msg in bad_weight_entries:
            print(f"    ✗ {raw_key}: {msg}")
    else:
        print("    ✓ Mọi entry trọng số đều có tổng ≈ 1.0.")

    # ------------------------------------------------------------------
    # Kết luận
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    if has_critical_error:
        print("  KẾT QUẢ: THẤT BẠI — có lỗi nghiêm trọng cần khắc phục trước khi chạy app.")
        print("=" * 78)
        sys.exit(1)
    else:
        print("  KẾT QUẢ: HỢP LỆ — dự án sẵn sàng chạy predict_sentence_webcam.py.")
        print("=" * 78)
        sys.exit(0)


if __name__ == "__main__":
    main()
