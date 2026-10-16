"""
Danh sách người ký (hồ sơ) của dự án: chủ dự án và các tình nguyện viên, dùng TÊN THẬT (từ 2026-09-28).

  - Mã hồ sơ (id) là chữ Latinh không dấu, dùng làm tên file và tiền tố session_id: `Mai`, `Phi`, `NguyenAn`.
  - Chủ dự án có mã cố định `DAT` (hiển thị "Đạt"): bank sản xuất `webcam_templates.json` là của Đạt.
  - Tình nguyện viên: dữ liệu ở `profiles/<mã>.json`, nét thô ở `raw_volunteers/`.
  - Danh sách lưu ở `profiles/people.json` (tự tạo lần đầu; thư mục profiles/ không đưa lên git).

Các công cụ khác (separate_volunteer_raw.py, freeze_volunteer_data.py, dip.py) dùng `volunteer_ids()` để nhận ra
tình nguyện viên theo tên, ngoài quy ước cũ V01..V99.
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(PROJECT_ROOT, "profiles")
PEOPLE_PATH = os.path.join(PROFILES_DIR, "people.json")

OWNER_ID = "DAT"
ROLE_OWNER = "chu_du_an"
ROLE_VOLUNTEER = "tinh_nguyen_vien"

DEFAULT_PEOPLE: Dict[str, Dict[str, str]] = {
    OWNER_ID: {"ten": "Đạt", "vai_tro": ROLE_OWNER},
    "Mai": {"ten": "Mai", "vai_tro": ROLE_VOLUNTEER},
    "Phi": {"ten": "Phi", "vai_tro": ROLE_VOLUNTEER},
}


def slugify(name: str) -> str:
    """'Nguyễn An' -> 'NguyenAn'. Bỏ dấu, bỏ ký tự lạ, mỗi từ viết hoa chữ đầu. Phải bắt đầu bằng chữ cái."""
    s = unicodedata.normalize("NFD", str(name).replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    words = re.findall(r"[A-Za-z0-9]+", s)
    out = "".join(w[:1].upper() + w[1:] for w in words)
    if not out or not out[0].isalpha():
        raise ValueError(f"Tên '{name}' không tạo được mã hồ sơ hợp lệ (phải có ít nhất một chữ cái ở đầu).")
    return out


def load_people(path: str = PEOPLE_PATH) -> Dict[str, Dict[str, str]]:
    """Đọc danh sách; chưa có file thì tạo với mặc định (Đạt, Mai, Phi)."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                if OWNER_ID not in data:
                    data = {OWNER_ID: DEFAULT_PEOPLE[OWNER_ID], **data}
                return data
        except (OSError, ValueError):
            pass
    data = {k: dict(v) for k, v in DEFAULT_PEOPLE.items()}
    save_people(data, path)
    return data


def save_people(data: Dict[str, Dict[str, str]], path: str = PEOPLE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_person(name: str, role: str = ROLE_VOLUNTEER, path: str = PEOPLE_PATH) -> str:
    """Thêm người mới, trả về mã hồ sơ. Lỗi ValueError nếu mã đã tồn tại."""
    pid = slugify(name)
    data = load_people(path)
    if any(pid.casefold() == k.casefold() for k in data):
        raise ValueError(f"Đã có hồ sơ '{pid}'.")
    data[pid] = {"ten": str(name).strip(), "vai_tro": role}
    save_people(data, path)
    return pid


def volunteer_ids(path: str = PEOPLE_PATH) -> List[str]:
    return [k for k, v in load_people(path).items() if v.get("vai_tro") != ROLE_OWNER]


def resolve(name: str, path: str = PEOPLE_PATH) -> Optional[str]:
    """Tên/mã người dùng gõ -> mã hồ sơ ('đạt', 'Dat', 'dat' -> 'DAT'; 'mai' -> 'Mai'). Không thấy -> None."""
    if not name:
        return None
    data = load_people(path)
    q = str(name).strip().casefold()
    for k, v in data.items():
        if q == k.casefold() or q == str(v.get("ten", "")).casefold():
            return k
    try:
        q2 = slugify(name).casefold()
    except ValueError:
        return None
    for k in data:
        if q2 == k.casefold():
            return k
    return None


def skipped_words(pid: str, path: str = PEOPLE_PATH) -> List[str]:
    """Các từ được miễn cho người này (không bắt buộc thu), lưu ở trường `bo_qua_tu`."""
    return [str(w).upper() for w in load_people(path).get(pid, {}).get("bo_qua_tu", [])]


def set_skipped_words(pid: str, words: List[str], path: str = PEOPLE_PATH) -> List[str]:
    data = load_people(path)
    if pid not in data:
        raise ValueError(f"Không có hồ sơ '{pid}'.")
    ws = sorted({str(w).strip().upper() for w in words if str(w).strip()})
    if ws:
        data[pid]["bo_qua_tu"] = ws
    else:
        data[pid].pop("bo_qua_tu", None)
    save_people(data, path)
    return ws


def display_name(pid: str, path: str = PEOPLE_PATH) -> str:
    return load_people(path).get(pid, {}).get("ten", pid)


def profile_path(pid: str) -> str:
    """Bank của người ký: Đạt = bank sản xuất; còn lại = profiles/<mã>.json (đường dẫn tương đối gốc dự án)."""
    return "webcam_templates.json" if pid == OWNER_ID else f"profiles/{pid}.json"


def profile_stats(pid: str) -> Dict[str, int]:
    """Số từ / số mẫu hiện có trong hồ sơ (0 nếu chưa có file)."""
    p = os.path.join(PROJECT_ROOT, profile_path(pid))
    if not os.path.exists(p):
        return {"tu": 0, "mau": 0}
    try:
        with open(p, "r", encoding="utf-8") as f:
            vocab = json.load(f).get("vocabulary", {})
        return {"tu": len(vocab), "mau": sum(len(v.get("samples", [])) for v in vocab.values())}
    except (OSError, ValueError):
        return {"tu": 0, "mau": 0}
