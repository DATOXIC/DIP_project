"""
Tien ich dong bang du lieu cho SPEC 12 (Buoc 0.7).

Chuc nang:
  - Sao chep cac file bank, negative, profiles, notes sang thu muc results/SPEC12_<timestamp>/
  - Tinh SHA-256 tung file de bao dam tinh toan ven
  - Thong ke so luong mau, phien, signer
  - Ham quy doi signer_of(session_id, notes)
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional


def sha256_file(filepath: str) -> str:
    """Tinh ma SHA-256 cua mot file tren dia."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Lay ma commit git hien tai."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def signer_of(session_id: str, notes_data: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Quy doi session_id sang ma nguoi ky (signer_id):
      1. Tra notes.csv (neu co)
      2. Regex ^V\\d{2}(?=_)
      3. Mac dinh 'DAT'
    """
    if not session_id:
        return "DAT"

    # 1. Tra trong notes_data
    if notes_data:
        for row in notes_data:
            if row.get("session_id") == session_id:
                sid = (row.get("signer_id") or "").strip()
                if sid:
                    return sid

    # 2. Regex ^V\d{2}
    m = re.match(r"^(V\d{2})_", session_id)
    if m:
        return m.group(1)

    # 2b. Tình nguyện viên theo tên (people.py): Mai_..., Phi_...
    try:
        import people
        for v in people.volunteer_ids():
            if session_id.startswith(f"{v}_"):
                return v
    except Exception:
        pass

    # 3. Mac dinh la DAT
    return "DAT"


def freeze_spec12_dataset(
    bank_path: str = "webcam_templates.json",
    neg_path: str = "negative_strokes.json",
    profiles_dir: str = "profiles",
    notes_file: str = os.path.join("profiles", "notes.csv"),
    out_dir: Optional[str] = None
) -> str:
    """
    Thuc hien dong bang toan bo du lieu cho SPEC 12 sang OUT.
    Tra ve duong dan thu muc OUT.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if out_dir is None:
        out_dir = os.path.join("results", f"SPEC12_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    profiles_frozen_dir = os.path.join(out_dir, "profiles_frozen")
    os.makedirs(profiles_frozen_dir, exist_ok=True)

    manifest: Dict[str, Any] = {
        "timestamp": stamp,
        "git_commit": get_git_commit(),
        "files": {},
        "summary": {}
    }

    # 1. Sao chep bank_frozen.json
    frozen_bank = os.path.join(out_dir, "bank_frozen.json")
    if os.path.exists(bank_path):
        shutil.copy2(bank_path, frozen_bank)
        manifest["files"]["bank"] = {
            "source": bank_path,
            "frozen": frozen_bank,
            "sha256": sha256_file(frozen_bank)
        }

    # 2. Sao chep neg_frozen.json
    frozen_neg = os.path.join(out_dir, "neg_frozen.json")
    if os.path.exists(neg_path):
        shutil.copy2(neg_path, frozen_neg)
        manifest["files"]["neg"] = {
            "source": neg_path,
            "frozen": frozen_neg,
            "sha256": sha256_file(frozen_neg)
        }

    # 3. Sao chep profiles
    profile_files = []
    if os.path.isdir(profiles_dir):
        for pattern in ["V*.json", "C_*.json", "DAT_cond*.json"]:
            for pf in glob.glob(os.path.join(profiles_dir, pattern)):
                dest_pf = os.path.join(profiles_frozen_dir, os.path.basename(pf))
                shutil.copy2(pf, dest_pf)
                profile_files.append((pf, dest_pf))

    manifest["files"]["profiles"] = [
        {"source": src, "frozen": dst, "sha256": sha256_file(dst)}
        for src, dst in profile_files
    ]

    # 4. Sao chep notes.csv
    frozen_notes = os.path.join(out_dir, "notes_frozen.csv")
    if os.path.exists(notes_file):
        shutil.copy2(notes_file, frozen_notes)
        manifest["files"]["notes"] = {
            "source": notes_file,
            "frozen": frozen_notes,
            "sha256": sha256_file(frozen_notes)
        }

    # Ghi manifest
    manifest_path = os.path.join(out_dir, "frozen_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[Freeze] Da dong bang du lieu thanh cong vao: '{out_dir}'")
    return out_dir
