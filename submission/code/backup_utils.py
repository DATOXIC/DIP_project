"""
Tien ich chung de tinh duong dan file backup.

Quy uoc: moi backup cua file nam TRONG project deu duoc dat o archive/backups/
(thay vi nam rai rac canh file goc). File nam NGOAI project (vi du thu muc tam cua test)
thi backup van dat canh file goc de khong chen vao repo.
"""

import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(PROJECT_ROOT, "archive", "backups")


def _is_inside_project(path: str) -> bool:
    try:
        return os.path.commonpath([PROJECT_ROOT, os.path.abspath(path)]) == PROJECT_ROOT
    except ValueError:  # khac o dia tren Windows
        return False


def backup_path_for(src_path: str, tag: str, stem: str = None) -> str:
    """
    Tra ve duong dan backup '<stem>.<tag>_<YYYYmmdd_HHMMSS>.json' va tao san thu muc dich.

    - stem mac dinh = ten file goc bo duoi mo rong.
    - File trong project -> archive/backups/; file ngoai project -> canh file goc.
    """
    if stem is None:
        stem = os.path.splitext(os.path.basename(src_path))[0]
    name = f"{stem}.{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    dest_dir = BACKUP_DIR if _is_inside_project(src_path) else (os.path.dirname(os.path.abspath(src_path)) or ".")
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, name)
