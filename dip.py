"""
DIP — một lệnh cho mọi việc của dự án nhận diện thủ ngữ.

    py dip.py                      # mở CỬA SỔ ỨNG DỤNG (làm mọi việc bằng chuột);  py dip.py --menu = menu chữ
    py dip.py tinh-nguyen          # MỘT lệnh cho cả buổi thu: chọn/tạo người, thu, kiểm tra, thử nhận diện
    py dip.py tien-do              # đã có bao nhiêu người đạt / mục tiêu
    py dip.py nhan-dien            # chạy nhận diện với hồ sơ của Đạt (bank sản xuất); hỏi xác nhận khi không
                                    # chắc là MẶC ĐỊNH BẬT (đòn bẩy chính cho người lạ, xem eval_enroll.py);
                                    # tầng DL dự phòng cũng MẶC ĐỊNH BẬT nếu có best_gesture_model_none.pth
                                    # (xem dl_fusion.py, docs/BAO_CAO_DEEP_LEARNING_HYBRID.md; tắt bằng --tat-dl)
    py dip.py nhan-dien --ho-so Mai   # mẫu xác nhận được CỘNG DỒN vào profiles/Mai.json qua nhiều lần dùng
    py dip.py nhan-dien --khong-xac-nhan   # tắt hỏi xác nhận (quay lại hành vi cũ)
    py dip.py nhan-dien --ho-so Mai --gop-dat   # bank gộp Đạt + CHÍNH Mai (đã đo: cách đúng, KHÔNG dùng
                                    # --gop <người khác> để gộp thêm người khác - đã đo là vô ích/có hại,
                                    # xem docs/BAO_CAO_DEEP_LEARNING_HYBRID.md)
    py dip.py thu --ho-so Phi      # thu dữ liệu cho hồ sơ có sẵn
    py dip.py thu --moi "Nguyễn An"   # tạo hồ sơ mới rồi thu luôn
    py dip.py thu --ho-so Đạt --dieu-kien dark   # Đạt thu điều kiện môi trường (không đụng bank sản xuất)
    py dip.py kiem-tra --ho-so Mai # kiểm tra chất lượng dữ liệu đã thu
    py dip.py danh-gia open --ho-so Mai   # loso | open | baseline | hoc | hybrid
    py dip.py ho-so                # liệt kê hồ sơ;  py dip.py ho-so --moi "Tên"
    py dip.py test                 # chạy toàn bộ kiểm thử
    py dip.py camera               # chẩn đoán camera
    Thêm tham số cho chương trình con: đặt sau `--`, ví dụ  py dip.py nhan-dien -- --time-norm

Quy ước dữ liệu:
  - Đạt (mã DAT): bank sản xuất webcam_templates.json (đã đóng băng — thu vào đó phải xác nhận).
    Điều kiện môi trường của Đạt (dark/bright/far/...) thu vào profiles/DAT_cond.json, nét thô ở raw_volunteers/.
  - Tình nguyện viên (Mai, Phi, ...): profiles/<mã>.json, nét thô ở raw_volunteers/ (KHÔNG lẫn vào raw_strokes/),
    session_id = <mã>_<yyyymmdd>. Danh sách người ở profiles/people.json (xem people.py).
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from typing import List, Optional

import people

ROOT = os.path.dirname(os.path.abspath(__file__))
NOTES_PATH = os.path.join(ROOT, "profiles", "notes.csv")
NOTES_HEADER = ["session_id", "signer_id", "condition", "dominant_hand", "height_cm_est", "distance_cm_est",
                "lighting", "background", "camera", "luma", "new_fps", "imitated", "variant_words", "notes"]
CONDITIONS = ["dark", "bright", "far", "lefthand", "bg2"]
EVAL_KINDS = {
    "loso": "Bảng chấm điểm LOSO trên bank của Đạt (nhanh)",
    "open": "Từ chối từ lạ và nhận người lạ (nhanh, ~15 giây)",
    "baseline": "So sánh DTW với học máy 1-NN/MLP/GRU (~13 phút)",
    "hoc": "Đường cong học: theo số mẫu mỗi từ (~23 phút)",
    "hybrid": "Kết hợp GRU + DTW (~18 phút)",
    "enroll": "Thêm mẫu của người lạ vào bank: 1-3 mẫu/từ có giúp không",
    "confirm": "Mô phỏng một phiên có hỏi xác nhận",
    "adapt": "Mô phỏng tự học không cần nhãn (kết quả: ít giúp)",
    "citizen": "Đo trên video ASL Citizen (không cần chọn người)",
}

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================================ dựng lệnh (thuần, dễ kiểm thử)

def today() -> str:
    return time.strftime("%Y%m%d")


def merge_banks(paths: List[str]) -> dict:
    """Gộp nhiều bank: mẫu của cùng một từ được nối lại; thông tin từ (ngưỡng, độ rộng...) lấy từ bank đầu tiên có từ đó."""
    import json
    merged: dict = {"metadata": {}, "vocabulary": {}}
    for i, path in enumerate(paths):
        with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
            bank = json.load(f)
        if i == 0:
            merged["metadata"] = bank.get("metadata", {})
        for word, d in bank.get("vocabulary", {}).items():
            slot = merged["vocabulary"].setdefault(word, {**{k: v for k, v in d.items() if k != "samples"}, "samples": []})
            slot["samples"].extend(d.get("samples", []))
    merged["vocabulary"] = {w: dict(d, num_samples=len(d["samples"])) for w, d in merged["vocabulary"].items() if d["samples"]}
    return merged


def build_merged_bank(pid: str, gop_dat: bool, gop: Optional[List[str]] = None) -> str:
    """Ghi bank gộp (Đạt nếu gop_dat, các hồ sơ trong gop, rồi chính pid) ra profiles/_gop_<pid>.json, trả về đường dẫn."""
    import json
    order: List[str] = []
    for x in ([people.OWNER_ID] if gop_dat else []) + list(gop or []) + [pid]:
        if x not in order:
            order.append(x)
    merged = merge_banks([people.profile_path(x) for x in order])
    out = f"profiles/_gop_{pid}.json"
    with open(os.path.join(ROOT, out), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    n = sum(len(d["samples"]) for d in merged["vocabulary"].values())
    print(f"[Bank gộp] {' + '.join(people.display_name(x) for x in order)}: {len(merged['vocabulary'])} từ, {n} mẫu -> {out}")
    return out


def cmd_nhan_dien(pid: str, extra: Optional[List[str]] = None, templates: Optional[str] = None,
                  xac_nhan: bool = True, dl_model: Optional[str] = None,
                  dl_nguong: Optional[float] = None, tat_dl: bool = False) -> List[str]:
    args = ["predict_sentence_webcam.py"]
    if not xac_nhan:
        args.append("--no-confirm-enroll")
    if tat_dl:
        args.append("--no-dl")
    if dl_model:
        args += ["--dl-model", dl_model]
    if dl_nguong is not None:
        args += ["--dl-conf-threshold", str(dl_nguong)]
    if templates:
        args += ["--templates", templates]
    elif pid != people.OWNER_ID:
        args += ["--templates", people.profile_path(pid)]
    return args + list(extra or [])


def cmd_thu(pid: str, dieu_kien: Optional[str] = None, ngay: Optional[str] = None,
            extra: Optional[List[str]] = None) -> List[str]:
    """Lệnh thu dữ liệu. Đạt không kèm điều kiện = thu vào bank sản xuất (thư mục raw_strokes mặc định)."""
    d = ngay or today()
    if pid == people.OWNER_ID and not dieu_kien:
        return ["collect_dynamic_templates.py"] + list(extra or [])
    if pid == people.OWNER_ID:
        if dieu_kien not in CONDITIONS:
            raise ValueError(f"Điều kiện phải là một trong {CONDITIONS}.")
        return ["collect_dynamic_templates.py", "--templates", "profiles/DAT_cond.json",
                "--session", f"C_{dieu_kien}_{d}", "--negative-file", "profiles/DAT_cond_neg.json",
                "--raw-dir", "raw_volunteers"] + list(extra or [])
    return ["collect_dynamic_templates.py", "--templates", people.profile_path(pid), "--session", f"{pid}_{d}",
            "--negative-file", f"profiles/{pid}_neg.json", "--raw-dir", "raw_volunteers"] + list(extra or [])


def cmd_kiem_tra(pid: str, bo_qua: Optional[List[str]] = None) -> List[str]:
    a = ["check_volunteer_data.py", "--signer", pid]
    if bo_qua:
        a += ["--bo-qua"] + list(bo_qua)
    return a


def cmd_danh_gia(kind: str, signer: Optional[str], extra: Optional[List[str]] = None) -> List[str]:
    if kind not in EVAL_KINDS:
        raise ValueError(f"Loại đánh giá phải là một trong {list(EVAL_KINDS)}.")
    sg = ["--signer", signer] if signer else []
    if kind == "loso":
        return ["scoreboard.py", "run", "--config", "configs/G1p90.json", "--bank", "webcam_templates.json",
                "--neg", "negative_strokes.json", "--tag", "dip"] + list(extra or [])
    if kind == "open":
        return ["eval_open_set.py", "--config", "configs/G1p90.json", "--config", "configs/C11_base.json"] + sg + list(extra or [])
    if kind == "citizen":
        return ["eval_citizen.py"] + list(extra or [])
    if kind in ("enroll", "confirm", "adapt"):
        tg = ["--targets", signer] if signer else []
        return [f"eval_{kind}.py"] + tg + list(extra or [])
    script = {"baseline": "eval_baselines.py", "hoc": "eval_learning_curve.py", "hybrid": "eval_hybrid.py"}[kind]
    return [script] + sg + list(extra or [])


def cmd_test(extra: Optional[List[str]] = None) -> List[str]:
    return ["-m", "pytest", "-q"] + list(extra or [])


def cmd_camera(extra: Optional[List[str]] = None) -> List[str]:
    return ["camera_diag.py"] + list(extra or [])


def chay(args: List[str]) -> int:
    """Chạy một chương trình Python của dự án, dùng chung cửa sổ dòng lệnh; trả về mã thoát."""
    print(f"\n>>> py {' '.join(args)}\n", flush=True)
    try:
        return subprocess.call([sys.executable] + args, cwd=ROOT)
    except KeyboardInterrupt:
        print("\n(đã dừng)")
        return 130


# ============================================================================ hồ sơ + ghi chú

def notes_da_co(session: str) -> bool:
    if not os.path.exists(NOTES_PATH):
        return False
    with open(NOTES_PATH, "r", encoding="utf-8", newline="") as f:
        return any(r.get("session_id") == session for r in csv.DictReader(f))


def ghi_notes_row(pid: str, session: str, condition: str = "normal", **fields: str) -> None:
    """Ghi MỘT dòng ghi chú buổi thu vào profiles/notes.csv (bỏ qua nếu phiên đã có dòng). fields: dominant_hand,
    distance_cm_est, lighting, camera, notes..."""
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    if notes_da_co(session):
        return
    row = {k: "" for k in NOTES_HEADER}
    row.update(session_id=session, signer_id=pid, condition=condition, imitated="yes")
    row.update({k: v for k, v in fields.items() if k in NOTES_HEADER})
    new_file = not os.path.exists(NOTES_PATH) or os.path.getsize(NOTES_PATH) == 0
    with open(NOTES_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NOTES_HEADER)
        if new_file:
            w.writeheader()
        w.writerow(row)


def ensure_notes_row(pid: str, session: str, hoi: bool = True, condition: str = "normal") -> None:
    """Bảo đảm profiles/notes.csv có đúng một dòng cho phiên này (check_volunteer_data yêu cầu)."""
    if notes_da_co(session):
        return
    fields = {"dominant_hand": "", "distance_cm_est": "", "lighting": "", "camera": "", "notes": ""}
    if hoi:
        try:
            print("\nGhi chú buổi thu (Enter để bỏ qua từng mục):")
            fields["dominant_hand"] = input("  Tay thuận (Right/Left) [Right]: ").strip() or "Right"
            fields["distance_cm_est"] = input("  Khoảng cách ngồi ước lượng (cm): ").strip()
            fields["lighting"] = input("  Ánh sáng (sáng/tối/chói) [sáng]: ").strip() or "sáng"
            fields["camera"] = input("  Camera (vd laptop, điện thoại): ").strip()
            fields["notes"] = input("  Ghi chú khác: ").strip()
        except EOFError:
            pass
    ghi_notes_row(pid, session, condition, **fields)
    print(f"  Đã ghi dòng cho phiên {session} vào profiles/notes.csv")


def liet_ke_ho_so() -> None:
    print("\nHỒ SƠ NGƯỜI KÝ")
    print(f"  {'mã':<12}{'tên':<16}{'vai trò':<18}{'từ':>4}{'mẫu':>6}   file")
    for pid, v in people.load_people().items():
        st = people.profile_stats(pid)
        role = "chủ dự án" if v.get("vai_tro") == people.ROLE_OWNER else "tình nguyện viên"
        print(f"  {pid:<12}{v.get('ten', pid):<16}{role:<18}{st['tu']:>4}{st['mau']:>6}   {people.profile_path(pid)}")


def tao_ho_so(ten: str) -> str:
    pid = people.add_person(ten)
    print(f"Đã tạo hồ sơ '{pid}' (tên: {ten}). File dữ liệu sẽ được tạo khi thu lần đầu.")
    return pid


# ============================================================================ phiên tình nguyện viên (làm tất cả)

TARGET_NGUOI = 8          # số tình nguyện viên nên có cho một kết luận đáng tin

HUONG_DAN_NGUOI_KY = """
  HƯỚNG DẪN CHO NGƯỜI KÝ (đọc cho tình nguyện viên nghe):
   - Ngồi cách camera khoảng một cánh tay, mặt và tay đều thấy trong hình, ánh sáng đủ sáng.
   - Đặt tay xuống nghỉ, rồi ký từ được yêu cầu, rồi hạ tay xuống. Ký tự nhiên, không cần nhanh.
   - Mỗi từ ký 3 lần. Chỉ cần ký theo cách của mình, không cần giống ai.
   - "Nét âm" (cử động KHÔNG phải từ nào: gãi mặt, vẫy tay...) là TÙY CHỌN, không làm cũng được.
   - Không chắc từ nào thì nhờ người thu bỏ qua từ đó (py dip.py bo-qua <tên> TỪ)."""


_TT_CACHE: dict = {}


def _dau_van_tay(pid: str) -> tuple:
    """Chữ ký rẻ của mọi thứ tinh_trang(pid) đọc: đổi khi thêm/xóa/sửa file dữ liệu của người này."""
    def mt(path: str) -> float:
        try:
            return os.stat(path).st_mtime
        except OSError:
            return 0.0

    sig: list = [mt(os.path.join(ROOT, "profiles", f"{pid}.json")), mt(os.path.join(ROOT, "profiles", f"{pid}_neg.json")),
                 mt(NOTES_PATH), tuple(people.skipped_words(pid))]
    for base in ("raw_volunteers", "raw_strokes"):
        d = os.path.join(ROOT, base)
        try:
            sig.append(tuple(sorted((e.name, e.stat().st_mtime) for e in os.scandir(d) if e.is_dir())))
        except OSError:
            sig.append(())
    return tuple(sig)


def tinh_trang(pid: str, chi_cache: bool = False) -> Optional[dict]:
    """Tình trạng dữ liệu của một tình nguyện viên (đạt hay chưa, còn thiếu gì, số từ/mẫu/nét âm).
    Có nhớ kết quả (kiểm tra đầy đủ mất ~1,5 giây vì đọc mọi nét thô); tự tính lại khi dữ liệu đổi.
    chi_cache=True: chỉ trả kết quả đã nhớ nếu còn đúng, không tính mới (trả None) — để giao diện không bị đứng."""
    sig = _dau_van_tay(pid)
    hit = _TT_CACHE.get(pid)
    if hit and hit[0] == sig:
        return hit[1]
    if chi_cache:
        return None
    res = _tinh_trang_that(pid)
    _TT_CACHE[pid] = (sig, res)
    return res


def _tinh_trang_that(pid: str) -> dict:
    """Tình trạng dữ liệu của một tình nguyện viên: đạt hay chưa, còn thiếu gì, số từ/mẫu/nét âm."""
    import json
    import check_volunteer_data as cvd
    skipped = people.skipped_words(pid)
    try:
        ok, fails, thieu, wc, _ = cvd.check_signer_data(pid, skip_words=skipped)
    except Exception as e:                                # hồ sơ chưa có gì -> coi như chưa đạt
        return {"dat": False, "thieu": [f"chưa đọc được dữ liệu ({e})"], "tu_du": 0, "tu_can": 13 - len(skipped),
                "mau": 0, "neg": 0}
    neg = 0
    neg_path = os.path.join(ROOT, "profiles", f"{pid}_neg.json")
    if os.path.exists(neg_path):
        try:
            with open(neg_path, "r", encoding="utf-8") as f:
                neg = len(json.load(f).get("samples", []))
        except (OSError, ValueError):
            pass
    return {"dat": bool(ok), "thieu": list(fails) + list(thieu),
            "tu_du": sum(1 for c in wc.values() if c >= cvd.MIN_SAMPLES_PER_WORD), "tu_can": len(wc),
            "mau": sum(wc.values()), "neg": neg}


def tien_do() -> int:
    """Bảng tiến độ thu dữ liệu tình nguyện viên. Trả về số người đã đạt."""
    print("\nTIẾN ĐỘ THU DỮ LIỆU TÌNH NGUYỆN VIÊN")
    print(f"  {'tên':<14}{'từ đủ':>8}{'mẫu':>6}{'nét âm*':>8}   trạng thái")
    n_dat = 0
    for pid in people.volunteer_ids():
        t = tinh_trang(pid)
        n_dat += t["dat"]
        tt = "ĐẠT" if t["dat"] else "còn thiếu: " + "; ".join(t["thieu"][:2])
        print(f"  {people.display_name(pid):<14}{t['tu_du']:>4}/{t['tu_can']:<3}{t['mau']:>6}{t['neg']:>8}   {tt}")
    print(f"\n  => {n_dat}/{TARGET_NGUOI} người đạt (mục tiêu {TARGET_NGUOI}: kết luận đáng tin; 5 = sơ bộ).   * nét âm: tùy chọn")
    return n_dat


def _co(prompt: str, mac_dinh: bool = True) -> bool:
    s = _hoi(prompt + (" [Enter=có, n=không]: " if mac_dinh else " [y=có, Enter=không]: ")).lower()
    return (s not in ("n", "no", "khong", "không")) if mac_dinh else s in ("y", "yes", "co", "có")


def phien_tinh_nguyen(pid: Optional[str] = None, moi: Optional[str] = None) -> int:
    """MỘT lệnh cho cả buổi thu: chọn/tạo người -> ghi chú -> thu -> kiểm tra (thu bù nếu thiếu) -> thử nhận diện có hỏi xác nhận."""
    if moi:
        try:
            pid = tao_ho_so(moi)
        except ValueError as e:
            print(f"  ! {e}")
            return 1
    if pid is None:
        pid = chon_ho_so(cho_phep_moi=True, chi_tinh_nguyen_vien=True)
    if pid is None:
        return 1
    print(f"\n=== PHIÊN THU CỦA {people.display_name(pid).upper()} ===")
    print(HUONG_DAN_NGUOI_KY)
    if _co("\nKiểm tra camera trước?", mac_dinh=False):
        chay(cmd_camera())
    lan = 0
    while True:
        lan += 1
        hanh_dong_thu(pid, hoi_ghi_chu=(lan == 1), xac_nhan=False)
        t = tinh_trang(pid)
        if t["dat"]:
            print(f"\n  ✓ Dữ liệu của {people.display_name(pid)} ĐẠT ({t['tu_du']}/{t['tu_can']} từ, {t['mau']} mẫu, {t['neg']} nét âm).")
            break
        print("\n  Còn thiếu: " + "; ".join(t["thieu"]))
        if not _co("  Thu bù ngay?"):
            break
    if t["dat"] and _co("\nThử nhận diện ngay với bank của Đạt + mẫu của họ, có hỏi xác nhận?"):
        chay(cmd_nhan_dien(pid, None, build_merged_bank(pid, True), True))
    tien_do()
    return 0


# ============================================================================ chọn hồ sơ / menu

def _hoi(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def chon_ho_so(cho_phep_moi: bool = True, chi_tinh_nguyen_vien: bool = False, can_du_lieu: bool = False) -> Optional[str]:
    """Hiện danh sách hồ sơ, người dùng chọn số (hoặc N để tạo mới). Trả về mã hoặc None nếu huỷ."""
    ds = [(k, v) for k, v in people.load_people().items()
          if not (chi_tinh_nguyen_vien and v.get("vai_tro") == people.ROLE_OWNER)]
    if can_du_lieu:
        ds = [(k, v) for k, v in ds if people.profile_stats(k)["mau"] > 0]
    if not ds and not cho_phep_moi:
        print("Chưa có hồ sơ nào có dữ liệu.")
        return None
    print("\nChọn hồ sơ:")
    for i, (k, v) in enumerate(ds, 1):
        st = people.profile_stats(k)
        print(f"  {i}. {v.get('ten', k)} ({k}) — {st['tu']} từ, {st['mau']} mẫu")
    if cho_phep_moi:
        print("  N. Tạo hồ sơ mới")
    print("  0. Quay lại")
    while True:
        s = _hoi("> ")
        if s in ("0", ""):
            return None
        if cho_phep_moi and s.lower() == "n":
            ten = _hoi("Tên người ký (có dấu cũng được): ")
            if not ten:
                return None
            try:
                return tao_ho_so(ten)
            except ValueError as e:
                print(f"  ! {e}")
                continue
        if s.isdigit() and 1 <= int(s) <= len(ds):
            return ds[int(s) - 1][0]
        print("  Chọn lại nhé.")


def hanh_dong_thu(pid: str, dieu_kien: Optional[str] = None, hoi_ghi_chu: bool = True, xac_nhan: bool = True) -> int:
    if pid == people.OWNER_ID and not dieu_kien:
        print("\n! Bank của Đạt (webcam_templates.json) là bank sản xuất đã đóng băng: thu thêm vào đây sẽ")
        print("  làm các số đo trước đó không còn so sánh được. Muốn thu điều kiện môi trường thì chọn --dieu-kien.")
        if xac_nhan and _hoi("  Vẫn thu vào bank sản xuất? Gõ YES để tiếp tục: ") != "YES":
            print("  Đã huỷ.")
            return 1
    args = cmd_thu(pid, dieu_kien)
    if pid != people.OWNER_ID or dieu_kien:
        sess = args[args.index("--session") + 1]
        ensure_notes_row(pid, sess, hoi=hoi_ghi_chu, condition=dieu_kien or "normal")
    rc = chay(args)
    if pid != people.OWNER_ID and rc == 0:
        print(f"\nThu xong. Kiểm tra chất lượng:  py dip.py kiem-tra --ho-so {pid}")
    return rc


def menu_danh_gia() -> None:
    print("\nĐÁNH GIÁ")
    keys = list(EVAL_KINDS)
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {EVAL_KINDS[k]}")
    print("  0. Quay lại")
    s = _hoi("> ")
    if not s.isdigit() or not (1 <= int(s) <= len(keys)):
        return
    kind = keys[int(s) - 1]
    signer = None
    if kind not in ("loso", "citizen"):
        print("Chọn người ký để làm 'người lạ' (dữ liệu của họ chỉ dùng để đo):")
        signer = chon_ho_so(cho_phep_moi=False, chi_tinh_nguyen_vien=True, can_du_lieu=True)
        if signer is None:
            return
    chay(cmd_danh_gia(kind, signer))


def menu() -> None:
    while True:
        print("\n" + "=" * 52)
        print("  DIP — Nhận diện thủ ngữ")
        print("=" * 52)
        print("  1. PHIÊN TÌNH NGUYỆN VIÊN (làm tất cả: chọn/tạo người, thu, kiểm tra, thử nhận diện)")
        print("  2. Chạy nhận diện (chọn hồ sơ; hỏi xác nhận)")
        print("  3. Thu dữ liệu riêng lẻ (chọn / tạo hồ sơ)")
        print("  4. Kiểm tra chất lượng dữ liệu đã thu")
        print("  5. Đánh giá và so sánh")
        print("  6. Tiến độ thu dữ liệu + danh sách hồ sơ")
        print("  7. Chạy kiểm thử")
        print("  8. Chẩn đoán camera")
        print("  0. Thoát")
        s = _hoi("> ")
        if s in ("0", "q", ""):
            return
        if s == "1":
            phien_tinh_nguyen()
        elif s == "2":
            pid = chon_ho_so(cho_phep_moi=False, can_du_lieu=True)
            if pid:
                gop = pid != people.OWNER_ID and _co("Gộp bank của Đạt vào (khuyên dùng cho người lạ)?")
                chay(cmd_nhan_dien(pid, None, build_merged_bank(pid, True) if gop else None,
                                   _co("Bật hỏi xác nhận khi không chắc?")))
        elif s == "3":
            pid = chon_ho_so(cho_phep_moi=True)
            if not pid:
                continue
            dk = None
            if pid == people.OWNER_ID:
                print("\nĐạt thu:\n  1. Vào bank sản xuất (cần xác nhận)")
                for i, c in enumerate(CONDITIONS, 2):
                    print(f"  {i}. Điều kiện môi trường: {c}")
                t = _hoi("> ")
                if not t.isdigit() or not (1 <= int(t) <= len(CONDITIONS) + 1):
                    continue
                dk = None if t == "1" else CONDITIONS[int(t) - 2]
            hanh_dong_thu(pid, dk)
        elif s == "4":
            pid = chon_ho_so(cho_phep_moi=False, chi_tinh_nguyen_vien=True, can_du_lieu=True)
            if pid:
                chay(cmd_kiem_tra(pid, people.skipped_words(pid)))
        elif s == "5":
            menu_danh_gia()
        elif s == "6":
            liet_ke_ho_so()
            tien_do()
        elif s == "7":
            chay(cmd_test())
        elif s == "8":
            chay(cmd_camera())
        else:
            print("Chọn lại nhé.")


# ============================================================================ dòng lệnh

def _tim(pid_text: Optional[str]) -> str:
    if not pid_text:
        return people.OWNER_ID
    pid = people.resolve(pid_text)
    if pid is None:
        raise SystemExit(f"Không có hồ sơ '{pid_text}'. Xem danh sách: py dip.py ho-so   |   tạo mới: py dip.py thu --moi \"Tên\"")
    return pid


def _mo_cua_so() -> bool:
    """Mở cửa sổ ứng dụng (dip_app.py). Trả False nếu không mở được (thiếu tkinter/màn hình) để quay về menu chữ."""
    try:
        import dip_app
        dip_app.main()
        return True
    except Exception as e:
        print(f"(Không mở được cửa sổ ứng dụng: {e}) -> dùng menu chữ.")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    argv = [a for a in argv if a != "--menu"]
    p = argparse.ArgumentParser(prog="py dip.py", description="Một lệnh cho mọi việc của dự án (không tham số = cửa sổ ứng dụng; --menu = menu chữ).")
    sub = p.add_subparsers(dest="lenh")
    a = sub.add_parser("nhan-dien", help="Chạy nhận diện thời gian thực")
    a.add_argument("--ho-so", default=None, help="Tên/mã hồ sơ (mặc định: Đạt)")
    a.add_argument("--khong-xac-nhan", action="store_true", help="TẮT hỏi xác nhận khi cử chỉ không chắc (mặc định đã BẬT — đòn bẩy chính giúp nhận đúng người lạ, xem eval_enroll.py). Bật lại bằng cách không truyền cờ này.")
    a.add_argument("--tat-dl", action="store_true", help="TẮT tầng DL dự phòng (xem dl_fusion.py). Mặc định đã BẬT tự động nếu "
                                                          "tìm thấy 'best_gesture_model_none.pth' (đã đo: nested CV 96.6% trung bình "
                                                          "4 người lạ thật, xem docs/BAO_CAO_DEEP_LEARNING_HYBRID.md).")
    a.add_argument("--dl-model", default=None, help="Checkpoint .pth tùy chỉnh cho tầng DL (mặc định 'best_gesture_model_none.pth').")
    a.add_argument("--dl-nguong", type=float, default=None, help="Ngưỡng tin cậy softmax cho tầng DL (mặc định 0.38, xem dl_fusion.py).")
    a.add_argument("--gop-dat", action="store_true", help="Gộp thêm bank của Đạt vào hồ sơ đang chọn")
    a.add_argument("--gop", nargs="+", default=None, metavar="HO_SO", help="Gộp thêm các hồ sơ khác, ví dụ --gop Phi Đạt")
    a = sub.add_parser("tinh-nguyen", help="MỘT lệnh cho cả buổi thu: chọn/tạo người, thu, kiểm tra, thử nhận diện")
    a.add_argument("--ho-so", default=None)
    a.add_argument("--moi", default=None, help="Tạo hồ sơ mới với tên này rồi thu")
    sub.add_parser("tien-do", help="Tiến độ thu dữ liệu tình nguyện viên")
    a = sub.add_parser("thu", help="Thu dữ liệu")
    a.add_argument("--ho-so", default=None)
    a.add_argument("--moi", default=None, help="Tạo hồ sơ mới với tên này rồi thu")
    a.add_argument("--dieu-kien", choices=CONDITIONS, default=None, help="(Chỉ Đạt) thu điều kiện môi trường")
    a.add_argument("--khong-hoi", action="store_true", help="Không hỏi ghi chú/xác nhận (dùng khi chạy tự động)")
    a = sub.add_parser("kiem-tra", help="Kiểm tra chất lượng dữ liệu tình nguyện viên")
    a.add_argument("--ho-so", required=True)
    a = sub.add_parser("bo-qua", help="Miễn một số từ cho một người (không bắt buộc thu). Không kèm từ = xem / xoá bằng --xoa")
    a.add_argument("ho_so")
    a.add_argument("tu", nargs="*", help="Ví dụ: YOU YES BAD")
    a.add_argument("--xoa", action="store_true", help="Bỏ miễn (thu đủ 13 từ)")
    a = sub.add_parser("danh-gia", help="Đánh giá / so sánh")
    a.add_argument("loai", choices=list(EVAL_KINDS))
    a.add_argument("--ho-so", default=None, help="Người ký làm 'người lạ' (mặc định: Mai nếu có)")
    a = sub.add_parser("ho-so", help="Liệt kê hồ sơ hoặc tạo mới")
    a.add_argument("--moi", default=None)
    sub.add_parser("test", help="Chạy kiểm thử")
    sub.add_parser("camera", help="Chẩn đoán camera")
    a = sub.add_parser("web", help="Bản web: nhận diện từ video, chọn mốc bắt đầu ngay trên trình phát (http://127.0.0.1:8000)")
    a.add_argument("--port", type=int, default=8000)
    a.add_argument("--khong-mo-trinh-duyet", action="store_true", help="Không tự mở trình duyệt")
    args = p.parse_args(argv)

    if args.lenh is None:
        if "--menu" not in sys.argv and _mo_cua_so():
            return 0
        menu()
        return 0
    if args.lenh == "nhan-dien":
        pid = _tim(args.ho_so)
        if pid != people.OWNER_ID and people.profile_stats(pid)["mau"] == 0:
            raise SystemExit(f"Hồ sơ '{pid}' chưa có dữ liệu. Thu trước: py dip.py thu --ho-so {pid}")
        others = [_tim(x) for x in (args.gop or [])]
        templates = None
        if args.gop_dat or others:
            if pid == people.OWNER_ID:
                raise SystemExit("Hồ sơ Đạt là bank gốc; hãy chọn --ho-so của một người khác để gộp.")
            for x in others:
                if x != people.OWNER_ID and people.profile_stats(x)["mau"] == 0:
                    raise SystemExit(f"Hồ sơ '{x}' chưa có dữ liệu.")
            templates = build_merged_bank(pid, args.gop_dat, others)
        return chay(cmd_nhan_dien(pid, extra, templates, not args.khong_xac_nhan, args.dl_model, args.dl_nguong, args.tat_dl))
    if args.lenh == "tinh-nguyen":
        return phien_tinh_nguyen(_tim(args.ho_so) if args.ho_so else None, args.moi)
    if args.lenh == "tien-do":
        tien_do()
        return 0
    if args.lenh == "thu":
        if args.moi:
            try:
                pid = tao_ho_so(args.moi)
            except ValueError as e:
                raise SystemExit(str(e))
        else:
            pid = _tim(args.ho_so)
        return hanh_dong_thu(pid, args.dieu_kien, hoi_ghi_chu=not args.khong_hoi, xac_nhan=not args.khong_hoi)
    if args.lenh == "kiem-tra":
        pid = _tim(args.ho_so)
        return chay(cmd_kiem_tra(pid, people.skipped_words(pid)))
    if args.lenh == "bo-qua":
        pid = _tim(args.ho_so)
        if args.xoa:
            people.set_skipped_words(pid, [])
        elif args.tu:
            people.set_skipped_words(pid, args.tu)
        sk = people.skipped_words(pid)
        print(f"{people.display_name(pid)}: " + (f"miễn {', '.join(sk)}" if sk else "không miễn từ nào (cần đủ 13 từ)"))
        return 0
    if args.lenh == "danh-gia":
        signer = None
        if args.loai not in ("loso", "citizen"):
            signer = _tim(args.ho_so) if args.ho_so else next(
                (v for v in people.volunteer_ids() if people.profile_stats(v)["mau"] > 0), "Mai")
        return chay(cmd_danh_gia(args.loai, signer, extra))
    if args.lenh == "ho-so":
        if args.moi:
            try:
                tao_ho_so(args.moi)
            except ValueError as e:
                raise SystemExit(str(e))
        liet_ke_ho_so()
        return 0
    if args.lenh == "test":
        return chay(cmd_test(extra))
    if args.lenh == "camera":
        return chay(cmd_camera(extra))
    if args.lenh == "web":
        return chay(["web_app.py", "--port", str(args.port)] + (["--no-browser"] if args.khong_mo_trinh_duyet else []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
