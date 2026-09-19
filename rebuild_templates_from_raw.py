"""
TÁI TẠO TEMPLATE 24D (17D + 7 CỘT SPEC 03) TỪ KHO STROKE THÔ (REBUILD TEMPLATES FROM RAW ARCHIVE)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

VẤN ĐỀ NÀY GIẢI QUYẾT ĐIỀU GÌ?
------------------------------
Mỗi lần bộ đặc trưng đổi (4D -> 9D -> 12D -> 17D), mọi mẫu đã thu trước đó lập tức
mất giá trị vì không thể suy ra vector mới từ vector cũ — buộc phải ĐI THU LẠI TỪ ĐẦU.
Đó là công sức đã mất một lần rồi.

Từ nay, collect_dynamic_templates.py lưu song song MỘT file thô cho mỗi stroke trong
raw_strokes/<TU>/<timestamp>.json, chứa đủ 21 mốc bàn tay + palm_len + góc gập + pháp
tuyến + nhãn dáng tay + nhãn tay thuận + timestamp CHO TỪNG FRAME. Script này đọc kho
thô đó và TÍNH LẠI vector 17D bằng ĐÚNG các công thức production đang dùng:

    compute_joint_flexion_angles(lm_3d)              -> 5 góc gập ngón (dims 4:9)
    compute_palm_normal_vector(lm_3d, handedness)    -> pháp tuyến (dims 9:12)
    neo gốc tại frame đầu stroke, chia palm_len[0]   -> (x, y)  (dims 0:2)
    sai phân từng frame                              -> (dx, dy) (dims 2:4) & d(góc) (dims 12:17)
    resample_9d_gesture_sequence(..., k_target=32)   -> chuẩn hóa thời gian về K = 32
    canonicalize_chirality(..., handedness)          -> đưa về không gian dấu chuẩn tắc

SPEC 03 — 7 CỘT MỚI + rep_count (cùng công thức với predict / collect / inspect, xem dynamic_features.py):
    z_norm (17), mốc cơ thể (18:20), độ xòe ngón (20:24); rep_count lưu vào dict mẫu.

BỐN THẾ HỆ SCHEMA CỦA KHO THÔ (đọc được CẢ BỐN):
    raw_stroke_v1.0 : KHÔNG có "face_box" => không tái tạo được mốc cơ thể. Stroke đó được đặt
                      body_anchor_valid=False (w_b bị vô hiệu cho mẫu này) và in cảnh báo. Thiếu session_id -> "unknown".
    raw_stroke_v1.1 : mỗi frame có "face_box": [cx, cy, w, h] hoặc null => tái tạo đầy đủ. Thiếu session_id -> "unknown".
    raw_stroke_v1.2 : bổ sung session_id top-level và trong meta => tái tạo đầy đủ kèm nhãn phiên thu.
    raw_stroke_v1.3 : bổ sung capture_fourcc và capture_fps_reported vào top-level metadata (SPEC 05B).

=> Đổi đặc trưng lần sau chỉ tốn một lệnh rebuild offline, KHÔNG phải thu lại mẫu.

LƯU Ý VỀ ĐỘ TRUNG THỰC (quan trọng, đọc kỹ)
-------------------------------------------
Tọa độ cổ tay dùng bản ĐÃ LÀM TRƠN EMA (wrist_smoothed) đọc từ file thô, KHÔNG tính lại
EMA, vì trạng thái bộ làm trơn phụ thuộc cả các frame TRƯỚC khi stroke bắt đầu — thông
tin đó không nằm trong file thô. Đây là lý do file thô lưu cả wrist_raw lẫn wrist_smoothed.

CÁCH DÙNG
---------
    py rebuild_templates_from_raw.py --out dynamic_templates_rebuilt.json
    py rebuild_templates_from_raw.py --out dynamic_templates.json --force
    py rebuild_templates_from_raw.py --out x.json --use-stored-features   # dùng góc/pháp tuyến đã lưu
    py rebuild_templates_from_raw.py --verify-against dynamic_templates.json  # đo sai số round-trip
"""

import os
import sys
import glob
import json
import time
import shutil
import argparse
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from predict_sentence_webcam import (
    compute_joint_flexion_angles,
    compute_palm_normal_vector,
)
from dynamic_features import (
    compute_extended_columns,
    append_extended_columns,
    compute_repetition_count,
)
from dynamic_dtw_engine import (
    resample_9d_gesture_sequence,
    canonicalize_chirality,
    is_stroke_duration_valid,
    STROKE_DURATION_MIN_SEC,
    STROKE_DURATION_MAX_SEC,
)

RAW_STROKES_DIR = "raw_strokes"
K_RESAMPLE = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tai tao file template 17D tu kho stroke tho raw_strokes/."
    )
    p.add_argument("--raw-dir", type=str, default=RAW_STROKES_DIR,
                   help=f"Thu muc kho tho (mac dinh: {RAW_STROKES_DIR}).")
    p.add_argument("--out", type=str, default=None,
                   help="File template JSON dau ra (BAT BUOC tru khi chi --verify-against).")
    p.add_argument("--force", action="store_true",
                   help="Cho phep ghi de file --out da ton tai (van tu dong backup truoc).")
    p.add_argument("--use-stored-features", action="store_true",
                   help="Dung goc gap / phap tuyen DA LUU thay vi tinh lai tu 21 moc ban tay.")
    p.add_argument("--no-duration-gate", action="store_true",
                   help="Khong loai stroke nam ngoai cong thoi luong [0.40s, 2.50s].")
    p.add_argument("--verify-against", type=str, default=None,
                   help="Doi chieu vector tai tao voi mot file template da co (do sai so round-trip).")
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="Nguong sai so round-trip coi la DAT (mac dinh 1e-4).")
    return p.parse_args()


# =============================================================================
# TÁI TẠO VECTOR 17D TỪ MỘT FILE STROKE THÔ
# =============================================================================

def recompute_palm_len(lm_px: np.ndarray) -> float:
    """
    Tinh lai Palm Scale bang DUNG cong thuc trong collect_dynamic_templates.py /
    predict_sentence_webcam.py (chong co ngan phoi canh - Foreshortening Invariant):
        palm_len = max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)
    Luu y lm_px[:, 2] da la z * frame_width, dung nhu production.
    """
    lm0, lm5, lm9, lm17 = lm_px[0], lm_px[5], lm_px[9], lm_px[17]
    palm_2d = float(np.hypot(lm9[0] - lm0[0], lm9[1] - lm0[1]))
    palm_3d = float(np.sqrt((lm9[0] - lm0[0]) ** 2 + (lm9[1] - lm0[1]) ** 2 + (lm9[2] - lm0[2]) ** 2))
    knuckle_span = float(np.hypot(lm17[0] - lm5[0], lm17[1] - lm5[1]))
    return max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)


def rebuild_one_stroke(
    raw: Dict[str, Any],
    use_stored_features: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Tinh lai vector 24D (17D + z_norm + moc co the + do xoe ngon) + nhan dang tay + rep_count
    cho 1 stroke tho. Tra ve sample_dict san sang ghi vao file template, hoac None neu stroke
    khong du dieu kien. Doc duoc ca schema raw_stroke_v1.0 (khong co face_box) va v1.1.
    """
    frames = raw.get("frames", [])
    n = len(frames)
    if n < 8:
        return None

    # --- Toa do co tay: dung ban DA LAM TRON EMA (xem ghi chu dau file) ---
    wrists = np.array([f["wrist_smoothed"] for f in frames], dtype=np.float32)

    # --- Goc gap & phap tuyen: tinh lai tu 21 moc, hoac dung ban da luu ---
    handed_labels = [f.get("handedness", "Right") for f in frames]
    dominant_handed = max(set(handed_labels), key=handed_labels.count) if handed_labels else "Right"

    if use_stored_features:
        angles_mat = np.array([f["flexion_angles"] for f in frames], dtype=np.float32)
        normals_mat = np.array([f["palm_normal"] for f in frames], dtype=np.float32)
        palm_lens = [float(f["palm_len"]) for f in frames]
    else:
        angles_list, normals_list, palm_lens = [], [], []
        for f in frames:
            lm_px = np.array(f["landmarks_px"], dtype=np.float32)
            angles_list.append(compute_joint_flexion_angles(lm_px))
            # Nhan tay thuan THEO TUNG FRAME, dung nhu production lam (khong dung nhan
            # chiem da so o day) — nhan chiem da so chi dung cho buoc chuan hoa chirality.
            normals_list.append(compute_palm_normal_vector(lm_px, handedness=f.get("handedness", "Right")))
            palm_lens.append(recompute_palm_len(lm_px))
        angles_mat = np.array(angles_list, dtype=np.float32)
        normals_mat = np.array(normals_list, dtype=np.float32)

    # --- Neo goc chuyen dong tai frame dau stroke, chia palm_len cua frame 0 ---
    x0, y0 = float(wrists[0, 0]), float(wrists[0, 1])
    ref_palm = max(palm_lens[0], 15.0)

    raw_xs = ((wrists[:, 0] - x0) / ref_palm).astype(np.float32)
    raw_ys = ((wrists[:, 1] - y0) / ref_palm).astype(np.float32)

    raw_dxs = np.zeros(n, dtype=np.float32)
    raw_dys = np.zeros(n, dtype=np.float32)
    raw_dxs[1:] = raw_xs[1:] - raw_xs[:-1]
    raw_dys[1:] = raw_ys[1:] - raw_ys[:-1]
    raw_dxs[0] = raw_dxs[1] if n > 1 else 0.0
    raw_dys[0] = raw_dys[1] if n > 1 else 0.0

    raw_dangles_mat = np.zeros_like(angles_mat)
    raw_dangles_mat[1:] = angles_mat[1:] - angles_mat[:-1]
    raw_dangles_mat[0] = raw_dangles_mat[1] if n > 1 else 0.0

    raw_17d = np.column_stack([
        raw_xs, raw_ys, raw_dxs, raw_dys, angles_mat, normals_mat, raw_dangles_mat
    ]).astype(np.float32)

    # --- SPEC 03: 7 cot moi. Landmark luon lay tu "landmarks_px" (z*frame_width, dung nhu lm_3d production) ---
    lm_seq = [np.array(f["landmarks_px"], dtype=np.float32) for f in frames]
    has_face_key = any("face_box" in f for f in frames)   # schema v1.0 khong co khoa nay
    face_seq = [f.get("face_box") for f in frames]
    raw_ext, body_ok = compute_extended_columns(palm_lens, wrists, lm_seq, face_seq)
    if not has_face_key:
        # Schema cu: khong co hop mat => khong tai tao duoc moc co the, danh dau de w_b bi vo hieu.
        body_ok = False
        print(f"  [Canh bao] Stroke '{raw.get('word_key')}' schema {raw.get('schema_version')} khong co face_box "
              f"-> body_anchor_valid=False (w_b se bi vo hieu cho mau nay).")
    elif not body_ok:
        print(f"  [Canh bao] Stroke '{raw.get('word_key')}': moi frame deu face_box=null -> body_anchor_valid=False.")
    raw_24d = append_extended_columns(raw_17d, raw_ext)

    shapes = [f.get("hand_shape", "Unknown") for f in frames]

    res_traj, res_shapes = resample_9d_gesture_sequence(raw_24d, shapes, k_target=K_RESAMPLE)
    if res_traj is None or res_shapes is None:
        return None

    res_traj, was_flipped = canonicalize_chirality(res_traj, handedness=dominant_handed)
    # So nhip lap tinh tren chuoi K=32 (cung cho voi match() va cac duong trich xuat khac).
    rep_count = compute_repetition_count(res_traj[:, 0], res_traj[:, 1])

    duration = float(raw.get("stroke_duration_sec") or 0.0)
    if duration <= 0.0:
        duration = float(frames[-1]["timestamp"] - frames[0]["timestamp"])

    session_id = raw.get("session_id")
    if session_id is None:
        session_id = "unknown"
        print(f"  [Canh bao] Stroke '{raw.get('word_key')}' schema {raw.get('schema_version')} khong co session_id -> dat 'unknown'.")

    return {
        # Ban FULL-PRECISION (khong lam tron) chi dung de do sai so round-trip; bi xoa
        # truoc khi ghi file. Nho no ta tach duoc sai so TINH TOAN that ra khoi sai so
        # LUONG TU HOA do lam tron 4 chu so thap phan luc ghi JSON (1 LSB = 1e-4).
        "_traj_full": res_traj,
        "traj_features": [[round(float(v), 4) for v in row] for row in res_traj],
        "hand_shapes": res_shapes,
        "duration_sec": round(duration, 3),
        "handedness": dominant_handed,
        "chirality_canonical": True,
        "chirality_flipped_on_save": bool(was_flipped),
        "session_id": session_id,
        "rep_count": int(rep_count),
        "body_anchor_valid": bool(body_ok),
        "rebuilt_from_raw": True,
        "raw_schema_version": raw.get("schema_version"),
        "template_sample_index_original": raw.get("template_sample_index"),
    }


# =============================================================================
# GHI FILE & BẢO VỆ DỮ LIỆU
# =============================================================================

def backup_existing(path: str) -> Optional[str]:
    """Sao luu file se bi ghi de, theo dung quy uoc ten cua project."""
    if not os.path.exists(path):
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(os.path.dirname(path) or ".", f"{stem}.backup_before_rebuild_{stamp}.json")
    shutil.copy2(path, bak)
    print(f"[Backup] Da sao luu file cu -> {bak}")
    return bak


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"[!] Khong tim thay thu muc kho tho '{args.raw_dir}'.")
        print("    Kho nay duoc collect_dynamic_templates.py tu dong sinh ra khi thu mau moi.")
        sys.exit(1)

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "*", "*.json")))
    if not raw_files:
        print(f"[!] Thu muc '{args.raw_dir}' khong co file stroke tho nao.")
        sys.exit(1)

    print("=" * 90)
    print("  TAI TAO TEMPLATE 24D TU KHO STROKE THO")
    print("=" * 90)
    print(f"  Thu muc tho        : {args.raw_dir}  ({len(raw_files)} stroke)")
    print(f"  Nguon goc/phap tuyen: {'DA LUU trong file tho' if args.use_stored_features else 'TINH LAI tu 21 moc ban tay'}")
    print(f"  Cong thoi luong    : {'TAT' if args.no_duration_gate else f'[{STROKE_DURATION_MIN_SEC}, {STROKE_DURATION_MAX_SEC}] s'}")

    vocab: Dict[str, Dict[str, Any]] = {}
    n_ok = n_skip_dur = n_skip_err = 0

    for rf in raw_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"  [!] Bo qua {rf}: {e}")
            n_skip_err += 1
            continue

        word_key = raw.get("word_key")
        if not word_key or word_key == "__NEG__":
            continue

        dur = float(raw.get("stroke_duration_sec") or 0.0)
        if (not args.no_duration_gate) and dur > 0 and not is_stroke_duration_valid(dur):
            print(f"  [Gate] Loai {os.path.basename(rf)} ({word_key}): thoi luong {dur:.2f}s ngoai cong.")
            n_skip_dur += 1
            continue

        sample = rebuild_one_stroke(raw, use_stored_features=args.use_stored_features)
        if sample is None:
            n_skip_err += 1
            continue

        sample["source_raw_file"] = os.path.relpath(rf).replace("\\", "/")
        entry = vocab.setdefault(word_key, {
            "display_name": raw.get("display_name", word_key),
            "threshold": 1.25,
            "samples": []
        })
        entry["samples"].append(sample)
        n_ok += 1

    for w in vocab.values():
        w["num_samples"] = len(w["samples"])

    print(f"\n  Tai tao thanh cong : {n_ok} mau / {len(vocab)} tu vung")
    print(f"  Bi loai vi thoi luong: {n_skip_dur} | Loi / khong du frame: {n_skip_err}")

    # --- Doi chieu round-trip (neu duoc yeu cau) ---
    if args.verify_against:
        verify_round_trip(vocab, args.verify_against, args.tolerance)

    if not args.out:
        print("\n  (Khong co --out nen khong ghi file nao.)")
        return

    if os.path.exists(args.out) and not args.force:
        print(f"\n[!] TU CHOI GHI DE: '{args.out}' da ton tai.")
        print("    Them --force neu that su muon ghi de (file cu se duoc backup tu dong truoc).")
        sys.exit(2)

    backup_existing(args.out)

    payload = {
        "schema_version": "1.0",
        "metadata": {
            "k_resample_frames": K_RESAMPLE,
            "feature_dim": 24,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Templates 24D tai tao tu kho stroke tho (rebuild_templates_from_raw.py)",
            "rebuilt_from_raw_dir": args.raw_dir,
            "features_source": "stored" if args.use_stored_features else "recomputed_from_landmarks",
            "chirality_canonical": True,
        },
        "vocabulary": vocab
    }

    # Xoa khoa noi bo _traj_full (chi phuc vu do sai so) truoc khi ghi ra dia.
    for w in vocab.values():
        for smp in w["samples"]:
            smp.pop("_traj_full", None)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Da ghi {n_ok} mau vao {args.out}")


def verify_round_trip(vocab: Dict[str, Any], ref_path: str, tol: float) -> None:
    """
    Doi chieu vector tai tao voi vector ma bo thu mau DA GHI TRUC TIEP, ghep cap qua
    truong 'template_sample_index_original'. In sai so tuyet doi lon nhat.
    """
    print("\n" + "-" * 90)
    print(f"  DOI CHIEU ROUND-TRIP voi: {ref_path}")
    print("-" * 90)
    if not os.path.exists(ref_path):
        print(f"  [!] Khong tim thay {ref_path}.")
        return

    with open(ref_path, "r", encoding="utf-8") as f:
        ref_vocab = json.load(f).get("vocabulary", {})

    n_pairs = 0
    worst = 0.0
    worst_where = ""
    per_word: List[Tuple[str, int, float, float]] = []

    for word_key, entry in vocab.items():
        ref_samples = ref_vocab.get(word_key, {}).get("samples", [])
        for s in entry["samples"]:
            idx = s.get("template_sample_index_original")
            if idx is None or not (0 <= int(idx) < len(ref_samples)):
                continue
            ref_s = ref_samples[int(idx)]
            a_r = np.array(s["traj_features"], dtype=np.float64)
            a_f = np.array(s.get("_traj_full", s["traj_features"]), dtype=np.float64)
            b = np.array(ref_s.get("traj_features_9d") or ref_s.get("traj_features"), dtype=np.float64)
            if a_r.shape[0] != b.shape[0]:
                print(f"  [!] {word_key}[{idx}]: lech kich thuoc {a_r.shape} vs {b.shape} -> bo qua")
                continue
            # SPEC 03: ban tai tao la 24D, ban tham chieu co the la 17D => chi doi chieu cac cot chung.
            n_common = min(a_r.shape[1], b.shape[1], a_f.shape[1])
            a_r, a_f, b = a_r[:, :n_common], a_f[:, :n_common], b[:, :n_common]
            # err_q : sai so giua HAI CHUOI DA GHI DIA (co ca luong tu hoa 4 chu so)
            # err   : sai so TINH TOAN that — so ban full-precision voi ban tham chieu da
            #         lam tron, nen theo ly thuyet <= 0.5 LSB = 5e-5 neu cong thuc trung khop.
            err_q = float(np.max(np.abs(a_r - b)))
            err = float(np.max(np.abs(a_f - b)))
            per_word.append((word_key, int(idx), err, err_q))
            n_pairs += 1
            if err > worst:
                worst = err
                worst_where = f"{word_key}[{idx}]"

    if n_pairs == 0:
        print("  [!] Khong ghep duoc cap nao (thieu 'template_sample_index_original' hoac file lech).")
        return

    for word_key, idx, err, err_q in per_word:
        flag = "OK " if err <= tol else "!! "
        print(f"   {flag} {word_key}[{idx}]: max|delta| = {err:.3e}  (sau lam tron 4 c.s.: {err_q:.3e})")

    mean_err = float(np.mean([e for _, _, e, _ in per_word]))
    worst_q = max(eq for _, _, _, eq in per_word)
    print(f"\n  So cap doi chieu            : {n_pairs}")
    print(f"  Sai so TINH TOAN lon nhat   : {worst:.3e}  (tai {worst_where})")
    print(f"  Sai so tinh toan trung binh : {mean_err:.3e}")
    print(f"  Sai so lon nhat SAU lam tron: {worst_q:.3e}   (1 LSB = 1e-4, luong tu hoa khi ghi JSON)")
    print(f"  Nguong cho phep             : {tol:.3e}  ->  {'DAT' if worst <= tol else 'KHONG DAT'}")


if __name__ == "__main__":
    main()
