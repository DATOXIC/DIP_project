"""
Tang cuong du lieu tu stroke tho (SPEC 06A, che do A: chi de DO, khong dua vao template runtime).

Y tuong:
  - Nguon: cac mau da co trong bank (webcam_templates.json). Moi mau bank duoc doi chieu voi stroke tho
    trong raw_strokes/ bang cach dung lai vector 24D tu stroke tho (rebuild_one_stroke) va so voi vector
    da luu trong bank. Chi cac mau bank TIM DUOC stroke tho tuong ung moi duoc sinh bien the
    (stroke tho cua cac mau da bi huy 'r'/'x' khi thu KHONG bao gio duoc dung).
  - Bien the duoc tao o muc stroke THO (toa do landmark + co tay), roi tinh lai TOAN BO dac trung 24D
    bang dung ham cua duong tai tao (rebuild_one_stroke): goc gap, phap tuyen, palm_len, do xoe ngon,
    moc co the, resample K=32, chuan hoa chirality, dem nhip lap.
  - Ghi ra file MOI (mac dinh webcam_templates_augmented.json). KHONG ghi de webcam_templates.json.
  - Moi mau co: sample_id, is_original, augment_of, augment_kind; bien the THUA HUONG session_id cua stroke goc
    (de scoreboard.py tach LOSO theo phien va loai moi bien the cung augment_of voi query).

CAM: lat guong ngang (dao chirality), xoay > 15 do, keo thoi gian ra ngoai cong thoi luong,
     bien the bi loai boi is_movement_epenthesis, augment mau am.

Cach dung:
    py augment_templates.py --seed 20260919 --strength 1.0
    py augment_templates.py --variants 0 --out results/bank_ids_only.json   # chi gan sample_id (nen de so sanh ghep cap)
"""

import argparse
import copy
import glob
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backup_utils import backup_path_for
from dynamic_dtw_engine import GestureFrame, is_movement_epenthesis, is_stroke_duration_valid
from rebuild_templates_from_raw import rebuild_one_stroke, recompute_palm_len

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_SEED = 20260919

# Khoang bien doi (nhan voi --strength).
TIME_RANGE = 0.20          # co gian thoi gian +-20%
SCALE_RANGE = 0.15         # co gian ti le +-15%
ROT_DEG = 10.0             # xoay +-10 do (tran cung 15 do)
MAX_ROT_DEG = 15.0
OFFSET_FRAC = 0.10         # lech diem khoi dau +-10% palm_len
DROP_LO, DROP_HI = 0.05, 0.10   # roi 5-10% frame
NOISE_RAD = 0.02           # sigma nhieu goc (rad)
NOISE_BONE_FRAC = 0.4      # do dai xuong ngon ~0.4 palm_len: sigma_px = NOISE_RAD * NOISE_BONE_FRAC * palm_len

KINDS = ["time", "scale", "rotate", "offset", "dropout", "noise", "combo"]
COMBO_POOL = ["time", "scale", "rotate", "offset", "noise"]


# =============================================================================
# 1. DONG GOI / GIAI NEN CAC FRAME THO
# =============================================================================

def pack_frames(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Doi danh sach frame (dict) thanh cac mang numpy de bien doi."""
    fr = raw["frames"]
    return {
        "t": np.array([float(f["timestamp"]) for f in fr], dtype=np.float64),
        "wrist": np.array([f["wrist_smoothed"] for f in fr], dtype=np.float64),          # (n, 2)
        "lm": np.array([f["landmarks_px"] for f in fr], dtype=np.float64),               # (n, 21, 3)
        "face": [f.get("face_box") for f in fr],
        "has_face_key": any("face_box" in f for f in fr),
        "shape": [f.get("hand_shape", "Unknown") for f in fr],
        "hand": [f.get("handedness", "Right") for f in fr],
        "time_factor": 1.0,
    }


def unpack_frames(raw: Dict[str, Any], P: Dict[str, Any]) -> Dict[str, Any]:
    """Tao stroke tho moi tu ban goc + mang da bien doi. Chi cac khoa ma rebuild_one_stroke can duoc ghi."""
    out = copy.deepcopy(raw)
    n = len(P["t"])
    frames = []
    for i in range(n):
        f = {
            "timestamp": float(P["t"][i]),
            "wrist_smoothed": [float(P["wrist"][i, 0]), float(P["wrist"][i, 1])],
            "landmarks_px": P["lm"][i].tolist(),
            "hand_shape": P["shape"][i],
            "handedness": P["hand"][i],
        }
        if P["has_face_key"]:
            f["face_box"] = P["face"][i]
        frames.append(f)
    out["frames"] = frames
    out["num_frames"] = n
    base_dur = float(raw.get("stroke_duration_sec") or (raw["frames"][-1]["timestamp"] - raw["frames"][0]["timestamp"]))
    out["stroke_duration_sec"] = base_dur * P["time_factor"]
    return out


def _lerp_rows(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    i0 = np.clip(np.floor(u).astype(int), 0, n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (u - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    return a[i0] * (1.0 - w) + a[i1] * w


def _nearest(lst: List[Any], u: np.ndarray) -> List[Any]:
    idx = np.clip(np.rint(u).astype(int), 0, len(lst) - 1)
    return [lst[i] for i in idx]


# =============================================================================
# 2. CAC PHEP BIEN DOI (moi phep: (P, rng, strength, info) -> sua P tai cho)
# =============================================================================

def _palm0(P: Dict[str, Any]) -> float:
    return max(recompute_palm_len(P["lm"][0].astype(np.float32)), 15.0)


def t_time(P, rng, s, info):
    f = 1.0 + s * rng.uniform(-TIME_RANGE, TIME_RANGE)
    n = len(P["t"])
    n2 = max(8, int(round(n * f)))
    u = np.linspace(0.0, n - 1.0, n2)
    t0, dur = P["t"][0], P["t"][-1] - P["t"][0]
    P["wrist"] = _lerp_rows(P["wrist"], u)
    P["lm"] = _lerp_rows(P["lm"], u)
    P["face"] = _nearest(P["face"], u)
    P["shape"] = _nearest(P["shape"], u)
    P["hand"] = _nearest(P["hand"], u)
    P["t"] = t0 + np.linspace(0.0, dur * f, n2)
    P["time_factor"] *= f
    info["time_factor"] = round(float(f), 4)


def t_scale(P, rng, s, info):
    k = 1.0 + s * rng.uniform(-SCALE_RANGE, SCALE_RANGE)
    p0 = P["wrist"][0].copy()
    P["wrist"] = p0 + k * (P["wrist"] - p0)
    P["lm"][:, :, :2] = p0 + k * (P["lm"][:, :, :2] - p0)
    P["lm"][:, :, 2] *= k
    info["scale"] = round(float(k), 4)


def t_rotate(P, rng, s, info):
    deg = float(np.clip(s * rng.uniform(-ROT_DEG, ROT_DEG), -MAX_ROT_DEG, MAX_ROT_DEG))
    th = math.radians(deg)
    c, sn = math.cos(th), math.sin(th)
    R = np.array([[c, -sn], [sn, c]])
    p0 = P["wrist"][0].copy()
    P["wrist"] = p0 + (P["wrist"] - p0) @ R.T
    P["lm"][:, :, :2] = p0 + (P["lm"][:, :, :2] - p0) @ R.T
    info["rot_deg"] = round(deg, 3)


def t_offset(P, rng, s, info):
    d = s * rng.uniform(-OFFSET_FRAC, OFFSET_FRAC, size=2) * _palm0(P)
    P["wrist"] = P["wrist"] + d
    P["lm"][:, :, :2] = P["lm"][:, :, :2] + d
    info["offset_px"] = [round(float(d[0]), 2), round(float(d[1]), 2)]


def t_dropout(P, rng, s, info):
    n = len(P["t"])
    m = int(round(n * rng.uniform(DROP_LO, DROP_HI) * s))
    m = min(m, max(0, n - 2))
    if m <= 0:
        info["dropped"] = 0
        return
    drop = np.sort(rng.choice(np.arange(1, n - 1), size=m, replace=False))
    kept = np.setdiff1d(np.arange(n), drop)
    grid = np.arange(n)

    def fill(a):
        flat = a.reshape(n, -1)
        out = np.empty_like(flat)
        for c in range(flat.shape[1]):
            out[:, c] = np.interp(grid, kept, flat[kept, c])
        return out.reshape(a.shape)

    P["wrist"] = fill(P["wrist"])
    P["lm"] = fill(P["lm"])
    prev = {j: int(kept[np.searchsorted(kept, j) - 1]) for j in drop}
    for j in drop:
        P["shape"][j] = P["shape"][prev[j]]
    info["dropped"] = int(m)


def t_noise(P, rng, s, info):
    sigma = NOISE_RAD * NOISE_BONE_FRAC * s * _palm0(P)
    P["lm"] = P["lm"] + rng.normal(0.0, sigma, size=P["lm"].shape) if sigma > 0 else P["lm"]
    info["noise_sigma_px"] = round(float(sigma), 3)


TRANSFORMS = {
    "time": t_time, "scale": t_scale, "rotate": t_rotate,
    "offset": t_offset, "dropout": t_dropout, "noise": t_noise,
}


def apply_kind(P, kind: str, rng, strength: float) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if kind == "combo":
        picks = rng.choice(COMBO_POOL, size=int(rng.integers(2, 4)), replace=False)
        info["combo"] = [str(p) for p in picks]
        for p in picks:
            TRANSFORMS[str(p)](P, rng, strength * 0.5, info)
    else:
        TRANSFORMS[kind](P, rng, strength, info)
    return info


# =============================================================================
# 3. CONG KIEM TRA (bien the phai la thu pipeline live TAO DUOC)
# =============================================================================

def passes_gates(raw_aug: Dict[str, Any]) -> Tuple[bool, str]:
    dur = float(raw_aug["stroke_duration_sec"])
    if not is_stroke_duration_valid(dur):
        return False, "duration"   # ngoai [STROKE_DURATION_MIN_SEC, STROKE_DURATION_MAX_SEC]
    gf = []
    for f in raw_aug["frames"]:
        lm = np.array(f["landmarks_px"], dtype=np.float32)
        gf.append(GestureFrame(x=f["wrist_smoothed"][0], y=f["wrist_smoothed"][1],
                               palm_len=max(recompute_palm_len(lm), 15.0),
                               hand_shape=f.get("hand_shape", "Unknown"), timestamp=f["timestamp"]))
    if is_movement_epenthesis(gf):
        return False, "epenthesis"
    return True, ""


# =============================================================================
# 4. DOI CHIEU MAU BANK <-> STROKE THO
# =============================================================================

def load_raw_strokes(raw_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*", "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue
        wk = raw.get("word_key")
        if not wk or wk == "__NEG__" or not raw.get("frames"):
            continue
        out.append((path, raw))
    return out


def match_bank_to_raw(
    bank_vocab: Dict[str, Any],
    raws: List[Tuple[str, Dict[str, Any]]],
    tol: float = 5e-3,
) -> Dict[Tuple[str, int], Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Voi moi mau bank tim stroke tho co vector tai tao khop (sai so lon nhat < tol), cung tu + cung session_id.
    Tra ve {(word, idx_bank): (duong_dan_raw, raw, sample_da_tai_tao)}. Moi stroke tho chi khop MOT mau bank.
    """
    cand: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any], np.ndarray, Dict[str, Any]]]] = {}
    for path, raw in raws:
        sample = rebuild_one_stroke(raw)
        if sample is None:
            continue
        cand.setdefault((raw["word_key"], str(raw.get("session_id"))), []).append((path, raw, sample["_traj_full"], sample))

    result = {}
    used = set()
    for word, wd in bank_vocab.items():
        for i, s in enumerate(wd.get("samples", [])):
            key = (word, str(s.get("session_id")))
            q = np.array(s["traj_features"], dtype=np.float32)
            best = None
            for path, raw, traj, sample in cand.get(key, []):
                if path in used or traj.shape != q.shape:
                    continue
                d = float(np.max(np.abs(traj - q)))
                if best is None or d < best[0]:
                    best = (d, path, raw, sample)
            if best is not None and best[0] < tol:
                used.add(best[1])
                result[(word, i)] = (best[1], best[2], best[3])
    return result


# =============================================================================
# 5. CHINH
# =============================================================================

PROTECTED_OUT = {"webcam_templates.json", "dynamic_templates.json", "asl_citizen_templates.json",
                 "asl_citizen_templates_24d.json", "negative_strokes.json"}

SAMPLE_KEYS = ("traj_features", "hand_shapes", "duration_sec", "handedness", "chirality_canonical",
               "session_id", "rep_count", "body_anchor_valid")


def build_augmented_bank(
    bank: Dict[str, Any],
    raws: List[Tuple[str, Dict[str, Any]]],
    seed: int = DEFAULT_SEED,
    strength: float = 1.0,
    n_variants: int = 7,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    vocab = bank["vocabulary"]
    matches = match_bank_to_raw(vocab, raws)
    kinds = KINDS[:max(0, min(n_variants, len(KINDS)))]

    out_vocab: Dict[str, Any] = {}
    stats: Dict[str, Any] = {
        "seed": seed, "strength": strength, "n_variants_per_stroke": len(kinds),
        "originals": 0, "originals_with_raw": 0, "generated": 0, "rejected": {}, "rep_changed": 0,
        "per_word": {}, "per_kind": {k: 0 for k in kinds},
    }
    stroke_no = 0
    for word, wd in vocab.items():
        wcopy = {k: v for k, v in wd.items() if k != "samples"}
        samples_out: List[Dict[str, Any]] = []
        variants_out: List[Dict[str, Any]] = []
        n_gen = 0
        for i, s in enumerate(wd.get("samples", [])):
            sid = f"{word}__{s.get('session_id')}__b{i}"
            orig = copy.deepcopy(s)
            orig["sample_id"] = sid
            orig["is_original"] = True
            samples_out.append(orig)
            stats["originals"] += 1
            m = matches.get((word, i))
            if m is None:
                continue
            stats["originals_with_raw"] += 1
            path, raw, rebuilt = m
            for k_no, kind in enumerate(kinds):
                rng = np.random.default_rng([seed, stroke_no, k_no])
                P = pack_frames(raw)
                info = apply_kind(P, kind, rng, strength)
                raw_aug = unpack_frames(raw, P)
                ok, why = passes_gates(raw_aug)
                if not ok:
                    stats["rejected"][why] = stats["rejected"].get(why, 0) + 1
                    continue
                rebuilt_aug = rebuild_one_stroke(raw_aug)
                if rebuilt_aug is None:
                    stats["rejected"]["rebuild_none"] = stats["rejected"].get("rebuild_none", 0) + 1
                    continue
                v = {k: rebuilt_aug[k] for k in SAMPLE_KEYS if k in rebuilt_aug}
                v["session_id"] = s.get("session_id")          # thua huong session cua stroke goc
                v["sample_id"] = f"{sid}__aug_{kind}"
                v["is_original"] = False
                v["augment_of"] = sid
                v["augment_kind"] = kind
                v["augment_params"] = info
                if int(v["rep_count"]) != int(rebuilt["rep_count"]):
                    stats["rep_changed"] += 1
                variants_out.append(v)
                stats["generated"] += 1
                stats["per_kind"][kind] += 1
                n_gen += 1
            stroke_no += 1
        wcopy["samples"] = samples_out + variants_out
        out_vocab[word] = wcopy
        stats["per_word"][word] = {"originals": len(samples_out), "variants": n_gen}

    out = {k: v for k, v in bank.items() if k != "vocabulary"}
    out["augmentation"] = {
        "spec": "SPEC_06A_che_do_A", "seed": seed, "strength": strength, "kinds": kinds,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Chi de DO (scoreboard). Khong dung lam template runtime.",
    }
    out["vocabulary"] = out_vocab
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Tang cuong du lieu tu stroke tho (SPEC 06A, che do A).")
    p.add_argument("--bank", default="webcam_templates.json")
    p.add_argument("--raw-dir", default="raw_strokes")
    p.add_argument("--out", default="webcam_templates_augmented.json")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--strength", type=float, default=1.0, help="Nhan voi moi khoang bien doi (thu 0.5 va 1.0).")
    p.add_argument("--variants", type=int, default=7, help="So bien the moi stroke (0 = chi gan sample_id cho ban goc).")
    args = p.parse_args()

    if os.path.basename(args.out) in PROTECTED_OUT:
        print(f"[!] TU CHOI: '{args.out}' la file du lieu that. Chon ten khac.")
        sys.exit(2)

    with open(args.bank, "r", encoding="utf-8") as f:
        bank = json.load(f)
    raws = load_raw_strokes(args.raw_dir)
    print(f"[Augment] bank='{args.bank}' | raw='{args.raw_dir}' ({len(raws)} stroke) | seed={args.seed} | strength={args.strength}")

    out, st = build_augmented_bank(bank, raws, seed=args.seed, strength=args.strength, n_variants=args.variants)

    print(f"  Mau goc trong bank        : {st['originals']}")
    print(f"  ... tim duoc stroke tho   : {st['originals_with_raw']} (chi cac mau nay duoc sinh bien the)")
    print(f"  Bien the da sinh          : {st['generated']}  theo loai: {st['per_kind']}")
    print(f"  Bien the bi loai (cong)   : {st['rejected'] or 'khong'}")
    if st["generated"]:
        print(f"  rep_count doi so voi goc  : {st['rep_changed']}/{st['generated']} ({100.0 * st['rep_changed'] / st['generated']:.1f}%)")
    for w, d in st["per_word"].items():
        print(f"    {w:<10} goc {d['originals']:3d} | bien the {d['variants']:4d}")

    if os.path.exists(args.out):
        bak = backup_path_for(args.out, "backup_before_augment")
        os.replace(args.out, bak)
        print(f"[Backup] file cu -> {bak}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Augment] Da ghi: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
