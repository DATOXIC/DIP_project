# -*- coding: utf-8 -*-
"""
HUẤN LUYỆN + ĐO nhánh chữ cái dựa trên landmark 3D chuẩn hóa.

Dữ liệu: letter_landmarks/<người>/*.npz (do tools/collect_letters_landmarks.py tạo).
Chia train/test THEO PHIÊN (không theo khung — khung trong một phiên gần như trùng nhau nên chia theo khung
sẽ phồng độ chính xác). Có thể giữ nguyên cả một người làm tập test (--holdout-person) để đo khả năng
tổng quát hóa sang người lạ.

Báo cáo:
  - Độ chính xác tổng + từng chữ + ma trận nhầm lẫn rút gọn.
  - Độ chính xác THEO GÓC NGHIÊNG (0-20 / 20-40 / 40-60 / 60+) trên dữ liệu thật — đúng điều cần đo.
  - Phép thử căng THUẦN MÔ PHỎNG (simulate_view): tay bị xoay + nhiễu độ sâu. Đây là phép thử ước lượng,
    KHÔNG thay thế dữ liệu thu thật nhiều góc.

Ví dụ:
    py train_letter_landmarks.py
    py train_letter_landmarks.py --holdout-person Mai
    py train_letter_landmarks.py --no-aug          # đối chứng: không tăng cường nhiễu độ sâu
    py train_letter_landmarks.py --fit-all         # sau khi đo, train lại trên TOÀN BỘ dữ liệu và lưu
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from hand_pose_features import FEATURE_DIM, landmarks_to_features, simulate_view, view_angle_deg
from letter_landmark_classifier import DEFAULT_MODEL_PATH, LandmarkMLP

BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 91)]


def load_dataset(root: str, space: str = "world") -> Dict[str, Any]:
    """space="world": landmark thế giới (mét). space="image": landmark ảnh dạng pixel (x*W, y*H, z*W) — đúng
    không gian của pipeline động và của mọi bản ghi cũ (raw_strokes/raw_volunteers: landmarks_px)."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    world, handed, labels, sessions, persons, angles = [], [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=False)
        if "world" not in d.files:
            continue
        n = len(d["world"])
        if space == "image":
            fs = d["frame_size"] if "frame_size" in d.files else np.array([0, 0])
            w_px, h_px = (float(fs[0]), float(fs[1])) if float(fs[0]) > 0 else (640.0, 480.0)
            world.append(d["image"].astype(np.float32) * np.array([w_px, h_px, w_px], dtype=np.float32))
        else:
            world.append(d["world"])
        handed += [str(h) for h in d["handed"]]
        labels += [str(d["letter"]).lower()] * n
        sessions += [str(d["session_id"])] * n
        persons += [str(d["person"])] * n
        angles.append(d["view_angle"])
    if not world:
        raise SystemExit(f"Không có dữ liệu trong {root}/ — chạy tools/collect_letters_landmarks.py trước.")
    return {"world": np.concatenate(world).astype(np.float32), "handed": np.array(handed), "label": np.array(labels),
            "session": np.array(sessions), "person": np.array(persons), "angle": np.concatenate(angles).astype(np.float32)}


def is_ext(data: Dict[str, Any]) -> np.ndarray:
    """Dữ liệu cộng đồng nhập bằng tools/import_letter_images.py (person = EXT_*): CHỈ để train, không bao giờ vào test."""
    return np.array([p.startswith("EXT_") for p in data["person"]], dtype=bool)


def split_chunks(data: Dict[str, Any], test_frac: float, chunk: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chia theo CỤM khung liên tiếp, PHÂN TẦNG theo (chữ, mặt tay hướng camera) để train và test đều có đủ hai mặt
    tay. Mỗi cụm `chunk` khung (theo thứ tự thu) là một đơn vị; cứ ~1/test_frac cụm thì một cụm vào test.
    Lỏng hơn chia theo phiên (các cụm kề nhau cùng ánh sáng/vị trí ngồi) nên số đo hơi lạc quan — dùng khi
    chưa đủ số phiên; khi đã thu nhiều phiên hoặc có người khác, ưu tiên --split session / --holdout-person.
    """
    from hand_pose_features import palm_faces_camera
    n = len(data["label"])
    ext = is_ext(data)
    face = np.array([palm_faces_camera(w, h) for w, h in zip(data["world"], data["handed"])])
    test_mask = np.zeros(n, dtype=bool)
    period = max(2, int(round(1.0 / test_frac)))
    for c in np.unique(data["label"]):
        for f in (True, False):
            idx = np.where((data["label"] == c) & (face == f) & ~ext)[0]
            for k, start in enumerate(range(0, len(idx), chunk)):
                if k % period == period - 1:
                    test_mask[idx[start:start + chunk]] = True
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def split_indices(data: Dict[str, Any], holdout_person: Optional[str], test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data["label"])
    ext = is_ext(data)
    if holdout_person:
        if holdout_person.startswith("EXT_"):
            raise SystemExit("Không được giữ dữ liệu cộng đồng (EXT_*) làm tập test.")
        test = data["person"] == holdout_person
        if not test.any():
            raise SystemExit(f"Không có dữ liệu của '{holdout_person}'")
        return np.where(~test)[0], np.where(test)[0]
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(n, dtype=bool)
    for c in np.unique(data["label"]):
        idx = np.where((data["label"] == c) & ~ext)[0]
        sess = np.unique(data["session"][idx])
        if len(sess) == 0:
            print(f"  [!] Chữ '{c}' chỉ có dữ liệu cộng đồng -> không có test thật cho chữ này.")
        elif len(sess) >= 2:
            k = max(1, int(round(len(sess) * test_frac)))
            pick = rng.choice(sess, size=k, replace=False)
            test_mask[idx[np.isin(data["session"][idx], pick)]] = True
        else:
            print(f"  [!] Chữ '{c}' chỉ có {len(sess)} phiên -> giữ 25% khung CUỐI làm test (kém tin cậy; thu thêm phiên).")
            test_mask[idx[int(len(idx) * (1 - test_frac)):]] = True
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def cap_external(data: Dict[str, Any], idx: np.ndarray, y: np.ndarray, max_ext: int, seed: int) -> np.ndarray:
    """Giới hạn số khung cộng đồng/chữ trong tập train để không áp đảo dữ liệu thu thật nhiều góc."""
    ext = is_ext(data)
    rng = np.random.default_rng(seed + 11)
    keep = []
    for c in np.unique(y[idx]):
        m = idx[y[idx] == c]
        e, r = m[ext[m]], m[~ext[m]]
        if len(e) > max_ext:
            e = rng.choice(e, size=max_ext, replace=False)
        keep += [r, e]
    return np.sort(np.concatenate(keep))


def featurize(world: np.ndarray, handed: np.ndarray) -> np.ndarray:
    return np.stack([landmarks_to_features(w, h) for w, h in zip(world, handed)]).astype(np.float32)


def augmented_features(world: np.ndarray, handed: np.ndarray, copies: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(copies):
        out.append(np.stack([landmarks_to_features(simulate_view(w, rng), h) for w, h in zip(world, handed)]))
    return np.concatenate(out).astype(np.float32)


def fit(x: np.ndarray, y: np.ndarray, num_classes: int, epochs: int, seed: int) -> Tuple[LandmarkMLP, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y.astype(np.int64))
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weight = torch.from_numpy(counts.sum() / (num_classes * np.maximum(counts, 1)))
    model = LandmarkMLP(num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    n, bs = len(xt), 256
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
        sched.step()
    model.eval()
    return model, mean, std


def predict(model: LandmarkMLP, mean: np.ndarray, std: np.ndarray, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy((x - mean) / std)).argmax(1).numpy()


def report(title: str, y_true: np.ndarray, y_pred: np.ndarray, angles: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    print(f"\n== {title}: n={len(y_true)}  acc={acc * 100:.1f}%")
    per_class = {}
    for i, c in enumerate(classes):
        m = y_true == i
        if m.any():
            per_class[c] = float((y_pred[m] == i).mean())
    print("  từng chữ : " + "  ".join(f"{c.upper()}={a * 100:.0f}%" for c, a in per_class.items()))
    per_bucket = {}
    for lo, hi in BUCKETS:
        m = (angles >= lo) & (angles < hi)
        if m.sum() >= 5:
            per_bucket[f"{lo}-{min(hi, 90)}"] = (float((y_true[m] == y_pred[m]).mean()), int(m.sum()))
    print("  theo góc : " + "  ".join(f"{k}°={a * 100:.0f}% (n={n})" for k, (a, n) in per_bucket.items()))
    errs: Dict[str, int] = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            key = f"{classes[t].upper()}->{classes[p].upper()}"
            errs[key] = errs.get(key, 0) + 1
    top = sorted(errs.items(), key=lambda kv: -kv[1])[:8]
    if top:
        print("  nhầm nhiều nhất: " + ", ".join(f"{k}:{v}" for k, v in top))
    return {"acc": acc, "n": int(len(y_true)), "per_class": per_class,
            "per_bucket": {k: {"acc": a, "n": n} for k, (a, n) in per_bucket.items()}, "top_errors": dict(top)}


def run(root: str, out: str, holdout_person: Optional[str], epochs: int, copies: int, no_aug: bool,
        fit_all: bool, seed: int = 0, test_frac: float = 0.25, results_path: Optional[str] = None,
        max_ext: int = 600, split: str = "session", space: str = "world") -> Dict[str, Any]:
    data = load_dataset(root, space)
    classes = sorted(set(data["label"].tolist()))
    cid = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cid[c] for c in data["label"]])
    print(f"Dữ liệu: {len(y_all)} khung, {len(classes)} chữ {classes}, {len(set(data['session']))} phiên, "
          f"{len(set(data['person']))} người")
    if split == "chunks" and not holdout_person:
        tr, te = split_chunks(data, test_frac)
    else:
        tr, te = split_indices(data, holdout_person, test_frac, seed)
    ext = is_ext(data)
    print(f"Dữ liệu cộng đồng (EXT_*): {int(ext.sum())} khung — chỉ dùng để train, tối đa {max_ext} khung/chữ")
    tr = cap_external(data, tr, y_all, max_ext, seed)
    print(f"Train {len(tr)} khung / Test {len(te)} khung" + (f" (giữ nguyên người '{holdout_person}')" if holdout_person else " (tách theo phiên)"))

    def build_train(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = featurize(data["world"][idx], data["handed"][idx])
        y = y_all[idx]
        if not no_aug and copies > 0:
            xa = augmented_features(data["world"][idx], data["handed"][idx], copies, seed + 1)
            x, y = np.concatenate([x, xa]), np.concatenate([y, np.tile(y, copies)])
        return x, y

    results: Dict[str, Any] = {"classes": classes, "no_aug": no_aug, "holdout_person": holdout_person}
    if len(te):
        x_tr, y_tr = build_train(tr)
        model, mean, std = fit(x_tr, y_tr, len(classes), epochs, seed)
        x_te = featurize(data["world"][te], data["handed"][te])
        results["test_real"] = report("TEST THẬT (landmark thu từ webcam)", y_all[te], predict(model, mean, std, x_te), data["angle"][te], classes)
        rng = np.random.default_rng(seed + 7)
        sim = [simulate_view(w, rng) for w in data["world"][te]]
        x_sim = np.stack([landmarks_to_features(s, h) for s, h in zip(sim, data["handed"][te])])
        a_sim = np.array([view_angle_deg(s) for s in sim])
        results["test_synthetic_stress"] = report("TEST CĂNG MÔ PHỎNG (xoay + nhiễu độ sâu; chỉ là ước lượng)",
                                                  y_all[te], predict(model, mean, std, x_sim), a_sim, classes)
        final = (model, mean, std)
    else:
        final = None

    if fit_all or final is None:
        x_all, y_fit = build_train(cap_external(data, np.arange(len(y_all)), y_all, max_ext, seed))
        final = fit(x_all, y_fit, len(classes), epochs, seed)
        print("\nĐã train lại trên TOÀN BỘ dữ liệu cho model lưu.")
    model, mean, std = final
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "classes": classes, "feature_dim": FEATURE_DIM, "space": space,
                "trained_on": "all" if (fit_all or not len(te)) else "train_split"}, out)
    print(f"Đã lưu model: {out}")
    if results_path:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="letter_landmarks")
    ap.add_argument("--out", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--holdout-person", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--aug-copies", type=int, default=6, help="số bản sao tăng cường (xoay + nhiễu độ sâu) mỗi mẫu")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--fit-all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ext", type=int, default=600, help="số khung cộng đồng tối đa/chữ khi train")
    ap.add_argument("--split", choices=["session", "chunks"], default="session",
                    help="session: giữ nguyên cả phiên làm test (nghiêm ngặt); chunks: theo cụm khung, phân tầng theo mặt tay")
    ap.add_argument("--space", choices=["world", "image"], default="world",
                    help="world: landmark thế giới; image: landmark pixel (dùng cho pipeline động và bản ghi cũ)")
    ap.add_argument("--results", default="results/letter_landmark_eval.json")
    a = ap.parse_args()
    run(a.data, a.out, a.holdout_person, a.epochs, a.aug_copies, a.no_aug, a.fit_all, a.seed, results_path=a.results, max_ext=a.max_ext, split=a.split, space=a.space)


if __name__ == "__main__":
    main()
