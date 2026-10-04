# -*- coding: utf-8 -*-
"""
NHẬP ẢNH CHỮ CÁI CỘNG ĐỒNG -> landmark 3D (cùng định dạng với collect_letters_landmarks.py).

Cấu trúc thư mục đầu vào:   <gốc>/<chữ>/*.jpg|png   (tên thư mục chữ không phân biệt hoa/thường; thư mục
không phải một chữ cái a-z như 'del', 'nothing', 'space' bị bỏ qua).

Kết quả: letter_landmarks/EXT_<tên>/*.npz. Dữ liệu EXT_* CHỈ DÙNG ĐỂ TRAIN (train_letter_landmarks.py không
bao giờ đưa vào tập test, vì tay dựng thẳng/chính diện làm số đo đẹp giả).

Nhãn trái/phải: MediaPipe giả định ảnh selfie (đã lật gương). Ảnh người khác chụp không lật thì nhãn có thể đảo.
  --flip              lật gương ảnh trước khi dò (dùng khi bộ dữ liệu là selfie KHÔNG lật)
  --handed right      ép mọi ảnh là tay PHẢI (dùng khi bạn biết toàn bộ người ký dùng tay phải)
Mặc định lấy nhãn của MediaPipe và IN PHÂN BỐ Left/Right mỗi chữ để bạn kiểm tra: nếu một chữ lệch nặng
sang Left trong khi người ký dùng tay phải, hãy chạy lại với --flip hoặc --handed right.

Ví dụ:  py import_letter_images.py --root D:\\data\\asl_alphabet_train --name kaggle --max-per-class 400
"""
import argparse
import glob
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from hand_pose_features import extract_hand, view_angle_deg

HAND_MODEL = "hand_landmarker.task"
OUT_ROOT = "letter_landmarks"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CHUNK = 25          # số ảnh liên tiếp (theo tên file) gộp thành một "phiên" giả để chia train/test không rò rỉ
MIN_SCORE = 0.7


def letter_dirs(root: str) -> Dict[str, str]:
    out = {}
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if os.path.isdir(full) and len(d) == 1 and d.isalpha():
            out[d.lower()] = full
    return out


def detect_image(detector, mp, path: str, flip: bool) -> Optional[Dict[str, Any]]:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    if flip:
        bgr = cv2.flip(bgr, 1)
    h, w = bgr.shape[:2]
    if max(h, w) > 1024:
        s = 1024.0 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hand = extract_hand(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))), MIN_SCORE)
    if hand is None or hand["world"] is None:
        return None
    img = hand["image"]
    if not np.all((img[:, :2] > 0.0) & (img[:, :2] < 1.0)):
        return None
    return hand


def save_chunk(out_dir: str, name: str, letter: str, idx: int, hands: List[Dict[str, Any]], handed_override: Optional[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sid = f"EXT_{name}_{letter}_c{idx:04d}"
    handed = [handed_override or h["handed"] for h in hands]
    np.savez_compressed(
        os.path.join(out_dir, sid + ".npz"),
        world=np.stack([h["world"] for h in hands]).astype(np.float32),
        image=np.stack([h["image"] for h in hands]).astype(np.float32),
        handed=np.array(handed), score=np.array([h["score"] for h in hands], dtype=np.float32),
        view_angle=np.array([view_angle_deg(h["world"]) for h in hands], dtype=np.float32),
        t=np.zeros(len(hands), dtype=np.float64), letter=np.array(letter), person=np.array(f"EXT_{name}"),
        session_id=np.array(sid), view_tag=np.array("ext_image"), frame_size=np.array([0, 0]), mirrored=np.array(False),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="thư mục chứa các thư mục con a..z")
    ap.add_argument("--name", required=True, help="tên bộ dữ liệu (thành EXT_<tên>)")
    ap.add_argument("--max-per-class", type=int, default=400, help="số ảnh tối đa/chữ (lấy rải đều theo tên file)")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--handed", choices=["detector", "right"], default="detector")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import mediapipe as mp
    detector = mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1))

    dirs = letter_dirs(a.root)
    if not dirs:
        raise SystemExit(f"Không thấy thư mục chữ cái (a..z) trong {a.root}")
    out_dir = os.path.join(OUT_ROOT, f"EXT_{a.name}")
    override = "Right" if a.handed == "right" else None
    t0 = time.time()
    print(f"{'chữ':<4}{'ảnh':>6}{'nhận':>6}{'tỉ lệ':>8}   Left/Right   góc trung vị")
    for letter, d in dirs.items():
        files = sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(IMG_EXT))
        if len(files) > a.max_per_class:       # lấy rải đều để giữ đa dạng người ký/góc chụp
            files = [files[i] for i in np.linspace(0, len(files) - 1, a.max_per_class).astype(int)]
        hands: List[Dict[str, Any]] = []
        chunk_i = 0
        counter: Counter = Counter()
        angles: List[float] = []
        for k, f in enumerate(files):
            hd = detect_image(detector, mp, f, a.flip)
            if hd is not None:
                hands.append(hd)
                counter[override or hd["handed"]] += 1
                angles.append(view_angle_deg(hd["world"]))
            if (k + 1) % CHUNK == 0 and hands:
                save_chunk(out_dir, a.name, letter, chunk_i, hands, override)
                chunk_i, hands = chunk_i + 1, []
        if hands:
            save_chunk(out_dir, a.name, letter, chunk_i, hands, override)
        ok = sum(counter.values())
        print(f"{letter.upper():<4}{len(files):>6}{ok:>6}{(ok / max(len(files), 1)) * 100:>7.0f}%   "
              f"{counter.get('Left', 0)}/{counter.get('Right', 0)}   {np.median(angles) if angles else float('nan'):.0f}°")
    print(f"Xong trong {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
