# -*- coding: utf-8 -*-
"""
THU DỮ LIỆU CHỮ CÁI ĐA GÓC NHÌN (landmark 3D) — MỖI LẦN BẤM SPACE = LƯU ĐÚNG MỘT KHUNG.

Tập công khai thường chụp tay dựng thẳng, hướng thẳng camera nên không đại diện cho thực tế. Công cụ này
lưu LANDMARK (không lưu ảnh: nhẹ, riêng tư, tính lại đặc trưng được bất cứ lúc nào) và theo dõi độ phủ theo
GÓC NGHIÊNG (góc giữa pháp tuyến lòng bàn tay và trục camera) để bạn thu cho đủ các góc khó.

Cách dùng:   py collect_letters_landmarks.py --person DAT
  [a-z 0-9]  chọn chữ cái / dáng số (vd. 5 = bàn tay xòe, 1 = trỏ một ngón, dùng cho BAD/BYE/YOU/ME)                   [ENTER]  bắt đầu PHIÊN MỚI (đổi ánh sáng / vị trí ngồi / nghỉ giải lao)
  [SPACE]     CHỤP: lưu đúng khung hiện tại (chỉ khi tay được nhận rõ; nếu không máy báo lý do và KHÔNG lưu)
  [BACKSPACE] xóa khung vừa chụp               [ESC]    thoát
Quy trình cho mỗi chữ: đặt tay vào một hướng (chính diện, nghiêng, ngửa, cúi, xoay ra xa, gần/xa) -> SPACE ->
đổi hướng khác -> SPACE ... cho đến khi 4 thanh góc nghiêng đều xanh. Chụp sai thì BACKSPACE.
Phủ CẢ HAI mặt: LÒNG bàn tay hướng camera và MU bàn tay hướng camera (vd. THANK-YOU: chữ B nhưng lòng tay hướng
vào người). Hai hàng thanh phủ ở đáy màn hình theo dõi từng mặt riêng.
Train tách theo PHIÊN nên nhớ bấm ENTER khi đổi điều kiện (đổi chữ tự mở phiên mới).
Ảnh lật gương trước khi dò (như selfie) để nhãn Left/Right của MediaPipe đúng với tay thật.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from hand_pose_features import extract_hand, palm_faces_camera, view_angle_deg

HAND_MODEL = "hand_landmarker.task"
OUT_ROOT = "letter_landmarks"
VIEW_TAGS = ["chinh_dien", "nghieng_trai", "nghieng_phai", "ngua_len", "cui_xuong",
             "xoay_ra_xa", "gan_camera", "xa_camera", "tu_do"]
BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 91)]
TARGET_PER_BUCKET = 15
MIN_SCORE = 0.75
MAX_STEP = 0.12          # bước nhảy cổ tay tối đa giữa 2 khung (đơn vị: palm) — loại khung nhòe do đang di chuyển nhanh
SAVE_PERIOD = 0.08       # giây giữa 2 khung trong một lượt
SHOT_MIN = 1             # số khung tối thiểu để lưu (mỗi lần chụp = 1 khung)


def bucket_of(angle: float) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= angle < hi:
            return i
    return len(BUCKETS) - 1


def count_existing(person_dir: str):
    """{chữ: [[số khung mỗi nhóm góc] cho LÒNG bàn tay hướng camera, [..] cho MU bàn tay hướng camera]}."""
    counts = {}
    if not os.path.isdir(person_dir):
        return counts
    for fn in os.listdir(person_dir):
        if not fn.endswith(".npz"):
            continue
        try:
            d = np.load(os.path.join(person_dir, fn), allow_pickle=False)
            c = counts.setdefault(str(d["letter"]), [[0] * len(BUCKETS), [0] * len(BUCKETS)])
            for w, hd, ang in zip(d["world"], d["handed"], d["view_angle"]):
                c[0 if palm_faces_camera(w, str(hd)) else 1][bucket_of(float(ang))] += 1
        except Exception:
            continue
    return counts


class Session:
    def __init__(self, person: str, letter: str, tag: str, size, session_id: str):
        self.person, self.letter, self.tag, self.size = person, letter, tag, size
        self.session_id = session_id
        self.file_id = f"{session_id}_{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
        self.world, self.image, self.handed, self.score, self.angle, self.t = [], [], [], [], [], []

    def add(self, hand, angle):
        self.world.append(hand["world"]); self.image.append(hand["image"])
        self.handed.append(hand["handed"]); self.score.append(hand["score"])
        self.angle.append(angle); self.t.append(time.time())

    def save(self, person_dir: str):
        if len(self.world) < SHOT_MIN:
            return None
        os.makedirs(person_dir, exist_ok=True)
        path = os.path.join(person_dir, self.file_id + ".npz")
        np.savez_compressed(
            path, world=np.stack(self.world).astype(np.float32), image=np.stack(self.image).astype(np.float32),
            handed=np.array(self.handed), score=np.array(self.score, dtype=np.float32),
            view_angle=np.array(self.angle, dtype=np.float32), t=np.array(self.t, dtype=np.float64),
            letter=np.array(self.letter), person=np.array(self.person), session_id=np.array(self.session_id),
            view_tag=np.array(self.tag), frame_size=np.array(self.size), mirrored=np.array(True),
        )
        return path


def draw_coverage(frame, counts, letter):
    c = counts.get(letter, [[0] * len(BUCKETS), [0] * len(BUCKETS)])
    names = ["LONG tay->cam", "MU tay->cam"]
    x0, w = 150, 100
    for r in range(2):
        y0 = frame.shape[0] - 70 + r * 26
        cv2.putText(frame, names[r], (8, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        for i, ((lo, hi), n) in enumerate(zip(BUCKETS, c[r])):
            frac = min(1.0, n / TARGET_PER_BUCKET)
            x = x0 + i * (w + 8)
            cv2.rectangle(frame, (x, y0), (x + w, y0 + 16), (60, 60, 60), -1)
            cv2.rectangle(frame, (x, y0), (x + int(w * frac), y0 + 16), (0, 200, 0) if frac >= 1.0 else (0, 165, 255), -1)
            cv2.putText(frame, f"{lo}-{min(hi, 90)}: {n}/{TARGET_PER_BUCKET}", (x + 3, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", default="DAT", help="mã người ký (thư mục con của letter_landmarks/)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    import mediapipe as mp
    if not os.path.exists(HAND_MODEL):
        print(f"Thiếu {HAND_MODEL}"); sys.exit(1)
    detector = mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Không mở được webcam"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    person_dir = os.path.join(OUT_ROOT, args.person)
    counts = count_existing(person_dir)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    block = 0                       # tăng khi bấm ENTER hoặc đổi chữ -> phiên mới
    letter, prev_wrist = "a", None
    saved = []                      # đường dẫn các khung đã chụp, để BACKSPACE xóa
    msg, msg_until = "", 0.0
    hints = ["chinh dien", "nghieng trai", "nghieng phai", "ngua len", "cui xuong", "xoay ra xa", "gan camera", "xa camera"]
    n_bursts = 0
    print(__doc__)

    def session_id():
        return f"{args.person}_{letter}_{run_stamp}_b{block}"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        hand = extract_hand(res)
        status, angle, good = "KHONG THAY TAY", None, False

        if hand is not None and hand["world"] is not None:
            img = hand["image"]
            angle = view_angle_deg(hand["world"])
            inside = bool(np.all((img[:, :2] > 0.02) & (img[:, :2] < 0.98)))
            wrist_px = img[0, :2] * np.array([w, h])
            palm_px = max(1.0, float(np.linalg.norm((img[9, :2] - img[0, :2]) * np.array([w, h]))))
            step = 0.0 if prev_wrist is None else float(np.linalg.norm(wrist_px - prev_wrist)) / palm_px
            prev_wrist = wrist_px
            if hand["score"] < MIN_SCORE:
                status = f"CHUA CHAC TAY ({hand['score']:.2f})"
            elif not inside:
                status = "TAY CHAM MEP KHUNG"
            elif step > MAX_STEP:
                status = "DI CHUYEN NHANH - GIU YEN"
            else:
                status, good = "OK - san sang", True
            col = (0, 255, 0) if good else (0, 165, 255)
            for x, y, _ in img:
                cv2.circle(frame, (int(x * w), int(y * h)), 3, col, -1)
        else:
            prev_wrist = None

        now = time.time()

        cv2.putText(frame, f"CHU: {letter.upper()}   phien b{block}   [SPACE = chup]",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{status}   goc nghieng: {'-' if angle is None else f'{angle:.0f} deg'}   {('LONG tay -> cam' if palm_faces_camera(hand['world'], hand['handed']) else 'MU tay -> cam') if angle is not None else ''}",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"goi y huong tiep: {hints[n_bursts % len(hints)]}", (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if now < msg_until:
            cv2.putText(frame, msg, (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        draw_coverage(frame, counts, letter)
        cv2.imshow("Thu chu cai da goc nhin", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 32:
            if not good:
                msg, msg_until = f"KHONG LUU: {status}", now + 2.0
            else:
                shot = Session(args.person, letter, "tu_do", (w, h), session_id())
                shot.add(hand, angle)
                p = shot.save(person_dir)
                saved.append(p)
                n_bursts += 1
                counts = count_existing(person_dir)
                msg, msg_until = f"DA CHUP #{n_bursts}  goc {angle:.0f} deg", now + 1.5
        elif k == 13:
            block += 1
            print(f"-> phien moi: b{block}")
        elif k == 8 and saved:
            p = saved.pop()
            os.remove(p)
            counts = count_existing(person_dir)
            msg, msg_until = f"DA XOA {os.path.basename(p)}", now + 2.0
        elif (ord('a') <= k <= ord('z') or ord('0') <= k <= ord('9')) and chr(k) != letter:
            letter = chr(k)
            block += 1              # đổi chữ = phiên mới

    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
