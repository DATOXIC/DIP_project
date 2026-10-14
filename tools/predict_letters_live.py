# -*- coding: utf-8 -*-
"""Chạy thử trực tiếp nhánh chữ cái landmark 3D: py predict_letters_live.py  (ESC để thoát)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import os
import sys

import cv2
import numpy as np

from hand_pose_features import extract_hand, view_angle_deg
from letter_landmark_classifier import DEFAULT_MODEL_PATH, LetterLandmarkClassifier

HAND_MODEL = "hand_landmarker.task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--smooth", type=float, default=0.5, help="hệ số EMA của xác suất (0 = không làm mượt)")
    a = ap.parse_args()
    if not os.path.exists(a.model):
        print(f"Chưa có model {a.model}: chạy train_letter_landmarks.py trước."); sys.exit(1)

    import mediapipe as mp
    clf = LetterLandmarkClassifier(a.model, min_conf=a.min_conf)
    det = mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1))
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(a.camera)
    ema = None
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        hand = extract_hand(res)
        text, color = "-", (200, 200, 200)
        if hand is not None and hand["world"] is not None:
            probs = clf.predict_features(__import__("hand_pose_features").landmarks_to_features(hand["world"], hand["handed"]))[0]
            ema = probs if (ema is None or a.smooth <= 0) else a.smooth * ema + (1 - a.smooth) * probs
            k = int(np.argmax(ema))
            ok_conf = ema[k] >= a.min_conf
            text = f"{clf.classes[k].upper() if ok_conf else '?'}  {ema[k] * 100:.0f}%   goc {view_angle_deg(hand['world']):.0f} deg"
            color = (0, 255, 0) if ok_conf else (0, 165, 255)
            for x, y, _ in hand["image"]:
                cv2.circle(frame, (int(x * w), int(y * h)), 3, color, -1)
        else:
            ema = None
        cv2.putText(frame, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.imshow("Chu cai (landmark 3D chuan hoa)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
