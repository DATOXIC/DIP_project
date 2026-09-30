# -*- coding: utf-8 -*-
"""
Script THU NGHIEM DOC LAP (khong dung production pipeline) de kiem tra gia thuyet:
"MediaPipe Tasks Gesture Recognizer co cho nhan dang tay ON DINH hon Two-Stream CNN cu khong?"

Boi canh: CNN cu (A/B/C/L/V, xem vsl_dataset/metadata.json) da bi tat trong DTW
(lambda_shape_global=0.0, xem dynamic_dtw_engine.py:1526) vi audit thuc nghiem cho thay
nhan "dang tay chiem da so" cua CNN KHONG on dinh giua cac mau CUNG MOT TU (vi du: MOTHER
4 mau ra 4 nhan khac nhau C/V/B/A).

Script nay CHI do 1 chi so: do ON DINH cua nhan dang tay (MediaPipe) giua nhieu lan lap lai
CUNG MOT TU, hoan toan tach roi khoi DTW (dung fusion muon - late fusion - theo dung y tuong
nguoi dung: DTW va HandShape hoat dong doc lap roi moi hop nhat).

QUAN TRONG: Gesture Recognizer mac dinh cua Google CHI co 7 nhan co san
(Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou, None)
- KHONG PHAI bang chu cai ASL/VSL. O buoc nay ta chi kiem tra TINH ON DINH cua tin hieu
(cung mot dang tay that co lap lai ra cung mot nhan khong), chua kiem tra do bao phu tu vung.
Landmarks tho (21 diem) cung duoc ghi lai trong log de sau nay huan luyen custom gesture
(MediaPipe Model Maker) neu buoc nay cho ket qua on dinh.

Cach dung (tren may co webcam, khong chay duoc trong sandbox khong co camera):

    python scratch/mediapipe_handshape_probe.py --word BYE
    # Nhan [SPACE] de bat dau ghi 1 mau (stroke), nhan [SPACE] lan nua de ket thuc mau.
    # Lap lai nhieu lan (>=4-5 lan) CUNG MOT TU --word de do do on dinh.
    # Nhan [q] de thoat.

    python scratch/mediapipe_handshape_probe.py --analyze
    # In bao cao on dinh nhan theo tung tu, doc tu scratch/mediapipe_handshape_probe_log.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "mediapipe_handshape_probe_log.jsonl"

GESTURE_MODEL_PATH = "gesture_recognizer.task"
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)


def _ensure_gesture_model() -> str:
    if not os.path.exists(GESTURE_MODEL_PATH):
        print(f"[-] Dang tai model {GESTURE_MODEL_PATH}...")
        urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
        print("OK: Da tai xong Gesture Recognizer.")
    return GESTURE_MODEL_PATH


def _open_camera_fallback(camera_index: int) -> Optional[cv2.VideoCapture]:
    """Mo camera dung chung quy uoc voi pipeline_common.open_camera (640x480, MJPG, FPS 30)."""
    try:
        from pipeline_common import open_camera  # tai dung helper co san cua project
        return open_camera(camera_index, prefer_mjpg=True)
    except Exception as e:
        print(f"[!] Khong tai duoc pipeline_common.open_camera ({e}), dung VideoCapture thuong.")
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap if cap.isOpened() else None


def _dominant_label(labels: List[str]) -> tuple[str, float]:
    """Nhan chiem da so + ty le, bo qua nhan 'None'/rong (giong _compute_dominant_shape)."""
    counted = [l for l in labels if l and l != "None"]
    if not counted:
        return "None", 0.0
    counts = collections.Counter(counted)
    label, n = counts.most_common(1)[0]
    return label, n / len(counted)


def record_mode(word: str, camera_index: int, min_detection_conf: float) -> None:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        GestureRecognizer, GestureRecognizerOptions, RunningMode,
    )

    model_path = _ensure_gesture_model()
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=min_detection_conf,
        min_hand_presence_confidence=min_detection_conf,
        min_tracking_confidence=min_detection_conf,
    )
    recognizer = GestureRecognizer.create_from_options(options)

    cap = _open_camera_fallback(camera_index)
    if cap is None or not cap.isOpened():
        print(f"[!] KHONG THE mo camera [{camera_index}]!")
        return

    print("=" * 70)
    print(f"  MEDIAPIPE HANDSHAPE PROBE - tu: {word}")
    print("  [SPACE] bat dau/ket thuc 1 mau  |  [q] thoat")
    print("=" * 70)

    recording = False
    frames_buf: List[Dict[str, Any]] = []
    t_start_ms = int(time.time() * 1000)
    sample_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000) - t_start_ms
        result = recognizer.recognize_for_video(mp_image, ts_ms)

        label, score = "None", 0.0
        landmarks_norm: Optional[List[List[float]]] = None
        if result.gestures:
            top = result.gestures[0][0]
            label, score = top.category_name, float(top.score)
        if result.hand_landmarks:
            landmarks_norm = [[p.x, p.y, p.z] for p in result.hand_landmarks[0]]

        if recording:
            frames_buf.append({
                "t_ms": ts_ms, "label": label, "score": score,
                "landmarks_norm": landmarks_norm,
            })

        disp = frame.copy()
        rec_txt = f"REC ({len(frames_buf)} khung)" if recording else "cho [SPACE]"
        color = (0, 0, 255) if recording else (0, 200, 0)
        cv2.putText(disp, f"{word} | {rec_txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(disp, f"label={label} score={score:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(disp, f"mau da luu: {sample_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("MediaPipe Handshape Probe", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            if not recording:
                recording = True
                frames_buf = []
                print("-> Bat dau ghi mau...")
            else:
                recording = False
                if frames_buf:
                    labels = [f["label"] for f in frames_buf]
                    dom_label, dom_ratio = _dominant_label(labels)
                    entry = {
                        "word": word,
                        "session_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "num_frames": len(frames_buf),
                        "dominant_label": dom_label,
                        "dominant_ratio": round(dom_ratio, 3),
                        "frames": frames_buf,
                    }
                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    sample_count += 1
                    print(f"-> Luu mau #{sample_count}: dominant_label={dom_label} "
                          f"(ty le {dom_ratio:.0%}, {len(frames_buf)} khung)")
                else:
                    print("-> Mau rong, bo qua.")

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    print(f"\nDa ghi {sample_count} mau cho tu '{word}' vao {LOG_PATH}")


def analyze_mode() -> None:
    if not LOG_PATH.exists():
        print(f"[!] Chua co log tai {LOG_PATH}. Chay --word <TU> truoc.")
        return

    by_word: Dict[str, List[dict]] = collections.defaultdict(list)
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            by_word[entry["word"]].append(entry)

    print("=" * 78)
    print("BAO CAO DO ON DINH NHAN DANG TAY (MediaPipe Gesture Recognizer)")
    print("=" * 78)
    print(f"{'Tu':<14} {'#mau':<6} {'nhan_da_so_moi_mau':<45} {'#nhan_khac_nhau':<16} {'on_dinh?'}")
    print("-" * 100)
    for word, entries in sorted(by_word.items()):
        dom_labels = [e["dominant_label"] for e in entries]
        distinct = sorted(set(dom_labels))
        stable = len(distinct) == 1
        print(f"{word:<14} {len(entries):<6} {str(dom_labels):<45} {len(distinct):<16} "
              f"{'ON DINH' if stable else 'KHONG ON DINH'}")

    print("\nChi tiet (nhu vi du MOTHER 4 mau -> 4 nhan khac nhau da phat hien voi CNN cu):")
    for word, entries in sorted(by_word.items()):
        print(f"\n  {word}:")
        for i, e in enumerate(entries, 1):
            print(f"    mau {i}: dominant={e['dominant_label']} "
                  f"(ty le {e['dominant_ratio']:.0%}, {e['num_frames']} khung, luc {e['session_ts']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--word", type=str, default=None, help="Tu dang ghi mau (vi du BYE, HELLO).")
    parser.add_argument("--cam", type=int, default=0, help="Chi so camera (mac dinh 0).")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Nguong tin cay phat hien tay.")
    parser.add_argument("--analyze", action="store_true", help="Chi phan tich log da co, khong mo camera.")
    args = parser.parse_args()

    if args.analyze:
        analyze_mode()
        return

    if not args.word:
        parser.error("Can --word <TU> de ghi mau, hoac dung --analyze de xem bao cao.")

    record_mode(args.word.upper(), args.cam, args.min_conf)


if __name__ == "__main__":
    main()
