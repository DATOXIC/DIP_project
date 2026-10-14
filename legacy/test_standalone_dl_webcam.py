"""
CHƯƠNG TRÌNH KIỂM THỬ THỜI GIAN THỰC ĐỘC LẬP MÔ HÌNH HỌC SÂU (STANDALONE DL WEBCAM)
=====================================================================================
Tác giả: Tô Phương Đạt (24110086) - Dự án DIP Project
Mục đích:
  - Kiểm thử 100% ĐỘC LẬP mô hình Deep Learning (LightweightGestureNet: 1D-CNN + Bi-GRU + Attention).
  - TẮT HOÀN TOÀN thuật toán DTW: Không dùng so khớp mẫu, không áp ngưỡng khoảng cách khắt khe.
  - Phản hồi tức thì ngay khi dứt cử chỉ: Độ trễ suy luận chỉ ~1-2 mili-giây trên CPU.
  - Giao diện trực quan:
      + Banner hiển thị Từ nhận diện lớn (Kèm % tin cậy và nghĩa Tiếng Việt).
      + Bảng phân phối Top-3 dự đoán để quan sát góc nhìn phân tích của mạng nơ-ron.
      + Vệt sáng quỹ đạo chuyển động tay thời gian thực.
"""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import os
import sys
import time
import math
import argparse
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_dynamic_dl import LightweightGestureNet, TARGET_WORDS
from dynamic_features import (
    CachedFaceDetector,
    build_segment_features,
    compute_palm_len
)
from dynamic_dtw_engine import KineticPauseSegmenter, GestureFrame
from pipeline_common import (
    open_camera,
    make_wrist_smoother,
    make_segmenter,
    ParallelHandDetector,
    SlidingFPSMeter
)
from predict_sentence_webcam import (
    compute_joint_flexion_angles,
    compute_palm_normal_vector,
    ASL_VOCAB_MAP
)


def draw_standalone_hud(
    frame: np.ndarray,
    status_text: str,
    status_color: Tuple[int, int, int],
    pred_word: Optional[str],
    pred_conf: float,
    top3_preds: List[Tuple[str, float]],
    inf_time_ms: float,
    fps: float,
    v_norm: float
):
    """
    Vẽ giao diện hiển thị thông số và kết quả nhận diện của Deep Learning.
    """
    h, w = frame.shape[:2]

    # 1. Khung trạng thái hệ thống ở góc trên bên trái
    cv2.rectangle(frame, (15, 15), (280, 115), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, 15), (280, 115), (80, 80, 80), 1)

    cv2.putText(frame, "DEEP LEARNING STANDALONE", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Trang thai : {status_text}", (25, 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1)
    cv2.putText(frame, f"Toc do FPS: {fps:4.1f} | Do tre: {inf_time_ms:3.1f}ms", (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Van toc tay: {v_norm:4.2f}", (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # 2. Bảng phân phối xác suất Top-3 ở góc trên bên phải
    cv2.rectangle(frame, (w - 300, 15), (w - 15, 115), (20, 20, 20), -1)
    cv2.rectangle(frame, (w - 300, 15), (w - 15, 115), (80, 80, 80), 1)
    cv2.putText(frame, "PHAN PHOI XAC SUAT (TOP 3)", (w - 290, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

    y_top = 62
    for rank, (cand_w, cand_prob) in enumerate(top3_preds[:3], start=1):
        bar_w = int(cand_prob * 110)
        col = (0, 255, 0) if rank == 1 else (0, 200, 255)
        cv2.putText(frame, f"{rank}. {cand_w:<9}: {cand_prob*100:4.1f}%", (w - 290, y_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        # Thanh đo xác suất
        cv2.rectangle(frame, (w - 130, y_top - 10), (w - 130 + bar_w, y_top), col, -1)
        cv2.rectangle(frame, (w - 130, y_top - 10), (w - 20, y_top), (80, 80, 80), 1)
        y_top += 22

    # 3. Banner lớn hiển thị kết quả từ vựng nhận diện ở chân màn hình
    if pred_word:
        vi_meaning = ASL_VOCAB_MAP.get(pred_word, {}).get("vi", "")
        banner_h = 75
        cv2.rectangle(frame, (15, h - banner_h - 15), (w - 15, h - 15), (10, 10, 10), -1)
        cv2.rectangle(frame, (15, h - banner_h - 15), (w - 15, h - 15), (0, 255, 0), 2)

        result_str = f"{pred_word} ({pred_conf*100:.1f}%)"
        if vi_meaning:
            result_str += f" - {vi_meaning}"

        cv2.putText(frame, result_str, (35, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 0), 3)


def run_standalone_dl_webcam(
    weights_path: str = "best_gesture_model_none.pth",
    camera_source: int = 1,
    min_confidence: float = 0.35
):
    print("=" * 72)
    print("  KIỂM THỬ THỜI GIAN THỰC ĐỘC LẬP MÔ HÌNH DEEP LEARNING (VSL/ASL)")
    print(f"  Trọng số mô hình : {weights_path}")
    print(f"  Nguồn Camera     : Index {camera_source}")
    print("=" * 72)

    # 1. Tự động tìm file model nếu file chỉ định không tồn tại
    if not os.path.exists(weights_path):
        candidates = ["best_gesture_model_none.pth", "best_gesture_model_Dang.pth", "best_gesture_model_Mai.pth"]
        found = False
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                found = True
                print(f"[!] Chuyển sang dùng mô hình tìm thấy: {weights_path}")
                break
        if not found:
            print(f"[LỖI] Không tìm thấy file trọng số mô hình tại {weights_path}!")
            return

    # 2. Khởi tạo mô hình Deep Learning trên CPU
    device = torch.device("cpu")
    model = LightweightGestureNet(in_features=24, num_classes=len(TARGET_WORDS), hidden_dim=64)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Đã nạp thành công mô hình Deep Learning ({sum(p.numel() for p in model.parameters()):,} tham số)!")

    # 3. Khởi tạo Webcam
    cap = open_camera(camera_source)
    if cap is None or not cap.isOpened():
        print(f"[!] Không thể mở camera index {camera_source}. Thử chuyển sang camera 0...")
        cap = open_camera(0)
    if cap is None or not cap.isOpened():
        print("[LỖI] Không thể mở bất kỳ camera nào!")
        return

    # 4. Khởi tạo Detector và Segmenter
    detector_pool = ParallelHandDetector(model_path="hand_landmarker.task", workers=2, num_hands=1)
    wrist_smoother = make_wrist_smoother()
    segmenter = make_segmenter()
    fps_meter = SlidingFPSMeter()
    face_detector = CachedFaceDetector(detect_every_n_frames=8)

    motion_trail: deque = deque(maxlen=25)
    buffered_frames: List[Dict[str, Any]] = []

    last_pred_word: Optional[str] = None
    last_pred_conf: float = 0.0
    last_top3: List[Tuple[str, float]] = []
    last_inf_time_ms: float = 0.0
    banner_expire_time: float = 0.0

    flip_cam = False
    ready_queue: deque = deque()

    print("\n👉 Hướng dẫn thao tác:")
    print("  - Đưa tay vào khung hình và thực hiện cử chỉ tự nhiên.")
    print("  - Khi bạn vừa dứt cử chỉ, mô hình sẽ tự động bắt đoạn và suy luận.")
    print("  - Phím [f]: Bật/Tắt lật gương camera (Flip)")
    print("  - Phím [c]: Xóa từ hiển thị (Clear)")
    print("  - Phím [q]: Thoát chương trình\n")

    frame_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            capture_ts = time.time()
            if flip_cam:
                frame = cv2.flip(frame, 1)

            h_f, w_f = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            meta = {"capture_ts": capture_ts, "frame": frame, "rgb": rgb_frame, "h": h_f, "w": w_f}
            frame_counter += 1
            detector_pool.submit(frame_counter, rgb_frame, meta)

            new_ready = detector_pool.poll(block_oldest=(detector_pool.in_flight >= detector_pool.max_in_flight))
            ready_queue.extend(new_ready)

            while ready_queue:
                seq, it_meta, det_res = ready_queue.popleft()
                cur_frame = it_meta["frame"]
                cur_rgb = it_meta["rgb"]
                h_f = it_meta["h"]
                w_f = it_meta["w"]

                fps = fps_meter.update()
                status_text = "CHO BAN TAY..."
                status_color = (150, 150, 150)
                cur_v_norm = 0.0

                if det_res is not None and det_res.hand_landmarks and len(det_res.hand_landmarks) > 0:
                    lms = det_res.hand_landmarks[0]
                    handed_label = det_res.handedness[0][0].category_name if det_res.handedness else "Right"

                    # 1. Trích xuất tọa độ xương & làm mượt cổ tay
                    wrist_raw_x = float(lms[0].x * w_f)
                    wrist_raw_y = float(lms[0].y * h_f)
                    wx, wy = wrist_smoother.update(wrist_raw_x, wrist_raw_y, it_meta["capture_ts"])
                    motion_trail.append((int(wx), int(wy)))

                    # 2. Tính palm_len — gọi đúng signature:
                    #    compute_palm_len(lm0, lm5, lm9, lm17, w_f, h_f)
                    #    với lm0/5/9/17 là landmark object có .x/.y/.z chuẩn hóa
                    palm_len = compute_palm_len(lms[0], lms[5], lms[9], lms[17], float(w_f), float(h_f))

                    # 3. Tính góc khớp & pháp tuyến lòng bàn tay
                    lm_pixels = np.array([[lm.x * w_f, lm.y * h_f, lm.z * w_f] for lm in lms], dtype=np.float32)
                    angles_5d = compute_joint_flexion_angles(lm_pixels)
                    normal_3d = compute_palm_normal_vector(lm_pixels, handed_label)

                    # 4. Phát hiện mặt (CachedFaceDetector.detect nhận ảnh RGB)
                    face_box = face_detector.detect(cur_rgb)

                    # 5. Đưa vào Bộ dò điểm dừng (Kinetic Pause Segmenter)
                    g_frame = GestureFrame(
                        x=wx, y=wy,
                        timestamp=it_meta["capture_ts"],
                        palm_len=palm_len,
                        hand_shape="Unknown",
                        hand_shape_conf=None,
                        v_norm=0.0
                    )
                    state, is_moving, cur_v_norm = segmenter.update(g_frame)

                    # Lưu lại bản ghi khung hình cho cử chỉ hiện tại
                    frame_record = {
                        "x": wx, "y": wy,
                        "palm_len": palm_len,
                        "angles": angles_5d,
                        "normal": normal_3d,
                        "landmarks": lm_pixels,
                        "face_box": face_box,
                        "handedness": handed_label
                    }

                    if is_moving:
                        status_text = "DANG KY CU CHI..."
                        status_color = (0, 165, 255)
                        buffered_frames.append(frame_record)
                    else:
                        status_text = "TAY DUNG YEN"
                        status_color = (0, 255, 0)
                        if len(buffered_frames) > 0:
                            buffered_frames.append(frame_record)

                    # 6. Khi bộ phân đoạn chốt dứt điểm cử chỉ (Stroke Ended)
                    if state.name in ["STROKE_END", "PAUSE_WAIT", "COMMITTED"] and len(buffered_frames) >= 8:
                        t0_inf = time.perf_counter()

                        # Trích xuất Tensor 32x24 chuẩn
                        res_traj_24d, _, _, _ = build_segment_features(buffered_frames, k_target=32)

                        if res_traj_24d is not None and res_traj_24d.shape == (32, 24):
                            # Suy luận trực tiếp qua Deep Learning
                            with torch.no_grad():
                                inp_tensor = torch.from_numpy(res_traj_24d[None, ...]).to(device)
                                logits = model(inp_tensor)
                                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

                            last_inf_time_ms = (time.perf_counter() - t0_inf) * 1000.0

                            # Sắp xếp Top-3
                            sorted_idx = np.argsort(probs)[::-1]
                            last_top3 = [(TARGET_WORDS[idx], float(probs[idx])) for idx in sorted_idx[:3]]

                            best_word, best_prob = last_top3[0]
                            if best_prob >= min_confidence:
                                last_pred_word = best_word
                                last_pred_conf = best_prob
                                banner_expire_time = time.time() + 2.5
                                print(f"[DEEP LEARNING] Dự đoán: {best_word:<10} ({best_prob*100:5.1f}%) | Độ trễ: {last_inf_time_ms:3.1f}ms")

                        buffered_frames.clear()

                else:
                    wrist_smoother.reset()
                    segmenter.reset()
                    motion_trail.clear()
                    buffered_frames.clear()

                # Vẽ vệt chuyển động của tay
                if len(motion_trail) >= 2:
                    for i in range(1, len(motion_trail)):
                        alpha = i / len(motion_trail)
                        color = (int(0 * alpha), int(255 * alpha), int(255 * alpha))
                        cv2.line(cur_frame, motion_trail[i - 1], motion_trail[i], color, 2)

                # Kiểm tra hạn banner
                cur_banner_word = last_pred_word if time.time() < banner_expire_time else None

                # Vẽ HUD và hiển thị
                draw_standalone_hud(
                    frame=cur_frame,
                    status_text=status_text,
                    status_color=status_color,
                    pred_word=cur_banner_word,
                    pred_conf=last_pred_conf,
                    top3_preds=last_top3,
                    inf_time_ms=last_inf_time_ms,
                    fps=fps,
                    v_norm=cur_v_norm
                )

                cv2.imshow("Test Doc Lap Mo Hinh Deep Learning (DIP Project)", cur_frame)

            # Phím điều khiển
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break
            elif key in [ord('f'), ord('F')]:
                flip_cam = not flip_cam
                print(f"-> Flip Camera: {'BẬT (Gương)' if flip_cam else 'TẮT'}")
            elif key in [ord('c'), ord('C')]:
                last_pred_word = None
                last_top3.clear()
                print("-> Đã xóa kết quả hiển thị.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector_pool.close()
        print("\n✓ Đã kết thúc phiên kiểm thử Deep Learning độc lập.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử thời gian thực độc lập mô hình Deep Learning qua Webcam")
    parser.add_argument("--weights", type=str, default="best_gesture_model_none.pth",
                        help="Đường dẫn tới file trọng số .pth của mô hình")
    parser.add_argument("--camera", type=int, default=1,
                        help="Chỉ số camera (mặc định: 1)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Ngưỡng tin cậy tối thiểu để hiển thị kết quả (mặc định: 0.35)")
    args = parser.parse_args()

    run_standalone_dl_webcam(
        weights_path=args.weights,
        camera_source=args.camera,
        min_confidence=args.threshold
    )
