"""
INSPECT & TRIM ASL CITIZEN - CÔNG CỤ TRỰC QUAN KIỂM TRA & CẮT VIDEO ASL CITIZEN
=============================================================================
Công cụ tương tác chuyên nghiệp giúp:
1. Duyệt từng video trong thư mục ASL_Citizen_videos theo từng từ vựng.
2. Di chuyển tới/lui từng frame (frame scrubbing), Play/Pause, xem vận tốc và dáng tay CNN.
3. Tự tay can thiệp đặt Start Frame và End Frame (loại bỏ pha nhấc tay và hạ tay).
4. Xem trước (Preview loop) đoạn cử chỉ chính đã cắt.
5. Đóng gói chuẩn 12D Golden Triad (32 frames) và lưu vào file template JSON.
"""

import os
import sys
import json
import math
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp

# Tích hợp các module sẵn có của dự án
from train import TwoStreamHandCNN
from improved_dip_pipeline import (
    extract_smart_hand_crop,
    process_lean_dip_tensor
)
from predict_sentence_webcam import (
    compute_joint_flexion_angles,
    compute_palm_normal_vector,
    resample_9d_gesture_sequence
)

# Kết nối xương bàn tay MediaPipe 21 điểm (để vẽ thủ công đẹp mắt)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),        # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),   # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16), # Ngón nhẫn
    (13, 17), (17, 18), (18, 19), (19, 20),# Ngón út
    (0, 17)                                 # Cạnh bàn tay
]


class ASLCitizenTrimmer:
    def __init__(
        self,
        input_dir: str = "ASL_Citizen_videos",
        output_json: str = "asl_citizen_templates.json",
        target_sign: Optional[str] = None
    ):
        self.input_dir = input_dir
        self.output_json = output_json
        self.target_sign = target_sign.upper() if target_sign else None

        # 1. Khởi tạo MediaPipe HandLandmarker (Chế độ IMAGE để chạy chính xác từng frame)
        model_task_path = "hand_landmarker.task"
        if not os.path.exists(model_task_path):
            print(f"[-] Đang tải {model_task_path}...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_task_path)
            print("✓ Tải xong model MediaPipe HandLandmarker!")

        base_options = mp.tasks.BaseOptions(model_asset_path=model_task_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        print("✓ Đã nạp MediaPipe HandLandmarker thành công.")

        # 2. Khởi tạo Two-Stream Hand CNN
        self.device = torch.device("cpu")
        model_path = "vsl_dataset/models/best_model_v2.pth"
        if not os.path.exists(model_path):
            model_path = "vsl_dataset/models/best_model_run2.pth"

        self.cnn_model = None
        self.classes = ['A', 'B', 'C', 'L', 'V']
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                id_map = checkpoint.get("id_to_label")
                if id_map:
                    self.classes = [id_map[i] for i in sorted(id_map.keys())]
                num_classes = checkpoint.get("num_classes", len(self.classes))
                self.cnn_model = TwoStreamHandCNN(num_classes=num_classes)
                state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
                self.cnn_model.load_state_dict(state_dict)
                self.cnn_model.to(self.device)
                self.cnn_model.eval()
                print(f"✓ Đã nạp Two-Stream CNN ({self.classes}) từ '{model_path}'.")
            except Exception as e:
                print(f"[!] Lỗi nạp Two-Stream CNN: {e}")
        else:
            print("[!] CẢNH BÁO: Không tìm thấy model CNN, sẽ dùng nhãn mặc định 'Unknown'.")

        # 3. Quét danh sách video
        self.video_list: List[Tuple[str, str]] = []  # List of (sign, filepath)
        self._scan_videos()

    def _scan_videos(self):
        if not os.path.exists(self.input_dir):
            print(f"[!] Thư mục {self.input_dir} không tồn tại!")
            return

        for root, dirs, files in os.walk(self.input_dir):
            rel = os.path.relpath(root, self.input_dir)
            if rel == ".":
                continue
            sign_name = os.path.basename(root).upper()
            if self.target_sign and sign_name != self.target_sign:
                continue

            for f in files:
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    full_path = os.path.join(root, f)
                    self.video_list.append((sign_name, full_path))

        print(f"✓ Tìm thấy {len(self.video_list)} video" + (f" cho từ '{self.target_sign}'" if self.target_sign else " trong tập dữ liệu."))

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Đọc và trích xuất đặc trưng toàn bộ frames của 1 video vào RAM."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            raw_frames.append(frame)
        cap.release()

        n_frames = len(raw_frames)
        if n_frames == 0:
            return {}

        parsed_data = {
            "fps": fps,
            "total_frames": n_frames,
            "raw_frames": raw_frames,
            "has_hand": [False] * n_frames,
            "wrist_coords": [(0.0, 0.0)] * n_frames,
            "palm_lens": [40.0] * n_frames,
            "landmarks_3d": [None] * n_frames,
            "angles_5d": [np.zeros(5, dtype=np.float32)] * n_frames,
            "normals_3d": [np.array([0, 0, -1], dtype=np.float32)] * n_frames,
            "hand_shapes": ["Unknown"] * n_frames,
            "shape_confs": [0.0] * n_frames,
            "v_norms": [0.0] * n_frames,
            "handedness": ["Right"] * n_frames,
            "crop_boxes": [None] * n_frames
        }

        h_f, w_f = raw_frames[0].shape[:2]

        for i, frame in enumerate(raw_frames):
            rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_f)
            det = self.detector.detect(mp_img)

            if det.hand_landmarks and len(det.hand_landmarks) > 0:
                parsed_data["has_hand"][i] = True
                lm = det.hand_landmarks[0]
                handed_label = det.handedness[0][0].category_name if det.handedness else "Right"
                parsed_data["handedness"][i] = handed_label

                # Cổ tay & kích thước lòng bàn tay
                lm0 = lm[0]
                lm5 = lm[5]
                lm9 = lm[9]
                lm17 = lm[17]
                wx, wy = lm0.x * w_f, lm0.y * h_f
                parsed_data["wrist_coords"][i] = (wx, wy)

                palm_2d = math.hypot((lm9.x - lm0.x) * w_f, (lm9.y - lm0.y) * h_f)
                palm_3d = math.sqrt(((lm9.x - lm0.x) * w_f)**2 + ((lm9.y - lm0.y) * h_f)**2 + ((lm9.z - lm0.z) * w_f)**2)
                knuckle_span = math.hypot((lm17.x - lm5.x) * w_f, (lm17.y - lm5.y) * h_f)
                plen = max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)
                parsed_data["palm_lens"][i] = plen

                # 3D Coordinates, 5D Angles, 3D Normal
                lm_3d = np.array([[l.x * w_f, l.y * h_f, l.z * w_f] for l in lm], dtype=np.float32)
                parsed_data["landmarks_3d"][i] = lm_3d
                parsed_data["angles_5d"][i] = compute_joint_flexion_angles(lm_3d)
                parsed_data["normals_3d"][i] = compute_palm_normal_vector(lm_3d, handedness=handed_label)

                # Dáng tay CNN
                hand_crop, (x1, y1, x2, y2), _, _ = extract_smart_hand_crop(
                    frame, lm, handedness_label=handed_label, auto_flip_left=False
                )
                parsed_data["crop_boxes"][i] = (x1, y1, x2, y2)

                if hand_crop.shape[0] >= 25 and hand_crop.shape[1] >= 25 and self.cnn_model is not None:
                    tensor_np, _, _ = process_lean_dip_tensor(hand_crop, target_size=(128, 128))
                    t_pt = torch.from_numpy(tensor_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.cnn_model(t_pt)
                        probs = F.softmax(logits, dim=1).numpy()[0]
                        max_idx = int(np.argmax(probs))
                        parsed_data["hand_shapes"][i] = self.classes[max_idx]
                        parsed_data["shape_confs"][i] = float(probs[max_idx])

        # Tính vận tốc chuẩn hóa v_norm = delta_w / palm_len
        for i in range(1, n_frames):
            if parsed_data["has_hand"][i] and parsed_data["has_hand"][i - 1]:
                x1, y1 = parsed_data["wrist_coords"][i - 1]
                x2, y2 = parsed_data["wrist_coords"][i]
                pl = (parsed_data["palm_lens"][i] + parsed_data["palm_lens"][i - 1]) / 2.0
                dist = math.hypot(x2 - x1, y2 - y1)
                parsed_data["v_norms"][i] = dist / max(pl, 15.0)

        # Gợi ý điểm cắt tự động (Auto-suggest Start & End)
        suggest_start, suggest_end = self._auto_suggest_cut(parsed_data, h_f)
        parsed_data["suggest_start"] = suggest_start
        parsed_data["suggest_end"] = suggest_end

        return parsed_data

    def _auto_suggest_cut(self, data: Dict[str, Any], h_f: int) -> Tuple[int, int]:
        """Thuật toán thông minh tự động tìm điểm bắt đầu Stroke và kết thúc Stroke."""
        n = data["total_frames"]
        if n <= 10:
            return 0, n - 1

        # 1. Tìm start_idx: Sau khi tay đã nhấc lên (y < 0.82 chiều cao ảnh)
        start_idx = 0
        for i in range(n):
            if data["has_hand"][i]:
                wy = data["wrist_coords"][i][1]
                if wy < h_f * 0.82:
                    start_idx = max(0, i)
                    break

        # Tinh chỉnh start_idx: Tìm điểm khựng vận tốc đầu tiên sau khi nhấc tay
        # Hoặc nơi vận tốc chuyển sang pha chuyển động chính
        for i in range(start_idx, min(start_idx + 15, n - 1)):
            if data["v_norms"][i] < 0.045 and data["v_norms"][min(i + 1, n - 1)] > 0.05:
                start_idx = i
                break

        # 2. Tìm end_idx: Khi cử chỉ kết thúc và tay khựng lại, hoặc trước khi rơi xuống đáy
        end_idx = n - 1
        # Tìm từ cuối lên: khi tay bắt đầu rơi xuống đáy (y > 0.85)
        for i in range(n - 1, start_idx + 5, -1):
            if data["has_hand"][i]:
                wy = data["wrist_coords"][i][1]
                if wy < h_f * 0.85:
                    end_idx = i
                    break

        # Tìm điểm khựng dừng cuối cùng của cử chỉ
        for i in range(start_idx + 8, end_idx):
            # Nếu vận tốc rất thấp liên tục 4-6 frames thì có thể cử chỉ đã dừng
            window = data["v_norms"][i:i + 5]
            if len(window) >= 4 and all(v < 0.038 for v in window):
                end_idx = min(i + 2, n - 1)
                break

        if end_idx <= start_idx + 4:
            end_idx = min(start_idx + 24, n - 1)

        return int(start_idx), int(end_idx)

    def save_sample_to_json(
        self,
        sign: str,
        video_filename: str,
        data: Dict[str, Any],
        start_frame: int,
        end_frame: int
    ) -> bool:
        """Trích xuất chuỗi 12D từ start_frame đến end_frame, resample về 32 frames và lưu JSON."""
        n_stroke = end_frame - start_frame + 1
        if n_stroke < 6:
            print("[!] Đoạn cắt quá ngắn (< 6 frames), không thể lưu mẫu!")
            return False

        # Thu thập tọa độ và các đặc trưng trong đoạn cắt
        stroke_wrists = [data["wrist_coords"][i] for i in range(start_frame, end_frame + 1)]
        stroke_palms = [data["palm_lens"][i] for i in range(start_frame, end_frame + 1)]
        stroke_angles = [data["angles_5d"][i] for i in range(start_frame, end_frame + 1)]
        stroke_normals = [data["normals_3d"][i] for i in range(start_frame, end_frame + 1)]
        stroke_shapes = [data["hand_shapes"][i] for i in range(start_frame, end_frame + 1)]

        # Chuẩn hóa tọa độ theo mỏ neo ban đầu (Stroke-origin) và Palm length
        x0, y0 = stroke_wrists[0]
        ref_palm = max(stroke_palms[0], 15.0)

        raw_xs = np.array([(w[0] - x0) / ref_palm for w in stroke_wrists], dtype=np.float32)
        raw_ys = np.array([(w[1] - y0) / ref_palm for w in stroke_wrists], dtype=np.float32)

        raw_dxs = np.zeros(n_stroke, dtype=np.float32)
        raw_dys = np.zeros(n_stroke, dtype=np.float32)
        raw_dxs[1:] = raw_xs[1:] - raw_xs[:-1]
        raw_dys[1:] = raw_ys[1:] - raw_ys[:-1]
        raw_dxs[0] = raw_dxs[1] if n_stroke > 1 else 0.0
        raw_dys[0] = raw_dys[1] if n_stroke > 1 else 0.0

        raw_angles_mat = np.array(stroke_angles, dtype=np.float32)
        raw_normals_mat = np.array(stroke_normals, dtype=np.float32)

        raw_12d = np.column_stack([
            raw_xs, raw_ys, raw_dxs, raw_dys, raw_angles_mat, raw_normals_mat
        ]).astype(np.float32)

        # Resample về đúng K = 32 frames
        res_traj_12d, res_shapes = resample_9d_gesture_sequence(raw_12d, stroke_shapes, k_target=32, min_frames=6)
        if res_traj_12d is None or res_shapes is None:
            print("[!] Lỗi khi resample chuỗi cử chỉ!")
            return False

        duration = n_stroke / data["fps"]

        # Đóng gói sample dictionary
        sample_dict = {
            "traj_features": [[round(float(val), 6) for val in row] for row in res_traj_12d],
            "hand_shapes": res_shapes,
            "duration_sec": round(float(duration), 3),
            "source_video": video_filename,
            "trim_range": [int(start_frame), int(end_frame)],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Nạp hoặc tạo mới file JSON
        db = {
            "metadata": {
                "k_resample_frames": 32,
                "feature_dim": 12,
                "description": "ASL Citizen Extracted 12D Templates (Interactive Trimmed)",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "vocabulary": {}
        }

        if os.path.exists(self.output_json):
            try:
                with open(self.output_json, "r", encoding="utf-8") as f:
                    db = json.load(f)
            except Exception as e:
                print(f"[!] Cảnh báo đọc file cũ {self.output_json}: {e}")

        vocab = db.setdefault("vocabulary", {})
        if sign not in vocab:
            vocab[sign] = {
                "display_name": sign,
                "threshold": 1.25,
                "num_samples": 0,
                "samples": []
            }

        vocab[sign]["samples"].append(sample_dict)
        vocab[sign]["num_samples"] = len(vocab[sign]["samples"])
        db["metadata"]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)

        print(f"✓ ĐÃ LƯU MẪU MỚI: Từ '{sign}' hiện có {vocab[sign]['num_samples']} mẫu trong '{self.output_json}'.")
        return True

    def run(self):
        if not self.video_list:
            print("[-] Không có video nào để xử lý. Vui lòng kiểm tra lại thư mục!")
            return

        cv2.namedWindow("ASL Citizen - Interactive Inspector & Trimmer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ASL Citizen - Interactive Inspector & Trimmer", 1024, 780)

        curr_vid_idx = 0
        total_vids = len(self.video_list)

        while 0 <= curr_vid_idx < total_vids:
            sign_name, video_path = self.video_list[curr_vid_idx]
            video_fname = os.path.basename(video_path)

            print("\n" + "=" * 75)
            print(f"ĐANG NẠP VIDEO [{curr_vid_idx + 1}/{total_vids}]: {sign_name} / {video_fname}")
            print("=" * 75)

            data = self.analyze_video(video_path)
            if not data or data.get("total_frames", 0) == 0:
                print(f"[!] Không đọc được video: {video_path}, tự động chuyển sang video tiếp theo.")
                curr_vid_idx += 1
                continue

            n_frames = data["total_frames"]
            start_f = data["suggest_start"]
            end_f = data["suggest_end"]

            curr_frame_idx = start_f
            is_playing = False
            is_preview_loop = False
            flash_message = f"Goi y Start: {start_f}, End: {end_f}"
            flash_timer = time.time() + 2.0

            while True:
                # 1. Lấy frame ảnh gốc
                frame = data["raw_frames"][curr_frame_idx].copy()
                h, w = frame.shape[:2]

                # Resize khung hiển thị cho đồng đều
                target_w, target_h = 760, 480
                display_frame = cv2.resize(frame, (target_w, target_h))
                scale_x = target_w / w
                scale_y = target_h / h

                # 2. Vẽ Landmarks & Khung xương
                if data["has_hand"][curr_frame_idx] and data["landmarks_3d"][curr_frame_idx] is not None:
                    lm_3d = data["landmarks_3d"][curr_frame_idx]
                    # Vẽ các đường nối xương tay
                    for p1, p2 in HAND_CONNECTIONS:
                        x1 = int(lm_3d[p1][0] * scale_x)
                        y1 = int(lm_3d[p1][1] * scale_y)
                        x2 = int(lm_3d[p2][0] * scale_x)
                        y2 = int(lm_3d[p2][1] * scale_y)
                        cv2.line(display_frame, (x1, y1), (x2, y2), (0, 220, 220), 2)

                    # Vẽ 21 khớp
                    for p in range(21):
                        px = int(lm_3d[p][0] * scale_x)
                        py = int(lm_3d[p][1] * scale_y)
                        cv2.circle(display_frame, (px, py), 4, (0, 0, 255) if p == 0 else (255, 100, 0), -1)

                    # Vẽ vector pháp tuyến (Normal Arrow) từ trọng tâm bàn tay (Knuckle 9)
                    norm_vec = data["normals_3d"][curr_frame_idx]
                    kn_x = int(lm_3d[9][0] * scale_x)
                    kn_y = int(lm_3d[9][1] * scale_y)
                    arrow_x = int(kn_x + norm_vec[0] * 50)
                    arrow_y = int(kn_y + norm_vec[1] * 50)
                    cv2.arrowedLine(display_frame, (kn_x, kn_y), (arrow_x, arrow_y), (0, 255, 255), 3, tipLength=0.3)

                # Vẽ vệt đuôi chuyển động (Motion Trail) từ start_f đến curr_frame_idx
                trail_pts = []
                for f_i in range(start_f, curr_frame_idx + 1):
                    if data["has_hand"][f_i]:
                        wx, wy = data["wrist_coords"][f_i]
                        trail_pts.append((int(wx * scale_x), int(wy * scale_y)))
                for k in range(1, len(trail_pts)):
                    alpha = k / max(len(trail_pts), 1)
                    thickness = int(1 + 3 * alpha)
                    cv2.line(display_frame, trail_pts[k - 1], trail_pts[k], (0, int(255 * alpha), 255), thickness)

                # Dán nhãn CNN và Bounding Box
                hand_shape = data["hand_shapes"][curr_frame_idx]
                shape_conf = data["shape_confs"][curr_frame_idx]
                cv2.rectangle(display_frame, (15, 15), (280, 85), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (15, 15), (280, 85), (0, 200, 200), 1)
                cv2.putText(display_frame, f"Sign: {sign_name}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(display_frame, f"CNN Shape: '{hand_shape}' ({shape_conf*100:.1f}%)", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

                # Huy hiệu trạng thái phân đoạn (Phase Badge)
                if curr_frame_idx < start_f:
                    phase_text = "[1. NHAC TAY LEN (BO QUA)]"
                    badge_color = (0, 0, 220)  # Đỏ
                elif start_f <= curr_frame_idx <= end_f:
                    phase_text = "[2. CU CHI CHINH (STROKE)]"
                    badge_color = (0, 200, 0)  # Xanh lá
                else:
                    phase_text = "[3. HA TAY XUONG (BO QUA)]"
                    badge_color = (0, 140, 255)  # Cam

                cv2.rectangle(display_frame, (target_w - 360, 15), (target_w - 15, 55), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (target_w - 360, 15), (target_w - 15, 55), badge_color, 2)
                cv2.putText(display_frame, phase_text, (target_w - 345, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, badge_color, 2)

                # 3. Tạo Bảng điều khiển HUD & Timeline bên dưới (Cao 260px)
                hud_h = 260
                hud = np.zeros((hud_h, target_w, 3), dtype=np.uint8)
                hud[:] = (25, 25, 28)

                # Dòng 1: Thông tin video & Trạng thái tua
                status_str = "PREVIEW LOOP" if is_preview_loop else ("PLAYING" if is_playing else "PAUSED")
                status_col = (0, 255, 255) if is_preview_loop else ((0, 255, 0) if is_playing else (180, 180, 180))
                info_line = f"Video [{curr_vid_idx + 1}/{total_vids}]: {video_fname} | {n_frames} frames"
                cv2.putText(hud, info_line, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                cv2.putText(hud, f"Trang thai: {status_str}", (target_w - 240, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2)

                # Dòng 2: THANH TIẾN TRÌNH TIMELINE TRỰC QUAN
                t_x1, t_x2 = 40, target_w - 40
                t_y = 65
                t_w = t_x2 - t_x1

                # Nền thanh timeline
                cv2.rectangle(hud, (t_x1, t_y - 12), (t_x2, t_y + 12), (45, 45, 50), -1)

                # Vùng chọn Stroke (Start -> End) màu xanh lá cây
                s_px = int(t_x1 + (start_f / max(n_frames - 1, 1)) * t_w)
                e_px = int(t_x1 + (end_f / max(n_frames - 1, 1)) * t_w)
                cv2.rectangle(hud, (s_px, t_y - 12), (e_px, t_y + 12), (0, 160, 60), -1)
                cv2.rectangle(hud, (s_px, t_y - 12), (e_px, t_y + 12), (0, 255, 100), 2)

                # Vạch Start & End Marker
                cv2.line(hud, (s_px, t_y - 18), (s_px, t_y + 18), (255, 255, 255), 2)
                cv2.line(hud, (e_px, t_y - 18), (e_px, t_y + 18), (255, 255, 255), 2)
                cv2.putText(hud, f"START: {start_f}", (max(10, s_px - 35), t_y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
                cv2.putText(hud, f"END: {end_f}", (min(target_w - 90, e_px - 20), t_y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

                # Con trỏ Current Frame
                c_px = int(t_x1 + (curr_frame_idx / max(n_frames - 1, 1)) * t_w)
                cv2.line(hud, (c_px, t_y - 15), (c_px, t_y + 15), (0, 220, 255), 3)
                cv2.circle(hud, (c_px, t_y), 6, (0, 220, 255), -1)

                # Dòng 3: Đồ thị vận tốc v_norm(t) thu nhỏ
                g_y_base = 150
                g_h = 45
                cv2.putText(hud, "Do thi van toc v(t):", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
                cv2.line(hud, (t_x1, g_y_base), (t_x2, g_y_base), (60, 60, 65), 1)

                # Vẽ đường vận tốc
                v_pts = []
                max_v = max(max(data["v_norms"]), 0.15)
                for i_v, val in enumerate(data["v_norms"]):
                    px = int(t_x1 + (i_v / max(n_frames - 1, 1)) * t_w)
                    py = int(g_y_base - min((val / max_v) * g_h, g_h))
                    v_pts.append((px, py))
                for k in range(1, len(v_pts)):
                    cv2.line(hud, v_pts[k - 1], v_pts[k], (200, 140, 50), 1)

                # Vạch ngưỡng khựng 0.04 trên đồ thị
                pause_y = int(g_y_base - (0.04 / max_v) * g_h)
                cv2.line(hud, (t_x1, pause_y), (t_x2, pause_y), (100, 100, 100), 1)

                # Điểm vận tốc hiện tại
                curr_v = data["v_norms"][curr_frame_idx]
                cv2.putText(hud, f"Frame: {curr_frame_idx}/{n_frames-1} | v_norm: {curr_v:.3f} | Do dai cat: {end_f - start_f + 1} frames ({(end_f - start_f + 1)/data['fps']:.2f}s)",
                            (180, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 230), 1)

                # Dòng 4: Bảng điều khiển phím tắt
                cv2.line(hud, (15, 170), (target_w - 15, 170), (50, 50, 55), 1)
                controls_1 = "[Space] Play/Pause  |  [A / D] Lui/Tien 1 frame  |  [Q / E] -5 / +5 frames"
                controls_2 = "[ [ / S ] Dat START  |  [ ] / F ] Dat END  |  [P] Preview Lap doan cat"
                controls_3 = "[Y] CHAP NHAN LUU MAU  |  [N] Bo qua video  |  [B] Quay lai  |  [Esc] Thoat"

                cv2.putText(hud, controls_1, (20, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
                cv2.putText(hud, controls_2, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 240, 100), 1)
                cv2.putText(hud, controls_3, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 200, 80), 1)

                # Banner thông báo (Flash Banner)
                if time.time() < flash_timer:
                    cv2.rectangle(hud, (target_w - 380, 178), (target_w - 20, 215), (0, 120, 0), -1)
                    cv2.putText(hud, flash_message, (target_w - 370, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Ghép Video và HUD thành 1 cửa sổ hoàn chỉnh
                combined_window = np.vstack([display_frame, hud])
                cv2.imshow("ASL Citizen - Interactive Inspector & Trimmer", combined_window)

                # Xử lý thời gian chờ phím (Play mượt mà)
                wait_time = int(1000 / data["fps"]) if (is_playing or is_preview_loop) else 40
                key = cv2.waitKey(wait_time) & 0xFF

                # Tự động tăng frame khi Play hoặc Preview
                if is_playing:
                    curr_frame_idx += 1
                    if curr_frame_idx >= n_frames:
                        curr_frame_idx = 0

                elif is_preview_loop:
                    curr_frame_idx += 1
                    if curr_frame_idx > end_f:
                        curr_frame_idx = start_f

                # Bắt phím bấm điều khiển
                if key in [ord('q'), ord('Q'), 27]:  # Esc hoặc Q (khi không scrub)
                    if key == 27:
                        print("[-] Thoát chương trình.")
                        cv2.destroyAllWindows()
                        return
                    else:
                        # Q lùi 5 frames
                        curr_frame_idx = max(0, curr_frame_idx - 5)

                elif key == 32:  # Phím Space: Play / Pause
                    is_playing = not is_playing
                    is_preview_loop = False

                elif key in [ord('d'), ord('D'), 83]:  # Tiến 1 frame (phím D hoặc Mũi tên phải)
                    is_playing = False
                    is_preview_loop = False
                    curr_frame_idx = min(n_frames - 1, curr_frame_idx + 1)

                elif key in [ord('a'), ord('A'), 81]:  # Lùi 1 frame (phím A hoặc Mũi tên trái)
                    is_playing = False
                    is_preview_loop = False
                    curr_frame_idx = max(0, curr_frame_idx - 1)

                elif key in [ord('e'), ord('E')]:  # Tiến nhanh 5 frames
                    is_playing = False
                    is_preview_loop = False
                    curr_frame_idx = min(n_frames - 1, curr_frame_idx + 5)

                elif key in [ord('['), ord('s'), ord('S')]:  # ĐẶT START FRAME
                    start_f = curr_frame_idx
                    if start_f >= end_f:
                        end_f = min(start_f + 10, n_frames - 1)
                    flash_message = f"Da dat START = {start_f}"
                    flash_timer = time.time() + 1.5

                elif key in [ord(']'), ord('f'), ord('F')]:  # ĐẶT END FRAME
                    end_f = curr_frame_idx
                    if end_f <= start_f:
                        start_f = max(0, end_f - 10)
                    flash_message = f"Da dat END = {end_f}"
                    flash_timer = time.time() + 1.5

                elif key in [ord('p'), ord('P')]:  # PREVIEW LOOP
                    is_preview_loop = not is_preview_loop
                    is_playing = False
                    curr_frame_idx = start_f

                elif key in [ord('y'), ord('Y'), 13]:  # LƯU MẪU (Accept)
                    is_playing = False
                    is_preview_loop = False
                    success = self.save_sample_to_json(
                        sign=sign_name,
                        video_filename=video_fname,
                        data=data,
                        start_frame=start_f,
                        end_frame=end_f
                    )
                    if success:
                        flash_message = "DA LUU MAU THANH CONG!"
                        flash_timer = time.time() + 2.0
                        # Tự động nhảy sang video tiếp theo sau khi lưu
                        time.sleep(0.5)
                        curr_vid_idx += 1
                        break

                elif key in [ord('n'), ord('N')]:  # BỎ QUA VIDEO NÀY
                    print(f"[-] Bỏ qua video {video_fname}")
                    curr_vid_idx += 1
                    break

                elif key in [ord('b'), ord('B')]:  # QUAY LẠI VIDEO TRƯỚC
                    curr_vid_idx = max(0, curr_vid_idx - 1)
                    break

        print("\n" + "=" * 70)
        print(f"✓ ĐÃ HOÀN TẤT DUYỆT TẤT CẢ VIDEO TRONG DANH SÁCH!")
        print(f"  Các mẫu chuẩn đã được lưu tại: '{self.output_json}'.")
        print("=" * 70)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ASL Citizen Video Inspector & Trimmer")
    parser.add_argument("--input-dir", type=str, default="ASL_Citizen_videos", help="Thư mục chứa video ASL Citizen")
    parser.add_argument("--output-json", type=str, default="asl_citizen_templates.json", help="File JSON lưu template trích xuất")
    parser.add_argument("--sign", type=str, default=None, help="Chỉ định từ cụ thể cần duyệt (ví dụ: --sign BYE)")
    args = parser.parse_args()

    app = ASLCitizenTrimmer(
        input_dir=args.input_dir,
        output_json=args.output_json,
        target_sign=args.sign
    )
    app.run()


if __name__ == "__main__":
    main()
