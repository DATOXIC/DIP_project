# -*- coding: utf-8 -*-
"""
pipeline.py — LUỒNG XỬ LÝ CHÍNH CỦA DỰ ÁN (bản rút gọn, đọc từ trên xuống là hiểu)
====================================================================================
Nhận diện ngôn ngữ ký hiệu ASL/VSL theo thời gian thực, chỉ dùng CPU và webcam thường.

    khung hình ──► 1 PERCEIVE ──► 2 SEGMENT ──► 3 SPOT ──► 4 FEATURIZE ──► 5 MATCH ──► 6 VERIFY ──► 7 COMMIT ──► 8 COMPOSE
                   MediaPipe      máy trạng thái  CNN 1D     đặc trưng 24D   DTW        dáng tay     khóa 1 lần   ngữ pháp
                   landmark       theo vận tốc    "đang ký"  + resample 32   Sakoe-Chiba landmark                 gloss -> câu

Mỗi bước là MỘT hàm ngắn bên dưới, kèm phương pháp áp dụng. Chi tiết nằm trong các module được gọi.

  1 PERCEIVE   Xử lý ảnh + thị giác: MediaPipe HandLandmarker (21 điểm 3D) + dò mặt có cache (mốc cơ thể);
               lọc 1€ làm mượt cổ tay; chuẩn hóa theo độ dài lòng bàn tay (palm_len) để bất biến khoảng cách camera;
               5 góc gập ngón + pháp tuyến lòng bàn tay làm đặc trưng dáng tay không phụ thuộc màu da / ánh sáng.
  2 SEGMENT    KineticPauseSegmenter: v_norm = |Δ cổ tay| / palm_len; máy trạng thái IDLE -> RECORDING -> PENDING_COMMIT;
               cổng thời lượng [0,4 s, 2,5 s]; loại chuyển động chuyển tiếp giữa hai từ; gạt trái = xóa từ.
  3 SPOT       Spotter (CNN 1D nhỏ, cửa sổ 16 khung): xác suất "đang ký" cho từng khung, cổng trễ kép mở 0,5 / đóng 0,35.
  4 FEATURIZE  Quỹ đạo -> vector 24D / khung (vị trí, hướng, góc ngón, pháp tuyến, mốc mặt), nội suy về K = 32 khung.
  5 MATCH      GoldenTriadDTWMatcher: DTW băng Sakoe-Chiba, khoảng cách có trọng số 3 nhóm đặc trưng, so với bank mẫu của người
               ký. Ngưỡng theo độ rộng nội lớp (global_k = 1,9) -> ACCEPTED / UNCERTAIN / REJECTED (từ chối cử chỉ lạ).
  6 VERIFY     HandshapeGate: bộ phân loại hình tay từ landmark 3D chuẩn hóa; chỉ giải quyết các ca DTW chưa chắc.
  7 COMMIT     OneShotLock: mỗi đoạn "đang ký" chốt tối đa một từ; chặn đọc lặp và cử chỉ vu vơ ngoài đoạn ký.
  8 COMPOSE    sentence_composer: chuỗi gloss ASL -> câu tự nhiên song ngữ (ví dụ ME LIKE YOU -> "I like you.").

Đây là bản ĐỌC HIỂU của vòng lặp sản xuất trong predict_sentence_webcam.py: dùng cùng các thành phần và tham số CONFIG,
nhưng bỏ các phần giao diện, giọng nói, dạy từ, đánh vần, chốt sớm và cắt dẫn đầu. Muốn chạy đầy đủ dùng `py dip.py`.

Cách chạy:   py pipeline.py --video continuous_data/Dat/<phiên>/session.mp4      (video thu từ collect_continuous)
             py pipeline.py --cam 0                                              (webcam, q để thoát)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from dynamic_dtw_engine import GoldenTriadDTWMatcher, is_movement_epenthesis, is_stroke_duration_valid
from dynamic_features import CachedFaceDetector, build_segment_features, compute_palm_len, extend_still_tail, records_to_gesture_frames
from handshape_gate import HandshapeGateRuntime
from pipeline_common import ParallelHandDetector, make_segmenter, make_wrist_smoother
from predict_sentence_webcam import (CONFIG, _segment_records, compute_joint_flexion_angles, compute_palm_normal_vector,
                                     get_asl_phrase)
from sentence_composer import compose
from spotter_gate import OneShotLock, make_spotter

MAX_SENTENCE_WORDS = 8        # đủ từ thì tự chốt câu
SENTENCE_IDLE_SEC = 1.8       # hạ tay lâu hơn thì chốt câu
REPEAT_SEC = 0.8              # cùng một từ trong khoảng này là đọc lặp


@dataclass
class Hand:
    """Kết quả bước 1: mọi thứ suy ra từ MỘT khung có tay."""
    landmarks: Any
    handedness: str
    wrist: Tuple[float, float]       # đã lọc 1€, đơn vị pixel
    palm_len: float
    size: Tuple[int, int]            # (rộng, cao) khung hình


class SignPipeline:
    def __init__(self, bank: Optional[str] = None, spotter_model: str = "spotter_model.pth") -> None:
        self.detector = ParallelHandDetector(workers=1)
        self.face = CachedFaceDetector(detect_every_n_frames=CONFIG.get("face_detect_every_n_frames", 8)) if CONFIG["w_b"] > 0 else None
        self.smoother = make_wrist_smoother()
        self.segmenter = make_segmenter(CONFIG, CONFIG["fast_chain"])
        self.matcher = _build_matcher(bank or CONFIG["templates_path"])
        self.spotter = make_spotter("dl", model_path=spotter_model) if os.path.exists(spotter_model) else None
        self.lock = OneShotLock(strict=True) if self.spotter else None
        gate = CONFIG["handshape_gate"]
        self.gate = HandshapeGateRuntime(gate["model"], gate["table"]) if os.path.exists(gate["model"]) else None
        self.seq = 0
        self.v_norm = 0.0
        self.hist: Dict[str, List[Any]] = {"angles": [], "normals": [], "ext": []}
        self.keys: List[str] = []            # câu hiện tại dưới dạng khóa từ vựng (gloss)
        self.sentences: List[str] = []       # các câu đã chốt
        self.finalized = False
        self.last_commit: Tuple[Optional[str], float] = (None, -1e9)
        self.last_seen = 0.0

    # ------------------------------------------------------------------ vòng xử lý một khung
    def step(self, frame: np.ndarray, t: float) -> Optional[Dict[str, Any]]:
        hand = self.perceive(frame, t)                                   # 1
        if hand is None:
            return self.idle(t)
        stroke = self.segment(hand, t)                                   # 2
        self.spot(hand, t)                                               # 3 (chạy song song với 2, chỉ cấp tín hiệu cho 7)
        if stroke is None:
            return None
        records = self.featurize_records(stroke)
        traj = self.featurize(records) if records else None              # 4
        if traj is None:
            return None
        result = self.match(traj, hand)                                  # 5
        result = self.verify(result, records, hand)                      # 6
        return self.commit(result, stroke, t)                            # 7

    # ------------------------------------------------------------------ 1. PERCEIVE
    def perceive(self, frame: np.ndarray, t: float) -> Optional[Hand]:
        """Ảnh -> 21 landmark 3D (MediaPipe) -> cổ tay làm mượt, palm_len, góc ngón, pháp tuyến, mốc mặt."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.detector.submit(self.seq, rgb, {})
        self.seq += 1
        (_, _, det), = self.detector.poll(block_oldest=True)
        if not det.hand_landmarks:
            return None
        lm = det.hand_landmarks[0]
        handed = det.handedness[0][0].category_name if det.handedness else "Right"
        lm3d = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
        face = self.face.detect(rgb, out_size=(w, h)) if self.face else None
        self.hist["angles"].append(compute_joint_flexion_angles(lm3d))
        self.hist["normals"].append(compute_palm_normal_vector(lm3d, handedness=handed))
        self.hist["ext"].append((lm3d, face))
        for k in self.hist:
            del self.hist[k][:-120]
        self.last_seen = t
        return Hand(lm, handed, self.smoother.update(lm[0].x * w, lm[0].y * h, timestamp=t),
                    compute_palm_len(lm[0], lm[5], lm[9], lm[17], w, h), (w, h))

    # ------------------------------------------------------------------ 2. SEGMENT
    def segment(self, hand: Hand, t: float) -> Optional[List[Any]]:
        """Vận tốc chuẩn hóa -> máy trạng thái động học. Trả về cử chỉ (danh sách khung) khi vừa kết thúc."""
        _, stroke, is_swipe, self.v_norm = self.segmenter.update(
            wrist_x=hand.wrist[0], wrist_y=hand.wrist[1], palm_len=hand.palm_len, hand_shape="Unknown", timestamp=t)
        if is_swipe and self.keys:                                       # gạt trái = xóa từ cuối
            self.keys.pop()
        return stroke

    # ------------------------------------------------------------------ 3. SPOT
    def spot(self, hand: Optional[Hand], t: float) -> None:
        """CNN 1D trên cửa sổ 16 khung landmark -> p(đang ký) -> cổng trễ kép; OneShotLock ghi lại các đoạn 'đang ký'."""
        if self.spotter is None:
            return
        if hand is None:
            self.lock.observe(t, self.spotter.update(t, False, 0.0))
            return
        gate_open = self.spotter.update(t, True, float(self.v_norm), hand.landmarks, hand.size)
        self.lock.observe(t, gate_open, getattr(self.spotter, "last_prob", None))

    # ------------------------------------------------------------------ 4. FEATURIZE
    def featurize_records(self, stroke: List[Any]) -> Optional[List[Dict[str, Any]]]:
        """Ghép khung cử chỉ với góc / pháp tuyến / mốc mặt; kéo đuôi yên như lúc thu mẫu; cổng thời lượng + lọc chuyển tiếp."""
        records = _segment_records(stroke, self.hist["angles"], self.hist["normals"], self.hist["ext"])
        for k in self.hist:
            self.hist[k].clear()
        tail = CONFIG["fast_chain"].get("tail_pad_sec")
        if tail:
            records, _ = extend_still_tail(records, float(tail), self.segmenter.v_stop_ratio, replay_jitter=True)
        if not is_stroke_duration_valid(records[-1]["t"] - records[0]["t"]):
            return None
        if is_movement_epenthesis(records_to_gesture_frames(records)):
            return None
        return records

    def featurize(self, records: List[Dict[str, Any]]) -> Optional[Tuple[np.ndarray, Any, bool]]:
        """Bản ghi -> quỹ đạo 24D nội suy K = 32 khung (đặc trưng dùng chung với lúc thu mẫu)."""
        traj, shapes, body_ok, _ = build_segment_features(records)
        return None if traj is None else (traj, shapes, body_ok)

    # ------------------------------------------------------------------ 5. MATCH
    def match(self, features: Tuple[np.ndarray, Any, bool], hand: Hand) -> Any:
        """DTW với từng mẫu trong bank -> từ gần nhất + trạng thái ACCEPTED / UNCERTAIN / REJECTED."""
        traj, shapes, body_ok = features
        return self.matcher.match(traj, shapes, handedness=hand.handedness, query_has_body=body_ok)

    # ------------------------------------------------------------------ 6. VERIFY
    def verify(self, result: Any, records: List[Dict[str, Any]], hand: Hand) -> Any:
        """Hình tay (landmark) chỉ xét lại khi DTW chưa chắc; không bao giờ lật một quyết định DTW đã tự tin."""
        return self.gate.apply(result, records, hand.handedness) if self.gate else result

    # ------------------------------------------------------------------ 7. COMMIT
    def commit(self, result: Any, stroke: List[Any], t: float) -> Dict[str, Any]:
        """Chỉ ACCEPTED mới vào câu; OneShotLock chặn cử chỉ ngoài đoạn ký và đọc lặp."""
        event = {"t": round(t, 2), "status": result.status, "word": result.top_word, "r_ratio": round(result.r_ratio, 2)}
        if result.status != "ACCEPTED" or not result.top_word:
            return event
        word = result.top_word
        if word == self.last_commit[0] and t - self.last_commit[1] < REPEAT_SEC:
            return {**event, "blocked": "repeat"}
        if self.lock is not None:
            allowed, reason = self.lock.allow_commit(float(stroke[0].timestamp), word)
            if not allowed:
                return {**event, "blocked": reason}
        if self.finalized:                                               # từ mới sau một câu đã chốt -> mở câu mới
            self.keys, self.finalized = [], False
        self.keys.append(word)
        self.last_commit = (word, t)
        if len(self.keys) >= MAX_SENTENCE_WORDS:
            self.finish_sentence()
        return {**event, "committed": True}

    # ------------------------------------------------------------------ 8. COMPOSE
    def finish_sentence(self) -> Optional[str]:
        """Chuỗi gloss -> câu tự nhiên (luật ngữ pháp SPEC 14); không có luật thì ghép tên hiển thị."""
        if not self.keys or self.finalized:
            return None
        text = compose(self.keys, "en", phrase_fn=lambda k, lang: get_asl_phrase(k, lang, default=k)).text
        self.sentences.append(text)
        self.finalized = True
        return text

    def idle(self, t: float) -> Optional[Dict[str, Any]]:
        """Không có tay: báo cho spotter, và hạ tay đủ lâu thì chốt câu."""
        self.spot(None, t)
        if self.keys and not self.finalized and t - self.last_seen > SENTENCE_IDLE_SEC:
            return {"t": round(t, 2), "sentence": self.finish_sentence()}
        return None

    def close(self) -> None:
        self.finish_sentence()
        self.detector.close()


def _build_matcher(bank: str) -> GoldenTriadDTWMatcher:
    """DTW 24D với tham số từ CONFIG (global_k = 1,9, ngưỡng theo độ rộng chung, trọng số toàn cục)."""
    return GoldenTriadDTWMatcher(
        templates_path=bank, sign_weights_path=CONFIG["sign_weights_path"] or None,
        w_p=CONFIG["w_wrist_pos"], w_a=CONFIG["w_finger_angles"], w_m=CONFIG["w_wrist_dir"], w_n=CONFIG["w_palm_normal"],
        lambda_shape=CONFIG["lambda_shape"], margin_threshold=CONFIG["margin_threshold"],
        uncertainty_ratio=CONFIG["uncertainty_ratio"], threshold_override=None, w_b=CONFIG["w_b"],
        threshold_mode=CONFIG["threshold_mode"], threshold_k=CONFIG["threshold_k"],
        uncertainty_k=CONFIG["uncertainty_k"], width_blend=CONFIG["width_blend"])


def run(video: Optional[str], cam: int, bank: Optional[str]) -> List[str]:
    pipe = SignPipeline(bank)
    cap = cv2.VideoCapture(video if video else cam)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if video is None:
                frame = cv2.flip(frame, 1)                               # webcam trực tiếp luôn lật gương (khớp dữ liệu huấn luyện)
            t = n / fps if video else cv2.getTickCount() / cv2.getTickFrequency()
            n += 1
            event = pipe.step(frame, t)
            if event and len(event) > 3:
                print(event)
            if video is None:
                cv2.putText(frame, " ".join(pipe.keys[-6:]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipe.close()
        cap.release()
        cv2.destroyAllWindows()
    return pipe.sentences


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Luồng xử lý chính (bản rút gọn dễ đọc).")
    ap.add_argument("--video", default=None, help="Video phiên thu (đã lật gương sẵn).")
    ap.add_argument("--cam", type=int, default=0, help="Chỉ số webcam khi không có --video.")
    ap.add_argument("--bank", default=None, help="Bank mẫu (mặc định CONFIG['templates_path']).")
    args = ap.parse_args()
    for s in run(args.video, args.cam, args.bank):
        print("CÂU:", s)
