"""
HỆ THỐNG PIPELINE DIP v3 CHO PHÂN TÍCH THỦ NGỮ (VSL)
========================================================
Kiến trúc chuẩn:
  1. Phân đoạn thích nghi: Histogram Backprojection + Otsu
  2. Chuẩn hóa hình học:   Image Moments (Principal Axis → 0°)
  3. Trích đặc trưng:       Tensor (128, 128, 3) = CLAHE + AutoCanny + Sobel
  4. Motion Tracking:       deque(maxlen=16) → Vector 32D cho cử chỉ ĐỘNG  [MỚI v3]

Lịch sử phiên bản:
  v1 — Baseline: HSV threshold cứng, không chuẩn hóa, 1 kênh
  v2 — [A] White Balance  [B] Online Hist Update  [C] Relative Area
       [D] Sat Sync S≥35  [E] Auto-Canny          [F] Moments Guard
       [G] Blur→CLAHE     [H] Solidity Check
  v3 — [I] LabAdaptiveHandSegmenter: CIELab (a*,b*) + Chroma Gate
            Kháng ám sắc tốt hơn, kênh L tách độc lập khỏi màu sắc
       [J] MotionTracker: deque(maxlen=16) — Vector 32D cho cử chỉ ĐỘNG
       [K] normalize_hand_orientation() trả thêm centroid (cx, cy)
       [L] full_preprocessing_pipeline() nhận motion_tracker tùy chọn
            — Backward compatible: vẫn trả tensor (128,128,3) hoặc None

Hai segmenter có cùng API, có thể thay thế lẫn nhau:
  AdaptiveHandSegmenter     — HSV, nhẹ, đủ tốt trong điều kiện ổn định
  LabAdaptiveHandSegmenter  — CIELab, mạnh hơn với ánh sáng phức tạp [KHUYẾN NGHỊ]
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional, List, Dict


# =============================================================================
# SECTION A: HELPER FUNCTIONS
# =============================================================================

def apply_gray_world_white_balance(bgr):
    """
    [A] Cân bằng trắng Gray-World: triệt tiêu ám sắc đèn vàng/xanh.
    Giả định mean(R) = mean(G) = mean(B) = mean_gray → scale từng kênh.
    Áp dụng trước CẢ HSV lẫn Lab để chuẩn hóa màu đầu vào nhất quán.
    """
    b, g, r  = cv2.split(bgr.astype(np.float32))
    mean_b   = np.mean(b) + 1e-6
    mean_g   = np.mean(g) + 1e-6
    mean_r   = np.mean(r) + 1e-6
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    return cv2.merge([
        np.clip(b * (mean_gray / mean_b), 0, 255).astype(np.uint8),
        np.clip(g * (mean_gray / mean_g), 0, 255).astype(np.uint8),
        np.clip(r * (mean_gray / mean_r), 0, 255).astype(np.uint8),
    ])


def auto_canny(gray, sigma=0.33):
    """
    [E] Tự động chọn ngưỡng Canny dựa theo median của VÙNG TAY (pixel > 0).
    Thích nghi với mọi mức độ sáng, vượt trội hơn cách tính median toàn ảnh.
      - Ảnh tối  → median thấp → lo/hi thấp → bắt được cạnh mờ
      - Ảnh sáng → median cao  → lo/hi cao  → không sinh cạnh giả
    """
    hand_pixels = gray[gray > 0]
    v  = float(np.median(hand_pixels)) if len(hand_pixels) > 100 else 127.0
    lo = int(max(10,  (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    hi = max(hi, lo * 2)       # Đảm bảo tỉ lệ lo : hi ≥ 1 : 2
    return cv2.Canny(gray, lo, hi)


# =============================================================================
# SECTION B: SEGMENTERS
# =============================================================================

class AdaptiveHandSegmenter:
    """
    Phân đoạn bàn tay thích nghi dựa trên HSV Histogram Backprojection (v2).
    Giữ nguyên cho backward compatibility với collect_webcam.py, dataset_builder.py.
    Khuyến nghị dùng LabAdaptiveHandSegmenter cho điều kiện ánh sáng phức tạp.

    API:
      segmenter.calibrate(frame)              → Học histogram màu da
      segmenter.draw_calibration_progress(frame) → Calibrate + vẽ UI
      segmenter.segment(frame)               → Trả mask nhị phân (ROI-sized)
      segmenter.reset_calibration()          → Bắt đầu lại
      segmenter.is_calibrated               → bool
      segmenter.min_hand_area               → float (5% diện tích ROI)
    """

    def __init__(self, calibration_roi=(80, 420, 320, 620), max_calibration_frames=30):
        self.top, self.bottom, self.left, self.right = calibration_roi
        self.max_calibration_frames = max_calibration_frames
        self.calibration_count = 0
        self.is_calibrated     = False
        self.hist              = None
        # [C] Ngưỡng diện tích tương đối: 5% diện tích ROI
        roi_area = (self.bottom - self.top) * (self.right - self.left)
        self.min_hand_area = roi_area * 0.05

    def reset_calibration(self):
        """Khởi động lại quá trình hiệu chuẩn."""
        self.calibration_count = 0
        self.is_calibrated     = False
        self.hist              = None

    def calibrate(self, frame):
        """
        Học Histogram 2D (H, S) từ ROI.
        [A] White Balance trước khi học.  [D] Skin Gate: S ≥ 35.
        """
        roi     = frame[self.top:self.bottom, self.left:self.right]
        roi_wb  = apply_gray_world_white_balance(roi)
        hsv_roi = cv2.cvtColor(roi_wb, cv2.COLOR_BGR2HSV)

        # [D] Skin Prior Gate — S ≥ 35 (đồng bộ với segment)
        lower1 = np.array([0,   35, 40], dtype=np.uint8)
        upper1 = np.array([25, 255, 255], dtype=np.uint8)
        lower2 = np.array([170,  35, 40], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        skin_gate = cv2.bitwise_or(
            cv2.inRange(hsv_roi, lower1, upper1),
            cv2.inRange(hsv_roi, lower2, upper2)
        )

        # Center Focus: lấy mẫu ở 70% tâm ROI
        h_r, w_r = roi.shape[:2]
        center   = np.zeros((h_r, w_r), dtype=np.uint8)
        py, px   = int(h_r * 0.15), int(w_r * 0.15)
        center[py:h_r - py, px:w_r - px] = 255
        cal_mask = cv2.bitwise_and(skin_gate, center)

        hist_cur = cv2.calcHist([hsv_roi], [0, 1], cal_mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_cur, hist_cur, 0, 255, cv2.NORM_MINMAX)

        self.hist = hist_cur if self.hist is None else (0.85 * self.hist + 0.15 * hist_cur)
        self.calibration_count += 1
        if self.calibration_count >= self.max_calibration_frames:
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
            self.is_calibrated = True

    def draw_calibration_progress(self, frame):
        """Thực hiện 1 bước calibration và vẽ tiến trình UI lên frame."""
        self.calibrate(frame)
        p = int((self.calibration_count / self.max_calibration_frames) * 100)
        cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (0, 255, 255), 2)
        h_r, w_r = self.bottom - self.top, self.right - self.left
        cv2.rectangle(frame,
                      (self.left + int(w_r * 0.15), self.top  + int(h_r * 0.15)),
                      (self.right - int(w_r * 0.15), self.bottom - int(h_r * 0.15)),
                      (255, 255, 0), 1)
        cv2.putText(frame, f"[HSV] HIEU CHUAN: {p}%",
                    (self.left - 30, self.top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        bar_w = int((self.right - self.left) * p / 100)
        cv2.rectangle(frame, (self.left, self.bottom + 10),
                      (self.left + bar_w, self.bottom + 25), (0, 255, 0), -1)

    def segment(self, frame):
        """
        Backprojection → Saturation Gate (S≥35) → Otsu → Morphology → Online Update.
        Trả về mask nhị phân kích thước ROI.
        """
        if not self.is_calibrated or self.hist is None:
            return np.zeros((self.bottom - self.top, self.right - self.left), dtype=np.uint8)

        roi     = frame[self.top:self.bottom, self.left:self.right]
        roi_wb  = apply_gray_world_white_balance(roi)
        hsv_roi = cv2.cvtColor(roi_wb, cv2.COLOR_BGR2HSV)

        backproj = cv2.calcBackProject([hsv_roi], [0, 1], self.hist, [0, 180, 0, 256], scale=1)
        # [D] Saturation Gate
        sat_gate = (hsv_roi[:, :, 1] >= 35).astype(np.uint8) * 255
        backproj = cv2.bitwise_and(backproj, sat_gate)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(backproj, -1, disc, backproj)
        _, mask = cv2.threshold(backproj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Lấp đầy lỗ thủng bóng đổ bằng cv2.fillPoly
        cnts, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None and len(cnts) > 0:
            for i in range(len(cnts)):
                if hier[0][i][3] != -1:
                    cv2.fillPoly(mask, [cnts[i]], 255)

        # Tắt cập nhật trực tuyến để tránh trôi dạt màu nền
        # self._online_histogram_update(hsv_roi, mask)
        return mask

    def _online_histogram_update(self, hsv_roi, confirmed_mask, alpha=0.005):
        """[B] EMA update histogram từ mask đã xác nhận. Bỏ qua nếu < 1000 px."""
        if cv2.countNonZero(confirmed_mask) < 1000:
            return
        hist_new = cv2.calcHist([hsv_roi], [0, 1], confirmed_mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_new, hist_new, 0, 255, cv2.NORM_MINMAX)
        self.hist = (1.0 - alpha) * self.hist + alpha * hist_new


# -----------------------------------------------------------------------------

class LabAdaptiveHandSegmenter:
    """
    [I] Phân đoạn bàn tay nâng cao dùng không gian màu CIELab (a*, b*) (v3).

    Ưu điểm so với HSV:
      ┌─────────────────────────────────────────────────────────────────────┐
      │ Vấn đề HSV                    │ Giải pháp Lab                      │
      ├───────────────────────────────┼────────────────────────────────────┤
      │ Hue bị lệch khi đèn thay đổi │ Kênh L tách hoàn toàn khỏi (a,b)  │
      │ Saturation Gate thô (tuyến   │ Chroma Gate chính xác hơn (Euclid) │
      │   tính theo từng kênh riêng) │   sqrt((a-128)²+(b-128)²) ≥ 12     │
      │ Histogram [180,256] = 46080  │ Histogram [64,64]  = 4096 bins     │
      │   bins — dư thừa             │   — đặc hơn, phân biệt tốt hơn     │
      └─────────────────────────────────────────────────────────────────────┘

    Quy ước OpenCV 8-bit Lab:
      L_opencv = L_true * 2.55          → [0, 255]
      a_opencv = a_true + 128           → tâm vô sắc tại a = 128
      b_opencv = b_true + 128           → tâm vô sắc tại b = 128
      Màu da người: a ≈ [130,175], b ≈ [130,188]  (góc phần tư Đỏ-Vàng)
      Chroma = sqrt((a-128)² + (b-128)²) ≥ 12 để loại màu trắng/xám/đen

    API: Giống hệt AdaptiveHandSegmenter — thay thế trực tiếp, không cần sửa code gọi.
    """

    # Ngưỡng màu da trên Lab 8-bit (empirically validated)
    _SKIN_A_LOW,  _SKIN_A_HIGH = 130, 175
    _SKIN_B_LOW,  _SKIN_B_HIGH = 124, 188
    _CHROMA_MIN                = 10.0

    def __init__(self, calibration_roi=(80, 420, 320, 620), max_calibration_frames=30):
        self.top, self.bottom, self.left, self.right = calibration_roi
        self.max_calibration_frames = max_calibration_frames
        self.calibration_count = 0
        self.is_calibrated     = False
        self.hist              = None
        roi_area = (self.bottom - self.top) * (self.right - self.left)
        self.min_hand_area = roi_area * 0.05
        # [Trừ nền DIP] Mẫu ảnh nền tĩnh của ROI để loại bỏ đồ vật/tủ gỗ trùng màu
        self.bg_roi_blur    = None
        self.is_bg_captured = False

    def capture_background(self, frame):
        """[DIP Pure] Chụp lại ảnh nền tĩnh thô của khung ROI (khi chưa giơ tay vào)."""
        roi_raw = frame[self.top:self.bottom, self.left:self.right]
        # Lưu ảnh nền BGR thô làm mịn nhẹ (không qua White Balance để không bị bàn tay làm đổi màu sau này)
        self.bg_roi_blur = cv2.GaussianBlur(roi_raw, (5, 5), 0)
        self.is_bg_captured = True
        print("-> [DIP Tru nen]: Da chup thanh cong mau nen tinh (Raw BGR) cua ROI!")

    def clear_background(self):
        """Xóa ảnh nền tham chiếu."""
        self.bg_roi_blur = None
        self.is_bg_captured = False

    def reset_calibration(self):
        """Khởi động lại quá trình hiệu chuẩn."""
        self.calibration_count = 0
        self.is_calibrated     = False
        self.hist              = None

    def _make_skin_gate(self, lab_roi):
        """
        Tạo mặt nạ da người trong Lab 8-bit bằng 4 lớp lọc:
          1. Ngưỡng hình chữ nhật (a, b): bắt đại đa số pixel da người.
          2. Chroma Gate (Euclid từ tâm 128,128): lọc màu vô sắc chính xác.
          3. [Apple ISP] Highlight Recovery: nới lỏng Chroma cho vùng cháy sáng L > 200.
          4. [Shadow Recovery]: cứu các pixel bị bóng đổ trong lòng bàn tay (chữ C, O).
        """
        lower    = np.array([0, self._SKIN_A_LOW,  self._SKIN_B_LOW],  dtype=np.uint8)
        upper    = np.array([255, self._SKIN_A_HIGH, self._SKIN_B_HIGH], dtype=np.uint8)
        rect_gate = cv2.inRange(lab_roi, lower, upper)

        a_f = lab_roi[:, :, 1].astype(np.float32) - 128.0
        b_f = lab_roi[:, :, 2].astype(np.float32) - 128.0
        chroma_dist = np.sqrt(a_f**2 + b_f**2)
        chroma_gate = (chroma_dist >= self._CHROMA_MIN).astype(np.uint8) * 255

        standard_skin = cv2.bitwise_and(rect_gate, chroma_gate)

        L_channel = lab_roi[:, :, 0]
        # [Apple ISP] Highlight Recovery: Cứu các pixel bị đèn trần dội cháy sáng (L > 200)
        hl_gate = ((L_channel > 200) & (chroma_dist >= 5.0) &
                   (lab_roi[:, :, 1] >= 125) & (lab_roi[:, :, 1] <= 180) &
                   (lab_roi[:, :, 2] >= 124) & (lab_roi[:, :, 2] <= 190)).astype(np.uint8) * 255

        # [Shadow Recovery]: Cứu các pixel bóng đổ trong lòng bàn tay (chữ C, ngón tay khum)
        sh_gate = ((L_channel >= 25) & (L_channel <= 135) & (chroma_dist >= 5.0) &
                   (lab_roi[:, :, 1] >= 125) & (lab_roi[:, :, 1] <= 175) &
                   (lab_roi[:, :, 2] >= 122) & (lab_roi[:, :, 2] <= 180)).astype(np.uint8) * 255

        recovered = cv2.bitwise_or(standard_skin, hl_gate)
        return cv2.bitwise_or(recovered, sh_gate)

    def calibrate(self, frame):
        """
        Học Histogram 2D trên kênh (a, b) của Lab.
        [A] White Balance → Lab → [I] 64×64 bins, range [0,256] cho (a,b).
        """
        roi     = frame[self.top:self.bottom, self.left:self.right]
        roi_wb  = apply_gray_world_white_balance(roi)
        lab_roi = cv2.cvtColor(roi_wb, cv2.COLOR_BGR2Lab)

        skin_gate = self._make_skin_gate(lab_roi)

        h_r, w_r = roi.shape[:2]
        center   = np.zeros((h_r, w_r), dtype=np.uint8)
        py, px   = int(h_r * 0.15), int(w_r * 0.15)
        center[py:h_r - py, px:w_r - px] = 255
        cal_mask = cv2.bitwise_and(skin_gate, center)

        # [I] Histogram 2D: kênh (a,b) — [1,2], 64×64 bins
        hist_cur = cv2.calcHist([lab_roi], [1, 2], cal_mask, [64, 64], [0, 256, 0, 256])
        cv2.normalize(hist_cur, hist_cur, 0, 255, cv2.NORM_MINMAX)

        self.hist = hist_cur if self.hist is None else (0.85 * self.hist + 0.15 * hist_cur)
        self.calibration_count += 1
        if self.calibration_count >= self.max_calibration_frames:
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
            self.is_calibrated = True

    def draw_calibration_progress(self, frame):
        """Thực hiện 1 bước calibration và vẽ tiến trình UI (màu cam để phân biệt với HSV)."""
        self.calibrate(frame)
        p = int((self.calibration_count / self.max_calibration_frames) * 100)
        cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (0, 165, 255), 2)
        h_r, w_r = self.bottom - self.top, self.right - self.left
        cv2.rectangle(frame,
                      (self.left + int(w_r * 0.15), self.top  + int(h_r * 0.15)),
                      (self.right - int(w_r * 0.15), self.bottom - int(h_r * 0.15)),
                      (0, 140, 255), 1)
        cv2.putText(frame, f"[Lab] HIEU CHUAN: {p}%",
                    (self.left - 30, self.top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        bar_w = int((self.right - self.left) * p / 100)
        cv2.rectangle(frame, (self.left, self.bottom + 10),
                      (self.left + bar_w, self.bottom + 25), (0, 165, 255), -1)

    def segment(self, frame):
        """
        Backprojection (a,b) → Chroma Gate → Otsu → Morphology → Online Update.
        [A] White Balance → [I] Lab pipeline → [B] Online EMA update.
        Trả về mask nhị phân kích thước ROI.
        """
        if not self.is_calibrated or self.hist is None:
            return np.zeros((self.bottom - self.top, self.right - self.left), dtype=np.uint8)

        roi     = frame[self.top:self.bottom, self.left:self.right]
        roi_wb  = apply_gray_world_white_balance(roi)
        lab_roi = cv2.cvtColor(roi_wb, cv2.COLOR_BGR2Lab)

        # 1. Backprojection trên kênh màu da (a, b)
        backproj = cv2.calcBackProject([lab_roi], [1, 2], self.hist, [0, 256, 0, 256], scale=1)

        # 2. Lọc sơ bộ bằng Skin Gate (ngưỡng a*, b* và Chroma) để loại bỏ vật thể lệch màu da
        skin_gate = self._make_skin_gate(lab_roi)
        backproj  = cv2.bitwise_and(backproj, skin_gate)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(backproj, -1, disc, backproj)
        _, skin_mask = cv2.threshold(backproj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. [DIP Trừ Nền & Ưu tiên màu da chắc chắn - Skin Confidence Priority]
        if self.is_bg_captured and self.bg_roi_blur is not None:
            # 3a. Trừ nền trên ảnh BGR thô gốc (tránh bị Gray-World làm đổi màu nền khi có tay vào)
            roi_raw_blur = cv2.GaussianBlur(roi, (5, 5), 0)
            diff_bgr = cv2.absdiff(roi_raw_blur, self.bg_roi_blur)
            diff_metric = np.max(diff_bgr, axis=2)
            _, bg_diff_mask = cv2.threshold(diff_metric, 15, 255, cv2.THRESH_BINARY)
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bg_diff_mask = cv2.dilate(bg_diff_mask, dilate_k, iterations=1)

            # 3b. [Bảo vệ màu da chắc chắn - Skin Confidence Priority]
            # Các pixel có điểm xác suất màu da cao trong Lab (backproj >= 25) chắc chắn là da thật!
            # Tuyệt đối không cho phép trừ nền xóa nhầm vùng này (đặc biệt là lòng bàn tay chữ C).
            high_conf_skin = (backproj >= 25).astype(np.uint8) * 255

            # Logic mềm: Giữ lại nếu (Là vật thể mới) HOẶC (Có xác suất da cực cao)
            valid_foreground = cv2.bitwise_or(bg_diff_mask, high_conf_skin)
            mask = cv2.bitwise_and(skin_mask, valid_foreground)
        else:
            mask = skin_mask

        # 4. [Khắc phục 2]: Giảm close_k xuống 5x5, 1 iteration (chống hàn dính mép tủ vào tay)
        open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k,  iterations=1)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)

        # 5. [Lấp đầy lỗ thủng bóng đổ bằng cv2.fillPoly]
        # Tự động phát hiện các lỗ thủng khép kín do bóng đổ trong lòng bàn tay (chữ C, O, A...)
        # và dùng cv2.fillPoly để lấp đầy đặc ruột trắng tinh.
        cnts, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None and len(cnts) > 0:
            for i in range(len(cnts)):
                if hier[0][i][3] != -1:  # Là lỗ thủng nội tại bên trong bàn tay
                    cv2.fillPoly(mask, [cnts[i]], 255)

        # FloodFill DIP từ góc (0,0) để đảm bảo 100% không còn lỗ thủng bóng đổ li ti nào
        h_m, w_m = mask.shape[:2]
        flood_filled = mask.copy()
        flood_mask = np.zeros((h_m + 2, w_m + 2), np.uint8)
        cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood_filled)
        mask = cv2.bitwise_or(mask, holes)

        # 6. [Khắc phục 1]: TẮT hoàn toàn online_histogram_update để chống Model Drift nuốt màu tủ gỗ
        # self._online_histogram_update(lab_roi, mask)
        return mask

    def _online_histogram_update(self, lab_roi, confirmed_mask, alpha=0.005):
        """[B] EMA update histogram Lab từ mask đã xác nhận. Bỏ qua nếu < 1000 px."""
        if cv2.countNonZero(confirmed_mask) < 1000:
            return
        hist_new = cv2.calcHist([lab_roi], [1, 2], confirmed_mask, [64, 64], [0, 256, 0, 256])
        cv2.normalize(hist_new, hist_new, 0, 255, cv2.NORM_MINMAX)
        self.hist = (1.0 - alpha) * self.hist + alpha * hist_new


# =============================================================================
# SECTION C: MOTION TRACKER
# =============================================================================

class MotionTracker:
    """
    [J] Theo dõi quỹ đạo tâm bàn tay theo thời gian dùng deque(maxlen=N).

    Nguyên lý "Cửa sổ trượt thời gian" (Sliding Window):
      ┌─ Frame f₁  → centroid (cx₁, cy₁)  → deque: [(cx₁,cy₁)]
      ├─ Frame f₂  → centroid (cx₂, cy₂)  → deque: [(cx₁,cy₁),(cx₂,cy₂)]
      │  ...
      └─ Frame f₁₆ → centroid (cx₁₆,cy₁₆) → deque đầy (16 phần tử)
         Frame f₁₇ → cx₁ bị đẩy ra, cx₂ trở thành phần tử đầu

    Tại sao 16 frame?
      30 FPS webcam → 16/30 ≈ 0.53 giây
      Cử chỉ vẫy tay, gật đầu, lướt tay: 0.4–0.8 giây
      → Vừa đủ bắt 1 cử chỉ, vừa đủ ngắn để phản hồi tức thì.

    get_normalized_trajectory() → vector (32,) cho LSTM/MLP:
      1. Trừ điểm đầu tiên → bất biến dịch chuyển vị trí
      2. Chia max_disp     → bất biến khoảng cách camera ↔ tay
      3. Flatten (16,2) → (32,) → nạp thẳng vào mạng nơ-ron
    """

    def __init__(self, history_len=16, movement_threshold=8):
        """
        Args:
            history_len:        Số frame lưu lịch sử (mặc định 16 ≈ 0.53s ở 30fps).
            movement_threshold: Pixel/frame để coi là "đang chuyển động" (mặc định 8).
        """
        self.history_len        = history_len
        self.movement_threshold = movement_threshold
        self.history            = deque(maxlen=history_len)

    def update(self, cx, cy):
        """Thêm tọa độ centroid (trong tọa độ ROI) của frame hiện tại vào lịch sử."""
        self.history.append((int(cx), int(cy)))

    def reset(self):
        """Xóa toàn bộ lịch sử. Gọi khi bàn tay rời khỏi ROI."""
        self.history.clear()

    def get_normalized_trajectory(self):
        """
        Trả về vector đặc trưng phẳng (2 × history_len,) đã chuẩn hóa.
        Bất biến dịch chuyển vị trí + bất biến tỉ lệ khoảng cách.

        Returns:
            np.ndarray shape (32,) — nạp vào LSTM/MLP để nhận cử chỉ động
            None                   — nếu quá ít điểm (< 4 frame)
        """
        if len(self.history) < 4:
            return None

        pts = np.array(self.history, dtype=np.float32)   # (N, 2)
        pts -= pts[0]                                      # Bất biến vị trí

        max_disp = np.max(np.abs(pts))
        if max_disp > 1e-6:
            pts /= max_disp                                # Bất biến tỉ lệ

        # Padding zeros về đúng history_len nếu deque chưa đầy
        if len(pts) < self.history_len:
            pad = np.zeros((self.history_len - len(pts), 2), dtype=np.float32)
            pts = np.vstack([pts, pad])

        return pts.flatten()   # Shape: (32,) với history_len=16

    def is_moving(self):
        """
        Kiểm tra bàn tay có đang chuyển động đáng kể không.
        Tính tốc độ trung bình từ 4 frame cuối.
        """
        if len(self.history) < 4:
            return False
        recent    = np.array(list(self.history)[-4:], dtype=np.float32)
        avg_speed = np.mean(np.linalg.norm(np.diff(recent, axis=0), axis=1))
        return avg_speed > self.movement_threshold

    def get_gesture_direction(self):
        """
        Phát hiện hướng cử chỉ đơn giản (Rule-based) từ 8 frame cuối.
        Hữu ích để debug và feedback tức thì trước khi có LSTM.

        Returns:
            str: 'STILL' | 'LEFT' | 'RIGHT' | 'UP' | 'DOWN'
        """
        if len(self.history) < 6:
            return "STILL"
        recent = np.array(list(self.history)[-8:], dtype=np.float32)
        dx = recent[-1, 0] - recent[0, 0]
        dy = recent[-1, 1] - recent[0, 1]
        if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
            return "STILL"
        if abs(dx) >= abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        return "DOWN" if dy > 0 else "UP"

    def draw_trail(self, frame, offset_x=0, offset_y=0):
        """
        Vẽ vệt chuyển động (sao băng) lên frame BGR in-place.
        Gradient màu: Xanh đậm (cũ) → Vàng sáng (mới). Dày dần về cuối.

        Args:
            frame:    Frame BGR cần vẽ lên.
            offset_x: Dịch x để chuyển từ tọa độ ROI sang tọa độ frame đầy đủ.
            offset_y: Dịch y tương tự.
        """
        pts = list(self.history)
        n   = len(pts)
        for i in range(1, n):
            alpha     = i / n                                   # 0→1: cũ→mới
            thickness = max(1, int(alpha * 5))
            color     = (0, int(100 + 155 * alpha), 255)        # Xanh lơ→Vàng
            p1 = (pts[i - 1][0] + offset_x, pts[i - 1][1] + offset_y)
            p2 = (pts[i    ][0] + offset_x, pts[i    ][1] + offset_y)
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

        if n > 0:   # Chấm tròn tại vị trí hiện tại
            cx_now = pts[-1][0] + offset_x
            cy_now = pts[-1][1] + offset_y
            cv2.circle(frame, (cx_now, cy_now), 7, (0, 255, 255), -1)
            cv2.circle(frame, (cx_now, cy_now), 7, (0, 200, 200), 2)


# =============================================================================
# SECTION D: NORMALIZATION & FEATURE EXTRACTION
# =============================================================================

def normalize_hand_orientation(frame, mask, roi_coords=(80, 420, 320, 620),
                                target_size=(128, 128), min_hand_area=None):
    """
    Chuẩn hóa hình học bàn tay bằng Image Moments:
    - Tính tâm trọng khối centroid (cx, cy)
    - Tính góc trục chính θ = 0.5 * atan2(2μ₁₁, μ₂₀ − μ₀₂)
    - Xoay về 0° (thẳng đứng), crop, padding về target_size

    Cải tiến v2: [C] Relative area  [F] Elongation guard  [H] Solidity check
    Cải tiến v3: [K] Trả thêm centroid (cx, cy) trong tọa độ ROI

    Returns:
        (normalized_hand, rotated_mask, (cx, cy))  — thành công
        (None, None, None)                          — thất bại (không có/phát hiện tay)
    """
    top, bottom, left, right = roi_coords
    roi = frame[top:bottom, left:right]

    if min_hand_area is None:
        min_hand_area = (bottom - top) * (right - left) * 0.05

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    max_contour  = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(max_contour)

    # [C] Ngưỡng diện tích tương đối
    if contour_area < min_hand_area:
        return None, None, None

    # [H] Solidity check: bàn tay thực có Solidity ∈ [0.45, 0.97]
    hull_area = cv2.contourArea(cv2.convexHull(max_contour))
    if hull_area > 0:
        if not (0.45 <= contour_area / hull_area <= 0.97):
            return None, None, None

    # Hole Filling: hàn kín lòng bàn tay, giữ khe ngón tay
    filled_hand = np.zeros_like(mask)
    cv2.drawContours(filled_hand, [max_contour], -1, 255, thickness=cv2.FILLED)
    clean_mask = filled_hand

    # [Google MediaPipe] Palm Anchor & Wrist Trimming bằng Distance Transform
    # 1. Tìm tâm lòng bàn tay (điểm nội tiếp xa đường biên nhất)
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    _, max_dist, _, max_loc = cv2.minMaxLoc(dist_transform)
    palm_cx, palm_cy = max_loc
    palm_r = max_dist  # Bán kính vòng tròn nội tiếp lòng bàn tay

    # 2. Gọt sạch phần cẳng tay thừa ở đáy nếu dính cẳng tay dài
    # Về mặt giải phẫu, cổ tay nằm cách tâm lòng bàn tay khoảng 1.6 * palm_r về phía đáy
    h_roi, w_roi = roi.shape[:2]
    if palm_r > 18 and palm_cy > h_roi * 0.25:
        wrist_cut_y = int(palm_cy + 1.6 * palm_r)
        if wrist_cut_y < h_roi and wrist_cut_y > int(h_roi * 0.65):
            clean_mask[wrist_cut_y:, :] = 0  # Cắt sạch cẳng tay phía dưới cổ tay

    # Cập nhật lại contour sau khi gọt cổ tay
    trimmed_cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if trimmed_cnts:
        max_contour = max(trimmed_cnts, key=cv2.contourArea)

    # Image Moments
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return None, None, None

    cx   = int(M["m10"] / M["m00"])
    cy   = int(M["m01"] / M["m00"])
    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]

    # [F] Elongation Guard: nâng ngưỡng lên 0.20 để nắm đấm chữ A không bị xoay xiên
    total_var = mu20 + mu02 + 1e-6
    if abs(mu20 - mu02) / total_var < 0.20:
        rot_angle = 0.0
    else:
        angle_deg = np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))
        rot_angle = angle_deg - 90 if angle_deg > 0 else angle_deg + 90

    rot_matrix   = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    h_roi, w_roi = roi.shape[:2]
    rotated_roi  = cv2.warpAffine(roi,        rot_matrix, (w_roi, h_roi), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(clean_mask, rot_matrix, (w_roi, h_roi), flags=cv2.INTER_NEAREST)

    rot_contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not rot_contours:
        return None, None, None

    rx, ry, rw, rh = cv2.boundingRect(max(rot_contours, key=cv2.contourArea))
    pad = 10
    y1, y2 = max(0, ry - pad), min(h_roi, ry + rh + pad)
    x1, x2 = max(0, rx - pad), min(w_roi, rx + rw + pad)

    cropped_hand = rotated_roi[y1:y2, x1:x2]
    if cropped_hand.size == 0 or min(cropped_hand.shape[:2]) < 20:
        return None, None, None

    # Flip Detection: đầu ngón tay phải hướng lên (tức là nửa trên ít pixel hơn)
    cropped_mask = rotated_mask[y1:y2, x1:x2]
    mid_y        = cropped_mask.shape[0] // 2
    if cv2.countNonZero(cropped_mask[:mid_y, :]) > cv2.countNonZero(cropped_mask[mid_y:, :]) * 1.5:
        cropped_hand = cv2.rotate(cropped_hand, cv2.ROTATE_180)
        cropped_mask = cv2.rotate(cropped_mask, cv2.ROTATE_180)

    # Áp mặt nạ: xóa nền thành đen tuyền
    cropped_hand = cv2.bitwise_and(cropped_hand, cropped_hand, mask=cropped_mask)

    # Aspect Ratio Padding → target_size (không méo hình)
    ch_h, ch_w = cropped_hand.shape[:2]
    scale       = min(target_size[1] / ch_w, target_size[0] / ch_h)
    new_w, new_h = int(ch_w * scale), int(ch_h * scale)
    resized     = cv2.resize(cropped_hand, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out  = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    sy   = (target_size[0] - new_h) // 2
    sx   = (target_size[1] - new_w) // 2
    out[sy:sy + new_h, sx:sx + new_w] = resized

    return out, rotated_mask, (cx, cy)   # [K] Trả thêm centroid (ROI coords)


def extract_multi_channel_features(normalized_hand):
    """
    Trích xuất Tensor (128, 128, 3) từ bàn tay đã chuẩn hóa:
    - Kênh 0: [G] Blur(3×3) + CLAHE  — Cấu trúc & tương phản cục bộ
    - Kênh 1: [E] Auto-Canny          — Đường biên & hình dạng ngón tay
    - Kênh 2: Sobel Magnitude          — Chi tiết vi mô & nếp gấp
    """
    gray      = cv2.cvtColor(normalized_hand, cv2.COLOR_BGR2GRAY) \
                if len(normalized_hand.shape) == 3 else normalized_hand.copy()
    hand_mask = (gray > 0).astype(np.uint8)

    # Kênh 0: [G] Blur nhẹ trước CLAHE → giảm khuếch đại noise sensor
    blurred_clahe = cv2.GaussianBlur(gray, (3, 3), 0.5)
    clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result  = clahe.apply(blurred_clahe)
    channel0      = cv2.bitwise_and(clahe_result, clahe_result, mask=hand_mask)

    # Blur 5×5 cho Canny và Sobel
    blurred  = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Kênh 1: [E] Auto-Canny
    channel1 = auto_canny(blurred)

    # Kênh 2: Sobel Gradient Magnitude
    sobelx   = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely   = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    channel2 = cv2.normalize(np.sqrt(sobelx**2 + sobely**2),
                              None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return np.stack([channel0, channel1, channel2], axis=2)


def full_preprocessing_pipeline(frame, segmenter, motion_tracker=None, target_size=(128, 128)):
    """
    Pipeline hoàn chỉnh từ frame thô → tensor CNN.

    [L] Backward compatible: vẫn trả tensor (128,128,3) hoặc None.
    [J] Nếu motion_tracker được cung cấp:
        - Khi phát hiện tay: cập nhật tracker với centroid mới
        - Khi mất tay: reset tracker (xóa lịch sử)

    Args:
        frame:          Frame BGR từ webcam.
        segmenter:      AdaptiveHandSegmenter HOẶC LabAdaptiveHandSegmenter (đã calibrated).
        motion_tracker: MotionTracker (tùy chọn). Nếu None: bỏ qua motion tracking.
        target_size:    Kích thước output. Mặc định (128, 128).

    Returns:
        np.ndarray (128, 128, 3) — tensor sẵn sàng cho CNN
        None                     — không phát hiện bàn tay
    """
    if not segmenter.is_calibrated:
        return None

    mask = segmenter.segment(frame)

    normalized_hand, _, centroid = normalize_hand_orientation(
        frame, mask,
        roi_coords=(segmenter.top, segmenter.bottom, segmenter.left, segmenter.right),
        target_size=target_size,
        min_hand_area=segmenter.min_hand_area
    )

    if normalized_hand is None:
        if motion_tracker is not None:
            motion_tracker.reset()   # [J] Mất tay → xóa lịch sử chuyển động
        return None

    # [J] Cập nhật tracker với centroid bàn tay vừa tìm được (tọa độ ROI)
    if motion_tracker is not None and centroid is not None:
        motion_tracker.update(centroid[0], centroid[1])

    return extract_multi_channel_features(normalized_hand)


# =============================================================================
# SECTION E: DEMO REALTIME (v3)
# =============================================================================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở Webcam!")
        exit()

    # -------------------------------------------------------------------------
    # Khởi tạo: dùng LabAdaptiveHandSegmenter (v3) + MotionTracker
    # Để so sánh, đổi thành AdaptiveHandSegmenter để dùng HSV (v2)
    # -------------------------------------------------------------------------
    segmenter      = LabAdaptiveHandSegmenter(
        calibration_roi=(80, 420, 320, 620),
        max_calibration_frames=30
    )
    motion_tracker = MotionTracker(history_len=16, movement_threshold=8)

    print("=" * 70)
    print("  PIPELINE DIP v3 — CIELab Segmentation + Motion Tracking")
    print("-" * 70)
    print("  Segmenter: LabAdaptiveHandSegmenter (CIELab a*,b*)")
    print("  Tracker:   MotionTracker deque(maxlen=16) → vector 32D")
    print("-" * 70)
    print("  Đặt bàn tay vào khung màu CAM để Calibration (30 frames).")
    print("  'c' / 'r': Calibrate lại  |  'q': Thoát")
    print("=" * 70)

    top    = segmenter.top
    bottom = segmenter.bottom
    left   = segmenter.left
    right  = segmenter.right

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # =====================================================================
        # GIAI ĐOẠN 1: CALIBRATION
        # =====================================================================
        if not segmenter.is_calibrated:
            segmenter.draw_calibration_progress(frame)
            cv2.imshow("1. Webcam & Status", frame)

        # =====================================================================
        # GIAI ĐOẠN 2: XỬ LÝ DIP v3 ĐẦY ĐỦ
        # =====================================================================
        else:
            # Phân đoạn (Lab)
            mask = segmenter.segment(frame)

            # Chuẩn hóa hình học — nhận thêm centroid
            normalized_hand, _, centroid = normalize_hand_orientation(
                frame, mask,
                roi_coords=(top, bottom, left, right),
                target_size=(128, 128),
                min_hand_area=segmenter.min_hand_area
            )

            # Cập nhật / reset MotionTracker
            if normalized_hand is not None and centroid is not None:
                motion_tracker.update(centroid[0], centroid[1])
            else:
                motion_tracker.reset()

            # Vẽ vệt sao băng lên full frame (offset ROI → frame coords)
            motion_tracker.draw_trail(frame, offset_x=left, offset_y=top)

            # Thông tin hướng chuyển động
            direction = motion_tracker.get_gesture_direction()
            moving    = motion_tracker.is_moving()
            traj      = motion_tracker.get_normalized_trajectory()
            status    = f"{'>>> MOVING <<<' if moving else 'STILL'}  |  {direction}"

            # Khung ROI màu xanh lá (đã calibrate)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 100), 2)
            cv2.putText(frame, f"Lab v3  |  {status}",
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)

            # Hiển thị tensor 3 kênh nếu có bàn tay
            if normalized_hand is not None:
                tensor = extract_multi_channel_features(normalized_hand)
                ch0, ch1, ch2 = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]

                vis = cv2.cvtColor(np.hstack([ch0, ch1, ch2]), cv2.COLOR_GRAY2BGR)
                cv2.putText(vis, "CH0:CLAHE+blur",  (  5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
                cv2.putText(vis, "CH1:AutoCanny",   (133, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
                cv2.putText(vis, "CH2:Sobel",       (261, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

                cv2.imshow("3. Normalized Hand (Moments)", normalized_hand)
                cv2.imshow("4. Multi-Channel Tensor (128x128x3)", vis)

                # In trajectory vector ra console (debug)
                if traj is not None:
                    traj_str = " ".join(f"{x:+.2f}" for x in traj[:8])
                    print(f"\r[Traj 32D] [{traj_str} ...]  dir={direction:<6}", end="", flush=True)

            cv2.imshow("1. Webcam + Motion Trail", frame)
            cv2.imshow("2. Lab Skin Mask (Backprojection)", mask)

        # Phím điều khiển
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('c'), ord('r')):
            print("\n[Reset] Đang Reset Calibration + Motion Tracker...")
            segmenter.reset_calibration()
            motion_tracker.reset()

    print()   # Xuống dòng sau chuỗi print traj
    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# SECTION M: KIẾN TRÚC PIPELINE HỢP NHẤT TINH GỌN (LEAN CONSOLIDATED PIPELINE v4)
# =============================================================================

def extract_smart_hand_crop(frame: np.ndarray, landmarks, handedness_label: str = 'Right', auto_flip_left: bool = False):
    """
    [Giai đoạn 1] Cắt crop bàn tay thông minh từ 21 Landmarks:
    - Cắt Landmark 0 (khớp cổ tay): triệt tiêu 100% cẳng tay (Case 4).
    - Lề ngang 15%, lề trên 15%, lề đáy khớp cổ tay + 5% (tránh cụt ngón tay / ngón cái).
    - Kiểm tra chạm mép camera (Case 12).
    - Chuẩn hóa chiều tay trái về tay phải (Case 14 - tùy chọn, mặc định tắt để người dùng chủ động phím [f]).
    """
    h_f, w_f = frame.shape[:2]
    xs = [lm.x * w_f for lm in landmarks]
    ys = [lm.y * h_f for lm in landmarks]

    wrist_y = landmarks[0].y * h_f
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)

    x1 = max(0, int(min(xs) - 0.15 * bw))
    x2 = min(w_f, int(max(xs) + 0.15 * bw))
    y1 = max(0, int(min(ys) - 0.15 * bh))
    y2 = min(h_f, int(max(max(ys), wrist_y) + 0.05 * bh))

    touching_border = (x1 <= 5 or y1 <= 5 or x2 >= w_f - 5 or y2 >= h_f - 5)
    crop = frame[y1:y2, x1:x2].copy()

    is_flipped = False
    if auto_flip_left and handedness_label == 'Left':
        crop = cv2.flip(crop, 1)
        is_flipped = True

    return crop, (x1, y1, x2, y2), touching_border, is_flipped


def segment_hand_crop_dip(crop_bgr: np.ndarray) -> np.ndarray:
    """
    [DIP Pure] Phân đoạn bàn tay trên vùng ảnh cắt nhỏ (Tight Crop):
    - Sử dụng không gian màu chrominance kép: CIELab (a*, b*) kết hợp YCrCb (Cr, Cb).
      (Loại bỏ 100% tường trắng, trần nhà, ánh sáng lóa vì không gian sắc độ bất biến với độ sáng Y/L).
    - Biến đổi hình thái học DIP: Morphological Closing & Opening khử nhiễu hạt.
    - Lấy contour lớn nhất và tự động hàn kín lòng bàn tay bằng cv2.fillPoly.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    # 1. CIELab Chrominance (a*, b*)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab)
    mask_lab = cv2.inRange(lab, (20, 128, 125), (250, 185, 185))

    # 2. YCrCb Chrominance (Cr, Cb)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 175, 127))

    # Giao thoa 2 không gian màu để triệt tiêu nền ngoài da
    skin_raw = cv2.bitwise_and(mask_lab, mask_ycrcb)

    # 3. Phép biến đổi hình thái học DIP
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(skin_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Lấy contour lớn nhất và hàn kín lỗ hổng lòng bàn tay
    cnts, hier = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask_out = np.zeros_like(clean)
    if cnts and hier is not None:
        max_idx = int(np.argmax([cv2.contourArea(c) for c in cnts]))
        cv2.drawContours(mask_out, cnts, max_idx, 255, -1)
        for i in range(len(cnts)):
            if hier[0][i][3] == max_idx:
                cv2.fillPoly(mask_out, [cnts[i]], 255)
    else:
        mask_out = np.ones((crop_bgr.shape[0], crop_bgr.shape[1]), dtype=np.uint8) * 255

    return mask_out


def process_lean_dip_tensor(crop_bgr: np.ndarray, target_size=(128, 128)) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    [Giai đoạn 2] Bộ lọc DIP tinh gọn, chạy trực tiếp trên vùng Crop bàn tay:
    1. Bilateral Filter: Lọc phẳng nhiễu hạt cảm biến ISO (Case 1) và Block JPEG (Case 18) nhưng GIỮ SẮC BIÊN.
    2. DIP Skin Segmentation: CIELab (a*, b*) + YCrCb (Cr, Cb) + FillPoly lấp bóng đổ, xóa 100% phông nền tường trắng.
    3. Chuẩn hóa hình học Image Moments: Xoay trục chính về 0° thẳng đứng, gọt cổ tay, căn giữa và padding vào canvas (target_size) với phông nền đen tuyệt đối (0).
    4. Trích xuất Tensor 3 kênh DIP:
       - Kênh 0: Blur(3x3) + CLAHE (hiện rõ ngón cái đè lòng bàn tay - Case 16).
       - Kênh 1: AutoCanny (bắt viền ngón tay sắc nét, miễn nhiễm son móng tay - Case 13).
       - Kênh 2: Sobel Gradient Magnitude (bắt cấu trúc khi ngược sáng - Case 10).
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
        empty = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return empty, empty, 0.0

    # 1. Bilateral Filter khử nhiễu ISO giữ biên
    filtered = cv2.bilateralFilter(crop_bgr, d=5, sigmaColor=45, sigmaSpace=45)

    # 2. Phân đoạn màu da DIP tách phông nền tường trắng
    crop_mask = segment_hand_crop_dip(filtered)

    # 3. Chuẩn hóa hình học bằng Image Moments (xoay trục chính về 0°, xóa nền thành đen tuyền)
    ch, cw = filtered.shape[:2]
    norm_hand, _, _ = normalize_hand_orientation(
        filtered, crop_mask,
        roi_coords=(0, ch, 0, cw),
        target_size=target_size,
        min_hand_area=max(200, int(ch * cw * 0.05))
    )

    if norm_hand is None:
        # Fallback an toàn: cắt mặt nạ trực tiếp và căn giữa vào canvas đen
        scale = min((target_size[1] - 16) / cw, (target_size[0] - 16) / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        masked_crop = cv2.bitwise_and(filtered, filtered, mask=crop_mask)
        resized = cv2.resize(masked_crop, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        sy = (target_size[1] - nh) // 2
        sx = (target_size[0] - nw) // 2
        canvas[sy:sy + nh, sx:sx + nw] = resized
        rot_angle_deg = 0.0
    else:
        canvas = norm_hand
        rot_angle_deg = 0.0

    # 4. Trích xuất Tensor 3 kênh DIP chuẩn hóa
    tensor_np = extract_multi_channel_features(canvas)
    return tensor_np, canvas, rot_angle_deg

