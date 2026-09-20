"""
DYNAMIC FEATURES - REGISTRY ĐẶC TRƯNG + 4 NHÓM ĐẶC TRƯNG MỚI (17D -> 24D + rep_count)
=============================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086
Môn học: Xử Lý Ảnh Số (DIP)

MỤC ĐÍCH (SPEC 03)
------------------
Thêm 4 nhóm thông tin mà hệ thống trước đây THIẾU về mặt cấu trúc, theo cách CỘNG THÊM,
đo được từng nhóm một và tắt được (trọng số = 0.0 => hành vi cũ, bit-identical):

  1. z_norm        (cột 17)     : độ sâu ảo suy ra từ palm_len (tay đẩy ra / kéo vào).
  2. Mốc cơ thể    (cột 18:20)  : vị trí cổ tay so với hộp mặt (FATHER trán vs MOTHER cằm).
  3. Độ xòe ngón   (cột 20:24)  : 4 góc giữa các ngón LIỀN KỀ (V xòe vs U khép).
  4. Số nhịp lặp   (KHÔNG phải cột): hằng số cho cả chuỗi, cộng MỘT LẦN sau DTW (YES/BYE lặp).

LAYOUT ĐÍCH: 24 chiều/frame + 1 hằng số/chuỗi (17 cột đầu GIỮ NGUYÊN, chỉ thêm vào sau):
    [xs, ys, dxs, dys, angles(5), normals(3), dangles(5), z_norm(1), body(2), abduction(4)]

QUY ƯỚC QUAN TRỌNG: mọi công thức trong file này là NGUỒN SỰ THẬT DUY NHẤT. Cả 4 đường
trích xuất (predict_sentence_webcam / collect_dynamic_templates / inspect_and_trim_asl_citizen /
rebuild_templates_from_raw) + reextract_citizen_batch đều GỌI các hàm ở đây, không tự viết lại,
để không tái diễn thảm họa lệch miền đã xảy ra với chirality.

File này chỉ phụ thuộc numpy. mediapipe chỉ được import LƯỜI bên trong CachedFaceDetector.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# 1. REGISTRY KHAI BÁO CÁC NHÓM ĐẶC TRƯNG
# =============================================================================

SQRT2 = 1.41421356
SQRT5 = 2.23606798

FEATURE_DIM_LEGACY = 17     # Số chiều của bank cũ (webcam hiện tại, Citizen hiện tại)
FEATURE_DIM_EXTENDED = 24   # Số chiều sau SPEC 03

# (tên_nhóm, cột_bắt_đầu, cột_kết_thúc, khóa_trọng_số, kiểu_khoảng_cách, hệ_số_chuẩn_hóa)
# kiểu_khoảng_cách: "euclid" | "cosine" | "scalar_abs"
# Hệ số chuẩn hóa = sqrt(số chiều của nhóm) để nhóm nhiều chiều không âm thầm áp đảo.
# LƯU Ý: nhóm "dangles" dùng chung khóa w_a với "angles" và được TRUNG BÌNH với nhóm angles
# trong hàm khoảng cách (xem compute_golden_triad_dtw_distance) — registry chỉ khai báo cột.
FEATURE_GROUPS: List[Tuple[str, int, int, str, str, float]] = [
    ("position",    0,  2, "w_p", "euclid",     SQRT2),   # GIỮ NGUYÊN
    ("velocity",    2,  4, "w_m", "euclid",     SQRT2),   # GIỮ NGUYÊN
    ("angles",      4,  9, "w_a", "euclid",     SQRT5),   # GIỮ NGUYÊN
    ("normal",      9, 12, "w_n", "cosine",     1.0),     # GIỮ NGUYÊN
    ("dangles",    12, 17, "w_a", "euclid",     SQRT5),   # GIỮ NGUYÊN
    ("z_norm",     17, 18, "w_z", "scalar_abs", 1.0),     # MỚI
    ("body",       18, 20, "w_b", "euclid",     SQRT2),   # MỚI
    ("abduction",  20, 24, "w_s", "euclid",     2.0),     # MỚI (sqrt(4) = 2)
]

# Nhóm cấp CHUỖI (không phải cột per-frame): (tên, khóa_trọng_số, mô_tả)
SEQUENCE_LEVEL_GROUPS: List[Tuple[str, str, str]] = [
    ("rep_count", "w_r", "w_r * |rep_query - rep_template|, cộng 1 lần sau đường DTW"),
]

NEW_WEIGHT_KEYS = ("w_z", "w_b", "w_s", "w_r")

# Cột 18 = x của mốc cơ thể: đổi dấu khi lật gương chirality (x ảnh bị lật => x tương đối so
# với tâm mặt cũng đổi dấu). Xem canonicalize_chirality trong dynamic_dtw_engine.py.
BODY_X_COLUMN = 18


def get_group(name: str) -> Tuple[str, int, int, str, str, float]:
    """Tra một nhóm theo tên trong FEATURE_GROUPS."""
    for g in FEATURE_GROUPS:
        if g[0] == name:
            return g
    raise KeyError(f"Không có nhóm đặc trưng tên '{name}'")


def groups_available_for_dim(dim: int) -> List[str]:
    """Danh sách tên nhóm có mặt đầy đủ trong vector có `dim` cột."""
    return [g[0] for g in FEATURE_GROUPS if dim >= g[2]]


# =============================================================================
# 2. CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG (đúng công thức trong SPEC 03)
# =============================================================================

def compute_palm_len(lm0: Any, lm5: Any, lm9: Any, lm17: Any, w_f: float, h_f: float) -> float:
    """
    Kích thước lòng bàn tay (palm_len, đơn vị pixel) chống co ngắn phối cảnh — NGUỒN SỰ THẬT DUY NHẤT.

        palm_2d      = |(lm9 - lm0)| trên mặt phẳng ảnh              (x nhân w_f, y nhân h_f)
        palm_3d      = sqrt(palm_2d^2 + ((lm9.z - lm0.z) * w_f)^2)    (z của MediaPipe cùng thang với x => nhân w_f)
        knuckle_span = |(lm17 - lm5)| trên mặt phẳng ảnh              (x nhân w_f, y nhân h_f)
        palm_len     = max(palm_2d, palm_3d, 0.85 * knuckle_span, 30.0)

    lm0, lm5, lm9, lm17: bốn mốc MediaPipe (cổ tay, gốc ngón trỏ, gốc ngón giữa, gốc ngón út), mỗi mốc là
    đối tượng có thuộc tính .x, .y, .z CHUẨN HÓA [0, 1] (z cùng thang với x). w_f, h_f: bề rộng, bề cao khung
    hình (pixel).

    CẢNH BÁO LỊCH SỬ: trước đây collect_dynamic_templates.py sao chép công thức này nhưng nhân thành phần y của
    palm_3d với w_f (bề RỘNG) thay vì h_f (bề CAO) => ở khung 640x480 palm_len khi thu lớn hơn lúc chạy thật tới
    640/480 = 1.33 lần (khi tay hướng lên). Từ nay MỌI đường thu / chạy / dựng lại phải GỌI hàm này.
    """
    dx = (lm9.x - lm0.x) * w_f
    dy = (lm9.y - lm0.y) * h_f
    palm_2d = math.hypot(dx, dy)
    palm_3d = math.sqrt(dx ** 2 + dy ** 2 + ((lm9.z - lm0.z) * w_f) ** 2)
    knuckle_span = math.hypot((lm17.x - lm5.x) * w_f, (lm17.y - lm5.y) * h_f)
    return max(palm_2d, palm_3d, knuckle_span * 0.85, 30.0)


def compute_z_norm(palm_len_seq: Sequence[float]) -> np.ndarray:
    """
    Độ sâu ảo per-frame: z[t] = clip((palm_len[t] - palm_len[0]) / palm_len[0], -1, 1).

    Ý nghĩa: tay đẩy về camera => bàn tay to lên => z > 0; tay kéo về ngực => z < 0.
    Đây là thứ phân biệt YOU (chỉ ra) với ME (chỉ vào mình).
    Việc clip vào [-1, 1] là BẮT BUỘC: palm_len có thể tăng gấp đôi khi tay sát camera,
    không clip thì nhóm này áp đảo mọi nhóm khác. Nếu palm_len[0] < 1e-6 => toàn 0.
    """
    pl = np.asarray(palm_len_seq, dtype=np.float64).reshape(-1)
    if pl.size == 0:
        return np.zeros(0, dtype=np.float32)
    p0 = float(pl[0])
    if p0 < 1e-6:
        return np.zeros(pl.shape[0], dtype=np.float32)
    return np.clip((pl - p0) / p0, -1.0, 1.0).astype(np.float32)


# Các cặp landmark (đỉnh ngón, gốc ngón) định hướng từng ngón: cái, trỏ, giữa, nhẫn, út.
_FINGER_TIP_MCP = ((4, 2), (8, 5), (12, 9), (16, 13), (20, 17))


def compute_finger_abduction(landmarks_3d: np.ndarray) -> np.ndarray:
    """
    Độ xòe ngón: 4 góc giữa các cặp hướng ngón LIỀN KỀ (cái-trỏ, trỏ-giữa, giữa-nhẫn, nhẫn-út),
    chia pi để về [0, 1]. Hướng ngón = vector chuẩn hóa (tip - mcp).

    Vì sao cần: 5 góc gập chỉ đo ngón cong bao nhiêu, KHÔNG đo ngón cách nhau bao xa,
    nên V (2 ngón xòe) và U (2 ngón khép) trước đây không phân biệt được.
    Vector nào có norm < 1e-6 => góc liên quan đặt = 0.

    landmarks_3d: (21, 3). Trả về (4,) float32.
    """
    lm = np.asarray(landmarks_3d, dtype=np.float64)
    dirs = []
    valid = []
    for tip, mcp in _FINGER_TIP_MCP:
        v = lm[tip] - lm[mcp]
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            dirs.append(np.zeros(3))
            valid.append(False)
        else:
            dirs.append(v / n)
            valid.append(True)
    out = np.zeros(4, dtype=np.float64)
    for k in range(4):
        if valid[k] and valid[k + 1]:
            dot = float(np.clip(np.dot(dirs[k], dirs[k + 1]), -1.0, 1.0))
            out[k] = math.acos(dot) / math.pi
    return out.astype(np.float32)


def compute_finger_abduction_seq(landmarks_seq: Sequence[np.ndarray]) -> np.ndarray:
    """Áp compute_finger_abduction cho từng frame. Trả về (N, 4)."""
    if len(landmarks_seq) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack([compute_finger_abduction(lm) for lm in landmarks_seq]).astype(np.float32)


def compute_repetition_count(xs: Sequence[float], ys: Sequence[float]) -> int:
    """
    Đếm số nhịp lặp của quỹ đạo cổ tay (thuộc tính TOÀN CỤC của chuỗi, không phải cột per-frame).

    1. speed[t] = hypot(dx, dy), speed[0] = speed[1].
    2. Làm mượt trung bình trượt cửa sổ 3 (biên: lặp phần tử biên).
    3. Đếm cực đại địa phương có giá trị >= 0.30 * max(speed đã mượt) VÀ cách cực đại được
       chấp nhận trước đó >= 4 frame (tham lam từ trái sang phải).
    4. Kẹp vào [1, 6].

    CHỌN THỰC THI (báo cáo): mọi nơi đều gọi hàm này trên chuỗi ĐÃ RESAMPLE K=32 (cột 0, 1),
    vì match() chỉ nhận được chuỗi K=32 của query. Nhờ vậy không phụ thuộc FPS nguồn video.
    """
    x = np.asarray(xs, dtype=np.float64).reshape(-1)
    y = np.asarray(ys, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n < 3:
        return 1
    speed = np.hypot(np.diff(x[:n]), np.diff(y[:n]))          # độ dài n-1
    speed = np.concatenate([[speed[0]], speed])                # speed[0] = speed[1]
    padded = np.concatenate([[speed[0]], speed, [speed[-1]]])
    smooth = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    peak_val = float(np.max(smooth))
    if peak_val < 1e-9:
        return 1
    thr = 0.30 * peak_val
    count = 0
    last_peak = -10 ** 9
    for t in range(1, n - 1):
        if smooth[t] > smooth[t - 1] and smooth[t] >= smooth[t + 1] and smooth[t] >= thr:
            if t - last_peak >= 4:
                count += 1
                last_peak = t
    return int(min(max(count, 1), 6))


# Kẹp giá trị mốc cơ thể để chống giá trị bệnh lý khi hộp mặt bị lỗi.
BODY_ANCHOR_CLIP = 6.0


def compute_body_anchor(wrist_xy: Sequence[float], face_box: Optional[Sequence[float]]) -> Tuple[np.ndarray, bool]:
    """
    Mốc cơ thể: ((wx - face_cx) / face_w, (wy - face_cy) / face_w), kẹp mỗi giá trị vào [-6, 6].

    Chia BỀ RỘNG MẶT (ổn định), KHÔNG chia palm_len (co giãn theo tư thế/góc quay bàn tay).
    face_box = (cx, cy, w, h) tính bằng pixel. Nếu face_box là None (hoặc w <= 0) => trả về
    ((0, 0), False); cờ False báo cho phía gọi đặt w_b = 0 cho stroke đó.
    Trả về (mảng (2,), has_face).
    """
    if face_box is None:
        return np.zeros(2, dtype=np.float32), False
    cx, cy, fw, _fh = [float(v) for v in face_box]
    if not (fw > 1e-6) or not np.isfinite(fw):
        return np.zeros(2, dtype=np.float32), False
    wx, wy = float(wrist_xy[0]), float(wrist_xy[1])
    bx = float(np.clip((wx - cx) / fw, -BODY_ANCHOR_CLIP, BODY_ANCHOR_CLIP))
    by = float(np.clip((wy - cy) / fw, -BODY_ANCHOR_CLIP, BODY_ANCHOR_CLIP))
    return np.array([bx, by], dtype=np.float32), True


def fill_face_boxes(face_boxes: Sequence[Optional[Sequence[float]]]) -> Optional[List[Tuple[float, float, float, float]]]:
    """
    Điền các frame thiếu hộp mặt bằng hộp gần nhất theo thời gian (đầu gần như bất động khi ký).
    Trả về None nếu KHÔNG frame nào có hộp mặt (=> stroke này không có mốc cơ thể).
    """
    n = len(face_boxes)
    idx = [i for i, b in enumerate(face_boxes) if b is not None]
    if not idx:
        return None
    out: List[Optional[Tuple[float, float, float, float]]] = [None] * n
    for i in range(n):
        if face_boxes[i] is not None:
            b = face_boxes[i]
            out[i] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        else:
            j = min(idx, key=lambda k: abs(k - i))
            b = face_boxes[j]
            out[i] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return out  # type: ignore[return-value]


def compute_extended_columns(
    palm_lens: Sequence[float],
    wrist_xy: Sequence[Sequence[float]],
    landmarks_seq: Sequence[Optional[np.ndarray]],
    face_boxes: Sequence[Optional[Sequence[float]]],
) -> Tuple[np.ndarray, bool]:
    """
    Tính 7 cột MỚI cho từng frame ở FPS gốc của nguồn: [z_norm(1), body(2), abduction(4)].

    palm_lens     : (N,) palm_len từng frame (đúng biến palm_len của GestureFrame).
    wrist_xy      : (N, 2) tọa độ cổ tay pixel (bản đã làm trơn / đã nội suy, đúng cột 0:2 dùng).
    landmarks_seq : N phần tử, mỗi phần tử (21, 3) tọa độ pixel-z*w (đúng lm_3d), hoặc None.
                    Frame None dùng độ xòe của frame hợp lệ gần nhất.
    face_boxes    : N phần tử (cx, cy, w, h) hoặc None.

    Trả về (ext (N, 7) float32, body_valid). body_valid = False khi KHÔNG có hộp mặt nào:
    khi đó cột mốc cơ thể = 0 và phía gọi phải ghi cờ để w_b bị vô hiệu cho stroke này.
    """
    n = len(palm_lens)
    z = compute_z_norm(palm_lens)

    wr = np.asarray(wrist_xy, dtype=np.float64).reshape(n, 2)
    filled = fill_face_boxes(list(face_boxes))
    body = np.zeros((n, 2), dtype=np.float32)
    body_valid = filled is not None
    if body_valid:
        for t in range(n):
            body[t], _ = compute_body_anchor(wr[t], filled[t])

    abd = np.zeros((n, 4), dtype=np.float32)
    valid_idx = [t for t in range(n) if landmarks_seq[t] is not None]
    for t in range(n):
        if landmarks_seq[t] is not None:
            abd[t] = compute_finger_abduction(landmarks_seq[t])
        elif valid_idx:
            j = min(valid_idx, key=lambda k: abs(k - t))
            abd[t] = compute_finger_abduction(landmarks_seq[j])

    ext = np.concatenate([z.reshape(n, 1), body, abd], axis=1).astype(np.float32)
    return ext, body_valid


def append_extended_columns(base_17d: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """Ghép 17 cột cũ + 7 cột mới => (N, 24). Thứ tự: [..17.., z_norm, body(2), abduction(4)]."""
    return np.concatenate([np.asarray(base_17d, dtype=np.float32), np.asarray(ext, dtype=np.float32)], axis=1).astype(np.float32)


def build_segment_features(
    frames: Sequence[Dict[str, Any]],
    k_target: int = 32,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], bool, Optional[str]]:
    """
    MỘT hàm duy nhất dựng đặc trưng 24D đã resample K cho MỘT đoạn khung (SPEC 08 Nhiệm vụ "dùng chung", cũng là
    SPEC 09 Nhiệm vụ A). Đây chính là đoạn code trước kia viết inline ở predict_sentence_webcam.py (cả nhánh chốt
    thật lẫn nhánh "dự đoán tạm"); công thức GIỮ NGUYÊN, kết quả bit-giống hệt.

    `frames`: danh sách bản ghi từng khung (dict), theo thứ tự thời gian, có các khóa:
        x, y          : tọa độ cổ tay (đúng giá trị đã đưa vào bộ phân đoạn, tức bản đã làm mượt)
        palm_len      : độ dài lòng bàn tay (bản đã kẹp >= 15 như GestureFrame.palm_len)
        hand_shape    : nhãn dáng tay (chuỗi)
        angles        : (5,) góc gập ngón
        normal        : (3,) pháp tuyến lòng bàn tay
        landmarks     : (21, 3) landmark pixel (z nhân bề rộng khung) hoặc None
        face_box      : (cx, cy, w, h) hoặc None
        handedness    : nhãn tay thuận của khung, hoặc None (khóa này có thể vắng)
    Trả về (res_traj_24d (K, 24), res_shapes (K,), body_ok, dominant_handedness).
    res_traj_24d = None nếu đoạn quá ngắn cho resample. dominant_handedness = nhãn chiếm đa số trong các khung
    (None nếu không khung nào có nhãn).
    """
    from dynamic_dtw_engine import resample_9d_gesture_sequence

    n = len(frames)
    x0, y0 = frames[0]["x"], frames[0]["y"]
    ref_palm = max(frames[0]["palm_len"], 15.0)

    raw_xs = np.array([(f["x"] - x0) / ref_palm for f in frames], dtype=np.float32)
    raw_ys = np.array([(f["y"] - y0) / ref_palm for f in frames], dtype=np.float32)

    raw_dxs = np.zeros(n, dtype=np.float32)
    raw_dys = np.zeros(n, dtype=np.float32)
    raw_dxs[1:] = raw_xs[1:] - raw_xs[:-1]
    raw_dys[1:] = raw_ys[1:] - raw_ys[:-1]
    raw_dxs[0] = raw_dxs[1] if n > 1 else 0.0
    raw_dys[0] = raw_dys[1] if n > 1 else 0.0

    raw_angles_mat = np.array([f["angles"] for f in frames], dtype=np.float32)
    raw_dangles_mat = np.zeros_like(raw_angles_mat)
    raw_dangles_mat[1:] = raw_angles_mat[1:] - raw_angles_mat[:-1]
    raw_dangles_mat[0] = raw_dangles_mat[1] if n > 1 else 0.0

    raw_normals_mat = np.array([f["normal"] for f in frames], dtype=np.float32)
    raw_17d = np.column_stack(
        [raw_xs, raw_ys, raw_dxs, raw_dys, raw_angles_mat, raw_normals_mat, raw_dangles_mat]
    ).astype(np.float32)

    raw_ext, body_ok = compute_extended_columns(
        [f["palm_len"] for f in frames],
        [(f["x"], f["y"]) for f in frames],
        [f["landmarks"] for f in frames],
        [f["face_box"] for f in frames],
    )
    raw_24d = append_extended_columns(raw_17d, raw_ext)
    shapes = [f["hand_shape"] for f in frames]

    res_traj, res_shapes = resample_9d_gesture_sequence(raw_24d, shapes, k_target=k_target)

    labels = [f.get("handedness") for f in frames if f.get("handedness")]
    dominant = max(set(labels), key=labels.count) if labels else None
    return res_traj, res_shapes, body_ok, dominant


def extend_still_tail(
    records: Sequence[Dict[str, Any]],
    target_tail_sec: float,
    v_stop: float = 0.040,
    min_frames: int = 8,
    replay_jitter: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Kéo dài đoạn ĐỨNG YÊN ở đuôi một stroke cho đủ `target_tail_sec` giây, bằng cách nhân bản bản ghi khung cuối.

    Vì sao cần (SPEC 08 Nhiệm vụ D): stroke chốt theo kiểu cũ chứa ~0,45 s đứng yên ở đuôi (9 khung khựng + grace), và
    mọi mẫu trong bank đều mang đuôi đó. Khi rút ngắn thời gian chờ (chốt sớm), đuôi ngắn hơn => tỉ lệ thời gian của
    quỹ đạo sau khi resample K=32 lệch khỏi bank. Đuôi vốn là đứng yên (v <= v_stop) nên nhân bản khung cuối cho đúng
    độ dài cũ đưa truy vấn về gần như GIỐNG HỆT stroke chốt kiểu cũ, KHÔNG cần đổi bank.

    records: danh sách dict, mỗi phần tử có x, y, palm_len, t (dấu thời gian), các khóa khác được giữ nguyên.
    Đuôi hiện có = khoảng thời gian từ khung cuối còn chuyển động (v > v_stop, v = |Δcổ tay| / max(palm, 15)) tới khung cuối.
    Thêm số khung n_add = round((target_tail_sec - đuôi hiện có) / dt) với dt = trung vị khoảng cách giữa các khung;
    thêm nữa nếu tổng chưa đủ `min_frames` (resample cần >= 8 khung).
    replay_jitter=True: khung thêm KHÔNG đứng bất động tuyệt đối mà lặp lại các bước nhiễu cổ tay của chính đuôi yên đã ghi
    (luân phiên đổi dấu mỗi vòng để không trôi), giữ nguyên độ dài đường đi của đuôi như đuôi thật. Cần cho heuristic
    "chuyển động chêm" (is_movement_epenthesis: đo độ THẲNG = độ dời / độ dài đường đi): nhiễu ở đuôi 0,45 s làm đường đi dài hơn
    một chút nên các cử chỉ vốn rất thẳng (như THANK_YOU) nằm dưới ngưỡng 0.97; nếu đuôi bị rút ngắn và thay bằng khung
    bất động thì độ thẳng nhích lên trên ngưỡng và từ bị loại nhầm. Mặc định False (chỉ nhân bản khung cuối).
    Trả về (danh sách mới, số khung đã thêm). Không đổi danh sách gốc.
    """
    recs = list(records)
    n = len(recs)
    if n == 0:
        return recs, 0
    ts = np.array([float(r["t"]) for r in recs], dtype=np.float64)
    dts = np.diff(ts)
    dts = dts[dts > 1e-6]
    dt = float(np.median(dts)) if dts.size else (1.0 / 30.0)
    last_motion = 0
    for i in range(1, n):
        step = math.hypot(recs[i]["x"] - recs[i - 1]["x"], recs[i]["y"] - recs[i - 1]["y"])
        if step / max(float(recs[i]["palm_len"]), 15.0) > v_stop:
            last_motion = i
    tail_dur = float(ts[-1] - ts[last_motion])
    n_add = max(0, int(round((float(target_tail_sec) - tail_dur) / dt)))
    n_add = max(n_add, int(min_frames) - n)
    steps: List[Tuple[float, float]] = []
    if replay_jitter:
        steps = [(recs[i]["x"] - recs[i - 1]["x"], recs[i]["y"] - recs[i - 1]["y"]) for i in range(last_motion + 1, n)]
    m = len(steps)
    x_cur, y_cur = float(recs[-1]["x"]), float(recs[-1]["y"])
    for k in range(1, n_add + 1):
        r = dict(recs[-1])
        r["t"] = float(ts[-1]) + k * dt
        if m:
            sign = 1.0 if ((k - 1) // m) % 2 == 0 else -1.0
            x_cur += sign * steps[(k - 1) % m][0]
            y_cur += sign * steps[(k - 1) % m][1]
            r["x"], r["y"] = x_cur, y_cur
        recs.append(r)
    return recs, max(n_add, 0)


def records_to_gesture_frames(records: Sequence[Dict[str, Any]]) -> List[Any]:
    """Bản ghi khung (dict) -> danh sách GestureFrame, để dùng các hàm kiểm tra nhận GestureFrame (vd is_movement_epenthesis)."""
    from dynamic_dtw_engine import GestureFrame
    return [GestureFrame(x=float(r["x"]), y=float(r["y"]), palm_len=float(r["palm_len"]),
                         hand_shape=r.get("hand_shape", "Unknown"), timestamp=float(r["t"])) for r in records]


def finalize_stroke_features(
    raw_matrix: np.ndarray,
    shapes: List[str],
    handedness: Optional[str],
    k_target: int = 32,
    min_frames: int = 8,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], int, bool]:
    """
    Bước chốt CHUNG cho mọi đường trích xuất: resample K=32 -> chuẩn hóa chirality -> đếm nhịp lặp.
    Trả về (traj (K, D), shapes (K,), rep_count, was_flipped); traj = None nếu quá ngắn.
    """
    from dynamic_dtw_engine import resample_9d_gesture_sequence, canonicalize_chirality

    res, res_shapes = resample_9d_gesture_sequence(raw_matrix, shapes, k_target=k_target, min_frames=min_frames)
    if res is None or res_shapes is None:
        return None, None, 1, False
    res, flipped = canonicalize_chirality(res, handedness=handedness)
    rep = compute_repetition_count(res[:, 0], res[:, 1])
    return res, res_shapes, rep, flipped


# =============================================================================
# 3. BỘ DÒ MẶT CÓ CACHE
# =============================================================================

FACE_MODEL_NAME = "blaze_face_short_range.tflite"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")


class CachedFaceDetector:
    """
    Bộ dò mặt MediaPipe (FaceDetector, blaze_face_short_range) có CACHE.

    Đầu gần như bất động khi ký, nên chỉ cần dò mỗi `detect_every_n_frames` frame; giữa các
    lần dò trả lại hộp cache => chi phí khấu hao còn ~1/8.
    Nếu dò thất bại: giữ hộp cache cũ, nhưng nếu đã quá `max_stale_frames` (30) frame không
    thấy mặt thì trả None. `reset()` xóa cache.

    Hộp mặt trả về là (cx, cy, w, h) theo PIXEL của khung `out_size` (mặc định = kích thước
    ảnh đưa vào). Truyền out_size = (W, H) của khung gốc khi ảnh dò đã bị thu nhỏ (detect_scale).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        detect_every_n_frames: int = 8,
        max_stale_frames: int = 30,
        min_detection_confidence: float = 0.5,
        auto_download: bool = True,
    ):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FACE_MODEL_NAME)
        self.model_path = model_path
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self.max_stale_frames = int(max_stale_frames)

        if not os.path.exists(self.model_path):
            if not auto_download:
                raise FileNotFoundError(f"Không tìm thấy model dò mặt: {self.model_path}")
            print(f"[-] Đang tải {FACE_MODEL_NAME}...")
            import urllib.request
            urllib.request.urlretrieve(FACE_MODEL_URL, self.model_path)
            print("✓ Tải xong model MediaPipe FaceDetector!")

        import mediapipe as mp
        self._mp = mp
        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=float(min_detection_confidence),
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

        self._cache: Optional[Tuple[float, float, float, float]] = None
        self._frame_idx = 0
        self._last_seen_idx = -10 ** 9
        self.n_detect_calls = 0      # số lần thực sự chạy mạng dò mặt (để đo chi phí)
        self.n_detect_hits = 0       # số lần dò thấy mặt

    def reset(self) -> None:
        """Xóa cache hộp mặt và bộ đếm frame."""
        self._cache = None
        self._frame_idx = 0
        self._last_seen_idx = -10 ** 9

    def _run_detector(self, rgb_frame: np.ndarray, out_size: Optional[Tuple[int, int]]) -> Optional[Tuple[float, float, float, float]]:
        mp = self._mp
        h_i, w_i = rgb_frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        res = self._detector.detect(img)
        self.n_detect_calls += 1
        if not res.detections:
            return None
        # Chọn khuôn mặt có điểm tin cậy cao nhất (mặt của người ký, thường to nhất/rõ nhất).
        best = max(res.detections, key=lambda d: (d.categories[0].score if d.categories else 0.0))
        bb = best.bounding_box
        sx = (out_size[0] / float(w_i)) if out_size else 1.0
        sy = (out_size[1] / float(h_i)) if out_size else 1.0
        cx = (bb.origin_x + bb.width / 2.0) * sx
        cy = (bb.origin_y + bb.height / 2.0) * sy
        return (float(cx), float(cy), float(bb.width * sx), float(bb.height * sy))

    def detect(self, rgb_frame: np.ndarray, out_size: Optional[Tuple[int, int]] = None) -> Optional[Tuple[float, float, float, float]]:
        """
        Gọi MỘT LẦN cho MỖI frame (thứ tự thời gian). Trả về hộp mặt (cx, cy, w, h) pixel
        hoặc None nếu chưa từng thấy / đã quá lâu không thấy mặt.
        rgb_frame: ảnh RGB uint8 (H, W, 3).
        """
        idx = self._frame_idx
        self._frame_idx += 1
        if idx % self.detect_every_n_frames == 0:
            box = self._run_detector(rgb_frame, out_size)
            if box is not None:
                self._cache = box
                self._last_seen_idx = idx
                self.n_detect_hits += 1
        if self._cache is not None and (idx - self._last_seen_idx) > self.max_stale_frames:
            self._cache = None
        return self._cache
