"""
Unit tests for Golden Triad [P + A + JM] feature engineering and DTW matcher (v5.1).
Covers:
- Joint flexion angles computation (straight = 0.0, fully bent = 1.0)
- 3D rotation invariance of joint flexion angles
- Resampling of 9D feature sequences to K=32
- Golden Triad Dimension-Normalized Weighted DTW distance
- Relative Contrast Margin Guard safe division
- Backward compatibility with legacy 4D templates
"""

import math
import time
import numpy as np
import pytest

from predict_sentence_webcam import (
    compute_joint_flexion_angles,
    compute_palm_normal_vector,
    resample_9d_gesture_sequence,
    compute_golden_triad_dtw_distance,
    GoldenTriadDTWMatcher,
)


def test_straight_finger_angles():
    """Tat ca 5 ngon duoi thang hoan toan -> Goc uon (Flexion Score) phai bang 0.0."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Thumb: 1 -> 2 -> 3
    lm[1] = [0.0, 0.0, 0.0]
    lm[2] = [0.0, 1.0, 0.0]
    lm[3] = [0.0, 2.0, 0.0]

    # Index: 5 -> 6 -> 7
    lm[5] = [1.0, 0.0, 0.0]
    lm[6] = [1.0, 1.0, 0.0]
    lm[7] = [1.0, 2.0, 0.0]

    # Middle: 9 -> 10 -> 11
    lm[9] = [2.0, 0.0, 0.0]
    lm[10] = [2.0, 1.0, 0.0]
    lm[11] = [2.0, 2.0, 0.0]

    # Ring: 13 -> 14 -> 15
    lm[13] = [3.0, 0.0, 0.0]
    lm[14] = [3.0, 1.0, 0.0]
    lm[15] = [3.0, 2.0, 0.0]

    # Pinky: 17 -> 18 -> 19
    lm[17] = [4.0, 0.0, 0.0]
    lm[18] = [4.0, 1.0, 0.0]
    lm[19] = [4.0, 2.0, 0.0]

    angles = compute_joint_flexion_angles(lm)
    assert angles.shape == (5,)
    for k in range(5):
        assert abs(angles[k] - 0.0) < 1e-4, f"Finger {k} expected 0.0, got {angles[k]}"


def test_fully_bent_finger_angles():
    """Ngon tay gap sat 180 do -> Goc uon (Flexion Score) phai bang 1.0."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Index gap nguoc lai: LM5=[1, 0, 0], LM6=[1, 1, 0], LM7=[1, 0, 0]
    lm[5] = [1.0, 0.0, 0.0]
    lm[6] = [1.0, 1.0, 0.0]
    lm[7] = [1.0, 0.0, 0.0]

    angles = compute_joint_flexion_angles(lm)
    assert abs(angles[1] - 1.0) < 1e-4, f"Index finger expected 1.0, got {angles[1]}"


def test_angles_3d_rotation_invariance():
    """Xoay toan bo ban tay trong khong gian 3D -> Goc uon khong thay doi."""
    np.random.seed(42)
    lm = np.random.randn(21, 3).astype(np.float32)
    angles_orig = compute_joint_flexion_angles(lm)

    # Ma tran xoay Euler 3D (yaw, pitch, roll)
    alpha = np.radians(35)
    beta = np.radians(50)
    gamma = np.radians(70)

    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha),  np.cos(alpha), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ], dtype=np.float32)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma),  np.cos(gamma)]
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx
    lm_rotated = lm @ R.T

    angles_rot = compute_joint_flexion_angles(lm_rotated)
    np.testing.assert_allclose(angles_orig, angles_rot, atol=1e-4)


def test_resample_9d_shape_and_bounds():
    """Noi suy chuoi 9D tu T frames bat ky ve chuan K=32 frames."""
    np.random.seed(123)
    T = 24
    raw_frames_9d = np.random.randn(T, 9).astype(np.float32)
    shapes = ["A" if i < 12 else "B" for i in range(T)]

    res_traj, res_shapes = resample_9d_gesture_sequence(raw_frames_9d, shapes, k_target=32)

    assert res_traj.shape == (32, 9)
    assert len(res_shapes) == 32
    assert all(s in ["A", "B"] for s in res_shapes)
    assert not np.any(np.isnan(res_traj))
    assert not np.any(np.isinf(res_traj))


def test_golden_triad_dtw_distance():
    """Kiem tra khoang cach DTW 9D co rang buoc Sakoe-Chiba."""
    k = 32
    traj_q = np.ones((k, 9), dtype=np.float32)
    shapes_q = ["B"] * k

    # Trung khop hoan toan -> khoang cach = 0.0
    dist_zero = compute_golden_triad_dtw_distance(traj_q, shapes_q, traj_q, shapes_q)
    assert abs(dist_zero - 0.0) < 1e-4

    # Khac biet hoan toan -> khoang cach > 0
    traj_t = traj_q + 1.0
    shapes_t = ["V"] * k
    dist_diff = compute_golden_triad_dtw_distance(traj_q, shapes_q, traj_t, shapes_t)
    assert dist_diff > 0.5


def test_margin_guard_safe_division():
    """Margin guard voi can an toan khong bao gio gay loi ZeroDivisionError."""
    matcher = GoldenTriadDTWMatcher()
    ratio = matcher.compute_margin_ratio(d_top1=0.0, d_top2=0.5)
    assert ratio == 0.5 / 1e-4  # 5000.0


def test_feature_extraction_performance():
    """Do thoi gian tinh toan dac trung 9D < 0.30 ms tren CPU (du suc dat >3000 FPS)."""
    lm = np.random.randn(21, 3).astype(np.float32)
    N = 1000
    start = time.perf_counter()
    for _ in range(N):
        _ = compute_joint_flexion_angles(lm)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / N
    assert elapsed_ms < 0.30, f"Qua cham: {elapsed_ms:.4f} ms/frame"


def test_palm_normal_vector():
    """Kiem tra vector phap tuyen long ban tay 3D."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # LM0 (Wrist) = (0, 0, 0)
    # LM5 (Index MCP) = (0, 1, 0)  -> u = (0, 1, 0)
    # LM17 (Pinky MCP) = (1, 0, 0) -> v = (1, 0, 0)
    # u x v = (0*0 - 0*0, 0*1 - 0*0, 0*0 - 1*1) = (0, 0, -1)
    lm[5] = [0.0, 1.0, 0.0]
    lm[17] = [1.0, 0.0, 0.0]

    n_right = compute_palm_normal_vector(lm, handedness="Right")
    assert abs(np.linalg.norm(n_right) - 1.0) < 1e-4
    assert abs(n_right[0] - 0.0) < 1e-4
    assert abs(n_right[1] - 0.0) < 1e-4
    assert abs(n_right[2] - (-1.0)) < 1e-4

    # Handedness Left -> dao chieu n
    n_left = compute_palm_normal_vector(lm, handedness="Left")
    assert abs(n_left[2] - 1.0) < 1e-4


def test_12d_resample_and_dtw():
    """Kiem tra noi suy va so khop 12D voi khoang cach Cosine."""
    T = 20
    raw_12d = np.random.randn(T, 12).astype(np.float32)
    # 3 cot cuoi la normal vector can norm = 1
    for t in range(T):
        raw_12d[t, 9:12] /= np.linalg.norm(raw_12d[t, 9:12])
    shapes = ["B"] * T

    res_traj, res_shapes = resample_9d_gesture_sequence(raw_12d, shapes, k_target=32)
    assert res_traj.shape == (32, 12)
    assert len(res_shapes) == 32
    # Norm cua 3 cot cuoi phai luon la 1.0
    for i in range(32):
        assert abs(np.linalg.norm(res_traj[i, 9:12]) - 1.0) < 1e-3

    # So khop chinh no -> dist = 0
    d_same = compute_golden_triad_dtw_distance(res_traj, res_shapes, res_traj, res_shapes)
    assert abs(d_same - 0.0) < 1e-4


def test_sign_weights_integrity():
    """Kiem tra tinh toan ven toan hoc cua sign_weights_asllex.json."""
    import json
    with open("ASL_ALEX_2.0/sign_weights_asllex.json", "r", encoding="utf-8") as f:
        weights_data = json.load(f)

    expected_signs = [
        "HELLO", "BYE", "ME", "YOU", "THANK_YOU",
        "GO", "COME", "WANT", "MORE", "WATER",
        "HELP", "GOOD", "NAME", "DRINK", "STOP"
    ]
    assert len(weights_data) >= 15
    for sign_key in expected_signs:
        assert sign_key in weights_data, f"Thieu key: {sign_key}"
        w = weights_data[sign_key]
        for comp in ["P", "A", "JM", "N"]:
            assert comp in w, f"Thieu component {comp} trong {sign_key}"
            assert isinstance(w[comp], (int, float))
            assert w[comp] >= 0.0, f"Trong so am o {sign_key}: {w[comp]}"

        w_sum = w["P"] + w["A"] + w["JM"] + w["N"]
        assert abs(w_sum - 1.0) < 1e-4, f"Tong trong so {sign_key} != 1.0 (got {w_sum})"


def test_canonical_key_consistency():
    """Kiem tra 100% nhat quan giua ASL_VOCAB_MAP, dynamic_templates va sign_weights."""
    import json
    from predict_sentence_webcam import ASL_VOCAB_MAP

    with open("ASL_ALEX_2.0/sign_weights_asllex.json", "r", encoding="utf-8") as f:
        weights_data = json.load(f)
    with open("dynamic_templates.json", "r", encoding="utf-8") as f:
        template_data = json.load(f)

    weight_keys = set(weights_data.keys())
    template_keys = set(template_data["vocabulary"].keys())

    # Kiem tra tat ca 15 tu deu nam trong ca 3 nguon
    expected_signs = {
        "HELLO", "BYE", "ME", "YOU", "THANK_YOU",
        "GO", "COME", "WANT", "MORE", "WATER",
        "HELP", "GOOD", "NAME", "DRINK", "STOP"
    }

    assert expected_signs.issubset(weight_keys), f"Weight thieu tu: {expected_signs - weight_keys}"
    assert expected_signs.issubset(template_keys), f"Template thieu tu: {expected_signs - template_keys}"
    for sign in expected_signs:
        assert sign in ASL_VOCAB_MAP, f"ASL_VOCAB_MAP thieu tu: {sign}"

    # Dac biet: khong duoc xuat hien cac bien the gay mismatch
    for forbidden in ["GOOD-BYE", "GOOD_BYE", "bye"]:
        assert forbidden not in weight_keys
        assert forbidden not in template_keys


def test_anti_silent_failure_missing_key(tmp_path):
    """Matcher bat buoc nem loi KeyError khi tu dien chua tu khong co trong sign_weights o che do per-sign."""
    import json
    bad_template = {
        "schema_version": "1.0",
        "metadata": {"feature_dim": 12},
        "vocabulary": {
            "HELLO": {"display_name": "Hello", "threshold": 1.25, "num_samples": 0, "samples": []},
            "UNKNOWN_SIGN": {"display_name": "Unknown", "threshold": 1.25, "num_samples": 0, "samples": []}
        }
    }
    temp_file = tmp_path / "bad_templates.json"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(bad_template, f)

    with pytest.raises(KeyError) as exc_info:
        _ = GoldenTriadDTWMatcher(
            templates_path=str(temp_file),
            sign_weights_path="ASL_ALEX_2.0/sign_weights_asllex.json",
            strict_weights=True
        )

    assert "UNKNOWN_SIGN" in str(exc_info.value)


def test_per_sign_weight_differentiation():
    """Kiem tra su khac biet ro ret ve khoang cach DTW khi thay doi vector trong so W_s."""
    matcher = GoldenTriadDTWMatcher(
        sign_weights_path="ASL_ALEX_2.0/sign_weights_asllex.json",
        strict_weights=True
    )

    k = 32
    traj_q = np.zeros((k, 12), dtype=np.float32)
    traj_t = np.zeros((k, 12), dtype=np.float32)

    # Cho lech o vi tri co tay P (kenh 0..1)
    traj_t[:, 0:2] = 2.0
    shapes = ["B"] * k

    # Trong so ME uu tien P hon WANT khi lech o P
    wp_me, wa_me, wm_me, wn_me = matcher._get_weights_for_sign("ME")
    wp_want, wa_want, wm_want, wn_want = matcher._get_weights_for_sign("WANT")

    d_me = compute_golden_triad_dtw_distance(traj_q, shapes, traj_t, shapes, w_p=wp_me, w_a=wa_me, w_m=wm_me, w_n=wn_me)
    d_want = compute_golden_triad_dtw_distance(traj_q, shapes, traj_t, shapes, w_p=wp_want, w_a=wa_want, w_m=wm_want, w_n=wn_want)

    assert d_me > d_want * 2.0, f"ME ({d_me}) phai lon hon WANT ({d_want}) it nhat 2 lan vi lech o P!"


def test_global_weights_mode_when_none():
    """Kiem tra che do Global Weights khi sign_weights_path='': khong bao gio nem loi va tra ve config weights."""
    matcher = GoldenTriadDTWMatcher(
        sign_weights_path="",
        w_p=0.4,
        w_a=0.8,
        w_m=0.5,
        w_n=0.5,
        strict_weights=True
    )
    assert matcher.sign_weights == {}
    weights = matcher._get_weights_for_sign("ANY_SIGN")
    assert weights == (0.4, 0.8, 0.5, 0.5)


def test_template_migration_preservation():
    """Kiem tra 34 mau 12D ban dau khong bi mat sau qua trinh di tru sang 15 Canonical Keys."""
    import json
    with open("dynamic_templates.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocabulary"]
    total_samples = sum(v["num_samples"] for v in vocab.values())
    assert total_samples >= 34, f"Tong so mau phai it nhat la 34, thuc te: {total_samples}"

    # 5 tu goc phai co it nhat so mau ban dau
    expected_min_counts = {
        "HELLO": 6,
        "BYE": 7,
        "ME": 5,
        "YOU": 11,
        "THANK_YOU": 5
    }
    for sign, min_count in expected_min_counts.items():
        assert vocab[sign]["num_samples"] >= min_count, f"{sign} expected at least {min_count}, got {vocab[sign]['num_samples']}"
        assert len(vocab[sign]["samples"]) >= min_count
        for sample in vocab[sign]["samples"]:
            raw_traj = sample.get("traj_features_9d", sample.get("traj_features", []))
            assert len(raw_traj) == 32
            assert len(raw_traj[0]) == 12, "Mau phai la 12D [P + A + JM + N]"


def test_compute_dominant_shape():
    """Kiem tra ham tinh dominant shape theo majority vote va xu ly Unknown."""
    from dynamic_dtw_engine import _compute_dominant_shape

    assert _compute_dominant_shape(["A", "A", "B", "Unknown"]) == "A"
    assert _compute_dominant_shape(["V", "B", "V", "Unknown", "V"]) == "V"
    assert _compute_dominant_shape(["Unknown"] * 10) == "Unknown"
    assert _compute_dominant_shape([]) == "Unknown"


def test_global_dominant_shape_dtw_penalty():
    """Kiem tra hinh phat lambda_shape_global 1 lan cho ca chuoi."""
    k = 32
    traj = np.zeros((k, 9), dtype=np.float32)
    shapes_a = ["A"] * k
    shapes_b = ["B"] * k
    shapes_unk = ["Unknown"] * k

    # 1. Hai mau khac dominant shape -> tang dung lambda_shape_global (+0.35)
    d_off = compute_golden_triad_dtw_distance(traj, shapes_a, traj, shapes_b, lambda_shape=0.0, lambda_shape_global=0.0)
    d_on = compute_golden_triad_dtw_distance(traj, shapes_a, traj, shapes_b, lambda_shape=0.0, lambda_shape_global=0.35)
    assert abs((d_on - d_off) - 0.35) < 1e-4, f"Expected +0.35 penalty, got {d_on - d_off}"

    # 2. Hai mau cung dominant shape -> khoang cach khong doi
    d_same_off = compute_golden_triad_dtw_distance(traj, shapes_a, traj, shapes_a, lambda_shape=0.0, lambda_shape_global=0.0)
    d_same_on = compute_golden_triad_dtw_distance(traj, shapes_a, traj, shapes_a, lambda_shape=0.0, lambda_shape_global=0.35)
    assert abs(d_same_on - d_same_off) < 1e-4

    # 3. Mau chua Unknown -> khong phat neu 1 ben la Unknown
    d_unk = compute_golden_triad_dtw_distance(traj, shapes_a, traj, shapes_unk, lambda_shape=0.0, lambda_shape_global=0.35)
    assert abs(d_unk - d_off) < 1e-4


def test_pair_weight_override_father():
    """Xac nhan w_p duoc nhan 1.3 lan cho FATHER boi pair_weight_overrides."""
    matcher_default = GoldenTriadDTWMatcher(strict_weights=False)
    wp_def, wa_def, wm_def, wn_def = matcher_default._get_weights_for_sign("FATHER")

    matcher_no_override = GoldenTriadDTWMatcher(pair_weight_overrides={}, strict_weights=False)
    wp_raw, wa_raw, wm_raw, wn_raw = matcher_no_override._get_weights_for_sign("FATHER")

    assert abs(wp_def - wp_raw * 1.3) < 1e-4, f"Expected {wp_raw * 1.3}, got {wp_def}"
    assert abs(wa_def - wa_raw) < 1e-4
    assert abs(wm_def - wm_raw) < 1e-4
    assert abs(wn_def - wn_raw) < 1e-4

