"""
Kiểm chứng tự động cho Hybrid Blend DTW weights (Tests 1-4 theo đặc tả).
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dynamic_dtw_engine import GoldenTriadDTWMatcher

def run_tests():
    print("=" * 70)
    print("BẮT ĐẦU KIỂM CHỨNG HYBRID BLEND (TESTS 1 - 4)")
    print("=" * 70)

    # 1. Test tương thích ngược tuyệt đối (alpha=1.0, max_deviation=None)
    print("\n[TEST 1] Tương thích ngược tuyệt đối (default alpha=1.0, max_deviation=None):")
    matcher_default = GoldenTriadDTWMatcher(strict_weights=False)
    w_default_you = matcher_default._get_weights_for_sign("YOU")
    
    with open("ASL_ALEX_2.0/sign_weights_asllex.json", "r", encoding="utf-8") as f:
        sw_json = json.load(f)
    raw_you = sw_json["YOU"]
    expected_you = (raw_you["P"], raw_you["A"], raw_you["JM"], raw_you["N"])
    
    print(f"  Thực tế (_get_weights_for_sign): {w_default_you}")
    print(f"  Kỳ vọng (sign_weights_asllex.json): {expected_you}")
    diff = max(abs(a - b) for a, b in zip(w_default_you, expected_you))
    print(f"  Sai số tối đa: {diff:.8e}")
    assert diff < 1e-6, f"Test 1 FAILED: Sai số {diff} >= 1e-6"
    print("  ✓ PASS: Khớp 100% giá trị thô ASL-LEX tại alpha=1.0!")

    # 2. Test alpha=0.0 (hoàn toàn dùng mặc định __init__)
    print("\n[TEST 2] alpha=0.0 (chỉ dùng trọng số mặc định __init__):")
    w_p, w_a, w_m, w_n = 0.6, 1.0, 0.4, 0.8
    matcher_alpha0 = GoldenTriadDTWMatcher(
        w_p=w_p, w_a=w_a, w_m=w_m, w_n=w_n,
        asllex_blend_alpha=0.0,
        strict_weights=False
    )
    w_res0 = matcher_alpha0._get_weights_for_sign("YOU")
    expected0 = (w_p, w_a, w_m, w_n)
    print(f"  Thực tế (_get_weights_for_sign): {w_res0}")
    print(f"  Kỳ vọng (default base weights):   {expected0}")
    diff0 = max(abs(a - b) for a, b in zip(w_res0, expected0))
    print(f"  Sai số tối đa: {diff0:.8e}")
    assert diff0 < 1e-6, f"Test 2 FAILED: Sai số {diff0} >= 1e-6"
    print("  ✓ PASS: Khớp 100% trọng số mặc định gốc khi alpha=0.0!")

    # 3. Test alpha=0.5 (pha trộn tuyến tính)
    print("\n[TEST 3] alpha=0.5 (pha trộn tuyến tính):")
    matcher_alpha05 = GoldenTriadDTWMatcher(
        w_p=w_p, w_a=w_a, w_m=w_m, w_n=w_n,
        asllex_blend_alpha=0.5,
        strict_weights=False
    )
    w_res05 = matcher_alpha05._get_weights_for_sign("YOU")
    base_arr = np.array([w_p, w_a, w_m, w_n], dtype=np.float64)
    base_sum = float(base_arr.sum())
    asllex_arr = np.array(expected_you, dtype=np.float64)
    target_sum = 0.5 * base_sum + 0.5 * 1.0
    blend_norm = 0.5 * (base_arr / base_sum) + 0.5 * asllex_arr
    expected05 = blend_norm * target_sum

    sum_actual = sum(w_res05)
    print(f"  Thực tế (_get_weights_for_sign): {w_res05}")
    print(f"  Kỳ vọng công thức Option 2:       {tuple(expected05)}")
    print(f"  Tổng trọng số: {sum_actual:.6f} (Kỳ vọng: {target_sum:.6f}, nằm giữa 1.0 và base_sum={base_sum:.2f})")
    assert 1.0 <= sum_actual <= base_sum, f"Test 3 FAILED: Tổng {sum_actual} không nằm trong [1.0, {base_sum}]"
    diff05 = max(abs(a - b) for a, b in zip(w_res05, expected05))
    assert diff05 < 1e-6, f"Test 3 FAILED: Sai lệch công thức {diff05} >= 1e-6"
    print("  ✓ PASS: Pha trộn tuyến tính và thang đo Option 2 hoàn toàn chính xác!")

    # 4. Test max_deviation=0.0
    print("\n[TEST 4] max_deviation=0.0 (kẹp cứng 0% sai lệch so với mặc định):")
    matcher_dev0 = GoldenTriadDTWMatcher(
        w_p=w_p, w_a=w_a, w_m=w_m, w_n=w_n,
        asllex_blend_alpha=1.0,
        asllex_max_deviation=0.0,
        strict_weights=False
    )
    w_res_dev0 = matcher_dev0._get_weights_for_sign("YOU")
    print(f"  Thực tế (_get_weights_for_sign): {w_res_dev0}")
    print(f"  Kỳ vọng (kẹp cứng về base):      {expected0}")
    diff_dev0 = max(abs(a - b) for a, b in zip(w_res_dev0, expected0))
    print(f"  Sai số tối đa: {diff_dev0:.8e}")
    assert diff_dev0 < 1e-6, f"Test 4 FAILED: Sai số {diff_dev0} >= 1e-6"
    print("  ✓ PASS: Kẹp cứng biên 0% trả về đúng trọng số mặc định!")

    print("\n" + "=" * 70)
    print("TẤT CẢ CÁC BÀI TEST 1 - 4 ĐÃ PASS 100%!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
