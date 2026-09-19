"""
AUDIT SIGN QUALITY - CONG CU CHUAN HOA KIEM DINH CHAT LUONG MAU CU CHI DYNAMIC DTW
===================================================================================
Dung de kiem tra nhanh sau khi thu thap/cat xong mot tu vung moi:
1. Dong hoc: Do dich chuyen (dx, dy), Chieu dai quy dao (Path), Goc gap 5 ngon [T, I, M, R, P], Phap tuyen (nx, ny, nz).
2. Tu dong canh bao: Ky tay trai (nx < 0), Quy dao nguoc huong, Dang tay bat thuong.
3. Ma tran khoang cach DTW noi cum (Intra-cluster matrix).
4. Kiem dinh Leave-One-Out (LOO-CV): Kiem tra tung mau khi loai tru co bi nham sang tu khac khong.
5. Kiem dinh thuc te: GoldenTriadDTWMatcher.match() xem Top-1 va khoang cach an toan (margin).

Cach dung:
  py audit_sign_quality.py --word TOMORROW
  py audit_sign_quality.py --word YOU
  py audit_sign_quality.py --all
"""

import os
import sys
import json
import argparse
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dynamic_dtw_engine import GoldenTriadDTWMatcher, compute_golden_triad_dtw_distance, canonicalize_chirality


def load_sample_traj(sample) -> "np.ndarray":
    """
    Đọc vector đặc trưng của 1 mẫu và CHUẨN HÓA CHIRALITY bằng CÙNG hàm mà
    GoldenTriadDTWMatcher.load_templates() dùng, để mọi con số audit nằm trong ĐÚNG
    không gian dấu mà production so khớp (xem canonicalize_chirality).
    """
    traj_raw = sample.get("traj_features_9d")
    if traj_raw is None:
        traj_raw = sample.get("traj_features", [])
    traj = np.array(traj_raw, dtype=np.float32)
    if not sample.get("chirality_canonical", False):
        traj, _ = canonicalize_chirality(traj, handedness=sample.get("handedness"))
    return traj


def audit_single_word(matcher: GoldenTriadDTWMatcher, vocab: dict, word_key: str):
    word_key = word_key.strip().upper()
    if word_key not in vocab:
        print(f"[!] Tu '{word_key}' khong co trong tu dien template!")
        return

    word_entry = vocab[word_key]
    samples = word_entry.get("samples", [])
    n = len(samples)
    if n == 0:
        print(f"[!] Tu '{word_key}' khong co mau nao!")
        return

    print("=" * 95)
    print(f"         BAO CAO KIEM DINH CHAT LUONG CU CHI: '{word_key}' ({n} MAU)")
    print("=" * 95)

    # 1. Trong so ASL-LEX
    wp, wa, wm, wn = matcher._get_weights_for_sign(word_key)
    print(f"\n[1] Trong so ASL-LEX 2.0: w_pos={wp:.3f}, w_angles={wa:.3f}, w_motion={wm:.3f}, w_normal={wn:.3f}")

    # 2. Dong hoc & Canh bao bat thuong
    print(f"\n[2] Dong hoc chi tiet & Phat hien bat thuong tung mau:")
    warnings = []
    for i, s in enumerate(samples):
        traj = load_sample_traj(s)
        disp = traj[-1, :2] - traj[0, :2]
        path_len = np.sum(np.sqrt(np.diff(traj[:, 0])**2 + np.diff(traj[:, 1])**2))
        dur = s.get("duration_sec", 0.0)
        dropout = s.get("dropout_ratio", 0.0) * 100.0
        max_gap = s.get("max_gap_frames", 0)

        angles_mean = np.mean(traj[:, 4:9], axis=0)
        normal_mean = np.mean(traj[:, 9:12], axis=0)

        ang_str = f"T={angles_mean[0]:.2f}, I={angles_mean[1]:.2f}, M={angles_mean[2]:.2f}, R={angles_mean[3]:.2f}, P={angles_mean[4]:.2f}"
        norm_str = f"[{normal_mean[0]:+.2f}, {normal_mean[1]:+.2f}, {normal_mean[2]:+.2f}]"
        src = s.get("source_video", "N/A")

        sample_warns = []
        if normal_mean[0] < -0.30:
            sample_warns.append("❌ TAY TRAI (nx < 0)")
        if dropout > 0.0:
            sample_warns.append(f"⚠️ Dropout {dropout:.1f}%")
        if max_gap > 0:
            sample_warns.append(f"⚠️ Gap {max_gap}f")
        if dur < 0.25:
            sample_warns.append("⚠️ Qua ngan (<0.25s)")
        elif dur > 1.80:
            sample_warns.append("⚠️ Qua dai (>1.8s)")

        warn_txt = " | " + ", ".join(sample_warns) if sample_warns else " | ✅ Tot"
        if sample_warns:
            warnings.append((i, sample_warns, src))

        print(f"  #{i:<2}: dur={dur:.2f}s | Disp=({disp[0]:+.2f}, {disp[1]:+.2f}) | Path={path_len:.2f} | Flex: {ang_str} | Normal: {norm_str}{warn_txt}")

    # 3. Ma tran DTW noi cum
    print(f"\n[3] Ma tran khoang cach DTW noi cum ({word_key}):")
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        q_traj = load_sample_traj(samples[i])
        q_shapes = samples[i].get("hand_shapes", ["Unknown"] * len(q_traj))
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = 0.0
            elif i < j:
                t_traj = load_sample_traj(samples[j])
                t_shapes = samples[j].get("hand_shapes", ["Unknown"] * len(t_traj))
                d = compute_golden_triad_dtw_distance(
                    q_traj=q_traj, q_shapes=q_shapes, t_traj=t_traj, t_shapes=t_shapes,
                    w_p=wp, w_a=wa, w_m=wm, w_n=wn,
                    lambda_shape=matcher.lambda_shape,
                    lambda_shape_global=matcher.lambda_shape_global
                )
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    header = "       " + " ".join([f"#{j:<6}" for j in range(n)]) + "   Mean_dist"
    print(header)
    for i in range(n):
        row_str = f"  #{i:<3} " + " ".join([f"{dist_matrix[i, j]:6.3f}" for j in range(n)])
        mean_d = (dist_matrix[i].sum()) / (n - 1) if n > 1 else 0.0
        print(f"{row_str}  -> {mean_d:.4f}")

    if n > 1:
        intra_dists = dist_matrix[np.triu_indices(n, k=1)]
        print(f"\n  Tong ket noi cum: TB={intra_dists.mean():.4f} | Min={intra_dists.min():.4f} | Max={intra_dists.max():.4f}")

    # 4. Kiem dinh Leave-One-Out (LOO-CV)
    print(f"\n[4] Kiem dinh Leave-One-Out (LOO-CV) so voi toan bo tu vung khac:")
    pass_count = 0
    for i, s in enumerate(samples):
        q_traj = load_sample_traj(s)
        q_shapes = s.get("hand_shapes", ["Unknown"] * len(q_traj))

        own_dists = [dist_matrix[i, j] for j in range(n) if j != i]
        min_own = min(own_dists) if own_dists else 0.0

        best_other_word = None
        min_other = 999.0

        for other_key, entry in vocab.items():
            if other_key == word_key:
                continue
            wp_o, wa_o, wm_o, wn_o = matcher._get_weights_for_sign(other_key)
            for cand in entry.get("samples", []):
                t_traj = load_sample_traj(cand)
                t_shapes = cand.get("hand_shapes", ["Unknown"] * len(t_traj))
                d = compute_golden_triad_dtw_distance(
                    q_traj=q_traj, q_shapes=q_shapes, t_traj=t_traj, t_shapes=t_shapes,
                    w_p=wp_o, w_a=wa_o, w_m=wm_o, w_n=wn_o,
                    lambda_shape=matcher.lambda_shape,
                    lambda_shape_global=matcher.lambda_shape_global
                )
                if d < min_other:
                    min_other = d
                    best_other_word = other_key

        margin = min_other - min_own
        status = "PASS" if margin > 0 else "FAIL"
        if margin > 0:
            pass_count += 1
        src = s.get("source_video", "")
        print(f"  #{i:<2} {src:<26}: own_min={min_own:.3f} | other_min={min_other:.3f} ({best_other_word}) | Margin={margin:+.3f} [{status}]")

    pct = (pass_count / n * 100.0)
    print(f"\n  Do chinh xac LOO-CV: {pass_count}/{n} ({pct:.1f}%)")

    # 5. Nhan dien thuc te qua GoldenTriadDTWMatcher.match()
    print(f"\n[5] Thu nghiem nhan dien thoi gian thuc (GoldenTriadDTWMatcher.match):")
    for i, s in enumerate(samples):
        q_traj = load_sample_traj(s)
        q_shapes = s.get("hand_shapes", ["Unknown"] * len(q_traj))
        res = matcher.match(q_traj, q_shapes)
        top1 = res.candidates[0] if res.candidates else None
        top2 = res.candidates[1] if len(res.candidates) > 1 else None
        top_name = top1.word_key if top1 else "None"
        top_dist = top1.distance if top1 else 999.0
        sec_name = top2.word_key if top2 else "None"
        sec_dist = top2.distance if top2 else 999.0
        margin_sec = sec_dist - top_dist
        print(f"  #{i:<2}: Top1={top_name:<10} (d={top_dist:.3f}) | Top2={sec_name:<10} (d={sec_dist:.3f}) | Margin={margin_sec:+.3f} | Trang thai: {res.status}")

    # 6. Ket luan & Goi y hanh dong
    print(f"\n[6] KET LUAN & GOI Y HANH DONG:")
    if not warnings and pct == 100.0:
        print(f"  ✅ TAT CA {n} MAU CUA '{word_key}' DEU HOAN HAO! Khong can xoa mau nao.")
    else:
        if warnings:
            print(f"  ⚠️ CAC MAU CAN XEM XET XOA BO:")
            for idx, w_list, src in warnings:
                print(f"     - Mau #{idx} ({src}): {', '.join(w_list)}")
            print(f"\n  Goi y lenh xoa (xoa tu chi so lon xuong nho):")
            for idx, _, _ in sorted(warnings, key=lambda x: x[0], reverse=True):
                print(f"     py manage_asl_citizen_samples.py --word {word_key} --delete {idx}")
    print("=" * 95)


def audit_all_words(matcher: GoldenTriadDTWMatcher, vocab: dict):
    print("=" * 80)
    print(f"  KIEM DINH HOI QUY TOAN BO HE THONG ({len(vocab)} TU VUNG)")
    print("=" * 80)

    total_samples = 0
    correct_samples = 0
    per_word_stats = {}
    confusion_map = {}  # {word: {pred_word: count}}
    words_list = list(vocab.keys())

    for word_key, entry in vocab.items():
        s_list = entry.get("samples", [])
        w_correct = 0
        confusion_map[word_key] = {}
        for s in s_list:
            total_samples += 1
            q_traj = load_sample_traj(s)
            q_shapes = s.get("hand_shapes", ["Unknown"] * len(q_traj))
            res = matcher.match(q_traj, q_shapes)
            pred_word = res.candidates[0].word_key if res.candidates else "REJECTED"
            confusion_map[word_key][pred_word] = confusion_map[word_key].get(pred_word, 0) + 1
            if pred_word == word_key:
                correct_samples += 1
                w_correct += 1
        per_word_stats[word_key] = (w_correct, len(s_list))

    print(f"\n{'Tu vung':<14}{'So mau':>10}{'Dung Top-1':>14}{'Ty le':>12}{'Trang thai':>14}")
    print("-" * 65)
    for word_key, (c, tot) in per_word_stats.items():
        pct = (c / tot * 100.0) if tot > 0 else 0.0
        status = "✅ OK" if c == tot else "❌ VA CHAM"
        print(f"{word_key:<14}{tot:>10}{c:>14}{pct:>11.1f}%{status:>14}")
    print("-" * 65)
    total_pct = (correct_samples / total_samples * 100.0) if total_samples > 0 else 0.0
    print(f"TONG CONG: {correct_samples}/{total_samples} mau dat chuan ({total_pct:.1f}%)\n")

    # 2. Chi tiết các trường hợp va chạm nhầm lẫn (nếu có)
    has_collisions = False
    print("-" * 80)
    print("  CHI TIET VA CHAM (CONFUSION / COLLISION ANALYSIS):")
    print("-" * 80)
    for word_key, preds in confusion_map.items():
        for p_word, count in preds.items():
            if p_word != word_key:
                has_collisions = True
                print(f"  ❌ Tu '{word_key}' co {count} mau bi he thong doan nham sang '{p_word}'!")
    if not has_collisions:
        print("  ✅ Khong co mau nao bi nhan nham sang tu khac (Zero Collision)!")

    # 3. Đo khoảng cách phân biệt liên từ (Inter-word Distance & Safety Margin)
    if len(words_list) >= 2:
        print("\n" + "-" * 80)
        print("  DO KHOANG CACH PHAN BIET LIEN TU (INTER-WORD DISTANCE & SAFETY MARGIN):")
        print("-" * 80)
        close_pairs = []
        for i in range(len(words_list)):
            w1 = words_list[i]
            s1_list = vocab[w1].get("samples", [])
            wp1, wa1, wm1, wn1 = matcher._get_weights_for_sign(w1)
            for j in range(i + 1, len(words_list)):
                w2 = words_list[j]
                s2_list = vocab[w2].get("samples", [])
                if not s1_list or not s2_list:
                    continue
                wp2, wa2, wm2, wn2 = matcher._get_weights_for_sign(w2)

                # Tinh khoang cach DTW 2 chieu doi xung giua cac mau cua w1 va w2
                d12, d21 = [], []
                for s1 in s1_list:
                    t1 = load_sample_traj(s1)
                    sh1 = s1.get("hand_shapes", ["Unknown"] * len(t1))
                    for s2 in s2_list:
                        t2 = load_sample_traj(s2)
                        sh2 = s2.get("hand_shapes", ["Unknown"] * len(t2))
                        d12.append(compute_golden_triad_dtw_distance(t1, sh1, t2, sh2, wp2, wa2, wm2, wn2))
                        d21.append(compute_golden_triad_dtw_distance(t2, sh2, t1, sh1, wp1, wa1, wm1, wn1))

                if d12 and d21:
                    avg_sym = (float(np.mean(d12)) + float(np.mean(d21))) / 2.0
                    close_pairs.append((w1, w2, avg_sym))

        if close_pairs:
            for w1, w2, d in sorted(close_pairs, key=lambda x: x[2]):
                if d < 0.45:
                    print(f"  ❌ NGUY CO VA CHAM: ({w1} <-> {w2}): Khoang cach DTW = {d:.3f} (Qua sat nhau, can thu ky hon)")
                elif d < 0.65:
                    print(f"  ⚠️ CANH BAO NHE:    ({w1} <-> {w2}): Khoang cach DTW = {d:.3f} (Khoang cach trung binh)")
                else:
                    print(f"  ✅ PHAN BIET TOT:   ({w1} <-> {w2}): Khoang cach DTW = {d:.3f} (An toan, gap ~3-4x noi cum)")
    print("=" * 80 + "\n")


def main():
    default_templates = "webcam_templates.json" if os.path.exists("webcam_templates.json") else "asl_citizen_templates.json"
    parser = argparse.ArgumentParser(description="Audit chat luong mau cu chi Dynamic DTW")
    parser.add_argument("--word", type=str, default=None, help="Ten tu vung can kiem tra (vi du: HELLO, ME, YOU)")
    parser.add_argument("--all", action="store_true", help="Kiem tra hoi quy toan bo tat ca cac tu trong database")
    parser.add_argument("--templates", type=str, default=default_templates,
                        help=f"Duong dan file template JSON (Mac dinh: {default_templates})")
    args = parser.parse_args()

    if not os.path.exists(args.templates):
        print(f"[!] Khong tim thay file template: {args.templates}")
        sys.exit(1)

    print(f"[Audit] Su dung template bank: '{args.templates}'")
    matcher = GoldenTriadDTWMatcher(templates_path=args.templates, strict_weights=False)
    vocab = matcher.vocabulary

    if args.all:
        audit_all_words(matcher, vocab)
    elif args.word:
        audit_single_word(matcher, vocab, args.word)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
