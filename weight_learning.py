"""
Weight Learning Module for SPEC 11 (Task 3 & 4).
Learns optimal global DTW group weights under relative margin loss with regularization
and strict Leave-One-Session-Out (LOSO) nested cross-validation.
"""

import os
import sys
import copy
import time
import json
import csv
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize

from dynamic_dtw_engine import (
    GoldenTriadDTWMatcher,
    canonicalize_chirality,
    compute_golden_triad_dtw_distance
)
from scoreboard import (
    DataFold,
    QueryResult,
    QuerySample,
    build_matcher_from_config,
    evaluate_query,
    compute_metrics_and_confusion,
    paired_bootstrap_delta_margin,
    mcnemar_exact_test,
    load_bank,
    load_negative_strokes,
    split_dataset
)

# Production baseline weight vector w0 = [w_p, w_a, w_m, w_n, w_b, w_z, w_s, w_r]
W0_DEFAULT = np.array([0.6, 1.0, 0.4, 0.8, 0.8, 0.0, 0.0, 0.0], dtype=np.float64)
GROUP_NAMES = ["w_p", "w_a", "w_m", "w_n", "w_b", "w_z", "w_s", "w_r"]


_FAST_K = 32
_FAST_R = 4
_FAST_NB = 2 * _FAST_R + 1
_FAST_SQRT2 = np.float32(1.41421356)
_FAST_SQRT5 = np.float32(2.23606798)
_FAST_II = np.arange(_FAST_K)[:, None]
_FAST_OO = np.arange(_FAST_NB)[None, :]
_FAST_JJ = _FAST_II - _FAST_R + _FAST_OO
_FAST_VALID = (_FAST_JJ >= 0) & (_FAST_JJ < _FAST_K)
_FAST_JC = np.clip(_FAST_JJ, 0, _FAST_K - 1)
_FAST_PENALTY = np.where(_FAST_VALID, 0.0, np.inf).astype(np.float32)


def pair_group_costs(
    q_traj: np.ndarray,
    t_traj: np.ndarray,
    w: np.ndarray,
    query_has_body: bool = True,
    t_body_valid: bool = True,
    q_rep: Optional[int] = None,
    t_rep: Optional[int] = None,
    sakoe_chiba_radius: int = 4
) -> np.ndarray:
    """
    Phân rã chi phí từng nhóm dọc theo đường dóng DTW tối ưu dưới trọng số hiện tại w.
    Trả về vector S = [S_p, S_a, S_m, S_n, S_b, S_z, S_s, S_r].
    Tính chất cốt lõi: dot(w, S) == compute_golden_triad_dtw_distance(..., lambda_shape=0.0).
    """
    K, R, NB = _FAST_K, _FAST_R, _FAST_NB
    wp, wa, wm, wn, wb, wz, ws, wr = [float(x) for x in w]
    wb_eff = wb if (query_has_body and t_body_valid) else 0.0

    q = np.asarray(q_traj, dtype=np.float32)
    t = np.asarray(t_traj, dtype=np.float32)
    Tb = t[_FAST_JC, :]

    dpos = np.hypot(q[:, None, 0] - Tb[:, :, 0], q[:, None, 1] - Tb[:, :, 1]) / _FAST_SQRT2
    ddir = np.hypot(q[:, None, 2] - Tb[:, :, 2], q[:, None, 3] - Tb[:, :, 3]) / _FAST_SQRT2

    da_stat = np.sqrt(np.square(q[:, None, 4:9] - Tb[:, :, 4:9]).sum(2))
    da_dyn = np.sqrt(np.square(q[:, None, 12:17] - Tb[:, :, 12:17]).sum(2))
    dang = (da_stat + da_dyn) * (np.float32(0.5) / _FAST_SQRT5)

    cos = np.clip(np.einsum('ic,ioc->io', q[:, 9:12], Tb[:, :, 9:12]), -1.0, 1.0)
    dn = (np.float32(1.0) - cos) * np.float32(0.5)

    dz = np.abs(q[:, None, 17] - Tb[:, :, 17])
    dbody = np.hypot(q[:, None, 18] - Tb[:, :, 18], q[:, None, 19] - Tb[:, :, 19]) / _FAST_SQRT2 if (query_has_body and t_body_valid) else np.zeros((K, NB), dtype=np.float32)
    ds = np.sqrt(np.square(q[:, None, 20:24] - Tb[:, :, 20:24]).sum(2)) / np.float32(2.0)

    C = (
        wp * dpos +
        wm * ddir +
        wa * dang +
        wn * dn +
        wb_eff * dbody +
        wz * dz +
        ws * ds +
        _FAST_PENALTY
    )

    D = np.full((K + 1, NB + 1), np.inf, dtype=np.float32)
    ptr = np.zeros((K, NB), dtype=np.int8)

    D[0, R] = 0.0

    for i in range(K):
        prev_row = D[i]
        curr_row = D[i + 1]
        Ci = C[i]

        for o in range(NB):
            if not _FAST_VALID[i, o]:
                continue
            v_diag = prev_row[o]
            v_up = prev_row[o + 1] if o + 1 < NB else np.inf
            v_left = curr_row[o - 1] if o > 0 else np.inf

            if v_diag <= v_up and v_diag <= v_left:
                best_v = v_diag
                p = 0
            elif v_up <= v_left:
                best_v = v_up
                p = 1
            else:
                best_v = v_left
                p = 2

            curr_row[o] = Ci[o] + best_v
            ptr[i, o] = p

    i, o = K - 1, R
    path_io = []
    while i >= 0:
        path_io.append((i, o))
        if i == 0 and o == R:
            break
        p = ptr[i, o]
        if p == 0:
            i -= 1
        elif p == 1:
            i -= 1
            o += 1
        else:
            o -= 1

    rows = [p[0] for p in path_io]
    cols = [p[1] for p in path_io]

    S_p = float(np.sum(dpos[rows, cols])) / float(K)
    S_a = float(np.sum(dang[rows, cols])) / float(K)
    S_m = float(np.sum(ddir[rows, cols])) / float(K)
    S_n = float(np.sum(dn[rows, cols])) / float(K)
    S_b = float(np.sum(dbody[rows, cols])) / float(K)
    S_z = float(np.sum(dz[rows, cols])) / float(K)
    S_s = float(np.sum(ds[rows, cols])) / float(K)
    S_r = float(abs(int(q_rep) - int(t_rep))) if (q_rep is not None and t_rep is not None) else 0.0

    return np.array([S_p, S_a, S_m, S_n, S_b, S_z, S_s, S_r], dtype=np.float64)


def _process_query_chunk(chunk_data: Tuple[List[QuerySample], Dict[str, List[Dict[str, Any]]], np.ndarray]) -> List[Dict[str, Any]]:
    """Xử lý tính phân rã chi phí cho một tập query con (hỗ trợ đa tiến trình)."""
    queries_chunk, ref_by_word, w = chunk_data
    w = np.asarray(w, dtype=np.float64)
    results = []
    for q in queries_chunk:
        cand_costs = {}
        for c, t_list in ref_by_word.items():
            s_list = []
            for t in t_list:
                if t.get("session_id") == q.session_id:
                    continue
                t_traj = t.get("_traj_np")
                if t_traj is None:
                    t_traj = np.array(t.get("traj_features_9d") or t.get("traj_features"), dtype=np.float32)
                S = pair_group_costs(
                    q.traj, t_traj, w,
                    query_has_body=q.has_body,
                    t_body_valid=t.get("body_anchor_valid", True),
                    q_rep=q.rep_count,
                    t_rep=t.get("rep_count")
                )
                s_list.append(S)
            if s_list:
                cand_costs[c] = np.stack(s_list)
        results.append({
            "true_word": q.true_word,
            "session_id": q.session_id,
            "cand_costs": cand_costs
        })
    return results


def compute_all_pair_costs(
    train_queries: List[QuerySample],
    ref_templates_by_word: Dict[str, List[Dict[str, Any]]],
    w: np.ndarray,
    pool: Optional[ProcessPoolExecutor] = None
) -> List[Dict[str, Any]]:
    """
    Tính ma trận phân rã chi phí S(q, t) cho tất cả (query q, từ ứng viên c, template t).
    Tối ưu hóa: lưu mảng numpy _traj_np và hỗ trợ song song hóa qua pool nếu có.
    """
    # Đảm bảo mảng numpy đã được cache trên template
    for t_list in ref_templates_by_word.values():
        for t in t_list:
            if "_traj_np" not in t:
                t["_traj_np"] = np.array(t.get("traj_features_9d") or t.get("traj_features"), dtype=np.float32)

    if pool is not None and len(train_queries) >= 16:
        n_workers = pool._max_workers
        chunks = [train_queries[i::n_workers] for i in range(n_workers)]
        args_list = [(ch, ref_templates_by_word, w) for ch in chunks if ch]
        results_nested = list(pool.map(_process_query_chunk, args_list))
        merged = []
        for r_chunk in results_nested:
            merged.extend(r_chunk)
        return merged
    else:
        return _process_query_chunk((train_queries, ref_templates_by_word, w))


def loss_function(
    w: np.ndarray,
    pair_data: List[Dict[str, Any]],
    w0: np.ndarray,
    lam: float,
    tau: float = 0.25,
    class_counts: Optional[Dict[str, int]] = None,
    mode: str = "L8"
) -> float:
    """
    Hàm mất mát relative margin loss:
    L(w) = sum_q (1 / n_y_q) * sum_{c != y_q} max(0, tau - (D_c - D_y)/max(D_c, 1e-6)) + lam * ||w - w0||^2
    """
    w = np.asarray(w, dtype=np.float64)
    if mode == "L5":
        w_eff = np.copy(w)
        w_eff[5:] = 0.0
    else:
        w_eff = w

    if class_counts is None:
        class_counts = {}
        for pd in pair_data:
            yw = pd["true_word"]
            class_counts[yw] = class_counts.get(yw, 0) + 1

    total_loss = 0.0
    for pd in pair_data:
        yw = pd["true_word"]
        cand_costs = pd["cand_costs"]
        if yw not in cand_costs:
            continue

        S_y = cand_costs[yw]
        D_y = float(np.min(S_y @ w_eff))

        loss_q = 0.0
        for c, S_c in cand_costs.items():
            if c == yw:
                continue
            D_c = float(np.min(S_c @ w_eff))
            margin_c = (D_c - D_y) / max(D_c, 1e-6)
            viol = max(0.0, tau - margin_c)
            loss_q += viol

        n_y = max(1, class_counts.get(yw, 1))
        total_loss += (1.0 / float(n_y)) * loss_q

    reg = float(lam) * float(np.sum((w_eff - w0) ** 2))
    return float(total_loss + reg)


def solve_weights_fixed_paths(
    pair_data: List[Dict[str, Any]],
    w_init: np.ndarray,
    w0: np.ndarray,
    lam: float,
    mode: str = "L8",
    tau: float = 0.25
) -> np.ndarray:
    """
    Tối ưu hóa vector trọng số w với đường dóng cố định bằng thuật toán SLSQP.
    Ràng buộc: w >= 0, sum(w) == sum(w0).
    """
    sum_w0 = float(np.sum(w0))
    n_vars = len(w0)

    class_counts = {}
    for pd in pair_data:
        yw = pd["true_word"]
        class_counts[yw] = class_counts.get(yw, 0) + 1

    def obj(w_arr):
        return loss_function(w_arr, pair_data, w0, lam, tau=tau, class_counts=class_counts, mode=mode)

    bounds = [(0.0, None) for _ in range(n_vars)]
    if mode == "L5":
        bounds[5] = (0.0, 0.0)
        bounds[6] = (0.0, 0.0)
        bounds[7] = (0.0, 0.0)

    constraints = [
        {"type": "eq", "fun": lambda w_arr: np.sum(w_arr) - sum_w0}
    ]

    res = minimize(
        obj,
        w_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100, "ftol": 1e-6, "disp": False}
    )

    w_opt = np.array(res.x, dtype=np.float64)
    w_opt = np.maximum(w_opt, 0.0)
    if mode == "L5":
        w_opt[5:] = 0.0
    w_opt = w_opt * (sum_w0 / np.sum(w_opt))
    return w_opt


def learn_weights_alternating(
    train_queries: List[QuerySample],
    ref_templates_by_word: Dict[str, List[Dict[str, Any]]],
    w0: np.ndarray = W0_DEFAULT,
    lam: float = 0.3,
    mode: str = "L8",
    max_rounds: int = 5,
    tol: float = 1e-3,
    tau: float = 0.25,
    pool: Optional[ProcessPoolExecutor] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Thuật toán lặp luân phiên: tìm đường DTW và tối ưu hóa trọng số w.
    """
    w_cur = np.copy(w0)
    if mode == "L5":
        w_cur[5:] = 0.0
        w_cur = w_cur * (np.sum(w0) / np.sum(w_cur))

    loss_history = []

    for r in range(max_rounds):
        pair_data = compute_all_pair_costs(train_queries, ref_templates_by_word, w_cur, pool=pool)
        loss_r = loss_function(w_cur, pair_data, w0, lam, tau=tau, mode=mode)
        loss_history.append(loss_r)

        w_next = solve_weights_fixed_paths(pair_data, w_cur, w0, lam, mode=mode, tau=tau)

        max_dw = float(np.max(np.abs(w_next - w_cur)))
        w_cur = w_next
        if max_dw < tol:
            break

    pair_data = compute_all_pair_costs(train_queries, ref_templates_by_word, w_cur, pool=pool)
    final_loss = loss_function(w_cur, pair_data, w0, lam, tau=tau, mode=mode)
    loss_history.append(final_loss)

    return w_cur, loss_history


def create_loso_queries_and_refs_from_bank(
    bank_dict: Dict[str, Any]
) -> Tuple[List[QuerySample], Dict[str, List[Dict[str, Any]]]]:
    """Tạo danh sách QuerySample và từ điển template tham chiếu từ dữ liệu bank."""
    matcher = GoldenTriadDTWMatcher(templates_dict=bank_dict, asllex_mode="none")
    queries = []
    ref_by_word = {}

    for wk, wd in matcher.vocabulary.items():
        samples = wd.get("samples", [])
        ref_by_word[wk] = []
        for s_idx, s in enumerate(samples):
            traj_raw = np.array(s.get("traj_features_9d") or s.get("traj_features"), dtype=np.float32)
            s["_traj_np"] = traj_raw
            sid = s.get("sample_id") or f"{wk}__idx_{s_idx}"
            sess = s.get("session_id") or "UNKNOWN"
            has_body = s.get("body_anchor_valid", True)
            rep = s.get("rep_count")
            handed = s.get("handedness")

            q = QuerySample(
                query_id=str(sid),
                true_word=wk,
                sample_id=str(sid),
                session_id=sess,
                traj=traj_raw,
                hand_shapes=s.get("hand_shapes", ["Unknown"] * 32),
                handedness="Right" if s.get("chirality_canonical") else handed,
                has_body=has_body,
                rep_count=rep,
                is_original=True,
                augment_of=s.get("augment_of"),
                chirality_canonical=bool(s.get("chirality_canonical"))
            )
            queries.append(q)
            ref_by_word[wk].append(s)

    return queries, ref_by_word


def _log_progress(progress_file: str, msg: str) -> None:
    """Ghi log tiến trình ra màn hình và nối vào file progress.txt."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if progress_file:
        os.makedirs(os.path.dirname(os.path.abspath(progress_file)), exist_ok=True)
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_nested_loso_pipeline(
    bank_path: str,
    neg_path: Optional[str] = None,
    out_dir: str = "results",
    modes: List[str] = ["B0", "L5", "L8"],
    lambdas: List[float] = [0.0, 0.3, 3.0],
    random_seed: int = 20260919,
    max_workers: int = 6
) -> Dict[str, Any]:
    """
    Thực thi toàn diện quy trình Leave-One-Session-Out (LOSO) lồng nhau:
      - Chia 7 fold ngoài (outer folds).
      - Trong mỗi fold ngoài, dùng 6 fold trong (inner folds) để chọn siêu tham số lambda.
      - Huấn luyện trên tập ngoài với lambda tốt nhất.
      - Đánh giá trên fold ngoài và mẫu âm.
      - Xuất đầy đủ các file kết quả theo chuẩn SPEC 11.
    """
    os.makedirs(out_dir, exist_ok=True)
    progress_file = os.path.join(out_dir, "progress.txt")
    _log_progress(progress_file, f"Khởi động nested LOSO pipeline. Modes: {modes}, Lambdas: {lambdas}")

    bank_data, bank_hash = load_bank(bank_path)
    neg_strokes, neg_hash = load_negative_strokes(neg_path)

    split_mode, outer_folds = split_dataset(bank_data, random_seed=random_seed)
    assert split_mode == "loso", "Kiểu chia dữ liệu bắt buộc phải là LOSO"

    w0 = np.copy(W0_DEFAULT)
    pipeline_results: Dict[str, Any] = {}

    # Khởi tạo persistent ProcessPoolExecutor để tăng tốc tính ma trận chi phí
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for mode in modes:
            _log_progress(progress_file, f"================ BẮT ĐẦU CHẾ ĐỘ: {mode} ================")
            t_mode_start = time.perf_counter()

            outer_fold_weights = []
            outer_fold_losses = []
            outer_fold_lambdas = []
            all_query_results: List[QueryResult] = []
            all_latencies = []
            all_neg_latencies = []
            false_accept_neg_total = 0

            for fold_idx, fold in enumerate(outer_folds):
                f_name = fold.fold_name
                sess_test = f_name.replace("LOSO_", "")
                _log_progress(progress_file, f"--- Outer Fold [{fold_idx + 1}/{len(outer_folds)}]: {f_name} (Test: {len(fold.queries)} truy vấn) ---")

                ref_bank = fold.reference_bank
                train_queries, train_ref_by_word = create_loso_queries_and_refs_from_bank(ref_bank)

                if mode == "B0":
                    w_fold = np.copy(w0)
                    best_lam = 0.0
                    loss_hist = [0.0]
                elif mode.startswith("B1"):
                    w_fold = np.copy(w0)
                    best_lam = 0.0
                    loss_hist = [0.0]
                else:
                    _log_progress(progress_file, f"  Inner LOSO chọn siêu tham số lambda trong {lambdas}...")
                    inner_split_mode, inner_folds = split_dataset(ref_bank, random_seed=random_seed)
                    best_lam = lambdas[0]
                    best_inner_margin = -1e9

                    # Tiền trích xuất inner fold dữ liệu và chi phí ban đầu under w_init
                    w_init = np.copy(w0)
                    if mode == "L5":
                        w_init[5:] = 0.0
                        w_init = w_init * (np.sum(w0) / np.sum(w_init))

                    inner_fold_cached = []
                    for inf in inner_folds:
                        in_train_q, in_train_ref = create_loso_queries_and_refs_from_bank(inf.reference_bank)
                        pair_data_0 = compute_all_pair_costs(in_train_q, in_train_ref, w_init, pool=pool)
                        inner_fold_cached.append((inf, in_train_q, in_train_ref, pair_data_0))

                    for cand_lam in lambdas:
                        t_lam0 = time.perf_counter()
                        inner_margins = []
                        for inf, in_train_q, in_train_ref, pair_data_0 in inner_fold_cached:
                            # Vòng 1 giải w từ pair_data_0
                            w_1 = solve_weights_fixed_paths(pair_data_0, w_init, w0, lam=cand_lam, mode=mode)
                            # Vòng 2 cập nhật lại đường DTW dưới w_1
                            pair_data_1 = compute_all_pair_costs(in_train_q, in_train_ref, w_1, pool=pool)
                            w_inner = solve_weights_fixed_paths(pair_data_1, w_1, w0, lam=cand_lam, mode=mode)

                            cfg_in = {
                                "name": f"inner_{mode}",
                                "weights": {GROUP_NAMES[i]: float(w_inner[i]) for i in range(8)},
                                "asllex_mode": "none",
                                "flip_body_x": True,
                                "thresholds": "scale_k:1.75",
                                "matcher": {"margin_threshold": 0.12, "uncertainty_ratio": 1.40, "uncertainty_k": 2.35}
                            }
                            in_matcher = build_matcher_from_config(cfg_in, inf.reference_bank)
                            for in_q in inf.queries:
                                res = evaluate_query(in_matcher, in_q)
                                inner_margins.append(res.margin)

                        mean_in_margin = float(np.mean(inner_margins)) if inner_margins else 0.0
                        _log_progress(progress_file, f"    lambda={cand_lam:4.1f} -> Inner mean margin: {mean_in_margin:+.4f} (took {time.perf_counter() - t_lam0:.2f}s)")
                        if mean_in_margin > best_inner_margin:
                            best_inner_margin = mean_in_margin
                            best_lam = cand_lam

                    _log_progress(progress_file, f"  Fold {f_name}: Chọn lambda = {best_lam} (margin nội = {best_inner_margin:+.4f})")

                    # Học trên toàn bộ fold ngoài với lambda đã chọn
                    t_train0 = time.perf_counter()
                    w_fold, loss_hist = learn_weights_alternating(
                        train_queries, train_ref_by_word,
                        w0=w0, lam=best_lam, mode=mode, max_rounds=4, pool=pool
                    )
                    _log_progress(progress_file, f"  Huấn luyện fold ngoài xong trong {time.perf_counter() - t_train0:.2f}s, w={np.round(w_fold, 4).tolist()}")

                outer_fold_weights.append(w_fold)
                outer_fold_losses.append(loss_hist)
                outer_fold_lambdas.append(best_lam)

                # Dựng matcher đánh giá trên fold test ngoài
                cfg_fold = {
                    "name": f"outer_{mode}_{f_name}",
                    "weights": {GROUP_NAMES[i]: float(w_fold[i]) for i in range(8)},
                    "asllex_mode": "none",
                    "flip_body_x": True,
                    "thresholds": "scale_k:1.75",
                    "matcher": {"margin_threshold": 0.12, "uncertainty_ratio": 1.40, "uncertainty_k": 2.35}
                }
                test_matcher = build_matcher_from_config(cfg_fold, ref_bank)

                # Đánh giá các mẫu dương trong test fold
                for q in fold.queries:
                    assert q.session_id == sess_test, "Sai lệch session_id của query"
                    res = evaluate_query(test_matcher, q)
                    all_query_results.append(res)
                    all_latencies.append(res.latency_ms)

                # Đánh giá mẫu âm trên template bank ngoài
                if neg_strokes:
                    for s in neg_strokes:
                        raw_traj = np.array(s.get("traj_features_9d") or s.get("traj_features"), dtype=np.float32)
                        shapes = s.get("hand_shapes", ["Unknown"] * 32)
                        handed = "Right" if s.get("chirality_canonical") else s.get("handedness")
                        hb = s.get("body_anchor_valid", True)
                        rep = s.get("rep_count")
                        t0 = time.perf_counter()
                        mres = test_matcher.match(
                            query_traj_9d=raw_traj, query_shapes=shapes,
                            handedness=handed, query_has_body=hb, query_rep_count=rep
                        )
                        all_neg_latencies.append((time.perf_counter() - t0) * 1000.0)
                        if mres.status == "ACCEPTED":
                            false_accept_neg_total += 1

            # Tổng hợp chỉ số thống kê qua tất cả các fold
            w_matrix = np.stack(outer_fold_weights)  # (N_folds, 8)
            w_mean = np.mean(w_matrix, axis=0)
            w_std = np.std(w_matrix, axis=0)
            cv = np.where(w_mean > 1e-4, w_std / (w_mean + 1e-9), 0.0)
            unstable = bool(np.any(cv > 0.5))

            # SỬA THEO MỤC A.1: Thang mẫu âm = số mẫu âm * số fold
            total_neg_evals = (len(neg_strokes) * len(outer_folds)) if neg_strokes else 0
            metrics, confusion = compute_metrics_and_confusion(
                all_query_results,
                false_accept_neg_total,
                total_neg_evals,
                all_neg_latencies
            )
            metrics["false_accept_neg_display"] = f"{false_accept_neg_total}/{total_neg_evals}"
            metrics["weights_mean"] = [float(x) for x in w_mean]
            metrics["weights_std"] = [float(x) for x in w_std]
            metrics["weights_cv"] = [float(x) for x in cv]
            metrics["unstable"] = unstable
            metrics["selected_lambdas"] = outer_fold_lambdas
            metrics["mode"] = mode

            _log_progress(
                progress_file,
                f"Hoàn thành {mode} trong {time.perf_counter() - t_mode_start:.2f}s. "
                f"Top1={metrics['top1']*100:.2f}%, CleanAccept={metrics['clean_accept']*100:.2f}%, "
                f"WrongAccept={metrics['wrong_accept']*100:.2f}%, MarginMean={metrics['margin_mean']:+.4f}, "
                f"FalseAcceptNeg={metrics['false_accept_neg']*100:.2f}% ({metrics['false_accept_neg_display']}), "
                f"Unstable={unstable}"
            )

            # KIỂM TRA HỒI QUY KHUNG CHO B0 (MỤC A.4)
            if mode == "B0":
                top1 = metrics["top1"]
                clean_accept = metrics["clean_accept"]
                margin_mean = metrics["margin_mean"]
                _log_progress(
                    progress_file,
                    f"[B0 Regression Check] top1={top1:.4f} (chuẩn 1.0000), "
                    f"clean_accept={clean_accept:.4f} (chuẩn 0.8857), margin_mean={margin_mean:.4f} (chuẩn 0.5383)"
                )
                assert abs(top1 - 1.0) < 1e-4, f"B0 top1 mismatch: {top1}"
                assert abs(clean_accept - 0.8857) < 1e-3, f"B0 clean_accept mismatch: {clean_accept}"
                assert abs(margin_mean - 0.5383) < 1e-3, f"B0 margin_mean mismatch: {margin_mean}"

            pipeline_results[mode] = {
                "mode": mode,
                "metrics": metrics,
                "confusion": confusion,
                "query_results": all_query_results,
                "weights_by_fold": w_matrix.tolist(),
                "weights_mean": w_mean.tolist(),
                "weights_std": w_std.tolist(),
                "weights_cv": cv.tolist(),
                "unstable": unstable,
                "loss_histories": outer_fold_losses,
                "selected_lambdas": outer_fold_lambdas
            }

            # Lưu kết quả ra đĩa theo từng mode (Mục B)
            _save_mode_files(out_dir, mode, outer_folds, pipeline_results[mode])

    # Ghi file RERUN.txt
    rerun_cmd = (
        f"py weight_learning.py --bank {bank_path} --neg {neg_path} "
        f"--out {out_dir} --modes {' '.join(modes)} "
        f"--lambdas {' '.join(str(x) for x in lambdas)} --seed {random_seed}\n"
    )
    with open(os.path.join(out_dir, "RERUN.txt"), "w", encoding="utf-8") as f:
        f.write(rerun_cmd)

    # Đánh giá các cổng chấp nhận (Section C & D)
    _evaluate_gates_and_proposal(pipeline_results, bank_data, out_dir)

    _log_progress(progress_file, "Toàn bộ nested LOSO pipeline đã hoàn thành thành công.")
    return pipeline_results


def _save_mode_files(out_dir: str, mode: str, outer_folds: List[DataFold], mode_data: Dict[str, Any]) -> None:
    """Lưu các tệp kết quả metrics, per_query, confusion, weights_by_fold, loss_history."""
    metrics = mode_data["metrics"]
    query_results = mode_data["query_results"]
    confusion = mode_data["confusion"]
    weights_by_fold = mode_data["weights_by_fold"]
    loss_histories = mode_data["loss_histories"]
    selected_lambdas = mode_data["selected_lambdas"]

    # 1. metrics_<mode>.json
    with open(os.path.join(out_dir, f"metrics_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 2. per_query_<mode>.csv
    per_query_path = os.path.join(out_dir, f"per_query_{mode}.csv")
    with open(per_query_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "true_word", "pred_word", "status",
            "d_correct", "d_wrong", "margin", "latency_ms",
            "top1", "clean_accept", "wrong_accept"
        ])
        for r in query_results:
            writer.writerow([
                r.query_id, r.true_word, r.pred_word, r.status,
                f"{r.d_correct:.6f}", f"{r.d_wrong:.6f}", f"{r.margin:.6f}", f"{r.latency_ms:.3f}",
                int(r.top1_correct), int(r.clean_accept), int(r.wrong_accept)
            ])

    # 3. confusion_<mode>.csv
    all_cols = sorted(list({col for row in confusion.values() for col in row.keys()}))
    confusion_path = os.path.join(out_dir, f"confusion_{mode}.csv")
    with open(confusion_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_word"] + all_cols)
        for tw in sorted(confusion.keys()):
            row_counts = [confusion[tw].get(col, 0) for col in all_cols]
            writer.writerow([tw] + row_counts)

    # 4. weights_by_fold_<mode>.csv
    weights_path = os.path.join(out_dir, f"weights_by_fold_{mode}.csv")
    with open(weights_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "session", "lambda", "w_p", "w_a", "w_m", "w_n", "w_b", "w_z", "w_s", "w_r"])
        for idx, fold in enumerate(outer_folds):
            f_name = fold.fold_name
            sess = f_name.replace("LOSO_", "")
            lam = selected_lambdas[idx]
            w_row = [f"{x:.6f}" for x in weights_by_fold[idx]]
            writer.writerow([f_name, sess, lam] + w_row)

    # 5. loss_history_<mode>.csv
    loss_path = os.path.join(out_dir, f"loss_history_{mode}.csv")
    with open(loss_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "session", "round", "loss"])
        for idx, fold in enumerate(outer_folds):
            f_name = fold.fold_name
            sess = f_name.replace("LOSO_", "")
            for r_idx, loss_val in enumerate(loss_histories[idx]):
                writer.writerow([f_name, sess, r_idx, f"{loss_val:.6f}"])


def _evaluate_gates_and_proposal(
    pipeline_results: Dict[str, Any],
    bank_data: Dict[str, Any],
    out_dir: str
) -> Dict[str, Any]:
    """
    Đánh giá các cổng chấp nhận (Section C & D) và tạo proposed_weights.json.
    """
    progress_file = os.path.join(out_dir, "progress.txt")
    _log_progress(progress_file, "Đánh giá các cổng chấp nhận SPEC 11...")

    b0_res = pipeline_results.get("B0")
    if not b0_res:
        return {}

    b0_metrics = b0_res["metrics"]
    b0_queries = b0_res["query_results"]
    b0_margins = np.array([r.margin for r in b0_queries], dtype=np.float64)

    # Số lượng đúng theo từ ở B0
    b0_word_correct: Dict[str, int] = {}
    for r in b0_queries:
        if r.top1_correct:
            b0_word_correct[r.true_word] = b0_word_correct.get(r.true_word, 0) + 1

    comparison_results: Dict[str, Any] = {}

    for mode in ["L5", "L8"]:
        if mode not in pipeline_results:
            continue
        m_res = pipeline_results[mode]
        m_metrics = m_res["metrics"]
        m_queries = m_res["query_results"]
        m_margins = np.array([r.margin for r in m_queries], dtype=np.float64)

        # 1. Delta margin và Bootstrap KTC 95%
        delta_arr = m_margins - b0_margins
        d_mean, ci_lower, ci_upper, se = paired_bootstrap_delta_margin(delta_arr, n_resamples=5000, random_seed=20260919)
        gate_ci = bool(ci_lower > 0.0)

        # 2. wrong_accept và false_accept_neg không tăng
        gate_wrong = bool(m_metrics["wrong_accept"] <= b0_metrics["wrong_accept"])
        gate_fa_neg = bool(m_metrics["false_accept_neg"] <= b0_metrics["false_accept_neg"])

        # 3. Không từ nào mất quá 1 mẫu đúng
        m_word_correct: Dict[str, int] = {}
        for r in m_queries:
            if r.top1_correct:
                m_word_correct[r.true_word] = m_word_correct.get(r.true_word, 0) + 1
        word_retention_ok = True
        lost_words = {}
        for tw, c_b0 in b0_word_correct.items():
            c_m = m_word_correct.get(tw, 0)
            if c_b0 - c_m > 1:
                word_retention_ok = False
                lost_words[tw] = c_b0 - c_m
        gate_words = word_retention_ok

        # 4. Không bị cờ không ổn định
        gate_stability = bool(not m_res["unstable"])

        # 5. Clean accept (McNemar chính xác)
        b_clean = sum(1 for rb, rm in zip(b0_queries, m_queries) if not rb.clean_accept and rm.clean_accept)
        c_clean = sum(1 for rb, rm in zip(b0_queries, m_queries) if rb.clean_accept and not rm.clean_accept)
        p_mcnemar_clean = mcnemar_exact_test(b_clean, c_clean)

        # 6. Độ trễ: tương đối và tuyệt đối
        lat_base = b0_metrics["latency_ms"]
        lat_mode = m_metrics["latency_ms"]
        lat_rel_change = ((lat_mode - lat_base) / lat_base) if lat_base > 0 else 0.0
        gate_latency_rel = bool(lat_rel_change <= 0.20)
        gate_latency_abs = bool(lat_mode <= 10.0)

        passed_all = bool(gate_ci and gate_wrong and gate_fa_neg and gate_words and gate_stability and gate_latency_abs)

        comparison_results[mode] = {
            "mode": mode,
            "delta_margin_mean": d_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "gate_ci_lower_gt_0": gate_ci,
            "gate_wrong_accept_not_increased": gate_wrong,
            "gate_false_accept_neg_not_increased": gate_fa_neg,
            "gate_word_retention": gate_words,
            "lost_words": lost_words,
            "gate_stability": gate_stability,
            "cv_by_group": m_res["weights_cv"],
            "mcnemar_clean_p": p_mcnemar_clean,
            "mcnemar_b": b_clean,
            "mcnemar_c": c_clean,
            "latency_base_ms": lat_base,
            "latency_mode_ms": lat_mode,
            "latency_rel_change_pct": lat_rel_change * 100.0,
            "gate_latency_rel_under_20pct": gate_latency_rel,
            "gate_latency_abs_under_10ms": gate_latency_abs,
            "passed_all_gates": passed_all
        }

        _log_progress(
            progress_file,
            f"Cổng so sánh {mode} vs B0: DeltaMargin={d_mean:+.4f} (KTC95: [{ci_lower:+.4f}, {ci_upper:+.4f}] -> {gate_ci}), "
            f"WrongAcceptOk={gate_wrong}, FalseNegOk={gate_fa_neg}, WordsOk={gate_words}, StabilityOk={gate_stability}, "
            f"Latency={lat_mode:.2f}ms ({lat_rel_change*100:+.1f}%), Passed={passed_all}"
        )

    # Lựa chọn phương án
    chosen_mode = None
    if comparison_results.get("L5", {}).get("passed_all_gates") and comparison_results.get("L8", {}).get("passed_all_gates"):
        # Cả hai đều đạt: chọn L5 trừ khi L8 hơn rõ về delta margin
        d5 = comparison_results["L5"]["delta_margin_mean"]
        d8 = comparison_results["L8"]["delta_margin_mean"]
        chosen_mode = "L8" if (d8 - d5 > 0.02) else "L5"
    elif comparison_results.get("L5", {}).get("passed_all_gates"):
        chosen_mode = "L5"
    elif comparison_results.get("L8", {}).get("passed_all_gates"):
        chosen_mode = "L8"

    # Kiểm tra giả thuyết handshape (Mục 3.7) cho HELLO-BYE, YOU-YES, ME-HUNGRY
    handshape_pairs = [("HELLO", "BYE"), ("YOU", "YES"), ("ME", "HUNGRY")]
    handshape_analysis: Dict[str, Any] = {}
    for pair in handshape_pairs:
        w1, w2 = pair
        pair_key = f"{w1}_{w2}"
        b0_confused = sum(1 for r in b0_queries if (r.true_word == w1 and r.pred_word == w2) or (r.true_word == w2 and r.pred_word == w1))
        handshape_analysis[pair_key] = {"pair": pair, "b0_confusions": b0_confused}
        for mode in ["L5", "L8"]:
            if mode in pipeline_results:
                m_q = pipeline_results[mode]["query_results"]
                m_confused = sum(1 for r in m_q if (r.true_word == w1 and r.pred_word == w2) or (r.true_word == w2 and r.pred_word == w1))
                handshape_analysis[pair_key][f"{mode}_confusions"] = m_confused

    proposal_path = os.path.join(out_dir, "proposed_weights.json")

    if chosen_mode is not None:
        _log_progress(progress_file, f"Phương án qua cổng chấp nhận: {chosen_mode}")
        # Huấn luyện trọng số trên TOÀN BỘ bank với lambda được chọn nhiều nhất
        chosen_lambdas = pipeline_results[chosen_mode]["selected_lambdas"]
        vals, counts = np.unique(chosen_lambdas, return_counts=True)
        best_overall_lambda = float(vals[np.argmax(counts)])

        all_q, all_ref = create_loso_queries_and_refs_from_bank(bank_data)
        w_final, _ = learn_weights_alternating(
            all_q, all_ref, w0=W0_DEFAULT, lam=best_overall_lambda, mode=chosen_mode, max_rounds=5
        )

        proposal_data = {
            "decision": f"PROPOSE_{chosen_mode}",
            "reason": f"Phương án {chosen_mode} đã vượt qua toàn bộ các cổng đặt trước.",
            "chosen_mode": chosen_mode,
            "chosen_lambda": best_overall_lambda,
            "proposed_weights": {GROUP_NAMES[i]: float(w_final[i]) for i in range(8)},
            "comparison_results": comparison_results,
            "handshape_analysis": handshape_analysis,
            "config_changes": {
                "w_p": float(w_final[0]), "w_a": float(w_final[1]), "w_m": float(w_final[2]),
                "w_n": float(w_final[3]), "w_b": float(w_final[4]), "w_z": float(w_final[5]),
                "w_s": float(w_final[6]), "w_r": float(w_final[7])
            }
        }
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal_data, f, indent=2, ensure_ascii=False)

        # Xuất configs/C11_learned.json
        cfg_learned = {
            "name": f"C11_learned_{chosen_mode}",
            "description": f"Trọng số học được từ SPEC 11 ({chosen_mode})",
            "weights": proposal_data["proposed_weights"],
            "asllex_mode": "none",
            "flip_body_x": True,
            "thresholds": "scale_k:1.75",
            "matcher": {"margin_threshold": 0.12, "uncertainty_ratio": 1.40, "uncertainty_k": 2.35}
        }
        os.makedirs("configs", exist_ok=True)
        with open("configs/C11_learned.json", "w", encoding="utf-8") as f:
            json.dump(cfg_learned, f, indent=2, ensure_ascii=False)

    else:
        reasons = []
        for m in ["L5", "L8"]:
            if m in comparison_results:
                cr = comparison_results[m]
                if not cr["gate_ci_lower_gt_0"]:
                    reasons.append(f"{m}: Cận dưới KTC95 Delta margin <= 0 ([{cr['ci_lower']:+.4f}, {cr['ci_upper']:+.4f}])")
                if not cr["gate_stability"]:
                    reasons.append(f"{m}: Trọng số không ổn định qua các fold (CV > 0.5)")
                if not cr["gate_word_retention"]:
                    reasons.append(f"{m}: Mất quá 1 mẫu đúng ở từ: {cr['lost_words']}")
                if not cr["gate_wrong_accept_not_increased"]:
                    reasons.append(f"{m}: Tăng wrong_accept")
                if not cr["gate_false_accept_neg_not_increased"]:
                    reasons.append(f"{m}: Tăng false_accept_neg")

        _log_progress(progress_file, "Không phương án nào vượt qua toàn bộ cổng đặt trước. Quyết định: GIỮ TRỌNG SỐ TAY.")
        proposal_data = {
            "decision": "KEEP_HAND_WEIGHTS",
            "reason": "Không có phương án nào vượt qua các cổng chấp nhận định lượng khắt khe đã đặt trước. " + "; ".join(reasons),
            "comparison_results": comparison_results,
            "handshape_analysis": handshape_analysis,
            "learned_weights": {
                m: {
                    "mean": pipeline_results[m]["weights_mean"],
                    "std": pipeline_results[m]["weights_std"],
                    "cv": pipeline_results[m]["weights_cv"]
                }
                for m in ["L5", "L8"] if m in pipeline_results
            }
        }
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal_data, f, indent=2, ensure_ascii=False)

    return proposal_data


def main() -> None:
    parser = argparse.ArgumentParser(description="SPEC 11 DTW Group Weight Learning CLI")
    parser.add_argument("--bank", type=str, default="results/SPEC11_20260921_133822/bank_frozen.json", help="Đường dẫn bank_frozen.json")
    parser.add_argument("--neg", type=str, default="results/SPEC11_20260921_133822/neg_frozen.json", help="Đường dẫn neg_frozen.json")
    parser.add_argument("--out", type=str, default="results/SPEC11_20260921_133822/nested", help="Thư mục xuất kết quả")
    parser.add_argument("--modes", nargs="+", default=["B0", "L5", "L8"], help="Danh sách chế độ chạy: B0 L5 L8")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.3, 3.0], help="Danh sách giá trị lambda")
    parser.add_argument("--seed", type=int, default=20260919, help="Random seed tất định")
    parser.add_argument("--workers", type=int, default=6, help="Số worker processes cho tính ma trận chi phí")

    args = parser.parse_args()

    run_nested_loso_pipeline(
        bank_path=args.bank,
        neg_path=args.neg,
        out_dir=args.out,
        modes=args.modes,
        lambdas=args.lambdas,
        random_seed=args.seed,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
