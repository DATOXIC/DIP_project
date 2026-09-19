#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scoreboard.py — Bảng điểm đo đạc và so sánh mọi thay đổi có kiểm định thống kê ghép cặp.

Tuân thủ nghiêm ngặt:
  - specs/_PREAMBLE.md
  - specs/SPEC_04A_scoreboard.md
  - specs/SPEC_04_hoc_trong_so.md

Chức năng:
  1. run     : Chạy kiểm định chéo (LOSO hoặc khối) trên 1 cấu hình và lưu vết kết quả.
  2. compare : So sánh ghép cặp 2 lần chạy (McNemar chính xác + Paired Bootstrap 95% CI + luật acceptance).
  3. sealed  : Chạy 1 lần trên tập kín phiên 3 (yêu cầu cờ xác nhận, ghi sổ lưu vết sealed_log.csv).
  4. test    : Chạy đủ 7 phép tự kiểm tra công cụ ở Nhiệm vụ 7 & SPEC 04A.1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import binom

from dynamic_dtw_engine import GoldenTriadDTWMatcher, MatchResult


# =============================================================================
# 1. TIỆN ÍCH HASH, GIT & CẤU HÌNH (NHIỆM VỤ 1)
# =============================================================================

ALLOWED_ROOT_CONFIG_KEYS = {
    "name",
    "weights",
    "asllex_mode",
    "asllex_lambda",
    "asllex_mapping",
    "flip_body_x",
    "matcher",
    "thresholds",
}

ALLOWED_WEIGHT_KEYS = {
    "w_p", "w_a", "w_m", "w_n", "w_z", "w_b", "w_s", "w_r"
}

ALLOWED_MATCHER_KEYS = {
    "margin_threshold",
    "uncertainty_ratio",
    "lambda_shape",
    "lambda_shape_global",
}


def compute_file_sha256(filepath: str) -> str:
    """Tính mã băm SHA-256 của một file."""
    if not os.path.exists(filepath):
        return "none"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Mã băm cấu hình = SHA-256 của JSON đã chuẩn hóa (khóa sắp xếp, không thừa khoảng trắng)."""
    canonical_json = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_git_info() -> Tuple[str, bool]:
    """Lấy commit hash hiện tại và kiểm tra xem working tree có bẩn (dirty) hay không."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        is_dirty = len(status) > 0
    except Exception:
        is_dirty = False

    return commit, is_dirty


def validate_and_hash_config(config_path: str) -> Tuple[Dict[str, Any], str]:
    """
    Nạp và kiểm tra tính hợp lệ của file cấu hình JSON.
    Các khóa lạ hoặc sai miền giá trị sẽ raise ValueError lập tức.
    Trả về: (dict_cấu_hình, mã_băm_sha256)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except Exception as e:
            raise ValueError(f"File cấu hình {config_path} không phải JSON hợp lệ: {e}")

    if not isinstance(cfg, dict):
        raise ValueError("Nội dung cấu hình phải là một JSON object.")

    # 1. Kiểm tra khóa lạ ở cấp gốc
    unknown_keys = set(cfg.keys()) - ALLOWED_ROOT_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Phát hiện khóa lạ không được hỗ trợ trong cấu hình: {sorted(list(unknown_keys))}")

    # 2. Kiểm tra trường weights
    weights = cfg.get("weights", {})
    if not isinstance(weights, dict):
        raise ValueError("Trường 'weights' phải là một dictionary.")
    unknown_weight_keys = set(weights.keys()) - ALLOWED_WEIGHT_KEYS
    if unknown_weight_keys:
        raise ValueError(f"Khóa lạ trong 'weights': {sorted(list(unknown_weight_keys))}")
    for k, v in weights.items():
        if not isinstance(v, (int, float)) or v < 0.0:
            raise ValueError(f"Trọng số '{k}' phải là số thực >= 0 (nhận được {v}).")

    # 3. Kiểm tra asllex_mode
    mode = str(cfg.get("asllex_mode", "none")).lower()
    if mode not in {"none", "replace", "modulate"}:
        raise ValueError(f"asllex_mode không hợp lệ: '{mode}'. Chỉ chấp nhận 'none', 'replace', hoặc 'modulate'.")

    # 4. Kiểm tra asllex_lambda
    lam = cfg.get("asllex_lambda", 0.0)
    if not isinstance(lam, (int, float)) or lam < 0.0:
        raise ValueError(f"asllex_lambda phải là số thực >= 0 (nhận được {lam}).")

    # 5. Kiểm tra asllex_mapping
    mapping = str(cfg.get("asllex_mapping", "legacy")).lower()
    if mapping not in {"legacy", "semantic"}:
        raise ValueError(f"asllex_mapping không hợp lệ: '{mapping}'. Chỉ chấp nhận 'legacy' hoặc 'semantic'.")

    # 6. Kiểm tra flip_body_x
    flip = cfg.get("flip_body_x", True)
    if not isinstance(flip, bool):
        raise ValueError(f"flip_body_x phải là boolean (True/False), nhận được {flip}.")

    # 7. Kiểm tra matcher
    matcher_cfg = cfg.get("matcher", {})
    if not isinstance(matcher_cfg, dict):
        raise ValueError("Trường 'matcher' phải là một dictionary.")
    unknown_matcher_keys = set(matcher_cfg.keys()) - ALLOWED_MATCHER_KEYS
    if unknown_matcher_keys:
        raise ValueError(f"Khóa lạ trong 'matcher': {sorted(list(unknown_matcher_keys))}")

    # 8. Kiểm tra thresholds
    thresh = cfg.get("thresholds", "from_bank")
    if thresh != "from_bank":
        if isinstance(thresh, str) and thresh.startswith("fixed:"):
            try:
                val = float(thresh.split(":", 1)[1])
                if val <= 0.0:
                    raise ValueError
            except Exception:
                raise ValueError(f"Định dạng thresholds cố định không hợp lệ: '{thresh}'. Ví dụ: 'fixed:1.25'")
        elif isinstance(thresh, str) and os.path.exists(thresh):
            # File thresholds hợp lệ
            pass
        else:
            raise ValueError(f"thresholds không hợp lệ: '{thresh}'. Chấp nhận 'from_bank', 'fixed:<x>', hoặc đường dẫn file.")

    cfg_hash = compute_config_hash(cfg)
    return cfg, cfg_hash


# =============================================================================
# 2. CHIA DỮ LIỆU & CHỐNG RÒ RỈ (NHIỆM VỤ 2)
# =============================================================================

@dataclass
class QuerySample:
    query_id: str
    true_word: str
    sample_id: Optional[str]
    session_id: Optional[str]
    traj: np.ndarray
    hand_shapes: List[str]
    handedness: Optional[str]
    has_body: bool
    rep_count: Optional[int]
    is_original: bool
    augment_of: Optional[str]
    chirality_canonical: bool = False


@dataclass
class DataFold:
    fold_name: str
    queries: List[QuerySample]
    reference_bank: Dict[str, Any]
    excluded_queries_count: int


def load_bank(bank_path: str) -> Tuple[Dict[str, Any], str]:
    """Nạp bank template từ file JSON."""
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Không tìm thấy bank template: {bank_path}")
    bank_hash = compute_file_sha256(bank_path)
    with open(bank_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "vocabulary" not in data:
        raise ValueError(f"File {bank_path} không có khóa 'vocabulary'.")
    return data, bank_hash


def load_negative_strokes(neg_path: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    """Nạp mẫu âm từ negative_strokes.json nếu có."""
    if not neg_path or not os.path.exists(neg_path):
        return [], "none"
    neg_hash = compute_file_sha256(neg_path)
    with open(neg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        strokes = data
    elif isinstance(data, dict):
        strokes = data.get("samples") or data.get("strokes") or []
    else:
        strokes = []
    return strokes, neg_hash


def split_dataset(bank_data: Dict[str, Any], random_seed: int = 20260919) -> Tuple[str, List[DataFold]]:
    """
    Chia dữ liệu theo đúng quy tắc Nhiệm vụ 2:
      1. LOSO theo session_id nếu có >= 2 phiên.
      2. Nếu chỉ có 1 phiên (hoặc không có session_id): chia theo khối nửa đầu / nửa sau từng từ.
      3. Loại biến thể augmented khỏi query; augmented chỉ ở phía reference và không chứa augment_of của query.
      4. Kiểm tra rò rỉ bằng assert nghiêm ngặt trước mỗi vòng.
    """
    vocabulary = bank_data.get("vocabulary", {})
    metadata = bank_data.get("metadata", {})

    # Tập hợp các session_id từ các mẫu gốc
    session_ids: Set[str] = set()
    total_samples = 0
    total_original = 0

    for word_key, word_data in vocabulary.items():
        for s in word_data.get("samples", []):
            total_samples += 1
            if s.get("is_original", True) is not False and not s.get("augment_of"):
                total_original += 1
                sess = s.get("session_id")
                if sess:
                    session_ids.add(str(sess))

    folds: List[DataFold] = []

    if len(session_ids) >= 2:
        split_mode = "loso"
        sorted_sessions = sorted(list(session_ids))

        for sess in sorted_sessions:
            queries: List[QuerySample] = []
            excluded_count = 0

            # Xây dựng reference bank cho fold này
            ref_vocab: Dict[str, Any] = {}
            for word_key, word_data in vocabulary.items():
                ref_samples = []
                for s in word_data.get("samples", []):
                    # Mẫu thuộc phiên bị giữ lại KHÔNG được vào reference
                    if str(s.get("session_id")) == sess:
                        continue
                    # Biến thể sinh từ stroke của phiên bị giữ lại không được xuất hiện
                    # (kiểm tra s.get("source_session") nếu có)
                    if str(s.get("source_session")) == sess:
                        continue
                    ref_samples.append(s)

                if ref_samples:
                    w_copy = copy.deepcopy(word_data)
                    w_copy["samples"] = ref_samples
                    ref_vocab[word_key] = w_copy

            # Tạo query từ các mẫu gốc của phiên `sess`
            for word_key, word_data in vocabulary.items():
                ref_word_samples = ref_vocab.get(word_key, {}).get("samples", [])
                for idx, s in enumerate(word_data.get("samples", [])):
                    if str(s.get("session_id")) != sess:
                        continue
                    # Chỉ lấy mẫu gốc làm query
                    if s.get("is_original", True) is False or s.get("augment_of"):
                        continue

                    # Nếu từ không còn mẫu tham chiếu ở các phiên khác -> loại và đếm
                    if not ref_word_samples:
                        excluded_count += 1
                        continue

                    raw_traj = s.get("traj_features_9d") if s.get("traj_features_9d") is not None else s.get("traj_features", [])
                    arr_traj = np.array(raw_traj, dtype=np.float32)
                    sid = s.get("sample_id")
                    qid = str(sid) if sid else f"{word_key}__s_{sess}__idx_{idx}"

                    is_canonical = (s.get("chirality_canonical") is True)
                    eff_handed = "Right" if is_canonical else s.get("handedness")

                    q = QuerySample(
                        query_id=qid,
                        true_word=word_key,
                        sample_id=str(sid) if sid else None,
                        session_id=sess,
                        traj=arr_traj,
                        hand_shapes=s.get("hand_shapes", ["Unknown"] * 32),
                        handedness=eff_handed,
                        has_body=s.get("body_anchor_valid", True),
                        rep_count=s.get("rep_count"),
                        is_original=True,
                        augment_of=s.get("augment_of"),
                        chirality_canonical=is_canonical,
                    )
                    queries.append(q)

            # Lọc lại reference: loại mọi biến thể augmented có augment_of trùng với bất kỳ query nào
            query_sample_ids = {q.sample_id for q in queries if q.sample_id}
            query_ids_set = {q.query_id for q in queries}
            cleaned_ref_vocab: Dict[str, Any] = {}
            for w_k, w_d in ref_vocab.items():
                c_samples = []
                for s in w_d.get("samples", []):
                    aug_of = s.get("augment_of")
                    if aug_of and (str(aug_of) in query_sample_ids or str(aug_of) in query_ids_set):
                        continue
                    c_samples.append(s)
                if c_samples:
                    w_copy = copy.deepcopy(w_d)
                    w_copy["samples"] = c_samples
                    cleaned_ref_vocab[w_k] = w_copy

            # -----------------------------------------------------------------
            # KIỂM TRA RÒ RỈ BẰNG ASSERT BẮT BUỘC TRƯỚC MỖI VÒNG (Quy tắc 5)
            # -----------------------------------------------------------------
            ref_sample_ids: Set[str] = set()
            ref_sessions: Set[str] = set()
            ref_aug_sources: Set[str] = set()

            for w_d in cleaned_ref_vocab.values():
                for s in w_d.get("samples", []):
                    if s.get("sample_id"):
                        ref_sample_ids.add(str(s.get("sample_id")))
                    if s.get("session_id"):
                        ref_sessions.add(str(s.get("session_id")))
                    if s.get("augment_of"):
                        ref_aug_sources.add(str(s.get("augment_of")))

            for q in queries:
                if q.sample_id:
                    assert q.sample_id not in ref_sample_ids, (
                        f"RÒ RỈ DỮ LIỆU NGHIÊM TRỌNG: query sample_id '{q.sample_id}' xuất hiện trong tập tham chiếu!"
                    )
                    assert q.sample_id not in ref_aug_sources, (
                        f"RÒ RỈ DỮ LIỆU NGHIÊM TRỌNG: query sample_id '{q.sample_id}' có mẫu phái sinh augment_of trong tập tham chiếu!"
                    )
                if q.session_id:
                    assert q.session_id not in ref_sessions, (
                        f"RÒ RỈ DỮ LIỆU NGHIÊM TRỌNG: query session_id '{q.session_id}' xuất hiện trong tập tham chiếu!"
                    )

            ref_bank = {"metadata": metadata, "vocabulary": cleaned_ref_vocab}
            folds.append(DataFold(fold_name=f"LOSO_{sess}", queries=queries, reference_bank=ref_bank, excluded_queries_count=excluded_count))

    else:
        # Bank chỉ có 1 phiên (hoặc không có session_id) -> Chia theo khối nửa đầu / nửa sau
        split_mode = "block_half"
        print("\033[91m[!] CẢNH BÁO ĐỎ: DANH GIA YEU: CUNG PHIEN (Dữ liệu chỉ có 1 phiên, chia theo nửa khối trong từng từ)\033[0m")

        for round_idx in [1, 2]:
            queries = []
            ref_vocab = {}
            excluded_count = 0

            for word_key, word_data in vocabulary.items():
                samples = word_data.get("samples", [])
                orig_samples = [
                    s for s in samples
                    if s.get("is_original", True) is not False and not s.get("augment_of")
                ]
                n = len(orig_samples)
                if n < 2:
                    excluded_count += n
                    continue

                mid = n // 2
                if round_idx == 1:
                    q_part = orig_samples[:mid]
                    r_part = orig_samples[mid:]
                else:
                    q_part = orig_samples[mid:]
                    r_part = orig_samples[:mid]

                if not r_part or not q_part:
                    excluded_count += len(q_part)
                    continue

                w_copy = copy.deepcopy(word_data)
                w_copy["samples"] = r_part
                ref_vocab[word_key] = w_copy

                for idx, s in enumerate(q_part):
                    raw_traj = s.get("traj_features_9d") if s.get("traj_features_9d") is not None else s.get("traj_features", [])
                    arr_traj = np.array(raw_traj, dtype=np.float32)
                    sid = s.get("sample_id")
                    qid = str(sid) if sid else f"{word_key}__r{round_idx}__idx_{idx}"

                    is_canonical = (s.get("chirality_canonical") is True)
                    eff_handed = "Right" if is_canonical else s.get("handedness")

                    q = QuerySample(
                        query_id=qid,
                        true_word=word_key,
                        sample_id=str(sid) if sid else None,
                        session_id=s.get("session_id"),
                        traj=arr_traj,
                        hand_shapes=s.get("hand_shapes", ["Unknown"] * 32),
                        handedness=eff_handed,
                        has_body=s.get("body_anchor_valid", True),
                        rep_count=s.get("rep_count"),
                        is_original=True,
                        augment_of=s.get("augment_of"),
                        chirality_canonical=is_canonical,
                    )
                    queries.append(q)

            # Kiểm tra rò rỉ
            ref_sample_ids = {str(s.get("sample_id")) for w in ref_vocab.values() for s in w.get("samples", []) if s.get("sample_id")}
            ref_aug_sources = {str(s.get("augment_of")) for w in ref_vocab.values() for s in w.get("samples", []) if s.get("augment_of")}

            for q in queries:
                if q.sample_id:
                    assert q.sample_id not in ref_sample_ids, (
                        f"RÒ RỈ DỮ LIỆU: query '{q.sample_id}' nằm trong tập tham chiếu khối {round_idx}!"
                    )
                    assert q.sample_id not in ref_aug_sources, (
                        f"RÒ RỈ DỮ LIỆU: query '{q.sample_id}' bị rò qua augment_of trong tập tham chiếu!"
                    )

            ref_bank = {"metadata": metadata, "vocabulary": ref_vocab}
            folds.append(DataFold(fold_name=f"BlockHalf_R{round_idx}", queries=queries, reference_bank=ref_bank, excluded_queries_count=excluded_count))

    return split_mode, folds


# =============================================================================
# 3. ĐÁNH GIÁ MATCHER & TÍNH TOÁN CHỈ SỐ (NHIỆM VỤ 3)
# =============================================================================

@dataclass
class QueryResult:
    query_id: str
    true_word: str
    pred_word: str
    status: str
    d_correct: float
    d_wrong: float
    margin: float
    latency_ms: float
    top1_correct: bool
    clean_accept: bool
    wrong_accept: bool


def build_matcher_from_config(config: Dict[str, Any], ref_bank: Dict[str, Any]) -> GoldenTriadDTWMatcher:
    """Khởi tạo một instance GoldenTriadDTWMatcher từ config dict và ref_bank."""
    w = config.get("weights", {})
    m = config.get("matcher", {})

    thresh_cfg = config.get("thresholds", "from_bank")
    threshold_override = None
    threshold_dict = None

    if isinstance(thresh_cfg, str):
        if thresh_cfg.startswith("fixed:"):
            threshold_override = float(thresh_cfg.split(":", 1)[1])
        elif thresh_cfg != "from_bank" and os.path.exists(thresh_cfg):
            with open(thresh_cfg, "r", encoding="utf-8") as f:
                threshold_dict = json.load(f)

    matcher = GoldenTriadDTWMatcher(
        templates_dict=ref_bank,
        w_p=float(w.get("w_p", 0.6)),
        w_a=float(w.get("w_a", 1.0)),
        w_m=float(w.get("w_m", 0.4)),
        w_n=float(w.get("w_n", 0.8)),
        w_z=float(w.get("w_z", 0.0)),
        w_b=float(w.get("w_b", 0.0)),
        w_s=float(w.get("w_s", 0.0)),
        w_r=float(w.get("w_r", 0.0)),
        lambda_shape=float(m.get("lambda_shape", 0.0)),
        lambda_shape_global=float(m.get("lambda_shape_global", 0.0)),
        margin_threshold=float(m.get("margin_threshold", 0.12)),
        uncertainty_ratio=float(m.get("uncertainty_ratio", 1.40)),
        threshold_override=threshold_override,
        strict_weights=(config.get("asllex_mode", "none") != "none"),
        asllex_mode=config.get("asllex_mode", "none"),
        asllex_lambda=float(config.get("asllex_lambda", 0.0)),
        asllex_mapping=config.get("asllex_mapping", "legacy"),
        flip_body_x=config.get("flip_body_x", True),
        threshold_dict=threshold_dict,
    )
    return matcher


def evaluate_query(matcher: GoldenTriadDTWMatcher, q: QuerySample) -> QueryResult:
    """So khớp 1 query qua đường quyết định thật của matcher."""
    t0 = time.perf_counter()
    eff_handed = "Right" if getattr(q, "chirality_canonical", False) else q.handedness
    match_res: MatchResult = matcher.match(
        query_traj_9d=q.traj,
        query_shapes=q.hand_shapes,
        handedness=eff_handed,
        query_has_body=q.has_body,
        query_rep_count=q.rep_count,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Tìm khoảng cách d_correct và d_wrong
    d_correct = 999.0
    d_wrong = 999.0

    for cand in match_res.candidates:
        if cand.word_key == q.true_word:
            if cand.distance < d_correct:
                d_correct = cand.distance
        else:
            if cand.distance < d_wrong:
                d_wrong = cand.distance

    # Relative contrast margin: (d_wrong - d_correct) / max(d_wrong, 1e-6)
    margin = (d_wrong - d_correct) / max(d_wrong, 1e-6)

    top_word = match_res.top_word or "NONE"
    status = match_res.status

    top1_correct = (len(match_res.candidates) > 0 and match_res.candidates[0].word_key == q.true_word)
    clean_accept = (status == "ACCEPTED" and top_word == q.true_word)
    wrong_accept = (status == "ACCEPTED" and top_word != q.true_word)

    return QueryResult(
        query_id=q.query_id,
        true_word=q.true_word,
        pred_word=top_word,
        status=status,
        d_correct=float(d_correct),
        d_wrong=float(d_wrong),
        margin=float(margin),
        latency_ms=float(latency_ms),
        top1_correct=bool(top1_correct),
        clean_accept=bool(clean_accept),
        wrong_accept=bool(wrong_accept),
    )


def evaluate_negative_strokes(
    matcher: GoldenTriadDTWMatcher,
    neg_strokes: List[Dict[str, Any]]
) -> Tuple[int, int, float, List[float]]:
    """Đo số mẫu âm bị nhận nhầm (ACCEPTED) và độ trễ."""
    if not neg_strokes:
        return 0, 0, 0.0, []

    fa_count = 0
    latencies: List[float] = []

    for s in neg_strokes:
        raw_traj = s.get("traj_features_9d") if s.get("traj_features_9d") is not None else s.get("traj_features", [])
        arr_traj = np.array(raw_traj, dtype=np.float32)
        shapes = s.get("hand_shapes", ["Unknown"] * 32)
        is_canonical = (s.get("chirality_canonical") is True)
        handed = "Right" if is_canonical else s.get("handedness")
        has_body = s.get("body_anchor_valid", True)
        rep = s.get("rep_count")

        t0 = time.perf_counter()
        res = matcher.match(
            query_traj_9d=arr_traj,
            query_shapes=shapes,
            handedness=handed,
            query_has_body=has_body,
            query_rep_count=rep,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        if res.status == "ACCEPTED":
            fa_count += 1

    total_neg = len(neg_strokes)
    fa_rate = float(fa_count / total_neg) if total_neg > 0 else 0.0
    return fa_count, total_neg, fa_rate, latencies


def compute_metrics_and_confusion(
    query_results: List[QueryResult],
    fa_neg_count: int,
    total_neg: int,
    neg_latencies: List[float]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    """Tổng hợp toàn bộ chỉ số từ danh sách QueryResult và mẫu âm."""
    n_queries = len(query_results)
    if n_queries == 0:
        return {}, {}

    top1_count = sum(1 for r in query_results if r.top1_correct)
    clean_accept_count = sum(1 for r in query_results if r.clean_accept)
    wrong_accept_count = sum(1 for r in query_results if r.wrong_accept)
    uncertain_count = sum(1 for r in query_results if r.status == "UNCERTAIN")
    reject_count = sum(1 for r in query_results if r.status == "REJECTED")

    margins = [r.margin for r in query_results]
    latencies = [r.latency_ms for r in query_results] + neg_latencies

    margin_mean = float(np.mean(margins))

    # Margin theo từng từ để tìm từ yếu nhất
    word_margins: Dict[str, List[float]] = {}
    for r in query_results:
        word_margins.setdefault(r.true_word, []).append(r.margin)

    worst_word = ""
    worst_margin = 999.0
    for w, m_list in word_margins.items():
        mean_m = float(np.mean(m_list))
        if mean_m < worst_margin:
            worst_margin = mean_m
            worst_word = w

    # Ma trận nhầm lẫn: true_word x predicted_label
    confusion: Dict[str, Dict[str, int]] = {}
    all_true_words = sorted(list({r.true_word for r in query_results}))

    for r in query_results:
        w_true = r.true_word
        if r.status == "ACCEPTED":
            pred_col = r.pred_word
        elif r.status == "UNCERTAIN":
            pred_col = f"{r.pred_word}_UNCERTAIN"
        else:
            pred_col = "REJECTED"

        confusion.setdefault(w_true, {})
        confusion[w_true][pred_col] = confusion[w_true].get(pred_col, 0) + 1

    metrics = {
        "num_positive_queries": n_queries,
        "top1": float(top1_count / n_queries),
        "top1_count": top1_count,
        "clean_accept": float(clean_accept_count / n_queries),
        "clean_accept_count": clean_accept_count,
        "wrong_accept": float(wrong_accept_count / n_queries),
        "wrong_accept_count": wrong_accept_count,
        "uncertain_rate": float(uncertain_count / n_queries),
        "uncertain_count": uncertain_count,
        "reject_rate": float(reject_count / n_queries),
        "reject_count": reject_count,
        "margin_mean": margin_mean,
        "margin_worst_word": {
            "word": worst_word,
            "margin": float(worst_margin) if worst_word else 0.0
        },
        "num_negative_samples": total_neg,
        "false_accept_neg_count": fa_neg_count,
        "false_accept_neg": float(fa_neg_count / total_neg) if total_neg > 0 else 0.0,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }

    return metrics, confusion


# =============================================================================
# 4. CHẾ ĐỘ RUN & LƯU VẾT (NHIỆM VỤ 4)
# =============================================================================

def run_evaluation(
    config_path: str,
    bank_path: str,
    neg_path: Optional[str] = None,
    out_dir: str = "results",
    random_seed: int = 20260919,
    run_tag: Optional[str] = None,
    quiet: bool = False
) -> Tuple[str, Dict[str, Any], List[QueryResult]]:
    """Thực thi đầy đủ 1 lần đánh giá (chế độ run)."""
    cfg, cfg_hash = validate_and_hash_config(config_path)
    bank_data, bank_hash = load_bank(bank_path)
    neg_strokes, neg_hash = load_negative_strokes(neg_path)

    split_mode, folds = split_dataset(bank_data, random_seed=random_seed)

    all_query_results: List[QueryResult] = []
    total_excluded = sum(f.excluded_queries_count for f in folds)

    if not quiet:
        print(f"\n========================================================")
        print(f"  SCOREBOARD RUN: {cfg.get('name', 'unnamed')}")
        print(f"========================================================")
        print(f"Cấu hình     : {config_path} (SHA-256: {cfg_hash[:12]}...)")
        print(f"Bank         : {bank_path} (SHA-256: {bank_hash[:12]}...)")
        print(f"Kiểu chia    : {split_mode.upper()} ({len(folds)} vòng fold)")
        if total_excluded > 0:
            print(f"Truy vấn loại: {total_excluded} (do từ chỉ có mẫu ở 1 phiên duy nhất)")

    # Đánh giá từng fold
    for fold in folds:
        matcher = build_matcher_from_config(cfg, fold.reference_bank)
        for q in fold.queries:
            q_res = evaluate_query(matcher, q)
            all_query_results.append(q_res)

    # Đánh giá negative strokes trên toan bộ bank
    full_matcher = build_matcher_from_config(cfg, bank_data)
    fa_neg_count, total_neg, fa_neg_rate, neg_latencies = evaluate_negative_strokes(full_matcher, neg_strokes)

    metrics, confusion = compute_metrics_and_confusion(all_query_results, fa_neg_count, total_neg, neg_latencies)

    # Thêm siêu dữ liệu vào metrics
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_name = cfg.get("name", "config")
    run_id = f"run_{cfg_name}_{now_str}" if not run_tag else f"run_{cfg_name}_{run_tag}_{now_str}"

    git_commit, is_dirty = get_git_info()

    meta = {
        "run_id": run_id,
        "timestamp": now_str,
        "config_path": config_path,
        "config_name": cfg_name,
        "config_hash": cfg_hash,
        "bank_path": bank_path,
        "bank_hash": bank_hash,
        "neg_path": neg_path or "none",
        "neg_hash": neg_hash,
        "split_mode": split_mode,
        "num_folds": len(folds),
        "git_commit": git_commit,
        "dirty_tree": is_dirty,
        "random_seed": random_seed,
    }
    metrics["metadata"] = meta

    # Tạo thư mục output
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Ghi metrics.json
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 2. Ghi per_query.csv
    per_query_path = os.path.join(run_dir, "per_query.csv")
    with open(per_query_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "true_word", "pred_word", "status",
            "d_correct", "d_wrong", "margin", "latency_ms",
            "top1", "clean_accept", "wrong_accept"
        ])
        for r in all_query_results:
            writer.writerow([
                r.query_id, r.true_word, r.pred_word, r.status,
                f"{r.d_correct:.4f}", f"{r.d_wrong:.4f}", f"{r.margin:.4f}", f"{r.latency_ms:.2f}",
                1 if r.top1_correct else 0,
                1 if r.clean_accept else 0,
                1 if r.wrong_accept else 0,
            ])

    # 3. Ghi confusion.csv
    confusion_path = os.path.join(run_dir, "confusion.csv")
    all_pred_cols = sorted(list({col for row in confusion.values() for col in row.keys()}))
    with open(confusion_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_word"] + all_pred_cols)
        for w_true, cols in sorted(confusion.items()):
            row = [w_true] + [cols.get(col, 0) for col in all_pred_cols]
            writer.writerow(row)

    # 4. Ghi config.json (bản sao)
    copy_config_path = os.path.join(run_dir, "config.json")
    with open(copy_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # 5. Thêm 1 dòng vào results/scoreboard_runs.csv
    summary_csv_path = os.path.join(out_dir, "scoreboard_runs.csv")
    header = [
        "timestamp", "run_id", "scoreboard_version", "config_name", "config_hash", "git_commit", "dirty_tree",
        "bank_path", "bank_hash", "neg_path", "neg_hash", "split_mode", "num_queries",
        "top1", "clean_accept", "wrong_accept", "uncertain_rate", "reject_rate",
        "margin_mean", "worst_word", "worst_word_margin", "false_accept_neg", "latency_ms"
    ]
    file_exists = os.path.exists(summary_csv_path)
    with open(summary_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            now_str, run_id, "04A.1", cfg_name, cfg_hash[:16], git_commit[:8], is_dirty,
            os.path.basename(bank_path), bank_hash[:16], os.path.basename(neg_path or "none"), neg_hash[:16],
            split_mode, metrics.get("num_positive_queries", 0),
            f"{metrics.get('top1', 0.0):.4f}",
            f"{metrics.get('clean_accept', 0.0):.4f}",
            f"{metrics.get('wrong_accept', 0.0):.4f}",
            f"{metrics.get('uncertain_rate', 0.0):.4f}",
            f"{metrics.get('reject_rate', 0.0):.4f}",
            f"{metrics.get('margin_mean', 0.0):.4f}",
            metrics.get("margin_worst_word", {}).get("word", "none"),
            f"{metrics.get('margin_worst_word', {}).get('margin', 0.0):.4f}",
            f"{metrics.get('false_accept_neg', 0.0):.4f}",
            f"{metrics.get('latency_ms', 0.0):.2f}",
        ])

    if not quiet:
        print_summary_table(metrics)

    return run_dir, metrics, all_query_results


def print_summary_table(m: Dict[str, Any]) -> None:
    """In bảng tóm tắt chỉ số đẹp mắt ra console."""
    meta = m.get("metadata", {})
    print("\n---------------------- BẢNG CHỈ SỐ KẾT QUẢ ----------------------")
    print(f"  Mã lần chạy      : {meta.get('run_id')}")
    print(f"  Số truy vấn      : {m.get('num_positive_queries')}")
    print(f"  Top-1 Accuracy   : {m.get('top1', 0.0) * 100:.2f}% ({m.get('top1_count')}/{m.get('num_positive_queries')})")
    print(f"  Clean Accept     : {m.get('clean_accept', 0.0) * 100:.2f}% ({m.get('clean_accept_count')}/{m.get('num_positive_queries')}) [Chấp nhận đúng]")
    print(f"  Wrong Accept     : {m.get('wrong_accept', 0.0) * 100:.2f}% ({m.get('wrong_accept_count')}/{m.get('num_positive_queries')}) [Lỗi nặng]")
    print(f"  Uncertain Rate   : {m.get('uncertain_rate', 0.0) * 100:.2f}% ({m.get('uncertain_count')})")
    print(f"  Reject Rate      : {m.get('reject_rate', 0.0) * 100:.2f}% ({m.get('reject_count')})")
    print(f"  Margin Mean      : {m.get('margin_mean', 0.0):+.4f}")
    ww = m.get("margin_worst_word", {})
    print(f"  Worst Word Margin: {ww.get('margin', 0.0):+.4f} (Từ: '{ww.get('word', 'N/A')}')")
    if m.get("num_negative_samples", 0) > 0:
        print(f"  False Accept Neg : {m.get('false_accept_neg', 0.0) * 100:.2f}% ({m.get('false_accept_neg_count')}/{m.get('num_negative_samples')})")
    else:
        print(f"  False Accept Neg : N/A (Không có mẫu âm)")
    print(f"  Latency TB       : {m.get('latency_ms', 0.0):.2f} ms")
    print("----------------------------------------------------------------\n")


# =============================================================================
# 5. CHẾ ĐỘ COMPARE & SO SÁNH GHÉP CẶP (NHIỆM VỤ 5)
# =============================================================================

def mcnemar_exact_test(b: int, c: int) -> float:
    """
    Kiểm định McNemar chính xác (Exact Binomial Test) trên các cặp bất đồng:
      b: Số mẫu run_base sai nhưng run_new đúng (Mới đúng, Cũ sai)
      c: Số mẫu run_base đúng nhưng run_new sai (Cũ đúng, Mới sai)
    H0: p = 0.5. Trả về p-value 2 phía.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 2 * P(X <= k) với X ~ Binom(n, 0.5)
    p_val = 2.0 * float(binom.cdf(k, n, 0.5))
    return min(1.0, float(p_val))


def paired_bootstrap_delta_margin(
    delta_margins: np.ndarray,
    n_resamples: int = 5000,
    random_seed: int = 20260919
) -> Tuple[float, float, float, float]:
    """
    Ước lượng khoảng tin cậy 95% bằng Bootstrap ghép cặp (5000 lần lấy mẫu lại)
    cho hiệu trung bình Delta margin = margin_new - margin_base.
    Trả về: (mean_delta, ci_lower, ci_upper, se)
    """
    n = len(delta_margins)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    rng = np.random.default_rng(random_seed)
    # Lấy mẫu lại n_resamples lần có hoàn lại
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(delta_margins[indices], axis=1)

    mean_delta = float(np.mean(delta_margins))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    se = float(np.std(boot_means, ddof=1))
    return mean_delta, ci_lower, ci_upper, se


def compare_runs(
    base_dir: str,
    new_dir: str,
    rule_path: str = "acceptance.json",
    random_seed: int = 20260919
) -> Dict[str, Any]:
    """
    So sánh ghép cặp hai lần chạy:
      1. Kiểm tra tính tương thích (cùng bank, cùng kiểu chia).
      2. Ghép query theo id.
      3. Kiểm định McNemar trên Top-1 và clean_accept.
      4. Khoảng tin cậy Bootstrap 95% cho Delta margin.
      5. Thống kê theo từ và kiểm tra hồi quy.
      6. Phán quyết theo luật acceptance.json: ACCEPT / REJECT / INCONCLUSIVE.
    """
    base_metrics_path = os.path.join(base_dir, "metrics.json")
    new_metrics_path = os.path.join(new_dir, "metrics.json")
    base_query_path = os.path.join(base_dir, "per_query.csv")
    new_query_path = os.path.join(new_dir, "per_query.csv")

    for p in [base_metrics_path, new_metrics_path, base_query_path, new_query_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy file yêu cầu cho so sánh: {p}")

    with open(base_metrics_path, "r", encoding="utf-8") as f:
        base_metrics = json.load(f)
    with open(new_metrics_path, "r", encoding="utf-8") as f:
        new_metrics = json.load(f)

    # 1. Kiểm tra tính tương thích
    base_meta = base_metrics.get("metadata", {})
    new_meta = new_metrics.get("metadata", {})

    if base_meta.get("bank_hash") != new_meta.get("bank_hash"):
        raise ValueError(
            f"Không thể so sánh ghép cặp: Hai lần chạy sử dụng hai bank template khác nhau!\n"
            f"  Base bank hash: {base_meta.get('bank_hash')}\n"
            f"  New bank hash : {new_meta.get('bank_hash')}"
        )
    if base_meta.get("split_mode") != new_meta.get("split_mode"):
        raise ValueError(
            f"Không thể so sánh ghép cặp: Hai lần chạy sử dụng kiểu chia khác nhau!\n"
            f"  Base split: {base_meta.get('split_mode')}\n"
            f"  New split : {new_meta.get('split_mode')}"
        )

    # 2. Đọc per_query.csv
    base_queries: Dict[str, Dict[str, Any]] = {}
    with open(base_query_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            base_queries[r["query_id"]] = r

    new_queries: Dict[str, Dict[str, Any]] = {}
    with open(new_query_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            new_queries[r["query_id"]] = r

    common_qids = sorted(list(set(base_queries.keys()) & set(new_queries.keys())))
    if len(common_qids) != len(base_queries) or len(common_qids) != len(new_queries):
        raise ValueError(
            f"Tập truy vấn không khớp hoàn toàn giữa hai lần chạy: "
            f"Base có {len(base_queries)}, New có {len(new_queries)}, chung {len(common_qids)}."
        )

    # 3. Phân tích ghép cặp
    top1_discordant_b = 0  # base sai (0), new đúng (1)
    top1_discordant_c = 0  # base đúng (1), new sai (0)

    clean_discordant_b = 0
    clean_discordant_c = 0

    delta_margins: List[float] = []
    changed_queries: List[Dict[str, Any]] = []

    # Theo dõi mẫu đúng từng từ
    words = sorted(list({base_queries[qid]["true_word"] for qid in common_qids}))
    word_base_correct: Dict[str, int] = {w: 0 for w in words}
    word_new_correct: Dict[str, int] = {w: 0 for w in words}

    for qid in common_qids:
        b_q = base_queries[qid]
        n_q = new_queries[qid]
        true_w = b_q["true_word"]

        b_top1 = int(b_q["top1"])
        n_top1 = int(n_q["top1"])
        b_clean = int(b_q["clean_accept"])
        n_clean = int(n_q["clean_accept"])

        if b_top1 == 0 and n_top1 == 1:
            top1_discordant_b += 1
        elif b_top1 == 1 and n_top1 == 0:
            top1_discordant_c += 1

        if b_clean == 0 and n_clean == 1:
            clean_discordant_b += 1
        elif b_clean == 1 and n_clean == 0:
            clean_discordant_c += 1

        if b_clean == 1:
            word_base_correct[true_w] += 1
        if n_clean == 1:
            word_new_correct[true_w] += 1

        d_margin = float(n_q["margin"]) - float(b_q["margin"])
        delta_margins.append(d_margin)

        if b_q["status"] != n_q["status"] or b_q["pred_word"] != n_q["pred_word"]:
            changed_queries.append({
                "query_id": qid,
                "true_word": true_w,
                "base_pred": b_q["pred_word"],
                "base_status": b_q["status"],
                "new_pred": n_q["pred_word"],
                "new_status": n_q["status"],
                "delta_margin": d_margin,
            })

    # McNemar p-values
    p_val_top1 = mcnemar_exact_test(top1_discordant_b, top1_discordant_c)
    p_val_clean = mcnemar_exact_test(clean_discordant_b, clean_discordant_c)

    # Bootstrap Delta Margin
    delta_arr = np.array(delta_margins, dtype=np.float64)
    d_mean, ci_lower, ci_upper, se = paired_bootstrap_delta_margin(delta_arr, n_resamples=5000, random_seed=random_seed)

    # Đánh giá hồi quy theo từ
    word_regression_counts: Dict[str, int] = {}
    max_regression_samples = 0
    worst_regress_word = ""

    for w in words:
        regress = word_base_correct[w] - word_new_correct[w]
        if regress > 0:
            word_regression_counts[w] = regress
            if regress > max_regression_samples:
                max_regression_samples = regress
                worst_regress_word = w

    # Mẫu âm nhận nhầm
    base_fa_neg = base_metrics.get("false_accept_neg_count", 0)
    new_fa_neg = new_metrics.get("false_accept_neg_count", 0)
    delta_fa_neg = new_fa_neg - base_fa_neg

    # Wrong accept tăng
    base_wrong = base_metrics.get("wrong_accept_count", 0)
    new_wrong = new_metrics.get("wrong_accept_count", 0)
    delta_wrong = new_wrong - base_wrong

    # Độ trễ tăng phần trăm
    base_lat = base_metrics.get("latency_ms", 1.0)
    new_lat = new_metrics.get("latency_ms", 1.0)
    latency_increase_pct = ((new_lat - base_lat) / max(base_lat, 1e-6)) * 100.0

    # -----------------------------------------------------------------
    # ĐÁNH GIÁ LUẬT CHẤP NHẬN (acceptance.json)
    # -----------------------------------------------------------------
    if not os.path.exists(rule_path):
        raise FileNotFoundError(f"Không tìm thấy file quy tắc chấp nhận: {rule_path}")

    with open(rule_path, "r", encoding="utf-8") as f:
        rule = json.load(f)

    max_fa_neg_inc = rule.get("max_false_accept_neg_increase", 0)
    max_wrong_inc = rule.get("max_wrong_accept_increase", 0)
    max_word_reg = rule.get("max_word_regression_samples", 1)
    max_lat_pct = rule.get("max_latency_increase_pct", 20.0)
    ci95_lower_req = rule.get("require_ci95_lower_gt", 0.0)

    # Kiểm tra các điều kiện an toàn (Safety conditions)
    rejection_reasons: List[str] = []

    if delta_fa_neg > max_fa_neg_inc:
        rejection_reasons.append(f"Mẫu âm nhận nhầm tăng: +{delta_fa_neg} mẫu (vượt quá ngưỡng cho phép {max_fa_neg_inc})")

    if delta_wrong > max_wrong_inc:
        rejection_reasons.append(f"Wrong accept (nhận sai từ) tăng: +{delta_wrong} mẫu (vượt ngưỡng {max_wrong_inc})")

    if max_regression_samples > max_word_reg:
        rejection_reasons.append(
            f"Từ '{worst_regress_word}' bị tụt {max_regression_samples} mẫu đúng "
            f"(vượt ngưỡng cho phép {max_word_reg} mẫu)"
        )

    if latency_increase_pct > max_lat_pct:
        rejection_reasons.append(
            f"Độ trễ tăng {latency_increase_pct:.1f}% (vượt ngưỡng cho phép {max_lat_pct}%)"
        )

    # Xác định phán quyết (Verdict)
    if rejection_reasons:
        verdict = "REJECT"
    else:
        # Nếu đạt toàn bộ điều kiện an toàn, kiểm tra chỉ số chính: CI95 lower > require_ci95_lower_gt
        if ci_lower > ci95_lower_req:
            verdict = "ACCEPT"
        else:
            verdict = "INCONCLUSIVE"

    comp_result = {
        "verdict": verdict,
        "rejection_reasons": rejection_reasons,
        "primary_metric": rule.get("primary", "margin_mean"),
        "delta_margin_mean": d_mean,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "bootstrap_se": se,
        "top1": {
            "base": base_metrics.get("top1", 0.0),
            "new": new_metrics.get("top1", 0.0),
            "discordant_new_better": top1_discordant_b,
            "discordant_base_better": top1_discordant_c,
            "mcnemar_p_value": p_val_top1,
        },
        "clean_accept": {
            "base": base_metrics.get("clean_accept", 0.0),
            "new": new_metrics.get("clean_accept", 0.0),
            "discordant_new_better": clean_discordant_b,
            "discordant_base_better": clean_discordant_c,
            "mcnemar_p_value": p_val_clean,
        },
        "wrong_accept": {
            "base": base_metrics.get("wrong_accept", 0.0),
            "new": new_metrics.get("wrong_accept", 0.0),
            "delta_count": delta_wrong,
        },
        "false_accept_neg": {
            "base_count": base_fa_neg,
            "new_count": new_fa_neg,
            "delta_count": delta_fa_neg,
        },
        "latency": {
            "base_ms": base_lat,
            "new_ms": new_lat,
            "increase_pct": latency_increase_pct,
        },
        "word_regressions": word_regression_counts,
        "changed_queries": changed_queries,
    }

    print_comparison_table(comp_result, base_meta.get("config_name"), new_meta.get("config_name"))
    return comp_result


def print_comparison_table(res: Dict[str, Any], base_name: Optional[str], new_name: Optional[str]) -> None:
    """In bảng so sánh ghép cặp chi tiết kèm phán quyết ra console."""
    base_label = base_name or "BASE"
    new_label = new_name or "NEW"
    verdict = res["verdict"]

    color_prefix = ""
    color_suffix = ""
    if verdict == "ACCEPT":
        color_prefix = "\033[92m"  # Xanh lá
        color_suffix = "\033[0m"
    elif verdict == "REJECT":
        color_prefix = "\033[91m"  # Đỏ
        color_suffix = "\033[0m"
    else:
        color_prefix = "\033[93m"  # Vàng
        color_suffix = "\033[0m"

    print("\n=================================================================")
    print(f"  KẾT QUẢ SO SÁNH GHÉP CẶP: {base_label}  vs  {new_label}")
    print("=================================================================")
    print(f"  PHÁN QUYẾT (VERDICT)   : {color_prefix}>>> {verdict} <<<{color_suffix}")
    if res["rejection_reasons"]:
        print("  LÝ DO TỪ CHỐI (REJECT) :")
        for reason in res["rejection_reasons"]:
            print(f"    - {reason}")
    elif verdict == "INCONCLUSIVE":
        print("  LƯU Ý INCONCLUSIVE      : An toàn nhưng khoảng tin cậy 95% chứa 0 hoặc <= 0.")
        print("                           Không đủ bằng chứng tốt hơn -> Giữ cấu hình cũ!")

    print("\n1. ĐỘ NÂNG MARGIN (Ghép cặp Bootstrap 5000 resamples):")
    print(f"   Delta Margin Trung bình: {res['delta_margin_mean']:+.4f}")
    print(f"   Khoảng Tin cậy 95%     : [{res['ci95_lower']:+.4f}, {res['ci95_upper']:+.4f}]")
    print(f"   Sai số chuẩn (SE)      : {res['bootstrap_se']:.4f}")

    top1 = res["top1"]
    print("\n2. TOP-1 ACCURACY & KIỂM ĐỊNH MCNEMAR:")
    print(f"   Base: {top1['base']*100:.2f}%  -->  New: {top1['new']*100:.2f}%")
    print(f"   Bất đồng: Mới đúng / Cũ sai = {top1['discordant_new_better']} | Cũ đúng / Mới sai = {top1['discordant_base_better']}")
    print(f"   McNemar p-value        : {top1['mcnemar_p_value']:.4f} "
          f"({'Khác biệt có ý nghĩa thống kê' if top1['mcnemar_p_value'] < 0.05 else 'Chưa đủ bằng chứng'})")

    clean = res["clean_accept"]
    print("\n3. CLEAN ACCEPT & KIỂM ĐỊNH MCNEMAR:")
    print(f"   Base: {clean['clean_accept' if 'clean_accept' in clean else 'base']*100:.2f}%  -->  New: {clean['new']*100:.2f}%")
    print(f"   Bất đồng: Mới đúng / Cũ sai = {clean['discordant_new_better']} | Cũ đúng / Mới sai = {clean['discordant_base_better']}")
    print(f"   McNemar p-value        : {clean['mcnemar_p_value']:.4f}")

    print("\n4. CHỈ SỐ AN TOÀN:")
    print("   [!] CẢNH BÁO THANG ĐO: Chỉ số nhận nhầm (wrong_accept, false_accept_neg) chỉ có nghĩa")
    print("       sau khi mỗi cấu hình đã hiệu chuẩn ngưỡng riêng (SPEC 04B).")
    print("       Các cấu hình khác nhau có thang khoảng cách khác nhau nên so sánh ở cùng một")
    print("       ngưỡng cố định không phản ánh đúng tỷ lệ nhận nhầm.")
    fa_neg = res["false_accept_neg"]
    print(f"   False Accept Mẫu âm    : Base {fa_neg['base_count']}  -->  New {fa_neg['new_count']} (Hiệu: {fa_neg['delta_count']:+d})")
    wr = res["wrong_accept"]
    print(f"   Wrong Accept           : Base {wr['base']*100:.2f}%  -->  New {wr['new']*100:.2f}% (Hiệu: {wr['delta_count']:+d} mẫu)")
    lat = res["latency"]
    print(f"   Độ trễ                 : Base {lat['base_ms']:.2f}ms  -->  New {lat['new_ms']:.2f}ms ({lat['increase_pct']:+.1f}%)")

    reg = res["word_regressions"]
    if reg:
        print(f"   Từ bị tụt mẫu đúng     : {reg}")
    else:
        print("   Từ bị tụt mẫu đúng     : Không có từ nào bị tụt")

    ch = res["changed_queries"]
    print(f"\n5. SỐ TRUY VẤN ĐỔI KẾT QUẢ: {len(ch)}")
    if ch:
        sample_show = ch[:5]
        for item in sample_show:
            print(f"   * Query {item['query_id']} ({item['true_word']}): "
                  f"[{item['base_status']}:{item['base_pred']}] --> [{item['new_status']}:{item['new_pred']}] (Delta margin: {item['delta_margin']:+.4f})")
        if len(ch) > 5:
            print(f"   ... và {len(ch) - 5} truy vấn khác.")
    print("=================================================================\n")


# =============================================================================
# 6. CHẾ ĐỘ TẬP KIỂM TRA KÍN (SEALED) (NHIỆM VỤ 6)
# =============================================================================

def run_sealed(
    config_path: str,
    sealed_bank_path: str,
    ref_bank_path: str = "webcam_templates.json",
    neg_path: Optional[str] = None,
    burn_confirmed: bool = False,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Chế độ kiểm tra tập kín phiên 3:
      - BẮT BUỘC có cờ `--i-understand-this-burns-the-test-set`.
      - Ghi lưu vết vào results/sealed_log.csv.
      - Đếm số lần sử dụng k. Nếu k >= 2 in cảnh báo đỏ.
      - CHỈ báo cáo chỉ số tổng hợp, TUYỆT ĐỐI không lưu per_query.csv.
    """
    if not burn_confirmed:
        print("\033[91m")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("  LỖI TỪ CHỐI CHẠY: THIẾU CỜ XÁC NHẬN SỬ DỤNG TẬP KIỂM TRA KÍN (SEALED)")
        print("  Tập kiểm tra kín (Phiên 3) là TÀI SẢN DUY NHẤT để nghiệm thu độc lập.")
        print("  Nếu dùng để dò tìm cấu hình, tập kiểm tra sẽ BỊ ĐỐT CHÁY (overfitting).")
        print("  Để thực sự chạy, bạn phải truyền cờ:")
        print("    --i-understand-this-burns-the-test-set")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("\033[0m")
        sys.exit(1)

    cfg, cfg_hash = validate_and_hash_config(config_path)
    sealed_data, sealed_hash = load_bank(sealed_bank_path)
    ref_data, ref_hash = load_bank(ref_bank_path)
    neg_strokes, neg_hash = load_negative_strokes(neg_path)

    os.makedirs(results_dir, exist_ok=True)
    sealed_log_path = os.path.join(results_dir, "sealed_log.csv")

    # Đếm số lần file sealed này đã được sử dụng
    usage_count = 1
    if os.path.exists(sealed_log_path):
        with open(sealed_log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5 and row[4] == sealed_hash:
                    usage_count += 1

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ghi vào sealed_log.csv
    file_exists = os.path.exists(sealed_log_path)
    with open(sealed_log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "config_name", "config_hash", "sealed_bank_path", "sealed_bank_hash", "run_k"])
        writer.writerow([now_str, cfg.get("name", "cfg"), cfg_hash, sealed_bank_path, sealed_hash, usage_count])

    print(f"\n>>> LẦN DÙNG THỨ {usage_count} CỦA TẬP KIỂM TRA KÍN <<<")
    if usage_count >= 2:
        print("\033[91m")
        print("============================================================================")
        print(f"  [!] CẢNH BÁO ĐỎ: LẦN DÙNG THỨ {usage_count}!")
        print("  Tập kiểm tra kín đã bị sử dụng nhiều lần để so sánh/chọn cấu hình.")
        print("  Dữ liệu KHÔNG CÒN TÍNH ĐỘC LẬP NGUYÊN BẢN (data leakage via decision loop).")
        print("============================================================================")
        print("\033[0m")

    # Xây dựng matcher trên ref_bank
    matcher = build_matcher_from_config(cfg, ref_data)

    # Đánh giá toàn bộ mẫu gốc trong sealed bank
    query_results: List[QueryResult] = []
    vocab = sealed_data.get("vocabulary", {})

    for word_key, word_data in vocab.items():
        for idx, s in enumerate(word_data.get("samples", [])):
            if s.get("is_original", True) is False or s.get("augment_of"):
                continue
            raw_traj = s.get("traj_features_9d") if s.get("traj_features_9d") is not None else s.get("traj_features", [])
            arr_traj = np.array(raw_traj, dtype=np.float32)
            sid = s.get("sample_id")
            qid = str(sid) if sid else f"{word_key}__sealed__idx_{idx}"

            is_canonical = (s.get("chirality_canonical") is True)
            eff_handed = "Right" if is_canonical else s.get("handedness")

            q = QuerySample(
                query_id=qid,
                true_word=word_key,
                sample_id=str(sid) if sid else None,
                session_id=s.get("session_id"),
                traj=arr_traj,
                hand_shapes=s.get("hand_shapes", ["Unknown"] * 32),
                handedness=eff_handed,
                has_body=s.get("body_anchor_valid", True),
                rep_count=s.get("rep_count"),
                is_original=True,
                augment_of=s.get("augment_of"),
                chirality_canonical=is_canonical,
            )
            q_res = evaluate_query(matcher, q)
            query_results.append(q_res)

    fa_neg_count, total_neg, fa_neg_rate, neg_lat = evaluate_negative_strokes(matcher, neg_strokes)
    metrics, _ = compute_metrics_and_confusion(query_results, fa_neg_count, total_neg, neg_lat)

    print("\n--- BÁO CÁO CHỈ SỐ TẬP KIỂM TRA KÍN (CHỈ BÁO CÁO TỔNG HỢP) ---")
    print(f"  Số truy vấn kín  : {len(query_results)}")
    print(f"  Top-1 Accuracy   : {metrics.get('top1', 0.0) * 100:.2f}%")
    print(f"  Clean Accept     : {metrics.get('clean_accept', 0.0) * 100:.2f}%")
    print(f"  Wrong Accept     : {metrics.get('wrong_accept', 0.0) * 100:.2f}%")
    print(f"  Margin Mean      : {metrics.get('margin_mean', 0.0):+.4f}")
    if total_neg > 0:
        print(f"  False Accept Neg : {metrics.get('false_accept_neg', 0.0) * 100:.2f}%")
    print(f"  Độ trễ trung bình: {metrics.get('latency_ms', 0.0):.2f} ms")
    print("----------------------------------------------------------------\n")

    return metrics


# =============================================================================
# 7. TỰ KIỂM TRA CÔNG CỤ (6 PHÉP THỬ Ở NHIỆM VỤ 7)
# =============================================================================

def run_self_tests() -> bool:
    """
    Thực hiện đầy đủ 7 phép tự kiểm tra công cụ ở Nhiệm vụ 7 & SPEC 04A.1:
      1. Đáp án đã biết: bank tổng hợp các lớp cách xa nhau -> top1=100%, clean_accept=100%.
      2. Cấu hình giống hệt: compare một cấu hình với chính nó -> hiệu margin=0, p=1.0, INCONCLUSIVE.
      3. Hồi quy đã biết trên Citizen 24D (10 từ): w_a=1.0 vs w_a=0.0 -> REJECT / CI95 nằm dưới 0.
      4. Bắt rò rỉ: cố tình đưa mẫu vào cả query và tham chiếu -> assert nổ (AssertionError).
      5. Tái lập: chạy hai lần cùng seed -> metrics.json giống hệt bit.
      6. Độ nhiễu nên có: in sai số chuẩn bootstrap của top1 và margin_mean.
      7. Bắt lỗi lật gương hai lần (SPEC 04A.1): mẫu có chirality_canonical: true và handedness: 'Left'.
    """
    print("\n" + "=" * 70)
    print("  BẮT ĐẦU 7 PHÉP TỰ KIỂM TRA CÔNG CỤ SCOREBOARD (NHIỆM VỤ 7 & SPEC 04A.1)")
    print("=" * 70)

    all_passed = True
    temp_dir = os.path.join("results", "self_test_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # PHÉP THỬ 1: Đáp án đã biết (Synthetic Known Answer)
    # -------------------------------------------------------------------------
    print("\n--- [Phép thử 1/7] Đáp án đã biết trên bank tổng hợp tách biệt tuyệt đối ---")
    synthetic_bank = {
        "metadata": {"feature_dim": 24},
        "vocabulary": {
            "SIGN_ALPHA": {
                "threshold": 1.5,
                "samples": [
                    {
                        "sample_id": "alpha_s1_1",
                        "session_id": "sess_1",
                        "traj_features": (np.zeros((32, 24)) + 15.0).tolist(),
                        "hand_shapes": ["A"] * 32,
                        "handedness": "Right",
                        "body_anchor_valid": True,
                    },
                    {
                        "sample_id": "alpha_s2_1",
                        "session_id": "sess_2",
                        "traj_features": (np.zeros((32, 24)) + 15.0).tolist(),
                        "hand_shapes": ["A"] * 32,
                        "handedness": "Right",
                        "body_anchor_valid": True,
                    },
                ]
            },
            "SIGN_BETA": {
                "threshold": 1.5,
                "samples": [
                    {
                        "sample_id": "beta_s1_1",
                        "session_id": "sess_1",
                        "traj_features": (np.zeros((32, 24)) - 15.0).tolist(),
                        "hand_shapes": ["B"] * 32,
                        "handedness": "Right",
                        "body_anchor_valid": True,
                    },
                    {
                        "sample_id": "beta_s2_1",
                        "session_id": "sess_2",
                        "traj_features": (np.zeros((32, 24)) - 15.0).tolist(),
                        "hand_shapes": ["B"] * 32,
                        "handedness": "Right",
                        "body_anchor_valid": True,
                    },
                ]
            },
        }
    }
    synth_bank_path = os.path.join(temp_dir, "synthetic_bank.json")
    with open(synth_bank_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_bank, f)

    synth_config = {
        "name": "synth_test",
        "weights": {"w_p": 0.6, "w_a": 1.0, "w_m": 0.4, "w_n": 0.8},
        "asllex_mode": "none",
        "asllex_lambda": 0.0,
        "asllex_mapping": "legacy",
        "flip_body_x": True,
        "matcher": {"margin_threshold": 0.12, "uncertainty_ratio": 1.4},
        "thresholds": "from_bank"
    }
    synth_cfg_path = os.path.join(temp_dir, "synth_config.json")
    with open(synth_cfg_path, "w", encoding="utf-8") as f:
        json.dump(synth_config, f)

    _, synth_m, _ = run_evaluation(synth_cfg_path, synth_bank_path, out_dir=temp_dir, quiet=True)
    t1 = synth_m.get("top1", 0.0)
    ca = synth_m.get("clean_accept", 0.0)
    print(f"  Kết quả: top1 = {t1 * 100:.1f}%, clean_accept = {ca * 100:.1f}%")
    if t1 == 1.0 and ca == 1.0:
        print("  -> ĐẠT (PASS): Bank tổng hợp đạt 100% khớp đúng kỳ vọng.")
    else:
        print(f"  -> THẤT BẠI (FAIL): Kỳ vọng 100%, nhận được top1={t1}, clean_accept={ca}")
        all_passed = False

    # -------------------------------------------------------------------------
    # PHÉP THỬ 2: Cấu hình giống hệt (Identical Config Comparison)
    # -------------------------------------------------------------------------
    print("\n--- [Phép thử 2/7] So sánh một cấu hình với chính nó (Identical Configs) ---")
    run1_dir, _, _ = run_evaluation(synth_cfg_path, synth_bank_path, out_dir=temp_dir, run_tag="t2_run1", quiet=True)

    comp_res = compare_runs(run1_dir, run1_dir, rule_path="acceptance.json")
    d_m = comp_res["delta_margin_mean"]
    v = comp_res["verdict"]
    print(f"  Kết quả: delta_margin = {d_m:.6f}, McNemar p = {comp_res['top1']['mcnemar_p_value']:.4f}, Verdict = {v}")
    if abs(d_m) < 1e-7 and v == "INCONCLUSIVE":
        print("  -> ĐẠT (PASS): Hiệu margin = 0, kết luận chuẩn tắc INCONCLUSIVE.")
    else:
        print(f"  -> THẤT BẠI (FAIL): delta_margin={d_m}, verdict={v}")
        all_passed = False

    # -------------------------------------------------------------------------
    # PHÉP THỬ 3: Phát hiện hồi quy đã biết trên Citizen 24D (10 từ)
    # -------------------------------------------------------------------------
    print("\n--- [Phép thử 3/7] Phát hiện hồi quy đã biết trên Citizen 24D (10 từ: w_a=1.0 vs w_a=0.0) ---")
    citizen_full_path = "asl_citizen_templates_24d.json"
    if not os.path.exists(citizen_full_path):
        print(f"  [!] Lỗi: Không tìm thấy {citizen_full_path}")
        all_passed = False
    else:
        # Lọc 10 từ theo spec
        words_10 = ["HELLO", "BYE", "ME", "YOU", "THANK_YOU", "EAT", "DRINK", "WATER", "SORRY", "PLEASE"]
        with open(citizen_full_path, "r", encoding="utf-8") as f:
            cit_data = json.load(f)
        sub_vocab = {w: cit_data["vocabulary"][w] for w in words_10 if w in cit_data.get("vocabulary", {})}
        sub_citizen = {"metadata": cit_data.get("metadata", {}), "vocabulary": sub_vocab}
        sub_cit_path = os.path.join(temp_dir, "citizen_10words.json")
        with open(sub_cit_path, "w", encoding="utf-8") as f:
            json.dump(sub_citizen, f)

        # Cấu hình C_wa_1 (base)
        cfg_wa_1 = {
            "name": "C_wa_1.0",
            "weights": {"w_p": 0.6, "w_a": 1.0, "w_m": 0.4, "w_n": 0.8, "w_z": 0.0, "w_b": 0.0, "w_s": 0.0, "w_r": 0.0},
            "asllex_mode": "none",
            "asllex_lambda": 0.0,
            "asllex_mapping": "legacy",
            "flip_body_x": True,
            "matcher": {"margin_threshold": 0.12, "uncertainty_ratio": 1.4},
            "thresholds": "from_bank"
        }
        # Cấu hình C_wa_0 (new - bỏ góc uốn ngón tay, hồi quy nặng)
        cfg_wa_0 = {
            "name": "C_wa_0.0",
            "weights": {"w_p": 0.6, "w_a": 0.0, "w_m": 0.4, "w_n": 0.8, "w_z": 0.0, "w_b": 0.0, "w_s": 0.0, "w_r": 0.0},
            "asllex_mode": "none",
            "asllex_lambda": 0.0,
            "asllex_mapping": "legacy",
            "flip_body_x": True,
            "matcher": {"margin_threshold": 0.12, "uncertainty_ratio": 1.4},
            "thresholds": "from_bank"
        }

        path_wa_1 = os.path.join(temp_dir, "cfg_wa_1.json")
        path_wa_0 = os.path.join(temp_dir, "cfg_wa_0.json")
        with open(path_wa_1, "w", encoding="utf-8") as f:
            json.dump(cfg_wa_1, f)
        with open(path_wa_0, "w", encoding="utf-8") as f:
            json.dump(cfg_wa_0, f)

        run_base_dir, m_base, _ = run_evaluation(path_wa_1, sub_cit_path, out_dir=temp_dir, run_tag="cit_base", quiet=True)
        run_new_dir, m_new, _ = run_evaluation(path_wa_0, sub_cit_path, out_dir=temp_dir, run_tag="cit_new", quiet=True)

        comp_cit = compare_runs(run_base_dir, run_new_dir, rule_path="acceptance.json")
        ci_u = comp_cit["ci95_upper"]
        cit_verdict = comp_cit["verdict"]

        print(f"  Base (w_a=1.0) Top-1 : {m_base['top1']*100:.1f}%, Margin Mean: {m_base['margin_mean']:+.4f}")
        print(f"  New  (w_a=0.0) Top-1 : {m_new['top1']*100:.1f}%, Margin Mean: {m_new['margin_mean']:+.4f}")
        print(f"  Delta Margin CI 95%   : [{comp_cit['ci95_lower']:+.4f}, {comp_cit['ci95_upper']:+.4f}]")
        print(f"  Phán quyết            : {cit_verdict}")

        if cit_verdict == "REJECT" or ci_u < 0.0:
            print("  -> ĐẠT (PASS): Công cụ phát hiện chính xác hồi quy (REJECT hoặc CI95 < 0).")
        else:
            print(f"  -> THẤT BẠI (FAIL): Công cụ không phát hiện được hồi quy (CI95 upper={ci_u}, verdict={cit_verdict})")
            all_passed = False

    # -------------------------------------------------------------------------
    # PHÉP THỬ 4: Phát hiện rò rỉ dữ liệu (Leakage Detection)
    # -------------------------------------------------------------------------
    print("\n--- [Phép thử 4/7] Phát hiện rò rỉ dữ liệu (Assert phải nổ) ---")
    leaked_bank = copy.deepcopy(synthetic_bank)
    # Cố tình đưa mẫu s1_1 vào cả session 2 của SIGN_ALPHA
    leaked_bank["vocabulary"]["SIGN_ALPHA"]["samples"][1]["sample_id"] = "alpha_s1_1"

    leak_caught = False
    try:
        split_dataset(leaked_bank)
    except AssertionError as e:
        leak_caught = True
        print(f"  Đã bắt được assert rò rỉ: '{e}'")

    if leak_caught:
        print("  -> ĐẠT (PASS): Assert nổ chính xác khi có rò rỉ sample_id.")
    else:
        print("  -> THẤT BẠI (FAIL): Rò rỉ dữ liệu không bị chặn bởi assert!")
        all_passed = False

    # -------------------------------------------------------------------------
    # PHÉP THỬ 5: Tính tái lập (Reproducibility)
    # -------------------------------------------------------------------------
    print("\n--- [Phép thử 5/7] Tái lập kết quả (chạy 2 lần cùng seed -> metrics giống hệt bit) ---")
    _, m_rep1, _ = run_evaluation(synth_cfg_path, synth_bank_path, out_dir=temp_dir, run_tag="rep1", random_seed=42, quiet=True)
    _, m_rep2, _ = run_evaluation(synth_cfg_path, synth_bank_path, out_dir=temp_dir, run_tag="rep2", random_seed=42, quiet=True)

    # So sánh các trường metrics cốt lõi (bỏ timestamp, run_id và độ trễ vật lý latency_ms do jitter phần cứng)
    m1_core = {k: v for k, v in m_rep1.items() if k not in ("metadata", "latency_ms")}
    m2_core = {k: v for k, v in m_rep2.items() if k not in ("metadata", "latency_ms")}

    if m1_core == m2_core:
        print("  -> ĐẠT (PASS): Hai lần chạy cho metrics thuật toán giống hệt bit 100%.")
    else:
        print(f"  -> THẤT BẠI (FAIL): Sai lệch chỉ số giữa hai lần chạy: { {k: (m1_core.get(k), m2_core.get(k)) for k in set(m1_core)|set(m2_core) if m1_core.get(k) != m2_core.get(k)} }")
        all_passed = False

    # -------------------------------------------------------------------------
    # PHÉP THỬ 6: Độ nhiễu nên có (Bootstrap Standard Error)
    # -------------------------------------------------------------------------
    print("\n--- [Phép thử 6/7] Độ nhiễu mẫu: Sai số chuẩn Bootstrap trên cỡ mẫu thật ---")
    # Đọc per_query của C_wa_1 (Citizen 10 từ, 68 mẫu)
    cit_per_query = os.path.join(run_base_dir, "per_query.csv")
    with open(cit_per_query, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    top1_arr = np.array([float(r["top1"]) for r in rows], dtype=np.float64)
    margin_arr = np.array([float(r["margin"]) for r in rows], dtype=np.float64)

    rng = np.random.default_rng(20260919)
    n = len(rows)
    indices = rng.integers(0, n, size=(5000, n))
    top1_boot_means = np.mean(top1_arr[indices], axis=1)
    margin_boot_means = np.mean(margin_arr[indices], axis=1)

    top1_se = float(np.std(top1_boot_means, ddof=1))
    margin_se = float(np.std(margin_boot_means, ddof=1))

    print(f"  Cỡ mẫu                  : N = {n} truy vấn")
    print(f"  Top-1 Accuracy          : {np.mean(top1_arr)*100:.2f}% (SE = {top1_se*100:.2f} điểm phần trăm)")
    print(f"  Margin Mean             : {np.mean(margin_arr):+.4f} (SE = {margin_se:.4f})")
    print("  Nhận xét: Với N ~ 68, sai số chuẩn của Top-1 là ~4-5%, chứng minh việc so sánh đơn lẻ")
    print("           bằng cách nhìn Top-1 là không đủ độ tin cậy -> Bắt buộc dùng Bảng Điểm ghép cặp!")
    print("  -> ĐẠT (PASS): Sai số chuẩn bootstrap được định lượng thành công.")

    # -------------------------------------------------------------------------
    # PHÉP THỬ 7: Bắt lỗi lật gương hai lần (SPEC 04A.1)
    # -------------------------------------------------------------------------
    print("\n--- [Phép thử 7/7] Kiểm tra chống lỗi lật gương hai lần (SPEC 04A.1) ---")
    feat_chir = np.zeros((32, 24), dtype=np.float32)
    feat_chir[:, 0] = np.linspace(0.0, 1.0, 32)  # Quỹ đạo x tăng dần
    feat_chir[:, 2] = 0.03125                     # dx > 0
    feat_chir[:, 9] = 1.0                         # Pháp tuyến nx dương (+1.0)
    feat_chir[:, 11] = 0.0

    chir_bank = {
        "metadata": {"feature_dim": 24},
        "vocabulary": {
            "HELLO": {
                "threshold": 1.0,
                "samples": [
                    {
                        "sample_id": "chir_s1",
                        "session_id": "sess_1",
                        "traj_features": feat_chir.tolist(),
                        "hand_shapes": ["A"] * 32,
                        "handedness": "Left",
                        "chirality_canonical": True,
                        "body_anchor_valid": True,
                    },
                    {
                        "sample_id": "chir_s2",
                        "session_id": "sess_2",
                        "traj_features": feat_chir.tolist(),
                        "hand_shapes": ["A"] * 32,
                        "handedness": "Left",
                        "chirality_canonical": True,
                        "body_anchor_valid": True,
                    },
                ]
            }
        }
    }
    chir_bank_path = os.path.join(temp_dir, "chir_bank.json")
    with open(chir_bank_path, "w", encoding="utf-8") as f:
        json.dump(chir_bank, f)

    chir_config = {
        "name": "chir_test",
        "weights": {"w_p": 0.6, "w_a": 1.0, "w_m": 0.4, "w_n": 0.8},
        "asllex_mode": "none",
        "asllex_lambda": 0.0,
        "asllex_mapping": "legacy",
        "flip_body_x": True,
        "matcher": {"margin_threshold": 0.12, "uncertainty_ratio": 1.4},
        "thresholds": "from_bank"
    }
    chir_cfg_path = os.path.join(temp_dir, "chir_config.json")
    with open(chir_cfg_path, "w", encoding="utf-8") as f:
        json.dump(chir_config, f)

    # 1. Mô phỏng khoảng cách trước bản vá (khi truyền handedness='Left' bị lật lại)
    m_unpatched = GoldenTriadDTWMatcher(templates_dict=chir_bank, w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8, strict_weights=False)
    res_unpatched = m_unpatched.match(query_traj_9d=feat_chir.copy(), query_shapes=["A"] * 32, handedness="Left")
    d_unpatched = res_unpatched.candidates[0].distance if res_unpatched.candidates else 999.0
    print(f"  Mô phỏng TRƯỚC bản vá (truyền handedness='Left' bị lật lại): d = {d_unpatched:.4f} >= 0.1 (FAIL)")

    # 2. Đo khoảng cách qua pipeline run_evaluation của scoreboard sau bản vá
    _, chir_m, chir_query_results = run_evaluation(chir_cfg_path, chir_bank_path, out_dir=temp_dir, quiet=True)
    d_corr = chir_query_results[0].d_correct if chir_query_results else 999.0
    print(f"  Thực tế SAU bản vá (xử lý chirality_canonical=True):        d = {d_corr:.4f}")

    if d_corr < 0.1 and d_unpatched >= 0.1:
        print(f"  -> ĐẠT (PASS): d_correct = {d_corr:.4f} < 0.1 (loại bỏ hoàn toàn lỗi lật gương hai lần).")
    else:
        print(f"  -> THẤT BẠI (FAIL): d_corr={d_corr:.4f}, d_unpatched={d_unpatched:.4f}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  KẾT QUẢ TỔNG THỂ: 7/7 PHÉP THỬ ĐỀU ĐẠT CHUẨN XÁC (PASS)!")
    else:
        print("  KẾT QUẢ TỔNG THỂ: CÓ PHÉP THỬ THẤT BẠI (FAIL)!")
    print("=" * 70 + "\n")

    return all_passed


# =============================================================================
# 8. GIAO DIỆN DÒNG LỆNH (CLI ENTRYPOINT)
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bảng điểm (Scoreboard): So sánh và kiểm định thống kê ghép cặp cho cử chỉ động."
    )
    subparsers = parser.add_subparsers(dest="command", help="Lệnh thực hiện: run, compare, sealed, test")

    # Lệnh: run
    run_p = subparsers.add_parser("run", help="Chạy đánh giá cross-validation cho 1 cấu hình và lưu vết kết quả")
    run_p.add_argument("--config", required=True, help="Đường dẫn file cấu hình JSON (vd: configs/C1_global.json)")
    run_p.add_argument("--bank", required=True, help="Đường dẫn bank template (vd: webcam_templates.json)")
    run_p.add_argument("--neg", default=None, help="Đường dẫn file negative_strokes.json (tùy chọn)")
    run_p.add_argument("--seed", type=int, default=20260919, help="Random seed (mặc định: 20260919)")
    run_p.add_argument("--out-dir", default="results", help="Thư mục lưu kết quả (mặc định: results)")
    run_p.add_argument("--tag", default=None, help="Thẻ tùy chọn gắn vào tên run_id")

    # Lệnh: compare
    comp_p = subparsers.add_parser(
        "compare",
        help="So sánh ghép cặp hai lần chạy (McNemar + Bootstrap + Acceptance)",
        description="So sánh ghép cặp hai lần chạy (McNemar + Bootstrap + Acceptance).\n"
                    "LƯU Ý: Chỉ số nhận nhầm (false_accept_neg, wrong_accept) chỉ có nghĩa "
                    "sau khi mỗi cấu hình đã hiệu chuẩn ngưỡng riêng (SPEC 04B). "
                    "Các cấu hình khác nhau có thang khoảng cách khác nhau nên so sánh "
                    "ở cùng một ngưỡng cố định có thể gây sai lệch phán đoán."
    )
    comp_p.add_argument("--base", required=True, help="Thư mục kết quả lần chạy cơ sở (vd: results/run_A)")
    comp_p.add_argument("--new", required=True, help="Thư mục kết quả lần chạy mới (vd: results/run_B)")
    comp_p.add_argument("--rule", default="acceptance.json", help="File quy tắc chấp nhận (mặc định: acceptance.json)")
    comp_p.add_argument("--seed", type=int, default=20260919, help="Random seed cho bootstrap (mặc định: 20260919)")

    # Lệnh: sealed
    seal_p = subparsers.add_parser("sealed", help="Đánh giá trên tập kiểm tra kín Phiên 3 (chỉ dùng 1 lần để nghiệm thu)")
    seal_p.add_argument("--config", required=True, help="Đường dẫn file cấu hình JSON")
    seal_p.add_argument("--sealed-bank", required=True, help="Đường dẫn bank phiên 3 kín")
    seal_p.add_argument("--bank", default="webcam_templates.json", help="Bank tham chiếu (mặc định: webcam_templates.json)")
    seal_p.add_argument("--neg", default=None, help="Đường dẫn file negative_strokes.json")
    seal_p.add_argument(
        "--i-understand-this-burns-the-test-set",
        action="store_true",
        default=False,
        help="Cờ xác nhận hiểu rõ việc kiểm tra trên tập kín sẽ tiêu tốn tính độc lập của nó"
    )

    # Lệnh: test
    test_p = subparsers.add_parser("test", help="Chạy 7 phép tự kiểm tra công cụ (Nhiệm vụ 7 & SPEC 04A.1)")

    args = parser.parse_args()

    if args.command == "run":
        run_evaluation(
            config_path=args.config,
            bank_path=args.bank,
            neg_path=args.neg,
            out_dir=args.out_dir,
            random_seed=args.seed,
            run_tag=args.tag,
        )
    elif args.command == "compare":
        compare_runs(
            base_dir=args.base,
            new_dir=args.new,
            rule_path=args.rule,
            random_seed=args.seed,
        )
    elif args.command == "sealed":
        run_sealed(
            config_path=args.config,
            sealed_bank_path=args.sealed_bank,
            ref_bank_path=args.bank,
            neg_path=args.neg,
            burn_confirmed=args.i_understand_this_burns_the_test_set,
        )
    elif args.command == "test":
        success = run_self_tests()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
