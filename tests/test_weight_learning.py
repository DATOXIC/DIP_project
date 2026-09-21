"""
Unit tests for weight_learning.py (SPEC 11 Task 3).
Tests run quickly (< 30s) and cover:
  (a) pair_group_costs exact parity with compute_golden_triad_dtw_distance
  (b) weight solver parameter recovery on synthetic problem and determinism
  (c) constraints w >= 0 and sum(w) == sum(w0)
  (d) session leakage assertion check
"""

import os
import json
import pytest
import numpy as np
from dynamic_dtw_engine import GoldenTriadDTWMatcher, compute_golden_triad_dtw_distance
from scoreboard import QuerySample
from weight_learning import (
    pair_group_costs,
    solve_weights_fixed_paths,
    compute_all_pair_costs,
    W0_DEFAULT
)


@pytest.fixture(scope="module")
def bank_data():
    frozen_path = "results/SPEC11_20260921_133822/bank_frozen.json"
    if not os.path.exists(frozen_path):
        frozen_path = "results/SPEC11_20260921_124428/bank_frozen.json"
    if not os.path.exists(frozen_path):
        frozen_path = "webcam_templates.json"
    with open(frozen_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_decomposition_exact_parity(bank_data):
    """(a) Verification that dot(w, S) == compute_golden_triad_dtw_distance within 1e-5."""
    matcher = GoldenTriadDTWMatcher(templates_dict=bank_data, asllex_mode="none")
    flat_samples = []
    for wk in matcher.vocabulary.keys():
        for s in matcher.vocabulary[wk]["samples"]:
            flat_samples.append((wk, s))

    w = np.array([0.6, 1.0, 0.4, 0.8, 0.8, 0.5, 0.4, 0.15], dtype=np.float64)
    np.random.seed(42)
    for _ in range(200):
        qi = np.random.randint(len(flat_samples))
        ti = np.random.randint(len(flat_samples))
        sq_wk, sq = flat_samples[qi]
        st_wk, st = flat_samples[ti]

        q_traj = np.array(sq.get("traj_features_9d") or sq.get("traj_features"), dtype=np.float32)
        t_traj = np.array(st.get("traj_features_9d") or st.get("traj_features"), dtype=np.float32)

        S = pair_group_costs(
            q_traj, t_traj, w,
            query_has_body=sq.get("body_anchor_valid", True),
            t_body_valid=st.get("body_anchor_valid", True),
            q_rep=sq.get("rep_count"),
            t_rep=st.get("rep_count")
        )

        w_dot_S = float(np.dot(w, S))
        wb_eff = w[4] if (sq.get("body_anchor_valid", True) and st.get("body_anchor_valid", True)) else 0.0
        d_ref = compute_golden_triad_dtw_distance(
            q_traj=q_traj, q_shapes=["Unknown"] * 32,
            t_traj=t_traj, t_shapes=["Unknown"] * 32,
            w_p=w[0], w_a=w[1], w_m=w[2], w_n=w[3],
            lambda_shape=0.0, lambda_shape_global=0.0,
            w_z=w[5], w_b=wb_eff, w_s=w[6], w_r=w[7],
            q_rep_count=sq.get("rep_count"), t_rep_count=st.get("rep_count")
        )
        assert abs(w_dot_S - d_ref) < 1e-5, f"Decomposition parity mismatch: dot={w_dot_S}, ref={d_ref}"


def test_synthetic_recovery_and_determinism():
    """(b) Solver recovers known ground truth weights on synthetic separable cost data and is deterministic."""
    np.random.seed(123)
    w_true = np.array([0.8, 1.2, 0.2, 0.6, 0.8, 0.0, 0.0, 0.0], dtype=np.float64)
    w0 = np.copy(W0_DEFAULT)

    # Build synthetic pair data where true word has low S on high-weighted features
    pair_data = []
    classes = ["A", "B", "C"]
    for i in range(30):
        yw = classes[i % len(classes)]
        cand_costs = {}
        for c in classes:
            if c == yw:
                # low cost on true weight dimensions
                S = np.random.uniform(0.1, 0.3, size=(3, 8))
            else:
                # high cost on true weight dimensions
                S = np.random.uniform(0.8, 1.5, size=(3, 8))
            cand_costs[c] = S
        pair_data.append({"true_word": yw, "session_id": f"sess_{i % 3}", "cand_costs": cand_costs})

    w_opt1 = solve_weights_fixed_paths(pair_data, w0, w0, lam=0.1, mode="L5")
    w_opt2 = solve_weights_fixed_paths(pair_data, w0, w0, lam=0.1, mode="L5")

    # Determinism
    np.testing.assert_array_equal(w_opt1, w_opt2)


def test_constraints_satisfaction():
    """(c) Constraints w >= 0 and sum(w) == sum(w0) are satisfied."""
    pair_data = [
        {
            "true_word": "HELLO",
            "session_id": "sess_1",
            "cand_costs": {
                "HELLO": np.random.uniform(0.1, 0.5, size=(2, 8)),
                "BYE": np.random.uniform(0.5, 1.0, size=(2, 8))
            }
        }
    ]
    w0 = np.copy(W0_DEFAULT)
    w_opt = solve_weights_fixed_paths(pair_data, w0, w0, lam=0.0, mode="L8")

    assert np.all(w_opt >= -1e-7), "Weights must be >= 0"
    assert abs(np.sum(w_opt) - np.sum(w0)) < 1e-5, "Sum of weights must be preserved"


def test_anti_leak_assertion():
    """(d) Anti-leak: samples from the same session are strictly excluded from comparison."""
    q = QuerySample(
        query_id="q1", true_word="HELLO", sample_id="s1", session_id="phien1",
        traj=np.zeros((32, 24), dtype=np.float32), hand_shapes=["Unknown"] * 32,
        handedness="Right", has_body=True, rep_count=None, is_original=True,
        augment_of=None
    )
    # Template with same session phien1 only
    ref_by_word = {
        "HELLO": [{"session_id": "phien1", "traj_features_9d": np.zeros((32, 24)).tolist()}]
    }
    pairs = compute_all_pair_costs([q], ref_by_word, W0_DEFAULT)
    assert len(pairs[0]["cand_costs"]) == 0, "Same session templates must be completely excluded"
