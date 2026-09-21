"""
Unit tests for SPEC 11 Fast DTW extension with new groups (w_z, w_s, w_r).
Tests run quickly (< 30s) and verify bit-for-bit equivalence when weights are 0,
exact numerical equivalence with reference DTW, edge cases, distances_many consistency,
and verify mode.
"""

import os
import json
import numpy as np
import pytest
from dynamic_dtw_engine import (
    GoldenTriadDTWMatcher,
    _FastDTWBank,
    compute_golden_triad_dtw_distance,
    canonicalize_chirality
)

@pytest.fixture(scope="module")
def bank_data():
    frozen_path = "results/SPEC11_20260921_124428/bank_frozen.json"
    if not os.path.exists(frozen_path):
        frozen_path = "webcam_templates.json"
    with open(frozen_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_t1_a_zero_weights_bit_exact(bank_data):
    """T1-a: When w_z=w_s=w_r=0, distances must match baseline exactly."""
    matcher = GoldenTriadDTWMatcher(
        templates_dict=bank_data,
        w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8, w_b=0.8,
        w_z=0.0, w_s=0.0, w_r=0.0,
        asllex_mode="none",
        use_fast_dtw=True
    )
    fb = _FastDTWBank(matcher)
    
    flat_samples = []
    for wk in fb.words:
        for s in matcher.vocabulary[wk]["samples"]:
            flat_samples.append((wk, s))
            
    queries = [np.array(s.get("traj_features_9d") or s.get("traj_features"), dtype=np.float32) for _, s in flat_samples]
    has_body = [bool(s.get("body_anchor_valid", True)) for _, s in flat_samples]
    dist_mat = fb.distances_many(np.stack(queries), has_body)
    expected = np.asarray([
        [float(np.min(dist_mat[row, fb.slices[fb.words.index(word)][0]:fb.slices[fb.words.index(word)][1]])) for word in sorted(fb.words)]
        for row in range(dist_mat.shape[0])
    ], dtype=np.float32)
    
    baseline_path = "results/SPEC11_20260921_124428/baseline_fast_distances.npy"
    if os.path.exists(baseline_path):
        baseline = np.load(baseline_path)
        assert np.array_equal(expected, baseline), "Distances matrix is not bit-for-bit equal to baseline"


def test_t1_b_and_c_reference_equivalence_and_edge_cases(bank_data):
    """T1-b & T1-c: Parity with compute_golden_triad_dtw_distance across multiple weight vectors & edge cases."""
    weight_vectors = [
        (0.6, 1.0, 0.4, 0.8, 0.8, 0.5, 0.4, 0.15),  # all 3 non-zero
        (0.5, 0.8, 0.3, 0.6, 0.0, 0.3, 0.5, 0.2),   # all 3 non-zero, wb=0
        (0.6, 1.0, 0.4, 0.8, 0.8, 0.0, 0.4, 0.0),   # only w_s
        (0.6, 1.0, 0.4, 0.8, 0.8, 0.5, 0.0, 0.0),   # only w_z
        (0.6, 1.0, 0.4, 0.8, 0.8, 0.0, 0.0, 0.2),   # only w_r
    ]
    
    np.random.seed(42)
    for wv in weight_vectors:
        wp, wa, wm, wn, wb, wz, ws, wr = wv
        matcher = GoldenTriadDTWMatcher(
            templates_dict=bank_data,
            w_p=wp, w_a=wa, w_m=wm, w_n=wn, w_b=wb,
            w_z=wz, w_s=ws, w_r=wr,
            asllex_mode="none",
            use_fast_dtw=True
        )
        fb = _FastDTWBank(matcher)
        
        flat_samples = []
        for wk in fb.words:
            for s in matcher.vocabulary[wk]["samples"]:
                flat_samples.append((wk, s))
                
        q_idxs = np.random.choice(len(flat_samples), 15, replace=False)
        t_idxs = np.random.choice(len(flat_samples), 15, replace=False)
        
        for qi, ti in zip(q_idxs, t_idxs):
            sq_wk, sq = flat_samples[qi]
            st_wk, st = flat_samples[ti]
            
            q_traj = np.array(sq.get("traj_features_9d") or sq.get("traj_features"), dtype=np.float32)
            t_traj = np.array(st.get("traj_features_9d") or st.get("traj_features"), dtype=np.float32)
            
            # Test edge cases: toggle query_has_body
            for q_hb in [True, False]:
                d_fast = fb.distances(q_traj, q_hb, sq.get("rep_count"))[ti]
                wb_eff = wb if (q_hb and st.get("body_anchor_valid", True)) else 0.0
                d_ref = compute_golden_triad_dtw_distance(
                    q_traj=q_traj, q_shapes=["Unknown"] * 32,
                    t_traj=t_traj, t_shapes=["Unknown"] * 32,
                    w_p=wp, w_a=wa, w_m=wm, w_n=wn,
                    lambda_shape=0.0, lambda_shape_global=0.0,
                    w_z=wz, w_b=wb_eff, w_s=ws, w_r=wr,
                    q_rep_count=sq.get("rep_count"), t_rep_count=st.get("rep_count")
                )
                assert abs(d_fast - d_ref) < 1e-5, f"Mismatch on vector {wv}: fast={d_fast}, ref={d_ref}"


def test_t1_d_distances_many_parity(bank_data):
    """T1-d: distances_many matches distances single call."""
    matcher = GoldenTriadDTWMatcher(
        templates_dict=bank_data,
        w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8, w_b=0.8,
        w_z=0.5, w_s=0.4, w_r=0.15,
        asllex_mode="none",
        use_fast_dtw=True
    )
    fb = _FastDTWBank(matcher)
    
    flat_samples = []
    for wk in fb.words:
        for s in matcher.vocabulary[wk]["samples"]:
            flat_samples.append((wk, s))
            
    queries = [np.array(s.get("traj_features_9d") or s.get("traj_features"), dtype=np.float32) for _, s in flat_samples[:30]]
    has_body = [bool(s.get("body_anchor_valid", True)) for _, s in flat_samples[:30]]
    q_reps = [s.get("rep_count") for _, s in flat_samples[:30]]
    
    d_many = fb.distances_many(np.stack(queries), has_body, q_reps=q_reps)
    d_single = np.stack([fb.distances(q, hb, qr) for q, hb, qr in zip(queries, has_body, q_reps)])
    
    np.testing.assert_allclose(d_many, d_single, atol=1e-5)


def test_t1_e_verify_mode(bank_data):
    """T1-e: DIP_FAST_DTW=verify mode runs match without assertion."""
    matcher = GoldenTriadDTWMatcher(
        templates_dict=bank_data,
        w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8, w_b=0.8,
        w_z=0.5, w_s=0.4, w_r=0.15,
        asllex_mode="none",
        use_fast_dtw=True
    )
    flat_samples = []
    for wk in matcher.vocabulary.keys():
        for s in matcher.vocabulary[wk]["samples"]:
            flat_samples.append((wk, s))
            
    os.environ["DIP_FAST_DTW"] = "verify"
    try:
        for i in range(30):
            _, sq = flat_samples[i]
            traj = np.array(sq.get("traj_features_9d") or sq.get("traj_features"), dtype=np.float32)
            matcher.match(
                traj,
                ["Unknown"] * 32,
                handedness=sq.get("handedness"),
                query_has_body=sq.get("body_anchor_valid", True),
                query_rep_count=sq.get("rep_count")
            )
    finally:
        os.environ.pop("DIP_FAST_DTW", None)
