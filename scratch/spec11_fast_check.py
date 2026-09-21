import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamic_dtw_engine import GoldenTriadDTWMatcher, compute_golden_triad_dtw_distance, compute_repetition_count

out = sorted(os.path.join(ROOT, 'results', name) for name in os.listdir(os.path.join(ROOT, 'results')) if name.startswith('SPEC11_'))[-1]
with open(os.path.join(out, 'bank_frozen.json'), encoding='utf-8') as f:
    bank = json.load(f)

def matcher(weights):
    return GoldenTriadDTWMatcher(templates_dict=bank, w_p=weights[0], w_a=weights[1], w_m=weights[2], w_n=weights[3], w_b=weights[4], w_z=weights[5], w_s=weights[6], w_r=weights[7], strict_weights=False, asllex_mode='none', lambda_shape=0.0, lambda_shape_global=0.0, flip_body_x=True)

def valid_rows(m):
    rows = []
    for word, wd in m.vocabulary.items():
        for s in wd.get('samples', []):
            raw = s.get('traj_features_9d') if s.get('traj_features_9d') is not None else s.get('traj_features', [])
            t = np.asarray(raw, dtype=np.float32)
            if t.shape == (32, 24):
                rows.append((word, s, t))
    return rows

base = matcher((0.6, 1.0, 0.4, 0.8, 0.8, 0.0, 0.0, 0.0))
rows = valid_rows(base)
fast = base._fast_bank or None
base._fast_min_distances(rows[0][2], True)
fast = base._fast_bank
baseline = np.load(os.path.join(out, 'baseline_fast_distances.npy'))
current = []
for word, s, q in rows:
    d = base._fast_min_distances(q, bool(s.get('body_anchor_valid', True)))
    current.append([d[k] for k in sorted(d)])
print('T1-a', np.array_equal(np.asarray(current, dtype=np.float32), baseline), len(rows))

vectors = [
    (0.6, 1.0, 0.4, 0.8, 0.8, 0.5, 0.0, 0.15),
    (0.6, 1.0, 0.4, 0.8, 0.8, 0.0, 0.4, 0.0),
    (0.7, 0.8, 0.5, 0.7, 0.6, 0.3, 0.2, 0.2),
    (0.4, 1.2, 0.3, 0.9, 0.8, 0.2, 0.5, 0.1),
    (0.8, 0.9, 0.6, 0.5, 0.4, 0.1, 0.3, 0.25),
]
max_error = 0.0
checks = 0
for weights in vectors:
    m = matcher(weights)
    m._fast_min_distances(rows[0][2], True)
    b = m._fast_bank
    flat = 0
    for word, s, q in rows[:12]:
        q_rep = compute_repetition_count(q[:, 0], q[:, 1])
        got = b.distances(q, bool(s.get('body_anchor_valid', True)), q_rep)
        flat = 0
        for wk, wd in m.vocabulary.items():
            eff = m._get_effective_weights(wk)
            for ts in wd.get('samples', []):
                raw = ts.get('traj_features_9d') if ts.get('traj_features_9d') is not None else ts.get('traj_features', [])
                t = np.asarray(raw, dtype=np.float32)
                if t.shape != (32, 24):
                    continue
                ref = compute_golden_triad_dtw_distance(q, s.get('hand_shapes', ['Unknown'] * 32), t, ts.get('hand_shapes', ['Unknown'] * 32), w_p=eff['w_p'], w_a=eff['w_a'], w_m=eff['w_m'], w_n=eff['w_n'], w_b=eff['w_b'] if s.get('body_anchor_valid', True) and ts.get('body_anchor_valid', True) else 0.0, w_z=eff['w_z'], w_s=eff['w_s'], w_r=eff['w_r'], q_rep_count=q_rep, t_rep_count=ts.get('rep_count'))
                max_error = max(max_error, abs(float(got[flat]) - ref))
                checks += 1
                flat += 1
print('T1-b-c', checks, max_error)

m = matcher(vectors[0])
m._fast_min_distances(rows[0][2], True)
b = m._fast_bank
qs = np.stack([r[2] for r in rows[:5]])
hb = [bool(r[1].get('body_anchor_valid', True)) for r in rows[:5]]
reps = [compute_repetition_count(q[:, 0], q[:, 1]) for q in qs]
many = b.distances_many(qs, hb, q_reps=reps)
one = np.stack([b.distances(q, h, rep) for q, h, rep in zip(qs, hb, reps)])
print('T1-d', np.array_equal(many, one), float(np.max(np.abs(many - one))))
