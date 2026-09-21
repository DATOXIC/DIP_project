import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamic_dtw_engine import GoldenTriadDTWMatcher

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out = os.path.join(ROOT, 'results', f'SPEC11_{stamp}')
os.makedirs(out, exist_ok=False)
bank_src = os.path.join(ROOT, 'webcam_templates.json')
neg_src = os.path.join(ROOT, 'negative_strokes.json')
bank_path = os.path.join(out, 'bank_frozen.json')
neg_path = os.path.join(out, 'neg_frozen.json')
shutil.copy2(bank_src, bank_path)
shutil.copy2(neg_src, neg_path)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def sample_counts(data):
    by_word = {k: len(v.get('samples', [])) for k, v in data['vocabulary'].items()}
    sessions = {}
    total = 0
    for v in data['vocabulary'].values():
        for s in v.get('samples', []):
            total += 1
            sessions[str(s.get('session_id', '<none>'))] = sessions.get(str(s.get('session_id', '<none>')), 0) + 1
    return total, by_word, sessions

bank = load(bank_path)
neg = load(neg_path)
matcher = GoldenTriadDTWMatcher(
    templates_dict=bank,
    w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8, w_z=0.0, w_b=0.8, w_s=0.0, w_r=0.0,
    lambda_shape=0.0, lambda_shape_global=0.0, strict_weights=False,
    asllex_mode='none', flip_body_x=True, threshold_mode='scale_k', threshold_k=1.75,
    uncertainty_k=2.35,
)
rows = []
for word, wd in matcher.vocabulary.items():
    for s in wd.get('samples', []):
        raw = s.get('traj_features_9d') if s.get('traj_features_9d') is not None else s.get('traj_features', [])
        q = np.asarray(raw, dtype=np.float32)
        if q.shape != (32, 24):
            continue
        rows.append(matcher._fast_min_distances(q, bool(s.get('body_anchor_valid', True))))
np.save(os.path.join(out, 'baseline_fast_distances.npy'), np.asarray([[r[k] for k in sorted(r)] for r in rows], dtype=np.float32))

total, by_word, sessions = sample_counts(bank)
neg_samples = neg if isinstance(neg, list) else (neg.get('samples') or neg.get('strokes') or [])
status = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True)
with open(os.path.join(out, 'step0.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'out': out, 'bank_sha256': sha(bank_path), 'neg_sha256': sha(neg_path),
        'total_samples': total, 'samples_by_word': by_word, 'samples_by_session': sessions,
        'negative_samples': len(neg_samples), 'git_commit': status.stdout.strip(),
        'baseline_words': sorted(rows[0]) if rows else [], 'baseline_rows': len(rows),
        'rerun_command': 'py scratch/spec11_step0.py',
    }, f, ensure_ascii=False, indent=2)
print(out)
print(json.dumps({'bank_sha256': sha(bank_path), 'neg_sha256': sha(neg_path), 'total': total, 'by_word': by_word, 'by_session': sessions, 'negative': len(neg_samples), 'commit': status.stdout.strip(), 'baseline_rows': len(rows)}, ensure_ascii=False))
