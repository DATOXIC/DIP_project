# c:/DIP/DIP_project/scratch/loo_grid_search.py
import argparse, sys, os, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import chain_replay
from lead_trim import select_best_lead_trim, DEFAULT_LEAD_TRIM_CONFIG

ns = argparse.Namespace(bank='scratch/bank_snapshot_clean.json', raw_dir='raw_strokes', exclude_session_prefix=[], threshold_k=1.75, uncertainty_k=2.35, all_strokes=False, study=None)
matcher, words, n_map, tk, uk = chain_replay.setup(ns)

def loo_test(cfg_override):
    cfg = dict(DEFAULT_LEAD_TRIM_CONFIG, **cfg_override)
    top1_ok = acc_ok = wrong_acc = n = tau_gt0 = 0
    hitch_times = []
    tail_pad_sec = chain_replay.CONFIG['fast_chain'].get('tail_pad_sec', 0.45)
    for w in words:
        frames = []
        for i in range(len(w.t)):
            frames.append(dict(x=float(w.wrist[i,0]), y=float(w.wrist[i,1]), palm_len=max(float(w.palm[i]),15.0),
                               hand_shape=w.shapes[i], t=float(w.t[i]), angles=w.angles[i], normal=w.normals[i],
                               landmarks=w.lm[i], face_box=w.face[i]))
        matcher.excluded = (matcher.global_index(w.bank_key),) if w.bank_key is not None else ()
        ts = time.time()
        res = select_best_lead_trim(frames, matcher, cfg, handedness=w.hand[-1], query_has_body=True, tail_pad_sec=tail_pad_sec)
        hitch_times.append((time.time()-ts)*1000)
        n += 1
        ok = (res.match_result.top_word == w.word)
        top1_ok += ok
        if res.match_result.status == 'ACCEPTED':
            acc_ok += ok
            wrong_acc += (not ok)
        tau_gt0 += int(res.trim_sec > 0)
    hp95 = float(np.percentile(hitch_times, 95))
    return dict(n=n, top1=top1_ok, acc_ok=acc_ok, wrong_acc=wrong_acc, tau_gt0=tau_gt0, hitch_p95_ms=hp95)

print('LOO Grid Search: min_gain x good_r x trim_penalty x max_variants')
print('Baseline (no trim): top1=203/203 acc=203 wrong=0 tau_gt0=0')
for mv in [2, 3]:
    for mg in [0.15, 0.20, 0.25]:
        for gr in [0.75, 0.80]:
            for pen in [0.40, 0.60, 0.80]:
                r = loo_test(dict(enabled=True, max_variants=mv, max_sec=0.45, good_r=gr, min_gain=mg, trim_penalty_per_sec=pen, min_gain_other_word=mg+0.15))
                reg = r['n'] - r['top1']
                flag = 'OK' if reg <= 1 and r['hitch_p95_ms'] <= 25 else ('REG>1' if reg > 1 else 'HITCH')
                print(f'[{flag}] mv={mv} mg={mg} gr={gr} pen={pen}: top1={r["top1"]}/{r["n"]} acc={r["acc_ok"]} wrong={r["wrong_acc"]} tau_gt0={r["tau_gt0"]} hp95={r["hitch_p95_ms"]:.1f}ms')
