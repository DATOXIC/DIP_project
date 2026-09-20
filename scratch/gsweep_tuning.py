# c:/DIP/DIP_project/scratch/gsweep_tuning.py
import argparse, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import chain_replay
from lead_trim import DEFAULT_LEAD_TRIM_CONFIG

ns = argparse.Namespace(bank='scratch/bank_snapshot_clean.json', raw_dir='raw_strokes', exclude_session_prefix=[], threshold_k=1.75, uncertainty_k=2.35, all_strokes=False, study=None)
matcher, words, n_map, tk, uk = chain_replay.setup(ns)
sentences_A = chain_replay.make_sentences(words, 200, 20260920)
mp = chain_replay.multi_phase_for(matcher, chain_replay.CONFIG['fast_chain'])

S0 = dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(DEFAULT_LEAD_TRIM_CONFIG, enabled=False), transition_lockout_sec=0.0)
# S2 voi cau hinh bao thu hon: min_gain=0.20, good_r=0.80, penalty=0.60, min_gain_other_word=0.35
S2 = dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(DEFAULT_LEAD_TRIM_CONFIG, enabled=True, max_variants=3, max_sec=0.45, good_r=0.80, min_gain=0.20, trim_penalty_per_sec=0.60, min_gain_other_word=0.35), transition_lockout_sec=0.0)
S4 = None

cache = {}
print('G-sweep tap chinh (seed A=20260920, 200 cau, 30 Hz, realistic)')
print('g(s) | cfg | CauDung | TuDung | Nhiem(%) | MatDau(%) | HitchP95(ms)')
for g in [0.10, 0.20, 0.30]:
    for name, cfg in [('S0_fast',S0),('S2_lead_trim',S2),('S4_prod',S4)]:
        res = chain_replay.run_level(sentences_A, g, 30.0, 'realistic', matcher, cfg, mp, 20260920, cache=cache)
        print(f'{g:.2f} | {name:15s} | {round(100*res["acc"],1)}% ({round(res["acc"]*res["sentences"])}/{res["sentences"]}) | {round(100*res["word_acc"],1)}% | {res["leadin_contam"]}({round(100*res["leadin_contam_rate"],1)}%) | {res["head_loss_words"]}({round(100*res["head_loss_rate"],1)}%) | {round(res["hitch_p95_ms"],1)}ms', flush=True)

print()
print('G-sweep 15 Hz (seed A, 200 cau, realistic)')
for g in [0.10, 0.20, 0.30]:
    for name, cfg in [('S0_fast',S0),('S2_lead_trim',S2),('S4_prod',S4)]:
        res = chain_replay.run_level(sentences_A, g, 15.0, 'realistic', matcher, cfg, mp, 20260920, cache={})
        print(f'15Hz g={g:.2f} | {name:15s} | {round(100*res["acc"],1)}% | Contam={res["leadin_contam"]}({round(100*res["leadin_contam_rate"],1)}%) | HeadLoss={res["head_loss_words"]}({round(100*res["head_loss_rate"],1)}%)', flush=True)
