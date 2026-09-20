# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import chain_replay

def main():
    ns = argparse.Namespace(
        bank='scratch/bank_snapshot_clean.json', raw_dir='raw_strokes', exclude_session_prefix=[],
        threshold_k=1.75, uncertainty_k=2.35, all_strokes=False, study=None
    )
    matcher, words, n_map, tk, uk = chain_replay.setup(ns)
    sentences = chain_replay.make_sentences(words, 200, 20260920)

    configs = {
        'S0_fast_chain': dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(chain_replay.DEFAULT_LEAD_TRIM_CONFIG, enabled=False), transition_lockout_sec=0.0),
        'S1_lockout_0.10': dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(chain_replay.DEFAULT_LEAD_TRIM_CONFIG, enabled=False), transition_lockout_sec=0.10),
        'S2_lead_trim': dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(chain_replay.DEFAULT_LEAD_TRIM_CONFIG, enabled=True, max_variants=3, max_sec=0.45, good_r=0.80, min_gain=0.10, trim_penalty_per_sec=0.30), transition_lockout_sec=0.0),
        'S3_both': dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(chain_replay.DEFAULT_LEAD_TRIM_CONFIG, enabled=True, max_variants=3, max_sec=0.45, good_r=0.80, min_gain=0.10, trim_penalty_per_sec=0.30), transition_lockout_sec=0.10),
        'S4_baseline_prod': None,
    }

    out_tuning = {}
    cache = {}
    mp = chain_replay.multi_phase_for(matcher, chain_replay.CONFIG['fast_chain'])

    for name, cfg in configs.items():
        res = chain_replay.run_level(sentences, 0.0, 15.0, 'realistic', matcher, cfg, mp, 20260920, cache=cache)
        out_tuning[name] = res
        acc_str = f"{100.0*res['acc']:.1f}%"
        wacc_str = f"{100.0*res['word_acc']:.1f}%"
        cnt_str = f"{res['leadin_contam']} ({100.0*res['leadin_contam_rate']:.1f}%)"
        hl_str = f"{res['head_loss_words']} ({100.0*res['head_loss_rate']:.1f}%)"
        lat_str = f"{res['lat_median']:.3f}s"
        hitch_str = f"{res['hitch_p95_ms']:.1f}ms"
        print(f"{name:20s}: Acc={acc_str} | WordAcc={wacc_str} | Contam={cnt_str} | HeadLoss={hl_str} | Lat={lat_str} | HitchP95={hitch_str}", flush=True)

    with open('scratch/tuning_results_15hz.json', 'w', encoding='utf-8') as f:
        json.dump(out_tuning, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
