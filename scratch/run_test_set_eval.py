# -*- coding: utf-8 -*-
import argparse, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import chain_replay
SEED_B = 20260921
def main():
    ns = argparse.Namespace(bank='scratch/bank_snapshot_clean.json', raw_dir='raw_strokes', exclude_session_prefix=[], threshold_k=1.75, uncertainty_k=2.35, all_strokes=False, study=None)
    matcher, words, n_map, tk, uk = chain_replay.setup(ns)
    sentences_test = chain_replay.make_sentences(words, 200, SEED_B)
    configs = {
        'S0_fast_chain': dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(chain_replay.DEFAULT_LEAD_TRIM_CONFIG, enabled=False), transition_lockout_sec=0.0),
        'S2_lead_trim_frozen': dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(chain_replay.DEFAULT_LEAD_TRIM_CONFIG, enabled=True, max_variants=3, max_sec=0.45, good_r=0.80, min_gain=0.10, trim_penalty_per_sec=0.30), transition_lockout_sec=0.0),
        'S3_lockout0.10_lead_trim': dict(chain_replay.CONFIG['fast_chain'], enabled=True, lead_trim=dict(chain_replay.DEFAULT_LEAD_TRIM_CONFIG, enabled=True, max_variants=3, max_sec=0.45, good_r=0.80, min_gain=0.10, trim_penalty_per_sec=0.30), transition_lockout_sec=0.10),
        'S4_baseline_prod': None,
    }
    mp = chain_replay.multi_phase_for(matcher, chain_replay.CONFIG['fast_chain'])
    cache = {}
    results = {}
    print('30 Hz realistic g=0.0 TEST SET (seed B=' + str(SEED_B) + ')')
    for name, cfg in configs.items():
        res = chain_replay.run_level(sentences_test, 0.0, 30.0, 'realistic', matcher, cfg, mp, SEED_B, cache=cache)
        results[name] = res
        print(name + ': Acc=' + str(round(100.0*res['acc'],1)) + '% | WdAcc=' + str(round(100.0*res['word_acc'],1)) + '% | Contam=' + str(res['leadin_contam']) + '(' + str(round(100.0*res['leadin_contam_rate'],1)) + '%) | HeadLoss=' + str(res['head_loss_words']) + '(' + str(round(100.0*res['head_loss_rate'],1)) + '%) | Lat=' + str(round(res['lat_median'],3)) + '/' + str(round(res['lat_p95'],3)) + 's | HitchP95=' + str(round(res['hitch_p95_ms'],1)) + 'ms', flush=True)
    with open('scratch/test_set_results_30hz.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print()
    print('15 Hz realistic g=0.0 TEST SET')
    results_15 = {}
    for name, cfg in configs.items():
        res = chain_replay.run_level(sentences_test, 0.0, 15.0, 'realistic', matcher, cfg, mp, SEED_B, cache={})
        results_15[name] = res
        print(name + ': Acc=' + str(round(100.0*res['acc'],1)) + '% | Contam=' + str(res['leadin_contam']) + '(' + str(round(100.0*res['leadin_contam_rate'],1)) + '%) | HeadLoss=' + str(res['head_loss_words']) + '(' + str(round(100.0*res['head_loss_rate'],1)) + '%)', flush=True)
    with open('scratch/test_set_results_15hz.json', 'w', encoding='utf-8') as f:
        json.dump(results_15, f, ensure_ascii=False, indent=2)
    print('Xong.')
if __name__ == '__main__':
    main()
