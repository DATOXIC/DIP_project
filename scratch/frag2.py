import csv, json, sys, collections, re
import numpy as np
sys.path.insert(0, '.')
from collect_continuous import _collect_attempts, _events_list, derive_frame_labels, P_SIGN, P_PAIR
from eval_replay_log import attach_words, parse_t0

ev = json.load(open('scratch/replay/events_103331.json', encoding='utf-8'))
z = np.load('scratch/replay/frames_103331.npz', allow_pickle=False)
t = z['t'].astype(float); t_end = float(t[-1]); t_first = float(t[0])
words_by = attach_words(ev)
atts = _collect_attempts(_events_list(ev), t_end)
for aid, a in atts.items(): a['words'] = words_by.get(aid, [])
gt = []
for a in atts.values():
    if a['bad'] or a['discarded'] or not a['signs']: continue
    ss, se = a['signs'][0]
    if a['kind'] == P_SIGN and a['words']: gt.append((a['words'][0], ss, se))
    elif a['kind'] == P_PAIR and len(a['words']) == 2 and len(a['boundaries']) >= 2:
        b0, b1 = sorted(a['boundaries'])[:2]
        gt.append((a['words'][0], ss, b0)); gt.append((a['words'][1], b1, se))
gt.sort(key=lambda x: x[1])
t0 = {'final': float(re.search(r'T0=([0-9.]+)',open('scratch/replay/t0_final.txt',encoding='utf-8').read()).group(1)), 'E': float(re.search(r'T0=([0-9.]+)',open('scratch/replay/t0_E_gap1.2.txt',encoding='utf-8').read()).group(1))}
mode = sys.argv[1] if len(sys.argv) > 1 else 'base'
rows = list(csv.DictReader(open(f'scratch/replay/vc_{mode}.csv', encoding='utf-8')))
dec = []
for r in rows:
    if r['event'] == 'SUPPRESSED_DUP': ev_name = 'SUPP'
    else: ev_name = r['event']
    ts = float(r['t_start']) - t0[mode] + t_first - 1/30
    dec.append((ts, ts + float(r['duration_sec']), ev_name, r['predicted_word']))
print(f'mode={mode} GT words={len(gt)}')
nfrag = collections.Counter(); missed = []; RES=[]
for w, a, b in gt:
    hits = [d for d in dec if min(b, d[1]) - max(a, d[0]) > 0]
    ok = any(d[2] == 'ACCEPTED' and d[3].upper() == w.upper() for d in hits)
    wrong=[(d[2],d[3]) for d in hits if d[2]=='ACCEPTED' and d[3].upper()!=w.upper()]
    RES.append((round(a-t_first,1),w,ok,wrong))
    nfrag[len(hits)] += 1
    if not ok: missed.append((w, round(a - t_first, 1), round(b - a, 2), [(d[2], d[3], round(d[1] - d[0], 2), round(max(a, d[0]) - a, 2)) for d in hits]))
print('so stroke giao moi tu that:', dict(sorted(nfrag.items())))
print('TU KHONG CO ACCEPTED DUNG:')
for m in missed: print('  ', m)


import pickle; pickle.dump(RES,open(f'scratch/replay/res_{mode}.pkl','wb'))
