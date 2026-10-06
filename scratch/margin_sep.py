import csv, json, numpy as np
import eval_replay_log as E
from collect_continuous import _collect_attempts, _events_list, P_SIGN, P_PAIR

ev = json.load(open('scratch/replay/events_103331.json', encoding='utf-8'))
z = np.load('scratch/replay/frames_103331.npz', allow_pickle=False); t = z['t'].astype(float)
atts = _collect_attempts(_events_list(ev), float(t[-1]))
wb = E.attach_words(ev)
gt = []
for aid, a in atts.items():
    a['words'] = wb.get(aid, [])
    if a['bad'] or a['discarded'] or not a['signs']: continue
    ss, se = a['signs'][0]
    if a['kind'] == P_SIGN and a['words']: gt.append((a['words'][0], ss, se))
    elif a['kind'] == P_PAIR and len(a['words']) == 2 and len(a['boundaries']) >= 2:
        b0, b1 = sorted(a['boundaries'])[:2]; gt += [(a['words'][0], ss, b0), (a['words'][1], b1, se)]
t0 = E.parse_t0('scratch/spnew/h_nosp.txt')['nosp']
rows = list(csv.DictReader(open('scratch/spnew/nosp.csv', encoding='utf-8')))
t_first = t[0]; used = {}; good = []; bad = []


def feat_of(r):
    return (float(r['r_ratio']), float(r['margin_pct']), float(r['match_quality']), float(r['best_distance']), float(r['duration_sec']))


for r in rows:
    if r['event'] != 'ACCEPTED': continue
    ts = float(r['t_start']) - t0; te = ts + float(r['duration_sec'])
    a, b = t_first + ts - 1 / 30, t_first + te - 1 / 30
    hit = [i for i, (_, x, y) in enumerate(gt) if min(y, b) - max(x, a) > 0]
    if hit:
        i = max(hit, key=lambda k: min(gt[k][2], b) - max(gt[k][1], a))
        ok = used.get(i, 0) == 0 and gt[i][0].upper() == r['predicted_word'].upper()
        used[i] = used.get(i, 0) + 1
        (good if ok else bad).append(feat_of(r))
    else:
        bad.append(feat_of(r))
nsig_bad = len(bad)
for tag in ['20261003_163235_neg', '20261003_163834_neg', '20261003_164439_neg']:
    for r in csv.DictReader(open(f'scratch/negeval/{tag}_0.csv', encoding='utf-8')):
        if r['event'] == 'ACCEPTED':
            bad.append(feat_of(r))
G, B = np.array(good), np.array(bad)
print('dung', len(G), 'sai', len(B), '(trong do phien ky', nsig_bad, ')')
names = ['r_ratio(thap tot)', 'margin_pct(cao tot)', 'match_quality(cao tot)', 'best_dist(thap tot)', 'duration']
for k, n in enumerate(names):
    print(f'{n:<24} dung med {np.median(G[:, k]):.2f} [{np.percentile(G[:, k], 10):.2f},{np.percentile(G[:, k], 90):.2f}]  sai med {np.median(B[:, k]):.2f} [{np.percentile(B[:, k], 10):.2f},{np.percentile(B[:, k], 90):.2f}]')


def sweep(k, higher_better, qs):
    print('--', names[k])
    for q in qs:
        th = np.percentile(G[:, k], q)
        keep_g = (G[:, k] >= th).mean() if higher_better else (G[:, k] <= th).mean()
        keep_b = (B[:, k] >= th).mean() if higher_better else (B[:, k] <= th).mean()
        print(f'  nguong {th:.2f}: giu {keep_g * 100:.0f}% tu dung, con {keep_b * 100:.0f}% bao nham ({int(round(keep_b * len(B)))}/{len(B)})')


sweep(0, False, [100, 95, 90, 80]); sweep(1, True, [0, 5, 10, 20]); sweep(2, True, [0, 5, 10, 20])

print('\n=== tach theo r_ratio > 1.7 (fusion ghi de) ===')
thr = 1.7
print('phien ky: dung r<=1.7:', int((G[:,0] <= thr).sum()), ' dung r>1.7:', int((G[:,0] > thr).sum()))
Bs = B[:nsig_bad]; Bn = B[nsig_bad:]
print('phien ky: sai r<=1.7:', int((Bs[:,0] <= thr).sum()), ' sai r>1.7:', int((Bs[:,0] > thr).sum()))
print('mau am  : sai r<=1.7:', int((Bn[:,0] <= thr).sum()), ' sai r>1.7:', int((Bn[:,0] > thr).sum()))
