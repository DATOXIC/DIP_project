import json, sys, numpy as np
sys.path.insert(0,'.')
from eval_baselines import _is_orig, _dtw_call, _traj
from scoreboard import build_matcher_from_config, load_bank, validate_and_hash_config
cfg,_=validate_and_hash_config('configs/G1p90.json')
cit=json.load(open('asl_citizen_templates_24d.json',encoding='utf-8'))['vocabulary']
def auc(pos,neg):  # xac suat r(pos) < r(neg)
    pos,neg=np.array(pos),np.array(neg); return float(np.mean([(p<n)+0.5*(p==n) for p in pos for n in neg]))
for tag,path in (('cu','archive/backups/webcam_templates.truoc_gop_xacnhan_20261003_131806.json'),('hien tai','webcam_templates.json')):
    dat,_=load_bank(path); m=build_matcher_from_config(cfg,dat)
    pos=[];neg=[];rows=[]
    for w,d in cit.items():
        for s in d['samples']:
            if not _is_orig(s): continue
            r=_dtw_call(m,_traj(s),s.get('hand_shapes',['Unknown']*32),s.get('handedness'),bool(s.get('chirality_canonical')),s.get('body_anchor_valid',True),s.get('rep_count'))
            top=r.candidates[0].word_key if r.candidates else None
            if w in dat['vocabulary']:
                if top==w: pos.append(r.r_ratio)
            else: neg.append(r.r_ratio)
    print(f'[{tag}] in-vocab dung top-1: {len(pos)} | OOV: {len(neg)} | AUROC(r_ratio phan biet in-vocab vs OOV) = {auc(pos,neg):.3f}')
    print('    r_ratio trung vi: in-vocab', round(float(np.median(pos)),2), '| OOV', round(float(np.median(neg)),2))
    for thr in (1.5,2.0,2.5,3.0,3.5):
        print(f'    nguong r<={thr}: giu in-vocab {100*np.mean(np.array(pos)<=thr):.0f}% | nhan nham OOV {100*np.mean(np.array(neg)<=thr):.0f}%')
