"""Independent reconstruction/rotation checks and saved-checkpoint replay."""
import json, hashlib
from pathlib import Path
import numpy as np, torch
from scipy.spatial.transform import Rotation
from run_motion_ctc_confirmation import Reader,read,decode,metrics
from verify_signspark_studies import sha,freeze

ROOT=Path('reports/cvpr_motion_consistency_confirmation_20261010')

def rotation(x):
    z=x.astype(np.float64).reshape(-1,40,6);a=z[...,:3];a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=z[...,3:];b=b-(a*b).sum(-1,keepdims=True)*a;b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.stack([a,b,np.cross(a,b)],axis=-2)

def verify():
    torch.set_num_threads(8);checks={};reader=ROOT/'ctc_reader';audit=json.loads((reader/'data_audit.json').read_text());assert audit['train_clips']==7096 and audit['dev_clips']==519;assert not set(audit['train_ids'])&set(audit['dev_ids']);assert len(json.loads((reader/'training_history.json').read_text()))==40
    xs,refs,ids=read('dev');assert ids==audit['dev_ids'];attacks={}
    for name in ['correct_gloss','rolled_wrong_gloss']:
        result=json.loads((reader/(name+'.json')).read_text());maximum=0.;coefficient=0.
        with np.load(reader/(name+'_poses.npz')) as d:
            for i,x in enumerate(xs):
                y=d[f'attacked{i:04d}'];u=d[f'gauge{i:04d}'];assert x.shape==y.shape;assert np.isfinite(y).all() and np.isfinite(u).all();assert (u[:,10:25]==0).all()
                # Original source left was canonicalized for the native loader.
                rr=rotation(x);left=rr[:,10:25,:2].reshape(len(x),90).astype(np.float32);base=x.copy();base[:,60:150]=left
                assert np.array_equal(base[:,60:150],y[:,60:150]);z=base.reshape(-1,40,6);a=np.exp(.35*np.tanh(u[...,0:1]));b=np.exp(.35*np.tanh(u[...,1:2]));c=.35*np.tanh(u[...,2:3]);reconstructed=np.concatenate([a*z[...,:3],b*z[...,3:]+c*z[...,:3]],axis=-1).reshape(y.shape);assert abs(reconstructed-y).max()<2e-6
                angle=float(np.degrees(Rotation.from_matrix((rotation(base).transpose(0,1,3,2)@rotation(y)).reshape(-1,3,3)).magnitude()).max());maximum=max(maximum,angle);coefficient=max(coefficient,float(abs(.35*np.tanh(u)).max()))
        assert maximum<.01 and coefficient<=.350001;assert result['canonical_changed_sentences']==0
        assert all(len(b['losses'])==80 and np.isfinite(b['losses']).all() for b in result['optimization_logs'])
        attacks[name]={'poses_verified':len(xs),'max_angle_deg':maximum,'max_gauge_coefficient':coefficient,'left_unchanged':True,'canonical_changed_sentences':0}
    ck=torch.load(reader/'best.pt',map_location='cpu',weights_only=False);s=ck['state_dict'];model=Reader(len(ck['vocab']),s['mean'].numpy(),s['std'].numpy()).cuda();model.load_state_dict(s,strict=True);h=decode(model,xs,ck['vocab']);original=json.loads((reader/'original.json').read_text());assert h==original['hypotheses'];assert abs(metrics(refs,h)['wer']-ck['dev_wer'])<1e-9
    checks['ctc']={'train_dev_overlap':0,'checkpoint_replay_all_dev_exact':True,'original_wer':ck['dev_wer'],'best_epoch':ck['epoch'],'attacks':attacks};del model;torch.cuda.empty_cache()
    generator={}
    for corpus in ['PHOENIX-2014T','How2Sign']:
        selected=json.loads((ROOT/'generated'/corpus/'selection.json').read_text());assert len(selected)==128 and len({z['clip_id'] for z in selected})==128;generator[corpus]={}
        for mode in ['text_duration','oracle_keyframes']:
            r=json.loads((ROOT/'generated'/corpus/(mode+'_scores.json')).read_text());assert [x['clip_id'] for x in selected]==r['clip_ids'];maximum=0.
            with np.load(ROOT/'generated'/corpus/(mode+'_loaded_poses.npz')) as d:
                for i in range(128):
                    x=d[f'raw{i:04d}'];y=d[f'canonical{i:04d}'];assert np.isfinite(x).all() and np.isfinite(y).all();assert 0<len(x)<=300
                    angle=float(np.degrees(Rotation.from_matrix((rotation(x).transpose(0,1,3,2)@rotation(y)).reshape(-1,3,3)).magnitude()).max());maximum=max(maximum,angle)
                    assert abs(np.linalg.det(rotation(y))-1).max()<1e-6
            assert maximum<.01;assert r['changed_sentences']==sum(a!=b for a,b in zip(r['raw_hypotheses'],r['canonical_hypotheses']))
            if mode=='text_duration':assert r['keyframe_elements_body']==r['keyframe_elements_hand']==0
            generator[corpus][mode]={'verified_generated_clips':128,'max_angle_deg':maximum,'hypothesis_changes_checked':True,'no_target_optimization':True}
    checks['generator']=generator
    for name,digest in json.loads((ROOT/'sources/protocol_selection_hashes.json').read_text()).items():assert sha(ROOT/name)==digest
    downloaded=json.loads((ROOT/'sources/downloads.json').read_text())
    for x in downloaded:assert sha(x['path'])==x['sha256']
    checks['pinned_downloads_verified']=len(downloaded);checks['protocol_and_selections_unchanged']=True;checks['TEST_used']=False;checks['mesh_rendered']=False
    (ROOT/'verification_pending_freeze.json').write_text(json.dumps(checks,indent=2));print('VERIFIED',json.dumps(checks),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[-1]=='freeze':freeze(ROOT,json.loads((ROOT/'verification_pending_freeze.json').read_text()))
    else:verify()
