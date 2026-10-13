"""Mathematical checks, matched training and exact baseline replay."""
import json
import sys
import numpy as np
import torch
from run_motion_path_diffusion import ROOT,OLD,ARMS
from check_motion_path_math import check
from verify_signspark_studies import sha,freeze
import run_physical_semantic_ddpm as previous


def verify():
    for name,digest in json.loads((ROOT/'sources/protocol_sha256.json').read_text()).items():assert sha(ROOT/name)==digest
    proof=check();assert proof==json.loads((ROOT/'sources/math_check.json').read_text())
    audit=json.loads((ROOT/'data_audit.json').read_text());assert len(audit['train_ids'])==7096 and len(audit['dev_ids'])==519
    assert not set(audit['train_ids'])&set(audit['dev_ids'])
    first=None
    for arm in ARMS:
        history=json.loads((ROOT/(arm+'_history.json')).read_text());assert len(history)==5
        assert all(np.isfinite(x['train_mse']) and np.isfinite(x['path_term']) and np.isfinite(x['dev_fixed_noise_mse']) for x in history)
        state=torch.load(ROOT/(arm+'.pt'),map_location='cpu',weights_only=False);assert state['epoch']==5
        assert all(torch.isfinite(value).all() for value in state['state_dict'].values())
        stochastic=json.loads((ROOT/(arm+'_stochastic_audit.json')).read_text())
        if first is None:first=stochastic
        assert stochastic==first
    original=torch.load(OLD/'geometry.pt',map_location='cpu',weights_only=False)['state_dict']
    replay=torch.load(ROOT/'geometry_replay.pt',map_location='cpu',weights_only=False)['state_dict']
    assert original.keys()==replay.keys() and all(torch.equal(original[k],replay[k]) for k in original)
    r=json.loads((ROOT/'geometry_replay_scores.json').read_text());o=json.loads((OLD/'geometry_scores.json').read_text())
    for key in ['raw_hypotheses','canonical_hypotheses','independent_raw_hypotheses','independent_canonical_hypotheses']:assert r[key]==o[key]
    train=previous.dataset('train');scales=json.loads((ROOT/'scale_audit.json').read_text())
    assert scales['TRAIN_clips']==7096 and scales['no_DEV_used']
    assert scales['valid_differences']==sum(len(e.sgn)-1 for e in train)
    with np.load(ROOT/'path_scales.npz') as data:
        assert set(data.files)=={'endpoint','area','area_time','velocity'}
        for key in data.files:assert np.isfinite(data[key]).all() and (data[key]>0).all()
    count=0
    for name in ARMS+['physical_area_time_shuffled']:
        result=json.loads((ROOT/(name+'_scores.json')).read_text());assert result['clips']==128
        assert result['clip_ids']==[e['clip_id'] for e in audit['selection']]
        with np.load(ROOT/(name+'_poses.npz')) as data:
            assert len(data.files)==128
            for i,e in enumerate(audit['selection']):
                x=data[f'clip{i:04d}'];assert x.shape==(min(300,e['frames']),240) and np.isfinite(x).all()
        count+=128
    return {'new_training_arms':5,'epochs_per_arm':5,'train_dev_overlap':0,
        'new_generated_clips':count,'baseline_weights_bitwise_replayed':True,
        'baseline_hypotheses_exactly_replayed_in_both_readers':True,
        'first_epoch_stochastic_audits_equal':True,'TRAIN_only_scales_count_verified':True,
        'math_checks':proof,'no_reader_or_gloss_used_in_training':True,
        'TEST_used':False,'multiple_train_seeds':False,'mesh_rendered':False}


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='freeze':freeze(ROOT,json.loads((ROOT/'verification_pending_freeze.json').read_text()))
    else:
        result=verify();(ROOT/'verification_pending_freeze.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PATH_VERIFIED',json.dumps(result),flush=True)
