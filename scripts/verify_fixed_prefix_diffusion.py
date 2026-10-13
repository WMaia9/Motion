"""Independent replay of mining, interpolation and frozen-study checks."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from run_fixed_prefix_diffusion import ROOT, OLD, ARMS, align_negative
import run_physical_semantic_ddpm as previous
from verify_signspark_studies import sha, freeze


def verify():
    for name, digest in json.loads((ROOT/'sources/protocol_sha256.json').read_text()).items():
        assert sha(ROOT/name) == digest
    audit=json.loads((ROOT/'data_audit.json').read_text())
    assert len(audit['train_ids'])==7096 and len(audit['dev_ids'])==519
    assert not set(audit['train_ids']) & set(audit['dev_ids'])
    train=previous.dataset('train');lookup={e.sequence:i for i,e in enumerate(train)}
    captions=[tuple(e.txt) for e in train];first=None;gradients={}
    for arm in ARMS:
        history=json.loads((ROOT/(arm+'_history.json')).read_text())
        assert len(history)==5 and all(x['semantic_examples']==1776 for x in history)
        assert all(np.isfinite(x['train_mse']) and np.isfinite(x['semantic_term']) and
                   np.isfinite(x['dev_fixed_noise_mse']) for x in history)
        state=torch.load(ROOT/(arm+'.pt'),map_location='cpu',weights_only=False)
        assert state['epoch']==5 and all(torch.isfinite(v).all() for v in state['state_dict'].values())
        stochastic=json.loads((ROOT/(arm+'_stochastic_audit.json')).read_text())
        if first is None:first=stochastic
        assert stochastic==first
        negatives=json.loads((ROOT/(arm+'_negative_audit.json')).read_text())
        assert len(negatives)==111
        for record in negatives:
            assert record['all_captions_distinct']
            batch=stochastic[record['batch']];ids=batch['ids'];drop=batch['drop'];ts=batch['timesteps']
            expected=[];partners=[]
            for j in range(len(ids)):
                if drop[j] or ts[j]>800:continue
                choices=[k for k in range(len(ids)) if not drop[k] and captions[ids[k]]!=captions[ids[j]]]
                if not choices:continue
                tokens=set(captions[ids[j]])
                similarities={k:len(tokens&set(captions[ids[k]]))/max(1,len(tokens|set(captions[ids[k]]))) for k in choices}
                k=(min(choices,key=lambda k:(k-j)%len(ids)) if arm.endswith('cyclic') else
                   min(choices,key=lambda k:(-similarities[k],ids[k])))
                expected.append(train[ids[j]].sequence);partners.append(train[ids[k]].sequence)
                if len(expected)==16:break
            assert record['positive_ids']==expected and record['negative_ids']==partners
            assert all(name in lookup for name in expected+partners)
        gradient=json.loads((ROOT/(arm+'_gradient_audit.json')).read_text());gradients[arm]=gradient
        assert gradient['same_batch_prefix_target_masks'] and gradient['positive']['gradient_norm']>0
        if arm.startswith('physical_'):
            assert gradient['positive']['maximum_vertical_dot']<1e-6
            if gradient['negative'] is not None:assert gradient['negative']['maximum_vertical_dot']<1e-6
        else:assert gradient['positive']['maximum_vertical_dot']>1e-5
        assert gradient['negative_receives_gradient']==('generated_' in arm)
    # NumPy interpolation independently checks the source/target duration contract.
    generator=torch.Generator().manual_seed(2027)
    x=torch.randn(3,13,240,dtype=torch.float64,generator=generator,requires_grad=True)
    length=torch.tensor([7,11,13]);chosen=[0,2];negative=[1,0]
    aligned=align_negative(x,length,chosen,negative,13)
    for row,(i,j) in enumerate(zip(chosen,negative)):
        source=x[j,:length[j]].detach().numpy();axis=np.arange(int(length[j]))
        expected=np.stack([np.interp(np.linspace(0,int(length[j])-1,int(length[i])),axis,source[:,k]) for k in range(240)],1)
        assert np.max(abs(expected-aligned[row,:length[i]].detach().numpy()))<1e-12
        assert not aligned[row,length[i]:].detach().any()
    aligned.square().sum().backward();assert x.grad.norm()>0
    count=0
    for name in ARMS+['physical_generated_near_shuffled']:
        result=json.loads((ROOT/(name+'_scores.json')).read_text());assert result['clips']==128
        assert result['clip_ids']==[e['clip_id'] for e in audit['selection']]
        with np.load(ROOT/(name+'_poses.npz')) as data:
            assert len(data.files)==128
            for i,e in enumerate(audit['selection']):
                value=data[f'clip{i:04d}'];assert np.isfinite(value).all()
                assert value.shape==(min(300,e['frames']),240)
        count+=128
    manifest=json.loads((OLD/'manifest_sha256.json').read_text())
    assert all(sha(OLD/name)==digest for name,digest in manifest.items())
    return {'train_dev_overlap':0,'new_training_arms':5,'epochs_per_arm':5,
        'new_generated_clips':count,'first_epoch_stochastic_audits_equal':True,
        'negative_pairs_reconstructed_from_TRAIN_only':True,
        'negative_interpolation_independently_checked':True,
        'negative_gradient_paths_confirmed':True,'gradient_audits':gradients,
        'previous_frozen_report_verified_files':len(manifest),'TEST_used':False,
        'multiple_train_seeds':False,'mesh_rendered':False}


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='freeze':
        freeze(ROOT,json.loads((ROOT/'verification_pending_freeze.json').read_text()))
    else:
        result=verify();(ROOT/'verification_pending_freeze.json').write_text(json.dumps(result,indent=2)+'\n')
        print('FIXED_PREFIX_VERIFIED',json.dumps(result),flush=True)
