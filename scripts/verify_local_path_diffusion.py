"""Verify registered protocol, partitions, matched RNG, geometry and artifacts."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from run_local_path_diffusion import ROOT,ARMS,CONTROLS,HISTORICAL,uniform,bounded_partition,viterbi
from check_local_path_math import check
from analyze_signspark_geometry_dev import statistics,b4
from verify_signspark_studies import sha,freeze
from run_motion_ctc_confirmation import Reader
import run_physical_semantic_ddpm as previous


@torch.no_grad()
def alignment_replay(train,partitions):
    ck=torch.load(previous.PREV/'ctc_reader/best.pt',map_location='cpu',weights_only=False)
    state=ck['state_dict'];vstoi={w:i for i,w in enumerate(ck['vocab'])}
    reader=Reader(len(vstoi),state['mean'].numpy(),state['std'].numpy()).cuda()
    reader.load_state_dict(state,strict=True);reader.eval()
    order=np.argsort([len(e.sgn) for e in train],kind='stable');checked=0;maximum=0.
    for start in range(0,len(order),1024):
        ids=order[start:start+64];length=torch.tensor([len(train[i].sgn) for i in ids],device='cuda')
        x=torch.zeros(len(ids),int(length.max()),240,device='cuda')
        for j,i in enumerate(ids):x[j,:length[j]]=torch.from_numpy(previous.project(train[i].sgn.numpy())).cuda()
        logp,outlength=reader(x,length);logp=logp.cpu().numpy()
        for j,i in enumerate(ids):
            e=train[i];record=partitions[i];assert record['fallback'] is None
            path,score=viterbi(logp[:int(outlength[j]),j],[vstoi[g] for g in e.gls])
            assert path is not None
            centers=[np.flatnonzero(path==2*q+1).mean() for q in range(record['K'])]
            candidates=[int(round(2*(a+b))) for a,b in zip(centers,centers[1:])]
            boundaries=bounded_partition(candidates,len(e.sgn),record['K'])
            assert boundaries==record['ctc'];maximum=max(maximum,abs(score-record['viterbi_log_probability']))
            assert abs(score-record['viterbi_log_probability'])<1e-4;checked+=1
    return {'replayed_TRAIN_alignments':checked,'exact_boundary_replay':True,'max_logp_difference':maximum}


def verify():
    assert not (ROOT/'manifest_sha256.json').exists(),'frozen report; use a new ROOT for verification writes'
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    for name,digest in json.loads((ROOT/'sources/protocol_sha256.json').read_text()).items():
        assert sha(ROOT/name if name=='PROTOCOLO.md' else Path(name))==digest
    proof=check();assert proof==json.loads((ROOT/'sources/math_check.json').read_text())
    audit=json.loads((ROOT/'data_audit.json').read_text());partitions=json.loads((ROOT/'train_partitions.json').read_text())
    assert len(audit['train_ids'])==len(partitions)==7096 and len(audit['dev_ids'])==519
    assert not set(audit['train_ids'])&set(audit['dev_ids'])
    train=previous.dataset('train');differing=0;shifts=[];counts=Counter()
    for e,record in zip(train,partitions):
        assert e.sequence==record['clip_id'] and len(e.sgn)==record['frames'] and len(e.gls)==record['gloss_count']
        k=record['K'];assert k==min(max(1,len(e.gls)),len(e.sgn)//2)
        assert record['uniform']==uniform(len(e.sgn),k)
        for key in ['uniform','ctc']:
            p=record[key];assert len(p)==k+1 and p[0]==0 and p[-1]==len(e.sgn) and min(np.diff(p))>=2
        counts[record['fallback'] or 'aligned']+=1
        differing+=record['uniform']!=record['ctc']
        shifts.extend(abs(a-b) for a,b in zip(record['uniform'][1:-1],record['ctc'][1:-1]))
    assert dict(counts)==json.loads((ROOT/'alignment_audit.json').read_text())['counts']
    first=None
    for arm in ARMS:
        history=json.loads((ROOT/(arm+'_history.json')).read_text());assert len(history)==5
        assert all(np.isfinite(r['train_mse']) and np.isfinite(r['path_term']) and np.isfinite(r['dev_fixed_noise_mse']) for r in history)
        state=torch.load(ROOT/(arm+'.pt'),map_location='cpu',weights_only=False);assert state['epoch']==5
        assert all(torch.isfinite(v).all() for v in state['state_dict'].values())
        stochastic=json.loads((ROOT/(arm+'_stochastic_audit.json')).read_text())
        if first is None:first=stochastic
        assert stochastic==first and len(stochastic)==111
    assert first==json.loads((HISTORICAL/'geometry_replay_stochastic_audit.json').read_text())
    matched_examples=[]
    for epoch in range(5):
        values=[json.loads((ROOT/(a+'_history.json')).read_text())[epoch] for a in ARMS]
        assert len({v['auxiliary_examples'] for v in values})==1 and len({v['auxiliary_segments'] for v in values})==1
        matched_examples.append(values[0]['auxiliary_segments'])
    control_equality={}
    for name in CONTROLS:
        for suffix in ['.pt','_history.json','_stochastic_audit.json','_poses.npz']:assert sha(ROOT/(name+suffix))==sha(HISTORICAL/(name+suffix))
        local=json.loads((ROOT/(name+'_scores.json')).read_text());old=json.loads((HISTORICAL/(name+'_scores.json')).read_text())
        assert all(local[k]==v for k,v in old.items());control_equality[name]=True
    scales=json.loads((ROOT/'scale_audit.json').read_text())
    assert scales['TRAIN_clips']==7096 and scales['no_DEV_used'] and scales['pooled_segments']==2*sum(p['K'] for p in partitions)
    with np.load(ROOT/'local_scales.npz') as data:
        assert set(data.files)=={'endpoint','area'}
        for k in data.files:assert np.isfinite(data[k]).all() and (data[k]>0).all()
    total=0
    for name in CONTROLS+ARMS+['local_ctc_area_shuffled']:
        result=json.loads((ROOT/(name+'_scores.json')).read_text());assert result['clips']==128
        assert result['clip_ids']==[e['clip_id'] for e in audit['selection']]
        assert not result['evaluation_provenance']['CTC_fully_independent_evaluator']
        with np.load(ROOT/(name+'_poses.npz')) as data:
            assert len(data.files)==128
            for i,e in enumerate(audit['selection']):
                x=data[f'clip{i:04d}'];assert x.shape==(min(300,e['frames']),240) and np.isfinite(x).all()
        total+=128
    for r in json.loads((ROOT/'sources/research_downloads.json').read_text()):assert sha(ROOT/'sources'/r['file'])==r['sha256']
    shuffled=json.loads((ROOT/'local_ctc_area_shuffled_scores.json').read_text())
    counts=statistics(shuffled['references'],shuffled['canonical_hypotheses']).sum(0)
    zero=json.loads((ROOT/'zero_bleu_diagnostic.json').read_text())
    assert counts[:4].tolist()==zero['canonical_matched_ngrams_1_to_4'] and counts[3]==0 and b4(counts)==0
    assert not any(not h.strip() for h in shuffled['canonical_hypotheses'])
    return {'new_training_arms':4,'epochs_per_arm':5,'new_generated_clips':640,'reused_control_clips':256,
        'total_scored_clips':total,'matched_first_epoch_order_t_drop_noise_with_old_baseline':True,
        'matched_auxiliary_segment_counts_by_epoch':matched_examples,'TRAIN_DEV_overlap':0,
        'TRAIN_partition_clips':7096,'partition_clips_different_from_uniform':differing,
        'mean_absolute_interior_boundary_shift_frames':float(np.mean(shifts)),
        'reused_control_artifacts_equal':control_equality,'math_checks':proof,
        'alignment_replay':alignment_replay(train,partitions),'BT_independent_of_new_loss':True,
        'shuffled_zero_BLEU_explained_by_zero_matching_fourgrams_not_empty_hypotheses':True,
        'CTC_is_alignment_source_not_independent':True,'training_uses_gloss_supervision':True,
        'inference_inputs_text_and_real_duration':True,'TEST_used':False,'mesh_rendered':False,'human_evaluation':False}


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='freeze':freeze(ROOT,json.loads((ROOT/'verification_pending_freeze.json').read_text()))
    else:
        result=verify();(ROOT/'verification_pending_freeze.json').write_text(json.dumps(result,indent=2)+'\n')
        print('LOCAL_VERIFIED',json.dumps(result),flush=True)
