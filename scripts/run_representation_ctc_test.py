"""Frozen second architecture: fixed TEST audit, no new fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from run_evaluation_heldout import dataset,geometry
from run_signspark_geometry_dev import project
from run_signspark_gauge_attack import transform
from run_motion_ctc_confirmation import Reader,batch,decode,metrics
from verify_signspark_studies import sha

ROOT=Path('reports/cvpr_representation_paper_20261011')
PRIOR=Path('reports/cvpr_evaluation_confirmation_20261011')
CHECKPOINT=Path('reports/cvpr_motion_consistency_confirmation_20261010/ctc_reader/best.pt')


def interval(reference,hypothesis,before):
    after=metrics(reference,hypothesis);base=metrics(reference,before);n=len(reference)
    weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=2000)
    den=weights@np.asarray(after['reference_lengths']);delta=100*((weights@np.asarray(after['errors']))-(weights@np.asarray(base['errors'])))/den
    return {'delta_WER':after['wer']-base['wer'],'ci95':np.percentile(delta,[2.5,97.5]).tolist()}


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    registered=json.loads((ROOT/'sources/registered_inputs.json').read_text())
    assert sha(ROOT/'PROTOCOLO.md')==registered['protocol_sha256'] and sha(CHECKPOINT)==registered['ctc_checkpoint_sha256']
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.enabled=False
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);state=ck['state_dict'];vocab=ck['vocab'];stoi={w:i for i,w in enumerate(vocab)}
    model=Reader(len(vocab),state['mean'].numpy(),state['std'].numpy()).cuda();model.load_state_dict(state,strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    entries,_=dataset('PHOENIX-2014T');xs=[e.sgn.numpy().copy() for e in entries];refs=[' '.join(e.gls) for e in entries]
    ids=[e.sequence for e in entries];audit=json.loads((PRIOR/'test_audit.json').read_text())['PHOENIX-2014T'];assert ids==audit['test_ids']
    native=json.loads(Path('reports/cvpr_motion_consistency_confirmation_20261010/ctc_reader/data_audit.json').read_text())
    assert not set(ids)&set(native['train_ids']) and not set(ids)&set(native['dev_ids'])
    folder=ROOT/'ctc_TEST';controls={}
    identity=[]
    for x in xs:
        y=x.copy();y[:,60:]=np.tile([1,0,0,0,1,0],30);identity.append(y)
    for name,poses in [('raw',xs),('canonical',[project(x) for x in xs]),('reverse_time',[x[::-1].copy() for x in xs]),('identity_hands',identity)]:
        hyps=decode(model,poses,vocab);controls[name]={'hypotheses':hyps,'metrics':metrics(refs,hyps)}
        print('CTC_TEST_CONTROL',name,controls[name]['metrics']['wer'],flush=True)
    (folder/'full_controls.json').write_text(json.dumps({'clips':len(xs),'clip_ids':ids,'references':refs,'controls':controls},ensure_ascii=False,indent=2)+'\n')
    indices=audit['attack_indices'];selected=[entries[i] for i in indices];xs=[xs[i] for i in indices];refs=[refs[i] for i in indices]
    selection=[{'clip_id':e.sequence,'frames':len(e.sgn),'reference':r,'test_index':int(i)} for e,r,i in zip(selected,refs,indices)]
    (folder/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    raw=decode(model,xs,vocab);canonical=decode(model,[project(x) for x in xs],vocab)
    mask=torch.ones(1,1,40,1,device='cuda');mask[:,:,10:25]=0;lossfn=nn.CTCLoss(blank=0,zero_infinity=True);summary={}
    for condition in ['correct_gloss','rolled_wrong_gloss']:
        targets=refs if condition=='correct_gloss' else refs[1:]+refs[:1];outputs=[];gauges=[];logs=[];infeasible=0
        unknown=sum(w not in stoi for r in targets for w in r.split())
        for start in range(0,128,32):
            inds=list(range(start,start+32));x,length=batch(xs,inds);d=x.detach().reshape(32,-1,40,6)
            u=torch.zeros((*d.shape[:-1],3),device='cuda',requires_grad=True);opt=torch.optim.Adam([u],lr=.1)
            tokenlists=[[stoi.get(w,1) for w in targets[i].split()] for i in inds]
            infeasible+=sum(len(row)+sum(a==b for a,b in zip(row,row[1:]))>(len(xs[i])+3)//4 for i,row in zip(inds,tokenlists))
            toks=torch.tensor([w for row in tokenlists for w in row],device='cuda');tl=torch.tensor(list(map(len,tokenlists)),device='cuda');losses=[]
            for step in range(80):
                y=transform(d,u*mask).reshape(x.shape);lp,ol=model(y,length);loss=lossfn(lp,toks,ol,tl)
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert torch.isfinite(u.grad).all();opt.step();losses.append(float(loss.detach()))
            ys=transform(d,u*mask).detach().cpu().numpy().reshape(x.shape);coef=(u*mask).detach().cpu().numpy()
            for j,i in enumerate(inds):outputs.append(ys[j,:len(xs[i])].copy());gauges.append(coef[j,:len(xs[i])].copy())
            logs.append({'start':start,'losses':losses});print('CTC_TEST_ATTACK',condition,start,losses[0],losses[-1],flush=True)
        h=decode(model,outputs,vocab);ch=decode(model,[project(x) for x in outputs],vocab)
        record={'clips':128,'clip_ids':[e.sequence for e in selected],'references':refs,'targets':targets,'baseline_hypotheses':raw,
            'attacked_hypotheses':h,'canonical_before_hypotheses':canonical,'canonical_after_hypotheses':ch,
            'baseline_actual':metrics(refs,raw),'baseline_target':metrics(targets,raw),'attacked_actual':metrics(refs,h),'attacked_target':metrics(targets,h),
            'canonical_actual':metrics(refs,ch),'canonical_changed_sentences':sum(a!=b for a,b in zip(canonical,ch)),
            'target_change':interval(targets,h,raw),'actual_change':interval(refs,h,raw),
            'target_unknown_tokens':unknown,'ctc_infeasible_targets':infeasible,'different_targets':sum(a!=b for a,b in zip(refs,targets)),
            'optimization_logs':logs,**geometry(xs,outputs),'new_training':False,'independently_published_BT':False,'task':'gloss recognition, not translation'}
        np.savez_compressed(folder/(condition+'_poses.npz'),**{f'{prefix}{i:04d}':x for prefix,array in [('source',xs),('attacked',outputs),('gauge',gauges)] for i,x in enumerate(array)})
        (folder/(condition+'_scores.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        summary[condition]={k:record[k] for k in ['baseline_actual','baseline_target','attacked_actual','attacked_target','canonical_actual','canonical_changed_sentences','target_change','actual_change','target_unknown_tokens','ctc_infeasible_targets','different_targets','max_rotation_error_deg']}
        print('CTC_TEST_RESULT',condition,record['baseline_target']['wer'],record['attacked_target']['wer'],record['attacked_actual']['wer'],record['canonical_changed_sentences'],flush=True)
    (folder/'summary.json').write_text(json.dumps({'full_controls':{k:v['metrics']['wer'] for k,v in controls.items()},'conditions':summary,'checkpoint_best_epoch':ck['epoch'],'training_seed_count':1,'bootstrap_unit':'clip'},indent=2)+'\n')
    print('CTC_TEST_COMPLETE',flush=True)


if __name__=='__main__':main()
