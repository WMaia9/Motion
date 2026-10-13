"""Fixed TEST audit; no training or selection on results. New report only."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
import run_physical_semantic_ddpm as previous
from run_signspark_geometry_dev import matrices,project,_build_split,load_config
from run_signspark_gauge_attack import transform,SignCollator,Batch,PAD_TOKEN
from verify_signspark_studies import sha

ROOT=Path('reports/cvpr_evaluation_confirmation_20261011')
CORPORA=['PHOENIX-2014T','How2Sign']
CHECKPOINTS={
    'geometry':Path('reports/cvpr_physical_semantic_ddpm_20261010/geometry.pt'),
    'velocity':Path('reports/cvpr_motion_path_diffusion_20261010/physical_velocity.pt'),
    'physical_ce':Path('reports/cvpr_fixed_prefix_diffusion_20261010/physical_ce.pt'),
    'physical_generated_near':Path('reports/cvpr_fixed_prefix_diffusion_20261010/physical_generated_near.pt')}
REPRESENTATIONS=['raw','canonical','gauge_scale090','gauge_scale110','gauge_shear010']


def dataset(corpus):
    config=load_config(previous.BASE/'checkpoints'/corpus/'config.yaml')
    dc=dict(config['data']);dc.pop('pseudo_gloss_dir',None)
    dc.update(base_path=str((ROOT/'data/lmdb').resolve()),index_cache=False,max_sgn_length=300)
    d=_build_split(dc,'test',None)
    entries=[d[i] for i in range(len(d))]
    for e in entries:
        x=e.sgn.numpy().copy();x[:,60:150]=project(x)[:,60:150];e.sgn=torch.from_numpy(x)
    return entries,dc


def representation(xs,name):
    if name=='raw':return [x.copy() for x in xs]
    if name=='canonical':return [project(x) for x in xs]
    scale=.9 if name=='gauge_scale090' else (1.1 if name=='gauge_scale110' else 1.)
    shear=.1 if name=='gauge_shear010' else 0.
    output=[]
    for x in xs:
        d=x.reshape(-1,40,6).copy();mask=np.ones(40,bool);mask[10:25]=False;a=d[:,mask,:3].copy()
        d[:,mask,:3]*=scale;d[:,mask,3:]=scale*d[:,mask,3:]+shear*a
        output.append(d.reshape(x.shape))
    return output


def geometry(xs,ys):
    errors=[];canonical=[]
    for x,y in zip(xs,ys):
        relative=matrices(x).transpose(0,1,3,2)@matrices(y)
        errors.append(float(np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).max()))
        canonical.append(float(abs(project(x)-project(y)).max()))
    assert max(errors)<.01
    return {'max_rotation_error_deg':max(errors),'per_clip_rotation_error_deg':errors,
        'max_canonical_feature_difference':max(canonical)}


def score(model,xs,refs,dc):
    hypotheses=previous.back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32)
    return {'hypotheses':hypotheses,'scores':previous.score_translations(refs,hypotheses,dc.get('version',''),dc.get('level','word'))}


def prepare():
    paths=[*CHECKPOINTS.values(),previous.ROOT/'normalization.npz',previous.ROOT/'data_audit.json']
    inputs={'frozen_models':{str(p):sha(p) for p in paths},'TEST_used_for_model_training_or_selection':False}
    (ROOT/'sources/frozen_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    corpora={}
    for corpus in CORPORA:
        entries,dc=dataset(corpus);names=[e.sequence for e in entries]
        # Compare to exact prior dataset IDs, not text matches.
        oldfolder=Path('reports/cvpr_signspark_gauge_full_dev_20261010')/corpus
        devnames=[r['clip_id'] for r in json.loads((oldfolder/'selection.json').read_text())]
        assert not set(names)&set(devnames)
        if corpus=='PHOENIX-2014T':
            old=json.loads((previous.ROOT/'data_audit.json').read_text());assert not set(names)&set(old['train_ids'])
            assert len(entries)==642
        selected=np.sort(np.random.default_rng(2027).choice(len(entries),128,replace=False))
        folder=ROOT/'heldout'/corpus;folder.mkdir(exist_ok=True)
        (folder/'selection.json').write_text(json.dumps([{'test_index':int(i),'clip_id':entries[i].sequence,
            'frames':len(entries[i].sgn),'reference':' '.join(entries[i].txt)} for i in selected],ensure_ascii=False,indent=2)+'\n')
        corpora[corpus]={'test_clips':len(entries),'test_ids':names,'attack_indices':selected.tolist(),
            'overlap_with_prior_DEV':0,'subset_selected_before_scores':True,
            'captions_longer_than_64_tokens':sum(len(e.txt)>64 for e in entries)}
    (ROOT/'test_audit.json').write_text(json.dumps(corpora,indent=2)+'\n')
    print('TEST_PREPARED',json.dumps({c:{k:v for k,v in a.items() if k!='test_ids'} for c,a in corpora.items()}),flush=True)


def attacks():
    audit=json.loads((ROOT/'test_audit.json').read_text())
    for corpus in CORPORA:
        entries,dc=dataset(corpus);xs=[e.sgn.numpy() for e in entries];refs=[' '.join(e.txt) for e in entries]
        folder=ROOT/'heldout'/corpus;model=previous.make_back_translation_model(previous.BASE/'checkpoints'/corpus)
        model.do_recognition=False
        for p in model.parameters():p.requires_grad_(False)
        fullfile=folder/'full_real_scores.json'
        if not fullfile.exists():
            raw=score(model,xs,refs,dc);can=score(model,[project(x) for x in xs],refs,dc)
            result={'clips':len(xs),'clip_ids':[e.sequence for e in entries],'references':refs,
                'raw':raw,'canonical':can,'raw_canonical_changed_sentences':sum(a!=b for a,b in zip(raw['hypotheses'],can['hypotheses']))}
            fullfile.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
            print('FULL_TEST_REAL',corpus,raw['scores']['bleu']['bleu4'],can['scores']['bleu']['bleu4'],flush=True)
        selection=audit[corpus]['attack_indices'];selected=[entries[i] for i in selection]
        xs=[e.sgn.numpy() for e in selected];refs=[' '.join(e.txt) for e in selected]
        baseline=score(model,xs,refs,dc);canonical=score(model,[project(x) for x in xs],refs,dc)
        collator=SignCollator(model.txt_vocab,None,240);pad=model.txt_vocab.stoi[PAD_TOKEN]
        jointmask=torch.ones(1,1,40,1,device='cuda');jointmask[:,:,10:25]=0
        for name in ['correct_caption','rolled_wrong_caption']:
            if (folder/(name+'_scores.json')).exists():continue
            targets=refs if name=='correct_caption' else refs[1:]+refs[:1]
            outputs=[];coefficients=[];logs=[]
            for start in range(0,len(selected),32):
                chunk=[copy.copy(e) for e in selected[start:start+32]]
                for e,target in zip(chunk,targets[start:start+32]):e.txt=target.split()
                batch=Batch(collator(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True)
                original=batch.sgn.detach().reshape(len(chunk),-1,40,6)
                u=torch.zeros((*original.shape[:-1],3),device='cuda',requires_grad=True)
                optimizer=torch.optim.Adam([u],lr=.1);lossvalues=[]
                for step in range(80):
                    batch.sgn=transform(original,u*jointmask).reshape(len(chunk),-1,240)
                    prediction,_=model.forward(batch.sgn,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                    logits=prediction[0];loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad)
                    assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();assert torch.isfinite(u.grad).all();optimizer.step()
                    lossvalues.append(float(loss.detach()))
                y=transform(original,u*jointmask).detach().cpu().numpy().reshape(len(chunk),-1,240)
                gauge=(u*jointmask).detach().cpu().numpy()
                for j,e in enumerate(chunk):outputs.append(y[j,:len(e.sgn)].copy());coefficients.append(gauge[j,:len(e.sgn)].copy())
                logs.append({'start':start,'cross_entropy':lossvalues});print('TEST_ATTACK',corpus,name,start,lossvalues[0],lossvalues[-1],flush=True)
            raw_after=score(model,outputs,refs,dc);can_after=score(model,[project(x) for x in outputs],refs,dc)
            changed=sum(a!=b for a,b in zip(canonical['hypotheses'],can_after['hypotheses']))
            result={'clips':128,'clip_ids':[e.sequence for e in selected],'references':refs,'targets':targets,
                'baseline_actual':baseline,'baseline_target_scores':previous.score_translations(targets,baseline['hypotheses'],dc.get('version',''),dc.get('level','word')),
                'raw_after_actual':raw_after,'raw_after_target_scores':previous.score_translations(targets,raw_after['hypotheses'],dc.get('version',''),dc.get('level','word')),
                'canonical_before_actual':canonical,'canonical_after_actual':can_after,'canonical_changed_sentences':changed,
                'target_is_actual_reference':name=='correct_caption','different_target_captions':sum(a!=b for a,b in zip(refs,targets)),
                'losslogs':logs,**geometry(xs,outputs)}
            assert changed==0
            np.savez_compressed(folder/(name+'_poses.npz'),**{f'{prefix}{i:04d}':x for prefix,array in [('source',xs),('attacked',outputs),('gauge',coefficients)] for i,x in enumerate(array)})
            (folder/(name+'_scores.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
            print('TEST_ATTACK_RESULT',corpus,name,result['baseline_target_scores']['bleu']['bleu4'],result['raw_after_target_scores']['bleu']['bleu4'],raw_after['scores']['bleu']['bleu4'],flush=True)
        del model;torch.cuda.empty_cache()


@torch.no_grad()
def sample_without_reference_pose(model,entries,stoi,mean,std,alpha):
    """Equivalent to historical sampler, but never reads the reference poses."""
    model.eval();previous.seed(102);generator=torch.Generator(device='cuda').manual_seed(102)
    mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda');times=np.linspace(999,0,50).round().astype(int);outputs=[]
    for start in range(0,len(entries),16):
        group=entries[start:start+16];length=torch.tensor([e['frames'] for e in group],device='cuda')
        tokens=np.zeros((len(group),64),np.int64)
        for j,e in enumerate(group):
            caption=[stoi.get(w,1) for w in e['text']][:64] or [2];tokens[j,:len(caption)]=caption
        tok=torch.tensor(tokens,device='cuda');null=torch.zeros_like(tok);null[:,0]=2
        xt=torch.randn((len(group),int(length.max()),240),device='cuda',generator=generator)
        for pos,ti in enumerate(times):
            t=torch.full((len(group),),int(ti),device='cuda');c=model(xt,t,tok,length);u=model(xt,t,null,length);pred=u+3*(c-u)
            at=alpha[ti];nt=times[pos+1] if pos+1<len(times) else -1;an=alpha[nt] if nt>=0 else torch.tensor(1.,device='cuda')
            epsilon=(xt-at.sqrt()*pred)/(1-at).sqrt();xt=an.sqrt()*pred+(1-an).sqrt()*epsilon
        pose=(xt*st+mt).cpu().numpy();assert np.isfinite(pose).all()
        for j,n in enumerate(length.tolist()):outputs.append(pose[j,:n].copy())
    return outputs


def rankings():
    examples,dc=dataset('PHOENIX-2014T');refs=[' '.join(e.txt) for e in examples]
    entries=[{'clip_id':e.sequence,'frames':len(e.sgn),'text':list(e.txt)} for e in examples]
    del examples
    audit=json.loads((previous.ROOT/'data_audit.json').read_text());stoi={w:i for i,w in enumerate(audit['vocab'])}
    with np.load(previous.ROOT/'normalization.npz') as data:mean=data['mean'].copy();std=data['std'].copy()
    alpha=previous.alpha_schedule();teacher=previous.make_back_translation_model(previous.BASE/'checkpoints/PHOENIX-2014T')
    for name,checkpoint in CHECKPOINTS.items():
        folder=ROOT/'rankings'/name;folder.mkdir(exist_ok=True);path=folder/'poses.npz'
        if path.exists():
            with np.load(path) as data:xs=[data[f'clip{i:04d}'].copy() for i in range(len(entries))]
        else:
            model=previous.Denoiser(len(stoi)).cuda();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
            xs=sample_without_reference_pose(model,entries,stoi,mean,std,alpha)
            np.savez_compressed(path,**{f'clip{i:04d}':x for i,x in enumerate(xs)});del model
            print('TEST_GENERATED',name,len(xs),flush=True)
        for kind in REPRESENTATIONS:
            output=folder/(kind+'_scores.json')
            if output.exists():continue
            ys=representation(xs,kind);result=score(teacher,ys,refs,dc)
            result.update(clips=len(xs),clip_ids=[e['clip_id'] for e in entries],references=refs,
                generated_from_text_and_duration_only=True,representation=kind,**geometry(xs,ys))
            if kind.startswith('gauge_'):
                defended=score(teacher,[project(x) for x in ys],refs,dc)
                original=json.loads((folder/'canonical_scores.json').read_text())
                result['defended']=defended;result['defended_changed_sentences']=sum(a!=b for a,b in zip(original['hypotheses'],defended['hypotheses']))
                assert result['defended_changed_sentences']==0
            output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print('TEST_RANK_SCORE',name,kind,result['scores']['bleu']['bleu4'],flush=True)


def main():
    assert not (ROOT/'manifest_sha256.json').exists(),'frozen report'
    protocol=json.loads((ROOT/'sources/protocol_sha256.json').read_text());assert sha(ROOT/'PROTOCOLO.md')==protocol['PROTOCOLO.md']
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    if not (ROOT/'test_audit.json').exists():prepare()
    attacks();rankings();print('EVALUATION_HELDOUT_COMPLETE',flush=True)


if __name__=='__main__':main()
