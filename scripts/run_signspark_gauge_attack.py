"""Targeted oracle-caption audit of physically redundant 6D coordinates."""
import argparse,copy,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
from run_signspark_geometry_dev import ROOT,matrices,project,load_config,_build_split,make_back_translation_model,back_translate,score_translations
from signspark_bt.data.batch import SignCollator,Batch
from signspark_bt.data.vocabulary import PAD_TOKEN

OUT=Path('reports/cvpr_signspark_gauge_attack_20261010')

def transform(original,u):
    alpha=torch.exp(.35*torch.tanh(u[...,0:1]));beta=torch.exp(.35*torch.tanh(u[...,1:2]));gamma=.35*torch.tanh(u[...,2:3])
    a,b=original[...,:3],original[...,3:]
    return torch.cat((alpha*a,beta*b+gamma*a),-1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args();started=time.time()
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    os.environ['DATA_ROOT']=str((ROOT/'data').resolve());cfg=load_config(ROOT/'checkpoints'/args.dataset/'config.yaml');dc=dict(cfg['data']);dc.pop('pseudo_gloss_dir',None)
    data=_build_split(dc,'dev',None);idx=np.sort(np.random.default_rng(20261010).choice(len(data),64,replace=False))
    examples=[data[int(i)] for i in idx];xs=[e.sgn.numpy().copy() for e in examples];refs=[' '.join(e.txt) for e in examples]
    folder=OUT/args.dataset;folder.mkdir(exist_ok=True)
    (folder/'selection.json').write_text(json.dumps([{'dev_index':int(i),'clip_id':e.sequence,'reference':r,'frames':len(x)} for i,e,r,x in zip(idx,examples,refs,xs)],indent=2,ensure_ascii=False))
    model=make_back_translation_model(ROOT/'checkpoints'/args.dataset);model.do_recognition=False
    for parameter in model.parameters():parameter.requires_grad_(False)
    baseline=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=8)
    canonical_before=back_translate(model,[torch.from_numpy(project(x)) for x in xs],batch_size=8)
    summary={'clips':64,'raw_original_scores':score_translations(refs,baseline,dc.get('version',''),dc.get('level','word')),'canonical_original_scores':score_translations(refs,canonical_before,dc.get('version',''),dc.get('level','word')),'conditions':{}}
    collator=SignCollator(model.txt_vocab,None,240);pad=model.txt_vocab.stoi[PAD_TOKEN]
    for condition in ['correct_caption','rolled_wrong_caption']:
        targetrefs=refs if condition=='correct_caption' else refs[1:]+refs[:1]
        attacked=[];coefficients=[];losslogs=[]
        for start in range(0,64,8):
            chunk=[copy.copy(e) for e in examples[start:start+8]]
            for e,target in zip(chunk,targetrefs[start:start+8]):e.txt=target.split()
            batch=Batch(collator(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True)
            original=batch.sgn.detach().reshape(8,-1,40,6);u=torch.zeros((*original.shape[:-1],3),device='cuda',requires_grad=True)
            optimizer=torch.optim.Adam([u],lr=.1);lossvalues=[]
            for step in range(80):
                y=transform(original,u).reshape(batch.sgn.shape);batch.sgn=y
                output,_=model.forward(batch.sgn,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                logits=output[0];loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad)
                assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();assert torch.isfinite(u.grad).all();optimizer.step();lossvalues.append(float(loss.detach()))
            y=transform(original,u).detach().cpu().numpy().reshape(8,-1,240);saved=u.detach().cpu().numpy()
            for i,e in enumerate(chunk):attacked.append(y[i,:len(e.sgn)].copy());coefficients.append(saved[i,:len(e.sgn)].copy())
            losslogs.append({'batch_start':start,'cross_entropy':lossvalues})
            print(args.dataset,condition,start,'CE',round(lossvalues[0],4),'->',round(lossvalues[-1],4),flush=True)
        maxangle=0.;feature_delta=0.
        for x,y in zip(xs,attacked):
            assert np.isfinite(y).all()
            rel=matrices(x).transpose(0,1,3,2)@matrices(y)
            maxangle=max(maxangle,float(np.degrees(Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude()).max()))
            feature_delta=max(feature_delta,float(abs(x-y).max()))
        assert maxangle<.01,(args.dataset,condition,maxangle)
        raw_after=back_translate(model,[torch.from_numpy(x) for x in attacked],batch_size=8)
        canonical_after=back_translate(model,[torch.from_numpy(project(x)) for x in attacked],batch_size=8)
        result={'actual_reference_scores':score_translations(refs,raw_after,dc.get('version',''),dc.get('level','word')),'target_scores':score_translations(targetrefs,raw_after,dc.get('version',''),dc.get('level','word')),'baseline_target_scores':score_translations(targetrefs,baseline,dc.get('version',''),dc.get('level','word')),'canonical_actual_scores':score_translations(refs,canonical_after,dc.get('version',''),dc.get('level','word')),'canonical_target_scores':score_translations(targetrefs,canonical_after,dc.get('version',''),dc.get('level','word')),'raw_changed_sentences':sum(a!=b for a,b in zip(raw_after,baseline)),'canonical_changed_sentences':sum(a!=b for a,b in zip(canonical_after,canonical_before)),'max_rotation_error_deg':maxangle,'max_feature_abs_delta':feature_delta,'different_target_references':sum(a!=b for a,b in zip(refs,targetrefs)),'raw_original_hypotheses':baseline,'canonical_original_hypotheses':canonical_before,'raw_after_hypotheses':raw_after,'canonical_after_hypotheses':canonical_after,'targets':targetrefs,'losslogs':losslogs}
        (folder/f'{condition}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        np.savez_compressed(folder/f'{condition}_poses.npz',**{f'{prefix}{i:02d}':x for prefix,arrays in [('source',xs),('attacked',attacked),('gauge',coefficients)] for i,x in enumerate(arrays)})
        summary['conditions'][condition]={k:v for k,v in result.items() if 'hypotheses' not in k and k not in ('targets','losslogs')};(folder/'summary.json').write_text(json.dumps(summary,indent=2));print('FINISHED',args.dataset,condition,json.dumps(summary['conditions'][condition]),flush=True)
    summary['seconds']=time.time()-started;(folder/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',args.dataset,summary['seconds'],flush=True)

if __name__=='__main__':main()
