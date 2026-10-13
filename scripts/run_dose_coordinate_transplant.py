"""Prospective exploratory dose mechanism diagnostic on full DEV.

Keeps the source output's rotations, replaces only its redundant coefficients.
No extra module is trained, no per-clip gradient optimization is performed.
"""
import json
from pathlib import Path
import numpy as np
import torch
import run_physical_semantic_ddpm as base
from run_quotient_training import ROOT as TRAINING
from run_reviewer_training import score
from run_evaluation_heldout import sample_without_reference_pose,geometry
from verify_signspark_studies import sha,matrices
ROOT=Path('reports/cvpr_dose_transplant_20261012');SEEDS=[42,123,777]
NEW=Path('reports/cvpr_seed_dose_response_20261012')
CONDITIONS=[(NEW,f'geodesic_raw_w{w:g}',e) for w in [1.,3.,10.] for e in [20,40]]+[(TRAINING,a,20) for a in ['geodesic_none_w0','geodesic_canonical_w1']]
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def register():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
 files=[NEW/'protocol.json',base.ROOT/'shared_best.pt',base.ROOT/'normalization.npz',base.ROOT/'data_audit.json',base.BASE/'checkpoints/PHOENIX-2014T/best.ckpt',Path(__file__)]
 p={'purpose':'caption-coordinate mechanism diagnostic for every dose endpoint, distinct from same-output score inflation',
 'registered_before_new_dose_BLEU_results':True,'seeds':SEEDS,'conditions':[{'root':str(r),'arm':a,'epoch':e} for r,a,e in CONDITIONS],
 'source':'full DEV519, saved source outputs of each fixed model','alternative':'same sampler seed and source duration, next caption in fixed DEV ordering',
 'operation':'keep source canonical u,v, replace rho/sigma/tau with alternate output scalars; original left stream protected',
 'no_added_trained_module':True,'no_per_clip_gradient_optimization':True,'no_TEST_use':True,'all_grid_endpoints_reported':True,
 'statistics':'20000 paired clip and seed+clip bootstrap, family eight aggregate conditions',
 'inputs_sha256':{str(p):sha(p) for p in files}}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==p
 else:write('protocol.json',p)


def main():
 register();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;audit=read(base.ROOT/'data_audit.json');stoi={w:i for i,w in enumerate(audit['vocab'])}
 with np.load(base.ROOT/'normalization.npz') as z:mean,std=z['mean'],z['std']
 alpha=base.alpha_schedule();teacher=base.make_back_translation_model(base.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
 for seed in SEEDS:
  for training,arm,epoch in CONDITIONS:
   folder=training/f'training/seed{seed}/{arm}/dev_epoch{epoch}'
   if training==NEW:
    saved=read(folder/'result.json');original,canonical=saved['raw'],saved['canonical']
   else:original=read(folder/'raw_scores.json');canonical=read(folder/'canonical_scores.json')
   refs=original['references'];ids=original['clip_ids']
   with np.load(folder/'poses.npz') as z:source=[z[f'clip{i:04d}'].copy() for i in range(len(ids))]
   targets=refs[1:]+refs[:1];entries=[{'clip_id':name,'frames':len(p),'text':caption.split()} for name,p,caption in zip(ids,source,targets)]
   model=base.Denoiser(len(stoi)).cuda();model.load_state_dict(torch.load(training/f'training/seed{seed}/{arm}/epoch{epoch}.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True);alternate=sample_without_reference_pose(model,entries,stoi,mean,std,alpha);hybrid=[]
   for x,y in zip(source,alternate):
    C=matrices(x)[...,:2,:];R=matrices(y)[...,:2,:];d=y.reshape(-1,40,6).astype(float);rho=np.linalg.norm(d[...,:3],axis=-1)[...,None];tau=(R[...,0,:]*d[...,3:]).sum(-1)[...,None];sigma=np.linalg.norm(d[...,3:]-tau*R[...,0,:],axis=-1)[...,None]
    changed=np.concatenate((rho*C[...,0,:],sigma*C[...,1,:]+tau*C[...,0,:]),-1).reshape(x.shape).astype(np.float32);changed[:,60:150]=x[:,60:150];hybrid.append(changed)
   raw=score(teacher,hybrid,refs,ids);can=score(teacher,[base.project(x) for x in hybrid],refs,ids)
   record={'seed':seed,'arm':arm,'epoch':epoch,'clip_ids':ids,'references':refs,'targets':targets,'different_caption_targets':sum(x!=y for x,y in zip(refs,targets)),
    'baseline_raw':original,'raw':raw,'canonical':can,'canonical_hypothesis_changes':sum(a!=b for a,b in zip(can['hypotheses'],canonical['hypotheses'])),
    'baseline_wrong_target_scores':base.score_translations(targets,original['hypotheses'],'phoenix_2014_trans','word'),
    'after_wrong_target_scores':base.score_translations(targets,raw['hypotheses'],'phoenix_2014_trans','word'),**geometry(source,hybrid)}
   write(f'evaluation/seed{seed}_{arm}_epoch{epoch}.json',record);np.savez_compressed(ROOT/'evaluation'/f'seed{seed}_{arm}_epoch{epoch}_poses.npz',**{f'{k}{i:04d}':x for k,arr in [('source',source),('alternate',alternate),('hybrid',hybrid)] for i,x in enumerate(arr)})
   print('ORDINARY_TRANSPLANT',seed,arm,record['baseline_wrong_target_scores']['bleu']['bleu4'],record['after_wrong_target_scores']['bleu']['bleu4'],original['scores']['bleu']['bleu4'],raw['scores']['bleu']['bleu4'],record['canonical_hypothesis_changes'],flush=True)
   del model;torch.cuda.empty_cache()
 write('complete.json',{'conditions':24,'DEV_clips':519,'no_new_fitting':True,'no_TEST_use':True})
if __name__=='__main__':
 import sys
 if len(sys.argv)>1 and sys.argv[1]=='register':register()
 else:
  record=read(NEW/'weights_frozen_before_inference.json')
  for p,h in record['checkpoints_sha256'].items():assert sha(p)==h
  write('weights_before_diagnostic.json',record)
  main()
