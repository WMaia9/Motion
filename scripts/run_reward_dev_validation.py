"""Conditional three-seed validation of a finite DEV-qualified reward objective.

All final weights are sealed before new TEST sampling. If the finite grid fails
its criteria, no additional grid or TEST training is attempted.
"""
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t
from motion_release_audit_common import A,read,write,sha,seed,evaluator,score_embeddings,canonical,metric
from run_motion_release_audit import mdm_model,generate_mdm
from run_humanml_reward_pilot import train_text,reward_embeddings,canonical as project
from run_additional_motion_audit import cohort_saved
GRID=Path('reports/cvpr_reward_dev_search_20261013');R=Path('reports/cvpr_reward_dev_validation_20261013');TRAIN=Path('reports/cvpr_humanml3d_blind_fid_20261012');OLD=Path('reports/cvpr_motion_prevalence_20261012');SEEDS=[997,1907,31415]

def main():
 R.mkdir(exist_ok=True)
 while not (GRID/'selection.json').exists():time.sleep(10)
 selection=read(GRID/'selection.json');chosen=selection['chosen'];files=[Path(__file__),Path('scripts/run_humanml_reward_pilot.py'),Path('scripts/run_motion_release_audit.py'),Path('scripts/motion_release_audit_common.py'),GRID/'selection.json',GRID/'protocol.json',TRAIN/'train_real.npz',TRAIN/'train_selection.json',A/'humanml_enc_512_50steps/model000750000.pt',A/'t2m/text_mot_match/model/finest.tar']
 p={'scope':'DEV-qualified finite objective validation, no new grid','chosen_DEV_configuration':chosen,'fitting_seeds':SEEDS,'arms':['mse','raw_reward','canonical_reward'],'steps':500,'batch':16,'optimizer':'AdamW wd1e-4,clipping1, chosenlr','sampling_seeds':[20261111,20261112],'TEST_cohorts':'first1024 native MDM cohorts rep00/01 with their full4640independentreferences','all9weights_hashed_before_new_TEST_sampling':True,'primary':'same-output raw-minus-C gap, not recovery relative to a collapsed noCE model','secondary':'reward improvement over matched MSE sampling','negative_or_failed_optimization_not_absence_of_channel_evidence':True,'source_sha256':{str(f):sha(f) for f in files}}
 if (R/'protocol.json').exists():assert read(R/'protocol.json')==p
 else:write(R/'protocol.json',p)
 if chosen is None:
  write(R/'analysis.json',{'DEV_grid_qualified':False,'no_TEST_training_or_inference':True,'interpretation':'finite one-step reward calibration failed to meet DEV retrieval/FID/optimization criteria; no inference about successful reward exploitation'});write(R/'complete.json',{'workflow_complete':True,'valid_reward_pilot_obtained':False});print('DEV_VALIDATION_SKIPPED_NO_QUALIFYING_CONFIGURATION',flush=True);return
 wrapper,em,es=evaluator();items=read(TRAIN/'train_selection.json');text=train_text(wrapper,items)
 with np.load(TRAIN/'train_real.npz') as z:raw=z['raw'].copy();lengths=z['lengths'].copy()
 mu=torch.tensor(np.load('external/HumanML3D/HumanML3D/Mean.npy'),device='cuda',dtype=torch.float32);sd=torch.tensor(np.load('external/HumanML3D/HumanML3D/Std.npy'),device='cuda',dtype=torch.float32);mean=torch.tensor(em,device='cuda',dtype=torch.float32);std=torch.tensor(es,device='cuda',dtype=torch.float32)
 for s in SEEDS:
  for arm in p['arms']:
   folder=R/f'{arm}_seed{s}';folder.mkdir(exist_ok=True)
   if (folder/'training_complete.json').exists():assert sha(folder/'final.pt')==read(folder/'training_complete.json')['weights_sha256'];continue
   seed(s);model,_,diffusion=mdm_model();model.train();model.clip_model.eval();params=[v for v in model.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=chosen['lr'],weight_decay=1e-4);seed(70000+s);rng=np.random.default_rng(70000+s);history=[];losses=[];wrapper.motion_encoder.gru.train();assert wrapper.motion_encoder.gru.dropout==0
   for step in range(500):
    idx=rng.choice(len(raw),16,replace=False);L=torch.tensor(lengths[idx],device='cuda');mask=torch.arange(196,device='cuda')[None,:]<L[:,None];X=torch.tensor(raw[idx],device='cuda');normalized=(X-mu)/sd;normalized*=mask[...,None];x0=normalized.permute(0,2,1)[:,:,None];tt=torch.randint(diffusion.num_timesteps,(len(idx),),device='cuda');xt=diffusion.q_sample(x0,tt,noise=torch.randn_like(x0));y={'text':[items[i]['caption'] for i in idx],'lengths':L,'mask':mask[:,None,None]};pred=model(xt,diffusion._scale_timesteps(tt),y=y)[:,:,0].permute(0,2,1);mse=(((pred-normalized)**2)*mask[...,None]).sum()/(mask.sum()*263);physical=pred*sd+mu
    if arm=='canonical_reward':physical=project(physical)
    inputs=(physical-mean)/std;inputs*=mask[...,None];E=reward_embeddings(wrapper,inputs,L);T=text[idx].cuda();ce=F.cross_entropy(F.normalize(E,dim=-1)@F.normalize(T,dim=-1).T/.1,torch.arange(len(idx),device='cuda'));loss=mse+(0 if arm=='mse' else chosen['weight'])*ce;opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad) and torch.isfinite(loss);opt.step();losses.append(float(ce.detach()))
    if (step+1)%50==0:history.append({'step':step+1,'MSE':float(mse.detach()),'reward_CE_mean50':float(np.mean(losses[-50:]))});write(folder/'training_log.json',history);print('REWARD_VALIDATION_TRAIN',s,arm,history[-1],flush=True)
   state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('clip_model.')};torch.save(state,folder/'final.pt');write(folder/'training_complete.json',{'weights_sha256':sha(folder/'final.pt'),'CE_first100':float(np.mean(losses[:100])),'CE_last100':float(np.mean(losses[-100:]))});del model,diffusion,opt,params,state;torch.cuda.empty_cache()
 write(R/'weights_frozen_before_TEST.json',{'all9weights':{str(f):sha(f) for f in R.glob('*_seed*/final.pt')},'chosen_only_on_DEV':True,'no_new_TEST_sampling_before_seal':True})
 wrapper.motion_encoder.gru.eval();allrows={}
 for s in SEEDS:
  for arm in p['arms']:
   folder=R/f'{arm}_seed{s}';seed(s);model,_,diffusion=mdm_model();missing,extra=model.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True),strict=False);assert not extra and all(k.startswith('clip_model.') for k in missing);model.eval();from model.cfg_sampler import ClassifierFreeSampleModel
   for rep,draw in enumerate(p['sampling_seeds']):
    out=folder/f'test_draw{rep}';out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():continue
    cohort=cohort_saved(OLD/f'mdm50/rep_{rep:02d}');ref=np.load(OLD/f'mdm50/rep_{rep:02d}/reference_embeddings.npy');seed(draw);X=generate_mdm((model,ClassifierFreeSampleModel(model),diffusion),cohort,rep);np.save(out/'generated_raw.npy',X);scores={}
    for condition,x in [('raw',X),('canonical',canonical(X))]:T,E=score_embeddings(wrapper,x,cohort,em,es);scores[condition]=metric(T,E,ref);np.save(out/(condition+'_embeddings.npy'),E)
    write(out/'metrics.json',scores);print('REWARD_VALIDATION_TEST',s,arm,rep,scores,flush=True)
   del model,diffusion;torch.cuda.empty_cache()
 summaries={}
 for arm in p['arms']:
  summaries[arm]={}
  for field in ['FID','R1']:
   values={}
   for condition in ['raw','canonical']:
    values[condition]=np.array([np.mean([read(R/f'{arm}_seed{s}/test_draw{rep}/metrics.json')[condition]['R_precision'][0] if field=='R1' else read(R/f'{arm}_seed{s}/test_draw{rep}/metrics.json')[condition][field] for rep in range(2)]) for s in SEEDS])
   gap=values['raw']-values['canonical'];se=gap.std(ddof=1)/np.sqrt(3);summaries[arm][field]={'raw':float(values['raw'].mean()),'canonical':float(values['canonical'].mean()),'raw_minus_C':float(gap.mean()),'same_model_seed_gaps':gap.tolist(),'family6_seed_t2_CI95':(gap.mean()+np.array([-1,1])*t.ppf(1-.05/12,2)*se).tolist()}
 write(R/'analysis.json',{'DEV_grid_qualified':True,'chosen':chosen,'TEST_conditions':summaries,'three_fitting_seeds_two_sampling_draws':True,'no_TEST_selection':True,'negative_gap_not_absence_of_channel_evidence':True});write(R/'complete.json',{'workflow_complete':True,'valid_DEV_reward_pilot_obtained':True});print('REWARD_VALIDATION_COMPLETE',summaries,flush=True)
if __name__=='__main__':main()
