"""Finite learning-rate/reward grid selected exclusively on DEV free sampling.

No TEST inference in this driver. A failed DEV grid is reported as failure to
improve retrieval, never as evidence that reward optimization avoids the channel.
"""
import argparse,sys,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from motion_release_audit_common import DATA,A,read,write,sha,seed,evaluator,materialize,score_embeddings,canonical,metric
from run_motion_release_audit import mdm_model,generate_mdm,save_cohort
from run_humanml_reward_pilot import train_text,reward_embeddings,canonical as torch_canonical
R=Path('reports/cvpr_reward_dev_search_20261013');TRAIN=Path('reports/cvpr_humanml3d_blind_fid_20261012');GRID=[(lr,w) for lr in [1e-6,3e-6] for w in [0.,.001,.01,.1]]
def name(lr,w):return f'lr{lr:g}_w{w:g}'
def register():
 R.mkdir(exist_ok=True);files=[Path(__file__),Path('scripts/run_humanml_reward_pilot.py'),Path('scripts/run_motion_release_audit.py'),Path('scripts/motion_release_audit_common.py'),TRAIN/'train_real.npz',TRAIN/'train_selection.json',A/'humanml_enc_512_50steps/model000750000.pt',A/'t2m/text_mot_match/model/finest.tar',DATA/'val.txt']
 p={'scope':'finite DEV-only reward calibration; no TEST model or score selection','grid':[{'learning_rate':lr,'reward_weight':w} for lr,w in GRID],'training_seed':42,'steps':500,'batch':16,'optimizer':'AdamW wd1e-4, clipping1','MSE_control_each_lr':True,'reward':'one denoised x0 cosine InfoNCE temperature0.1; frozen259-feature standard evaluator incl126rotationchannels','native_dropout_and_condition_drop_retained':True,'DEV_selection':'fixed512native VAL outputs, fullnative VAL reference, common generation seed20261101','acceptance_criterion':'DEV R1 improvement at least0.005 over matched-lr MSE, and DEV FID <=1.10*matched-lr MSE; reward CE last100mean < first100mean','if_valid':'choose maximal qualifying DEV R1, tie smallerFID then smallerrewardweight; freeze before any new TEST fits; matched MSE/raw/canonical reward at3new declared seeds','if_invalid':'no TEST claim from a failed optimization; one-step pilot did not improve retrieval','public_VAL_already_inspected':True,'inputs_and_sources_sha256':{str(f):sha(f) for f in files}}
 if (R/'protocol.json').exists():assert read(R/'protocol.json')==p
 else:write(R/'protocol.json',p)
 return p

def prepare():
 register();sys.path.insert(0,'external/MDM')
 # Native pretokenized dataset imports spaCy but never calls its tokenizer.
 if 'spacy' not in sys.modules:
  m=types.ModuleType('spacy');m.load=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('unused tokenizer'));sys.modules['spacy']=m
 from data_loaders.humanml.data.dataset import Text2MotionDatasetV2
 from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
 opt=types.SimpleNamespace(dataset_name='t2m',fixed_len=0,use_cache=True,cache_dir=str(R/'native_cache'),max_motion_length=196,max_text_len=20,unit_length=4,disable_offset_aug=False,motion_dir=str(DATA/'new_joint_vecs'),text_dir=str(DATA/'texts'));(R/'native_cache/dataset').mkdir(parents=True,exist_ok=True)
 dataset=Text2MotionDatasetV2(opt,np.zeros(263),np.ones(263),str(DATA/'val.txt'),WordVectorizer(str(A/'glove'),'our_vab'));cohort=materialize(dataset,20261100,512);ref=materialize(dataset,20261102);(R/'dev_inputs').mkdir(exist_ok=True);(R/'dev_reference').mkdir(exist_ok=True);save_cohort(R/'dev_inputs',cohort);save_cohort(R/'dev_reference',ref);write(R/'prepared.json',{'DEV_clips':len(cohort['raw']),'reference_clips':len(ref['raw']),'DEV_only':True})

def load_cohort(folder):
 z=np.load(folder/'conditioning.npz');return {**{k:z[k].tolist() if k=='captions' else z[k].copy() for k in z.files},'raw':np.load(folder/'source_raw.npy')}

def run():
 p=register();assert (R/'prepared.json').exists();wrapper,evalmean,evalstd=evaluator();items=read(TRAIN/'train_selection.json');texts=train_text(wrapper,items);cohort=load_cohort(R/'dev_inputs');reference=load_cohort(R/'dev_reference');_,refE=score_embeddings(wrapper,reference['raw'],reference,evalmean,evalstd);np.save(R/'DEV_reference_embeddings.npy',refE);del reference
 with np.load(TRAIN/'train_real.npz') as z:raw=z['raw'].copy();lengths=z['lengths'].copy()
 mu=torch.tensor(np.load('external/HumanML3D/HumanML3D/Mean.npy'),device='cuda',dtype=torch.float32);sd=torch.tensor(np.load('external/HumanML3D/HumanML3D/Std.npy'),device='cuda',dtype=torch.float32);em=torch.tensor(evalmean,device='cuda',dtype=torch.float32);es=torch.tensor(evalstd,device='cuda',dtype=torch.float32)
 for lr,w in GRID:
  folder=R/name(lr,w);folder.mkdir(exist_ok=True)
  if (folder/'dev_metrics.json').exists():continue
  seed(42);model,_,diffusion=mdm_model()
  if (folder/'final.pt').exists():
   missing,extra=model.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True),strict=False);assert not extra and all(k.startswith('clip_model.') for k in missing)
  else:
   model.train();model.clip_model.eval();params=[v for v in model.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4);seed(60042);rng=np.random.default_rng(60042);losses=[];history=[];wrapper.motion_encoder.gru.train();assert wrapper.motion_encoder.gru.dropout==0
   for step in range(p['steps']):
    idx=rng.choice(len(raw),p['batch'],replace=False);L=torch.tensor(lengths[idx],device='cuda');mask=torch.arange(196,device='cuda')[None,:]<L[:,None];X=torch.tensor(raw[idx],device='cuda');norm=(X-mu)/sd;norm*=mask[...,None];x0=norm.permute(0,2,1)[:,:,None];t=torch.randint(diffusion.num_timesteps,(len(idx),),device='cuda');noise=torch.randn_like(x0);xt=diffusion.q_sample(x0,t,noise=noise);y={'text':[items[i]['caption'] for i in idx],'lengths':L,'mask':mask[:,None,None]};pred=model(xt,diffusion._scale_timesteps(t),y=y)[:,:,0].permute(0,2,1);mse=(((pred-norm)**2)*mask[...,None]).sum()/(mask.sum()*263)
    physical=pred*sd+mu;normalized=(physical-em)/es;normalized*=mask[...,None];E=reward_embeddings(wrapper,normalized,L);T=texts[idx].cuda();ce=F.cross_entropy(F.normalize(E,dim=-1)@F.normalize(T,dim=-1).T/.1,torch.arange(len(idx),device='cuda'));loss=mse+w*ce;opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad) and torch.isfinite(loss);opt.step();losses.append(float(ce.detach()))
    if (step+1)%50==0:history.append({'step':step+1,'MSE':float(mse.detach()),'reward_CE_mean50':float(np.mean(losses[-50:]))});write(folder/'training_log.json',history);print('DEV_GRID_TRAIN',lr,w,history[-1],flush=True)
   state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('clip_model.')};torch.save(state,folder/'final.pt');write(folder/'training_complete.json',{'weights_sha256':sha(folder/'final.pt'),'CE_first100':float(np.mean(losses[:100])),'CE_last100':float(np.mean(losses[-100:]))});del opt,params,state
  wrapper.motion_encoder.gru.eval();model.eval();from model.cfg_sampler import ClassifierFreeSampleModel
  seed(20261101);generated=generate_mdm((model,ClassifierFreeSampleModel(model),diffusion),cohort,0);np.save(folder/'generated_DEV.npy',generated);scores={}
  for c,x in [('raw',generated),('canonical',canonical(generated))]:T,E=score_embeddings(wrapper,x,cohort,evalmean,evalstd);scores[c]=metric(T,E,refE);np.save(folder/f'{c}_embeddings.npy',E)
  write(folder/'dev_metrics.json',{'lr':lr,'weight':w,'scores':scores});print('DEV_GRID_EVALUATED',lr,w,scores,flush=True);del model,diffusion;torch.cuda.empty_cache()
 qualifying=[];rows=[]
 for lr,w in GRID:
  x=read(R/name(lr,w)/'dev_metrics.json');rows.append(x)
  if w==0:continue
  baseline=read(R/name(lr,0.)/'dev_metrics.json')['scores']['raw'];a=x['scores']['raw'];h=read(R/name(lr,w)/'training_complete.json')
  if a['R_precision'][0]-baseline['R_precision'][0]>=.005 and a['FID']<=1.1*baseline['FID'] and h['CE_last100']<h['CE_first100']:qualifying.append(x)
 qualifying.sort(key=lambda x:(-x['scores']['raw']['R_precision'][0],x['scores']['raw']['FID'],x['weight']))
 write(R/'selection.json',{'all_DEV_rows':rows,'qualified':len(qualifying),'chosen':qualifying[0] if qualifying else None,'TEST_used_for_selection':False,'no_grid_extension_if_failed':True});print('DEV_GRID_COMPLETE',len(qualifying),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args();seed(42);(prepare if a.action=='prepare' else run)()
