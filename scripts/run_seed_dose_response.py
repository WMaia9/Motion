"""Post-review seed replication and exploratory dose response in a NEW report.

Existing evidence is read only. Fixed endpoints; never stop when significant.
Copied training loop retains the original objective and stochastic schedule.
"""
import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F
import run_physical_semantic_ddpm as old
from run_quotient_training import tag, geometric_loss, validation
from run_reviewer_training import batches, score
from run_signspark_gauge_attack import SignCollator, Batch, PAD_TOKEN
from run_evaluation_heldout import sample_without_reference_pose, geometry
from verify_signspark_studies import sha
ROOT = Path('reports/cvpr_seed_dose_response_20261012')
NEW_SEEDS = [11, 97, 909, 1337, 2027, 2718, 31415]
OLD_SEEDS = [42, 123, 777]
JOBS = [(s,k,1.,20) for s in NEW_SEEDS for k in ['raw','canonical']]
JOBS += [(s,'raw',w,40) for s in OLD_SEEDS for w in [1.,3.,10.]]

def read(p): return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n; p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def folder(s,k,w): return ROOT/f'training/seed{s}'/tag('geodesic',k,w)
def setup():
 assert not (ROOT/'manifest_sha256.json').exists()
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 a=read(old.ROOT/'data_audit.json')
 with np.load(old.ROOT/'normalization.npz') as z: mean,std=z['mean'].copy(),z['std'].copy()
 return a,mean,std

def register():
 ROOT.mkdir(exist_ok=True);setup()
 inputs=[old.ROOT/'shared_best.pt',old.ROOT/'normalization.npz',old.ROOT/'data_audit.json',old.BASE/'checkpoints/PHOENIX-2014T/best.ckpt',Path('reports/cvpr_quotient_training_20261012/manifest_sha256.json'),Path('reports/cvpr_second_review_confirmation_20261012/manifest_sha256.json')]
 sources=[Path(x) for x in read(ROOT/'protocol.json')['scientific_sources_sha256']] if (ROOT/'protocol.json').exists() else list(Path('scripts').glob('*.py'))
 p={'purpose':'post-review replication of same-model raw-minus-canonical gap, not recovery relative to collapsed no-CE training',
 'primary_new_seeds':NEW_SEEDS,'historical_seeds':OLD_SEEDS,'primary_endpoint':'raw CE1/geodesic/20epochs on seven new seeds; pooled ten seeds is a separately labelled extension',
 'matched_control':'canonical CE1/geodesic/20epochs on the same seven new seeds',
 'exploratory_dose_grid':{'seeds':OLD_SEEDS,'CE_weights':[1.,3.,10.],'epochs':[20,40],'fits_from_shared_init':True,'all_conditions_reported':True},
 'jobs':[{'seed':s,'reader_input':k,'CE_weight':w,'epochs':e} for s,k,w,e in JOBS],
 'batch_size':64,'semantic_batch_limit':16,'semantic_timesteps':'t<=800 and no caption dropout','caption_drop':.1,'optimizer':'AdamW lr1e-4 wd1e-4 clip1',
 'sampling':'DDIM50 CFG3 seed102, full DEV519 and TEST642; source caption and reference duration only',
 'selection':'fixed epochs; no outcome-dependent early stopping, seed removal or best TEST dose',
 'inference_order':'all weights saved and hashed before any new TEST inference',
 'statistics':'20000 paired clip and joint seed+clip bootstrap draws; primary seven-new-seed gap, pooled ten-seed gap separately; all six exploratory endpoints and five contrasts multiplicity-adjusted; leave-one-seed-out sensitivity',
 'limits':'one shared pretrained generator, one PHOENIX public reader; more seeds do not validate independent pretraining or guarantee venue acceptance',
 'prior_TEST_use':'same public TEST already inspected; this is prospective model replication after observed initial outcomes, not a pristine external holdout',
 'inputs_sha256':{str(p):sha(p) for p in inputs},'scientific_sources_sha256':{str(p):sha(p) for p in sources}}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==p,'protocol changed'
 else:write('protocol.json',p)
 print('SEED_DOSE_REGISTERED',len(JOBS),flush=True)
def fit(model,train,dev,stoi,mean,std,alpha,teacher,motion,kind,weight,seed,epochs):
 old.seed(seed);opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);collator=SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[PAD_TOKEN]
 mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda');name=tag(motion,kind,weight);folder=ROOT/f'training/seed{seed}'/name;folder.mkdir(parents=True,exist_ok=True)
 history=[];stochastic=[];gradient=None
 for epoch in range(epochs):
  started=time.time();model.train();values=[];cevalues=[];mses=[];count=0
  for ids in batches(train):
   x,tok,n=old.pack(train,ids,stoi,mean,std);drop=torch.rand(len(ids),device='cuda')<.1;tok=tok.clone();tok[drop]=0;tok[drop,0]=2;t=torch.randint(0,1000,(len(ids),),device='cuda');noise=torch.randn_like(x);a=alpha[t][:,None,None]
   pred=model(a.sqrt()*x+(1-a).sqrt()*noise,t,tok,n);mask=torch.arange(x.shape[1],device='cuda')[None]<n[:,None]
   poses=pred*st+mt;target=x*st+mt;mse=((pred-x).square()*mask[...,None]).sum()/(mask.sum()*240)
   physical=geometric_loss(poses,target,mask) if motion=='geodesic' else mse
   ce=torch.zeros((),device='cuda');chosen=((t<=800)&~drop).nonzero().flatten()[:16]
   if epoch==0:stochastic.append({'ids':list(map(int,ids)),'timesteps':t.cpu().tolist(),'drop':drop.cpu().tolist(),'noise_sum':float(noise.double().sum()),'noise_squared_sum':float(noise.double().square().sum())})
   if kind!='none' and len(chosen):
    ex=[copy.copy(train[ids[j]]) for j in chosen.tolist()];batch=Batch(collator(ex),txt_pad_index=pad,sgn_dim=240,use_cuda=True);mx=batch.sgn.shape[1]
    z=poses[chosen,:mx]*mask[chosen,:mx,None];consumed=old.physical(z) if kind=='canonical' else z
    out,_=teacher.forward(consumed,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask);logits=out[0];losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad,reduction='none').reshape(batch.txt.shape);ce=(losses.sum(1)/(batch.txt!=pad).sum(1)).mean();count+=len(chosen)
   if gradient is None and len(chosen) and kind!='none':
    gm=(torch.autograd.grad(physical,pred,retain_graph=True)[0]/st if motion=='mse' else torch.autograd.grad(physical,poses,retain_graph=True)[0]);gc=torch.autograd.grad(ce,poses,retain_graph=True)[0]
    def dots(g):
     q=g.reshape(-1,40,6);d=poses.reshape(-1,40,6);return torch.stack(((q[...,:3]*d[...,:3]).sum(-1),(q[...,3:]*d[...,3:]).sum(-1),(q[...,3:]*d[...,:3]).sum(-1)),-1)
    gradient={'physical_gradient_norm':float(gm.norm()),'CE_gradient_norm':float(gc.norm()),'physical_max_vertical_dot':float(dots(gm).abs().max()),'CE_max_vertical_dot':float(dots(gc).abs().max()),'weighted_CE_to_motion_gradient_norm':float(weight*gc.norm()/gm.norm().clamp_min(1e-12))}
   loss=physical+weight*ce;assert torch.isfinite(loss)
   opt.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step();values.append(float(physical.detach()));cevalues.append(float(ce.detach()));mses.append(float(mse.detach()))
  val=validation(model,dev,stoi,mean,std,alpha);r={'epoch':epoch+1,'motion_loss':float(np.mean(values)),'TRAIN_CE':float(np.mean(cevalues)),'TRAIN_coordinate_mse':float(np.mean(mses)),'semantic_examples':count,**val,'seconds':time.time()-started};history.append(r);write(str(folder.relative_to(ROOT)/'history.json'),history)
  print('QUOTIENT_EPOCH',seed,name,json.dumps(r),flush=True)
  if epoch+1 in [20,40]:torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),'seed':seed,'epoch':epoch+1,'motion_loss':motion,'reader_input':kind,'weight':weight,'selection':'fixed endpoint'},folder/f'epoch{epoch+1}.pt')
 write(str(folder.relative_to(ROOT)/'first_epoch_stochastic.json'),stochastic);write(str(folder.relative_to(ROOT)/'gradient_audit.json'),gradient)

def train():
 register();audit,mean,std=setup();data,dev=old.dataset('train'),old.dataset('dev')
 assert [e.sequence for e in data]==audit['train_ids'] and [e.sequence for e in dev]==audit['dev_ids']
 stoi={w:i for i,w in enumerate(audit['vocab'])};shared=torch.load(old.ROOT/'shared_best.pt',map_location='cpu',weights_only=False)['state_dict'];alpha=old.alpha_schedule()
 teacher=old.make_back_translation_model(old.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
 for p in teacher.parameters():p.requires_grad_(False)
 for s,k,w,e in JOBS:
  f=folder(s,k,w)
  if (f/f'epoch{e}.pt').exists():continue
  assert not (f/'history.json').exists(),'partial fit: do not overwrite or silently restart'
  model=old.Denoiser(len(stoi)).cuda();model.load_state_dict(shared,strict=True)
  fit(model,data,dev,stoi,mean,std,alpha,teacher,'geodesic',k,w,s,e)
  del model;torch.cuda.empty_cache()
 for s in NEW_SEEDS:
  assert read(folder(s,'raw',1.)/'first_epoch_stochastic.json')==read(folder(s,'canonical',1.)/'first_epoch_stochastic.json')
 for s in OLD_SEEDS:
  base=read(folder(s,'raw',1.)/'first_epoch_stochastic.json')
  assert all(read(folder(s,'raw',w)/'first_epoch_stochastic.json')==base for w in [3.,10.])
 checkpoints=[folder(s,k,w)/f'epoch{epoch}.pt' for s,k,w,e in JOBS for epoch in ([20,40] if e==40 else [20])]
 write('weights_frozen_before_inference.json',{'checkpoints_sha256':{str(p):sha(p) for p in checkpoints},'fits':len(JOBS),'matched_stochastic_inputs':True})

@torch.no_grad()
def evaluate(split):
 register();record=read(ROOT/'weights_frozen_before_inference.json')
 for p,h in record['checkpoints_sha256'].items():assert sha(p)==h
 audit,mean,std=setup();stoi={w:i for i,w in enumerate(audit['vocab'])}
 if split=='dev':
  data=old.dataset('dev');ids=[e.sequence for e in data];refs=[' '.join(e.txt) for e in data];lengths=[len(e.sgn) for e in data]
 else:
  source=read('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/raw_scores.json');ids=source['clip_ids'];refs=source['references']
  with np.load('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz') as z:lengths=[len(z[f'clip{i:04d}']) for i in range(len(ids))]
 assert not set(ids)&set(audit['train_ids'])
 if split=='test':assert not set(ids)&set(audit['dev_ids'])
 entries=[{'clip_id':i,'frames':n,'text':r.split()} for i,n,r in zip(ids,lengths,refs)]
 teacher=old.make_back_translation_model(old.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False;alpha=old.alpha_schedule()
 conditions=0
 for s,k,w,e in JOBS:
  for epoch in ([20,40] if e==40 else [20]):
   out=folder(s,k,w)/f'{split}_epoch{epoch}';out.mkdir(parents=True,exist_ok=True);conditions+=1
   if (out/'result.json').exists():continue
   model=old.Denoiser(len(stoi)).cuda();model.load_state_dict(torch.load(folder(s,k,w)/f'epoch{epoch}.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True)
   xs=sample_without_reference_pose(model,entries,stoi,mean,std,alpha);cs=[old.project(x) for x in xs]
   raw=score(teacher,xs,refs,ids);can=score(teacher,cs,refs,ids)
   rms=[float(np.sqrt(np.mean((x-y)**2))) for x,y in zip(xs,cs)]
   np.savez_compressed(out/'poses.npz',**{f'clip{i:04d}':x for i,x in enumerate(xs)})
   write(str(out.relative_to(ROOT)/'result.json'),{'seed':s,'reader_input':k,'CE_weight':w,'epoch':epoch,'split':split,'raw':raw,'canonical':can,'RMS_to_canonical':rms,'mean_RMS':float(np.mean(rms)),**geometry(xs,cs)})
   print('SEED_DOSE_SCORED',split,s,k,w,epoch,raw['scores']['bleu']['bleu4'],can['scores']['bleu']['bleu4'],flush=True)
   del model;torch.cuda.empty_cache()
 write(f'{split}_complete.json',{'conditions':conditions,'clips':len(ids),'checkpoint_sha256_verified':True})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','train','dev','test']);a=parser.parse_args()
 if a.phase in ['dev','test']:evaluate(a.phase)
 else:globals()[a.phase]()
