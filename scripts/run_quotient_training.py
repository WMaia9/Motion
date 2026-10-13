"""Predeclared conventional DDPM training: rotation-only vs coordinate MSE losses.

No learned gauge module; full denoiser is fitted from the same initialization.
Primary endpoints fixed before new fitting; no TEST use in development.
"""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run_physical_semantic_ddpm as old
from run_reviewer_training import batches,score
from run_signspark_gauge_attack import SignCollator,Batch,PAD_TOKEN
from run_evaluation_heldout import sample_without_reference_pose
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_quotient_training_20261012')
SEEDS=[42,123,777];WEIGHTS=[.02,.1,.3,1.]
ARMS=[(motion,kind,weight) for motion in ['geodesic','mse'] for kind,weight in [('none',0.)]+[('raw',w) for w in WEIGHTS]+[('canonical',.1),('canonical',1.)]]
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def tag(m,k,w):return f'{m}_{k}_w{w:g}'
def rotation(x):
 d=x.reshape(*x.shape[:-1],40,6);u=F.normalize(d[...,:3],dim=-1,eps=1e-8);b=d[...,3:];v=F.normalize(b-(u*b).sum(-1,keepdim=True)*u,dim=-1,eps=1e-8)
 return torch.stack((u,v,torch.linalg.cross(u,v)),dim=-2)
def squared_angles(x,y):
 rx=rotation(x);ry=rotation(y);rel=rx@ry.transpose(-1,-2)
 skew=torch.stack((rel[...,2,1]-rel[...,1,2],rel[...,0,2]-rel[...,2,0],rel[...,1,0]-rel[...,0,1]),-1)*.5
 sine=torch.linalg.vector_norm(skew,dim=-1);cosine=(rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
 return torch.atan2(sine,cosine).square()
def geometric_loss(pred,target,mask):return (squared_angles(pred,target)*mask[...,None]).sum()/(mask.sum()*40)
def setup():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 a=read(old.ROOT/'data_audit.json')
 with np.load(old.ROOT/'normalization.npz') as z:mean,std=z['mean'].copy(),z['std'].copy()
 return a,mean,std

def register():
 setup();inputs=[old.ROOT/'shared_best.pt',old.ROOT/'normalization.npz',old.ROOT/'data_audit.json',old.BASE/'checkpoints/PHOENIX-2014T/best.ckpt']
 proto={'purpose':'test ordinary generator training when motion loss is invariant to redundant rotation coordinates',
 'seeds':SEEDS,'arms':[{'motion_loss':m,'reader_input':k,'CE_weight':w,'tag':tag(m,k,w)} for m,k,w in ARMS],
 'epochs':{'all_arms':5,'geodesic_none_w0_and_geodesic_raw_w1_and_geodesic_canonical_w1':20},
 'primary_endpoint':'20-epoch geodesic raw CE weight1 versus geodesic no CE and matched canonical CE weight1',
 'secondary':'5-epoch full motion-loss/CE-weight sweep; all conditions reported; no best-DEV selection',
 'batch_size':64,'semantic_batch_limit':16,'semantic_timesteps':'t<=800 and no caption dropout',
 'optimizer':'AdamW lr1e-4 wd1e-4 clip1','caption_drop':.1,'motion_geodesic':'mean squared SO(3) angle in radians, atan2(skew norm,trace); no raw-feature MSE term',
 'full_denoiser_trainable':True,'additional_gauge_module':False,'init':'same frozen shared canonical-MSE pretrained DDPM',
 'sampling':'full DEV519, DDIM50 CFG3 seed102, source text and reference duration; no reference poses',
 'checkpoint_selection':'predeclared epoch5 and epoch20; last epochs, no score selection',
 'uncertainty':'paired clip bootstrap and three fine-tuning seeds; not independent pretraining',
 'interpretation':'raw-minus-canonical gain contrasts; physical motions may differ across fitted models; only raw/canonical encodings of one output share rotations',
 'no_new_TEST_use':True,'inputs':{str(p):sha(p) for p in inputs},'code_sha256':sha(__file__)}
 path=ROOT/'protocol.json'
 if path.exists():assert read(path)==proto,'changed protocol'
 else:write('protocol.json',proto)
 print('QUOTIENT_TRAINING_REGISTERED',len(ARMS)*len(SEEDS),flush=True)

def numerics():
 torch.manual_seed(42);x=torch.randn(3,5,240,dtype=torch.double,requires_grad=True);target=torch.randn_like(x);mask=torch.ones(3,5,dtype=torch.bool)
 c=.2*torch.randn(3,5,40,3,dtype=torch.double);d=x.reshape(3,5,40,6)
 y=torch.cat((torch.exp(c[...,0:1])*d[...,:3],torch.exp(c[...,1:2])*d[...,3:]+c[...,2:3]*d[...,:3]),-1).reshape_as(x)
 a=geometric_loss(x,target,mask);b=geometric_loss(y,target,mask);assert abs(float(a-b))<1e-12
 g=torch.autograd.grad(a,x)[0].reshape(3,5,40,6);v=torch.stack(((g[...,:3]*d[...,:3]).sum(-1),(g[...,3:]*d[...,3:]).sum(-1),(g[...,3:]*d[...,:3]).sum(-1)),-1)
 assert v.abs().max()<1e-12
 identity=geometric_loss(x,x.detach(),mask);gg=torch.autograd.grad(identity,x)[0];assert torch.isfinite(gg).all() and float(identity)<1e-24
 write('geometry_unit_checks.json',{'passed':True,'equivalent_loss_difference':float(abs(a-b)),'maximum_vertical_gradient_dot':float(v.abs().max()),'identity_loss':float(identity),'identity_gradient_finite':True})
 print('GEODESIC_CHECKS_PASSED',float(v.abs().max()),flush=True)

@torch.no_grad()
def validation(model,dev,stoi,mean,std,alpha):
 model.eval();rng=torch.Generator(device='cuda').manual_seed(2027);mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda');total=den=mse=0.
 for start in range(0,len(dev),64):
  ids=list(range(start,min(start+64,len(dev))));x,tok,n=old.pack(dev,ids,stoi,mean,std);t=torch.randint(0,1000,(len(ids),),device='cuda',generator=rng);noise=torch.randn(x.shape,device='cuda',generator=rng);a=alpha[t][:,None,None];p=model(a.sqrt()*x+(1-a).sqrt()*noise,t,tok,n);mask=torch.arange(x.shape[1],device='cuda')[None]<n[:,None]
  total+=float((squared_angles(p*st+mt,x*st+mt)*mask[...,None]).sum());den+=int(mask.sum())*40;mse+=float(((p-x).square()*mask[...,None]).sum())
 return {'mean_squared_angle_radians':total/den,'standardized_coordinate_mse':mse/(den*6)}

def fit(model,train,dev,stoi,mean,std,alpha,teacher,motion,kind,weight,seed):
 old.seed(seed);opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);collator=SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[PAD_TOKEN]
 mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda');name=tag(motion,kind,weight);folder=ROOT/f'training/seed{seed}'/name;folder.mkdir(parents=True,exist_ok=True)
 epochs=20 if motion=='geodesic' and (kind=='none' or weight==1) else 5
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
  if epoch+1 in [5,20]:torch.save({'state_dict':model.state_dict(),'seed':seed,'epoch':epoch+1,'motion_loss':motion,'reader_input':kind,'weight':weight,'selection':'fixed endpoint'},folder/f'epoch{epoch+1}.pt')
 write(str(folder.relative_to(ROOT)/'first_epoch_stochastic.json'),stochastic);write(str(folder.relative_to(ROOT)/'gradient_audit.json'),gradient)


def evaluation(model,teacher,dev,entries,stoi,mean,std,alpha,folder):
 xs=sample_without_reference_pose(model,entries,stoi,mean,std,alpha);ys=[old.project(x) for x in xs];refs=[' '.join(e.txt) for e in dev];ids=[e.sequence for e in dev]
 for kind,poses in [('raw',xs),('canonical',ys)]:
  r=score(teacher,poses,refs,ids);write(str(folder.relative_to(ROOT)/(kind+'_scores.json')),r)
 rms=np.asarray([np.sqrt(np.mean((x-y)**2)) for x,y in zip(xs,ys)]);maxima=np.asarray([np.max(abs(x-y)) for x,y in zip(xs,ys)])
 write(str(folder.relative_to(ROOT)/'noncanonicity.json'),{'clip_RMS_mean':float(rms.mean()),'clip_RMS_median':float(np.median(rms)),'clip_RMS_quantiles':np.percentile(rms,[5,25,75,95]).tolist(),'clip_RMS_values':rms.tolist(),'max_feature_difference':float(maxima.max()),'generated_motion_differs_across_checkpoints':True})
 np.savez_compressed(folder/'poses.npz',**{f'clip{i:04d}':x for i,x in enumerate(xs)})
 raw=read(folder/'raw_scores.json')['scores']['bleu']['bleu4'];can=read(folder/'canonical_scores.json')['scores']['bleu']['bleu4']
 print('QUOTIENT_SCORED',str(folder),raw,can,float(rms.mean()),flush=True)

def train():
 register();numerics();audit,mean,std=setup();data,dev=old.dataset('train'),old.dataset('dev');assert len(data)==7096 and len(dev)==519
 assert [e.sequence for e in data]==audit['train_ids'] and [e.sequence for e in dev]==audit['dev_ids'];stoi={w:i for i,w in enumerate(audit['vocab'])};shared=torch.load(old.ROOT/'shared_best.pt',map_location='cpu',weights_only=False)['state_dict'];alpha=old.alpha_schedule()
 teacher=old.make_back_translation_model(old.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
 for p in teacher.parameters():p.requires_grad_(False)
 entries=[{'clip_id':e.sequence,'frames':len(e.sgn),'text':list(e.txt)} for e in dev]
 for seed in SEEDS:
  matching=[]
  for motion,kind,weight in ARMS:
   name=tag(motion,kind,weight);folder=ROOT/f'training/seed{seed}'/name;last=20 if motion=='geodesic' and (kind=='none' or weight==1) else 5
   model=old.Denoiser(len(stoi)).cuda();model.load_state_dict(shared,strict=True)
   if not (folder/f'epoch{last}.pt').exists():fit(model,data,dev,stoi,mean,std,alpha,teacher,motion,kind,weight,seed)
   matching.append(read(folder/'first_epoch_stochastic.json'))
   for epoch in [5,20] if last==20 else [5]:
    out=folder/f'dev_epoch{epoch}';out.mkdir(exist_ok=True)
    if not (out/'canonical_scores.json').exists():
     model.load_state_dict(torch.load(folder/f'epoch{epoch}.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True);evaluation(model,teacher,dev,entries,stoi,mean,std,alpha,out)
   del model;torch.cuda.empty_cache()
  assert all(v==matching[0] for v in matching),'stochastic inputs differ across arms'
 write('complete.json',{'fits':len(SEEDS)*len(ARMS),'seeds':SEEDS,'DEV_clips':519,'TRAIN_clips':7096,'all_first_epoch_stochastic_inputs_matched':True,'no_TEST_use':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','numerics']);a=p.parse_args();globals()[a.phase]()
