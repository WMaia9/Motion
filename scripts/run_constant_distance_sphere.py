"""Exploratory full-sphere control: fixed canonical distance, varying total norm.

Declared after the stricter circle study; all circle and sphere outcomes remain
reported. This tests which removed freedom limits the stricter norm control.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run_constant_noncanonicity_layer as circle
import run_learned_equivalence as old
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_constant_distance_sphere_20261012');SEEDS=[42,123,777];ARMS=['text_raw','motion_raw','text_canonical']
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def auxiliary_rotation(w):
 theta=torch.linalg.vector_norm(w,dim=-1);a,b,c=w.unbind(-1);zero=torch.zeros_like(a);k=torch.stack((zero,-c,b,c,zero,-a,-b,a,zero),-1).reshape(*w.shape[:-1],3,3)
 return torch.eye(3,dtype=w.dtype,device=w.device)+torch.sinc(theta/torch.pi)[...,None,None]*k+.5*torch.sinc(theta/(2*torch.pi)).square()[...,None,None]*(k@k)
def transform(x,w):
 shape=x.shape;d=x.reshape(*shape[:-1],40,6);u=F.normalize(d[...,:3],dim=-1,eps=1e-8);b=d[...,3:];tau=(u*b).sum(-1);perp=b-tau[...,None]*u;v=F.normalize(perp,dim=-1,eps=1e-8)
 rho=torch.linalg.vector_norm(d[...,:3],dim=-1);sigma=torch.linalg.vector_norm(perp,dim=-1);s=torch.stack((rho-1,sigma-1,tau),-1);radius=torch.linalg.vector_norm(s,dim=-1);w=w*(radius<.95)[...,None];delta=(auxiliary_rotation(w)@s[...,None])[...,0]-s
 return (d+torch.cat((delta[...,0:1]*u,delta[...,1:2]*v+delta[...,2:3]*u),-1)).reshape(shape),s,delta
class Layer(old.OutputLayer):
 def forward(self,pose,tokens,lengths):
  _,c=super().forward(pose,tokens,lengths);w=c/.35;y,_,_=transform(pose,w);return y,w

def setup():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 audit=read(old.base.ROOT/'data_audit.json');stoi={w:i for i,w in enumerate(audit['vocab'])};g=old.base.Denoiser(len(stoi)).cuda();g.load_state_dict(torch.load(old.GENERATOR,map_location='cpu',weights_only=False)['state_dict'],strict=True);g.eval()
 for p in g.parameters():p.requires_grad_(False)
 inputs=[old.GENERATOR,circle.ROOT/'generated/train/poses.npz',circle.ROOT/'generated/dev/poses.npz',old.base.BASE/'checkpoints/PHOENIX-2014T/best.ckpt']
 protocol={'purpose':'relax stricter circle control to full sphere, while exactly fixing per-joint distance to canonical','exploratory_after_circle_study':True,'seeds':SEEDS,'arms':ARMS,'epochs':20,'batch_size':64,'optimizer':'AdamW lr.001 wd.0001 clip1','angle_coeff_bound':1.,'safety':'identity for radius>=.95','total_raw_6D_norm_not_fixed':True,'per_frame_joint_canonical_distance_fixed':True,'frozen_source_sampling_reused':True,'no_TEST_use':True,'selection':'last epoch; all outcomes retained','input_SHA256':{str(p):sha(p) for p in inputs},'source_code_sha256':sha(__file__)}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==protocol
 else:write('protocol.json',protocol)
 return stoi,g

@torch.no_grad()
def evaluate(model,teacher,records,xs,stoi,folder,arm):
 from run_evaluation_heldout import geometry
 from scipy.stats import rankdata
 ys=[];ws=[];model.eval()
 for start in range(0,len(records),32):
  group=records[start:start+32];n=torch.tensor([len(x) for x in xs[start:start+32]],device='cuda');x=torch.zeros(len(group),int(n.max()),240,device='cuda')
  for j,p in enumerate(xs[start:start+32]):x[j,:len(p)]=torch.from_numpy(p).cuda()
  tok=old.token_tensor(group,stoi)
  if arm=='motion_raw':tok=torch.zeros_like(tok);tok[:,0]=2
  y,w=model(x,tok,n)
  for j,l in enumerate(n.tolist()):ys.append(y[j,:l].cpu().numpy().copy());ws.append(w[j,:l].cpu().numpy().copy())
 refs=[' '.join(e['text']) for e in records];ids=[e['clip_id'] for e in records];raw=old.score(teacher,ys,refs,ids);can=old.score(teacher,[old.base.project(y) for y in ys],refs,ids);baseline=read(ROOT/'evaluation/baseline_canonical.json');assert can['hypotheses']==baseline['hypotheses']
 emax=nmax=0.;ra=[];rb=[]
 for x,y in zip(xs,ys):
  ca=old.base.project(x);cb=old.base.project(y);ea=((x-ca).reshape(-1,40,6).astype(float)**2).sum(-1);eb=((y-cb).reshape(-1,40,6).astype(float)**2).sum(-1);emax=max(emax,float(abs(ea-eb).max()));nmax=max(nmax,float(abs((x.reshape(-1,40,6).astype(float)**2).sum(-1)-(y.reshape(-1,40,6).astype(float)**2).sum(-1)).max()));ra.append(float(np.sqrt(np.mean((x-ca)**2))));rb.append(float(np.sqrt(np.mean((y-cb)**2))))
 assert emax<2e-5;ranks=rankdata(np.r_[ra,rb]);n=len(ra);auc=float((ranks[n:].sum()-n*(n+1)/2)/(n*n));r={'raw':raw,'canonical':can,'arm':arm,'per_frame_joint_canonical_distance_squared_max_difference':emax,'per_frame_joint_6D_squared_norm_max_difference':nmax,'mean_source_RMS':float(np.mean(ra)),'mean_layer_RMS':float(np.mean(rb)),'RMS_threshold_detector_AUC':auc,'RMS_best_orientation_AUC':max(auc,1-auc),**geometry(xs,ys)}
 write(str(folder.relative_to(ROOT))+'.json',r);np.savez_compressed(str(folder)+'_poses.npz',**{f'{k}{i:04d}':x for k,arr in [('attacked',ys),('auxiliary',ws)] for i,x in enumerate(arr)});print('SPHERE_SCORED',folder.name,raw['scores']['bleu']['bleu4'],can['scores']['bleu']['bleu4'],auc,flush=True)

def train():
 stoi,g=setup();records,xs=circle.cache('train');dev,dx=circle.cache('dev');teacher=old.base.make_back_translation_model(old.base.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
 for p in teacher.parameters():p.requires_grad_(False)
 collator=old.SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[old.PAD_TOKEN]
 for kind,poses in [('raw',dx),('canonical',[old.base.project(x) for x in dx])]:write('evaluation/baseline_'+kind+'.json',old.score(teacher,poses,[' '.join(e['text']) for e in dev],[e['clip_id'] for e in dev]))
 for seed in SEEDS:
  for arm in ARMS:
   old.base.seed(seed);model=Layer(g).cuda();model.train();folder=ROOT/f'training/seed{seed}/{arm}';folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/'layer.pt'
   if not checkpoint.exists():
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001);history=[];orders=[]
    for epoch in range(20):
     started=time.time();order=np.random.permutation(len(records));ordered=[]
     for start in range(0,len(order),256):ordered.extend(sorted(order[start:start+256],key=lambda i:len(xs[i])))
     groups=[ordered[p:p+64] for p in range(0,len(ordered),64)];random.shuffle(groups);losses=[]
     for ids in groups:
      if epoch==0:orders.append(list(map(int,ids)))
      batch,tok=old.pack(records,xs,ids,stoi,collator,pad)
      if arm=='motion_raw':tok=torch.zeros_like(tok);tok[:,0]=2
      y,w=model(batch.sgn,tok,batch.sgn_lengths.to(batch.sgn.device));consumed=old.base.physical(y) if arm=='text_canonical' else y;out,_=teacher.forward(consumed,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask);logits=out[0];lt=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad,reduction='none').reshape(batch.txt.shape);loss=(lt.sum(1)/(batch.txt!=pad).sum(1)).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();losses.append(float(loss.detach()))
     r={'epoch':epoch+1,'TRAIN_CE':float(np.mean(losses)),'seconds':time.time()-started};history.append(r);write(str(folder.relative_to(ROOT)/'history.json'),history);print('SPHERE_EPOCH',seed,arm,json.dumps(r),flush=True)
    assert orders==read(old.ROOT/f'training/seed{seed}/text_raw_batches.json');torch.save({'state_dict':model.state_dict(),'seed':seed,'arm':arm,'epoch':20,'selection':'last epoch'},checkpoint);write(str(folder.relative_to(ROOT)/'first_epoch_batches.json'),orders)
   model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True);p=ROOT/f'evaluation/seed{seed}_{arm}'
   if not Path(str(p)+'.json').exists():evaluate(model,teacher,dev,dx,stoi,p,arm)
   del model;torch.cuda.empty_cache()
 write('complete.json',{'fits':9,'seeds':SEEDS,'arms':ARMS,'same_raw_sources_as_circle':True,'all_first_epoch_batches_match':True,'no_TEST_use':True})
if __name__=='__main__':train()
