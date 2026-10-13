"""Rotation-preserving layers with fixed per-joint distance to canonical encoding.

An auxiliary SO(3) acts on three redundant coordinates, not physical joints.
This controls a simple RMS detector without claiming general undetectability.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import rankdata
import run_learned_equivalence as old
from run_evaluation_heldout import sample_without_reference_pose,geometry
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_constant_noncanonicity_20261012');SEEDS=[42,123,777];ARMS=['text_raw','motion_raw','text_canonical']
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def auxiliary_rotation(w):
 phase=torch.pi*w.mean(-1);zero=torch.zeros_like(phase);w=torch.stack((phase/(2**.5),phase/(2**.5),zero),-1)
 theta=torch.linalg.vector_norm(w,dim=-1);a,b,c=w.unbind(-1);zero=torch.zeros_like(a)
 k=torch.stack((zero,-c,b,c,zero,-a,-b,a,zero),-1).reshape(*w.shape[:-1],3,3)
 return torch.eye(3,dtype=w.dtype,device=w.device)+torch.sinc(theta/torch.pi)[...,None,None]*k+.5*torch.sinc(theta/(2*torch.pi)).square()[...,None,None]*(k@k)
def transform(x,w):
 shape=x.shape;d=x.reshape(*shape[:-1],40,6);u=F.normalize(d[...,:3],dim=-1,eps=1e-8);b=d[...,3:];tau=(u*b).sum(-1);perp=b-tau[...,None]*u;v=F.normalize(perp,dim=-1,eps=1e-8)
 rho=torch.linalg.vector_norm(d[...,:3],dim=-1);sigma=torch.linalg.vector_norm(perp,dim=-1);s=torch.stack((rho-1,sigma-1,tau),-1);radius=torch.linalg.vector_norm(s,dim=-1)
 w=w*(radius<.95)[...,None];delta=(auxiliary_rotation(w)@s[...,None])[...,0]-s
 change=torch.cat((delta[...,0:1]*u,delta[...,1:2]*v+delta[...,2:3]*u),-1)
 return (d+change).reshape(shape),s,delta
class Layer(old.OutputLayer):
 def forward(self,pose,tokens,lengths):
  _,c=super().forward(pose,tokens,lengths);w=c/.35
  y,_,_=transform(pose,w);return y,w

def setup():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 a=read(old.base.ROOT/'data_audit.json');stoi={w:i for i,w in enumerate(a['vocab'])};g=old.base.Denoiser(len(stoi)).cuda();g.load_state_dict(torch.load(old.GENERATOR,map_location='cpu',weights_only=False)['state_dict'],strict=True);g.eval()
 for p in g.parameters():p.requires_grad_(False)
 return a,stoi,g

def register():
 inputs=[old.GENERATOR,old.ROOT/'manifest_sha256.json',old.base.BASE/'checkpoints/PHOENIX-2014T/best.ckpt']
 p={'purpose':'test score gains while rotations and per-frame/joint RMS-to-canonical distance are held fixed',
 'seeds':SEEDS,'arms':ARMS,'epochs':20,'batch_size':64,'optimizer':'AdamW lr.001 wd.0001 clip1','source':'raw outputs of same frozen generator, not canonicalized before layer',
 'auxiliary_angle_coeff_bound':1.,'auxiliary_phase_bound_radians':float(np.pi),'auxiliary_axis':[float(1/np.sqrt(2)),float(1/np.sqrt(2)),0.],'safety':'identity where redundant-radius>=.95; guarantees rho,sigma positive elsewhere','left_joint_mask':'10:25 unchanged',
 'mathematics':'s=(rho-1,sigma-1,tau); rotation about axis(1,1,0) preserves squared distance to canonical AND total 6D norm for each frame/joint; not physical kinetic energy',
 'sampling':'TRAIN7096/DEV519, DDIM50 CFG3 seed102 source text and reference durations; no reference poses',
 'selection':'last epoch; no TEST use or score selection','claim_scope':'not general steganographic security or invisible to all detectors; only declared scalar budget is fixed',
 'inputs':{str(p):sha(p) for p in inputs},'code_sha256':sha(__file__)}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==p
 else:write('protocol.json',p)

@torch.enable_grad()
def numerics():
 torch.manual_seed(42);u=torch.randn(3,5,240,dtype=torch.double);c=old.base.physical(u);d=c.reshape(3,5,40,6);q=.1*torch.randn(3,5,40,3,dtype=torch.double)
 x=torch.cat(((1+q[...,0:1])*d[...,:3],(1+q[...,1:2])*d[...,3:]+q[...,2:3]*d[...,:3]),-1).reshape_as(c);w=torch.randn_like(q,requires_grad=True);y,s,delta=transform(x,w)
 before=(x-old.base.physical(x)).square().reshape(3,5,40,6).sum(-1);after=(y-old.base.physical(y)).square().reshape(3,5,40,6).sum(-1);err=float(abs(before-after).max());rot=float(abs(old.base.physical(x)-old.base.physical(y)).max());assert err<1e-12 and rot<1e-12
 normerr=float(abs(x.square().reshape(3,5,40,6).sum(-1)-y.square().reshape(3,5,40,6).sum(-1)).max());assert normerr<1e-12
 zero=torch.zeros_like(w,requires_grad=True);z,_,_=transform(x,zero);g=torch.autograd.grad(z.sum(),zero)[0];assert torch.equal(x,z) and torch.isfinite(g).all() and g.norm()>0
 write('unit_checks.json',{'passed':True,'max_energy_difference_double':err,'max_6D_squared_norm_difference_double':normerr,'max_canonical_difference_double':rot,'identity_exact_and_finite_nonzero_gradient':True})
 print('CONSTANT_BUDGET_NUMERICS',err,rot,flush=True)

@torch.no_grad()
def prepare():
 _,stoi,g=setup();register();numerics()
 with np.load(old.base.ROOT/'normalization.npz') as z:mean,std=z['mean'],z['std']
 alpha=old.base.alpha_schedule()
 for split in ['train','dev']:
  records=read(old.ROOT/'data_audit.json')[split];folder=ROOT/'generated'/split;folder.mkdir(parents=True,exist_ok=True)
  if not (folder/'poses.npz').exists():
   xs=sample_without_reference_pose(g,records,stoi,mean,std,alpha);np.savez_compressed(folder/'poses.npz',**{f'clip{i:05d}':x for i,x in enumerate(xs)})
   oldrecords,canonical=old.cache(split);assert records==oldrecords;maximum=max(float(abs(old.base.project(x)-y).max()) for x,y in zip(xs,canonical));assert maximum<2e-6
   radius=[]
   for x in xs:
    _,s,_=transform(torch.from_numpy(x)[None],torch.zeros(1,len(x),40,3));radius.append(torch.linalg.vector_norm(s,dim=-1).numpy().reshape(-1))
   rr=np.concatenate(radius);write(f'generated/{split}/audit.json',{'clips':len(xs),'canonical_replay_max_difference':maximum,'fraction_frame_joints_radius_above_safety':float(np.mean(rr>=.95))})
  print('CONSTANT_BUDGET_PREPARED',split,len(records),flush=True)
 write('prepared.json',{'TRAIN_clips':7096,'DEV_clips':519,'no_reference_pose_tensors':True})
def cache(split):
 records=read(old.ROOT/'data_audit.json')[split]
 with np.load(ROOT/f'generated/{split}/poses.npz') as z:xs=[z[f'clip{i:05d}'].copy() for i in range(len(records))]
 return records,xs

@torch.no_grad()
def evaluate(model,teacher,records,xs,stoi,folder,arm):
 model.eval();ys=[];ws=[]
 for start in range(0,len(records),32):
  group=records[start:start+32];n=torch.tensor([len(x) for x in xs[start:start+32]],device='cuda');x=torch.zeros(len(group),int(n.max()),240,device='cuda')
  for j,p in enumerate(xs[start:start+32]):x[j,:len(p)]=torch.from_numpy(p).cuda()
  tokens=old.token_tensor(group,stoi)
  if arm=='motion_raw':tokens=torch.zeros_like(tokens);tokens[:,0]=2
  y,w=model(x,tokens,n)
  for j,l in enumerate(n.tolist()):ys.append(y[j,:l].cpu().numpy().copy());ws.append(w[j,:l].cpu().numpy().copy())
 refs=[' '.join(e['text']) for e in records];ids=[e['clip_id'] for e in records];raw=old.score(teacher,ys,refs,ids);canonical=old.score(teacher,[old.base.project(y) for y in ys],refs,ids);baseline=read(ROOT/'evaluation/baseline_canonical.json');assert canonical['hypotheses']==baseline['hypotheses']
 emax=nmax=0.;rmsa=[];rmsb=[]
 for x,y in zip(xs,ys):
  ca=old.base.project(x);cb=old.base.project(y);ea=((x-ca).reshape(-1,40,6).astype(float)**2).sum(-1);eb=((y-cb).reshape(-1,40,6).astype(float)**2).sum(-1);emax=max(emax,float(abs(ea-eb).max()));nmax=max(nmax,float(abs((x.reshape(-1,40,6).astype(float)**2).sum(-1)-(y.reshape(-1,40,6).astype(float)**2).sum(-1)).max()));rmsa.append(float(np.sqrt(np.mean((x-ca)**2))));rmsb.append(float(np.sqrt(np.mean((y-cb)**2))))
 ranks=rankdata(np.r_[rmsa,rmsb]);n=len(rmsa);auc=float((ranks[n:].sum()-n*(n+1)/2)/(n*n));assert emax<2e-5 and nmax<2e-5
 r={'raw':raw,'canonical':canonical,'arm':arm,'per_frame_joint_canonical_distance_squared_max_difference':emax,'per_frame_joint_6D_squared_norm_max_difference':nmax,'mean_source_RMS':float(np.mean(rmsa)),'mean_layer_RMS':float(np.mean(rmsb)),'RMS_threshold_detector_AUC':auc,'RMS_best_orientation_AUC':max(auc,1-auc),**geometry(xs,ys)}
 write(str(folder.relative_to(ROOT))+'.json',r);np.savez_compressed(str(folder)+'_poses.npz',**{f'{kind}{i:04d}':x for kind,arr in [('attacked',ys),('auxiliary',ws)] for i,x in enumerate(arr)})
 print('CONSTANT_BUDGET_SCORED',folder.name,raw['scores']['bleu']['bleu4'],canonical['scores']['bleu']['bleu4'],r['mean_source_RMS'],r['mean_layer_RMS'],auc,flush=True)


def train():
 _,stoi,g=setup();register();assert (ROOT/'prepared.json').exists();records,xs=cache('train');dev,dx=cache('dev');teacher=old.base.make_back_translation_model(old.base.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
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
     groups=[ordered[p:p+64] for p in range(0,len(ordered),64)];random.shuffle(groups);lossvalues=[]
     for ids in groups:
      if epoch==0:orders.append(list(map(int,ids)))
      batch,tok=old.pack(records,xs,ids,stoi,collator,pad)
      if arm=='motion_raw':tok=torch.zeros_like(tok);tok[:,0]=2
      y,w=model(batch.sgn,tok,batch.sgn_lengths.to(batch.sgn.device));consumed=old.base.physical(y) if arm=='text_canonical' else y;out,_=teacher.forward(consumed,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask);logits=out[0];lt=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad,reduction='none').reshape(batch.txt.shape);loss=(lt.sum(1)/(batch.txt!=pad).sum(1)).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();lossvalues.append(float(loss.detach()))
     r={'epoch':epoch+1,'TRAIN_CE':float(np.mean(lossvalues)),'seconds':time.time()-started};history.append(r);write(str(folder.relative_to(ROOT)/'history.json'),history);print('CONSTANT_BUDGET_EPOCH',seed,arm,json.dumps(r),flush=True)
    assert orders==read(old.ROOT/f'training/seed{seed}/text_raw_batches.json');torch.save({'state_dict':model.state_dict(),'seed':seed,'arm':arm,'epoch':20,'selection':'last epoch'},checkpoint);write(str(folder.relative_to(ROOT)/'first_epoch_batches.json'),orders)
   model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True);path=ROOT/f'evaluation/seed{seed}_{arm}'
   if not Path(str(path)+'.json').exists():evaluate(model,teacher,dev,dx,stoi,path,arm)
   del model;torch.cuda.empty_cache()
 write('complete.json',{'fits':9,'seeds':SEEDS,'arms':ARMS,'all_first_epoch_batches_match_original_study':True,'no_TEST_use':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train']);a=p.parse_args();globals()[a.phase]()
