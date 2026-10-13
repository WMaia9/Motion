"""Caption-blind Gaussian transport targets; withheld TEST reference scored last."""
import numpy as np
import torch
from pathlib import Path
from humanml_audit_common import A,configure,load_evaluator,encode_raw,transform,canonical,sha,read,write
R=Path('reports/cvpr_humanml3d_blind_fid_20261012')
def symmetric_root(x,inverse=False):
 eigen,u=np.linalg.eigh((x+x.T)/2);floor=1e-4*np.trace(x)/len(x);eigen=np.maximum(eigen,floor);v=1/np.sqrt(eigen) if inverse else np.sqrt(eigen);return (u*v)@u.T

def main():
 assert not (R/'manifest_sha256.json').exists();p=read(R/'protocol.json')
 for key in ['inputs_sha256','scientific_sources_sha256']:
  for path,h in p[key].items():assert sha(path)==h,path
 configure();movement,motion,text,mean,std=load_evaluator();mean=torch.tensor(mean,device='cuda');std=torch.tensor(std,device='cuda');assert motion.gru.dropout==0
 def embeddings(X,L):
  out=[]
  with torch.no_grad():
   for i in range(0,len(X),32):out.append(encode_raw(torch.tensor(X[i:i+32],device='cuda'),torch.tensor(L[i:i+32],device='cuda'),movement,motion,mean,std).cpu().numpy())
  return np.concatenate(out)
 # Only TRAIN motion is loaded for distribution targets. The attack never calls
 # the text encoder, opens TEST captions or loads TEST real motion/reference.
 with np.load(R/'train_real.npz') as z:train=embeddings(z['raw'],z['lengths'])
 np.save(R/'TRAIN_motion_embeddings.npy',train)
 with np.load(R/'test_mdm.npz') as z:X=z['raw'].copy();L=z['lengths'].copy()
 src=embeddings(X,L);mus=src.mean(0).astype(float);mut=train.mean(0).astype(float);Cs=np.cov(src,rowvar=False);Ct=np.cov(train,rowvar=False)
 sq=symmetric_root(Cs);iv=symmetric_root(Cs,True);map_=iv@symmetric_root(sq@Ct@sq)@iv;goals=(src-mus)@map_+mut
 np.savez_compressed(R/'transport_targets.npz',generated_embeddings=src,targets=goals,TRAIN_mean=mut,TRAIN_covariance=Ct,source_mean=mus,source_covariance=Cs,map=map_)
 write(R/'blind_inputs.json',{'captions_used_by_attack':False,'TEST_reference_used_by_attack':False,'reference_distribution':'TRAIN motions only','TRAIN_clips':len(train),'TEST_output_clips':len(src),'transport_targets_sha256':sha(R/'transport_targets.npz'),'code_sha256':sha(Path(__file__))})
 for condition in p['conditions']:
  out=R/'attacks'/(condition+'.npz');assert not out.exists()
  ys=[];cs=[];es=[];ces=[];losses=[];rng=np.random.default_rng(42)
  for start in range(0,len(X),32):
   x=torch.tensor(X[start:start+32],device='cuda');ll=torch.tensor(L[start:start+32],device='cuda');mask=torch.arange(196,device='cuda')[None,:]<ll[:,None];shape=(*x.shape[:2],21,3)
   if condition=='identity':c=torch.zeros(shape,device='cuda');y=x
   elif condition=='canonical':c=torch.zeros(shape,device='cuda');y=canonical(x)
   elif condition.startswith('random'):
    c=torch.tensor(rng.uniform(-.35,.35,size=shape),device='cuda',dtype=torch.float32)*mask[...,None,None];y=transform(x,c)
   else:
    bound=.1 if condition.endswith('010') else .35;q=torch.zeros(shape,device='cuda',requires_grad=True);opt=torch.optim.Adam([q],lr=.1);goal=torch.tensor(goals[start:start+len(x)],device='cuda',dtype=torch.float32);motion.gru.train();history=[]
    for step in range(80):
     c=bound*torch.tanh(q)*mask[...,None,None];y=transform(x,c);z=canonical(y) if condition.startswith('canonical_reader') else y;e=encode_raw(z,ll,movement,motion,mean,std);loss=((e-goal)**2).mean();opt.zero_grad();loss.backward();opt.step();history.append(float(loss.detach()))
    motion.gru.eval();c=bound*torch.tanh(q.detach())*mask[...,None,None];y=transform(x,c);losses.append(history)
   with torch.no_grad():e=encode_raw(y,ll,movement,motion,mean,std);ce=encode_raw(canonical(y),ll,movement,motion,mean,std)
   ys.append(y.detach().cpu().numpy());cs.append(c.detach().cpu().numpy());es.append(e.cpu().numpy());ces.append(ce.cpu().numpy())
   if 'OT_' in condition:print('BLIND_ATTACK',condition,start+len(x),len(X),flush=True)
  out.parent.mkdir(exist_ok=True);np.savez_compressed(out,raw=np.concatenate(ys),coefficients=np.concatenate(cs),embeddings=np.concatenate(es),canonical_embeddings=np.concatenate(ces));write(out.with_suffix('.json'),{'condition':condition,'clips':len(X),'caption_blind':True,'TRAIN_distribution_target_only':True,'step_loss_histories':losses,'attack_arrays_sha256':sha(out)})
 write(R/'attack_outputs_before_TEST_scoring.json',{'files_sha256':{str(path):sha(path) for path in sorted((R/'attacks').glob('*.npz'))},'TEST_reference_not_loaded':True,'text_encoder_never_called_by_attack':True,'TEST_scores_not_observed':True,'all_conditions_completed':True})
 print('BLIND_ATTACKS_FIXED_BEFORE_SCORING',flush=True)
if __name__=='__main__':main()
