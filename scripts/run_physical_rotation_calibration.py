"""Physical angular sensitivity on coherent FK/positions/velocity features.

The zero-angle reconstruction is scored explicitly. Rotation-only changes that
leave stale position/velocity features would not be a physical calibration.
"""
from pathlib import Path
import sys
import numpy as np
import torch
from motion_release_audit_common import R as INPUT,read,write,sha,seed,evaluator,score_embeddings,canonical,metric
from run_motion_release_audit import recode
R=Path('reports/cvpr_physical_calibration_20261013')
ANGLES=[.1,.3,1.,3.,10.,30.]
KIN=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]

def physical_features(raw,L,degrees,kind,rep,start):
 from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_root_rot_pos
 from data_loaders.humanml.common.quaternion import quaternion_to_cont6d,cont6d_to_matrix,qrot
 from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets
 x=torch.tensor(raw,dtype=torch.float32);B,T,_=x.shape;q,root=recover_root_rot_pos(x);positions=recover_from_ric(x,22);local=cont6d_to_matrix(torch.tensor(canonical(raw)[...,67:193].reshape(B,T,21,6)))
 # Per-record axes avoid dependence on memory or scoring batches.
 axis=[]
 for j in range(B):
  rng=np.random.default_rng(20261013+rep*10000+start+j);shape=(1,21,3) if kind=='static' else (T,21,3);a=rng.normal(size=shape);a/=np.linalg.norm(a,axis=-1,keepdims=True);axis.append(np.broadcast_to(a,(T,21,3)))
 axis=torch.tensor(np.stack(axis),dtype=torch.float32);angle=np.radians(degrees);K=torch.zeros((*axis.shape[:-1],3,3));K[...,0,1]=-axis[...,2];K[...,0,2]=axis[...,1];K[...,1,0]=axis[...,2];K[...,1,2]=-axis[...,0];K[...,2,0]=-axis[...,1];K[...,2,1]=axis[...,0]
 E=torch.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K);changed=local@E;rootR=cont6d_to_matrix(quaternion_to_cont6d(q));raw_offsets=torch.tensor(t2m_raw_offsets).float();offsets=raw_offsets[None].expand(B,-1,-1).clone();parents=[-1]*22
 for chain in KIN:
  for p,c in zip(chain[:-1],chain[1:]):parents[c]=p
 for j in range(1,22):offsets[:,j]*=torch.linalg.norm(positions[:,0,j]-positions[:,0,parents[j]],dim=-1,keepdim=True)
 P=torch.zeros((B,T,22,3));P[:,:,0]=root
 for chain in KIN:
  rot=rootR
  for parent,joint in zip(chain[:-1],chain[1:]):
   rot=rot@changed[:,:,joint-1];P[:,:,joint]=P[:,:,parent]+(rot@offsets[:,None,joint,:,None]).squeeze(-1)
 y=x.clone();translated=P.clone();translated[...,0]-=root[...,0,None];translated[...,2]-=root[...,2,None];ric=qrot(q[:,:,None].expand(-1,-1,22,-1),translated)
 y[...,4:67]=ric[:,:,1:].reshape(B,T,63);y[...,67:193]=torch.cat([changed[...,0],changed[...,1]],dim=-1).reshape(B,T,126)
 v=P[:,1:]-P[:,:-1];v=qrot(q[:,:-1,None].expand(-1,-1,22,-1),v);y[:,:-1,193:259]=v.reshape(B,T-1,66)
 for col,joint in enumerate([7,10,8,11]):y[:,:-1,259+col]=(torch.sum((P[:,1:,joint]-P[:,:-1,joint])**2,-1)<.002).float()
 newL=np.asarray(L)-4;assert newL.min()>=4;mask=torch.arange(T)[None,:]<torch.tensor(newL)[:,None];y[~mask]=0
 # Return FK states for independent sampled native reconstruction.
 return y.numpy(),newL,P.numpy(),positions.numpy()

def register():
 R.mkdir(exist_ok=True);files=[Path(__file__),Path('scripts/run_motion_release_audit.py'),Path('scripts/motion_release_audit_common.py'),INPUT/'manifest_sha256.json']
 p={'scope':'angular sensitivity, not equivalence in human-perceived quality','replications':20,'clips_each':4640,'source':'all saved real_controls native TEST cohorts, no subset chosen by scores','angles_degrees':ANGLES,'noise_modes':['static joint axis per clip','independent frame/joint axis'],'amplitude':'fixed local SO(3) angle at all21nonroot joints; global consequences accumulate along kinematic chains','coherence':'FK with per-clip initial bone lengths; recomputed RIC positions, local velocities and foot contacts; root trajectory fixed','zero_angle_reconstruction_scored':True,'crop':'last4frames removed to derive velocities from available next-frame positions; common crop for all controls','FID_reference':'paired zero-angle coherent embedding cohort; also report original-to-zero reconstruction FID separately','controls':['coherent_zero','C_zero','global_plus010','global_minus010','global_plus035','global_minus035','random035_seed42'],'not_a_universal_conversion_between_FID_and_degrees':True,'native_evaluator':'primary common MDM-linked weights, 32candidate retrieval','axis_seeds':'20261013+10000*rep+clipindex; shared axes across angle doses','inputs_and_sources_sha256':{str(f):sha(f) for f in files}}
 if (R/'protocol.json').exists():assert read(R/'protocol.json')==p
 else:write(R/'protocol.json',p)
 return p

def main():
 assert not (R/'manifest_sha256.json').exists();register();seed(434);wrapper,mean,std=evaluator();sys.path.insert(0,'external/MDM')
 from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot
 from data_loaders.humanml.common.skeleton import Skeleton
 from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
 for rep in range(20):
  folder=R/f'rep_{rep:02d}';folder.mkdir(exist_ok=True)
  if (folder/'metrics.json').exists():continue
  source=INPUT/'real_controls'/f'rep_{rep:02d}';x=np.load(source/'source_raw.npy',mmap_mode='r');z=np.load(source/'conditioning.npz');cohort={k:z[k].copy() for k in z.files};L=cohort['lengths'].copy();cohort['lengths']=L-4;zero=[];checks=[]
  for i in range(0,len(x),64):
   y,n,P,old=physical_features(np.array(x[i:i+64]),L[i:i+64],0.,'static',rep,i);zero.append(y)
   a=torch.tensor(y[0,:int(n[0])]);p=recover_from_ric(a,22);s=Skeleton(torch.tensor(t2m_raw_offsets),t2m_kinematic_chain,'cpu');s.get_offsets_joints(p[0]);f=recover_from_rot(a,22,s);assert torch.max(abs(p-f))<2e-4;checks.append({'RIC_FK_max':float(abs(p-f).max()),'source_vs_zero_RMS':float(np.sqrt(np.mean((P[0,:n[0]]-old[0,:n[0]])**2)))})
  zero=np.concatenate(zero);np.save(folder/'coherent_zero.npy',zero);T,ref=score_embeddings(wrapper,zero,cohort,mean,std);np.save(folder/'reference_embeddings.npy',ref);scores={};Tx,Ex=score_embeddings(wrapper,x,cohort,mean,std);scores['original_vs_coherent_zero']=metric(Tx,Ex,ref)
  for name in ['coherent_zero','C_zero','global_plus010','global_minus010','global_plus035','global_minus035','random035_seed42']:
   y=zero if name=='coherent_zero' else canonical(zero) if name=='C_zero' else recode(zero,name);T,E=score_embeddings(wrapper,y,cohort,mean,std);scores[name]=metric(T,E,ref);np.save(folder/f'{name}_embeddings.npy',E)
  for kind in ['static','iid']:
   for angle in ANGLES:
    blocks=[];displacement=[]
    for i in range(0,len(x),64):
     y,n,P,_=physical_features(np.array(x[i:i+64]),L[i:i+64],angle,kind,rep,i)
     subset={k:v[i:i+64] for k,v in cohort.items()};T,E=score_embeddings(wrapper,y,subset,mean,std);blocks.append((T,E))
     # Sampled native position control applies to the perturbed states too.
     a=torch.tensor(y[0,:int(n[0])]);p=recover_from_ric(a,22);s=Skeleton(torch.tensor(t2m_raw_offsets),t2m_kinematic_chain,'cpu');s.get_offsets_joints(p[0]);f=recover_from_rot(a,22,s);assert abs(p-f).max()<2e-4
    T=np.concatenate([q[0] for q in blocks]);E=np.concatenate([q[1] for q in blocks]);name=f'{kind}_{angle:g}deg';scores[name]=metric(T,E,ref);np.save(folder/f'{name}_embeddings.npy',E);print('PHYSICAL_SCORED',rep,name,scores[name],flush=True)
  write(folder/'metrics.json',{'rep':rep,'scores':scores,'sampled_zero_checks':checks});print('PHYSICAL_REP_COMPLETE',rep,flush=True)
 write(R/'complete.json',{'all20repetitions':True,'all_noise_and_fixed_controls_retained':True})
if __name__=='__main__':main()
