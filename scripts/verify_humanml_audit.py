"""Independently check native converters, physical controls and metric sensitivity."""
import json,sys,types,os
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanml_audit_common import R,A,read,write,sha,load_evaluator,motion_embedding,configure

def matrices(d):
 a=d[...,:3].astype(float);b=d[...,3:].astype(float);u=a/np.linalg.norm(a,axis=-1,keepdims=True);v=b-(u*b).sum(-1,keepdims=True)*u;v/=np.linalg.norm(v,axis=-1,keepdims=True);return np.stack([u,v,np.cross(u,v)],axis=-1)
def main():
 assert not (R/'manifest_sha256.json').exists();configure();p=read(R/'protocol.json')
 for key in ['inputs_sha256','scientific_sources_sha256']:
  for path,h in p[key].items():assert sha(path)==h,path
 sys.path.insert(0,str(Path('external/MDM').resolve()))
 from data_loaders.humanml.common.quaternion import cont6d_to_matrix
 from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot
 from data_loaders.humanml.common.skeleton import Skeleton
 from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
 from data_loaders.humanml.utils.metrics import calculate_R_precision,calculate_matching_score,calculate_activation_statistics,calculate_frechet_distance
 from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
 cwd=os.getcwd();os.chdir(A);native=EvaluatorMDMWrapper('humanml',torch.device('cuda'));os.chdir(cwd)
 move,motion,text,mean,std=load_evaluator();pairs=0;amax=mmax=reconmax=ricmax=cmax=embeddingparity=0.;sorting_changes={};fk_pairs=0;fkmax=0.
 for split in ['val','test']:
  selection=read(R/f'{split}_selection.json');N=len(selection);T=np.load(R/f'{split}_text_embeddings.npy');orders=np.load(R/f'{split}_orders.npy');reference=np.load(R/f'{split}_reference_embeddings.npy')
  for source in ['real','mdm']:
   with np.load(R/f'{split}_{source}.npz') as z:X=z['raw'].copy();lengths=z['lengths'].copy()
   if split=='val':
    norm=(X[:32]-mean)/std;mask=np.arange(196)[None,:]<lengths[:32,None];norm*=mask[...,None]
    xx=torch.tensor(norm,device='cuda');ll=torch.tensor(lengths[:32]);idx=np.argsort(lengths[:32])[::-1].copy()
    with torch.no_grad():a=motion_embedding(xx,ll.to('cuda'),move,motion).cpu().numpy();b=native.get_motion_embeddings(xx,ll).cpu().numpy()
    er=float(abs(a[idx]-b).max());assert er<1e-5;embeddingparity=max(embeddingparity,er)
   for condition in p['fixed_conditions']+p['adaptive_conditions']:
    path=R/split/source/(condition+'.npz');native_r1=[];stable_r1=[]
    with np.load(path) as z:
     Y=z['raw'];C=z['coefficients'];emb=z['embeddings'];can=z['canonical_embeddings']
     assert X.shape==Y.shape and np.isfinite(Y).all()
     assert np.array_equal(X[...,:67].view(np.uint32),Y[...,:67].view(np.uint32))
     assert np.array_equal(X[...,193:].view(np.uint32),Y[...,193:].view(np.uint32))
     base=np.load(R/split/source/'identity.npz')['canonical_embeddings'];er=float(abs(can-base).max());assert er<.002;cmax=max(cmax,er)
     if condition!='canonical':
      d=X[...,67:193].reshape(N,196,21,6).astype(float);rebuilt=np.concatenate([np.exp(C[...,0:1].astype(float))*d[...,:3],np.exp(C[...,1:2].astype(float))*d[...,3:]+C[...,2:3].astype(float)*d[...,:3]],-1)
      er=float(abs(rebuilt-Y[...,67:193].reshape(d.shape)).max());assert er<2e-6;reconmax=max(reconmax,er)
     for i,n in enumerate(lengths):
      d=X[i,:n,67:193].reshape(n,21,6);e=Y[i,:n,67:193].reshape(n,21,6);r0,r1=matrices(d),matrices(e)
      er=float(abs(r0-r1).max());assert er<1e-5;mmax=max(mmax,er)
      angle=float(np.degrees(Rotation.from_matrix((r0.transpose(0,1,3,2)@r1).reshape(-1,3,3)).magnitude()).max());assert angle<.01;amax=max(amax,angle)
      nr0=cont6d_to_matrix(torch.tensor(d));nr1=cont6d_to_matrix(torch.tensor(e));assert float(abs(nr0-nr1).max())<2e-5
      pos0=recover_from_ric(torch.tensor(X[i,:n]),22);pos1=recover_from_ric(torch.tensor(Y[i,:n]),22);assert torch.equal(pos0,pos1);ricmax=max(ricmax,float(abs(pos0-pos1).max()));pairs+=1
      if i<8:
       skel=Skeleton(torch.tensor(t2m_raw_offsets),t2m_kinematic_chain,'cpu');skel.get_offsets_joints(pos0[0]);f0=recover_from_rot(torch.tensor(X[i,:n]),22,skel);f1=recover_from_rot(torch.tensor(Y[i,:n]),22,skel);fer=float(abs(f0-f1).max());assert fer<1e-4;fkmax=max(fkmax,fer);fk_pairs+=1
     for order in orders:
      total=0;stable=0
      for idx in order.reshape(-1,32):
       total+=int(calculate_R_precision(T[idx],emb[idx],top_k=1,sum_all=True)[0])
       D=np.linalg.norm(T[idx,None,:]-emb[None,idx,:],axis=-1);stable+=int((np.argsort(D,axis=1,kind='stable')[:,0]==np.arange(32)).sum())
      native_r1.append(total/N);stable_r1.append(stable/N)
     sorting_changes[f'{split}/{source}/{condition}']={'native_R1':float(np.mean(native_r1)),'audit_R1':float(np.mean(stable_r1)),'difference':float(np.mean(native_r1)-np.mean(stable_r1))}
     er=float(abs(calculate_matching_score(T,emb)-np.linalg.norm(T-emb,axis=1)).max());assert er<2e-6
     mu,cov=calculate_activation_statistics(emb);mr,cr=calculate_activation_statistics(reference);fid=float(calculate_frechet_distance(mu,cov,mr,cr));saved=read(path.with_suffix('.json'))['actual_caption']['FID'];assert abs(max(fid,0)-saved)<1e-5
 write(R/'verification.json',{'passed':True,'rotation_and_bitwise_physical_pairs':pairs,'max_float64_rotation_error_deg':amax,'max_rotation_matrix_abs_error':mmax,'max_coefficient_reconstruction_error':reconmax,'native_recover_from_ric_max_error':ricmax,'native_FK_pairs_checked':fk_pairs,'native_FK_max_absolute_position_change':fkmax,'nonrotation_channels_bitwise_unchanged':True,
  'native_evaluator_embedding_adapter_max_error':embeddingparity,'canonical_embedding_max_absolute_change':cmax,'native_matching_score_and_FID_verified':True,'native_R1_distance_and_sort_sensitivity':sorting_changes,
  'data_mirror_not_full_byte_verified_against_original_Drive':True,'not_training_or_human_quality_evidence':True,'code_sha256':sha(Path(__file__))})
 print('HUMANML_VERIFIED',pairs,amax,embeddingparity,flush=True)
if __name__=='__main__':main()
