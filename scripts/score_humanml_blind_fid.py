"""Score sealed caption-blind attack outputs; verify physical controls independently."""
from pathlib import Path
import sys,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanml_audit_common import A,configure,load_evaluator,encode_raw,text_embeddings,metrics,read,write,sha
R=Path('reports/cvpr_humanml3d_blind_fid_20261012')

def matrices(d):
 a=d[...,:3].astype(float);b=d[...,3:].astype(float);u=a/np.linalg.norm(a,axis=-1,keepdims=True);v=b-(u*b).sum(-1,keepdims=True)*u;v/=np.linalg.norm(v,axis=-1,keepdims=True);return np.stack([u,v,np.cross(u,v)],-1)
def root(x,inverse=False):
 e,u=np.linalg.eigh((x+x.T)/2);e=np.maximum(e,1e-4*np.trace(x)/len(x));return (u*(1/np.sqrt(e) if inverse else np.sqrt(e)))@u.T

def main():
 assert not (R/'manifest_sha256.json').exists();assert not (R/'analysis.json').exists()
 p=read(R/'protocol.json');seal=read(R/'attack_outputs_before_TEST_scoring.json');assert seal['all_conditions_completed']
 for path,h in seal['files_sha256'].items():assert sha(path)==h,path
 configure();movement,motion,text,mean,std=load_evaluator();mean=torch.tensor(mean,device='cuda');std=torch.tensor(std,device='cuda')
 # Held-out motion and target text are first numerically consumed here, after
 # all attack outputs have been saved and hashed. The preflight attack did
 # hash these input files, but did not deserialize or encode their contents.
 selection=read(R/'test_selection.json');N=len(selection)
 with np.load(R/'test_real.npz') as z:ref=[];LR=z['lengths'].copy()
 with np.load(R/'test_real.npz') as z:
  with torch.no_grad():
   for start in range(0,N,32):ref.append(encode_raw(torch.tensor(z['raw'][start:start+32],device='cuda'),torch.tensor(LR[start:start+32],device='cuda'),movement,motion,mean,std).cpu().numpy())
 reference=np.concatenate(ref);T=text_embeddings(selection,text);orders=np.array([np.random.default_rng(s).permutation(N) for s in [11,97,909,1337,2027]])
 np.savez_compressed(R/'withheld_TEST_reference_embeddings.npz',motion=reference,text=T,orders=orders)
 with np.load(R/'test_mdm.npz') as z:X=z['raw'].copy();L=z['lengths'].copy()
 sys.path.insert(0,str(Path('external/MDM').resolve()))
 from data_loaders.humanml.common.quaternion import cont6d_to_matrix
 from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot
 from data_loaders.humanml.common.skeleton import Skeleton
 from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
 from data_loaders.humanml.utils.metrics import calculate_R_precision,calculate_activation_statistics,calculate_frechet_distance
 results={};pairs=0;amax=mmax=reconmax=fkmax=cmax=0.;fkcount=0
 with np.load(R/'attacks/identity.npz') as z:basecan=z['canonical_embeddings'].copy()
 for name in p['conditions']:
  with np.load(R/'attacks'/(name+'.npz')) as z:Y=z['raw'];C=z['coefficients'];E=z['embeddings'];CE=z['canonical_embeddings']
  assert X.shape==Y.shape and np.isfinite(Y).all()
  assert np.array_equal(X[...,:67].view(np.uint32),Y[...,:67].view(np.uint32));assert np.array_equal(X[...,193:].view(np.uint32),Y[...,193:].view(np.uint32))
  er=float(abs(CE-basecan).max());assert er<.002;cmax=max(cmax,er)
  if name!='canonical':
   d=X[...,67:193].reshape(N,196,21,6).astype(float);reb=np.concatenate([np.exp(C[...,0:1].astype(float))*d[...,:3],np.exp(C[...,1:2].astype(float))*d[...,3:]+C[...,2:3].astype(float)*d[...,:3]],-1);er=float(abs(reb-Y[...,67:193].reshape(d.shape)).max());assert er<2e-6;reconmax=max(reconmax,er)
  for i,n in enumerate(L):
   d=X[i,:n,67:193].reshape(n,21,6);e=Y[i,:n,67:193].reshape(n,21,6);r0,r1=matrices(d),matrices(e);er=float(abs(r0-r1).max());assert er<1e-5;mmax=max(mmax,er)
   angle=float(np.degrees(Rotation.from_matrix((r0.transpose(0,1,3,2)@r1).reshape(-1,3,3)).magnitude()).max());assert angle<.01;amax=max(amax,angle)
   assert float(abs(cont6d_to_matrix(torch.tensor(d))-cont6d_to_matrix(torch.tensor(e))).max())<2e-5
   pos0=recover_from_ric(torch.tensor(X[i,:n]),22);pos1=recover_from_ric(torch.tensor(Y[i,:n]),22);assert torch.equal(pos0,pos1);pairs+=1
   if i<8:
    s=Skeleton(torch.tensor(t2m_raw_offsets),t2m_kinematic_chain,'cpu');s.get_offsets_joints(pos0[0]);f0=recover_from_rot(torch.tensor(X[i,:n]),22,s);f1=recover_from_rot(torch.tensor(Y[i,:n]),22,s);er=float(abs(f0-f1).max());assert er<1e-4;fkmax=max(fkmax,er);fkcount+=1
  score=metrics(E,T,reference,orders);can=metrics(CE,T,reference,orders);native=[]
  for order in orders:native.append(sum(calculate_R_precision(T[idx],E[idx],1,True)[0] for idx in order.reshape(-1,32))/N)
  assert abs(float(np.mean(native))-score['R_precision_top1'])<1e-10
  mu,cov=calculate_activation_statistics(E);mr,cr=calculate_activation_statistics(reference);fid=calculate_frechet_distance(mu,cov,mr,cr);assert abs(max(float(fid),0)-score['FID'])<1e-5
  results[name]={'raw':score,'canonical':can,'canonical_embedding_max_abs_change':float(abs(CE-basecan).max())}
 with np.load(R/'transport_targets.npz') as z:
  Cs=z['source_covariance'];Ct=z['TRAIN_covariance'];sq=root(Cs);iv=root(Cs,True);mapping=iv@root(sq@Ct@sq)@iv;goals=(z['generated_embeddings']-z['source_mean'])@mapping+z['TRAIN_mean'];assert np.max(abs(mapping-z['map']))<1e-10 and np.max(abs(goals-z['targets']))<1e-10
 train={x['clip_id'].lstrip('M') for x in read(R/'train_selection.json')};test={x['clip_id'].lstrip('M') for x in selection};old={x['clip_id'].lstrip('M') for x in read('reports/cvpr_humanml3d_audit_20261012/test_selection.json')};assert not test&old and not test&train
 baseline=results['identity']['raw']['FID']
 for v in results.values():v['FID_change']=v['raw']['FID']-baseline;v['relative_FID_reduction']=1-v['raw']['FID']/baseline
 write(R/'analysis.json',{'clips':N,'TRAIN_reference_clips':2048,'TEST_original_mirror_groups':len(test),'fresh_TEST_group_overlap':0,'conditions':results,'FID_descriptive_only':True,'not_generator_quality_improvement':True,'not_training_exploitation':True,'one_MDM_generation_seed':42,'attack_caption_content_and_TEST_reference_not_numerically_used':True,'source_code_sha256':sha(Path(__file__))})
 write(R/'verification.json',{'passed':True,'pairs':pairs,'native_FK_pairs':fkcount,'max_rotation_error_deg':amax,'max_rotation_matrix_error':mmax,'max_coefficient_reconstruction_error':reconmax,'native_RIC_positions_bitwise_unchanged':True,'other_feature_channels_bitwise_unchanged':True,'max_native_FK_position_error':fkmax,'max_canonical_embedding_change':cmax,'transport_targets_reconstructed_independently':True,'native_metrics_verified':True,'attack_arrays_hashed_before_TEST_reference_encoding':True,'TEST_group_overlap_with_previous_audit':0,'TRAIN_TEST_group_overlap':0,'preflight_hash_only_reads_of_TEST_files_disclosed':True,'source_code_sha256':sha(Path(__file__))})
 print(json.dumps({k:{'FID':v['raw']['FID'],'FID_change':v['FID_change'],'reduction':v['relative_FID_reduction'],'R1':v['raw']['R_precision_top1']} for k,v in results.items()},indent=2),flush=True)
if __name__=='__main__':main()
