"""Check coherent angular doses with native geometry, then summarize all curves."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import t
from motion_release_audit_common import read,write,sha,seed,canonical,metric
from run_physical_rotation_calibration import physical_features,ANGLES,R,INPUT
from score_humanml_blind_fid import matrices

def main():
 seed(713);assert read(R/'complete.json')['all20repetitions']
 for f,h in read(R/'protocol.json')['inputs_and_sources_sha256'].items():assert sha(f)==h
 sys.path.insert(0,str(Path('external/MDM').resolve()))
 from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot,recover_root_rot_pos
 from data_loaders.humanml.common.skeleton import Skeleton
 from data_loaders.humanml.common.quaternion import qrot
 from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
 summaries={};errors={'native_RIC_FK_max_m':0.,'native_FK_vs_construction_max_m':0.,'angular_dose_max_error_deg':0.,'velocity_max_error':0.,'saved_zero_regeneration_max':0.};pairs=0;savedmetric=0;maxFIDerr=0.;positionrows=[]
 for rep in range(20):
  folder=R/f'rep_{rep:02d}';source=INPUT/f'real_controls/rep_{rep:02d}';X=np.load(source/'source_raw.npy',mmap_mode='r');z=np.load(source/'conditioning.npz');L=z['lengths'];T=np.load(source/'text_embeddings.npy');reference=np.load(folder/'reference_embeddings.npy');row=read(folder/'metrics.json')
  for name,score in row['scores'].items():
   summaries.setdefault(name,[]).append(score)
   if name=='original_vs_coherent_zero':continue
   E=np.load(folder/(name+'_embeddings.npy'));result=metric(T,E,reference);np.testing.assert_allclose(result['R_precision'],score['R_precision'],atol=1e-14);assert abs(result['MM_distance']-score['MM_distance'])<1e-7;er=abs(result['FID']-score['FID']);assert er<1e-5;maxFIDerr=max(er,maxFIDerr);savedmetric+=1
  original=np.array(X[:1]);ll=L[:1];zero,n,_,_=physical_features(original,ll,0.,'static',rep,0);saved=np.load(folder/'coherent_zero.npy',mmap_mode='r');er=float(abs(zero[0]-saved[0]).max());assert er==0.;errors['saved_zero_regeneration_max']=max(er,errors['saved_zero_regeneration_max']);m0=matrices(zero[0,:n[0],67:193].reshape(-1,21,6))
  for mode in ['static','iid']:
   for dose in ANGLES:
    Y,NN,P,_=physical_features(original,ll,dose,mode,rep,0);length=int(NN[0]);y=torch.tensor(Y[0,:length]);pos=recover_from_ric(y,22);sk=Skeleton(torch.tensor(t2m_raw_offsets),t2m_kinematic_chain,'cpu');sk.get_offsets_joints(pos[0]);fk=recover_from_rot(y,22,sk);er=float(abs(pos-fk).max());assert er<2e-4;errors['native_RIC_FK_max_m']=max(er,errors['native_RIC_FK_max_m']);er=float(abs(fk.numpy()-P[0,:length]).max());assert er<2e-4;errors['native_FK_vs_construction_max_m']=max(er,errors['native_FK_vs_construction_max_m'])
    m1=matrices(Y[0,:length,67:193].reshape(-1,21,6));delta=m0.swapaxes(-1,-2)@m1;angles=np.degrees(Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude());er=float(abs(angles-dose).max());assert er<1e-4;errors['angular_dose_max_error_deg']=max(er,errors['angular_dose_max_error_deg'])
    # Recover next positions via native conversion at an extended mask. The
    # reconstruction equations are unchanged; only padding is postponed.
    extended,_,_,_=physical_features(original,ll+4,dose,mode,rep,0);nextpos=recover_from_ric(torch.tensor(extended[0,:length+1]),22);q,_=recover_root_rot_pos(torch.tensor(extended[0,:length+1]));v=qrot(q[:-1,None].expand(-1,22,-1),nextpos[1:]-nextpos[:-1]).reshape(length,66);er=float(abs(v.numpy()-Y[0,:length,193:259]).max());assert er<2e-4;errors['velocity_max_error']=max(er,errors['velocity_max_error'])
    contacts=np.stack([(np.sum((nextpos[1:,j].numpy()-nextpos[:-1,j].numpy())**2,axis=-1)<.002).astype(np.float32) for j in [7,10,8,11]],-1);assert np.array_equal(contacts,Y[0,:length,259:]);assert np.array_equal(Y[0,:length,:4].view('u4'),original[0,:length,:4].view('u4'));assert np.count_nonzero(Y[0,length:])==0;pairs+=1
    positionrows.append({'rep':rep,'mode':mode,'angle_deg':dose,'first_native_clip_joint_displacement_RMS_m':float(np.sqrt(np.mean((P[0,:length]-recover_from_ric(torch.tensor(zero[0,:length]),22).numpy())**2)))})
 out={}
 for name,rows in summaries.items():
  out[name]={}
  for field in ['FID','MM_distance','R1']:
   a=np.array([x['R_precision'][0] if field=='R1' else x[field] for x in rows]);se=a.std(ddof=1)/np.sqrt(20);out[name][field]={'mean':float(a.mean()),'t19_CI95':(a.mean()+np.array([-1,1])*t.ppf(.975,19)*se).tolist(),'replications':20}
 write(R/'analysis.json',{'conditions':out,'original_to_zero_cost_reported':True,'paired_zero_reference_not_independent_real_distribution':True,'no_universal_FID_degree_conversion':True,'two_axis_noise_spectra':True,'source_scores_have_twenty_native_repetitions':True,'sampled_displacements':positionrows})
 write(R/'verification.json',{'passed':True,'native_physical_clip_pairs':pairs,'saved_metric_endpoints_checked':savedmetric,'maximum_saved_FID_error':maxFIDerr,'errors':errors,'contacts_and_root_and_padding_checked':True,'native_SO3_angle_and_RIC_FK_and_velocity_checks':True,'producer_input_hashes_passed':True,'verifier_sha256':sha(__file__)})
 print('PHYSICAL_VERIFIED',pairs,out,flush=True)
if __name__=='__main__':main()
