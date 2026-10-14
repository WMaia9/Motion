"""Independently check saved metrics, native inference and physical controls."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from motion_release_audit_common import read,write,sha,seed,canonical
from tmr_audit_common import NativeTMR,retrieval
from score_humanml_blind_fid import matrices
NAT=Path('reports/cvpr_tmr_audit_20261013');ATT=Path('reports/cvpr_tmr_attack_20261013');OLD=Path('reports/cvpr_motion_prevalence_20261012')

def main():
 seed(424);assert read(NAT/'complete.json')['all120repetition_cells'];assert read(ATT/'complete.json')['all_sources_and_controls_completed']
 for root in [NAT,ATT]:
  p=read(root/'protocol.json')
  for key in ['input_and_source_sha256','source_sha256']:
   for path,h in p.get(key,{}).items():assert sha(path)==h,path
 model=NativeTMR();metriccells=0;infermax=0.;batchmax=0.;textmax=0.
 # Match the producer's first64-caption batches/cache before comparing its
 # saved text vectors. Changing BERT padding/batch sizes has separately
 # measured float32 differences up to4.7e-5, not a changed native equation.
 with np.load(OLD/'real_controls/rep_00/conditioning.npz') as z:model.texts(z['captions'].tolist())
 for method in read(NAT/'protocol.json')['methods']:
  for rep in range(20):
   folder=NAT/method/f'rep_{rep:02d}';row=read(folder/'metrics.json');T=np.load(folder/'text_embeddings.npy')
   for condition,reported in row['scores'].items():
    E=np.load(folder/(condition+'_embeddings.npy'));result=retrieval(T,E)
    np.testing.assert_allclose(result['R_precision'],reported['R_precision'],atol=0,rtol=0);assert abs(result['matched_cosine']-reported['matched_cosine'])<1e-7;metriccells+=1
   if rep==0:
    source=OLD/method/'rep_00';z=np.load(source/'conditioning.npz');L=z['lengths'][:8] if method=='real_controls' else np.load(source/'generated_lengths.npy')[:8];X=np.array(np.load(source/('source_raw.npy' if method=='real_controls' else 'generated_raw.npy'),mmap_mode='r')[:8]);E=model.motions(X,L);saved=np.load(folder/'raw_embeddings.npy')[:8];er=float(abs(E-saved).max());assert er<3e-5;infermax=max(er,infermax)
    serial=model.motions(X,L,batch=1);er=float(abs(serial-E).max());assert er<3e-5;batchmax=max(er,batchmax)
    target=model.texts(z['captions'][:8].tolist());er=float(abs(target-T[:8]).max());assert er<3e-5;textmax=max(er,textmax)
 sys.path.insert(0,str(Path('external/MDM').resolve()))
 from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot
 from data_loaders.humanml.common.skeleton import Skeleton
 from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
 from data_loaders.humanml.common.quaternion import cont6d_to_matrix
 allrows={};pairs=0;amax=0.;mmax=0.;fkmax=0.;nativeMmax=0.
 for source in ['real','mdm']:
  folder=ATT/source;z=np.load(folder/'source.npz');X=z['raw'];L=z['lengths'];T=z['text'];mask=np.arange(196)[None,:]<L[:,None];scores=read(folder/'metrics.json');baseline=z['canonical_embedding'];scale=model.std.cpu().numpy()+1e-12;rows={}
  for condition in ['actual','wrong','canonical_gradient','generic_L2']:
   a=np.load(folder/(condition+'.npz'));Y=a['raw'];E=a['embedding'];CE=a['canonical_embedding'];expected=scores[condition]['actual'];np.testing.assert_allclose(retrieval(T,E)['R_precision'],expected['R_precision'],rtol=0,atol=0)
   if condition!='generic_L2':
    assert np.array_equal(X[...,:67].view('u4'),Y[...,:67].view('u4'));assert np.array_equal(X[...,193:].view('u4'),Y[...,193:].view('u4'))
    c=a['coefficients'];assert abs(c).max()<=.3500001;assert np.array_equal(c[~mask],np.zeros_like(c[~mask]))
    d=X[...,67:193].reshape(*X.shape[:2],21,6)[mask];e=Y[...,67:193].reshape(*Y.shape[:2],21,6)[mask];r0=matrices(d);r1=matrices(e);er=float(abs(r0-r1).max());mmax=max(mmax,er);assert er<2e-5
    angle=float(np.degrees(Rotation.from_matrix((r0.swapaxes(-1,-2)@r1).reshape(-1,3,3)).magnitude()).max());amax=max(amax,angle);assert angle<.01
    ce_error=float(abs(CE-baseline).max());assert ce_error<3e-5
    nm=float(abs(cont6d_to_matrix(torch.tensor(d))-cont6d_to_matrix(torch.tensor(e))).max());nativeMmax=max(nativeMmax,nm);assert nm<5e-5
    for i in range(8):
     x=torch.tensor(X[i,:L[i]]);y=torch.tensor(Y[i,:L[i]]);P=recover_from_ric(x,22);assert torch.equal(P,recover_from_ric(y,22));sk=Skeleton(torch.tensor(t2m_raw_offsets),t2m_kinematic_chain,'cpu');sk.get_offsets_joints(P[0]);er=float(abs(recover_from_rot(x,22,sk)-recover_from_rot(y,22,sk)).max());fkmax=max(fkmax,er);assert er<2e-4
    pairs+=len(X);rows[condition]={'canonical_embedding_max_change':ce_error,'positions_and_protected_channels_bitwise_equal':True,'native_FK_sampled_clips':8}
   else:
    norm=np.sqrt(np.sum((((Y-X)/scale)*mask[...,None])**2,axis=(1,2)));radii=a['radii'];excess=float(np.max(norm-radii));assert excess<1e-3
    assert np.array_equal(Y[...,259:].view('u4'),X[...,259:].view('u4'));assert np.array_equal(X[~mask].view('u4'),Y[~mask].view('u4'))
    d=X[...,67:193].reshape(*X.shape[:2],21,6)[mask];e=Y[...,67:193].reshape(*Y.shape[:2],21,6)[mask];angles=np.degrees(Rotation.from_matrix((matrices(d).swapaxes(-1,-2)@matrices(e)).reshape(-1,3,3)).magnitude());displacements=[]
    for i in range(len(X)):
     p=recover_from_ric(torch.tensor(X[i,:L[i]]),22).numpy();q=recover_from_ric(torch.tensor(Y[i,:L[i]]),22).numpy();displacements.extend(np.linalg.norm(p-q,axis=-1).reshape(-1))
    rows[condition]={'standardized_L2_max_excess':excess,'standardized_L2_mean_gauge_budget':float(radii.mean()),'standardized_L2_generic_mean':float(norm.mean()),'mean_rotation_change_deg':float(angles.mean()),'max_rotation_change_deg':float(angles.max()),'mean_native_RIC_position_change_m':float(np.mean(displacements)),'contacts_and_padding_bitwise_preserved':True,'physical_consistency_between_all_feature_blocks_not_guaranteed':True}
  allrows[source]=rows
 write(NAT/'verification.json',{'passed':True,'saved_metric_cells':metriccells,'native_embedding_recompute_max':infermax,'batch_serial_max':batchmax,'native_text_recompute_max':textmax,'parent_frozen_manifest_sha256':sha(OLD/'manifest_sha256.json'),'protocol_source_hashes_passed':True,'verifier_sha256':sha(__file__)})
 write(ATT/'verification.json',{'passed':True,'geometry_clip_pairs':pairs,'native_FK_pairs':48,'max_rotation_error_deg':amax,'max_rotation_matrix_difference':mmax,'max_native_rotation_matrix_difference':nativeMmax,'max_native_FK_error_m':fkmax,'conditions':allrows,'native_score_recompute_and_protocol_hashes_passed':True,'verifier_sha256':sha(__file__)})
 print('TMR_VERIFIED',metriccells,allrows,flush=True)
if __name__=='__main__':main()
