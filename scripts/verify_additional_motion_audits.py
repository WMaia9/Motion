"""Recompute all saved metrics and verify native geometry for every completed cell."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from motion_release_audit_common import read,write,sha,seed,metric
from score_humanml_blind_fid import matrices
from build_generalization_presentation import summarize,ADD,MD,OLD
from tools.canonicalize_motion import canonicalize_motion

def verify(root,dataset,method):
    assert (root/'complete.json').exists();p=read(root/'protocol.json')
    for f,h in p.get('source_and_assets_sha256',{}).items():assert sha(f)==h,f
    rows=[];pairs=0;fkcount=0.;maxangle=maxmatrix=maxnative=maxFK=maxFKrelative=maxFID=0.;joints=22 if dataset=='humanml' else 21;start=4+3*(joints-1);end=start+6*(joints-1);keep=np.r_[0:start,end:(263 if dataset=='humanml' else 251)]
    from data_loaders.humanml.common.quaternion import cont6d_to_matrix
    from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain,kit_raw_offsets,kit_kinematic_chain
    offsets,chain=(t2m_raw_offsets,t2m_kinematic_chain) if dataset=='humanml' else (kit_raw_offsets,kit_kinematic_chain)
    for rep in range(20):
        out=root/f'rep_{rep:02d}';row=read(out/'metrics.json');X=np.load(out/'generated_raw.npy',mmap_mode='r');L=np.load(out/'conditioning.npz')['lengths'];T=np.load(out/'text_embeddings.npy');reference=np.load(OLD/f'real_controls/rep_{rep:02d}/reference_embeddings.npy') if dataset=='humanml' else np.load(out/'reference_embeddings.npy')
        for condition in ['raw','canonical']:
            E=np.load(out/(condition+'_motion_embeddings.npy'));scores=metric(T,E,reference);expected=row['scores'][condition];np.testing.assert_allclose(scores['R_precision'],expected['R_precision'],atol=0,rtol=0);er=abs(scores['FID']-expected['FID']);assert er<1e-5;maxFID=max(maxFID,er);assert abs(scores['MM_distance']-expected['MM_distance'])<1e-7
        for i in range(0,len(X),32):
            raw=np.array(X[i:i+32]);ll=L[i:i+32];Y=canonicalize_motion(raw,ll);mask=np.arange(196)[None,:]<ll[:,None];assert np.array_equal(raw[...,keep].view('u4'),Y[...,keep].view('u4'));assert np.array_equal(raw[~mask].view('u4'),Y[~mask].view('u4'))
            a=raw[...,start:end].reshape(*raw.shape[:2],joints-1,6)[mask];b=Y[...,start:end].reshape(*raw.shape[:2],joints-1,6)[mask];m0=matrices(a);m1=matrices(b);er=float(abs(m0-m1).max());assert er<2e-5;maxmatrix=max(maxmatrix,er);angle=float(np.degrees(Rotation.from_matrix((m0.swapaxes(-1,-2)@m1).reshape(-1,3,3)).magnitude()).max());assert angle<.01;maxangle=max(maxangle,angle)
            er=float(abs(cont6d_to_matrix(torch.tensor(a))-cont6d_to_matrix(torch.tensor(b))).max());assert er<5e-5;maxnative=max(maxnative,er);pairs+=len(raw)
            if i==0:
                for j in range(2):
                    x=torch.tensor(raw[j,:ll[j]]);y=torch.tensor(Y[j,:ll[j]]);pos=recover_from_ric(x,joints);assert torch.equal(pos,recover_from_ric(y,joints));sk=Skeleton(torch.tensor(offsets),chain,'cpu');sk.get_offsets_joints(pos[0]);er=float(abs(recover_from_rot(x,joints,sk)-recover_from_rot(y,joints,sk)).max())
                    # Native KIT coordinates are hundreds/thousands of units,
                    # unlike meter-scale HumanML. Do not label KIT units as m
                    # or apply a meter threshold to its raw coordinates.
                    height=float((pos[:,:,1].max(1).values-pos[:,:,1].min(1).values).mean());assert height>0;relative=er/height;assert relative<2e-6
                    maxFK=max(maxFK,er);maxFKrelative=max(maxFKrelative,relative);fkcount+=1
        rows.append(row);print('ADDITIONAL_VERIFIED',dataset,method,rep,flush=True)
    record={'passed':True,'clip_pairs':pairs,'native_FK_pairs':int(fkcount),'saved_metric_endpoints':40,'max_rotation_error_deg':maxangle,'max_matrix_error':maxmatrix,'max_native_float32_matrix_error':maxnative,'max_native_FK_error_native_units':maxFK,'max_native_FK_error_divided_by_mean_body_height':maxFKrelative,'coordinate_units':'m' if dataset=='humanml' else 'native_processed_KIT_units_not_assumed_m','protected_features_and_padding_bitwise_equal':True,'max_saved_FID_error':maxFID,'portable_projection_used_independently':True,'source_and_asset_hashes_passed':True,'verifier_sha256':sha(__file__)}
    write(root/'verification.json',record);return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('study',choices=['additional','motiondiffuse']);a=parser.parse_args();seed(140);sys.path.insert(0,str(Path('external/MDM').resolve()))
    # Native metric modules import only these qualified MDM namespaces.
    from data_loaders.humanml.utils import metrics
    if a.study=='additional':cases=[(ADD/'humanml/momask','humanml','momask'),(ADD/'humanml/dip','humanml','dip'),(ADD/'kit/momask','kit','momask'),(ADD/'kit/real_controls','kit','real_controls')];root=ADD
    else:cases=[(MD/'humanml/motiondiffuse','humanml','motiondiffuse'),(MD/'kit/motiondiffuse','kit','motiondiffuse')];root=MD
    assert not (root/'manifest_sha256.json').exists(), 'Frozen study: verify_frozen_research.py checks it read-only; use a new study root for recomputation'
    results={};summaries={}
    for path,dataset,method in cases:
        if (path/'verification.json').exists():assert read(path/'verification.json')['passed'];record=read(path/'verification.json')
        else:record=verify(path,dataset,method)
        results[dataset+'/'+method]=record
        if method!='real_controls':summaries[dataset+'/'+method]=summarize(path)
    write(root/'analysis.json',{'completed_methods':summaries,'comparison_family':10,'all20repetitions_each':True,'MotionDiffuse_eager_partial_attempts_excluded':True,'source_sha256':sha(__file__)})
    write(root/'verification.json',{'passed':True,'studies':results,'all_completed_cells_verified':True,'verifier_sha256':sha(__file__)})
    write(root/'complete.json',{'all_scientific_endpoints_for_this_root_complete':True,'initial_aborted_attempts_not_results':True});print('ADDITIONAL_STUDY_VERIFIED',a.study,flush=True)
if __name__=='__main__':main()
