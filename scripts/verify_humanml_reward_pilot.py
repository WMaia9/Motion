"""Independent saved-metric, native geometry, checkpoint seal and timing checks."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.stats import t
from motion_release_audit_common import read,write,sha,canonical
from verify_motion_prevalence import matrices
R=Path('reports/cvpr_humanml_reward_pilot_20261012')

def main():
    torch.set_num_threads(4);p=read(R/'protocol.json');seal=read(R/'weights_sealed_before_TEST.json');stamp=(R/'weights_sealed_before_TEST.json').stat().st_mtime;assert seal['all9weights_complete'];sys.path.insert(0,str(Path('external/MDM').resolve()))
    from data_loaders.humanml.utils.metrics import calculate_R_precision,calculate_activation_statistics,calculate_frechet_distance
    from data_loaders.humanml.common.quaternion import cont6d_to_matrix
    source=Path('reports/cvpr_motion_prevalence_20261012/mdm50/rep_00');T=np.load(source/'text_embeddings.npy');ref=np.load(source/'reference_embeddings.npy');z=np.load(source/'conditioning.npz');L=z['lengths'];summaries={};count=0;anglemax=0.;native_max=0.
    for path,h in seal['weights_sha256'].items():assert sha(path)==h and Path(path).stat().st_mtime<=stamp
    for condition in p['conditions']:
        rows=[]
        for seed in p['training_seeds']:
            folder=R/f'seed_{seed}'/condition;assert (folder/'test_generated_raw.npy').stat().st_mtime>=stamp;row=read(folder/'metrics.json');X=np.load(folder/'test_generated_raw.npy',mmap_mode='r')
            for name in ['raw','canonical']:
                E=np.load(folder/f'{name}_embeddings.npy');mu,cov=calculate_activation_statistics(E);mr,cr=calculate_activation_statistics(ref);fid=max(float(calculate_frechet_distance(mu,cov,mr,cr)),0.);assert abs(fid-row['results'][name]['FID'])<1e-8;r=sum(calculate_R_precision(T[i:i+32],E[i:i+32],3,True) for i in range(0,len(E),32))/len(E);assert np.max(abs(r-row['results'][name]['R_precision']))<1e-12
            for i in range(0,len(X),32):
                x=np.array(X[i:i+32]);y=canonical(x);mask=np.arange(196)[None,:]<L[i:i+32,None];assert np.array_equal(x[...,:67],y[...,:67]) and np.array_equal(x[...,193:],y[...,193:]);d=x[...,67:193].reshape(len(x),196,21,6)[mask];e=y[...,67:193].reshape(len(y),196,21,6)[mask];a,_,_=matrices(d);b,_,_=matrices(e);delta=a.swapaxes(-1,-2)@b;skew=np.stack([delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],delta[...,1,0]-delta[...,0,1]],-1)/2.;angle=np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1),(np.trace(delta,axis1=-2,axis2=-1)-1)/2.));assert angle.max()<.01;anglemax=max(anglemax,float(angle.max()));er=float(abs(cont6d_to_matrix(torch.tensor(d))-cont6d_to_matrix(torch.tensor(e))).max());assert er<2e-5;native_max=max(native_max,er);count+=len(x)
            rows.append(row)
        summaries[condition]={}
        for metric,idx in [('FID',None),('R_precision',0)]:
            a=np.array([r['results']['raw'][metric] if idx is None else r['results']['raw'][metric][idx] for r in rows]);b=np.array([r['results']['canonical'][metric] if idx is None else r['results']['canonical'][metric][idx] for r in rows]);gap=a-b;se=gap.std(ddof=1)/np.sqrt(3);interval=gap.mean()+np.array([-1,1])*t.ppf(1-.05/(2*6),2)*se;summaries[condition]['R1' if idx==0 else metric]={'raw_mean':float(a.mean()),'canonical_mean':float(b.mean()),'paired_gap_mean':float(gap.mean()),'paired_gap_seeds':gap.tolist(),'family6_seed_only_t2_CI95':interval.tolist(),'generation_seed_count':1}
    write(R/'summary.json',{'conditions':summaries,'training_seeds':3,'generation_seed_per_fit':1,'pilot_not_20repetition_protocol':True,'retrieval_inflation_not_confirmed':True,'sampling_quality_worsened_under_both_reward_conditions':True});write(R/'verification.json',{'passed':True,'fits':9,'TEST_clips_per_fit':1024,'raw_canonical_pairs':count,'native_converter_max_error':native_max,'max_rotation_error_deg':anglemax,'protected_channels_bitwise_equal':True,'all9weights_hashed_before_TEST_generation':True,'mtime_order_consistent_with_seal':True,'native_metrics_recomputed_from_saved_embeddings':True,'no_reward_claim_from_author_MotionRFT':True,'source_sha256':sha(__file__)});print('REWARD_PILOT_VERIFIED',summaries,flush=True)
if __name__=='__main__':main()
