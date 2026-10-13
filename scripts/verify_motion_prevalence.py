"""Independent native-converter/FK checks and saved-embedding metric checks."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy import linalg
from motion_release_audit_common import R,read,write,sha,canonical

def matrices(d):
    d=d.astype(float);a,b=d[...,:3],d[...,3:];rho=np.linalg.norm(a,axis=-1,keepdims=True);u=a/rho;v=b-(b*u).sum(-1,keepdims=True)*u;sigma=np.linalg.norm(v,axis=-1,keepdims=True);v/=sigma;assert rho.min()>1e-8 and sigma.min()>1e-8;return np.stack([u,v,np.cross(u,v)],-1),float(rho.min()),float(sigma.min())

def main():
    assert read(R/'analysis.json')['replications']==20
    sys.path.insert(0,str(Path('external/MDM').resolve()))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot
    from data_loaders.humanml.common.quaternion import cont6d_to_matrix
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
    from data_loaders.humanml.utils.metrics import calculate_R_precision,calculate_activation_statistics,calculate_frechet_distance
    totals={};globalmax=0.;fkmax=0.;fkcount=0
    for method in ['mdm50','mld','motionlcm_v1','t2mgpt','t2m']:
        checked=0;anglemax=0.;native_max=0.;metricmax=0.;rho_min=100.;sigma_min=100.
        for rep in range(20):
            folder=R/method/f'rep_{rep:02d}';row=read(folder/'metrics.json');X=np.load(folder/'generated_raw.npy',mmap_mode='r');L=np.load(folder/'generated_lengths.npy');N=len(X);T=np.load(folder/'text_embeddings.npy');ref=np.load(folder/'reference_embeddings.npy')
            for condition in ['raw','canonical']:
                E=np.load(folder/f'{condition}_motion_embeddings.npy');mu,cov=calculate_activation_statistics(E);mr,cr=calculate_activation_statistics(ref);fid=max(float(calculate_frechet_distance(mu,cov,mr,cr)),0.);error=abs(fid-row['results'][condition]['FID']);assert error<1e-8;metricmax=max(metricmax,error)
                counts=sum(calculate_R_precision(T[i:i+32],E[i:i+32],3,True) for i in range(0,N,32))/N;assert np.max(abs(counts-row['results'][condition]['R_precision']))<1e-12
            for i in range(0,N,32):
                x=np.array(X[i:i+32]);y=canonical(x);mask=np.arange(196)[None,:]<L[i:i+32,None];assert np.array_equal(x[...,:67],y[...,:67]) and np.array_equal(x[...,193:],y[...,193:])
                d=x[...,67:193].reshape(len(x),196,21,6)[mask];e=y[...,67:193].reshape(len(x),196,21,6)[mask];r0,rho,sigma=matrices(d);r1,_,_=matrices(e);rho_min=min(rho_min,rho);sigma_min=min(sigma_min,sigma)
                delta=r0.swapaxes(-1,-2)@r1;skew=np.stack([delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],delta[...,1,0]-delta[...,0,1]],-1)/2.;angle=np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1),(np.trace(delta,axis1=-2,axis2=-1)-1)/2.));anglemax=max(anglemax,float(angle.max()));assert angle.max()<.01
                native=float(abs(cont6d_to_matrix(torch.tensor(d))-cont6d_to_matrix(torch.tensor(e))).max());native_max=max(native_max,native);assert native<2e-5;checked+=len(x)
                if i==0:
                    for j in range(2):
                        n=int(L[j]);a=torch.tensor(x[j,:n]);b=torch.tensor(y[j,:n]);pa=recover_from_ric(a,22);pb=recover_from_ric(b,22);assert torch.equal(pa,pb);skel=Skeleton(torch.tensor(t2m_raw_offsets),t2m_kinematic_chain,'cpu');skel.get_offsets_joints(pa[0]);fa=recover_from_rot(a,22,skel);fb=recover_from_rot(b,22,skel);er=float((fa-fb).abs().max());assert er<1e-4;fkmax=max(fkmax,er);fkcount+=1
            print('VERIFIED_RELEASE',method,rep,flush=True)
        totals[method]={'clips':checked,'max_rotation_error_deg':anglemax,'max_native_converter_error':native_max,'saved_metric_error':metricmax,'minimum_valid_a_norm':rho_min,'minimum_valid_b_orthogonal_norm':sigma_min};globalmax=max(globalmax,anglemax)
    write(R/'verification.json',{'passed':True,'all_five_releases20repetitions':True,'methods':totals,'native_FK_pairs':fkcount,'max_native_FK_position_error':fkmax,'max_rotation_error_deg':globalmax,'protected_channels_bitwise_equal':True,'native_saved_embedding_metrics_recomputed':True,'geometry_check_code_sha256':sha(__file__),'not_a_human_quality_or_intentional_exploitation_test':True});print('ALL_RELEASES_VERIFIED',flush=True)
if __name__=='__main__':main()
