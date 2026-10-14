"""Descriptive family mechanism check using frozen scores and natural 6D RMS.

Eight release settings are not independent training replicas of model families.
Canonicalization tests only the within-rotation encoding fiber; it does not
remove velocity/contact errors or test all of MARDM's proposed mechanisms.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import itertools
import numpy as np
from scipy.stats import rankdata
from motion_release_audit_common import R as OLD,read,write,sha
from tools.canonicalize_motion import canonicalize_motion
ADD=Path('reports/cvpr_additional_motion_audit_20261013');MD=Path('reports/cvpr_motiondiffuse_compiled_20261013');R=Path('reports/cvpr_rotation_mechanism_20261013')
METHODS=['mdm50','mld','motionlcm_v1','motiondiffuse','t2mgpt','momask','t2m','dip']
FAMILY={'mdm50':'diffusion','mld':'diffusion','motionlcm_v1':'diffusion','motiondiffuse':'diffusion','t2mgpt':'VQ','momask':'VQ','t2m':'other_VAE','dip':'prefix_conditioned_diffusion'}


def main():
    assert not (R/'manifest_sha256.json').exists();R.mkdir(exist_ok=True)
    old=read(OLD/'analysis.json');extra=read('reports/cvpr_generalization_review_20261013/additional_methods_analysis.json')['additional_methods']
    p={'methods':METHODS,'family_map':FAMILY,'RMS':'sqrt(mean squared raw-minus-canonical over all valid126 rotation values)), not global263 or padding; aggregate each original20draw then method mean','scores':'existing signed primaryGuoFID raw-minus-canonical, no new evaluation or source selection','primary_family_contrast':'descriptive mean of four unconditioned diffusion/consistency releases vs two VQ releases; T2M and prefixDiP shown separately','correlations':'release-level n8 Spearman RMS vs signed gap and absolute gap, exact two-sided permutation with two-endpoint Bonferroni; model family confounding and correlated release settings prevent causal/population inference','limitations':['rotation fiber only, not velocity/contact or whole representation errors','latent diffusion/consistency and raw-noise/x0 models have different mechanisms','one frozen set of weights per method, no family-level causal claim','shared public TEST development history','raw FID numerical scales tied to chosen reader'],'source_sha256':{str(q):sha(q) for q in [Path(__file__),Path('tools/canonicalize_motion.py'),OLD/'manifest_sha256.json',ADD/'manifest_sha256.json',MD/'manifest_sha256.json']}}
    if (R/'protocol.json').exists():assert read(R/'protocol.json')==p
    else:write(R/'protocol.json',p)
    out={}
    for name in METHODS:
        records=[]
        if name in old['methods']:
            f=old['methods'][name]['FID']
            for rep in range(20):
                row=read(OLD/name/f'rep_{rep:02d}/metrics.json');records.append(row['geometry']['canonical']['raw_6D_RMS_change'])
            score={'raw':f['raw_mean'],'canonical':f['canonical_mean'],'delta':f['paired_raw_minus_canonical'],'CI95':f['paired_family_CI95']};rms_source='verified original per-repetition geometry fields'
        else:
            key='humanml/'+name;f=extra[key]['FID'];score={'raw':f['raw'],'canonical':f['canonical'],'delta':f['delta'],'CI95':f['family10_CI95']};root=MD/'humanml/motiondiffuse' if name=='motiondiffuse' else ADD/'humanml'/name
            for rep in range(20):
                folder=root/f'rep_{rep:02d}';X=np.load(folder/'generated_raw.npy',mmap_mode='r');L=np.load(folder/'conditioning.npz')['lengths'];sum2=0.;count=0
                for i in range(0,len(X),32):
                    x=np.array(X[i:i+32]);ll=L[i:i+32];c=canonicalize_motion(x,ll);mask=np.arange(x.shape[1])[None,:]<ll[:,None];d=x[...,67:193][mask].astype(float)-c[...,67:193][mask];sum2+=float(np.sum(d*d));count+=d.size
                records.append(float(np.sqrt(sum2/count)))
                print('ROTATION_RMS_COMPUTED',name,rep,records[-1],flush=True)
            rms_source='recomputed from every saved raw output with portable projection and valid-frame mask'
        out[name]={'family':FAMILY[name],'FID':score,'rotation_RMS_per_draw':records,'rotation_RMS_mean':float(np.mean(records)),'RMS_source':rms_source}
    group={f:{'methods':[n for n in METHODS if FAMILY[n]==f],'mean_signed_gap':float(np.mean([out[n]['FID']['delta'] for n in METHODS if FAMILY[n]==f]))} for f in ['diffusion','VQ','other_VAE','prefix_conditioned_diffusion']}
    x=np.array([out[n]['rotation_RMS_mean'] for n in METHODS]);x=rankdata(x)-4.5;perms=np.array(list(itertools.permutations(range(8))));corr={}
    for label,values in [('signed',np.array([out[n]['FID']['delta'] for n in METHODS])),('absolute',np.array([abs(out[n]['FID']['delta']) for n in METHODS]))]:
        y=rankdata(values)-4.5;rho=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)));null=(y[perms]@x)/(np.linalg.norm(x)*np.linalg.norm(y));pval=float(np.mean(np.abs(null)>=abs(rho)-1e-12));corr[label]={'Spearman_rho':rho,'n_releases':8,'exact_permutation_p':pval,'two_endpoint_adjusted_p':min(1.,2*pval),'exploratory_not_causal':True}
    write(R/'analysis.json',{'methods':out,'family_descriptive_means':group,'VQ_minus_diffusion_descriptive_gap':group['VQ']['mean_signed_gap']-group['diffusion']['mean_signed_gap'],'correlations':corr,'does_not_refute_full_MARDM_mechanism':True,'no_uniform_rotation_noncanonicality_penalty_for_diffusion_observed':True,'original_five_mean_ranking_unchanged':old['FID_rank_order_changed'] is False});write(R/'complete.json',{'all8release20draw_RMS_sources':True});print('ROTATION_MECHANISM_COMPLETE',group,corr,flush=True)

if __name__=='__main__':main()
