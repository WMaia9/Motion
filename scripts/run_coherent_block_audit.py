"""Registered temporal-direction controls on twenty existing real cohorts."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits
from scripts.coherent_motion_blocks import intervene, CONDITIONS, AXIS
from motion_release_audit_common import R as INPUT, read,write,sha,seed,evaluator,score_embeddings,metric
from tmr_audit_common import NativeTMR,retrieval

R=Path('reports/cvpr_coherent_blocks_20261014')
PREVIOUS=Path('reports/cvpr_matched_blocks_20261013')
RADII=[1.,3.]
FAMILY=32


def registration():
    return {'cohorts':20,'clips_each':4640,'radii':RADII,'conditions':CONDITIONS,
        'axis':AXIS.tolist(),'both_signs_retained':True,
        'direction':'same unnormalized local unit axis for every joint and valid frame; contacts use all-ones vector',
        'rotation':'right-multiply each local rotation by same per-clip SO3 change; signed axis fixed over joints/time; angle in [0,pi/2] solved from input norm',
        'matching':'per clip ||(y-x)/native_reader_std||_2 = radius*sqrt(valid_frames) on all263 channels before selection',
        'float32_relative_budget_tolerance':{'contact':2e-5,'other_blocks':5e-6},
        'scalar':'input geometry only, no evaluator gradients, captions, score selection or optimization',
        'coherence_scope':'temporal direction only; held-fixed blocks can be physically inconsistent; not full coherent motion editing',
        'paired_references':'identical source cohorts and original reader embeddings from previous frozen block audit',
        'Guo_contacts':'excluded by architecture; consumed budget zero',
        'contact_domain':'continuous changes may leave binary domain',
        'primary_family':'32 endpoints per reader:8conditions x2radii x2scores; Bonferroni paired t19 CI',
        'secondary_family':'32 fixed-minus-random changes per reader against previous corresponding rotation/velocity/contact/position controls, separate family32 paired t19 CI',
        'old_null_results':'unchanged prior registered 28-endpoint family; no pooled family across historical and prospective phases',
        'no_new_generators_training_model_seeds_or_reward_fits':True,
        'source_manifests':{str(p):sha(p/'manifest_sha256.json') for p in [INPUT,PREVIOUS]},
        'producer_sha256':{str(p):sha(p) for p in [Path(__file__),Path('scripts/coherent_motion_blocks.py'),Path('scripts/matched_motion_blocks.py'),Path('scripts/motion_release_audit_common.py'),Path('scripts/tmr_audit_common.py')]}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader',choices=['guo','tmr'],required=True);args=parser.parse_args()
    assert not (R/'manifest_sha256.json').exists()
    p=R/'protocol.json'
    assert p.exists(), 'Register the full protocol before running either reader'
    assert read(p)==registration()
    seed(4301);root=R/args.reader;root.mkdir(exist_ok=True)
    if args.reader=='guo':model,mean,std=evaluator()
    else:model=NativeTMR();std=model.std.cpu().numpy()+1e-12
    np.testing.assert_array_equal(std,read(PREVIOUS/args.reader/'normalizer.json')['std'])
    write(root/'normalizer.json',read(PREVIOUS/args.reader/'normalizer.json'))
    with threadpool_limits(limits=2):
        for rep in range(20):
            folder=root/f'rep_{rep:02d}';folder.mkdir(exist_ok=True)
            if (folder/'metrics.json').exists():continue
            source=INPUT/'real_controls'/f'rep_{rep:02d}'
            X=np.load(source/'source_raw.npy',mmap_mode='r');z=np.load(source/'conditioning.npz');cohort={k:z[k].copy() for k in z.files};L=cohort['lengths']
            prev=PREVIOUS/args.reader/f'rep_{rep:02d}'
            T=np.load(prev/'text_embeddings.npy');E0=np.load(prev/'reference_embeddings.npy');old=read(prev/'metrics.json')
            scores={'source':old['scores']['source']};checks={}
            for radius in RADII:
                for condition in CONDITIONS:
                    label=f'{condition}_r{radius:g}';emb=[];norms=[];scalars=[]
                    for i in range(0,len(X),64):
                        x=np.array(X[i:i+64]);ll=L[i:i+64]
                        y,s,n=intervene(x,ll,std,condition,radius)
                        if args.reader=='guo':
                            c={k:v[i:i+64] for k,v in cohort.items()};tt,ee=score_embeddings(model,y,c,mean,std)
                            np.testing.assert_array_equal(tt,T[i:i+64])
                        else:ee=model.motions(y,ll)
                        emb.append(ee);norms.append(n);scalars.append(s)
                        if i==0:np.save(folder/(label+'_sample.npy'),y[:8])
                    E=np.concatenate(emb);N=np.concatenate(norms);S=np.concatenate(scalars)
                    np.save(folder/(label+'_embeddings.npy'),E)
                    np.savez_compressed(folder/(label+'_parameters.npz'),scalar=S,realized_normalized_L2=N,lengths=L)
                    scores[label]=metric(T,E,E0) if args.reader=='guo' else retrieval(T,E)
                    checks[label]={'max_relative_L2_mismatch':float(np.max(abs(N-radius*np.sqrt(L))/(radius*np.sqrt(L))))}
                    print('COHERENT_BLOCK_SCORED',args.reader,rep,label,scores[label],flush=True)
            write(folder/'metrics.json',{'rep':rep,'reader':args.reader,'scores':scores,'checks':checks})
    rows=[read(root/f'rep_{i:02d}/metrics.json') for i in range(20)]
    previous=[read(PREVIOUS/args.reader/f'rep_{i:02d}/metrics.json') for i in range(20)]
    result={};contrasts={}
    for radius in RADII:
        for condition in CONDITIONS:
            label=f'{condition}_r{radius:g}';result[label]={};contrasts[label]={}
            random_label=f'{condition.split("_")[0]}_r{radius:g}'
            for endpoint in ['FID','R1'] if args.reader=='guo' else ['R1','matched_cosine']:
                def value(v):return v['R_precision'][0] if endpoint=='R1' else v[endpoint]
                a=np.array([value(row['scores'][label]) for row in rows]);b=np.array([value(row['scores']['source']) for row in rows]);d=a-b
                e=np.array([value(row['scores'][random_label]) for row in previous]);c=a-e
                def summary(delta):
                    half=t.ppf(1-.05/(2*FAMILY),19)*delta.std(ddof=1)/np.sqrt(20)
                    return {'mean_change':float(delta.mean()),'family32_CI95':[float(delta.mean()-half),float(delta.mean()+half)],'all20_draws':True}
                result[label][endpoint]={'source_mean':float(b.mean()),'perturbed_mean':float(a.mean()),**summary(d)}
                contrasts[label][endpoint]={'random_condition':random_label,**summary(c)}
    write(root/'analysis.json',{'reader':args.reader,'conditions':result,'fixed_minus_random':contrasts,'twenty_draws_complete':True,'L2_before_reader_selection':True,'coherence_means_temporally_fixed_feature_direction':True})
    write(root/'complete.json',{'all20_draws_and_all16_conditions':True})
    print('COHERENT_BLOCK_READER_COMPLETE',args.reader,flush=True)


if __name__=='__main__':main()
