"""Prospective equal-L2 block audit on all twenty frozen real-motion cohorts."""
import argparse,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from matched_motion_blocks import intervene,basis,CONDITIONS,BLOCKS
from motion_release_audit_common import R as INPUT,read,write,sha,seed,evaluator,score_embeddings,metric,canonical
from tmr_audit_common import NativeTMR,retrieval,MODEL,SOURCE
R=Path('reports/cvpr_matched_blocks_20261013')
RADII=[1.,3.]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reader',choices=['guo','tmr'],required=True);args=parser.parse_args()
    assert not (R/'manifest_sha256.json').exists();R.mkdir(exist_ok=True)
    registration={'cohorts':20,'clips_each':4640,'sources':'all saved original real_controls cohorts, unchanged captions/crops/masks','blocks':{'null':'positive scale/shear only, rotation equivalence fiber','rotation':'canonical 6D of actual local SO(3) rotation changes, other blocks fixed','velocity':'193:259 (66)','contact':'259:263 (4)','position':'4:67 (63), root four channels held fixed'},'conditions':CONDITIONS,'radii':RADII,'matching':'per clip ||(y-x)/native_reader_std||_2 = radius*sqrt(valid_frames), before reader feature selection; no per-dimension noise amplitude matching','reader_specific_normalization':True,'Guo_discards_contacts':'Contact budget is matched in full pre-selection263 normalized representation; consumed Guo budget is zero by native architecture','contacts':'continuous input ablation; values may leave binary/physical domain','non_null_controls':'block-only interventions can be inconsistent with unchanged remaining features; this is input sensitivity, not coherent physical-motion calibration','random_fields':'20261013+10000*rep+clipindex, shared across radii/readers; independent Gaussian fields; fixed null signs both retained','null_matching':'positive exponential scales and shear; root solved from norm, never from evaluator scores','outside_rotation_matching':'local isotropic axis and angle solved from norm; angle <=pi/2','primary_family':'28 endpoints per reader:14 conditions x (GuoFID,R1 or TMRR1,matchedCosine), Bonferroni t19 mean intervals','closure_diagnostics':'all null outputs canonicalized and encoded, separate numerical diagnostics, no additional significance claims','FID_reference':'paired original real cohort; zero baseline by construction, not real-world generation quality','not_official_full_TMR_annotation_protocol':True,'no_new_generators_training_or_reward_fits':True,'producer_sha256':{str(p):sha(p) for p in [Path(__file__),Path('scripts/matched_motion_blocks.py'),Path('scripts/tmr_audit_common.py'),Path('scripts/motion_release_audit_common.py')]},'source_manifest_sha256':sha(INPUT/'manifest_sha256.json')}
    p=R/'protocol.json'
    if p.exists():assert read(p)==registration
    else:write(p,registration)
    seed(4301);root=R/args.reader;root.mkdir(exist_ok=True)
    if args.reader=='guo':model,mean,std=evaluator()
    else:model=NativeTMR();mean=model.mean.cpu().numpy();std=model.std.cpu().numpy()+1e-12
    write(root/'normalizer.json',{'reader':args.reader,'mean':np.asarray(mean).tolist(),'std':np.asarray(std).tolist(),'source':'native fixed checkpoint normalization','no_metric_selected_budget':True})
    for rep in range(20):
        folder=root/f'rep_{rep:02d}';folder.mkdir(exist_ok=True)
        if (folder/'metrics.json').exists():continue
        source=INPUT/'real_controls'/f'rep_{rep:02d}';X=np.load(source/'source_raw.npy',mmap_mode='r');z=np.load(source/'conditioning.npz');cohort={k:z[k].copy() for k in z.files};L=cohort['lengths']
        def encode(y,c):
            if args.reader=='guo':return score_embeddings(model,y,c,mean,std)
            return model.texts(c['captions'].tolist()),model.motions(y,c['lengths'])
        T0,E0=encode(X,cohort);_,EC0=encode(canonical(X),cohort);np.save(folder/'text_embeddings.npy',T0);np.save(folder/'reference_embeddings.npy',E0);np.save(folder/'canonical_source_embeddings.npy',EC0)
        def scores(T,E):return metric(T,E,E0) if args.reader=='guo' else retrieval(T,E)
        result={'source':scores(T0,E0),'canonical_source':scores(T0,EC0)};checks={}
        for radius in RADII:
            for condition in CONDITIONS:
                label=f'{condition}_r{radius:g}';texts=[];embeddings=[];closed=[];norms=[];scalars=[];maxbasis=0.;samples=[]
                begun=time.monotonic()
                for i in range(0,len(X),64):
                    x=np.array(X[i:i+64]);ll=L[i:i+64];c={k:v[i:i+64] for k,v in cohort.items()};y,params,n=intervene(x,ll,std,condition,radius,rep,i);tt,ee=encode(y,c);texts.append(tt);embeddings.append(ee);norms.append(n);scalars.append(params['scalar'])
                    if condition.startswith('null'):
                        _,ce=encode(canonical(y),c);closed.append(ce);mask=np.arange(196)[None,:]<ll[:,None];err=float(abs(basis(y)[mask]-basis(x)[mask]).max());assert err<2e-6;maxbasis=max(maxbasis,err)
                    if i==0:samples.append(y[:8]);np.save(folder/(label+'_sample.npy'),y[:8])
                T=np.concatenate(texts);E=np.concatenate(embeddings);N=np.concatenate(norms);S=np.concatenate(scalars);result[label]=scores(T,E);np.save(folder/(label+'_embeddings.npy'),E);np.savez_compressed(folder/(label+'_parameters.npz'),scalar=S,realized_normalized_L2=N,lengths=L)
                check={'max_relative_L2_mismatch':float(np.max(abs(N-radius*np.sqrt(L))/(radius*np.sqrt(L)))),'block':BLOCKS.get(condition,(67,193)),'samples':'first8source records, not chosen by scores','max_null_basis_difference':maxbasis,'null_other_features_preserved':condition.startswith('null'),'elapsed':time.monotonic()-begun}
                if closed:
                    CE=np.concatenate(closed);np.save(folder/(label+'_canonical_embeddings.npy'),CE);check['canonical_embedding_max_change']=float(abs(CE-EC0).max());check['canonical_scores']=scores(T,CE)
                checks[label]=check;print('MATCHED_BLOCK_SCORED',args.reader,rep,label,result[label],check['max_relative_L2_mismatch'],flush=True)
        write(folder/'metrics.json',{'rep':rep,'reader':args.reader,'scores':result,'checks':checks,'source_input_sha256_from_frozen_inventory':read(INPUT/'manifest_sha256.json')[str((source/'source_raw.npy').relative_to(INPUT))]})
    out={};rows=[read(root/f'rep_{i:02d}/metrics.json') for i in range(20)]
    for radius in RADII:
        for condition in CONDITIONS:
            label=f'{condition}_r{radius:g}';out[label]={}
            for name in (['FID','R1'] if args.reader=='guo' else ['R1','matched_cosine']):
                def value(x):return x['R_precision'][0] if name=='R1' else x[name]
                a=np.array([value(row['scores'][label]) for row in rows]);b=np.array([value(row['scores']['source']) for row in rows]);delta=a-b;half=t.ppf(1-.05/(2*28),19)*delta.std(ddof=1)/np.sqrt(20)
                out[label][name]={'source_mean':float(b.mean()),'perturbed_mean':float(a.mean()),'mean_change':float(delta.mean()),'family28_CI95':[float(delta.mean()-half),float(delta.mean()+half)],'all20_draws':True}
    write(root/'analysis.json',{'reader':args.reader,'conditions':out,'twenty_draws_complete':True,'paired_source_baseline_zero_by_construction':True,'L2_before_reader_feature_selection':True});write(root/'complete.json',{'all20_draws_and_all14_conditions':True});print('MATCHED_BLOCK_READER_COMPLETE',args.reader,flush=True)

if __name__=='__main__':main()
