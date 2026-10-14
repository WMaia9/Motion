"""Independently reconstruct fixed directions and recompute all saved scores."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import t
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from motion_release_audit_common import read,write,sha,seed,evaluator,score_embeddings,metric
from score_humanml_blind_fid import matrices
from tmr_audit_common import NativeTMR
from verify_tmr1024_controls import score as cosine_score

R=Path('reports/cvpr_coherent_blocks_20261014')
OLD=Path('reports/cvpr_matched_blocks_20261013')
INPUT=Path('reports/cvpr_motion_prevalence_20261012')
BLOCKS={'velocity':(193,259),'contact':(259,263),'position':(4,67)}


def wait(path):
    while not path.exists():time.sleep(10)


def reconstruct(x,L,condition,scalar):
    block,_,direction=condition.split('_');sign=1 if direction=='plus' else -1
    axis=sign*np.ones(3)/np.sqrt(3.)
    y=x.astype(float).copy()
    if block=='rotation':
        d=x[...,67:193].reshape(*x.shape[:2],21,6)
        rot=matrices(d).reshape(*x.shape[:2],21,3,3)
        for i in range(len(x)):
            Q=Rotation.from_rotvec(scalar[i]*axis).as_matrix()
            U=rot[i]@Q
            y[i,:,67:193]=np.concatenate([U[...,0],U[...,1]],axis=-1).reshape(x.shape[1],126)
        begin,end=67,193
    else:
        begin,end=BLOCKS[block]
        v=sign*np.ones(4) if block=='contact' else np.tile(axis,(end-begin)//3)
        y[...,begin:end]+=scalar[:,None,None]*v
    y=y.astype(x.dtype);y[...,:begin]=x[...,:begin];y[...,end:]=x[...,end:]
    mask=np.arange(x.shape[1])[None,:]<L[:,None];y[~mask]=x[~mask]
    return y


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reader',choices=['guo','tmr'],required=True);args=p.parse_args()
    assert not (R/'manifest_sha256.json').exists()
    protocol=read(R/'protocol.json')
    for file,h in protocol['producer_sha256'].items():assert sha(file)==h,file
    for folder,h in protocol['source_manifests'].items():assert sha(Path(folder)/'manifest_sha256.json')==h
    seed(444)
    if args.reader=='guo':model,mean,std=evaluator()
    else:model=NativeTMR();std=model.std.cpu().numpy()+1e-12
    root=R/args.reader;wait(root/'normalizer.json');np.testing.assert_array_equal(std,read(root/'normalizer.json')['std'])
    cells=fixtures=0;max_norm=max_forward=0.;rows=[];old_rows=[];max_matrix_change=0.
    with threadpool_limits(limits=2):
        for rep in range(20):
            folder=root/f'rep_{rep:02d}';wait(folder/'metrics.json');row=read(folder/'metrics.json');rows.append(row)
            prev=OLD/args.reader/f'rep_{rep:02d}';old_rows.append(read(prev/'metrics.json'))
            T=np.load(prev/'text_embeddings.npy');E0=np.load(prev/'reference_embeddings.npy')
            src=INPUT/'real_controls'/f'rep_{rep:02d}'
            X=np.load(src/'source_raw.npy',mmap_mode='r');z=np.load(src/'conditioning.npz');c={k:z[k].copy() for k in z.files};L=c['lengths']
            x=np.array(X[:8]);ll=L[:8];mask=np.arange(x.shape[1])[None,:]<ll[:,None]
            for label,reported in row['scores'].items():
                E=E0 if label=='source' else np.load(folder/(label+'_embeddings.npy'))
                if args.reader=='guo':
                    v=metric(T,E,E0)
                    assert abs(v['FID']-reported['FID'])<2e-5
                    assert abs(v['MM_distance']-reported['MM_distance'])<1e-6
                    np.testing.assert_array_equal(v['R_precision'],reported['R_precision'])
                else:
                    ranks,cosine=cosine_score(T,E)
                    np.testing.assert_array_equal(ranks,reported['R_precision']);assert abs(cosine-reported['matched_cosine'])<1e-7
                cells+=1
                if label=='source':continue
                condition,radius=label.rsplit('_r',1);radius=float(radius);block=condition.split('_')[0]
                params=np.load(folder/(label+'_parameters.npz'));np.testing.assert_array_equal(params['lengths'],L)
                error=float(np.max(abs(params['realized_normalized_L2']-radius*np.sqrt(L))/(radius*np.sqrt(L))))
                tol=protocol['float32_relative_budget_tolerance']['contact' if block=='contact' else 'other_blocks'];assert error<tol
                max_norm=max(max_norm,error)
                Y=np.load(folder/(label+'_sample.npy'));expected=reconstruct(x,ll,condition,params['scalar'][:8])
                assert abs(Y-expected).max()<2e-6
                n=np.linalg.norm((((Y.astype(float)-x)/std)*mask[...,None]).reshape(8,-1),axis=1)
                np.testing.assert_allclose(n,params['realized_normalized_L2'][:8],atol=2e-7,rtol=2e-7)
                begin,end=BLOCKS.get(block,(67,193))
                for a,b in [(Y[...,:begin],x[...,:begin]),(Y[...,end:],x[...,end:]),(Y[~mask],x[~mask])]:assert np.array_equal(a.view('u4'),b.view('u4'))
                if block=='rotation':
                    a=x[...,67:193].reshape(8,196,21,6);b=Y[...,67:193].reshape(8,196,21,6)
                    delta=float(abs(matrices(a[mask])-matrices(b[mask])).max());assert delta>1e-6
                    max_matrix_change=max(max_matrix_change,delta)
                if block=='contact' and args.reader=='guo':np.testing.assert_array_equal(E,E0)
                if args.reader=='tmr':inferred=model.motions(Y,ll);saved=E[:8]
                else:
                    cc={k:v[:8] for k,v in c.items()};_,inferred=score_embeddings(model,Y,cc,mean,std)
                    ix32=np.argsort(-c['sent_lengths'][:32],kind='stable');order32=ix32[np.argsort(L[:32][ix32])[::-1]]
                    ix8=np.argsort(-c['sent_lengths'][:8],kind='stable');order8=ix8[np.argsort(ll[ix8])[::-1]]
                    saved=E[np.argsort(order32)[order8]]
                error=float(abs(inferred-saved).max());assert error<3e-5;max_forward=max(max_forward,error);fixtures+=8
            print('COHERENT_DRAW_VERIFIED',args.reader,rep,flush=True)
    wait(root/'complete.json');analysis=read(root/'analysis.json')
    for label,records in analysis['conditions'].items():
        for endpoint,record in records.items():
            def value(v):return v['R_precision'][0] if endpoint=='R1' else v[endpoint]
            a=np.array([value(row['scores'][label]) for row in rows]);b=np.array([value(row['scores']['source']) for row in rows]);d=a-b
            previous_label=analysis['fixed_minus_random'][label][endpoint]['random_condition']
            random=np.array([value(row['scores'][previous_label]) for row in old_rows])
            for delta,target in [(d,record),(a-random,analysis['fixed_minus_random'][label][endpoint])]:
                half=t.ppf(1-.05/(2*32),19)*delta.std(ddof=1)/np.sqrt(20)
                np.testing.assert_allclose(target['family32_CI95'],[delta.mean()-half,delta.mean()+half],atol=1e-12,rtol=0)
                assert abs(target['mean_change']-delta.mean())<1e-12
    write(root/'verification.json',{'passed':True,'saved_metric_cells_recomputed':cells,'all_norm_arrays_checked':True,'first8_fixture_pairs_independently_reconstructed':fixtures,'max_relative_L2_error':max_norm,'native_forward_max_difference':max_forward,'changed_rotation_matrix_max_difference':max_matrix_change,'all_primary_and_fixed_minus_random_intervals_recomputed':True,'verifier_sha256':sha(__file__),'full_unsaved_geometry_replay_claimed':False})
    print('COHERENT_READER_VERIFIED',args.reader,flush=True)


if __name__=='__main__':main()
