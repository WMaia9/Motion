"""Independent all-clip reconstruction, geometry, strings and native fixture replay."""
import csv
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from motion_release_audit_common import read,write,sha
from run_signspark_geometry_dev import project,matrices,make_back_translation_model,back_translate

R=Path('reports/cvpr_sign_stability_20261014')


def tokenize(s,corpus):return tuple(s.replace(' ','')) if corpus=='CSL-Daily' else tuple(s.split())


def distance(a,b):
    # Full matrix rather than the analyst's one-row recursion.
    D=np.empty((len(a)+1,len(b)+1),dtype=np.int32);D[:,0]=np.arange(len(a)+1);D[0]=np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):D[i,j]=min(D[i-1,j]+1,D[i,j-1]+1,D[i-1,j-1]+int(a[i-1]!=b[j-1]))
    return int(D[-1,-1])


def reconstruction(x,condition,index,corpus_index,parameter):
    if condition=='raw':return x.copy()
    if condition=='canonical':return project(x)
    if condition.startswith('gaussian'):
        rng=np.random.default_rng(int(condition.rsplit('_',1)[1])+corpus_index*1000000+index)
        noise=rng.normal(size=(len(x),40,6)).reshape(x.shape)
        if condition.startswith('gaussian_support'):noise[:,60:150]=0.
        y=(x.astype(float)+parameter['scalar']*noise).astype('float32')
    else:
        axes=np.random.default_rng(20261014+corpus_index*1000000+index).normal(size=(40,3));axes/=np.linalg.norm(axes,axis=1)[:,None]
        Q=Rotation.from_rotvec(parameter['scalar']*axes).as_matrix()
        y=(x.reshape(len(x),40,2,3).astype(float)@Q[None]).reshape(x.shape).astype('float32')
    if condition.startswith('physical') or condition.startswith('gaussian_support'):y[:,60:150]=x[:,60:150]
    return y


def main():
    assert not (R/'manifest_sha256.json').exists();p=read(R/'protocol.json');a=read(R/'analysis.json')
    for key in ['inputs_sha256','scientific_sources_sha256']:
        for f,h in p[key].items():assert sha(f)==h,(f,'registered input/source changed')
    for r,h in p['source_manifest_sha256'].items():assert sha(Path(r)/'manifest_sha256.json')==h
    checked=0;hypothesis_pairs=0;geometry={};intervals=0;forward=0
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with threadpool_limits(limits=2):
        for corpus_index,corpus in enumerate(p['corpora']):
            root=R/corpus;meta=read(root/'inputs.json');N=len(meta['clip_ids']);assert sha(root/'source_raw.npz')==meta['source_raw_sha256']
            with np.load(root/'source_raw.npz') as z:X=[z[f'clip{i:04d}'].copy() for i in range(N)]
            raw=read(root/'raw.json')['hypotheses'];measurements={};max_norm=0.;max_canonical_angle=0.;max_gram=0.;max_physical_angle_error=0.
            expected_checkpoint=Path('reports/cvpr_signspark_baseline_20261010/checkpoints/How2Sign') if corpus=='How2Sign' else Path('reports/cvpr_representation_paper_20261011/checkpoints/CSL-Daily')
            model=make_back_translation_model(expected_checkpoint);model.do_recognition=False
            for parameter in model.parameters():parameter.requires_grad_(False)
            for condition in p['conditions']:
                record=read(root/(condition+'.json'));stored=np.load(root/(condition+'_fixtures.npz'));values=[]
                assert len(record['hypotheses'])==N and all(record['feasible'])
                for i,x in enumerate(X):
                    y=reconstruction(x,condition,i,corpus_index,record['parameters'][i]);budget=float(np.linalg.norm(project(x).astype(float)-x))
                    assert abs(budget-record['parameters'][i]['budget'])<1e-10
                    actual=float(np.linalg.norm(y.astype(float)-x))
                    if condition not in ['raw','physical_1deg']:assert abs(actual-budget)<=5e-6*max(budget,1.)
                    if condition not in ['raw','canonical']:assert abs(actual-record['parameters'][i]['realized_L2'])<1e-8
                    if condition not in ['raw','physical_1deg']:max_norm=max(max_norm,abs(actual-budget)/max(budget,1.))
                    if condition.startswith('physical') or condition.startswith('gaussian_support'):assert np.array_equal(y[:,60:150].view('u4'),x[:,60:150].view('u4'))
                    key=f'clip{i:04d}'
                    if key in stored:assert abs(y-stored[key]).max()<2e-7
                    if condition=='canonical':
                        Q=matrices(x)@matrices(y).transpose(0,1,3,2)
                        angle=float(np.degrees(Rotation.from_matrix(Q.reshape(-1,3,3)).magnitude()).max());assert angle<2e-4;max_canonical_angle=max(max_canonical_angle,angle)
                    if condition.startswith('physical'):
                        original=x.reshape(-1,40,2,3).astype(float);changed=y.reshape(-1,40,2,3).astype(float)
                        error=float(abs(original@original.transpose(0,1,3,2)-changed@changed.transpose(0,1,3,2)).max());assert error<3e-6;max_gram=max(max_gram,error)
                        Q=matrices(x)@matrices(y).transpose(0,1,3,2);theta=Rotation.from_matrix(Q.reshape(-1,3,3)).magnitude().reshape(len(x),40)
                        active=np.r_[np.arange(10),np.arange(25,40)];error=float(abs(theta[:,active]-record['parameters'][i]['scalar']).max());assert error<3e-6;max_physical_angle_error=max(max_physical_angle_error,error)
                    s,t=raw[i],record['hypotheses'][i];u,v=tokenize(s,corpus),tokenize(t,corpus);d=distance(u,v)
                    values.append([float(s!=t),float(u!=v),float(d),d/max(len(u),len(v),1)])
                    checked+=1;hypothesis_pairs+=1
                stored.close();measurements[condition]=np.array(values)
                # Reproduce exact original batch boundaries and padding, without
                # passing any references/text to the decoder.
                for start in [0,((N-1)//32)*32]:
                    indices=range(start,min(start+32,N));ys=[torch.from_numpy(reconstruction(X[i],condition,i,corpus_index,record['parameters'][i])) for i in indices]
                    decoded=back_translate(model,ys,batch_size=32)
                    assert decoded==record['hypotheses'][start:start+len(ys)],(corpus,condition,start,'native fixture decode differs')
                    forward+=len(ys)
                expected=a['results'][corpus]['individual_conditions'][condition]
                for name,j in [('string_changed_fraction',0),('token_changed_fraction',1),('mean_token_edit_distance',2),('mean_normalized_edit_distance',3)]:assert abs(expected[name]-measurements[condition][:,j].mean())<1e-12
                print('SIGN_STABILITY_CONDITION_VERIFIED',corpus,condition,flush=True)
            del model;torch.cuda.empty_cache()
            measurements['gaussian_support_mean']=np.mean([measurements[f'gaussian_support_{s}'] for s in p['Gaussian_seeds']],axis=0)
            saved=np.load(root/'per_clip_measurements.npz')
            for name,values in measurements.items():np.testing.assert_array_equal(saved[name],values)
            saved.close()
            if corpus=='How2Sign':
                with Path('reports/cvpr_evaluation_confirmation_20261011/sources/how2sign_test.csv').open() as f:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(f,delimiter='\t')}
                names=[lookup[key] for key in meta['clip_ids']]
            else:names=meta['clip_ids']
            unique=sorted(set(names));ix=np.array([unique.index(n) for n in names]);counts=np.bincount(ix);G=len(unique);assert G==a['results'][corpus]['bootstrap_groups']
            W=np.random.default_rng(42026+corpus_index).multinomial(G,np.ones(G)/G,size=20000);denominator=W@counts;B={}
            for name,expected in a['results'][corpus]['primary_summary'].items():
                sums=np.stack([measurements[name][ix==j].sum(axis=0) for j in range(G)])
                B[name]=(W@sums)/denominator[:,None]
                for key,j in [('token_changed_fraction',1),('mean_normalized_edit_distance',3)]:
                    ci=np.quantile(B[name][:,j],[.05/40,1-.05/40]);np.testing.assert_allclose(expected[key+'_family20_CI95'],ci,atol=1e-12,rtol=0);intervals+=1
            for name,expected in a['results'][corpus]['canonical_minus_control'].items():
                delta=B['canonical']-B[name]
                for key,j in [('token_changed_fraction',1),('mean_normalized_edit_distance',3)]:
                    ci=np.quantile(delta[:,j],[.05/16,1-.05/16]);np.testing.assert_allclose(expected[key]['family8_CI95'],ci,atol=1e-12,rtol=0)
                    assert abs(expected[key]['canonical_minus_control']-(measurements['canonical']-measurements[name])[:,j].mean())<1e-12;intervals+=1
            geometry[corpus]={'all8_conditions_allclips_reconstructed':True,'budget_error_scaled_by_max_B_1':max_norm,'canonical_native_matrix_angle_max_deg':max_canonical_angle,'physical_Gram_max_change':max_gram,'physical_angle_error_max_rad':max_physical_angle_error}
    ratio=read(R/'rotation_ratio.json')
    for f,h in ratio['inputs_sha256'].items():assert sha(f)==h
    old=read('reports/cvpr_matched_blocks_20261013/guo/analysis.json');new=read('reports/cvpr_coherent_blocks_20261014/guo/analysis.json')
    for s,row in ratio['directions'].items():assert row['ratio_of_mean_FID_changes']==old['conditions'][f'null_{s}_r3']['FID']['mean_change']/new['conditions'][f'rotation_fixed_{s}_r3']['FID']['mean_change']
    write(R/'verification.json',{'passed':True,'allclip_condition_reconstructions':checked,'all_saved_hypothesis_pairs_independently_scored':hypothesis_pairs,'native_fixture_decodes_exact':forward,'all28_adjusted_intervals_recomputed':intervals==28,'interval_count':intervals,'geometry':geometry,'ratio_same_sign_arithmetic_verified':True,'source_sha256':sha(__file__)})
    print('SIGN_STABILITY_VERIFIED',checked,forward,intervals,flush=True)


if __name__=='__main__':main()
