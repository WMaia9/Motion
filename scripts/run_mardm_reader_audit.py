"""Compare essential-reader and canonical-reader ranks on eight frozen releases.

The same saved real-source cohort is used as reference for every reader. Guo
scores are recomputed from saved embeddings with that reference, so reader
rank comparisons do not mix independently selected reference motions.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t,kendalltau
from motion_release_audit_common import R as OLD,read,write,sha,seed,metric,canonical
from mardm_eval_common import EssentialEvaluators,SOURCE,MODELS
R=Path('reports/cvpr_mardm_readers_20261013')
ADD=Path('reports/cvpr_additional_motion_audit_20261013');MD=Path('reports/cvpr_motiondiffuse_compiled_20261013')
METHODS={m:OLD/m for m in ['mdm50','mld','motionlcm_v1','t2mgpt','t2m']};METHODS.update(momask=ADD/'humanml/momask',dip=ADD/'humanml/dip',motiondiffuse=MD/'humanml/motiondiffuse')


def main():
    assert not (R/'manifest_sha256.json').exists();R.mkdir(exist_ok=True)
    p={'methods':list(METHODS),'draws':20,'no_generation_or_model_fitting':True,'MARDM_models':'strict public67-feature GRU and CLIP models','source_revision':'9b05ad867be916afb9fcbff663b42a03f933014a','model_revision':'592ff3cb511c477349ea5947623f289b35ee2cea','normalization':'official MARDM utils/eval_mean_std/t2m (67 dimensions)','reference':'same saved real_controls source cohort for all readers, encoded separately; Guo saved generation embeddings re-scored against saved identity real embeddings, rather than original independent reference draw','cohort_counts':'MDM/DiP/MotionDiffuse1024; others4640; DiP source-prefix conditioned','invariance':'unchanged first67 features under rotation recoding; structural property plus native repeated-forward checks, not accuracy-cost evidence','CLIP_metric':'native normalized latent vectors and 32candidate retrieval; FID of this encoder is descriptive embedding distance, not an official leaderboard score','comparison':'within-reader mean FID ranks and Kendall agreement; not numerical FID scale comparison or causal attribution across training regimes','inputs_source_sha256':{str(f):sha(f) for f in [Path(__file__),Path('scripts/mardm_eval_common.py'),SOURCE/'utils/evaluators.py',SOURCE/'utils/eval_mean_std/t2m/eval_mean.npy',SOURCE/'utils/eval_mean_std/t2m/eval_std.npy',MODELS/'text_mot_match/model/finest.tar',MODELS/'text_mot_match_clip/model/finest.tar',OLD/'manifest_sha256.json',ADD/'manifest_sha256.json',MD/'manifest_sha256.json']}}
    if (R/'protocol.json').exists():assert read(R/'protocol.json')==p
    else:write(R/'protocol.json',p)
    seed(4321);model=EssentialEvaluators();sys.path.append(str(Path('external/MDM').resolve()))
    for rep in range(20):
        folder=R/f'rep_{rep:02d}';folder.mkdir(exist_ok=True)
        if (folder/'metrics.json').exists():continue
        source=OLD/'real_controls'/f'rep_{rep:02d}';raw=np.load(source/'source_raw.npy',mmap_mode='r');z=np.load(source/'conditioning.npz');cohort={k:z[k].copy() for k in z.files};ref=model.encode(raw,cohort)
        for reader,(T,E) in ref.items():np.save(folder/(reader+'_real_reference.npy'),E)
        guoref=np.load(source/'identity_motion_embeddings.npy');results={};checks={}
        for name,root in METHODS.items():
            src=root/f'rep_{rep:02d}';X=np.load(src/'generated_raw.npy',mmap_mode='r');z=np.load(src/'conditioning.npz');c={k:z[k].copy() for k in z.files};
            if (src/'generated_lengths.npy').exists():c['lengths']=np.load(src/'generated_lengths.npy')
            encoded=model.encode(X,c);record={}
            for reader,(T,E) in encoded.items():
                record[reader]=metric(T,E,ref[reader][1]);np.save(folder/(name+'_'+reader+'_embedding.npy'),E);np.save(folder/(name+'_'+reader+'_text.npy'),T)
            for cond in ['raw','canonical']:
                T=np.load(src/'text_embeddings.npy');E=np.load(src/(cond+'_motion_embeddings.npy'));record['guo_'+cond]=metric(T,E,guoref)
            # Fixed gauge transformation, before any scores; first native32 pool.
            x=np.array(X[:32]);y=x.copy();d=x[...,67:193].reshape(*x.shape[:2],21,6);y[...,67:193]=np.concatenate([np.exp(.35)*d[...,:3],np.exp(-.35)*d[...,3:]+.35*d[...,:3]],-1).reshape(*x.shape[:2],126);assert np.array_equal(x[...,:67],y[...,:67])
            cc={k:v[:32] for k,v in c.items()};a=model.encode(x,cc);b=model.encode(y,cc);changes={reader:float(abs(a[reader][1]-b[reader][1]).max()) for reader in ['gru','clip']};assert max(changes.values())==0
            checks[name]={'first67_bitwise_equal':True,'native_gauge_embedding_max_change':changes};results[name]=record;print('MARDM_READER_SCORED',rep,name,record,flush=True)
        write(folder/'metrics.json',{'rep':rep,'methods':results,'invariance_checks':checks})
    out={}
    for name in METHODS:
        rows=[read(R/f'rep_{i:02d}/metrics.json')['methods'][name] for i in range(20)];out[name]={}
        for reader in ['guo_raw','guo_canonical','gru','clip']:
            values=np.array([r[reader]['FID'] for r in rows]);out[name][reader]={'FID_mean':float(values.mean()),'t19_CI95':(values.mean()+np.array([-1,1])*t.ppf(.975,19)*values.std(ddof=1)/np.sqrt(20)).tolist(),'R1_mean':float(np.mean([r[reader]['R_precision'][0] for r in rows]))}
    ranks={r:sorted(METHODS,key=lambda name:out[name][r]['FID_mean']) for r in ['guo_raw','guo_canonical','gru','clip']};agreement={r:float(kendalltau([ranks['guo_canonical'].index(n) for n in METHODS],[ranks[r].index(n) for n in METHODS]).statistic) for r in ['gru','clip']}
    write(R/'analysis.json',{'methods':out,'mean_FID_rankings':ranks,'Kendall_vs_canonical_Guo':agreement,'all20_draws_complete':True,'essential_invariance_structural_not_accuracy_cost':True,'rank_comparison_not_direct_FID_scale_comparison':True});write(R/'complete.json',{'eight_releases_twenty_draws':True});print('MARDM_READER_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
