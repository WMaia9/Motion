"""Twenty paired repetitions using native1000-step MotionDiffuse components."""
from pathlib import Path
import argparse,time
import numpy as np
from motion_release_audit_common import read,write,sha,seed,evaluator,score_embeddings,metric
import run_additional_motion_audit as base
from motiondiffuse_vectorized_adapter import VectorizedMotionDiffuse
R=Path('reports/cvpr_additional_motion_fast_20261013')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['humanml','kit']);a=parser.parse_args();base.R=R;folder=R/a.dataset/'motiondiffuse';folder.mkdir(parents=True,exist_ok=True);assert not (R/'manifest_sha256.json').exists();seed(20261013)
    if a.dataset=='humanml':wrapper,mean,std=evaluator()
    else:ds=base.dataset_kit();wrapper,mean,std=base.kit_evaluator()
    gen=VectorizedMotionDiffuse(a.dataset);files=[Path(__file__),Path('scripts/motiondiffuse_vectorized_adapter.py'),Path('scripts/additional_motion_adapters.py'),Path('scripts/run_additional_motion_audit.py'),Path('scripts/motion_release_audit_common.py'),Path('external/MotionDiffuse/text2motion/models/transformer.py'),Path('external/MotionDiffuse/text2motion/models/gaussian_diffusion.py'),*gen.files]
    p={'scope':'native paired output audit, not exact leaderboard reproduction','dataset':a.dataset,'method':'motiondiffuse','replications':20,'draw_seeds':list(range(1234,1254)),'generation_seed_offset':400000,'generation_batch':128,'native_steps':1000,'native_min_length':40 if a.dataset=='humanml' else 24,'conditions':['raw','canonical'],'outputs':1024 if a.dataset=='humanml' else 'full native KIT drop_last32','candidate_pool':32,'reference':'full native independently drawn reference, same saved HumanML20draw study or KIT seed+100000','vectorization':'native0/1 length mask only; exact mask and full forward parity; no denoiser/distribution/weights changed','previous_slow_incomplete_attempt_excluded':'reports/cvpr_additional_motion_audit_20261013/humanml/motiondiffuse','source_and_assets_sha256':{str(f):sha(f) for f in files}}
    if (folder/'protocol.json').exists():assert read(folder/'protocol.json')==p
    else:write(folder/'protocol.json',p)
    write(folder/'native_mask_parity.json',gen.parity)
    for rep,s in enumerate(p['draw_seeds']):
        out=folder/f'rep_{rep:02d}';out.mkdir(exist_ok=True)
        if (out/'metrics.json').exists():continue
        start=time.time()
        if a.dataset=='humanml':cohort=base.cohort_saved(base.PREVIOUS/f'real_controls/rep_{rep:02d}',1024);refE=np.load(base.PREVIOUS/f'real_controls/rep_{rep:02d}/reference_embeddings.npy')
        else:cohort=base.kit_materialize(ds,s);ref=base.kit_materialize(ds,s+100000);_,refE=score_embeddings(wrapper,ref['raw'],ref,mean,std);np.save(out/'reference_embeddings.npy',refE);del ref
        cohort['lengths']=np.maximum(cohort['lengths'],p['native_min_length']);np.savez_compressed(out/'conditioning.npz',**{k:v for k,v in cohort.items() if k!='raw'});seed(s+400000)
        if (out/'generated_raw.npy').exists():X=np.load(out/'generated_raw.npy')
        else:X=gen.generate(cohort);np.save(out/'generated_raw.npy',X)
        scores={};Y=base.canonical(X);check=base.geometry(X,Y,cohort['lengths'])
        for condition,raw in [('raw',X),('canonical',Y)]:
            T,E=score_embeddings(wrapper,raw,cohort,mean,std);scores[condition]=metric(T,E,refE);np.save(out/(condition+'_motion_embeddings.npy'),E)
            if condition=='raw':np.save(out/'text_embeddings.npy',T)
        write(out/'metrics.json',{'dataset':a.dataset,'method':'motiondiffuse','rep':rep,'scores':scores,'geometry':check,'elapsed_seconds':time.time()-start});print('MOTIONDIFFUSE_NATIVE_SCORED',a.dataset,rep,scores,flush=True);del cohort,X,Y,refE
    write(folder/'complete.json',{'twenty_native_repetitions':True});print('MOTIONDIFFUSE_NATIVE_COMPLETE',a.dataset,flush=True)
if __name__=='__main__':main()
