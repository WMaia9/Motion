"""Prospective paired public-output audit in HumanML3D and standard KIT-ML.

Native generation components; paired audit, not a published-score reproduction.
HumanML reference/cohorts reuse the sealed twenty-draw study. KIT is loaded
from the official processed archive, with its own 251-feature evaluator.
"""
import argparse,sys,types,os,time
from pathlib import Path
import numpy as np
import torch
from motion_release_audit_common import A,read,write,sha,seed,materialize,score_embeddings,metric,evaluator
from additional_motion_adapters import NativeAdditional,ASSETS
R=Path('reports/cvpr_additional_motion_audit_20261013')
PREVIOUS=Path('reports/cvpr_motion_prevalence_20261012')

def canonical(x):
    n=21 if x.shape[-1]==263 else 20;start=4+3*n;stop=start+6*n
    y=x.copy();d=x[...,start:stop].reshape(*x.shape[:-1],n,6).astype(float)
    u=d[...,:3]/np.maximum(np.linalg.norm(d[...,:3],axis=-1,keepdims=True),1e-12)
    v=d[...,3:]-(u*d[...,3:]).sum(-1,keepdims=True)*u;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
    y[...,start:stop]=np.concatenate([u,v],-1).reshape(*x.shape[:-1],6*n)
    return y

def dataset_kit():
    sys.path.insert(0,str(Path('external/MDM').resolve()))
    if 'spacy' not in sys.modules:
        m=types.ModuleType('spacy');m.load=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('unused pretokenized import'));sys.modules['spacy']=m
    from data_loaders.humanml.data.dataset import Text2MotionDatasetV2
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
    data=ASSETS/'KIT-ML';cache=R/'kit/native_cache';(cache/'dataset').mkdir(parents=True,exist_ok=True)
    opt=types.SimpleNamespace(dataset_name='kit',fixed_len=0,use_cache=True,cache_dir=str(cache),max_motion_length=196,max_text_len=20,unit_length=4,disable_offset_aug=False,motion_dir=str(data/'new_joint_vecs'),text_dir=str(data/'texts'))
    return Text2MotionDatasetV2(opt,np.zeros(251),np.ones(251),str(data/'test.txt'),WordVectorizer(str(A/'glove'),'our_vab'))

def kit_materialize(ds,s):
    # The generic native materializer fixes263 output channels; use the same
    # equation/order with the native KIT feature width251.
    seed(s);order=np.random.permutation(len(ds));order=order[:len(order)//32*32]
    rows=[ds[int(i)] for i in order]
    return {'raw':np.array([x[4] for x in rows],dtype=np.float32),'words':np.array([x[0] for x in rows],dtype=np.float32),'pos':np.array([x[1] for x in rows],dtype=np.float32),'captions':[x[2] for x in rows],'sent_lengths':np.array([x[3] for x in rows]),'lengths':np.array([x[5] for x in rows]),'tokens':[x[6] for x in rows],'clip_ids':[ds.name_list[int(i)] for i in order]}

def kit_evaluator():
    sys.path.insert(0,str(Path('external/MDM').resolve()))
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    cwd=Path.cwd();os.chdir(ASSETS/'kit_evaluator')
    try:model=EvaluatorMDMWrapper('kit',torch.device('cuda'))
    finally:os.chdir(cwd)
    for module in [model.movement_encoder,model.motion_encoder,model.text_encoder]:
        module.eval()
        for p in module.parameters():p.requires_grad_(False)
    folder=ASSETS/'kit_evaluator/kit/Comp_v6_KLD005/meta'
    mean=np.load(folder/'mean.npy');std=np.load(folder/'std.npy');assert mean.shape==std.shape==(251,)
    return model,mean,std

def cohort_saved(path,limit=None):
    with np.load(path/'conditioning.npz') as z:cohort={k:z[k].tolist() if k in ['captions','tokens','clip_ids'] else z[k].copy() for k in z.files}
    cohort['raw']=np.load(path/'source_raw.npy')
    if limit is not None:cohort={k:v[:limit] for k,v in cohort.items()}
    return cohort

def geometry(X,Y,L):
    n=21 if X.shape[-1]==263 else 20;start=4+3*n;stop=start+6*n;mask=np.arange(196)[None,:]<L[:,None]
    keep=np.r_[0:start,stop:X.shape[-1]];assert np.array_equal(X[...,keep],Y[...,keep])
    err=float(abs(canonical(X)[...,start:stop][mask]-Y[...,start:stop][mask]).max())
    assert err==0
    return {'nonrotation_channels_bitwise_equal':True,'canonical_array_recomputed_exactly':True,'valid_frames':int(mask.sum()),'rotation_block':[start,stop]}

def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['momask','motiondiffuse','dip','real_controls']);p.add_argument('dataset',choices=['humanml','kit']);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    assert not (R/'manifest_sha256.json').exists();folder=R/args.dataset/args.method;folder.mkdir(parents=True,exist_ok=True)
    seed(20261013)
    if args.dataset=='kit':ds=dataset_kit();wrapper,mean,std=kit_evaluator()
    else:wrapper,mean,std=evaluator()
    generator=None if args.method=='real_controls' else NativeAdditional(args.method,args.dataset)
    files=[Path(__file__),Path('scripts/additional_motion_adapters.py'),Path('scripts/motion_release_audit_common.py')]
    if generator:files+=generator.files
    if args.dataset=='humanml':files+=[PREVIOUS/'manifest_sha256.json']
    else:files+=[ASSETS/'KIT-ML/test.txt',ASSETS/'kit_evaluator/kit/text_mot_match/model/finest.tar',ASSETS/'kit_evaluator/kit/Comp_v6_KLD005/meta/mean.npy',ASSETS/'kit_evaluator/kit/Comp_v6_KLD005/meta/std.npy']
    protocol={'method':args.method,'dataset':args.dataset,'repetitions':20,'draw_seeds':list(range(1234,1254)),'generation_seed_offset':400000,'conditions':['raw','canonical'],'candidate_pool':32,'reference':'full native drop_last32; HumanML saved independently drawn reference from frozen20rep study; KIT new native reference seed+100000','outputs':'full native test for MoMask and KIT,1024 HumanML for MotionDiffuse/DiP','generation_batch':128,'inference':'MoMask18masksteps CFG4 topk0.9 temp1 residualCFG5; MotionDiffuse native1000 DDPM steps linear EPS/FIXED_SMALL; DiP10 native autoregressive steps context20 predict40 CFG7.5 model200000 EMA','DiP_source_motion_prefix_conditioning':args.method=='dip','DiP_prefix':'first20frames of each saved cropped source motion; no future source frames passed; prefix excluded from generated output, not a text-only comparison' if args.method=='dip' else None,'not_exact_published_score_reproduction':True,'paired_audit_not_generator_quality_or_reward_exploitation':True,'KIT_TMR263_AMASS_checkpoint_not_applied_to251_MMM_features':True,'source_and_assets_sha256':{str(f):sha(f) for f in files}}
    if (folder/'protocol.json').exists():assert read(folder/'protocol.json')==protocol
    else:write(folder/'protocol.json',protocol)
    if args.preflight:
        cohort=kit_materialize(ds,1234) if args.dataset=='kit' else cohort_saved(PREVIOUS/'real_controls/rep_00',32)
        cohort={k:v[:32] for k,v in cohort.items()};seed(401234);X=cohort['raw'] if generator is None else generator.generate(cohort,batch=32)
        assert X.shape==(32,196,251 if args.dataset=='kit' else 263)
        T,E=score_embeddings(wrapper,X,cohort,mean,std);TC,EC=score_embeddings(wrapper,canonical(X),cohort,mean,std);assert np.array_equal(T,TC)
        # Forward scores are deliberately not used to select checkpoints or settings.
        write(folder/'preflight.json',{'shape':list(X.shape),'native_components_loaded':True,'strict_trainable_state_load':True,'finite_embeddings':bool(np.isfinite(E).all() and np.isfinite(EC).all()),'test_metrics_not_reported_or_used_for_selection':True});print('ADDITIONAL_PREFLIGHT_PASSED',args.method,args.dataset,flush=True);return
    for rep,s in enumerate(protocol['draw_seeds']):
        out=folder/f'rep_{rep:02d}';out.mkdir(exist_ok=True)
        if (out/'metrics.json').exists():continue
        started=time.time()
        if args.dataset=='humanml':
            cohort=cohort_saved(PREVIOUS/f'real_controls/rep_{rep:02d}',1024 if args.method in ['motiondiffuse','dip'] else None);refE=np.load(PREVIOUS/f'real_controls/rep_{rep:02d}/reference_embeddings.npy')
        else:
            cohort=kit_materialize(ds,s);ref=kit_materialize(ds,s+100000);_,refE=score_embeddings(wrapper,ref['raw'],ref,mean,std);np.save(out/'reference_embeddings.npy',refE);del ref
        if not (out/'conditioning.npz').exists():
            np.save(out/'source_raw.npy',cohort['raw']);np.savez_compressed(out/'conditioning.npz',**{k:v for k,v in cohort.items() if k!='raw'})
        seed(s+400000)
        if (out/'generated_raw.npy').exists():X=np.load(out/'generated_raw.npy')
        else:
            X=cohort['raw'] if generator is None else generator.generate(cohort);np.save(out/'generated_raw.npy',X)
        Y=canonical(X);checks=geometry(X,Y,cohort['lengths']);scores={}
        for name,x in [('raw',X),('canonical',Y)]:
            T,E=score_embeddings(wrapper,x,cohort,mean,std);scores[name]=metric(T,E,refE);np.save(out/f'{name}_motion_embeddings.npy',E)
            if name=='raw':np.save(out/'text_embeddings.npy',T)
        write(out/'metrics.json',{'method':args.method,'dataset':args.dataset,'rep':rep,'scores':scores,'geometry':checks,'elapsed_seconds':time.time()-started});print('ADDITIONAL_SCORED',args.method,args.dataset,rep,scores,flush=True)
        del cohort,X,Y,refE
    write(folder/'complete.json',{'twenty_repetitions_complete':True});print('ADDITIONAL_METHOD_COMPLETE',args.method,args.dataset,flush=True)
if __name__=='__main__':main()
