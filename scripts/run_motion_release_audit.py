"""Paired native-component inference audits; preserves each emitted motion.

The twenty-repetition sample count and full-test FID reference are registered
in protocol.json. Public method inference adapters run in separate processes
to avoid the incompatible upstream ``utils`` / ``mld`` Python namespaces.
"""
import argparse, sys, types, time
from pathlib import Path
import numpy as np
import torch
from scipy import linalg
from motion_release_audit_common import R,A,DATA,read,write,sha,seed,native_dataset,materialize,evaluator,score_embeddings,canonical,metric

def register(method):
    p=R/f'producer_{method}.json'
    files=[Path(__file__),Path('scripts/motion_release_audit_common.py')]
    if method not in ['mdm50','real_controls']:
        files.append(Path('scripts/motion_release_native_adapters.py'))
    record={'method':method,'source_sha256':{str(x):sha(x) for x in files},'reference_draw_seed_offset':100000,'generation_rng_seed_offset':200000,'real_random_seed':42,'native_component_adapter':True}
    if p.exists():
        saved=read(p);saved.pop('revision_history',None)
        assert saved==record,'registered producer changed'
    else:write(p,record)
    protocol=read(R/'protocol.json')
    for p,h in protocol['producer_sha256'].items():assert sha(p)==h,(p,'registered preparation changed')

def mdm_model():
    sys.path[:0]=[str(Path(p).resolve()) for p in ['external/humanml_python','external/CLIP','external/MDM']]
    import clip
    old=clip.load
    def load(*a,**kw):kw['download_root']='models/humanml3d_audit/clip';return old(*a,**kw)
    clip.load=load
    stub=types.ModuleType('model.rotation2xyz')
    class NoMesh:
        def __init__(self,*a,**k):self.smpl_model=torch.nn.Identity()
        def __call__(self,*a,**k):raise RuntimeError('Mesh conversion unavailable')
    stub.Rotation2xyz=NoMesh;sys.modules['model.rotation2xyz']=stub
    from utils.model_util import create_model_and_diffusion
    from model.cfg_sampler import ClassifierFreeSampleModel
    cfg=types.SimpleNamespace(**read(A/'humanml_enc_512_50steps/args.json'))
    model,diffusion=create_model_and_diffusion(cfg,types.SimpleNamespace(dataset=types.SimpleNamespace(num_actions=1)))
    state=torch.load(A/'humanml_enc_512_50steps/model000750000.pt',map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not unexpected and all(x.startswith('clip_model.') for x in missing)
    model.to('cuda');model.eval();return model,ClassifierFreeSampleModel(model),diffusion

def generate_mdm(state,cohort,rep):
    model,guided,diffusion=state
    mu=np.load('external/HumanML3D/HumanML3D/Mean.npy');sd=np.load('external/HumanML3D/HumanML3D/Std.npy')
    out=np.zeros_like(cohort['raw'])
    with torch.no_grad():
        for i in range(0,len(out),32):
            L=torch.tensor(cohort['lengths'][i:i+32],device='cuda');mask=torch.arange(196,device='cuda')[None,:]<L[:,None]
            y={'text':cohort['captions'][i:i+32],'lengths':L,'mask':mask[:,None,None,:],'scale':torch.full((len(L),),2.5,device='cuda')}
            y['text_embed']=model.encode_text(y['text'])
            x=diffusion.p_sample_loop(guided,(len(L),263,1,196),clip_denoised=False,model_kwargs={'y':y},skip_timesteps=0,init_image=None,progress=False,dump_steps=None,noise=None,const_noise=False)
            x=x[:,:,0,:].permute(0,2,1).cpu().numpy()*sd+mu;x[~mask.cpu().numpy()]=0;out[i:i+len(L)]=x
            if i%256==0:print('GENERATE',rep,i,len(out),flush=True)
    return out

def recode(raw,name):
    if name=='identity':return raw
    if name=='canonical':return canonical(raw)
    d=raw[...,67:193].reshape(*raw.shape[:-1],21,6)
    if name=='random035_seed42':c=np.random.default_rng(42).uniform(-.35,.35,(*d.shape[:-1],3))
    else:
        b=float(name[-3:])/100
        if 'minus' in name:b=-b
        c=np.array([b,-b,b])
    y=raw.copy();y[...,67:193]=np.concatenate([np.exp(c[...,0:1])*d[...,:3],np.exp(c[...,1:2])*d[...,3:]+c[...,2:3]*d[...,:3]],-1).reshape(*raw.shape[:-1],126)
    return y

def geometry(raw,changed,L):
    mask=np.arange(196)[None,:]<L[:,None]
    a=canonical(raw)[...,67:193][mask];b=canonical(changed)[...,67:193][mask]
    keep=np.r_[0:67,193:263]
    assert np.array_equal(raw[...,keep][mask],changed[...,keep][mask])
    return {'valid_frames':int(mask.sum()),'max_canonical_basis_difference':float(np.max(np.abs(a-b))),'nonrotation_channels_bitwise_equal':True,'raw_6D_RMS_change':float(np.sqrt(np.mean((raw[...,67:193][mask].astype(float)-changed[...,67:193][mask])**2)))}

def save_cohort(folder,cohort):
    np.save(folder/'source_raw.npy',cohort['raw'])
    np.savez_compressed(folder/'conditioning.npz',**{k:v for k,v in cohort.items() if k!='raw'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['mdm50','real_controls','mld','motionlcm_v1','t2mgpt','t2m']);args=parser.parse_args()
    assert not (R/'manifest_sha256.json').exists();register(args.method)
    protocol=read(R/'protocol.json');seed(20261012);dataset=native_dataset();wrapper,mean,std=evaluator()
    if args.method=='mdm50':state=mdm_model()
    elif args.method!='real_controls':
        from motion_release_native_adapters import NativeGenerator
        state=NativeGenerator(args.method)
    for rep,s in enumerate(protocol['replication_seeds']):
        folder=R/args.method/f'rep_{rep:02d}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'metrics.json').exists():continue
        started=time.time();cohort=materialize(dataset,s,1000 if args.method=='mdm50' else None)
        reference=materialize(dataset,s+100000)
        _,refE=score_embeddings(wrapper,reference['raw'],reference,mean,std)
        np.save(folder/'reference_embeddings.npy',refE)
        write(folder/'reference_draw.json',{'seed':s+100000,'clips':len(refE),'clip_ids':reference['clip_ids'],'lengths':reference['lengths'].tolist(),'caption_draws':reference['captions']})
        del reference
        save_cohort(folder,cohort);seed(s+200000)
        if (folder/'generated_raw.npy').exists():raw=np.load(folder/'generated_raw.npy');saved=np.load(folder/'generated_lengths.npy');cohort['lengths']=saved
        elif args.method=='real_controls':raw=cohort['raw']
        else:
            if args.method=='mdm50':raw=generate_mdm(state,cohort,rep)
            else:raw,cohort['lengths']=state.generate(cohort)
            np.save(folder/'generated_raw.npy',raw);np.save(folder/'generated_lengths.npy',cohort['lengths'])
        assert np.isfinite(raw).all()
        results={};checks={};conditions=protocol['real_fixed_encoding_controls'] if args.method=='real_controls' else ['raw','canonical']
        for name in conditions:
            changed=raw if name=='raw' else recode(raw,name)
            T,E=score_embeddings(wrapper,changed,cohort,mean,std);results[name]=metric(T,E,refE);checks[name]=geometry(raw,changed,cohort['lengths'])
            np.save(folder/f'{name}_motion_embeddings.npy',E)
            if name in ['raw','identity']:np.save(folder/'text_embeddings.npy',T)
            print('SCORED',args.method,rep,name,results[name],flush=True)
            if changed is not raw:del changed
        write(folder/'metrics.json',{'method':args.method,'repetition':rep,'seed':s,'elapsed_seconds':time.time()-started,'results':results,'geometry':checks,'reference_clips':len(refE),'source_ground_truth_not_generation_input':args.method!='real_controls'})
        del cohort,raw,refE
    print('ALL_REPETITIONS_COMPLETE',args.method,flush=True)
if __name__=='__main__':main()
