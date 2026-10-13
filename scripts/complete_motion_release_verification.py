"""Independent saved-metric and native-geometry checks for new release studies.

No generation, fitting, condition selection or change to previous producers.
"""
from pathlib import Path
import argparse, copy, sys
import numpy as np
import torch
from scipy.stats import t
from motion_release_audit_common import R, read, write, sha, evaluator, score_embeddings, canonical, seed
from run_motion_release_audit import recode
from verify_motion_prevalence import matrices

METHODS=['mdm50','mld','motionlcm_v1','t2mgpt','t2m']
REWARD=Path('reports/cvpr_motion_reward_release_20261013')

def native():
    sys.path.insert(0,str(Path('external/MDM').resolve()))
    from data_loaders.humanml.utils.metrics import calculate_R_precision,calculate_activation_statistics,calculate_frechet_distance
    from data_loaders.humanml.common.quaternion import cont6d_to_matrix
    from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
    return locals()

def check_metrics(api,T,E,ref,result):
    mu,cov=api['calculate_activation_statistics'](E);mr,cr=api['calculate_activation_statistics'](ref)
    fid=max(float(api['calculate_frechet_distance'](mu,cov,mr,cr)),0.)
    error=abs(fid-result['FID']);assert error<1e-8,(fid,result)
    r=sum(api['calculate_R_precision'](T[i:i+32],E[i:i+32],3,True) for i in range(0,len(E),32))/len(E)
    assert np.max(abs(r-result['R_precision']))<1e-12,(r,result['R_precision'])
    return error

def check_geometry(api,x,y,L,fk=2):
    mask=np.arange(x.shape[1])[None,:]<L[:,None]
    assert np.array_equal(x[...,:67],y[...,:67]) and np.array_equal(x[...,193:],y[...,193:])
    d=x[...,67:193].reshape(*x.shape[:-1],21,6)[mask];e=y[...,67:193].reshape(*y.shape[:-1],21,6)[mask]
    a,_,_=matrices(d);b,_,_=matrices(e);delta=a.swapaxes(-1,-2)@b
    skew=np.stack([delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],delta[...,1,0]-delta[...,0,1]],-1)/2
    angle=float(np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1),(np.trace(delta,axis1=-2,axis2=-1)-1)/2)).max())
    native_error=float(abs(api['cont6d_to_matrix'](torch.tensor(d))-api['cont6d_to_matrix'](torch.tensor(e))).max());assert angle<.01 and native_error<3e-5
    fkerr=0.
    for j in range(min(fk,len(x))):
        aa=torch.tensor(x[j,:int(L[j])]);bb=torch.tensor(y[j,:int(L[j])]);pa=api['recover_from_ric'](aa,22);pb=api['recover_from_ric'](bb,22);assert torch.equal(pa,pb)
        sk=api['Skeleton'](torch.tensor(api['t2m_raw_offsets']),api['t2m_kinematic_chain'],'cpu');sk.get_offsets_joints(pa[0])
        er=float(abs(api['recover_from_rot'](aa,22,sk)-api['recover_from_rot'](bb,22,sk)).max());assert er<1e-4;fkerr=max(fkerr,er)
    return {'max_rotation_error_deg':angle,'max_native_converter_error':native_error,'max_native_FK_error':fkerr}

def readers():
    wrapper,mean,std=evaluator();second=copy.deepcopy(wrapper)
    state=torch.load('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar',map_location='cpu',weights_only=False)
    for k,m in [('movement_encoder',second.movement_encoder),('motion_encoder',second.motion_encoder),('text_encoder',second.text_encoder)]:m.load_state_dict(state[k],strict=True);m.eval()
    return {'common_MDM_checkpoint':wrapper,'official_GPT_checkpoint':second},mean,std

def extras():
    api=native();models,mean,std=readers();maximum=0.;details=[]
    for rep in range(20):
        folder=R/'real_controls'/f'rep_{rep:02d}';row=read(folder/'metrics.json');x=np.load(folder/'source_raw.npy',mmap_mode='r');L=np.load(folder/'conditioning.npz')['lengths'];ref=np.load(folder/'reference_embeddings.npy');T=np.load(folder/'text_embeddings.npy')
        for condition,result in row['results'].items():
            maximum=max(maximum,check_metrics(api,T,np.load(folder/f'{condition}_motion_embeddings.npy'),ref,result))
            # Directional transforms are identical operators at every clip. Random
            # draws must be reconstructed with the full source shape for parity.
            y=recode(x,condition);details.append({'rep':rep,'condition':condition,'sampled_clips':32,**check_geometry(api,np.array(x[:32]),np.array(y[:32]),L[:32])});del y
            assert row['geometry'][condition]['nonrotation_channels_bitwise_equal']
        for method in METHODS:
            src=R/method/f'rep_{rep:02d}';z=np.load(src/'conditioning.npz');cohort={k:z[k].copy() for k in z.files};cohort['lengths']=np.load(src/'generated_lengths.npy')
            # Text embeddings do not depend on the dummy motion; preserve native
            # wrapper ordering by passing the actual generated durations.
            T,_=score_embeddings(models['official_GPT_checkpoint'],np.zeros((len(cohort['lengths']),196,263),np.float32),cohort,mean,std)
            target=R/'secondary_gpt_reader'/method/f'rep_{rep:02d}';row=read(target/'metrics.json');ref=np.load(R/'secondary_gpt_reader'/f'reference_rep_{rep:02d}.npy')
            for condition in ['raw','canonical']:maximum=max(maximum,check_metrics(api,T,np.load(target/f'{condition}_embeddings.npy'),ref,row['scores'][condition]))
        print('EXTRA_VERIFIED',rep,flush=True)
    write(R/'additional_verification.json',{'passed':True,'secondary_saved_metric_endpoints':200,'real_fixed_saved_metric_endpoints':140,'native_real_geometry_sample_pairs':20*7*32,'native_real_FK_pairs':20*7*2,'geometry':details,'max_saved_FID_error':maximum,'code_sha256':sha(__file__)})

def stats(a,b,family):
    a=np.asarray(a,float);b=np.asarray(b,float);d=a-b;n=len(d);se=d.std(ddof=1)/np.sqrt(n)
    return {'raw_mean':float(a.mean()),'canonical_mean':float(b.mean()),'paired_raw_minus_canonical':float(d.mean()),'paired_family_CI95':(d.mean()+np.array([-1,1])*t.ppf(1-.05/(2*family),n-1)*se).tolist(),'family_size':family,'replications':n}

def reward():
    assert read(REWARD/'complete.json')['generation_seeds']==5
    api=native();models,mean,std=readers();checks=[];maxmetric=0.;rows={}
    for method in ['base','reward_finetuned']:
        rows[method]=[]
        for rep in range(5):
            folder=REWARD/method/f'rep_{rep:02d}';row=read(folder/'metrics.json');rows[method].append(row);src=R/'mdm50'/f'rep_{rep:02d}';z=np.load(src/'conditioning.npz');cohort={k:z[k].copy() for k in z.files};x=np.load(folder/'generated_raw.npy',mmap_mode='r')
            for i in range(0,len(x),32):checks.append(check_geometry(api,np.array(x[i:i+32]),canonical(x[i:i+32]),cohort['lengths'][i:i+32],fk=2 if i==0 else 0))
            for name,model in models.items():
                T,_=score_embeddings(model,np.zeros_like(x),cohort,mean,std)
                ref=np.load(src/'reference_embeddings.npy') if name=='common_MDM_checkpoint' else np.load(R/'secondary_gpt_reader'/f'reference_rep_{rep:02d}.npy')
                for cond in ['raw','canonical']:maxmetric=max(maxmetric,check_metrics(api,T,np.load(folder/f'{cond}_{name}_embeddings.npy'),ref,row['results'][cond][name]))
            print('REWARD_RELEASE_VERIFIED',method,rep,flush=True)
    out={}
    for name in models:
        out[name]={}
        for method in rows:
            out[name][method]={}
            for metric,idx in [('FID',None),('R1',0)]:
                def values(cond):return [r['results'][cond][name]['FID'] if idx is None else r['results'][cond][name]['R_precision'][idx] for r in rows[method]]
                out[name][method][metric]=stats(values('raw'),values('canonical'),8)
        out[name]['finetuned_minus_base']={cond:{'FID':float(np.mean([rows['reward_finetuned'][i]['results'][cond][name]['FID']-rows['base'][i]['results'][cond][name]['FID'] for i in range(5)])),'R1':float(np.mean([rows['reward_finetuned'][i]['results'][cond][name]['R_precision'][0]-rows['base'][i]['results'][cond][name]['R_precision'][0] for i in range(5)]))} for cond in ['raw','canonical']}
    write(REWARD/'analysis.json',{'readers':out,'same_architecture_two_distinct_checkpoints':True,'not_native20_repetition_protocol':True,'all_endpoints_retained':True,'analysis_code_sha256':sha(__file__)})
    write(REWARD/'verification.json',{'passed':True,'outputs':10240,'native_FK_pairs':20,'protected_channels_bitwise_equal':True,'max_rotation_error_deg':max(r['max_rotation_error_deg'] for r in checks),'max_native_converter_error':max(r['max_native_converter_error'] for r in checks),'max_native_FK_error':max(r['max_native_FK_error'] for r in checks),'saved_metric_endpoints_recomputed':40,'max_saved_FID_error':maxmetric,'code_sha256':sha(__file__)})
    print(out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['extras','reward']);args=parser.parse_args();seed(434)
    (extras if args.phase=='extras' else reward)()
