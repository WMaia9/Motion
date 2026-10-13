"""Matched ordinary MDM fine-tuning with frozen retrieval rewards.

Full denoiser parameters are updated, without any appended gauge layer or
reference-dependent test optimization. This is a controlled TRAIN-subset
pilot using the standard evaluator, not a reproduction of MotionReward.
"""
import argparse,sys,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from motion_release_audit_common import A,read,write,sha,seed,evaluator
from run_motion_release_audit import mdm_model
R=Path('reports/cvpr_humanml_reward_pilot_20261012')
TRAIN=Path('reports/cvpr_humanml3d_blind_fid_20261012')

def canonical(x):
    y=x.clone();d=x[...,67:193].reshape(*x.shape[:-1],21,6);u=F.normalize(d[...,:3],dim=-1);v=F.normalize(d[...,3:]-(u*d[...,3:]).sum(-1,keepdim=True)*u,dim=-1);y[...,67:193]=torch.cat([u,v],-1).reshape(*x.shape[:-1],126);return y

def reward_embeddings(wrapper,x,L):
    order=torch.argsort(L,descending=True,stable=True);inv=torch.argsort(order);moves=wrapper.movement_encoder(x[order,:,:259]);return wrapper.motion_encoder(moves,(L[order]//4).cpu())[inv]

def train_text(wrapper,items):
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
    vec=WordVectorizer(str(A/'glove'),'our_vab');W=[];P=[];L=[]
    for x in items:
        t=['sos/OTHER']+x['tokens'].split()[:20]+['eos/OTHER'];L.append(len(t));t+=['unk/OTHER']*(22-len(t));v=[vec[z] for z in t];W.append(np.stack([z[0] for z in v]));P.append(np.stack([z[1] for z in v]))
    out=[]
    with torch.no_grad():
        for i in range(0,len(W),64):
            l=torch.tensor(L[i:i+64]);order=torch.argsort(l,descending=True,stable=True);inv=torch.argsort(order);w=torch.tensor(np.array(W[i:i+64]),device='cuda').float();p=torch.tensor(np.array(P[i:i+64]),device='cuda').float();out.append(wrapper.text_encoder(w[order],p[order],l[order])[inv].cpu())
    return torch.cat(out)

def register():
    R.mkdir(exist_ok=True);path=R/'protocol.json';inputs=[TRAIN/'train_selection.json',TRAIN/'train_real.npz',A/'humanml_enc_512_50steps/model000750000.pt',A/'t2m/text_mot_match/model/finest.tar',Path('reports/cvpr_motion_prevalence_20261012/mdm50/rep_00/conditioning.npz'),Path('reports/cvpr_motion_prevalence_20261012/mdm50/rep_00/reference_embeddings.npy')]
    sources=[Path(__file__),Path('scripts/run_motion_release_audit.py'),Path('scripts/motion_release_audit_common.py'),Path('scripts/score_humanml_reward_pilot.py')]
    record={'scope':'Controlled ordinary full-denoiser MDM reward pilot, not MotionRFT replication','TRAIN_subset_clips':2048,'conditions':['mse_only','raw_reward','canonical_reward'],'training_seeds':[42,123,777],'steps':1000,'batch_size':16,'optimizer':'AdamW lr1e-5, weight_decay1e-4, clipping1','initialization':'same official MDM50 checkpoint','motion_loss':'masked coordinate MSE on normalized x0','reward':'frozen standard T2M reader, cosine InfoNCE against source TRAIN text, temperature0.1, weight1','reward_timestep':'native50step q_sample uniformly over its50 timesteps, one x0 prediction, not recursive rollout reward','fixed_final_iterate':True,'clip_encoder_frozen':True,'no_appended_attack_layer':True,'evaluation':'one fixed1024-output native TEST cohort and full4640-output reference per training seed, common generation seed20261031; descriptive pilot, not20repetition published-score protocol','TEST_already_used_in_previous_audits':True,'sealing':'all9final weights hashed before any pilot TEST generation or scoring','primary':'same-output raw-minus-canonical FID/R1 and matched raw/canonical reward contrasts, three training seeds; no claim of spontaneous caption coding from gap alone','inputs_sha256':{str(p):sha(p) for p in inputs},'scientific_sources_sha256':{str(p):sha(p) for p in sources}}
    if path.exists():assert read(path)==record,'registered pilot changed'
    else:write(path,record)
    return record

def fit(s):
    protocol=register();wrapper,evalmean,evalstd=evaluator();items=read(TRAIN/'train_selection.json');texts=train_text(wrapper,items)
    with np.load(TRAIN/'train_real.npz') as z:raw=z['raw'].copy();lengths=z['lengths'].copy()
    mean=torch.tensor(np.load('external/HumanML3D/HumanML3D/Mean.npy'),device='cuda',dtype=torch.float32);std=torch.tensor(np.load('external/HumanML3D/HumanML3D/Std.npy'),device='cuda',dtype=torch.float32);evalmean=torch.tensor(evalmean,device='cuda',dtype=torch.float32);evalstd=torch.tensor(evalstd,device='cuda',dtype=torch.float32)
    for condition in protocol['conditions']:
        folder=R/f'seed_{s}'/condition;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():continue
        seed(s);model,_,diffusion=mdm_model();model.train();model.clip_model.eval();params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-4);seed(s+60000);rng=np.random.default_rng(s+60000);history=[]
        # cuDNN needs recurrent train mode for input gradients; dropout is zero.
        wrapper.motion_encoder.gru.train();assert wrapper.motion_encoder.gru.dropout==0
        for step in range(protocol['steps']):
            idx=rng.choice(len(raw),protocol['batch_size'],replace=False);L=torch.tensor(lengths[idx],device='cuda');mask=torch.arange(196,device='cuda')[None,:]<L[:,None];X=torch.tensor(raw[idx],device='cuda');norm=(X-mean)/std;norm*=mask[...,None];x0=norm.permute(0,2,1)[:,:,None,:];t=torch.randint(diffusion.num_timesteps,(len(idx),),device='cuda');noise=torch.randn_like(x0);xt=diffusion.q_sample(x0,t,noise=noise)
            y={'text':[items[i]['caption'] for i in idx],'lengths':L,'mask':mask[:,None,None,:]};prediction=model(xt,diffusion._scale_timesteps(t),y=y);pred=prediction[:,:,0,:].permute(0,2,1);mse=(((pred-norm)**2)*mask[...,None]).sum()/(mask.sum()*263);ce=torch.zeros((),device='cuda')
            if condition!='mse_only':
                physical=pred*std+mean
                if condition=='canonical_reward':physical=canonical(physical)
                normalized=(physical-evalmean)/evalstd;normalized*=mask[...,None];E=reward_embeddings(wrapper,normalized,L);T=texts[idx].cuda();logits=F.normalize(E,dim=-1)@F.normalize(T,dim=-1).T/.1;ce=F.cross_entropy(logits,torch.arange(len(idx),device='cuda'))
            loss=mse+ce;opt.zero_grad(set_to_none=True);loss.backward();normgrad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(loss) and torch.isfinite(normgrad);opt.step()
            if (step+1)%50==0:
                row={'step':step+1,'mse':float(mse.detach()),'reward_CE':float(ce.detach()),'grad_norm':float(normgrad)};history.append(row);write(folder/'training_log.json',history);print('REWARD_FIT',s,condition,row,flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('clip_model.')};torch.save(state,folder/'final.pt');write(folder/'complete.json',{'seed':s,'condition':condition,'steps':protocol['steps'],'trainable_parameters':sum(p.numel() for p in params),'weights_sha256':sha(folder/'final.pt'),'protocol_sha256':sha(R/'protocol.json')});del model,opt,params,state;torch.cuda.empty_cache()
    print('REWARD_SEED_COMPLETE',s,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('seed',type=int,choices=[42,123,777]);args=parser.parse_args();fit(args.seed)
if __name__=='__main__':main()
