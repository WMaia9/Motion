"""TRAIN-only CTC pseudo partitions and matched local physical path losses."""
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import run_physical_semantic_ddpm as previous
import run_fixed_prefix_diffusion as scorer
from run_motion_ctc_confirmation import Reader

ROOT=Path('reports/cvpr_local_path_diffusion_20261011')
HISTORICAL=Path('reports/cvpr_motion_path_diffusion_20261010')
ARMS=['local_uniform_endpoint','local_uniform_area','local_ctc_endpoint','local_ctc_area']
CONTROLS=['geometry_replay','physical_velocity']


def viterbi(logp, target):
    """Exact max log-probability CTC path, including repeated target labels."""
    target=list(target); t=len(logp)
    if not target:return np.zeros(t,dtype=np.int32),float(logp[:,0].sum())
    if len(target)+sum(a==b for a,b in zip(target,target[1:]))>t:return None,None
    labels=np.zeros(2*len(target)+1,np.int64);labels[1::2]=target
    n=len(labels);p=np.full(n,-np.inf);p[:2]=logp[0,labels[:2]]
    back=np.zeros((t,n),np.int16)
    skip=np.zeros(n,bool);skip[2:]=(labels[2:]!=0)&(labels[2:]!=labels[:-2])
    for i in range(1,t):
        candidates=np.stack([p,np.r_[-np.inf,p[:-1]],np.r_[-np.inf,-np.inf,p[:-2]]])
        candidates[2,~skip]=-np.inf
        choices=candidates.argmax(0);back[i]=choices
        p=candidates[choices,np.arange(n)]+logp[i,labels]
    end=n-1 if p[-1]>=p[-2] else n-2
    score=float(p[end])
    if not np.isfinite(score):return None,None
    states=np.empty(t,np.int32);states[-1]=end
    for i in range(t-1,0,-1):states[i-1]=states[i]-back[i,states[i]]
    return states,score


def uniform(length,k):
    return np.rint(np.linspace(0,length,k+1)).astype(int).tolist()


def bounded_partition(candidates,length,k):
    boundaries=[0]
    for i,b in enumerate(candidates,1):
        boundaries.append(max(boundaries[-1]+2,min(int(b),length-2*(k-i))))
    boundaries.append(length)
    assert len(boundaries)==k+1 and min(np.diff(boundaries))>=2
    return boundaries


@torch.no_grad()
def align_train(train):
    filename=ROOT/'train_partitions.json'
    if filename.exists():return json.loads(filename.read_text())
    ck=torch.load(previous.PREV/'ctc_reader/best.pt',map_location='cpu',weights_only=False)
    state=ck['state_dict'];vocab=ck['vocab'];vstoi={w:i for i,w in enumerate(vocab)}
    reader=Reader(len(vocab),state['mean'].numpy(),state['std'].numpy()).cuda()
    reader.load_state_dict(state,strict=True);reader.eval()
    records=[None]*len(train);counts={};edited=0;occupied=[];scores=[]
    order=np.argsort([len(e.sgn) for e in train],kind='stable')
    for start in range(0,len(order),64):
        ids=order[start:start+64];length=torch.tensor([len(train[i].sgn) for i in ids],device='cuda')
        x=torch.zeros(len(ids),int(length.max()),240,device='cuda')
        for j,i in enumerate(ids):x[j,:length[j]]=torch.from_numpy(previous.project(train[i].sgn.numpy())).cuda()
        lp,ol=reader(x,length);lp=lp.cpu().numpy()
        for j,i in enumerate(ids):
            e=train[i];n=len(e.sgn);gloss=list(e.gls);k=min(max(1,len(gloss)),n//2)
            assert n>=2
            reason=None;bound=uniform(n,k);path=None;score=None
            if not gloss:reason='empty_gloss'
            elif k!=len(gloss):reason='count_capped'
            elif any(g not in vstoi for g in gloss):reason='unknown_gloss'
            else:
                path,score=viterbi(lp[:int(ol[j]),j],[vstoi[g] for g in gloss])
                if path is None:reason='infeasible_ctc'
                else:
                    centers=[np.flatnonzero(path==2*q+1).mean() for q in range(k)]
                    candidate=[int(round(2*(a+b))) for a,b in zip(centers,centers[1:])]
                    bound=bounded_partition(candidate,n,k)
                    edited+=sum(a!=b for a,b in zip(candidate,bound[1:-1]))
                    occupied.extend([int(np.sum(path==2*q+1)) for q in range(k)])
                    scores.append(score/len(path))
            counts[reason or 'aligned']=counts.get(reason or 'aligned',0)+1
            records[int(i)]={'clip_id':e.sequence,'frames':n,'gloss_count':len(gloss),'K':k,
                'uniform':uniform(n,k),'ctc':bound,'fallback':reason,'viterbi_log_probability':score}
        if start%1024==0:print('TRAIN_ALIGNMENT',start,len(train),flush=True)
    filename.write_text(json.dumps(records,ensure_ascii=False)+'\n')
    (ROOT/'alignment_audit.json').write_text(json.dumps({'TRAIN_clips':len(train),'counts':counts,
        'boundaries_adjusted_for_minimum_two_frames':edited,'mean_token_occupancy_downsampled':float(np.mean(occupied)),
        'mean_viterbi_logp_per_step':float(np.mean(scores)),'annotated_boundaries':False,
        'DEV_poses_or_glosses_aligned':False,'reader_checkpoint_selected_on_DEV_previously':True},indent=2)+'\n')
    return records


def local_signatures(canonical, boundaries):
    """Prefix/Chen computation; no Python loop over joints or path increments."""
    b,t,_=canonical.shape;x=canonical.reshape(b,t,40,6)
    origin=x-x[:,:1];d=x[:,1:]-x[:,:-1]
    indices=torch.triu_indices(6,6,1,device=x.device);i,j=indices
    areas=.5*(origin[:,:-1,:,i]*d[:,:,:,j]-origin[:,:-1,:,j]*d[:,:,:,i])
    cumulative=F.pad(areas.cumsum(1),(0,0,0,0,1,0))
    endpoints=[];all_areas=[];owners=[]
    for batch,partition in enumerate(boundaries):
        starts=torch.tensor(partition[:-1],device=x.device);last=torch.tensor(partition[1:],device=x.device)-1
        first=origin[batch,last]-origin[batch,starts]
        anchor=origin[batch,starts]
        area=cumulative[batch,last]-cumulative[batch,starts]-.5*(anchor[:,:,i]*first[:,:,j]-anchor[:,:,j]*first[:,:,i])
        endpoints.append(first);all_areas.append(area);owners.extend([batch]*len(starts))
    return torch.cat(endpoints),torch.cat(all_areas),torch.tensor(owners,device=x.device)


def equal_clip_mean(elements,owners,count):
    persegment=elements.flatten(1).mean(1)
    numerator=torch.zeros(count,device=elements.device).scatter_add(0,owners,persegment)
    denominator=torch.bincount(owners,minlength=count)
    return (numerator/denominator).mean()


@torch.no_grad()
def scales(train,partitions):
    filename=ROOT/'local_scales.npz'
    if filename.exists():
        with np.load(filename) as data:return {k:torch.tensor(data[k].copy(),device='cuda') for k in data.files}
    endpoints=[];areas=[];count=0
    for start in range(0,len(train),64):
        ids=list(range(start,min(start+64,len(train))));n=max(len(train[i].sgn) for i in ids)
        xs=torch.zeros(len(ids),n,240,device='cuda')
        for j,i in enumerate(ids):xs[j,:len(train[i].sgn)]=torch.from_numpy(previous.project(train[i].sgn.numpy())).cuda()
        for kind in ['uniform','ctc']:
            first,area,_=local_signatures(xs,[partitions[i][kind] for i in ids])
            endpoints.append(first.cpu().numpy());areas.append(area.cpu().numpy());count+=len(first)
    result={'endpoint':np.maximum(np.concatenate(endpoints).std(0),.05),
            'area':np.maximum(np.concatenate(areas).std(0),.01)}
    np.savez(filename,**result)
    (ROOT/'scale_audit.json').write_text(json.dumps({'TRAIN_clips':len(train),'pooled_segments':count,
        'same_scale_for_uniform_and_ctc':True,'no_DEV_used':True,
        'features':{k:{'shape':list(v.shape),'minimum':float(v.min()),'maximum':float(v.max())} for k,v in result.items()}},indent=2)+'\n')
    return {k:torch.tensor(v,device='cuda') for k,v in result.items()}


def fit(model,train,dev,stoi,mean,std,alpha,scale,partitions,name):
    previous.seed(42);optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda');history=[];stochastic=[]
    for epoch in range(5):
        started=time.time();model.train();order=np.random.permutation(len(train));ordered=[]
        for pos in range(0,len(order),256):ordered.extend(sorted(order[pos:pos+256],key=lambda i:len(train[i].sgn)))
        groups=[ordered[pos:pos+64] for pos in range(0,len(ordered),64)];random.shuffle(groups)
        losses=[];extras=[];used=0;segment_count=0
        for ids in groups:
            x,text,length=previous.pack(train,ids,stoi,mean,std)
            drop=torch.rand(len(ids),device='cuda')<.1;text=text.clone();text[drop]=0;text[drop,0]=2
            t=torch.randint(0,1000,(len(ids),),device='cuda');noise=torch.randn_like(x);a=alpha[t][:,None,None]
            prediction=model(a.sqrt()*x+(1-a).sqrt()*noise,t,text,length)
            mask=torch.arange(x.shape[1],device='cuda')[None]<length[:,None]
            mse=((prediction-x).square()*mask[:,:,None]).sum()/(mask.sum()*240)
            extra=torch.zeros((),device='cuda')
            if epoch==0:stochastic.append({'ids':list(map(int,ids)),'t':t.cpu().tolist(),'drop':drop.cpu().tolist(),
                'noise_sum':float(noise.double().sum()),'noise_squared_sum':float(noise.double().square().sum())})
            chosen=(t<=800).nonzero().flatten()[:16]
            if len(chosen):
                physical=previous.physical(prediction[chosen]*st+mt);target=x[chosen]*st+mt
                kind='ctc' if '_ctc_' in name else 'uniform'
                bounds=[partitions[ids[j]][kind] for j in chosen.cpu().tolist()]
                pf,pa,owners=local_signatures(physical,bounds);gf,ga,_=local_signatures(target,bounds)
                endpoint=equal_clip_mean(F.smooth_l1_loss((pf-gf)/scale['endpoint'],torch.zeros_like(pf),reduction='none'),owners,len(chosen))
                if name.endswith('endpoint'):extra=endpoint
                else:
                    area=equal_clip_mean(F.smooth_l1_loss((pa-ga)/scale['area'],torch.zeros_like(pa),reduction='none'),owners,len(chosen))
                    extra=(endpoint+area)/2
                used+=len(chosen);segment_count+=len(pf)
            loss=mse+.02*extra;assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            losses.append(float(mse.detach()));extras.append(float(extra.detach()))
        val=previous.validation(model,dev,stoi,mean,std,alpha)
        record={'epoch':epoch+1,'train_mse':float(np.mean(losses)),'path_term':float(np.mean(extras)),
                'auxiliary_examples':used,'auxiliary_segments':segment_count,'dev_fixed_noise_mse':val,'seconds':time.time()-started}
        history.append(record);(ROOT/(name+'_history.json')).write_text(json.dumps(history,indent=2)+'\n')
        print('LOCAL_TRAIN',name,json.dumps(record),flush=True)
    torch.save({'state_dict':model.state_dict(),'epoch':5,'selection':'last epoch'},ROOT/(name+'.pt'))
    (ROOT/(name+'_stochastic_audit.json')).write_text(json.dumps(stochastic,indent=2)+'\n')


def main():
    assert not (ROOT/'manifest_sha256.json').exists(),'frozen report'
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    audit=json.loads((ROOT/'data_audit.json').read_text());train=previous.dataset('train');dev=previous.dataset('dev')
    assert [e.sequence for e in train]==audit['train_ids'] and [e.sequence for e in dev]==audit['dev_ids']
    stoi={w:i for i,w in enumerate(audit['vocab'])};names=[e['clip_id'] for e in audit['selection']]
    with np.load(ROOT/'normalization.npz') as data:mean=data['mean'].copy();std=data['std'].copy()
    partitions=align_train(train);scale=scales(train,partitions);alpha=previous.alpha_schedule()
    shared=torch.load(previous.ROOT/'shared_best.pt',map_location='cpu',weights_only=False)['state_dict'];models={}
    for name in ARMS:
        model=previous.Denoiser(len(stoi)).cuda();model.load_state_dict(shared,strict=True)
        if not (ROOT/(name+'.pt')).exists():fit(model,train,dev,stoi,mean,std,alpha,scale,partitions,name)
        model.load_state_dict(torch.load(ROOT/(name+'.pt'),map_location='cpu',weights_only=False)['state_dict'],strict=True)
        models[name]=model
    for name in CONTROLS:
        for suffix in ['_scores.json','_poses.npz','.pt','_history.json','_stochastic_audit.json']:
            shutil.copy2(HISTORICAL/(name+suffix),ROOT/(name+suffix))
    scorer.ROOT=ROOT;scorer.ARMS=ARMS
    teacher=previous.make_back_translation_model(previous.BASE/'checkpoints/PHOENIX-2014T')
    scorer.evaluate(models,dev,names,stoi,mean,std,alpha,teacher,shuffled_arm='local_ctc_area')
    print('LOCAL_COMPLETE',flush=True)


if __name__=='__main__':main()
