"""DDPM training with physical endpoint/area signatures; no semantic teacher."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import run_physical_semantic_ddpm as previous
from run_fixed_prefix_diffusion import evaluate as score
import run_fixed_prefix_diffusion as scorer

ROOT=Path('reports/cvpr_motion_path_diffusion_20261010')
OLD=Path('reports/cvpr_physical_semantic_ddpm_20261010')
ARMS=['geometry_replay','physical_velocity','physical_endpoint','physical_area','physical_area_time']


def signatures(canonical, lengths, with_time=False):
    b,t,_=canonical.shape
    path=canonical.reshape(b,t,40,6)
    if with_time:
        clock=torch.arange(t,device=path.device,dtype=path.dtype)[None]/(lengths-1).clamp_min(1)[:,None]
        path=torch.cat([clock[:,:,None,None].expand(-1,-1,40,1),path],-1)
    mask=torch.arange(t-1,device=path.device)[None]<(lengths-1)[:,None]
    delta=(path[:,1:]-path[:,:-1])*mask[:,:,None,None]
    prefix=delta.cumsum(1)-delta
    second=torch.einsum('btjd,btje->bjde',prefix,delta)
    area=.5*(second-second.transpose(-1,-2))
    indices=torch.triu_indices(path.shape[-1],path.shape[-1],1,device=path.device)
    first=delta.sum(1)[...,-6:]
    return first,area[...,indices[0],indices[1]]


@torch.no_grad()
def scales(train,stoi,mean,std):
    path=ROOT/'path_scales.npz'
    if path.exists():
        with np.load(path) as data:return {k:torch.as_tensor(data[k].copy(),device='cuda',dtype=torch.float32) for k in data.files}
    mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda')
    first=[];area=[];time_area=[];total=torch.zeros(240,dtype=torch.float64,device='cuda')
    squared=torch.zeros_like(total);count=0
    for start in range(0,len(train),64):
        ids=list(range(start,min(start+64,len(train))))
        x,_,length=previous.pack(train,ids,stoi,mean,std);physical=x*st+mt
        s1,a=signatures(physical,length);_,at=signatures(physical,length,True)
        first.append(s1.cpu().numpy());area.append(a.cpu().numpy());time_area.append(at.cpu().numpy())
        delta=(physical[:,1:]-physical[:,:-1]).double()
        mask=torch.arange(delta.shape[1],device='cuda')[None]<(length-1)[:,None]
        total+=(delta*mask[:,:,None]).sum((0,1));squared+=(delta.square()*mask[:,:,None]).sum((0,1))
        count+=int(mask.sum())
    result={'endpoint':np.maximum(np.concatenate(first).std(0),.05),
        'area':np.maximum(np.concatenate(area).std(0),.01),
        'area_time':np.maximum(np.concatenate(time_area).std(0),.01),
        'velocity':np.maximum(torch.sqrt((squared/count-(total/count).square()).clamp_min(0)).cpu().numpy(),.01)}
    np.savez(path,**result)
    (ROOT/'scale_audit.json').write_text(json.dumps({'TRAIN_clips':len(train),'valid_differences':count,
        'features':{k:{'shape':list(v.shape),'minimum':float(v.min()),'maximum':float(v.max())} for k,v in result.items()},
        'no_DEV_used':True},indent=2)+'\n')
    return {k:torch.as_tensor(v,device='cuda',dtype=torch.float32) for k,v in result.items()}


def fit(model,train,dev,stoi,mean,std,alpha,scale,name):
    previous.seed(42);optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda');history=[];stochastic=[]
    for epoch in range(5):
        started=time.time();model.train();order=np.random.permutation(len(train));ordered=[]
        for pos in range(0,len(order),256):ordered.extend(sorted(order[pos:pos+256],key=lambda i:len(train[i].sgn)))
        groups=[ordered[pos:pos+64] for pos in range(0,len(ordered),64)];random.shuffle(groups)
        losses=[];extras=[];used=0
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
            if name!='geometry_replay' and len(chosen):
                physical=previous.physical(prediction[chosen]*st+mt);target=x[chosen]*st+mt;n=length[chosen]
                if name=='physical_velocity':
                    delta=(physical[:,1:]-physical[:,:-1])-(target[:,1:]-target[:,:-1])
                    valid=torch.arange(delta.shape[1],device='cuda')[None]<(n-1)[:,None]
                    elements=F.smooth_l1_loss(delta/scale['velocity'],torch.zeros_like(delta),reduction='none')
                    extra=(elements*valid[:,:,None]).sum()/(valid.sum()*240)
                else:
                    clock=name=='physical_area_time'
                    pf,pa=signatures(physical,n,clock);gf,ga=signatures(target,n,clock)
                    endpoint=F.smooth_l1_loss((pf-gf)/scale['endpoint'],torch.zeros_like(pf))
                    if name=='physical_endpoint':extra=endpoint
                    else:
                        area_scale=scale['area_time' if clock else 'area']
                        area=F.smooth_l1_loss((pa-ga)/area_scale,torch.zeros_like(pa))
                        extra=(endpoint+area)/2
                used+=len(chosen)
            loss=mse+.02*extra;assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            losses.append(float(mse.detach()));extras.append(float(extra.detach()))
        val=previous.validation(model,dev,stoi,mean,std,alpha)
        record={'epoch':epoch+1,'train_mse':float(np.mean(losses)),'path_term':float(np.mean(extras)),
                'auxiliary_examples':used,'dev_fixed_noise_mse':val,'seconds':time.time()-started}
        history.append(record);(ROOT/(name+'_history.json')).write_text(json.dumps(history,indent=2)+'\n')
        print('PATH_TRAIN',name,json.dumps(record),flush=True)
    torch.save({'state_dict':model.state_dict(),'epoch':5,'selection':'last epoch'},ROOT/(name+'.pt'))
    (ROOT/(name+'_stochastic_audit.json')).write_text(json.dumps(stochastic,indent=2)+'\n')


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    audit=json.loads((ROOT/'data_audit.json').read_text());train=previous.dataset('train');dev=previous.dataset('dev')
    assert [e.sequence for e in train]==audit['train_ids'] and [e.sequence for e in dev]==audit['dev_ids']
    stoi={w:i for i,w in enumerate(audit['vocab'])};names=[e['clip_id'] for e in audit['selection']]
    with np.load(ROOT/'normalization.npz') as data:mean=data['mean'].copy();std=data['std'].copy()
    scale=scales(train,stoi,mean,std);alpha=previous.alpha_schedule()
    shared=torch.load(OLD/'shared_best.pt',map_location='cpu',weights_only=False)['state_dict'];models={}
    for name in ARMS:
        model=previous.Denoiser(len(stoi)).cuda();model.load_state_dict(shared,strict=True)
        if not (ROOT/(name+'.pt')).exists():fit(model,train,dev,stoi,mean,std,alpha,scale,name)
        model.load_state_dict(torch.load(ROOT/(name+'.pt'),map_location='cpu',weights_only=False)['state_dict'],strict=True)
        models[name]=model
    # Reuse the exact scoring producer while keeping outputs in this new report.
    scorer.ROOT=ROOT;scorer.ARMS=ARMS
    teacher=previous.make_back_translation_model(previous.BASE/'checkpoints/PHOENIX-2014T')
    score(models,dev,names,stoi,mean,std,alpha,teacher,shuffled_arm='physical_area_time')
    print('PATH_COMPLETE',flush=True)


if __name__=='__main__':main()
