"""Independent Conv/BiGRU/CTC reader, TRAIN fit and DEV audit only."""
import json, time, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from run_signspark_geometry_dev import project, matrices, _build_split
from run_signspark_gauge_attack import transform
from scipy.spatial.transform import Rotation

ROOT=Path('reports/cvpr_motion_consistency_confirmation_20261010')
OUT=ROOT/'ctc_reader';OUT.mkdir(exist_ok=True)

class Reader(nn.Module):
    def __init__(self,vocab_size,mean,std):
        super().__init__();self.register_buffer('mean',torch.tensor(mean));self.register_buffer('std',torch.tensor(std))
        self.conv=nn.Sequential(nn.Conv1d(240,128,5,2,2),nn.ReLU(),nn.Conv1d(128,128,5,2,2),nn.ReLU())
        self.gru=nn.GRU(128,256,2,batch_first=True,bidirectional=True,dropout=.1)
        self.head=nn.Linear(512,vocab_size)
        with torch.no_grad():self.head.bias[0]=-2
    def forward(self,x,lengths):
        # Re-mask after normalization; padded frames cannot encode mean offsets.
        mask=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        z=((x-self.mean)/self.std)*mask[:,:,None]
        z=self.conv(z.transpose(1,2)).transpose(1,2);outlen=(lengths+3)//4
        packed=pack_padded_sequence(z,outlen.cpu(),batch_first=True,enforce_sorted=False)
        z,_=self.gru(packed);z,_=pad_packed_sequence(z,batch_first=True,total_length=z.shape[1] if isinstance(z,torch.Tensor) else (x.shape[1]+3)//4)
        return self.head(z).log_softmax(-1).transpose(0,1),outlen

def batch(xs,ids):
    n=max(len(xs[i]) for i in ids);x=np.zeros((len(ids),n,240),np.float32)
    for j,i in enumerate(ids):x[j,:len(xs[i])]=xs[i]
    return torch.from_numpy(x).cuda(),torch.tensor([len(xs[i]) for i in ids],device='cuda')

def distance(ref,hyp):
    prev=list(range(len(hyp)+1))
    for i,r in enumerate(ref,1):
        row=[i]
        for j,h in enumerate(hyp,1):row.append(min(prev[j]+1,row[-1]+1,prev[j-1]+(r!=h)))
        prev=row
    return prev[-1]

def metrics(refs,hyps):
    err=[distance(r.split(),h.split()) for r,h in zip(refs,hyps)];den=[len(r.split()) for r in refs]
    return {'wer':100*sum(err)/sum(den),'errors':err,'reference_lengths':den}

@torch.no_grad()
def decode(model,xs,vocab):
    model.eval();out=['']*len(xs);order=np.argsort(list(map(len,xs)),kind='stable')
    for start in range(0,len(xs),64):
        ids=order[start:start+64];x,l=batch(xs,ids);logp,ol=model(x,l);tokens=logp.argmax(-1).T.cpu().tolist()
        for i,row,n in zip(ids,tokens,ol.tolist()):
            compact=[];prev=-1
            for token in row[:n]:
                if token and token!=prev:compact.append(vocab[token])
                prev=token
            out[i]=' '.join(compact)
    return out

def read(split):
    cfg={'dataset':'PHOENIX-2014T','base_path':str((ROOT/'data/lmdb').resolve()),'index_cache':False,'max_sgn_length':300}
    data=_build_split(cfg,split,None);examples=[data[i] for i in range(len(data))]
    return [e.sgn.numpy().copy() for e in examples],[' '.join(e.gls) for e in examples],[e.sequence for e in examples]

def train():
    xs,refs,ids=read('train');dv,dr,di=read('dev')
    vocab=['<blank>','<unknown>']+sorted(set(t for r in refs for t in r.split())-{'<blank>','<unknown>'});stoi={w:i for i,w in enumerate(vocab)}
    targets=[torch.tensor([stoi.get(w,1) for w in r.split()],dtype=torch.long) for r in refs]
    mean=np.mean(np.concatenate(xs),axis=0);std=np.maximum(np.std(np.concatenate(xs),axis=0),.05)
    model=Reader(len(vocab),mean,std).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40,eta_min=.0001);lossfn=nn.CTCLoss(blank=0,zero_infinity=True)
    audit={'train_clips':len(xs),'dev_clips':len(dv),'parameters':sum(p.numel() for p in model.parameters()),'vocab':vocab,'train_ids':ids,'dev_ids':di,'dev_references':dr,'dev_unknown_tokens':sum(w not in stoi for r in dr for w in r.split()),'torch':torch.__version__}
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False));best=1e9;history=[]
    for epoch in range(40):
        model.train();order=np.random.permutation(len(xs));ordered=[]
        for start in range(0,len(xs),256):ordered.extend(sorted(order[start:start+256],key=lambda i:len(xs[i])))
        groups=[ordered[s:s+64] for s in range(0,len(xs),64)];random.shuffle(groups);losses=[];starttime=time.time()
        for inds in groups:
            x,l=batch(xs,inds);logp,ol=model(x,l);t=torch.cat([targets[i] for i in inds]).cuda();tl=torch.tensor([len(targets[i]) for i in inds],device='cuda')
            loss=lossfn(logp,t,ol,tl);assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        hyps=decode(model,dv,vocab);score=metrics(dr,hyps)['wer'];record={'epoch':epoch+1,'loss':float(np.mean(losses)),'dev_wer':score,'lr':opt.param_groups[0]['lr'],'seconds':time.time()-starttime};history.append(record)
        if score<best:
            best=score;torch.save({'state_dict':model.state_dict(),'vocab':vocab,'epoch':epoch+1,'dev_wer':score},OUT/'best.pt')
        schedule.step();(OUT/'training_history.json').write_text(json.dumps(history,indent=2));print('TRAIN',json.dumps(record),flush=True)
    return model

def audit():
    ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);vocab=ck['vocab'];stoi={w:i for i,w in enumerate(vocab)};state=ck['state_dict'];model=Reader(len(vocab),state['mean'].numpy(),state['std'].numpy()).cuda();model.load_state_dict(state);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    xs,refs,ids=read('dev');xs=[np.concatenate((x[:,:60],project(x)[:,60:150],x[:,150:]),axis=1) for x in xs]
    original=decode(model,xs,vocab);can=decode(model,[project(x) for x in xs],vocab);results={}
    def save(name,ys):
        h=decode(model,ys,vocab);m=metrics(refs,h);m.update(hypotheses=h,changed_from_original=sum(a!=b for a,b in zip(h,original)))
        results[name]=m;(OUT/(name+'.json')).write_text(json.dumps(m,indent=2,ensure_ascii=False));print('AUDIT',name,m['wer'],m['changed_from_original'],flush=True)
    save('original',xs);save('canonical',[project(x) for x in xs]);save('reverse_time',[x[::-1].copy() for x in xs])
    yh=[]
    for x in xs:
        y=x.copy();y[:,60:]=np.tile([1,0,0,0,1,0],30);yh.append(y)
    save('identity_hands',yh)
    for name,scale,shear in [('scale09',.9,0),('scale11',1.1,0),('shear01',1,.1)]:
        ys=[]
        for x in xs:
            y=x.reshape(-1,40,6).copy();mask=np.ones(40,bool);mask[10:25]=False;a=y[:,mask,:3].copy();y[:,mask,:3]*=scale;y[:,mask,3:]=scale*y[:,mask,3:]+shear*a;ys.append(y.reshape(x.shape))
        save(name,ys)
    prior=Path('reports/cvpr_signspark_gauge_full_dev_20261010/PHOENIX-2014T')
    oldids=[e['clip_id'] for e in json.loads((prior/'selection.json').read_text())];assert ids==oldids
    for cond in ['correct_caption','rolled_wrong_caption']:
        with np.load(prior/(cond+'_poses.npz')) as z:ys=[z[f'attacked{i:04d}'].copy() for i in range(len(xs))]
        save('transfer_transformer_'+cond,ys)
    # Disable cuDNN so eval-mode GRU permits deterministic input gradients.
    torch.backends.cudnn.enabled=False;jointmask=torch.ones(1,1,40,1,device='cuda');jointmask[:,:,10:25]=0;lossfn=nn.CTCLoss(blank=0,zero_infinity=True)
    attacksummary={}
    for cond in ['correct_gloss','rolled_wrong_gloss']:
        tr=refs if cond=='correct_gloss' else refs[1:]+refs[:1];ys=[];coefs=[];logs=[];infeasible=0
        for start in range(0,len(xs),32):
            inds=list(range(start,min(start+32,len(xs))));x,l=batch(xs,inds);d=x.detach().reshape(len(inds),-1,40,6);u=torch.zeros((*d.shape[:-1],3),device='cuda',requires_grad=True);opt=torch.optim.Adam([u],lr=.1)
            ts=[[stoi.get(w,1) for w in tr[i].split()] for i in inds];tl=torch.tensor(list(map(len,ts)),device='cuda');t=torch.tensor([j for r in ts for j in r],device='cuda');losses=[]
            infeasible+=sum(len(r)+sum(a==b for a,b in zip(r,r[1:]))>(len(xs[i])+3)//4 for i,r in zip(inds,ts))
            for step in range(80):
                y=transform(d,u*jointmask).reshape(x.shape);lp,ol=model(y,l);loss=lossfn(lp,t,ol,tl);assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert torch.isfinite(u.grad).all();opt.step();losses.append(float(loss.detach()))
            y=transform(d,u*jointmask).detach().cpu().numpy().reshape(x.shape);coef=(u*jointmask).detach().cpu().numpy()
            for j,i in enumerate(inds):ys.append(y[j,:len(xs[i])].copy());coefs.append(coef[j,:len(xs[i])].copy())
            logs.append({'batch_start':start,'losses':losses});print('ATTACK',cond,start,round(losses[0],3),round(losses[-1],3),flush=True)
        h=decode(model,ys,vocab);ch=decode(model,[project(y) for y in ys],vocab);maxangle=0
        for x,y in zip(xs,ys):
            assert np.array_equal(x[:,60:150],y[:,60:150]);assert np.isfinite(y).all()
            rot=matrices(x).transpose(0,1,3,2)@matrices(y);angle=np.degrees(Rotation.from_matrix(rot.reshape(-1,3,3)).magnitude()).max();maxangle=max(maxangle,float(angle))
        result={'actual_reference':metrics(refs,h),'target':metrics(tr,h),'baseline_target':metrics(tr,original),'canonical_actual':metrics(refs,ch),'canonical_changed_sentences':sum(a!=b for a,b in zip(can,ch)),'raw_changed_sentences':sum(a!=b for a,b in zip(original,h)),'max_angle_deg':maxangle,'ctc_infeasible_targets':infeasible,'targets':tr,'hypotheses':h,'canonical_hypotheses':ch,'optimization_logs':logs}
        assert maxangle<.01;(OUT/(cond+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False));np.savez_compressed(OUT/(cond+'_poses.npz'),**{f'{p}{i:04d}':x for p,arr in [('attacked',ys),('gauge',coefs)] for i,x in enumerate(arr)})
        attacksummary[cond]={k:v for k,v in result.items() if k not in ['targets','hypotheses','canonical_hypotheses','optimization_logs']};print('ATTACK_COMPLETE',cond,json.dumps(attacksummary[cond]),flush=True)
    summary={'best_epoch':ck['epoch'],'original_best_dev_wer':ck['dev_wer'],'controls':{k:{'wer':v['wer'],'changed_sentences':v['changed_from_original']} for k,v in results.items()},'attacks':attacksummary};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('CTC_COMPLETE',flush=True)

if __name__=='__main__':
    torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cuda.matmul.allow_tf32=False
    if not (OUT/'best.pt').exists():train()
    audit()
