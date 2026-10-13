"""Matched DDPM pilot: semantic supervision on raw vs physical rotations."""
import copy,json,math,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from run_signspark_geometry_dev import _build_split,project,make_back_translation_model,back_translate,score_translations
from run_signspark_gauge_attack import SignCollator,Batch,PAD_TOKEN
from run_motion_ctc_confirmation import Reader,decode,metrics

ROOT=Path('reports/cvpr_physical_semantic_ddpm_20261010')
PREV=Path('reports/cvpr_motion_consistency_confirmation_20261010')
BASE=Path('reports/cvpr_signspark_baseline_20261010')

def physical(x):
    shape=x.shape;d=x.reshape(*shape[:-1],40,6);a=F.normalize(d[...,:3],dim=-1,eps=1e-6);b=d[...,3:];b=F.normalize(b-(a*b).sum(-1,keepdim=True)*a,dim=-1,eps=1e-6);return torch.cat([a,b],-1).reshape(shape)

def timestep_embedding(t,dim=128):
    freq=torch.exp(-math.log(10000)*torch.arange(dim//2,device=t.device)/(dim//2-1));phase=t[:,None]*freq[None,:]
    return torch.cat([phase.sin(),phase.cos()],-1)

class Denoiser(nn.Module):
    def __init__(self,vocab):
        super().__init__();self.token=nn.Embedding(vocab,128,padding_idx=0);self.token_pos=nn.Embedding(64,128)
        layer=nn.TransformerEncoderLayer(128,4,512,.1,batch_first=True,norm_first=True)
        self.text=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.frame_pos=nn.Embedding(300,128);self.time=nn.Sequential(nn.Linear(128,512),nn.SiLU(),nn.Linear(512,128));self.input=nn.Conv1d(240,128,1)
        self.convs=nn.ModuleList([nn.Conv1d(128,128,3,padding=d,dilation=d) for d in [1,2,4,8,16,32]]);self.norms=nn.ModuleList([nn.GroupNorm(8,128) for _ in range(6)]);self.cross=nn.ModuleDict({str(i):nn.MultiheadAttention(128,4,batch_first=True,dropout=.1) for i in [2,5]});self.output=nn.Conv1d(128,240,1)
    def forward(self,x,t,tokens,lengths):
        padding=tokens==0;z=self.token(tokens)+self.token_pos(torch.arange(tokens.shape[1],device=x.device))[None];z=self.text(z,src_key_padding_mask=padding)
        pooled=(z*(~padding)[:,:,None]).sum(1)/(~padding).sum(1)[:,None];cond=self.time(timestep_embedding(t.float()))+pooled
        mask=(torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]).to(x.dtype)[:,None,:];h=self.input(x.transpose(1,2))+self.frame_pos(torch.arange(x.shape[1],device=x.device)).T[None]
        for i,(conv,norm) in enumerate(zip(self.convs,self.norms)):
            residual=conv(F.silu(norm(h)+cond[:,:,None]))
            h=(h+residual)*mask
            if str(i) in self.cross:
                q=h.transpose(1,2);att,_=self.cross[str(i)](q,z,z,key_padding_mask=padding,need_weights=False);h=(h+att.transpose(1,2))*mask
        return self.output(h).transpose(1,2)

def seed(n):torch.manual_seed(n);np.random.seed(n);random.seed(n)

def dataset(split):
    cfg={'dataset':'PHOENIX-2014T','base_path':str((PREV/'data/lmdb').resolve()),'index_cache':False,'max_sgn_length':300}
    d=_build_split(cfg,split,None);return [d[i] for i in range(len(d))]

def pack(examples,ids,stoi,mean,std):
    size=max(len(examples[i].sgn) for i in ids);x=np.zeros((len(ids),size,240),np.float32);tokens=np.zeros((len(ids),64),np.int64);lengths=[]
    for j,i in enumerate(ids):
        e=examples[i];p=project(e.sgn.numpy());x[j,:len(p)]=(p-mean)/std;lengths.append(len(p));s=[stoi.get(w,1) for w in e.txt][:64];s=s or [2];tokens[j,:len(s)]=s
    return torch.from_numpy(x).cuda(),torch.from_numpy(tokens).cuda(),torch.tensor(lengths,device='cuda')

def alpha_schedule():
    t=torch.arange(1001,dtype=torch.float64);a=torch.cos(((t/1000+.008)/1.008)*math.pi/2).square();a=a/a[0];beta=(1-a[1:]/a[:-1]).clamp(max=.999);return torch.cumprod(1-beta,0).float().cuda()

@torch.no_grad()
def validation(model,examples,stoi,mean,std,alpha):
    model.eval();gen=torch.Generator(device='cuda').manual_seed(2027);total=0.;den=0.
    for start in range(0,len(examples),64):
        ids=list(range(start,min(start+64,len(examples))));x,tok,length=pack(examples,ids,stoi,mean,std);t=torch.randint(0,1000,(len(ids),),device='cuda',generator=gen);noise=torch.randn(x.shape,device='cuda',generator=gen);a=alpha[t][:,None,None];noisy=a.sqrt()*x+(1-a).sqrt()*noise;pred=model(noisy,t,tok,length);mask=torch.arange(x.shape[1],device='cuda')[None,:]<length[:,None];total+=float(((pred-x).square()*mask[:,:,None]).sum());den+=int(mask.sum())*240
    return total/den

def save_weights(path,model,metadata):torch.save({'state_dict':model.state_dict(),**metadata},path)

def fit(model,train,dev,stoi,mean,std,alpha,epochs,lr,name,teacher=None):
    seed(42);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001);history=[];best=1e9;shared=name=='shared';mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda')
    if teacher is not None:collator=SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[PAD_TOKEN]
    for epoch in range(epochs):
        started=time.time();model.train();order=np.random.permutation(len(train));ordered=[]
        for pos in range(0,len(order),256):ordered.extend(sorted(order[pos:pos+256],key=lambda i:len(train[i].sgn)))
        groups=[ordered[pos:pos+64] for pos in range(0,len(ordered),64)];random.shuffle(groups);mse_values=[];semantic_values=[]
        for ids in groups:
            x,tok,length=pack(train,ids,stoi,mean,std);drop=torch.rand(len(ids),device='cuda')<.1;tok=tok.clone();tok[drop]=0;tok[drop,0]=2;t=torch.randint(0,1000,(len(ids),),device='cuda');noise=torch.randn_like(x);a=alpha[t][:,None,None];xt=a.sqrt()*x+(1-a).sqrt()*noise;pred=model(xt,t,tok,length);mask=torch.arange(x.shape[1],device='cuda')[None,:]<length[:,None];mse=((pred-x).square()*mask[:,:,None]).sum()/(mask.sum()*240);loss=mse;semantic=torch.zeros((),device='cuda')
            if teacher is not None:
                chosen=(t<=800).nonzero().flatten()[:16]
                if len(chosen):
                    poses=pred[chosen]*st+mt
                    if name=='semantic_physical':poses=physical(poses)
                    losses=[]
                    for wrong in [False,True]:
                        chunk=[]
                        for j in chosen.tolist():
                            e=copy.copy(train[ids[j]])
                            if wrong:e.txt=train[(ids[j]+1)%len(train)].txt
                            chunk.append(e)
                        b=Batch(collator(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True);b.sgn=poses[:,:b.sgn.shape[1]];out,_=teacher.forward(b.sgn,b.sgn_mask,b.sgn_lengths,b.txt_input,b.txt_mask);lp=out[0];ce=F.cross_entropy(lp.reshape(-1,lp.shape[-1]),b.txt.reshape(-1),ignore_index=pad);losses.append(ce)
                    semantic=losses[0]+F.softplus(.5+losses[0]-losses[1]);loss=loss+.02*semantic
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();mse_values.append(float(mse.detach()));semantic_values.append(float(semantic.detach()))
        val=validation(model,dev,stoi,mean,std,alpha);record={'epoch':epoch+1,'train_mse':float(np.mean(mse_values)),'semantic_term':float(np.mean(semantic_values)),'dev_fixed_noise_mse':val,'seconds':time.time()-started};history.append(record)
        if shared and val<best:best=val;save_weights(ROOT/'shared_best.pt',model,{'epoch':epoch+1,'dev_mse':val})
        (ROOT/(name+'_history.json')).write_text(json.dumps(history,indent=2));print('DDPM_TRAIN',name,json.dumps(record),flush=True)
    if not shared:save_weights(ROOT/(name+'.pt'),model,{'epoch':epochs,'selection':'last epoch, no BLEU selection'})

@torch.no_grad()
def sample(model,dev,ids,stoi,mean,std,alpha,shuffled=False):
    model.eval();seed(102);outputs=[];idxglobal={e.sequence:i for i,e in enumerate(dev)};gen=torch.Generator(device='cuda').manual_seed(102);times=np.linspace(999,0,50).round().astype(int);mt=torch.tensor(mean,device='cuda');st=torch.tensor(std,device='cuda')
    for start in range(0,len(ids),16):
        group=ids[start:start+16];x,tok,length=pack(dev,[idxglobal[n] for n in group],stoi,mean,std);null=torch.zeros_like(tok);null[:,0]=2
        if shuffled:
            allnames=ids[1:]+ids[:1];replacement={n:m for n,m in zip(ids,allnames)}
            _,tok,_=pack(dev,[idxglobal[replacement[n]] for n in group],stoi,mean,std)
        xt=torch.randn(x.shape,device='cuda',generator=gen)
        for pos,ti in enumerate(times):
            t=torch.full((len(group),),int(ti),device='cuda');c=model(xt,t,tok,length);u=model(xt,t,null,length);pred=u+3*(c-u);at=alpha[ti];nt=times[pos+1] if pos+1<len(times) else -1;an=alpha[nt] if nt>=0 else torch.tensor(1.,device='cuda');epsilon=(xt-at.sqrt()*pred)/(1-at).sqrt();xt=an.sqrt()*pred+(1-an).sqrt()*epsilon
        pose=(xt*st+mt).cpu().numpy()
        assert np.isfinite(pose).all()
        for j,n in enumerate(length.tolist()):outputs.append(pose[j,:n].copy())
    return outputs

def evaluate(models,dev,ids,stoi,mean,std,alpha,teacher):
    native={e.sequence:e for e in dev};refs=[' '.join(native[n].txt) for n in ids];gloss=[' '.join(native[n].gls) for n in ids];results={};poses={}
    for name,model in models.items():
        xs=sample(model,dev,ids,stoi,mean,std,alpha);poses[name]=xs
        if name=='semantic_physical':poses[name+'_shuffled']=sample(model,dev,ids,stoi,mean,std,alpha,True)
    ck=torch.load(PREV/'ctc_reader/best.pt',map_location='cpu',weights_only=False);ss=ck['state_dict'];reader=Reader(len(ck['vocab']),ss['mean'].numpy(),ss['std'].numpy()).cuda();reader.load_state_dict(ss,strict=True)
    for name,xs in poses.items():
        ys=[project(x) for x in xs];raw=back_translate(teacher,[torch.from_numpy(x) for x in xs],batch_size=32);can=back_translate(teacher,[torch.from_numpy(x) for x in ys],batch_size=32);hr=decode(reader,xs,ck['vocab']);hc=decode(reader,ys,ck['vocab'])
        record={'clips':len(xs),'clip_ids':ids,'references':refs,'raw_hypotheses':raw,'canonical_hypotheses':can,'raw_scores':score_translations(refs,raw,'phoenix_2014_trans','word'),'canonical_scores':score_translations(refs,can,'phoenix_2014_trans','word'),'independent_raw_gloss':metrics(gloss,hr),'independent_canonical_gloss':metrics(gloss,hc),'independent_raw_hypotheses':hr,'independent_canonical_hypotheses':hc,'raw_canonical_changed_sentences':sum(a!=b for a,b in zip(raw,can))}
        (ROOT/(name+'_scores.json')).write_text(json.dumps(record,indent=2,ensure_ascii=False));np.savez_compressed(ROOT/(name+'_poses.npz'),**{f'clip{i:04d}':x for i,x in enumerate(xs)});results[name]={k:v for k,v in record.items() if k not in ['clip_ids','references','raw_hypotheses','canonical_hypotheses','independent_raw_hypotheses','independent_canonical_hypotheses']};print('DDPM_SCORED',name,record['raw_scores']['bleu']['bleu4'],record['canonical_scores']['bleu']['bleu4'],record['independent_canonical_gloss']['wer'],flush=True)
    (ROOT/'summary.json').write_text(json.dumps(results,indent=2));print('DDPM_COMPLETE',flush=True)

def main():
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;seed(42)
    train=dataset('train');dev=dataset('dev');vocab=['<pad>','<unknown>','<null>']+sorted(set(w for e in train for w in e.txt)-{'<pad>','<unknown>','<null>'});stoi={w:i for i,w in enumerate(vocab)};allx=np.concatenate([project(e.sgn.numpy()) for e in train]);mean=allx.mean(0);std=np.maximum(allx.std(0),.05);del allx;np.savez(ROOT/'normalization.npz',mean=mean,std=std)
    selected=json.loads((PREV/'generated/PHOENIX-2014T/selection.json').read_text());ids=[x['clip_id'] for x in selected];assert len(ids)==128
    audit={'train_clips':len(train),'dev_clips':len(dev),'train_ids':[e.sequence for e in train],'dev_ids':[e.sequence for e in dev],'vocab':vocab,'selection':selected,'train_caption_over_64_tokens':sum(len(e.txt)>64 for e in train),'dev_caption_over_64_tokens':sum(len(e.txt)>64 for e in dev),'torch':torch.__version__};assert not set(audit['train_ids'])&set(audit['dev_ids']);(ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False));alpha=alpha_schedule();model=Denoiser(len(vocab)).cuda();print('DDPM_PARAMETERS',sum(p.numel() for p in model.parameters()),flush=True)
    if not (ROOT/'shared_best.pt').exists():fit(model,train,dev,stoi,mean,std,alpha,20,.0003,'shared')
    shared=torch.load(ROOT/'shared_best.pt',map_location='cpu',weights_only=False)['state_dict'];teacher=make_back_translation_model(BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
    for p in teacher.parameters():p.requires_grad_(False)
    models={}
    for name in ['geometry','semantic_raw','semantic_physical']:
        model=Denoiser(len(vocab)).cuda();model.load_state_dict(shared,strict=True)
        if not (ROOT/(name+'.pt')).exists():fit(model,train,dev,stoi,mean,std,alpha,5,.0001,name,None if name=='geometry' else teacher)
        model.load_state_dict(torch.load(ROOT/(name+'.pt'),map_location='cpu',weights_only=False)['state_dict'],strict=True);models[name]=model
    evaluate(models,dev,ids,stoi,mean,std,alpha,teacher)

if __name__=='__main__':main()
