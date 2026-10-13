"""Amortized rotation-equivalent layers on real How2Sign and CSL TRAIN motion.

A source caption is an input by design; this extends coverage, not spontaneous
exploitation. Native reader word embeddings are frozen, source encoder fitted.
"""
import copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_signspark_geometry_dev import _build_split,load_config,project,make_back_translation_model,back_translate,score_translations
from run_signspark_gauge_attack import SignCollator,Batch,PAD_TOKEN
from run_reviewer_attack_ablation import transformed
from run_evaluation_heldout import geometry
from audit_reviewed_manuscript_evidence import metric
from verify_signspark_studies import sha
import run_canonical_readers as readers
import run_physical_semantic_ddpm as base
ROOT=Path('reports/cvpr_real_motion_layers_20261012');SEEDS=[42,123,777];BOUND=.35
ARMS={'text_raw':SEEDS,'motion_raw':[42],'text_canonical':[42]}
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
class RealLayer(nn.Module):
 def __init__(self,teacher,pad):
  super().__init__();self.embedding=copy.deepcopy(teacher.txt_embed.lut);self.pad=pad
  for p in self.embedding.parameters():p.requires_grad_(False)
  self.text_projection=nn.Linear(self.embedding.embedding_dim,128);self.token_pos=nn.Embedding(64,128)
  self.text=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,0.,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
  self.input=nn.Conv1d(240,128,1);self.position=nn.Embedding(300,128);self.attention=nn.MultiheadAttention(128,4,batch_first=True,dropout=0.)
  self.convs=nn.ModuleList([nn.Conv1d(128,128,3,padding=d,dilation=d) for d in [1,2,4]]);self.norms=nn.ModuleList([nn.GroupNorm(8,128) for _ in range(3)]);self.output=nn.Conv1d(128,120,1)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias);mask=torch.ones(1,1,40,1);mask[:,:,10:25]=0;self.register_buffer('joint_mask',mask)
 def forward(self,pose,tokens,lengths):
  padding=tokens==self.pad
  with torch.no_grad():z=self.embedding(tokens)
  z=self.text_projection(z)+self.token_pos(torch.arange(tokens.shape[1],device=pose.device))[None];z=self.text(z,src_key_padding_mask=padding)
  valid=torch.arange(pose.shape[1],device=pose.device)[None]<lengths[:,None];h=self.input(pose.transpose(1,2))+self.position(torch.arange(pose.shape[1],device=pose.device)).T[None]
  context,_=self.attention(h.transpose(1,2),z,z,key_padding_mask=padding,need_weights=False);h=(h+context.transpose(1,2))*valid[:,None]
  for conv,norm in zip(self.convs,self.norms):h=(h+conv(F.silu(norm(h))))*valid[:,None]
  c=BOUND*torch.tanh(self.output(h).transpose(1,2).reshape(*pose.shape[:-1],40,3))*self.joint_mask*valid[:,:,None,None]
  return transformed(pose.reshape(*pose.shape[:-1],40,6),c).reshape_as(pose),c

def setup():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 inputs=[readers.ROOT/'manifest_sha256.json']+[q/'best.ckpt' for q in readers.CHECKPOINTS.values()]
 p={'corpora':list(readers.CHECKPOINTS),'arms':ARMS,'epochs':20,'bound':BOUND,'batch_size':64,'optimizer':'AdamW lr.001 wd.0001 clip1',
 'source':'native real TRAIN motion, not generated; raw reoptimized representation retained',
 'coefficient_architecture':'same 128-wide temporal Conv/MHA stack as PHOENIX layer, head zero initialized, left joints10:25 fixed',
 'text_encoder':'native frozen public-reader word embeddings, trained linear512to128 and two-layer128 Transformer; no compatible generator text checkpoint available',
 'text_is_source_input':True,'interpretation':'designed caption-conditioned amortized attack on real motions; not spontaneous ordinary generator exploitation',
 'selection':'last epoch; no score-based selection','splits':'TRAIN fit, full DEV confirmation','no_TEST_use':True,
 'checks':'TRAIN/DEV disjoint clip IDs, canonical hypotheses, matrix reconstruction, native word/character BLEU',
 'inputs':{str(p):sha(p) for p in inputs},'code_sha256':sha(__file__)}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==p
 else:write('protocol.json',p)

def data(corpus):
 dc=dict(load_config(readers.CHECKPOINTS[corpus]/'config.yaml')['data']);dc.pop('pseudo_gloss_dir',None);dc.update(index_cache=False,max_sgn_length=300,base_path=str((readers.ROOT/'data/lmdb').resolve()))
 train=_build_split(dc,'train',dc.get('max_sent_length'));training=[train[i] for i in range(len(train))]
 if corpus=='How2Sign':dc['base_path']=str(Path('reports/cvpr_signspark_baseline_20261010/data/lmdb').resolve())
 dv=_build_split(dc,'dev',None);dev=[dv[i] for i in range(len(dv))];old=read(readers.ROOT/corpus/'data_audit.json');assert [e.sequence for e in training]==old['train_ids'] and [e.sequence for e in dev]==old['dev_ids'];assert not set(old['train_ids'])&set(old['dev_ids'])
 seen={' '.join(e.txt) for e in training};write(corpus+'/data_audit.json',{'TRAIN_clips':len(training),'DEV_clips':len(dev),'TRAIN_DEV_ID_overlap':0,'train_ids':old['train_ids'],'dev_ids':old['dev_ids'],'DEV_exact_caption_seen_in_TRAIN':[' '.join(e.txt) in seen for e in dev],'source_motion_is_real':True})
 print('REAL_LAYER_DATA',corpus,len(training),len(dev),flush=True);return training,dev,dc

def score(teacher,xs,dev,dc):
 refs=[' '.join(e.txt) for e in dev];hyps=back_translate(teacher,[torch.from_numpy(x) for x in xs],batch_size=32);scores=score_translations(refs,hyps,dc['version'],dc['level']);assert abs(metric(refs,hyps,char=corpus_char(dc))-scores['bleu']['bleu4'])<1e-10
 return {'clip_ids':[e.sequence for e in dev],'references':refs,'hypotheses':hyps,'scores':scores}
def corpus_char(dc):return dc['version']=='csldaily'
def tokens(batch,pad,arm):
 t=batch.txt_input[:,:64].clone()
 if arm=='motion_raw':first=t[:,0].clone();t.fill_(pad);t[:,0]=first
 return t

@torch.no_grad()
def evaluate(model,teacher,dev,dc,collator,pad,corpus,seed,arm):
 model.eval();xs=[];ys=[];cs=[]
 for start in range(0,len(dev),32):
  batch=Batch(collator(dev[start:start+32]),txt_pad_index=pad,sgn_dim=240,use_cuda=True);y,c=model(batch.sgn,tokens(batch,pad,arm),batch.sgn_lengths.to(batch.sgn.device))
  for j,n in enumerate(batch.sgn_lengths.tolist()):xs.append(batch.sgn[j,:n].cpu().numpy().copy());ys.append(y[j,:n].cpu().numpy().copy());cs.append(c[j,:n].cpu().numpy().copy())
 raw=score(teacher,ys,dev,dc);canonical=score(teacher,[project(y) for y in ys],dev,dc);baseline=read(ROOT/corpus/'baseline_canonical.json');assert canonical['hypotheses']==baseline['hypotheses']
 rms=[float(np.sqrt(np.mean((x-project(x))**2))) for x in ys];delta=[float(np.sqrt(np.mean((x-y)**2))) for x,y in zip(xs,ys)]
 result={'corpus':corpus,'seed':seed,'arm':arm,'clips':len(dev),'raw':raw,'canonical':canonical,'source_is_real_motion':True,'canonical_hypothesis_changes':0,
 'mean_RMS_to_canonical':float(np.mean(rms)),'mean_RMS_change_from_source':float(np.mean(delta)),**geometry(xs,ys)}
 folder=ROOT/corpus/'evaluation';folder.mkdir(exist_ok=True);tag=f'seed{seed}_{arm}';write(f'{corpus}/evaluation/{tag}.json',result)
 np.savez_compressed(folder/(tag+'_poses.npz'),**{f'{k}{i:04d}':v for k,arr in [('attacked',ys),('coefficient',cs)] for i,v in enumerate(arr)})
 print('REAL_LAYER_SCORED',corpus,seed,arm,raw['scores']['bleu']['bleu4'],canonical['scores']['bleu']['bleu4'],result['mean_RMS_change_from_source'],flush=True)


def train():
 setup()
 for corpus,checkpoint in readers.CHECKPOINTS.items():
  training,dev,dc=data(corpus);teacher=make_back_translation_model(checkpoint);teacher.eval();teacher.do_recognition=False
  for p in teacher.parameters():p.requires_grad_(False)
  collator=SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[PAD_TOKEN]
  for kind,poses in [('raw',[e.sgn.numpy() for e in dev]),('canonical',[project(e.sgn.numpy()) for e in dev])]:write(corpus+'/baseline_'+kind+'.json',score(teacher,poses,dev,dc))
  matching={}
  for arm,seeds in ARMS.items():
   for seed in seeds:
    base.seed(seed);model=RealLayer(teacher,pad).cuda();folder=ROOT/corpus/f'training/seed{seed}/{arm}';folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/'layer.pt'
    if not checkpoint.exists():
     opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001);history=[];orders=[];model.train()
     for epoch in range(20):
      started=time.time();order=np.random.permutation(len(training));ordered=[]
      for start in range(0,len(order),256):ordered.extend(sorted(order[start:start+256],key=lambda i:len(training[i].sgn)))
      groups=[ordered[p:p+64] for p in range(0,len(ordered),64)];random.shuffle(groups);lossvalues=[]
      for indices in groups:
       if epoch==0:orders.append(list(map(int,indices)))
       batch=Batch(collator([training[i] for i in indices]),txt_pad_index=pad,sgn_dim=240,use_cuda=True);y,c=model(batch.sgn,tokens(batch,pad,arm),batch.sgn_lengths.to(batch.sgn.device));consumed=base.physical(y) if arm=='text_canonical' else y
       out,_=teacher.forward(consumed,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask);logits=out[0];lt=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad,reduction='none').reshape(batch.txt.shape);loss=(lt.sum(1)/(batch.txt!=pad).sum(1)).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();lossvalues.append(float(loss.detach()))
      r={'epoch':epoch+1,'TRAIN_per_clip_CE':float(np.mean(lossvalues)),'seconds':time.time()-started};history.append(r);write(str(folder.relative_to(ROOT)/'history.json'),history);print('REAL_LAYER_EPOCH',corpus,seed,arm,json.dumps(r),flush=True)
     torch.save({'state_dict':model.state_dict(),'seed':seed,'arm':arm,'epoch':20,'selection':'last epoch','trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)},checkpoint);write(str(folder.relative_to(ROOT)/'first_epoch_batches.json'),orders)
    order=read(folder/'first_epoch_batches.json')
    if seed in matching:assert matching[seed]==order,'batches differ across arms'
    else:matching[seed]=order
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    if not (ROOT/corpus/'evaluation'/f'seed{seed}_{arm}.json').exists():evaluate(model,teacher,dev,dc,collator,pad,corpus,seed,arm)
    del model;torch.cuda.empty_cache()
  del teacher,training,dev;torch.cuda.empty_cache()
 write('complete.json',{'corpora':list(readers.CHECKPOINTS),'fits':10,'all_first_epoch_batches_matched_within_seed':True,'no_TEST_use':True})
if __name__=='__main__':train()
