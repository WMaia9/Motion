"""Train output layers at smaller bounds; compare full noncanonicity profiles."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import rankdata
import run_learned_equivalence as old
from run_reviewer_attack_ablation import transformed
from run_evaluation_heldout import sample_without_reference_pose,geometry
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_learned_budget_20261012');BOUNDS=[.01,.05,.1];SEEDS=[42,123,777]
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
class BoundedLayer(old.OutputLayer):
 def __init__(self,generator,bound):super().__init__(generator);self.bound=bound
 def forward(self,pose,tokens,lengths):
  _,c=super().forward(pose,tokens,lengths);c=c*(self.bound/old.BOUND)
  return transformed(pose.reshape(*pose.shape[:-1],40,6),c).reshape_as(pose),c

def setup():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 audit=read(old.base.ROOT/'data_audit.json');stoi={w:i for i,w in enumerate(audit['vocab'])};g=old.base.Denoiser(len(stoi)).cuda();g.load_state_dict(torch.load(old.GENERATOR,map_location='cpu',weights_only=False)['state_dict'],strict=True);g.eval()
 for p in g.parameters():p.requires_grad_(False)
 return audit,stoi,g

def register():
 inputs=[old.GENERATOR,old.ROOT/'manifest_sha256.json',old.ROOT/'data_audit.json',old.base.BASE/'checkpoints/PHOENIX-2014T/best.ckpt']
 p={'bounds':BOUNDS,'seeds':SEEDS,'epochs':20,'batch_size':64,'optimizer':'AdamW lr.001 wd.0001 clip1','model':'same frozen generator text encoder and same output-layer architecture as bound.35 study',
 'source':'identical cached TRAIN7096 and DEV519 canonicalized generated motions','loss':'mean per-clip raw-reader CE','selection':'last checkpoint, no score selection','baseline_bound035':'existing frozen three-seed results, no fitting changes',
 'natural_comparison':'raw motions from same frozen geometry generator; paired RMS-to-canonical perclip; simple threshold separability via ROC AUC',
 'no_TEST_use':True,'inputs':{str(q):sha(q) for q in inputs},'code_sha256':sha(__file__)}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==p
 else:write('protocol.json',p)

def profiles(xs):
 rms=[];norma=[];normb=[];inner=[]
 for x in xs:
  y=old.base.project(x);rms.append(float(np.sqrt(np.mean((x-y)**2))));d=x.reshape(-1,40,6).astype(float)
  norma.append(float(np.mean(abs(np.linalg.norm(d[...,:3],axis=-1)-1))));normb.append(float(np.mean(abs(np.linalg.norm(d[...,3:],axis=-1)-1))));inner.append(float(np.mean(abs((d[...,:3]*d[...,3:]).sum(-1)))))
 return {'clip_RMS_values':rms,'mean_RMS':float(np.mean(rms)),'median_RMS':float(np.median(rms)),'RMS_quantiles_5_25_75_95':np.percentile(rms,[5,25,75,95]).tolist(),'mean_abs_first_norm_minus1':float(np.mean(norma)),'mean_abs_second_norm_minus1':float(np.mean(normb)),'mean_abs_inner_product':float(np.mean(inner))}

def auc(negative,positive):
 values=np.r_[negative,positive];rank=rankdata(values);n=len(positive);m=len(negative)
 return float((rank[m:].sum()-n*(n+1)/2)/(n*m))

@torch.no_grad()
def evaluate(model,teacher,records,poses,stoi,path,natural):
 refs=[' '.join(e['text']) for e in records];ids=[e['clip_id'] for e in records];ys=[];cs=[]
 model.eval()
 for start in range(0,len(records),32):
  group=records[start:start+32];n=torch.tensor([len(x) for x in poses[start:start+32]],device='cuda');x=torch.zeros(len(group),int(n.max()),240,device='cuda')
  for j,p in enumerate(poses[start:start+32]):x[j,:len(p)]=torch.from_numpy(p).cuda()
  tokens=old.token_tensor(group,stoi);y,c=model(x,tokens,n)
  for j,l in enumerate(n.tolist()):ys.append(y[j,:l].cpu().numpy().copy());cs.append(c[j,:l].cpu().numpy().copy())
 raw=old.score(teacher,ys,refs,ids);canonical=old.score(teacher,[old.base.project(y) for y in ys],refs,ids);baseline=read(ROOT/'evaluation/baseline.json')
 assert canonical['hypotheses']==baseline['hypotheses']
 profile=profiles(ys);separability=auc(natural['clip_RMS_values'],profile['clip_RMS_values'])
 r={'raw':raw,'canonical':canonical,'bound':model.bound,'clips':len(ys),'noncanonicity':profile,'RMS_detector_AUC_high_is_attack':separability,'RMS_detector_best_orientation_AUC':max(separability,1-separability),**geometry(poses,ys)}
 write(str(path.relative_to(ROOT))+'.json',r);np.savez_compressed(str(path)+'_poses.npz',**{f'{kind}{i:04d}':x for kind,arr in [('attacked',ys),('coefficient',cs)] for i,x in enumerate(arr)})
 print('LEARNED_BOUND_SCORED',path.name,raw['scores']['bleu']['bleu4'],canonical['scores']['bleu']['bleu4'],profile['mean_RMS'],separability,flush=True)


def train():
 audit,stoi,g=setup();register();records,poses=old.cache('train');dev,xs=old.cache('dev');teacher=old.base.make_back_translation_model(old.base.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
 for p in teacher.parameters():p.requires_grad_(False)
 collator=old.SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[old.PAD_TOKEN]
 write('evaluation/baseline.json',old.score(teacher,xs,[' '.join(e['text']) for e in dev],[e['clip_id'] for e in dev]))
 with np.load(old.base.ROOT/'normalization.npz') as z:mean,std=z['mean'],z['std']
 naturalxs=sample_without_reference_pose(g,dev,stoi,mean,std,old.base.alpha_schedule());difference=max(float(abs(old.base.project(x)-y).max()) for x,y in zip(naturalxs,xs));assert difference<2e-6
 natural=profiles(naturalxs);write('evaluation/natural_generator.json',{'source':'same frozen local generator and sampling; not real human poses','canonical_cache_replay_max_difference':difference,**natural})
 for seed in SEEDS:
  for bound in BOUNDS:
   old.base.seed(seed);model=BoundedLayer(g,bound).cuda();model.train();folder=ROOT/f'training/seed{seed}/bound{bound:g}';folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/'layer.pt'
   if not checkpoint.exists():
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001);history=[];orders=[]
    for epoch in range(20):
     started=time.time();order=np.random.permutation(len(records));ordered=[]
     for start in range(0,len(order),256):ordered.extend(sorted(order[start:start+256],key=lambda i:len(poses[i])))
     groups=[ordered[p:p+64] for p in range(0,len(ordered),64)];random.shuffle(groups);losses=[]
     for ids in groups:
      if epoch==0:orders.append(list(map(int,ids)))
      batch,tokens=old.pack(records,poses,ids,stoi,collator,pad);y,c=model(batch.sgn,tokens,batch.sgn_lengths.to(batch.sgn.device));out,_=teacher.forward(y,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask);logits=out[0];losses_token=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad,reduction='none').reshape(batch.txt.shape);loss=(losses_token.sum(1)/(batch.txt!=pad).sum(1)).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();losses.append(float(loss.detach()))
     r={'epoch':epoch+1,'TRAIN_CE':float(np.mean(losses)),'seconds':time.time()-started};history.append(r);write(str(folder.relative_to(ROOT)/'history.json'),history);print('LEARNED_BOUND_EPOCH',seed,bound,json.dumps(r),flush=True)
    assert orders==read(old.ROOT/f'training/seed{seed}/text_raw_batches.json'),'batch order differs from bound.35'
    torch.save({'state_dict':model.state_dict(),'seed':seed,'bound':bound,'epoch':20,'selection':'last epoch'},checkpoint);write(str(folder.relative_to(ROOT)/'first_epoch_batches.json'),orders)
   model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True);p=ROOT/f'evaluation/seed{seed}_bound{bound:g}'
   if not p.with_suffix('.json').exists():evaluate(model,teacher,dev,xs,stoi,p,natural)
   del model;torch.cuda.empty_cache()
 write('complete.json',{'fits':9,'all_first_epoch_batches_match_bound035':True,'seeds':SEEDS,'bounds':BOUNDS,'DEV_clips':519,'no_TEST_use':True})
if __name__=='__main__':train()
