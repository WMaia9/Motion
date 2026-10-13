"""Prospective DEV-only per-clip attacks preserving auxiliary norm statistics.

Targets are explicitly supplied. This is neither a learned layer nor ordinary
training. Circle fixes both 6D norm and distance; sphere fixes distance only.
"""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F
import run_constant_noncanonicity_layer as circle
import run_constant_distance_sphere as sphere
import run_physical_semantic_ddpm as base
from run_signspark_gauge_attack import SignCollator,Batch,PAD_TOKEN
from run_evaluation_heldout import geometry
from run_reviewer_training import score
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_adaptive_distance_attack_20261012')
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def transform(x,q,kind):
 w=torch.tanh(q)* (1 if kind=='circle' else np.pi/np.sqrt(3))
 if kind=='circle':w=w.expand(*w.shape[:-1],3)
 w=w.clone();w[...,10:25,:]=0
 return (circle if kind=='circle' else sphere).transform(x,w)[0],w

def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;base.seed(42)
 records=read('reports/cvpr_learned_equivalence_20261011/data_audit.json')['dev']
 idx=np.sort(np.random.default_rng(20261012).choice(len(records),64,replace=False)).tolist()
 files=[circle.ROOT/'generated/dev/poses.npz',Path('reports/cvpr_learned_equivalence_20261011/data_audit.json'),base.BASE/'checkpoints/PHOENIX-2014T/best.ckpt',Path(__file__),Path('scripts/run_constant_noncanonicity_layer.py'),Path('scripts/run_constant_distance_sphere.py')]
 proto={'purpose':'separate failure of amortized fixed-statistic layers from reader resistance to targeted per-clip optimization',
 'split':'DEV only, previously inspected public split; no TEST','selection_indices':idx,'selection_seed':20261012,'clips':64,
 'targets':['actual source caption','circular shift by one selected caption'],'geometries':['circle fixes squared 6D norm and squared canonical distance per frame/joint','sphere fixes only squared canonical distance per frame/joint'],
 'optimization':'Adam lr.1, zero initialization, final steps80 and200 both retained, no best decoded checkpoint',
 'circle_phase_bound_radians':float(np.pi),'sphere_vector_bound':float(np.pi/np.sqrt(3)),'safety':'identity when auxiliary radius>=.95','left_hand_joints':'10:25 identity',
 'controls':'random coefficients seeds42/123/777 per geometry; canonical-reader target CE200 per geometry',
 'scope':'tests declared scalar detector statistics, not all possible detectors, learned transfer, ordinary training, encryption or linguistic improvement',
 'statistics':'20000 paired clip bootstrap with family-adjusted intervals for eight optimized target-gain endpoints',
 'inputs_sha256':{str(p):sha(p) for p in files}}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==proto
 else:write('protocol.json',proto)
 examples=base.dataset('dev');assert [e.sequence for e in examples]==[r['clip_id'] for r in records]
 with np.load(circle.ROOT/'generated/dev/poses.npz') as z:xs=[z[f'clip{i:05d}'].copy() for i in idx]
 examples=[copy.copy(examples[i]) for i in idx]
 for e,x in zip(examples,xs):e.sgn=torch.from_numpy(x)
 refs=[' '.join(e.txt) for e in examples];ids=[e.sequence for e in examples];targets=refs[1:]+refs[:1]
 write('selection.json',[{'source_index':i,'clip_id':c,'reference':r,'target':t,'frames':len(x)} for i,c,r,t,x in zip(idx,ids,refs,targets,xs)])
 teacher=base.make_back_translation_model(base.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
 for p in teacher.parameters():p.requires_grad_(False)
 collator=SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[PAD_TOKEN]
 raw=score(teacher,xs,refs,ids);can=score(teacher,[base.project(x) for x in xs],refs,ids)
 write('baseline_raw.json',raw);write('baseline_canonical.json',can)
 def save(name,ys,ws,target,logs):
  out=score(teacher,ys,refs,ids);canonical=score(teacher,[base.project(y) for y in ys],refs,ids)
  assert canonical['hypotheses']==can['hypotheses']
  emax=nmax=0.;ra=[];rb=[]
  for x,y in zip(xs,ys):
   assert np.array_equal(x[:,60:150],y[:,60:150]);assert np.isfinite(y).all()
   c=base.project(x);d=base.project(y)
   e=((x-c).astype(float).reshape(-1,40,6)**2).sum(-1);ee=((y-d).astype(float).reshape(-1,40,6)**2).sum(-1)
   emax=max(emax,float(abs(e-ee).max()));nmax=max(nmax,float(abs((x.astype(float).reshape(-1,40,6)**2).sum(-1)-(y.astype(float).reshape(-1,40,6)**2).sum(-1)).max()))
   ra.append(float(np.sqrt(e.mean()/6)));rb.append(float(np.sqrt(ee.mean()/6)))
  assert emax<2e-5
  if name.startswith('circle'):assert nmax<2e-5
  ranks=rankdata(np.r_[ra,rb]);auc=float((ranks[len(ra):].sum()-len(ra)*(len(ra)+1)/2)/len(ra)**2)
  write(f'evaluation/{name}.json',{'raw':out,'canonical':canonical,'targets':target,'canonical_hypothesis_changes':0,'canonical_distance_squared_max_difference':emax,'squared_norm_max_difference':nmax,'source_RMS':ra,'attacked_RMS':rb,'RMS_best_orientation_AUC':max(auc,1-auc),'losslogs':logs,**geometry(xs,ys)})
  np.savez_compressed(ROOT/f'evaluation/{name}_poses.npz',**{f'{kind}{i:04d}':x for kind,a in [('attacked',ys),('auxiliary',ws)] for i,x in enumerate(a)})
  print('ADAPTIVE_SCORED',name,out['scores']['bleu']['bleu4'],emax,nmax,auc,flush=True)
 for kind in ['circle','sphere']:
  dim=1 if kind=='circle' else 3
  for mode in ['correct','wrong','canonical_wrong']:
   wanted=refs if mode=='correct' else targets;ends=[80,200] if mode!='canonical_wrong' else [200]
   if all((ROOT/f'evaluation/{kind}_{mode}_step{s}.json').exists() for s in ends):continue
   attacked={s:[] for s in ends};aux={s:[] for s in ends};logs=[]
   for start in range(0,64,8):
    chunk=[copy.copy(e) for e in examples[start:start+8]]
    for e,t in zip(chunk,wanted[start:start+8]):e.txt=t.split()
    batch=Batch(collator(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True);original=batch.sgn.detach();q=torch.zeros(8,original.shape[1],40,dim,device='cuda',requires_grad=True);opt=torch.optim.Adam([q],lr=.1);losses=[]
    for step in range(1,201):
     y,w=transform(original,q,kind);z=base.physical(y) if mode=='canonical_wrong' else y
     output,_=teacher.forward(z,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask);logits=output[0]
     loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad)
     assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert torch.isfinite(q.grad).all();opt.step();losses.append(float(loss.detach()))
     if step in ends:
      with torch.no_grad():y,w=transform(original,q,kind)
      for j,e in enumerate(chunk):attacked[step].append(y[j,:len(e.sgn)].cpu().numpy().copy());aux[step].append(w[j,:len(e.sgn)].cpu().numpy().copy())
    logs.append({'batch_start':start,'CE':losses});print('ADAPTIVE_BATCH',kind,mode,start,losses[0],losses[-1],flush=True)
   for s in ends:save(f'{kind}_{mode}_step{s}',attacked[s],aux[s],wanted,logs)
  for seed in [42,123,777]:
   name=f'{kind}_random_seed{seed}'
   if (ROOT/f'evaluation/{name}.json').exists():continue
   rng=np.random.default_rng(seed);ys=[];ws=[]
   for x in xs:
    q=torch.from_numpy(rng.standard_normal((1,len(x),40,dim)).astype(np.float32))
    with torch.no_grad():y,w=transform(torch.from_numpy(x)[None],q,kind)
    ys.append(y[0].numpy());ws.append(w[0].numpy())
   save(name,ys,ws,targets,[])
 write('complete.json',{'conditions':16,'clips':64,'new_TEST_use':False,'per_clip_target_optimization':True,'not_amortized_or_ordinary_training':True})
if __name__=='__main__':main()
