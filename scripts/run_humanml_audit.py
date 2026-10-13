"""Fixed HumanML3D evaluator audit, retaining all controls and final endpoints."""
import argparse, json,time
from pathlib import Path
import numpy as np
import torch
from humanml_audit_common import R,A,DATA,configure,read,write,sha,load_evaluator,text_embeddings,motion_embedding,canonical,transform,encode_raw,metrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=['val','test']);args=ap.parse_args();split=args.split
 assert not (R/'manifest_sha256.json').exists();protocol=read(R/'protocol.json')
 for p,h in protocol['inputs_sha256'].items():assert sha(p)==h,p
 configure();movement,motion,text,mean,std=load_evaluator();assert motion.gru.dropout==0;device='cuda';mean=torch.tensor(mean,device=device);std=torch.tensor(std,device=device)
 selection=read(R/f'{split}_selection.json');N=len(selection);T=text_embeddings(selection,text);np.save(R/f'{split}_text_embeddings.npy',T)
 target=torch.tensor(T,device=device);wrong=torch.roll(target,shifts=N//2,dims=0)
 orders=np.array([np.random.default_rng(seed).permutation(N) for seed in protocol['R_precision_seeds']]);np.save(R/f'{split}_orders.npy',orders)
 sources={}
 for name in ['real','mdm']:
  with np.load(R/f'{split}_{name}.npz') as z:sources[name]=(z['raw'].copy(),z['lengths'].copy())
 reference=[]
 with torch.no_grad():
  X,L=sources['real']
  for i in range(0,N,32):reference.append(encode_raw(torch.tensor(X[i:i+32],device=device),torch.tensor(L[i:i+32],device=device),movement,motion,mean,std).cpu().numpy())
 reference=np.concatenate(reference);np.save(R/f'{split}_reference_embeddings.npy',reference)
 for source,(X,L) in sources.items():
  folder=R/split/source;folder.mkdir(parents=True,exist_ok=True)
  conditions=protocol['fixed_conditions']+protocol['adaptive_conditions']
  for condition in conditions:
   output=folder/(condition+'.npz');meta=folder/(condition+'.json')
   if output.exists() and meta.exists():continue
   ys=[];cs=[];embs=[];cembs=[];losses=[];started=time.monotonic()
   rng=np.random.default_rng(42)
   for start in range(0,N,32):
    x=torch.tensor(X[start:start+32],device=device);lengths=torch.tensor(L[start:start+32],device=device);mask=torch.arange(196,device=device)[None,:]<lengths[:,None];shape=(*x.shape[:2],21,3)
    if condition=='canonical':c=torch.zeros(shape,device=device);y=canonical(x)
    elif condition=='identity':c=torch.zeros(shape,device=device);y=x
    elif condition.startswith('global'):
     sign=1. if 'plus' in condition else -1.;b=sign*int(condition[-3:])/100;c=torch.tensor([b,-b,b],device=device).expand(shape).clone()*mask[...,None,None];y=transform(x,c)
    elif condition.startswith('random'):
     c=torch.tensor(rng.uniform(-.35,.35,size=shape),device=device,dtype=torch.float32)*mask[...,None,None];y=transform(x,c)
    else:
     q=torch.zeros(shape,device=device,requires_grad=True);opt=torch.optim.Adam([q],lr=.1);goal=target[start:start+len(x)] if condition=='target_caption' else wrong[start:start+len(x)]
     history=[]
     # cuDNN requires the no-dropout GRU training flag for input gradients;
     # all encoder parameters remain frozen and final scoring uses eval mode.
     with torch.no_grad():
      expected=encode_raw(x,lengths,movement,motion,mean,std)
      motion.gru.train();actual=encode_raw(x,lengths,movement,motion,mean,std)
      assert torch.max(abs(actual-expected))<1e-4
     for step in range(80):
      c=.35*torch.tanh(q)*mask[...,None,None];y=transform(x,c);z=canonical(y) if condition=='canonical_reader_shifted_caption' else y
      e=encode_raw(z,lengths,movement,motion,mean,std);loss=((e-goal)**2).mean();opt.zero_grad();loss.backward();opt.step();history.append(float(loss.detach()))
     motion.gru.eval()
     c=.35*torch.tanh(q.detach())*mask[...,None,None];y=transform(x,c);losses.append(history)
    with torch.no_grad():
     emb=encode_raw(y,lengths,movement,motion,mean,std);ce=encode_raw(canonical(y),lengths,movement,motion,mean,std)
    ys.append(y.detach().cpu().numpy());cs.append(c.detach().cpu().numpy());embs.append(emb.cpu().numpy());cembs.append(ce.cpu().numpy())
    if condition in protocol['adaptive_conditions']:print('ATTACK_PROGRESS',split,source,condition,start+len(x),N,flush=True)
   Y=np.concatenate(ys);C=np.concatenate(cs);E=np.concatenate(embs);CE=np.concatenate(cembs)
   np.savez_compressed(output,raw=Y,coefficients=C,embeddings=E,canonical_embeddings=CE)
   actual=metrics(E,T,reference,orders);shifted=metrics(E,np.roll(T,N//2,axis=0),reference,orders)
   write(meta,{'split':split,'source':source,'condition':condition,'clips':N,'actual_caption':actual,'shifted_caption':shifted,'elapsed_seconds':time.monotonic()-started,'step_loss_histories':losses,'max_coefficient':float(abs(C).max()),'source_sha256':sha(R/f'{split}_{source}.npz'),'code_sha256':sha(Path(__file__)),'gradient_GRU_flag_only':True,'GRU_dropout':0.,'all_evaluator_weights_frozen':True,'final_scoring_eval_mode':True})
   print('EVALUATED',split,source,condition,actual,flush=True)
 write(R/f'{split}_complete.json',{'passed':True,'sources':2,'conditions_per_source':12,'clips':N,'all_final_endpoints_retained':True})
if __name__=='__main__':main()
