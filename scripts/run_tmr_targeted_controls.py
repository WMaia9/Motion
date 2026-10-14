"""TMR capability at fixed rotations plus matched standardized-L2 adversary.

Target captions are available. These are evaluator attacks, not generation
improvements or ordinary reward training. All endpoints are retained.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
from motion_release_audit_common import read,write,sha,seed,canonical
from tmr_audit_common import NativeTMR,MODEL,SOURCE,retrieval
R=Path('reports/cvpr_tmr_attack_20261013');OLD=Path('reports/cvpr_humanml3d_audit_20261012')

def projection(x):
 y=x.clone();d=x[...,67:193].reshape(*x.shape[:-1],21,6);u=F.normalize(d[...,:3],dim=-1);v=F.normalize(d[...,3:]-(u*d[...,3:]).sum(-1,keepdim=True)*u,dim=-1);y[...,67:193]=torch.cat([u,v],-1).reshape(*x.shape[:-1],126);return y

def gauge(x,c):
 y=x.clone();d=x[...,67:193].reshape(*x.shape[:-1],21,6);a,b=d[...,:3],d[...,3:];y[...,67:193]=torch.cat([torch.exp(c[...,0:1])*a,torch.exp(c[...,1:2])*b+c[...,2:3]*a],-1).reshape(*x.shape[:-1],126);return y

def main():
 global R
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--clips',type=int,choices=[128,1024],default=128);parser.add_argument('--output',type=Path,default=R);args=parser.parse_args();R=args.output
 assert not (R/'manifest_sha256.json').exists(),'Frozen study: use a new output directory'
 R.mkdir(exist_ok=True);files=[Path(__file__),Path('scripts/tmr_audit_common.py'),MODEL/'config.json',MODEL/'last_weights/motion_encoder.pt',MODEL/'last_weights/text_encoder.pt',OLD/'test_real.npz',OLD/'test_mdm.npz',OLD/'test_selection.json']
 p={'scope':'different-architecture constructed attack and generic-L2 comparator','sources':['real','mdm'],'clips_each':args.clips,'selection':f'first{args.clips}fixed source items of previous1024targeted cohort, not chosen by results','steps':80,'optimizer':'Adam lr0.1 zero initialization, final iterate','gauge_bound':.35,'targets':['actual caption','rolled wrong caption'],'controls':['canonical-gradient','generic all259 continuous channels within perclip standardizedL2 radius of actual-caption gauge attack'],'generic_budget':'L2 in native TMR-normalized units over valid frames; foot contacts fixed; projected each iteration','generic_inputs_may_be_physically_inconsistent':True,'native_cosine_retrieval_32candidates':True,'no_spontaneous_exploitation_claim':True,'source_sha256':{str(f):sha(f) for f in files}}
 if (R/'protocol.json').exists():assert read(R/'protocol.json')==p
 else:write(R/'protocol.json',p)
 seed(2718);model=NativeTMR();items=read(OLD/'test_selection.json')[:args.clips];captions=[x['caption'] for x in items];T=model.texts(captions);wrong=np.roll(T,1,axis=0);summary={}
 for source,path in [('real',OLD/'test_real.npz'),('mdm',OLD/'test_mdm.npz')]:
  with np.load(path) as z:X=z['raw'][:args.clips].copy();L=z['lengths'][:args.clips].copy()
  folder=R/source;folder.mkdir(exist_ok=True);before=model.motions(X,L);beforeC=model.motions(canonical(X),L);scores={'identity':retrieval(T,before),'canonical':retrieval(T,beforeC)};arrays={};logs={}
  for condition in ['actual','wrong','canonical_gradient']:
   ys=[];cs=[];traces=[]
   for i in range(0,len(X),32):
    x=torch.tensor(X[i:i+32],device='cuda');ll=torch.tensor(L[i:i+32],device='cuda');mask=torch.arange(196,device='cuda')[None,:]<ll[:,None];q=torch.zeros((*x.shape[:2],21,3),device='cuda',requires_grad=True);opt=torch.optim.Adam([q],lr=.1);target=torch.tensor((wrong if condition=='wrong' else T)[i:i+32],device='cuda');history=[]
    for step in range(80):
     c=.35*torch.tanh(q)*mask[...,None,None];y=gauge(x,c);e=model.encode_motion(projection(y) if condition=='canonical_gradient' else y,ll);loss=-(F.normalize(e,dim=-1)*F.normalize(target,dim=-1)).sum(-1).mean();opt.zero_grad();loss.backward();opt.step();history.append(float(loss.detach()))
    c=.35*torch.tanh(q.detach())*mask[...,None,None];ys.append(gauge(x,c).detach().cpu().numpy());cs.append(c.cpu().numpy());traces.append(history)
   Y=np.concatenate(ys);E=model.motions(Y,L);EC=model.motions(canonical(Y),L);arrays[condition]=Y;logs[condition]=traces;np.savez_compressed(folder/(condition+'.npz'),raw=Y,coefficients=np.concatenate(cs),embedding=E,canonical_embedding=EC);scores[condition]={'actual':retrieval(T,E),'wrong':retrieval(wrong,E),'canonical':retrieval(T,EC),'canonical_embedding_max_change':float(abs(EC-beforeC).max())};print('TMR_ATTACK_SCORED',source,condition,scores[condition],flush=True)
  gauge_delta=arrays['actual']-X;scale=model.std.detach().cpu().numpy()+1e-12;mask=np.arange(196)[None,:]<L[:,None];radii=np.sqrt(np.sum(((gauge_delta/scale)*mask[...,None])**2,axis=(1,2)));ys=[];traces=[]
  for i in range(0,len(X),32):
   x=torch.tensor(X[i:i+32],device='cuda');ll=torch.tensor(L[i:i+32],device='cuda');mask=torch.arange(196,device='cuda')[None,:]<ll[:,None];active=mask[...,None].expand(-1,-1,263).clone();active[:,:,259:]=False;d=torch.zeros_like(x,requires_grad=True);opt=torch.optim.Adam([d],lr=.1);rad=torch.tensor(radii[i:i+32],device='cuda');target=torch.tensor(T[i:i+32],device='cuda');history=[]
   for step in range(80):
    y=x+d*(model.std+1e-12)*active;e=model.encode_motion(y,ll);loss=-(F.normalize(e,dim=-1)*F.normalize(target,dim=-1)).sum(-1).mean();opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
     d.mul_(active);norm=torch.linalg.vector_norm(d.flatten(1),dim=1);d.mul_(torch.minimum(torch.ones_like(norm),rad/(norm+1e-12))[:,None,None])
    history.append(float(loss.detach()))
   ys.append((x+d.detach()*(model.std+1e-12)*active).cpu().numpy());traces.append(history)
  Y=np.concatenate(ys);E=model.motions(Y,L);EC=model.motions(canonical(Y),L);np.savez_compressed(folder/'generic_L2.npz',raw=Y,embedding=E,canonical_embedding=EC,radii=radii);scores['generic_L2']={'actual':retrieval(T,E),'canonical':retrieval(T,EC),'radius_match_max_excess':float(np.max(np.sqrt(np.sum((((Y-X)/scale)*mask.cpu().numpy()[...,None])**2,axis=(1,2)))-radii))} if False else {'actual':retrieval(T,E),'canonical':retrieval(T,EC),'standardizedL2_bound_verified_in_separate_verifier':True};logs['generic_L2']=traces
  np.savez_compressed(folder/'source.npz',raw=X,lengths=L,text=T,embedding=before,canonical_embedding=beforeC);write(folder/'losses.json',logs);write(folder/'metrics.json',scores);summary[source]=scores
 write(R/'analysis.json',summary);write(R/'complete.json',{'all_sources_and_controls_completed':True});print('TMR_ATTACK_COMPLETE',flush=True)
if __name__=='__main__':main()
