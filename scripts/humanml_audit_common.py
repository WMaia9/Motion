"""Native HumanML3D evaluator adapter; transform only unnormalized rotations."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
from scipy import linalg

R=Path('reports/cvpr_humanml3d_audit_20261012')
A=Path('models/humanml3d_audit/extracted')
DATA=Path('models/humanml3d_audit/data/HumanML3D')
SOURCE=Path('external/HumanML3D_Evaluator')
sys.path.insert(0,str(SOURCE.resolve()))
from networks.modules import MovementConvEncoder,MotionEncoderBiGRUCo,TextEncoderBiGRUCo
from utils.word_vectorizer import WordVectorizer

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def configure():
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(20261012);np.random.seed(20261012)

def load_evaluator():
 device=torch.device('cuda');ck=A/'t2m/text_mot_match/model/finest.tar'
 state=torch.load(ck,map_location='cpu',weights_only=False)
 movement=MovementConvEncoder(259,512,512).to(device)
 motion=MotionEncoderBiGRUCo(512,1024,512,device).to(device)
 text=TextEncoderBiGRUCo(300,15,512,512,device).to(device)
 for model,key in [(movement,'movement_encoder'),(motion,'motion_encoder'),(text,'text_encoder')]:
  model.load_state_dict(state[key],strict=True);model.eval()
  for p in model.parameters():p.requires_grad_(False)
 mean=np.load(A/'t2m/Comp_v6_KLD01/meta/mean.npy').astype(np.float32)
 std=np.load(A/'t2m/Comp_v6_KLD01/meta/std.npy').astype(np.float32)
 assert mean.shape==std.shape==(263,) and (std>0).all()
 return movement,motion,text,mean,std

def motion_embedding(x,lengths,movement,motion):
 order=torch.argsort(lengths,descending=True,stable=True);inverse=torch.argsort(order)
 feats=movement(x[order,:,:259]);emb=motion(feats,(lengths[order]//4).cpu())
 return emb[inverse]

def text_embeddings(selection,text):
 vectors=WordVectorizer(str(A/'glove'),'our_vab');ws=[];ps=[];lengths=[]
 for clip in selection:
  tokens=clip['tokens'].split()[:20];tokens=['sos/OTHER']+tokens+['eos/OTHER'];lengths.append(len(tokens));tokens+=['unk/OTHER']*(22-len(tokens))
  pairs=[vectors[t] for t in tokens];ws.append(np.stack([x[0] for x in pairs]));ps.append(np.stack([x[1] for x in pairs]))
 out=[]
 with torch.no_grad():
  for start in range(0,len(ws),64):
   w=torch.tensor(np.array(ws[start:start+64]),device='cuda').float();p=torch.tensor(np.array(ps[start:start+64]),device='cuda').float();l=torch.tensor(lengths[start:start+64]);idx=torch.argsort(l,descending=True,stable=True);inv=torch.argsort(idx);out.append(text(w[idx],p[idx],l[idx])[inv].cpu().numpy())
 return np.concatenate(out)

def canonical(x):
 y=x.clone();d=x[...,67:193].reshape(*x.shape[:-1],21,6);a=d[...,:3];u=torch.nn.functional.normalize(a,dim=-1);b=d[...,3:];v=torch.nn.functional.normalize(b-(u*b).sum(-1,keepdim=True)*u,dim=-1);y[...,67:193]=torch.cat([u,v],-1).reshape(*x.shape[:-1],126);return y

def transform(raw,c):
 y=raw.clone();d=raw[...,67:193].reshape(*raw.shape[:-1],21,6);a,b=d[...,:3],d[...,3:];y[...,67:193]=torch.cat([torch.exp(c[...,0:1])*a,torch.exp(c[...,1:2])*b+c[...,2:3]*a],-1).reshape(*raw.shape[:-1],126);return y

def encode_raw(raw,lengths,movement,motion,mean,std):
 norm=(raw-mean)/std;mask=torch.arange(raw.shape[1],device=raw.device)[None,:]<lengths[:,None];norm=norm*mask[...,None]
 return motion_embedding(norm,lengths,movement,motion)

def metrics(emb,text,reference,orders):
 eu=np.linalg.norm(emb-text,axis=1);counts=[]
 for order in orders:
  rows=[]
  for idx in order.reshape(-1,32):
   D=np.linalg.norm(text[idx,None,:]-emb[None,idx,:],axis=-1);r=np.argsort(D,axis=1,kind='stable');rows.extend(np.any(r[:,:1]==np.arange(32)[:,None],axis=1))
  counts.append(np.mean(rows))
 mu=emb.mean(0);cov=np.cov(emb,rowvar=False);mur=reference.mean(0);covr=np.cov(reference,rowvar=False)
 root=linalg.sqrtm(cov@covr)
 if np.iscomplexobj(root):root=root.real
 fid=float((mu-mur)@(mu-mur)+np.trace(cov+covr-2*root))
 return {'R_precision_top1':float(np.mean(counts)),'R_precision_repetitions':counts,'MM_distance':float(eu.mean()),'FID':max(fid,0.),'mean_embedding_distance_to_reference_clip':float(np.linalg.norm(emb-reference,axis=1).mean())}
