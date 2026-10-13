"""Native full-test loader and evaluator for paired release-output audits.

This is a new study; previous frozen producers and reports are unchanged.
"""
from pathlib import Path
import json,hashlib,random,sys,types,os
import numpy as np
import torch
from scipy import linalg
R=Path('reports/cvpr_motion_prevalence_20261012')
DATA=Path('models/humanml3d_audit/data/HumanML3D')
A=Path('models/humanml3d_audit/extracted')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def native_dataset():
 sys.path.insert(0,str(Path('external/MDM').resolve()))
 # spaCy is an unused import for pretokenized native Text2MotionDatasetV2.
 if 'spacy' not in sys.modules:
  spacy=types.ModuleType('spacy')
  def unavailable(*args,**kwargs):raise RuntimeError('spaCy tokenization is outside the pretokenized audit')
  spacy.load=unavailable;sys.modules['spacy']=spacy
 from data_loaders.humanml.data.dataset import Text2MotionDatasetV2
 from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
 opt=types.SimpleNamespace(dataset_name='t2m',fixed_len=0,use_cache=True,cache_dir=str(R/'native_cache'),max_motion_length=196,max_text_len=20,unit_length=4,disable_offset_aug=False,motion_dir=str(DATA/'new_joint_vecs'),text_dir=str(DATA/'texts'))
 (R/'native_cache/dataset').mkdir(parents=True,exist_ok=True)
 vectors=WordVectorizer(str(A/'glove'),'our_vab')
 return Text2MotionDatasetV2(opt,np.zeros(263),np.ones(263),str(DATA/'test.txt'),vectors)

def materialize(dataset,s,limit=None):
 seed(s);order=np.random.permutation(len(dataset));N=(len(order)//32)*32
 if limit is not None:N=min(N,((limit+31)//32)*32)
 order=order[:N];raw=np.zeros((N,196,263),np.float32);words=[];pos=[];caps=[];tokens=[];lengths=[];sent=[];ids=[]
 for i,idx in enumerate(order):
  w,p,c,sl,x,n,t=dataset[int(idx)];raw[i]=x;words.append(w);pos.append(p);caps.append(c);tokens.append(t);lengths.append(n);sent.append(sl);ids.append(dataset.name_list[int(idx)])
 return {'raw':raw,'lengths':np.array(lengths),'words':np.array(words,dtype=np.float32),'pos':np.array(pos,dtype=np.float32),'sent_lengths':np.array(sent),'captions':caps,'tokens':tokens,'clip_ids':ids}

def evaluator():
 sys.path.insert(0,str(Path('external/MDM').resolve()))
 from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
 cwd=Path.cwd();os.chdir(A);model=EvaluatorMDMWrapper('humanml',torch.device('cuda'));os.chdir(cwd)
 for m in [model.movement_encoder,model.motion_encoder,model.text_encoder]:
  m.eval()
  for p in m.parameters():p.requires_grad_(False)
 mu=np.load('external/MDM/dataset/t2m_mean.npy');sd=np.load('external/MDM/dataset/t2m_std.npy');return model,mu,sd

def canonical(x):
 y=x.copy();d=x[...,67:193].reshape(*x.shape[:-1],21,6).astype(np.float64);a=d[...,:3];u=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12);b=d[...,3:];v=b-(u*b).sum(-1,keepdims=True)*u;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12);y[...,67:193]=np.concatenate([u,v],-1).reshape(*x.shape[:-1],126);return y

def score_embeddings(wrapper,raw,cohort,mean,std):
 texts=[];motions=[];N=len(raw)
 with torch.no_grad():
  for i in range(0,N,32):
   ix=np.argsort(-cohort['sent_lengths'][i:i+32],kind='stable');L=cohort['lengths'][i:i+32][ix];x=(raw[i:i+32][ix].astype(float)-mean)/std;mask=np.arange(196)[None,:]<L[:,None];x*=mask[...,None]
   t,m=wrapper.get_co_embeddings(torch.tensor(cohort['words'][i:i+32][ix]),torch.tensor(cohort['pos'][i:i+32][ix]),torch.tensor(cohort['sent_lengths'][i:i+32][ix]),torch.tensor(x),torch.tensor(L))
   texts.append(t.cpu().numpy());motions.append(m.cpu().numpy())
 return np.concatenate(texts),np.concatenate(motions)

def metric(T,E,reference):
 from data_loaders.humanml.utils.metrics import calculate_R_precision,calculate_activation_statistics,calculate_frechet_distance,calculate_diversity
 r=np.zeros(3);mm=0.
 for i in range(0,len(E),32):
  r+=calculate_R_precision(T[i:i+32],E[i:i+32],3,True);mm+=np.linalg.norm(T[i:i+32]-E[i:i+32],axis=1).sum()
 mu,cov=calculate_activation_statistics(E);mr,cr=calculate_activation_statistics(reference);fid=float(calculate_frechet_distance(mu,cov,mr,cr))
 return {'FID':max(fid,0.),'R_precision':(r/len(E)).tolist(),'MM_distance':float(mm/len(E)),'clips':len(E)}
