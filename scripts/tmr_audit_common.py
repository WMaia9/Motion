"""Author TMR encoders, native normalizer and sample-mean/cosine inference.

Components are imported directly to avoid optional Lightning training imports;
no layer, state, normalization or inference equation is replaced.
"""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from motion_release_audit_common import sha,read,write,seed
MODEL=Path('models/tmr_audit/models/tmr_humanml3d_guoh3dfeats')
TEXT=Path('models/generalization_audit/distilbert-base-uncased')
SOURCE=Path('external/TMR')

def native_module(file,name):
 spec=importlib.util.spec_from_file_location(name,SOURCE/file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class NativeTMR:
 def __init__(self,device='cuda'):
  self.device=torch.device(device);cfg=read(MODEL/'config.json');actor=native_module('src/model/actor.py','tmr_native_actor');self.modules={}
  for name in ['motion_encoder','text_encoder']:
   args={k:v for k,v in cfg['model'][name].items() if not k.startswith('_')};model=actor.ACTORStyleEncoder(**args);model.load_state_dict(torch.load(MODEL/'last_weights'/(name+'.pt'),map_location='cpu',weights_only=True),strict=True);model.to(self.device).eval()
   for p in model.parameters():p.requires_grad_(False)
   self.modules[name]=model
  self.mean=torch.load(SOURCE/'stats/humanml3d/guoh3dfeats/mean.pt',weights_only=True).to(self.device);self.std=torch.load(SOURCE/'stats/humanml3d/guoh3dfeats/std.pt',weights_only=True).to(self.device);assert self.mean.shape==self.std.shape==(263,)
  self.tokenizer=AutoTokenizer.from_pretrained(TEXT);self.bert=AutoModel.from_pretrained(TEXT).to(self.device).eval()
  for p in self.bert.parameters():p.requires_grad_(False)
  self.cache={};self.cfg=cfg

 def encode_motion(self,raw,L):
  x=torch.as_tensor(raw,device=self.device,dtype=torch.float32);l=torch.as_tensor(L,device=self.device);x=(x-self.mean)/(self.std+1e-12);mask=torch.arange(x.shape[1],device=self.device)[None,:]<l[:,None]
  return self.modules['motion_encoder']({'x':x,'mask':mask})[:,0]

 @torch.no_grad()
 def texts(self,captions,batch=64):
  for i in range(0,len(captions),batch):
   missing=list(dict.fromkeys(c for c in captions[i:i+batch] if c not in self.cache))
   if not missing:continue
   inputs=self.tokenizer(missing,return_tensors='pt',padding=True).to(self.device);tokens=self.bert(**inputs).last_hidden_state;E=self.modules['text_encoder']({'x':tokens,'mask':inputs.attention_mask.bool()})[:,0]
   for c,e in zip(missing,E):self.cache[c]=e.cpu().numpy().copy()
  return np.stack([self.cache[c] for c in captions])

 @torch.no_grad()
 def motions(self,raw,L,batch=64):
  return np.concatenate([self.encode_motion(np.array(raw[i:i+batch]),L[i:i+batch]).cpu().numpy() for i in range(0,len(raw),batch)])

def retrieval(T,E):
 T=T/np.linalg.norm(T,axis=1,keepdims=True);E=E/np.linalg.norm(E,axis=1,keepdims=True);counts=np.zeros(3)
 for i in range(0,len(E),32):
  similarity=T[i:i+32]@E[i:i+32].T;order=np.argsort(-similarity,axis=1);rank=np.argmax(order==np.arange(len(similarity))[:,None],axis=1);counts+=np.array([(rank<k).sum() for k in [1,2,3]])
 return {'R_precision':(counts/len(E)).tolist(),'matched_cosine':float(np.sum(T*E,axis=1).mean()),'clips':len(E),'candidate_pool':32,'native_cosine_and_sample_mean':True,'not_official_full_TMR_annotation_protocol':True}
