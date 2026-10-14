"""Prospective different-architecture audit on all saved release repetitions."""
from pathlib import Path
import numpy as np
from scipy.stats import t
import torch
from motion_release_audit_common import R as INPUT,read,write,sha,seed,canonical
from run_motion_release_audit import recode
from tmr_audit_common import NativeTMR,MODEL,SOURCE,TEXT,retrieval
R=Path('reports/cvpr_tmr_audit_20261013');METHODS=['real_controls','mdm50','mld','motionlcm_v1','t2mgpt','t2m']
CONDITIONS=['raw','canonical','global_plus010','global_minus010','global_plus035','global_minus035','random035_seed42']
def register():
 R.mkdir(exist_ok=True);files=[Path(__file__),Path('scripts/tmr_audit_common.py'),MODEL/'config.json',MODEL/'last_weights/motion_encoder.pt',MODEL/'last_weights/text_encoder.pt',SOURCE/'src/model/actor.py',SOURCE/'src/model/temos.py',SOURCE/'src/model/tmr.py',SOURCE/'stats/humanml3d/guoh3dfeats/mean.pt',SOURCE/'stats/humanml3d/guoh3dfeats/std.pt',Path('reports/cvpr_generalization_review_20261013/download_distilbert.json')]
 x={'scope':'paired independent ACTOR Transformer/DistilBERT architecture audit; all old saved outputs, no model fitting or sample selection','methods':METHODS,'replications':20,'conditions':CONDITIONS,'primary':'same-output raw-minus-canonical R1 and matched cosine, family12 paired t19 intervals','fixed_conditions':'all listed controls retained, descriptive diagnostics','TMR_native':'strict author weights, author normalizer eps1e-12, ACTOR6layers, VAEmean token, native cosine retrieval','sample_count':'MDM1024; other generators and real4640; unchanged native candidate cohorts from old audit','not_official_full_TMR_dataset_protocol':'author AMASS segment annotations and filtering differ; compatible Guo263 representation and native inference used on our fixed existing cohorts','processed_data_mirror_limitations_retained':True,'reference_captions_only_for_scoring':True,'input_and_source_sha256':{str(p):sha(p) for p in files}}
 if (R/'protocol.json').exists():assert read(R/'protocol.json')==x
 else:write(R/'protocol.json',x)
 return x

def main():
 assert not (R/'manifest_sha256.json').exists();register();seed(20261013);model=NativeTMR();assert torch.linalg.vector_norm(model.modules['motion_encoder'].projection.weight[:,67:193])>0
 write(R/'native_contract.json',{'motion_projection_shape':list(model.modules['motion_encoder'].projection.weight.shape),'rotation_columns_nonzero':True,'native_mu_equals_first_token':True,'no_canonicalization_in_native_input':True})
 for rep in range(20):
  for method in METHODS:
   source=INPUT/method/f'rep_{rep:02d}';folder=R/method/f'rep_{rep:02d}';folder.mkdir(parents=True,exist_ok=True)
   if (folder/'metrics.json').exists():continue
   z=np.load(source/'conditioning.npz');L=z['lengths'].copy() if method=='real_controls' else np.load(source/'generated_lengths.npy');caps=z['captions'].tolist();raw=np.load(source/('source_raw.npy' if method=='real_controls' else 'generated_raw.npy'),mmap_mode='r');T=model.texts(caps);np.save(folder/'text_embeddings.npy',T);scores={}
   for condition in CONDITIONS:
    E=[]
    for i in range(0,len(raw),64):
     block=np.array(raw[i:i+64]);x=block if condition=='raw' else recode(block,condition)
     # Random controls draw per clip so batching cannot reset the same field.
     if condition=='random035_seed42':
      rows=[]
      for j,clip in enumerate(block):
       d=clip[:,67:193].reshape(196,21,6);c=np.random.default_rng(42+i+j).uniform(-.35,.35,(196,21,3));y=clip.copy();y[:,67:193]=np.concatenate([np.exp(c[...,0:1])*d[...,:3],np.exp(c[...,1:2])*d[...,3:]+c[...,2:3]*d[...,:3]],-1).reshape(196,126);rows.append(y)
      x=np.stack(rows)
     with torch.no_grad():E.append(model.encode_motion(x,L[i:i+64]).cpu().numpy())
    E=np.concatenate(E);scores[condition]=retrieval(T,E);np.save(folder/(condition+'_embeddings.npy'),E)
   write(folder/'metrics.json',{'method':method,'repetition':rep,'scores':scores});print('TMR_SCORED',method,rep,scores['raw'],scores['canonical'],flush=True)
 out={}
 for method in METHODS:
  rows=[read(R/method/f'rep_{i:02d}/metrics.json') for i in range(20)];out[method]={}
  for metric in ['R1','matched_cosine']:
   a=np.array([x['scores']['raw']['R_precision'][0] if metric=='R1' else x['scores']['raw'][metric] for x in rows]);b=np.array([x['scores']['canonical']['R_precision'][0] if metric=='R1' else x['scores']['canonical'][metric] for x in rows]);d=a-b;se=d.std(ddof=1)/np.sqrt(20);out[method][metric]={'raw':float(a.mean()),'canonical':float(b.mean()),'raw_minus_C':float(d.mean()),'family12_CI95':(d.mean()+np.array([-1,1])*t.ppf(1-.05/24,19)*se).tolist()}
 write(R/'analysis.json',{'methods':out,'different_architecture':True,'full_official_TMR_protocol_reproduction':False});write(R/'complete.json',{'all120repetition_cells':True,'all7conditions_retained':True});print('TMR_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()
