"""Paired grouped uncertainty for fixed HumanML3D audit endpoints."""
import argparse,json
from pathlib import Path
import numpy as np
from humanml_audit_common import R,read,write,metrics,sha

def retrieval_rows(emb,text,orders):
 out=np.zeros(len(emb),float)
 for order in orders:
  for idx in order.reshape(-1,32):
   dist=np.linalg.norm(text[idx,None,:]-emb[None,idx,:],axis=-1)
   # This stable convention is recorded; native default-sort sensitivity is
   # checked independently for ties rather than silently changing outcomes.
   rank=np.argsort(dist,axis=1,kind='stable')
   out[idx]+=rank[:,0]==np.arange(32)
 return out/len(orders)
def interval(values,family=1):return np.quantile(values,[.025/family,1-.025/family]).tolist()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=['val','test']);args=ap.parse_args();split=args.split
 assert not (R/'manifest_sha256.json').exists();p=read(R/'protocol.json');assert read(R/f'{split}_complete.json')['passed']
 selection=read(R/f'{split}_selection.json');groups=np.array([x['clip_id'].lstrip('M') for x in selection]);unique=np.unique(groups);N=len(groups)
 rng=np.random.default_rng(20261012);sums=np.zeros((10000,len(unique)),np.int16)
 for i in range(10000):sums[i]=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
 loc={g:i for i,g in enumerate(unique)};counts=np.bincount([loc[g] for g in groups],minlength=len(unique));denom=sums@counts
 def bootstrap(x):
  aggregate=np.bincount([loc[g] for g in groups],weights=x,minlength=len(unique));return (sums@aggregate)/denom
 T=np.load(R/f'{split}_text_embeddings.npy');orders=np.load(R/f'{split}_orders.npy');reference=np.load(R/f'{split}_reference_embeddings.npy');results={}
 for source in ['real','mdm']:
  folder=R/split/source
  with np.load(folder/'identity.npz') as z:base=z['embeddings'].copy();basecan=z['canonical_embeddings'].copy()
  out={}
  for name in p['fixed_conditions']+p['adaptive_conditions']:
   with np.load(folder/(name+'.npz')) as z:emb=z['embeddings'];can=z['canonical_embeddings']
   target=np.roll(T,N//2,axis=0) if name in ['shifted_caption','canonical_reader_shifted_caption'] else T
   before=retrieval_rows(base,target,orders);after=retrieval_rows(emb,target,orders);d=after-before;b=bootstrap(d)
   md0=np.linalg.norm(base-target,axis=1);md1=np.linalg.norm(emb-target,axis=1);bm=bootstrap(md1-md0)
   actual=retrieval_rows(emb,T,orders);meta=read(folder/(name+'.json'))
   targetkey='shifted_caption' if name in ['shifted_caption','canonical_reader_shifted_caption'] else 'actual_caption'
   assert abs(float(after.mean())-meta[targetkey]['R_precision_top1'])<1e-8
   out[name]={'R1_before':float(before.mean()),'R1_after':float(after.mean()),'R1_change':float(d.mean()),'R1_change_group_ci95':interval(b),'R1_change_group_family_ci95_six_endpoints':interval(b,6),
    'MM_distance_before':float(md0.mean()),'MM_distance_after':float(md1.mean()),'MM_distance_change':float((md1-md0).mean()),'MM_distance_change_group_ci95':interval(bm),
    'MM_distance_change_group_family_ci95_six_endpoints':interval(bm,6),'actual_R1_after':float(actual.mean()),'FID_before':read(folder/'identity.json')['actual_caption']['FID'],'FID_after':meta['actual_caption']['FID'],
    'canonical_embedding_max_absolute_change':float(abs(can-basecan).max()),'canonical_embedding_mean_L2_change':float(np.linalg.norm(can-basecan,axis=1).mean()),
    'readout_is_on_same_physical_motion':True}
  results[source]=out
 write(R/f'{split}_analysis.json',{'split':split,'clips':N,'native_mirror_groups':len(unique),'replicates':10000,'R1_candidate_count':32,'R1_repetition_count':5,
  'grouping':'strip leading M from native clip ID; not verified actor identity','candidate_pool_fixed_within_each_repetition':True,'uncertainty_conditional_on_five_candidate_permutations':True,
  'FID_descriptive_only':True,'family_six_adaptive_source_target_endpoints':True,'results':results,'source_code_sha256':sha(Path(__file__))})
 print(json.dumps({s:{c:r for c,r in v.items() if c in p['adaptive_conditions']} for s,v in results.items()},indent=2),flush=True)
if __name__=='__main__':main()
