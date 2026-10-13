"""Post-hoc paired group-bootstrap sensitivity for the sealed blind FID audit.

Conditional on one generated cohort; does not estimate generation/training seed
variability. Primary registered FID remains descriptive. Both bounds are kept.
"""
from pathlib import Path
import numpy as np
import torch
from humanml_audit_common import read,write,sha,configure
R=Path('reports/cvpr_humanml3d_blind_fid_20261012')
def covariance(x,w):
 total=w.sum();mu=(x*w[:,None]).sum(0)/total;c=x-mu;cov=c.T@(c*w[:,None])/(total-1);return mu,(cov+cov.T)/2

def fid(x,y,w):
 mu,A=covariance(x,w);nu,B=covariance(y,w);d,U=torch.linalg.eigh(A);sq=(U*torch.sqrt(d.clamp_min(0)))@U.T;C=sq@B@sq;values=torch.linalg.eigvalsh((C+C.T)/2).clamp_min(0)
 return float(((mu-nu)**2).sum()+torch.trace(A+B)-2*torch.sqrt(values).sum())
def main():
 assert not (R/'manifest_sha256.json').exists();assert not (R/'FID_bootstrap_sensitivity.json').exists()
 configure();analysis=read(R/'analysis.json');selection=read(R/'test_selection.json');groups=np.array([x['clip_id'].lstrip('M') for x in selection]);unique,labels=np.unique(groups,return_inverse=True);N=len(groups)
 conditions=['identity','blind_OT_bound010','blind_OT_bound035'];inputs=[R/'analysis.json',R/'attack_outputs_before_TEST_scoring.json',R/'withheld_TEST_reference_embeddings.npz',Path(__file__)]
 write(R/'FID_bootstrap_protocol.json',{'scope':'post-hoc sensitivity after point FIDs inspected, not registered primary inference; no new attack or selection','replicates':200,'seed':20261012,'grouping':'native source clip ID without mirror prefix, not actor IDs','paired_cohort_and_reference_resampling':True,'conditions':conditions,'family_size':2,'one_MDM_generation_seed':True,'inputs_sha256':{str(p):sha(p) for p in inputs}})
 device='cuda'
 with np.load(R/'withheld_TEST_reference_embeddings.npz') as z:Y=torch.tensor(z['motion'].astype(float),device=device)
 E={}
 for c in conditions:
  with np.load(R/'attacks'/(c+'.npz')) as z:E[c]=torch.tensor(z['embeddings'].astype(float),device=device)
 weight=torch.ones(N,device=device,dtype=torch.float64);parity={}
 for c in conditions:
  point=fid(E[c],Y,weight);expected=analysis['conditions'][c]['raw']['FID'];assert abs(point-expected)<1e-5,(c,point,expected);parity[c]=abs(point-expected)
 rng=np.random.default_rng(20261012);out={c:[] for c in conditions}
 for k in range(200):
  count=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique));w=torch.tensor(count[labels],device=device,dtype=torch.float64)
  for c in conditions:out[c].append(fid(E[c],Y,w))
  if (k+1)%20==0:print('FID_BOOTSTRAP',k+1,200,flush=True)
 baseline=np.array(out['identity']);results={}
 for c in conditions[1:]:
  d=np.array(out[c])-baseline;results[c]={'point_change':analysis['conditions'][c]['FID_change'],'paired_group_change_ci95':np.quantile(d,[.025,.975]).tolist(),'paired_group_change_family_ci95_two_bounds':np.quantile(d,[.0125,.9875]).tolist()}
 write(R/'FID_bootstrap_sensitivity.json',{'scope':'post-hoc paired group sensitivity, conditional on one fixed generated cohort and evaluator; no seed-variability estimate','replicates':200,'groups':len(unique),'results':results,'native_point_FID_parity_max_error':max(parity.values()),'bootstrap_FIDs':out,'all_bounds_retained':True})
 print(results,flush=True)
if __name__=='__main__':main()
