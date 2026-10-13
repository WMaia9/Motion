"""Post-hoc scalar detector sensitivity; thresholds fitted on DEV only.

All prespecified statistics are reported. Matching E/N is not general stealth.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from verify_signspark_studies import matrices,sha
R=Path('reports/cvpr_critical_review_20261012')
FEATURES=['canonical_RMS','mean_squared_6D_norm','mean_abs_tau','mean_abs_rho_minus_sigma','mean_rho','mean_sigma','std_rho','std_sigma','std_tau']
def read(p):return json.loads(Path(p).read_text())
def features(x):
 d=x.reshape(-1,40,6).astype(float);C=matrices(x)[...,:2,:];u=C[...,0,:];rho=np.linalg.norm(d[...,:3],axis=-1);tau=(u*d[...,3:]).sum(-1);sigma=np.linalg.norm(d[...,3:]-tau[...,None]*u,axis=-1)
 return np.array([np.sqrt(np.mean((d-C.reshape(d.shape))**2)),np.mean((d**2).sum(-1)),np.mean(abs(tau)),np.mean(abs(rho-sigma)),np.mean(rho),np.mean(sigma),np.std(rho),np.std(sigma),np.std(tau)])
def auc(a,b):
 n=len(a);m=len(b);r=rankdata(np.r_[a,b]);return float((r[n:].sum()-m*(m+1)/2)/(n*m))
def main():
 assert not (R/'manifest_sha256.json').exists(),'frozen report: choose a new output root'
 dev=Path('reports/cvpr_adaptive_distance_attack_20261012');test=Path('reports/cvpr_adaptive_distance_confirmation_20261012');inputs=[dev/'manifest_sha256.json',test/'manifest_sha256.json',Path(__file__)]
 plan={'scope':'post-hoc diagnostic after positive TEST attack scores; no new attack optimization or TEST-fitted threshold; not pristine confirmation','features':FEATURES,'conditions':['circle_correct_step200','sphere_correct_step200'],'threshold_selection':'maximum DEV balanced accuracy; orientation and threshold fixed before applying to TEST; all features retained','inputs_sha256':{str(p):sha(p) for p in inputs}}
 (R/'detector_audit_protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
 idx=read(dev/'protocol.json')['selection_indices']
 with np.load('reports/cvpr_constant_noncanonicity_20261012/generated/dev/poses.npz') as z:dx=[z[f'clip{i:05d}'] for i in idx]
 with np.load('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz') as z:tx=[z[f'clip{i:04d}'] for i in range(642)]
 ds=np.array([features(x) for x in dx]);ts=np.array([features(x) for x in tx]);out={}
 for name in plan['conditions']:
  with np.load(dev/f'evaluation/{name}_poses.npz') as z:da=np.array([features(z[f'attacked{i:04d}']) for i in range(64)])
  with np.load(test/f'evaluation/{name}_poses.npz') as z:ta=np.array([features(z[f'attacked{i:04d}']) for i in range(642)])
  stats={}
  for j,f in enumerate(FEATURES):
   vals=np.sort(np.unique(np.r_[ds[:,j],da[:,j]]));thresholds=np.r_[-np.inf,(vals[:-1]+vals[1:])/2,np.inf];best=(-1,None,None)
   for direction in [1,-1]:
    for threshold in thresholds:
     s=ds[:,j]>threshold;a=da[:,j]>threshold
     if direction<0:s=~s;a=~a
     bacc=.5*((~s).mean()+a.mean())
     if bacc>best[0]:best=(float(bacc),float(threshold),direction)
   _,threshold,direction=best;s=ts[:,j]>threshold;a=ta[:,j]>threshold
   if direction<0:s=~s;a=~a
   ar=auc(ts[:,j],ta[:,j]);stats[f]={'DEV_balanced_accuracy':best[0],'threshold':threshold if np.isfinite(threshold) else ('-inf' if threshold<0 else '+inf'),'direction':direction,'TEST_balanced_accuracy':float(.5*((~s).mean()+a.mean())),'TEST_source_false_positive_rate':float(s.mean()),'TEST_attack_detection_rate':float(a.mean()),'TEST_AUC_with_DEV_orientation':ar if direction>0 else 1-ar,'TEST_best_orientation_AUC_diagnostic':max(ar,1-ar)}
  out[name]=stats
 (R/'detector_audit.json').write_text(json.dumps({'scope':plan['scope'],'statistics':out,'all_features_reported':True,'thresholds_not_fit_on_TEST':True},indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
