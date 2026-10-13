"""Scores at exactly declared per-joint norm and canonical-distance budgets."""
import json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric
ROOT=Path('reports/cvpr_constant_noncanonicity_20261012');SEEDS=[42,123,777];REPLICATES=20000
def read(p):return json.loads(Path(p).read_text())
def main():
 assert (ROOT/'complete.json').exists() and not (ROOT/'manifest_sha256.json').exists()
 base=read(ROOT/'evaluation/baseline_raw.json');cb=read(ROOT/'evaluation/baseline_canonical.json');refs=base['references'];n=len(refs);weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=REPLICATES)
 def score(h):
  stat=statistics(refs,h);p=b4(stat.sum(0));assert abs(p-metric(refs,h))<1e-10;return p,np.array([b4(x) for x in weights@stat])
 p0,b0=score(base['hypotheses']);pc,_=score(cb['hypotheses']);arms={}
 for arm in ['text_raw','motion_raw','text_canonical']:
  points=[];boot=[];rms=[];norm=[];energy=[];angle=[];auc=[]
  for seed in SEEDS:
   d=read(ROOT/'evaluation'/f'seed{seed}_{arm}.json');assert d['canonical']['hypotheses']==cb['hypotheses'];p,b=score(d['raw']['hypotheses']);points.append(p);boot.append(b);rms.append(d['mean_layer_RMS']);norm.append(d['per_frame_joint_6D_squared_norm_max_difference']);energy.append(d['per_frame_joint_canonical_distance_squared_max_difference']);angle.append(d['max_rotation_error_deg']);auc.append(d['RMS_best_orientation_AUC'])
  diff=np.stack(boot,1)-b0[:,None];draw=np.random.default_rng(2027).integers(0,3,size=(REPLICATES,3))
  arms[arm]={'raw_per_seed':points,'raw_mean':float(np.mean(points)),'raw_gain':float(np.mean(points)-p0),'canonical_score':pc,'paired_gain_ci95':np.percentile(diff.mean(1),[2.5,97.5]).tolist(),'seed_and_clip_gain_ci95':np.percentile(np.take_along_axis(diff,draw,1).mean(1),[2.5,97.5]).tolist(),'RMS_mean_per_seed':rms,'max_6D_norm_squared_difference':max(norm),'max_canonical_distance_squared_difference':max(energy),'max_rotation_error_deg':max(angle),'simple_RMS_detector_best_AUC_per_seed':auc,'canonical_changes':0}
 out={'DEV_clips':n,'baseline_raw':p0,'baseline_canonical':pc,'arms':arms,'replicates':REPLICATES,'budget_invariants':'total 6D norm and distance to canonical at every represented frame/joint; floating-point residuals reported',
 'no_general_detector_or_security_claim':True,'source_is_raw_frozen_generator':True,'no_TEST_use':True}
 (ROOT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('CONSTANT_BUDGET_ANALYZED',json.dumps(arms),flush=True)
if __name__=='__main__':main()
