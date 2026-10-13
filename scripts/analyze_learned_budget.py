"""Matched smaller-bound layers: scores, uncertainty and detectable norms."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata  # Load compiled SciPy before legacy NumPy module aliases.
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric
from run_learned_budget import ROOT,BOUNDS,SEEDS
import run_learned_equivalence as old
REPLICATES=20000

def read(p):return json.loads(Path(p).read_text())
def main():
 assert (ROOT/'complete.json').exists() and not (ROOT/'manifest_sha256.json').exists()
 base=read(ROOT/'evaluation/baseline.json');refs=base['references'];n=len(refs);w=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=REPLICATES)
 def scores(h):
  stat=statistics(refs,h);p=b4(stat.sum(0));assert abs(p-metric(refs,h))<1e-10;return p,np.asarray([b4(x) for x in w@stat])
 p0,b0=scores(base['hypotheses']);results={};natural=read(ROOT/'evaluation/natural_generator.json')
 for bound in BOUNDS+[.35]:
  points=[];bootstrap=[];rms=[];aucs=[];errors=[]
  for seed in SEEDS:
   if bound==.35:
    r=read(old.ROOT/f'evaluation/seed{seed}_text_raw_source_text.json');s=r['raw'];path=old.ROOT/f'evaluation/seed{seed}_text_raw_source_text_poses.npz'
    with np.load(path) as z:
     xs=[z[f'attacked{i:04d}'] for i in range(n)]
    from run_learned_budget import profiles,auc
    profile=profiles(xs);rms.append(profile['mean_RMS']);a=auc(natural['clip_RMS_values'],profile['clip_RMS_values']);aucs.append(max(a,1-a))
   else:
    r=read(ROOT/f'evaluation/seed{seed}_bound{bound:g}.json');s=r['raw'];rms.append(r['noncanonicity']['mean_RMS']);aucs.append(r['RMS_detector_best_orientation_AUC'])
   assert r['canonical']['hypotheses']==base['hypotheses'];p,b=scores(s['hypotheses']);points.append(p);bootstrap.append(b);errors.append(r['max_rotation_error_deg'])
  delta=np.stack(bootstrap,1)-b0[:,None];draw=np.random.default_rng(2027).integers(0,3,size=(REPLICATES,3));results[str(bound)]={'raw_per_seed':points,'raw_mean':float(np.mean(points)),'baseline_canonical':p0,'mean_gain':float(np.mean(points)-p0),'paired_gain_ci95':np.percentile(delta.mean(1),[2.5,97.5]).tolist(),'seed_and_clip_gain_ci95':np.percentile(np.take_along_axis(delta,draw,1).mean(1),[2.5,97.5]).tolist(),'mean_RMS_to_canonical_per_seed':rms,'RMS_detector_best_orientation_AUC_per_seed':aucs,'canonical_changes':0,'max_rotation_error_deg':max(errors)}
 out={'DEV_clips':n,'baseline_canonical':p0,'natural_generator_mean_RMS':natural['mean_RMS'],'results':results,'replicates':REPLICATES,'detection_scope':'one-dimensional clip RMS compared to raw same-generator output; no general stealth claim','no_TEST_use':True}
 (ROOT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('LEARNED_BUDGET_ANALYZED',json.dumps(results),flush=True)
if __name__=='__main__':main()
