"""Paired confirmation of the 'above real-motion BT' descriptive comparison."""
import json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric
ROOT=Path('reports/cvpr_second_review_20261012');NEW=Path('reports/cvpr_learned_confirmation_20261012')
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (ROOT/'manifest_sha256.json').exists()
 native=read('reports/cvpr_evaluation_confirmation_20261011/heldout/PHOENIX-2014T/full_real_scores.json');test=read(NEW/'learned/baseline.json');ids=test['clip_ids'];refs=test['references'];lookup={n:(r,h) for n,r,h in zip(native['clip_ids'],native['references'],native['raw']['hypotheses'])};real=[lookup[n][1] for n in ids];assert refs==[lookup[n][0] for n in ids]
 weights=np.random.default_rng(42).multinomial(len(ids),np.ones(len(ids))/len(ids),size=20000)
 def score(h):
  stat=statistics(refs,h);p=b4(stat.sum(0));assert abs(p-metric(refs,h))<1e-10;return p,np.asarray([b4(x) for x in weights@stat])
 p,b=score(real);points=[];boots=[]
 for seed in [42,123,777]:
  d=read(NEW/f'learned/seed{seed}_text_raw_source_text.json');q,c=score(d['raw']['hypotheses']);points.append(q);boots.append(c)
 diff=np.stack(boots,1)-b[:,None];ci=np.percentile(diff.mean(1),[2.5,97.5]).tolist()
 out={'clips':len(ids),'native_real_motion_raw_BT':p,'learned_generated_raw_BT_per_seed':points,'mean_learned_score':float(np.mean(points)),'mean_difference':float(np.mean(points)-p),'paired_ci95':ci,'point_estimate_above_real':float(np.mean(points))>p,'supported_superiority_if_ci_positive':ci[0]>0,'interpretation':'reader score comparison only; not better generated signing or statistical superiority unless interval excludes zero'}
 (ROOT/'generated_vs_real_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('GENERATED_VS_REAL_COMPARISON',json.dumps(out),flush=True)
if __name__=='__main__':main()
