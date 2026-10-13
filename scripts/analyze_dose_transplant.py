"""All eight registered coefficient-transplant endpoints, no best dose selection."""
import json
from pathlib import Path
import numpy as np
from analyze_second_review_results import Bootstrap,ci,joint
R=Path('reports/cvpr_dose_transplant_20261012');SEEDS=[42,123,777]
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (R/'manifest_sha256.json').exists() and read(R/'complete.json')['conditions']==24
 first=read(R/'evaluation/seed42_geodesic_raw_w1_epoch20.json');boot=Bootstrap(first['targets']);out={}
 for c in read(R/'protocol.json')['conditions']:
  arm,e=c['arm'],c['epoch'];ps=[];bs=[];changes=[];actual=[]
  for s in SEEDS:
   r=read(R/f'evaluation/seed{s}_{arm}_epoch{e}.json');assert r['targets']==first['targets'] and r['clip_ids']==first['clip_ids']
   p0,b0=boot.score(r['baseline_raw']['hypotheses']);p1,b1=boot.score(r['raw']['hypotheses']);ps.append([p0,p1]);bs.append(b1-b0);changes.append(r['canonical_hypothesis_changes'])
   actual.append([r['baseline_raw']['scores']['bleu']['bleu4'],r['raw']['scores']['bleu']['bleu4']])
  d=np.stack(bs,1);points=np.array(ps);out[f'{arm}/epoch{e}']={'baseline_wrong_target_per_seed':points[:,0].tolist(),'after_wrong_target_per_seed':points[:,1].tolist(),'wrong_target_gain':float((points[:,1]-points[:,0]).mean()),'paired_clip_ci95':ci(d.mean(1)),'seed_and_clip_ci95':ci(joint(d)),'family_adjusted_seed_and_clip_ci95_eight_conditions':ci(joint(d),8),'actual_caption_before_after_per_seed':actual,'canonical_hypothesis_changes':changes}
 (R/'analysis.json').write_text(json.dumps({'split':'DEV','clips':519,'conditions':out,'replicates':20000,'all_grid_conditions_reported':True,'registered_before_new_dose_BLEU_results':True,'no_per_clip_gradient_or_extra_trained_module':True,'not_TEST_confirmation':True},indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
