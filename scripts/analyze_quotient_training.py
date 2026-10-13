"""All prespecified arms; corpus paired intervals for ordinary-training inflation."""
import json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric
from run_quotient_training import ROOT,SEEDS,ARMS,tag
REPLICATES=20000

def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (ROOT/'manifest_sha256.json').exists() and (ROOT/'complete.json').exists()
 first=read(ROOT/'training/seed42/geodesic_none_w0/dev_epoch20/raw_scores.json');refs=first['references'];ids=first['clip_ids'];n=len(refs);weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=REPLICATES)
 points={};boots={};summary={}
 for seed in SEEDS:
  for motion,kind,w in ARMS:
   name=tag(motion,kind,w);epochs=[5,20] if motion=='geodesic' and (kind=='none' or w==1) else [5]
   for epoch in epochs:
    for encoding in ['raw','canonical']:
     r=read(ROOT/f'training/seed{seed}/{name}/dev_epoch{epoch}/{encoding}_scores.json');assert r['references']==refs and r['clip_ids']==ids
     stat=statistics(refs,r['hypotheses']);p=b4(stat.sum(0));assert abs(p-metric(refs,r['hypotheses']))<1e-10
     points[seed,name,epoch,encoding]=p;boots[seed,name,epoch,encoding]=np.asarray([b4(row) for row in weights@stat])
 for motion,kind,w in ARMS:
  name=tag(motion,kind,w);epochs=[5,20] if motion=='geodesic' and (kind=='none' or w==1) else [5]
  for epoch in epochs:
   base=tag(motion,'none',0.);raw=np.array([points[s,name,epoch,'raw'] for s in SEEDS]);can=np.array([points[s,name,epoch,'canonical'] for s in SEEDS]);gaps=raw-can
   B=np.stack([boots[s,name,epoch,'raw']-boots[s,name,epoch,'canonical'] for s in SEEDS],1)
   rawdelta=np.array([points[s,name,epoch,'raw']-points[s,base,epoch,'raw'] for s in SEEDS]);candelta=np.array([points[s,name,epoch,'canonical']-points[s,base,epoch,'canonical'] for s in SEEDS])
   excessB=B-np.stack([boots[s,base,epoch,'raw']-boots[s,base,epoch,'canonical'] for s in SEEDS],1)
   summary[name+f'/epoch{epoch}']={'raw_per_seed':raw.tolist(),'canonical_per_seed':can.tolist(),'raw_mean':float(raw.mean()),'canonical_mean':float(can.mean()),'raw_minus_canonical':float(gaps.mean()),'gap_ci95_conditional':np.percentile(B.mean(1),[2.5,97.5]).tolist(),
    'mean_raw_gain_vs_matched_motion_only':float(rawdelta.mean()),'mean_canonical_gain_vs_matched_motion_only':float(candelta.mean()),'excess_raw_gain':float((rawdelta-candelta).mean()),'excess_gain_ci95_conditional':np.percentile(excessB.mean(1),[2.5,97.5]).tolist()}
 def compare(a,b):
  A=np.stack([boots[s,a,20,'raw']-boots[s,a,20,'canonical'] for s in SEEDS],1);B=np.stack([boots[s,b,20,'raw']-boots[s,b,20,'canonical'] for s in SEEDS],1);diff=A-B
  draw=np.random.default_rng(2027).integers(0,3,size=(REPLICATES,3));point=np.mean([points[s,a,20,'raw']-points[s,a,20,'canonical']-points[s,b,20,'raw']+points[s,b,20,'canonical'] for s in SEEDS])
  return {'mean_difference_of_raw_canonical_gaps':float(point),'paired_clip_ci95':np.percentile(diff.mean(1),[2.5,97.5]).tolist(),'seed_and_clip_ci95':np.percentile(np.take_along_axis(diff,draw,1).mean(1),[2.5,97.5]).tolist(),'family_ci_three_primary_contrasts':np.percentile(diff.mean(1),[100*.05/(2*3),100*(1-.05/(2*3))]).tolist()}
 raw='geodesic_raw_w1';baseline='geodesic_none_w0';can='geodesic_canonical_w1';absolute=np.stack([boots[z,raw,20,'raw']-boots[z,raw,20,'canonical'] for z in SEEDS],1)
 primary={'raw_vs_no_CE':compare(raw,baseline),'raw_vs_canonical_CE':compare(raw,can),
 'raw_gap_absolute':{'mean_gap':summary[raw+'/epoch20']['raw_minus_canonical'],'paired_clip_ci95':summary[raw+'/epoch20']['gap_ci95_conditional'],'family_ci_three_primary_contrasts':np.percentile(absolute.mean(1),[100*.05/(2*3),100*(1-.05/(2*3))]).tolist()}}
 guards={'ordinary_denoiser_no_added_gauge_head':True,'motions_can_change_across_fits':True,'raw_canonical_of_one_output_preserve_rotations':True,
 'all_sweep_conditions_reported':True,'DEV_exploratory_not_pristine':True,'no_TEST_use':True,'one_shared_pretraining_init':True,
 'primary_positive_conditional':primary['raw_vs_no_CE']['paired_clip_ci95'][0]>0 and primary['raw_vs_canonical_CE']['paired_clip_ci95'][0]>0 and primary['raw_gap_absolute']['paired_clip_ci95'][0]>0}
 out={'DEV_clips':n,'replicates':REPLICATES,'bootstrap_seed':42,'training_seeds':SEEDS,'systems':summary,'primary_epoch20':primary,'interpretation':guards}
 (ROOT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('ORDINARY_TRAINING_ANALYZED',json.dumps(primary),json.dumps(guards),flush=True)
if __name__=='__main__':main()
