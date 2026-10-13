"""Paired native-unit uncertainty for designed real-motion equivalence layers."""
import csv,json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric
ROOT=Path('reports/cvpr_real_motion_layers_20261012');SEEDS=[42,123,777];REPLICATES=20000
def read(p):return json.loads(Path(p).read_text())
def main():
 assert (ROOT/'complete.json').exists() and not (ROOT/'manifest_sha256.json').exists()
 results={}
 for corpus in ['How2Sign','CSL-Daily']:
  folder=ROOT/corpus;base=read(folder/'baseline_raw.json');canbase=read(folder/'baseline_canonical.json');refs=base['references'];ids=base['clip_ids'];char=corpus=='CSL-Daily';groups=ids
  if corpus=='How2Sign':
   audit=read('reports/cvpr_canonical_readers_20261011/How2Sign/data_audit.json');assert audit['dev_ids']==ids;groups=audit['DEV_video_groups']
  names=sorted(set(groups));index=np.array([names.index(g) for g in groups]);n=len(names);weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=REPLICATES)
  def score(h):
   rr=[' '.join(list(r.replace(' ',''))) for r in refs] if char else refs;hh=[' '.join(list(x.replace(' ',''))) for x in h] if char else h
   stat=statistics(rr,hh);grouped=np.zeros((n,10),np.int64);np.add.at(grouped,index,stat);p=b4(stat.sum(0));assert abs(p-metric(refs,h,char))<1e-10;return p,np.array([b4(x) for x in weights@grouped])
  p0,b0=score(base['hypotheses']);pc,bc=score(canbase['hypotheses']);arms={}
  for arm,seeds in [('text_raw',SEEDS),('motion_raw',[42]),('text_canonical',[42])]:
   points=[];boot=[];angles=[];rms=[]
   for seed in seeds:
    d=read(folder/'evaluation'/f'seed{seed}_{arm}.json');assert d['canonical']['hypotheses']==canbase['hypotheses'];p,b=score(d['raw']['hypotheses']);points.append(p);boot.append(b);angles.append(d['max_rotation_error_deg']);rms.append(d['mean_RMS_change_from_source'])
   diff=np.stack(boot,1)-b0[:,None];draw=np.random.default_rng(2027).integers(0,len(seeds),size=(REPLICATES,len(seeds)))
   arms[arm]={'seeds':seeds,'raw_per_seed':points,'raw_mean':float(np.mean(points)),'canonical_per_seed':[pc]*len(seeds),'raw_gain_mean':float(np.mean(points)-p0),
    'gain_ci95_conditional':np.percentile(diff.mean(1),[2.5,97.5]).tolist(),'gain_ci95_seed_and_group':np.percentile(np.take_along_axis(diff,draw,1).mean(1),[2.5,97.5]).tolist(),'canonical_hypothesis_changes':0,'max_rotation_error_deg':max(angles),'RMS_change_mean':float(np.mean(rms))}
  results[corpus]={'DEV_clips':len(ids),'bootstrap_unit':'VIDEO_ID' if corpus=='How2Sign' else 'clip','groups':n,'baseline_raw':p0,'baseline_canonical':pc,'arms':arms}
 out={'results':results,'replicates':REPLICATES,'source_text_input_disclosed':True,'not_spontaneous_exploitation':True,'all_conditions_reported':True,'new_TEST_use':False,'source_encoder_differs_from_generator_layer_disclosed':True}
 (ROOT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('REAL_MOTION_LAYERS_ANALYZED',json.dumps(results),flush=True)
if __name__=='__main__':main()
