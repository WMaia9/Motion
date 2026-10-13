"""Same-output BLEU gaps, declared seed cohorts, all exploratory dose endpoints."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import t
from analyze_second_review_results import Bootstrap,ci,joint,N
from verify_signspark_studies import sha
R=Path('reports/cvpr_seed_dose_response_20261012')
A=Path('reports/cvpr_adaptive_distance_attack_20261012')
def read(p):return json.loads(Path(p).read_text())
def write(root,n,v):(root/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def interval(x,f=1):return ci(x,f)
def summary(ps,bs,seeds,family=1):
 ps=np.array(ps);bs=np.stack(bs,1);gap=ps[:,0]-ps[:,1];g=bs[:,:,0]-bs[:,:,1]
 spread=gap.std(ddof=1) if len(seeds)>1 else 0.;delta=t.ppf(.975,len(seeds)-1)*spread/np.sqrt(len(seeds)) if len(seeds)>1 else None
 out={'seeds':seeds,'raw_per_seed':ps[:,0].tolist(),'canonical_per_seed':ps[:,1].tolist(),'gaps_per_seed':gap.tolist(),'raw_mean':float(ps[:,0].mean()),'canonical_mean':float(ps[:,1].mean()),'gap':float(gap.mean()),'paired_clip_ci95':interval(g.mean(1)),'seed_and_clip_ci95':interval(joint(g)),
 'family_adjusted_seed_and_clip_ci95':interval(joint(g),family),'positive_seed_count':int((gap>0).sum()),'seed_SD':float(spread),'seed_only_t_ci95':[float(gap.mean()-delta),float(gap.mean()+delta)] if delta is not None else None,
 'leave_one_seed_out_means':[float(np.delete(gap,i).mean()) for i in range(len(seeds))] if len(seeds)>1 else []}
 return out,g

def seed_dose():
 assert not (R/'manifest_sha256.json').exists()
 for n in ['dev_complete.json','test_complete.json']:assert read(R/n)['conditions']==32
 protocol=read(R/'protocol.json');new=protocol['primary_new_seeds'];old=protocol['historical_seeds'];out={}
 for split in ['dev','test']:
  first=read(R/f'training/seed{new[0]}/geodesic_raw_w1/{split}_epoch20/result.json')['raw'];boot=Bootstrap(first['references']);points={};boots={};rms={}
  def add(key,r):
   assert r['raw']['clip_ids']==first['clip_ids'] and r['raw']['references']==first['references']
   ps=[];bs=[]
   for encoding in ['raw','canonical']:
    p,b=boot.score(r[encoding]['hypotheses']);assert abs(p-r[encoding]['scores']['bleu']['bleu4'])<1e-10;ps.append(p);bs.append(b)
   points[key]=ps;boots[key]=np.stack(bs,1);rms[key]=r.get('mean_RMS')
  for s in new:
   for k in ['raw','canonical']:add((s,k,1.,20),read(R/f'training/seed{s}/geodesic_{k}_w1/{split}_epoch20/result.json'))
  for s in old:
   for k in ['raw','canonical']:
    if split=='test':r=read(f'reports/cvpr_second_review_confirmation_20261012/ordinary/seed{s}_geodesic_{k}_w1/result.json')
    else:
     f=Path(f'reports/cvpr_quotient_training_20261012/training/seed{s}/geodesic_{k}_w1/dev_epoch20');r={'raw':read(f/'raw_scores.json'),'canonical':read(f/'canonical_scores.json')}
    add((s,k,1.,20,'historical'),r)
   for w in [1.,3.,10.]:
    for epoch in [20,40]:add((s,'raw',w,epoch),read(R/f'training/seed{s}/geodesic_raw_w{w:g}/{split}_epoch{epoch}/result.json'))
  def cohort(seeds,k):
   keys=[(s,k,1.,20) if s in new else (s,k,1.,20,'historical') for s in seeds]
   return summary([points[x] for x in keys],[boots[x] for x in keys],seeds)
  primary,g=cohort(new,'raw');pooled,pg=cohort(old+new,'raw');control,cg=cohort(new,'canonical')
  contrast=g-cg
  primary['matched_canonical_CE_gap_contrast']={'gap_difference':primary['gap']-control['gap'],'paired_clip_ci95':interval(contrast.mean(1)),'seed_and_clip_ci95':interval(joint(contrast))}
  doses={};dB={}
  for w in [1.,3.,10.]:
   for e in [20,40]:
    keys=[(s,'raw',w,e) for s in old];doses[f'weight{w:g}/epoch{e}'],dB[w,e]=summary([points[x] for x in keys],[boots[x] for x in keys],old,11)
    doses[f'weight{w:g}/epoch{e}']['mean_RMS_per_seed']=[rms[x] for x in keys]
  contrasts={}
  for e in [20,40]:
   for wa,wb in [(3.,1.),(10.,3.)]:
    d=dB[wa,e]-dB[wb,e];point=doses[f'weight{wa:g}/epoch{e}']['gap']-doses[f'weight{wb:g}/epoch{e}']['gap']
    contrasts[f'weight{wa:g}-weight{wb:g}/epoch{e}']={'gap_difference':point,'seed_and_clip_ci95':interval(joint(d)),'family_adjusted_seed_and_clip_ci95':interval(joint(d),11)}
  d=np.mean([dB[w,40]-dB[w,20] for w in [1.,3.,10.]],0);point=np.mean([doses[f'weight{w:g}/epoch40']['gap']-doses[f'weight{w:g}/epoch20']['gap'] for w in [1.,3.,10.]])
  contrasts['epoch40-epoch20/mean_over_weights']={'gap_difference':float(point),'seed_and_clip_ci95':interval(joint(d)),'family_adjusted_seed_and_clip_ci95':interval(joint(d),11)}
  replay={str(s):{'raw_delta':points[s,'raw',1.,20][0]-points[s,'raw',1.,20,'historical'][0],'canonical_delta':points[s,'raw',1.,20][1]-points[s,'raw',1.,20,'historical'][1]} for s in old}
  out[split]={'clips':len(first['references']),'primary_seven_new_seeds':primary,'historical_plus_new_ten_seeds':pooled,'new_seed_canonical_CE_control':control,'exploratory_dose_endpoints':doses,'exploratory_dose_contrasts':contrasts,'fresh_epoch20_replay_differences':replay}
 write(R,'analysis.json',{'replicates':N,'bootstrap_seed':42,'seed_resampling_seed':2027,'results':out,'same_model_raw_canonical_gap_is_primary':True,'not_project_wide_pristine_TEST':True,'all_conditions_retained':True,'one_shared_pretraining_initialization':True,'no_venue_acceptance_probability_estimated':True})
 print('SEED_REPLICATION_ANALYZED',json.dumps(out['test']['primary_seven_new_seeds']),flush=True)

def adaptive():
 assert not (A/'manifest_sha256.json').exists() and read(A/'complete.json')['conditions']==16
 base=read(A/'baseline_raw.json');can=read(A/'baseline_canonical.json');out={}
 for p in sorted((A/'evaluation').glob('*.json')):
  r=read(p);assert r['raw']['clip_ids']==base['clip_ids'];assert r['canonical']['hypotheses']==can['hypotheses']
  boot=Bootstrap(r['targets']);p0,b0=boot.score(base['hypotheses']);pa,ba=boot.score(r['raw']['hypotheses']);actual=Bootstrap(base['references']);pp,bb=actual.score(r['raw']['hypotheses']);q0,qb=actual.score(base['hypotheses']);diff=ba-b0
  out[p.stem]={'baseline_target_BLEU':p0,'attacked_target_BLEU':pa,'target_gain':pa-p0,'target_gain_paired_ci95':interval(diff),'target_gain_family_ci95_eight_endpoints':interval(diff,8),
   'actual_BLEU':pp,'actual_gain':pp-q0,'actual_gain_paired_ci95':interval(bb-qb),'canonical_hypothesis_changes':r['canonical_hypothesis_changes'],
   'RMS_best_orientation_AUC':r['RMS_best_orientation_AUC'],'mean_source_RMS':float(np.mean(r['source_RMS'])),'mean_attacked_RMS':float(np.mean(r['attacked_RMS'])),
   'max_canonical_distance_squared_difference':r['canonical_distance_squared_max_difference'],'max_6D_squared_norm_difference':r['squared_norm_max_difference']}
 write(A,'analysis.json',{'clips':64,'split':'DEV','replicates':N,'conditions':out,'all_optimized_endpoints_retained':True,'per_clip_target_optimization_not_amortized_or_ordinary_training':True,'scalar_statistic_matching_not_general_undetectability':True})
 print('ADAPTIVE_ANALYZED',json.dumps({k:(v['target_gain'],v['target_gain_family_ci95_eight_endpoints']) for k,v in out.items()}),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['seed_dose','adaptive']);args=parser.parse_args();globals()[args.phase]()
