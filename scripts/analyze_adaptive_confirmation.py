import json
from pathlib import Path
import numpy as np
from analyze_second_review_results import Bootstrap,ci,N
A=Path('reports/cvpr_adaptive_distance_confirmation_20261012')
def read(p):return json.loads(Path(p).read_text())
def write(root,n,v):(root/n).write_text(json.dumps(v,indent=2)+'\n')
def interval(x,f=1):return ci(x,f)
def adaptive():
 assert not (A/'manifest_sha256.json').exists() and read(A/'complete.json')['conditions']==12
 base=read(A/'baseline_raw.json');can=read(A/'baseline_canonical.json');out={}
 for p in sorted((A/'evaluation').glob('*.json')):
  r=read(p);assert r['raw']['clip_ids']==base['clip_ids'];assert r['canonical']['hypotheses']==can['hypotheses']
  boot=Bootstrap(r['targets']);p0,b0=boot.score(base['hypotheses']);pa,ba=boot.score(r['raw']['hypotheses']);actual=Bootstrap(base['references']);pp,bb=actual.score(r['raw']['hypotheses']);q0,qb=actual.score(base['hypotheses']);diff=ba-b0
  out[p.stem]={'baseline_target_BLEU':p0,'attacked_target_BLEU':pa,'target_gain':pa-p0,'target_gain_paired_ci95':interval(diff),'target_gain_family_ci95_four_endpoints':interval(diff,4),
   'actual_BLEU':pp,'actual_gain':pp-q0,'actual_gain_paired_ci95':interval(bb-qb),'canonical_hypothesis_changes':r['canonical_hypothesis_changes'],
   'RMS_best_orientation_AUC':r['RMS_best_orientation_AUC'],'mean_source_RMS':float(np.mean(r['source_RMS'])),'mean_attacked_RMS':float(np.mean(r['attacked_RMS'])),
   'max_canonical_distance_squared_difference':r['canonical_distance_squared_max_difference'],'max_6D_squared_norm_difference':r['squared_norm_max_difference']}
 write(A,'analysis.json',{'clips':642,'split':'TEST','replicates':N,'conditions':out,'all_optimized_endpoints_retained':True,'per_clip_target_optimization_not_amortized_or_ordinary_training':True,'scalar_statistic_matching_not_general_undetectability':True})
 print('ADAPTIVE_ANALYZED',json.dumps({k:(v['target_gain'],v['target_gain_family_ci95_four_endpoints']) for k,v in out.items()}),flush=True)
if __name__=='__main__':adaptive()
