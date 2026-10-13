"""Post-hoc source-name-prefix cluster sensitivity, without changing primary CI.

Prefix strips the native sentence suffix; it is not verified signer metadata.
"""
import json,re
from pathlib import Path
import numpy as np
from analyze_second_review_results import Bootstrap,ci,joint
from verify_signspark_studies import sha
R=Path('reports/cvpr_critical_review_20261012')
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (R/'manifest_sha256.json').exists(),'frozen report: choose a new output root'
 A=Path('reports/cvpr_adaptive_distance_confirmation_20261012');S=Path('reports/cvpr_seed_dose_response_20261012');base=read(A/'baseline_raw.json');ids=base['clip_ids'];assert all(re.fullmatch(r'.+-\d+',x) for x in ids)
 groups=[x.rsplit('-',1)[0] for x in ids];boot=Bootstrap(base['references'],groups);out={}
 for name in ['circle_correct_step200','circle_wrong_step200','sphere_correct_step200','sphere_wrong_step200']:
  r=read(A/f'evaluation/{name}.json');b=Bootstrap(r['targets'],groups);p0,B0=b.score(base['hypotheses']);p1,B1=b.score(r['raw']['hypotheses']);d=B1-B0
  out[name]={'gain':p1-p0,'prefix_group_ci95':ci(d),'prefix_group_family_ci95_four_endpoints':ci(d,4)}
 p=read(S/'protocol.json');new=p['primary_new_seeds'];old=p['historical_seeds'];cache={}
 for s in new+old:
  if s in new:r=read(S/f'training/seed{s}/geodesic_raw_w1/test_epoch20/result.json')
  else:r=read(f'reports/cvpr_second_review_confirmation_20261012/ordinary/seed{s}_geodesic_raw_w1/result.json')
  assert r['raw']['clip_ids']==ids
  _,a=boot.score(r['raw']['hypotheses']);_,b=boot.score(r['canonical']['hypotheses']);cache[s]=a-b
 out['primary_seven_new_seeds']={'seed_and_prefix_group_ci95':ci(joint(np.stack([cache[s] for s in new],1)))}
 out['historical_plus_new_ten_seeds']={'seed_and_prefix_group_ci95':ci(joint(np.stack([cache[s] for s in old+new],1)))}
 for w in [1.,3.,10.]:
  for e in [20,40]:
   bs=[];ps=[]
   for s in old:
    r=read(S/f'training/seed{s}/geodesic_raw_w{w:g}/test_epoch{e}/result.json');pa,a=boot.score(r['raw']['hypotheses']);pb,b=boot.score(r['canonical']['hypotheses']);bs.append(a-b);ps.append(pa-pb)
   d=np.stack(bs,1);out[f'weight{w:g}/epoch{e}']={'gap':float(np.mean(ps)),'seed_and_prefix_group_ci95':ci(joint(d)),'seed_and_prefix_group_family_ci95_eleven_endpoints_contrasts':ci(joint(d),11)}
 result={'scope':'post-hoc cluster sensitivity; primary registered clip intervals unchanged','grouping':'native clip-name prefix before final -sentence_number; not independently verified signer or source-video metadata','clips':len(ids),'groups':len(set(groups)),'group_sizes':{g:groups.count(g) for g in sorted(set(groups))},'replicates':20000,'results':out,'inputs_sha256':{str(p):sha(p) for p in [A/'manifest_sha256.json',S/'manifest_sha256.json',Path(__file__)]}}
 (R/'phoenix_cluster_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
