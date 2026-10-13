"""Paired native corpus BLEU intervals for fixed confirmation and diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric
N=20000;SEEDS=[42,123,777]
def read(p):return json.loads(Path(p).read_text())
def save(root,name,d):
 (root/name).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def ci(x,family=1):return np.percentile(x,[100*.05/(2*family),100*(1-.05/(2*family))]).tolist()
def joint(x):
 if x.ndim==1:x=x[:,None]
 k=x.shape[1];draw=np.random.default_rng(2027).integers(0,k,(N,k));return np.take_along_axis(x,draw,1).mean(1)
class Bootstrap:
 def __init__(self,refs,groups=None,char=False):
  self.refs=refs;self.char=char;groups=groups or list(map(str,range(len(refs))));names=sorted(set(groups));self.n=len(names);lookup={g:i for i,g in enumerate(names)};self.index=np.array([lookup[g] for g in groups]);self.w=np.random.default_rng(42).multinomial(self.n,np.ones(self.n)/self.n,size=N)
 def score(self,h):
  r=[' '.join(list(x.replace(' ',''))) for x in self.refs] if self.char else self.refs;h0=[' '.join(list(x.replace(' ',''))) for x in h] if self.char else h;s=statistics(r,h0);g=np.zeros((self.n,10),np.int64);np.add.at(g,self.index,s);p=b4(s.sum(0));assert abs(p-metric(self.refs,h,self.char))<1e-10;return p,np.array([b4(x) for x in self.w@g])
def layers(folder,char=False,how=False,budget=False):
 base=read(folder/('baseline.json' if budget else 'baseline_raw.json'));can=base if budget else read(folder/'baseline_canonical.json');groups=None
 if how:
  path=Path('/mnt/datasets/sign_datasets/raw/how2sign/How2Sign/sentence_level/test/text/en/raw_text/how2sign_test.csv')
  with path.open() as f:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(f,delimiter='\t')}
  groups=[lookup[id] for id in base['clip_ids']]
 boot=Bootstrap(base['references'],groups,char);p0,b0=boot.score(base['hypotheses']);pc,_=boot.score(can['hypotheses']);out={}
 conditions=[(f'bound{b:g}',SEEDS) for b in [.01,.05,.1]] if budget else [('text_raw',SEEDS),('motion_raw',[42]),('text_canonical',[42])]
 for arm,seeds in conditions:
  ps=[];bs=[];rms=[];aucs=[];changes=[]
  for seed in seeds:
   r=read(folder/f'seed{seed}_{arm}.json');assert r['raw']['references']==base['references'];p,b=boot.score(r['raw']['hypotheses']);ps.append(p);bs.append(b);changes.append(r['canonical_hypothesis_changes']);rms.append(r['noncanonicity']['mean_RMS'])
   if budget:aucs.append(r['RMS_detector_best_orientation_AUC'])
  diff=np.stack(bs,1)-b0[:,None];out[arm]={'seeds':seeds,'raw_per_seed':ps,'raw_mean':float(np.mean(ps)),'gain':float(np.mean(ps)-p0),'paired_ci95':ci(diff.mean(1)),'seed_and_group_ci95':ci(joint(diff)),'canonical_hypothesis_changes':changes,'RMS_to_canonical_per_seed':rms}
  if budget:out[arm]['RMS_detector_best_AUC_per_seed']=aucs
 return {'clips':len(base['references']),'groups':boot.n,'bootstrap_unit':'VIDEO_ID' if how else 'clip','baseline_raw':p0,'baseline_canonical':pc,'arms':out}
def confirmation():
 root=Path('reports/cvpr_second_review_confirmation_20261012');assert not (root/'manifest_sha256.json').exists()
 for part in ['ordinary','real','budget']:assert (root/part/'complete.json').exists()
 source=read(root/'ordinary/seed42_geodesic_none_w0/result.json')['raw'];boot=Bootstrap(source['references']);ps={};bs={};summ={}
 for arm in ['geodesic_none_w0','geodesic_raw_w1','geodesic_canonical_w1']:
  for seed in SEEDS:
   r=read(root/f'ordinary/seed{seed}_{arm}/result.json')
   for encoding in ['raw','canonical']:
    assert r[encoding]['clip_ids']==source['clip_ids'];ps[seed,arm,encoding],bs[seed,arm,encoding]=boot.score(r[encoding]['hypotheses'])
  raw=np.array([ps[s,arm,'raw'] for s in SEEDS]);can=np.array([ps[s,arm,'canonical'] for s in SEEDS]);g=np.stack([bs[s,arm,'raw']-bs[s,arm,'canonical'] for s in SEEDS],1)
  summ[arm]={'raw_per_seed':raw.tolist(),'canonical_per_seed':can.tolist(),'raw_mean':float(raw.mean()),'canonical_mean':float(can.mean()),'gap':float((raw-can).mean()),'gap_ci95':ci(g.mean(1)),'gap_seed_and_clip_ci95':ci(joint(g))}
 raw='geodesic_raw_w1';prim={}
 for ref,name in [('geodesic_none_w0','raw_vs_no_CE'),('geodesic_canonical_w1','raw_vs_canonical_CE')]:
  diff=np.stack([bs[s,raw,'raw']-bs[s,raw,'canonical']-bs[s,ref,'raw']+bs[s,ref,'canonical'] for s in SEEDS],1)
  prim[name]={'excess_raw_gain':summ[raw]['gap']-summ[ref]['gap'],'raw_gain':summ[raw]['raw_mean']-summ[ref]['raw_mean'],'canonical_gain':summ[raw]['canonical_mean']-summ[ref]['canonical_mean'],'paired_ci95':ci(diff.mean(1)),'family_ci95_three_contrasts':ci(diff.mean(1),3),'seed_and_clip_ci95':ci(joint(diff))}
 gap=np.stack([bs[s,raw,'raw']-bs[s,raw,'canonical'] for s in SEEDS],1);prim['raw_gap_absolute']={'gap':summ[raw]['gap'],'paired_ci95':ci(gap.mean(1)),'family_ci95_three_contrasts':ci(gap.mean(1),3),'seed_and_clip_ci95':ci(joint(gap))}
 real={c:layers(root/'real'/c,char=c=='CSL-Daily',how=c=='How2Sign') for c in ['How2Sign','CSL-Daily']};budget=layers(root/'budget',budget=True)
 save(root,'analysis.json',{'replicates':N,'ordinary':{'clips':642,'systems':summ,'primary_contrasts':prim},'real_motion_layers':real,'budget':budget,'all_conditions_reported':True,'no_TEST_fitting_or_selection':True,'one_shared_pretraining_initialization':True})
 print('SECOND_TEST_ANALYZED',json.dumps(prim),flush=True)
def retrieval():
 root=Path('reports/cvpr_retrieval_transfer_20261012');assert (root/'transfer_complete.json').exists() and not (root/'manifest_sha256.json').exists();base=read(root/'evaluation/generated_raw.json');boot=Bootstrap(base['references'],char=True);p0,b0=boot.score(base['hypotheses']);ps=[];bs=[]
 for seed in SEEDS:
  r=read(root/f'transfer/seed{seed}_text_raw.json');p,b=boot.score(r['raw']['hypotheses']);ps.append(p);bs.append(b)
 diff=np.stack(bs,1)-b0[:,None];save(root,'analysis.json',{'clips':128,'baseline_raw':p0,'baseline_canonical':read(root/'evaluation/generated_canonical.json')['scores']['bleu']['bleu4'],'raw_per_seed':ps,'raw_mean':float(np.mean(ps)),'gain':float(np.mean(ps)-p0),'paired_ci95':ci(diff.mean(1)),'seed_and_clip_ci95':ci(joint(diff)),'native_retrieval_replay_not_published_BT_reproduction':True})
 print('RETRIEVAL_TRANSFER_ANALYZED',float(np.mean(ps)-p0),ci(diff.mean(1)),flush=True)
def transplant():
 root=Path('reports/cvpr_ordinary_transplant_20261012');assert (root/'complete.json').exists() and not (root/'manifest_sha256.json').exists();out={}
 for arm in ['geodesic_none_w0','geodesic_raw_w1','geodesic_canonical_w1']:
  ps=[];bs=[];changed=[]
  first=read(root/f'evaluation/seed42_{arm}.json');boot=Bootstrap(first['targets'])
  for seed in SEEDS:
   r=read(root/f'evaluation/seed{seed}_{arm}.json');p0,b0=boot.score(r['baseline_raw']['hypotheses']);p,b=boot.score(r['raw']['hypotheses']);ps.append([p0,p]);bs.append(b-b0);changed.append(r['canonical_hypothesis_changes'])
  diff=np.stack(bs,1);out[arm]={'wrong_target_bleu_before_after_per_seed':ps,'wrong_target_gain':float(np.mean(np.array(ps)[:,1]-np.array(ps)[:,0])),'paired_ci95':ci(diff.mean(1)),'seed_and_clip_ci95':ci(joint(diff)),'canonical_hypothesis_changes':changed}
 save(root,'analysis.json',{'replicates':N,'arms':out,'exploratory_after_training':True,'no_new_fitting':True,'all_conditions_reported':True});print('TRANSPLANT_ANALYZED',json.dumps(out),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['confirmation','retrieval','transplant']);a=p.parse_args();globals()[a.phase]()
