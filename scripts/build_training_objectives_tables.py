"""Generate all new manuscript numbers from frozen result JSONs."""
import json,re
from pathlib import Path
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_second_review_20261012');PAPER=Path('paper/cvpr2027_training_objectives_revision_20261012')
def read(p):return json.loads(Path(p).read_text())
def f(v):return f'{v:.2f}'
def interval(v):return '$['+','.join(map(f,v))+']$'
def table(name,caption,label,cols,headers,rows,wide=True):
 star='*' if wide else '';text='\\begin{table'+star+'}[t]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\\toprule\n'+headers+r'\\\midrule'+'\n'+'\n'.join('&'.join(row)+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{table'+star+'}\n';(PAPER/(name+'.tex')).write_text(text)
def main():
 assert not (ROOT/'manifest_sha256.json').exists();roots=[Path('reports')/n for n in ['cvpr_quotient_training_20261012','cvpr_real_motion_layers_20261012','cvpr_learned_budget_20261012','cvpr_constant_noncanonicity_20261012','cvpr_constant_distance_sphere_20261012','cvpr_ordinary_transplant_20261012','cvpr_retrieval_transfer_20261012','cvpr_second_review_confirmation_20261012']]
 for root in roots:assert (root/'manifest_sha256.json').exists()
 q=read(roots[0]/'analysis.json');test=read(roots[-1]/'analysis.json');p=test['ordinary']['primary_contrasts'];mac={'OrdExcess':f(p['raw_vs_no_CE']['excess_raw_gain']),'OrdExcessCI':interval(p['raw_vs_no_CE']['paired_ci95']),'OrdGap':f(p['raw_gap_absolute']['gap']),'OrdGapCI':interval(p['raw_gap_absolute']['paired_ci95']),'OrdCanContrast':f(p['raw_vs_canonical_CE']['excess_raw_gain']),'OrdCanContrastCI':interval(p['raw_vs_canonical_CE']['paired_ci95']),'BoundSmallGain':f(test['budget']['arms']['bound0.05']['gain']),'BoundMediumGain':f(test['budget']['arms']['bound0.1']['gain'])}
 (PAPER/'new_results_macros.tex').write_text('% Generated exclusively from frozen JSONs\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in mac.items())+'\n')
 arms={'geodesic_none_w0':'Geodesic / no CE','geodesic_raw_w1':'Geodesic / raw CE1','geodesic_canonical_w1':'Geodesic / canonical CE1'};rows=[]
 for split,summ in [('DEV',{a:q['systems'][a+'/epoch20'] for a in arms}),('TEST',test['ordinary']['systems'])]:
  for a,label in arms.items():
   d=summ[a];rows.append([split,label,f(d['raw_mean']),f(d['canonical_mean']),f(d.get('gap',d.get('raw_minus_canonical')))])
 table('table_ordinary',r'\textbf{Ordinary denoiser training, not an attack layer.} Three seeds, declared20-epoch endpoint. Rotations can change across models; each raw/canonical pair evaluates the same output. TEST confirms the gap contrast over no CE, but not absolute positive gap or contrast over canonical CE.','tab:ordinary','llrrr','Split&Motion / semantic objective&Raw&Canonical&Raw--canonical',rows)
 rows=[]
 for c in ['How2Sign','CSL-Daily']:
  d=test['real_motion_layers'][c];a=d['arms']['text_raw'];rows.append([c,f(d['baseline_raw']),f(a['raw_mean']),f(d['baseline_canonical']),f(a['gain']),interval(a['paired_ci95'])])
 table('table_real_layers',r'\textbf{Learned layers on real motion.} Full native TEST, three text-conditioned seeds. Caption access is disclosed; corpus-specific source encoders are fitted. Canonical predictions of every transformed clip are identical to their source baseline. How2Sign intervals resample VIDEO\_ID groups.','tab:reallayers','lrrrrr','Corpus&Source raw&Layer raw&Canonical&Gain&95\% interval',rows)
 rows=[]
 for name,d in q['systems'].items():
  objective,epoch=name.split('/');motion,kind,weight=re.fullmatch(r'(geodesic|mse)_(none|raw|canonical)_w(.+)',objective).groups();rows.append([motion,kind,weight,epoch.replace('epoch',''),f(d['raw_mean']),f(d['canonical_mean']),f(d['excess_raw_gain']),interval(d['excess_gain_ci95_conditional'])])
 table('table_ordinary_sweep',r'Complete PHOENIX DEV objective sweep. All means average three seeds. Excess compares each objective with matched motion-only training at the same loss/epoch; intervals condition on frozen models. Only declared geodesic20-epoch endpoints define TEST confirmation.','tab:ordinarysweep','lllrrrrr','Motion loss&CE path&Weight&Epoch&Raw&Canonical&Excess&95\% interval',rows)
 rows=[]
 for a,label in arms.items():
  d=test['ordinary']['systems'][a]
  for seed,raw,can in zip([42,123,777],d['raw_per_seed'],d['canonical_per_seed']):rows.append([label, str(seed),f(raw),f(can),f(raw-can)])
 table('table_ordinary_seeds','Every ordinary-generator TEST seed; no checkpoint is selected by these scores. The raw/canonical gap varies across seeds.','tab:ordinaryseeds','lrrrr','Objective&Seed&Raw&Canonical&Gap',rows)
 rows=[]
 for name,root in [('Circle: distance + norm',roots[3]),('Sphere: distance only',roots[4])]:
  d=read(root/'analysis.json')
  for arm,label in [('text_raw','Text / raw CE'),('motion_raw','Motion / raw CE'),('text_canonical','Text / canonical CE')]:
   a=d['arms'][arm];rows.append([name,label,f(a['raw_mean']),f(a['raw_gain']),interval(a['paired_gain_ci95']),interval(a['seed_and_clip_gain_ci95'])])
 table('table_constrained_layers','Fixed-distance/norm controls, full PHOENIX DEV519, three seeds. All canonical hypotheses remain identical. Neither text-conditioned raw-loss layer gives a reliable positive gain.','tab:constrained','llrrrr','Constraint&Condition&Raw&Gain&Paired95\%&Seed+clip95\%',rows)
 rows=[]
 for c,d in test['real_motion_layers'].items():
  for arm,label in [('text_raw','Text / raw CE'),('motion_raw','Motion / raw CE'),('text_canonical','Text / canonical CE')]:
   a=d['arms'][arm];rows.append([c,label,str(len(a['seeds'])),f(a['raw_mean']),f(a['gain']),interval(a['paired_ci95']),interval(a['seed_and_group_ci95'])])
 table('table_real_layer_controls','Native real-motion TEST layers and every control. Controls use one seed, so their intervals do not estimate seed variation. All canonical predictions stay fixed.','tab:realcontrols','llrrrrr','Corpus&Objective&Seeds&Raw&Gain&Paired95\%&Seed+group95\%',rows)
 rows=[]
 for bound,a in test['budget']['arms'].items():
  rms=sum(a['RMS_to_canonical_per_seed'])/3;aucs=a['RMS_detector_best_AUC_per_seed'];rows.append([bound.replace('bound',''),f(a['raw_mean']),f(a['gain']),interval(a['paired_ci95']),interval(a['seed_and_group_ci95']),f'{rms:.4f}',f'{min(aucs):.3f}--{max(aucs):.3f}'])
 table('table_learned_budget','Small-bound PHOENIX TEST layers, all three seeds; baseline6.3905. RMS is distance to canonical features. Detector AUC uses the better threshold orientation against the same raw generator. At0.05the seed-sensitive gain interval includes zero.','tab:smallbudget','rrrrrrr','Bound&Raw&Gain&Paired95\%&Seed+clip95\%&RMS&AUC range',rows)
 for p in PAPER.glob('table_*.tex'):
  if p.name in ['table_ordinary.tex','table_real_layers.tex','table_ordinary_sweep.tex','table_ordinary_seeds.tex','table_constrained_layers.tex','table_real_layer_controls.tex','table_learned_budget.tex']:
   t=p.read_text();t=re.sub(r'(?<=\d)(?=epoch|the|[A-Z])',' ',t);t=t.replace('DEV519','DEV 519').replace('How2 Sign','How2Sign').replace('declared20','declared 20').replace('baseline6.3905','baseline 6.3905').replace('At0.05','At 0.05');p.write_text(t)
 sources={str(p):sha(p) for root in roots for p in [root/'manifest_sha256.json',root/'analysis.json']};(ROOT/'manuscript_numeric_source_map.json').write_text(json.dumps({'frozen_sources':sources,'table_files':{str(p):sha(p) for p in PAPER.glob('table_*.tex')},'generated_macros':mac,'rounding':'two decimals in main; full precision in frozen JSONs','original_tables_sources':'unchanged previous frozen paper/evidence map'},indent=2)+'\n')
 print('NEW_TABLES_AND_MACROS_READY')
if __name__=='__main__':main()
