"""Assemble current tables and figures without selecting successful endpoints.

--draft permits visibly pending release rows for layout work. Final assembly
requires every declared twenty-repetition endpoint; pending cells cannot freeze.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion_release_audit_common import read,write,sha
P=Path('paper/current');D=Path('reports/cvpr_generalization_review_20261013');OLD=Path('reports/cvpr_motion_prevalence_20261012');ADD=Path('reports/cvpr_additional_motion_audit_20261013');MD=Path('reports/cvpr_motiondiffuse_compiled_20261013')
EXTRA={('humanml','momask'):ADD/'humanml/momask',('humanml','dip'):ADD/'humanml/dip',('humanml','motiondiffuse'):MD/'humanml/motiondiffuse',('kit','momask'):ADD/'kit/momask',('kit','motiondiffuse'):MD/'kit/motiondiffuse'}
NAMES={'mdm50':'MDM50','mld':'MLD','motionlcm_v1':'MotionLCM-v1','t2mgpt':'T2M-GPT','t2m':'T2M','momask':'MoMask','dip':'DiP (prefix)','motiondiffuse':'MotionDiffuse','real_controls':'Real'}

def table(name,caption,label,columns,header,rows,wide=False):
 env='table*' if wide else 'table';text='\\begin{'+env+'}[t]\n\\centering\\footnotesize\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{@{}'+columns+'@{}}\\toprule\n'+header+'\\\\\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\\end{'+env+'}\n';(P/(name+'.tex')).write_text(text)

def summarize(root):
 rows=[read(root/f'rep_{i:02d}/metrics.json')['scores'] for i in range(20)];out={}
 for field in ['FID','R1']:
  a=np.array([r['raw']['R_precision'][0] if field=='R1' else r['raw'][field] for r in rows]);b=np.array([r['canonical']['R_precision'][0] if field=='R1' else r['canonical'][field] for r in rows]);delta=a-b;se=delta.std(ddof=1)/np.sqrt(20);out[field]={'raw':float(a.mean()),'canonical':float(b.mean()),'delta':float(delta.mean()),'family10_CI95':(delta.mean()+np.array([-1,1])*t.ppf(1-.05/20,19)*se).tolist()}
 return out

def main():
 assert not (D/'manifest_sha256.json').exists(), 'Presentation delivery is frozen; register a new delivery path before writing'
 parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');a=parser.parse_args();summaries={};pending=[]
 for (dataset,method),root in EXTRA.items():
  if not (root/'complete.json').exists():
   pending.append(dataset+'/'+method);assert a.draft,(dataset,method,'not complete');continue
  summaries[dataset+'/'+method]=summarize(root)
 if not pending:write(D/'additional_methods_analysis.json',{'additional_methods':summaries,'comparison_family':10,'original_five_study_unchanged':True,'all20repetitions':True,'not_exact_leaderboard_reproduction':True,'MotionDiffuse_final_registered_execution':str(MD),'source_sha256':sha(__file__)})
 original=read(OLD/'analysis.json')['methods'];rows=[]
 published={'mdm50':(.489,.047,r'MDM50, corrected release~\cite{tevet2023}'), 'mld':(.473,.013,r'MLD~\cite{mld2023}'), 'motionlcm_v1':(.467,.012,r'MotionLCM, one step~\cite{motionlcm2024}'), 't2mgpt':(.141,.005,r'T2M-GPT, $\tau\sim U[0,1]$~\cite{t2mgpt2023}'), 't2m':(1.087,.021,r'T2M, estimated length~\cite{guo2022}')}
 scale=[]
 for method,(mean,margin,label) in published.items():
  delta=original[method]['FID']['paired_raw_minus_canonical'];scale.append(f'{label}&${mean:.3f}\\pm{margin:.3f}$&{delta:+.4f}'+r'\\')
 more={('humanml','momask'):(.045,.002,r'MoMask / HumanML~\cite{momask2024}'),('humanml','dip'):(.228,.027,r'DiP / HumanML, corrected release~\cite{closd2025}'),('humanml','motiondiffuse'):(.630,.001,r'MotionDiffuse / HumanML, preprint~\cite{motiondiffuse2024}'),('kit','momask'):(.204,.011,r'MoMask / KIT, final table~\cite{momask2024}'),('kit','motiondiffuse'):(1.954,.062,r'MotionDiffuse / KIT, preprint~\cite{motiondiffuse2024}')}
 for (dataset,method),(mean,margin,label) in more.items():
  key=dataset+'/'+method;delta=f"{summaries[key]['FID']['delta']:+.4f}" if key in summaries else r'\textit{Pending}'
  scale.append(f'{label}&${mean:.3f}\\pm{margin:.3f}$&{delta}'+r'\\')
 table('table_published_scale',r'\textbf{Published margins give scale, not paired significance.} Author-reported FID and its reported 95\% mean margin, beside our signed same-output raw$-C$ gap. Original five rows use HumanML3D; other datasets are labeled. Samples, readers, checkpoints and conditioning differ. MDM50/DiP use corrected author results; T2M uses estimated length. DiP is prefix-conditioned. These numbers do not test our gap against a published uncertainty or establish leaderboard reproduction.','tab:publishedscale','lrr',r'Method/dataset/setting&Published FID&Our raw$-C$',scale,True)
 for method,r in original.items():
  f=r['FID'];lo,hi=f['paired_family_CI95'];rows.append(f"{NAMES[method]}&{f['raw_mean']:.3f}&{f['canonical_mean']:.3f}&{f['paired_raw_minus_canonical']:+.4f}&$[{lo:+.4f},{hi:+.4f}]$\\\\")
 for method in ['momask','dip','motiondiffuse']:
  key='humanml/'+method
  if key not in summaries:rows.append(NAMES[method]+r'&\multicolumn{4}{c}{\textit{Running; excluded from frozen claims}}\\');continue
  f=summaries[key]['FID'];lo,hi=f['family10_CI95'];lower=f'{lo:+.5f}' if 0<abs(lo)<.0001 else f'{lo:+.4f}';rows.append(f"{NAMES[method]}&{f['raw']:.3f}&{f['canonical']:.3f}&{f['delta']:+.4f}&$[{lower},{hi:+.4f}]$\\\\")
 table('table_generalization_releases',r'\textbf{Identical public outputs receive different FID.} Twenty paired HumanML3D draws. $\Delta=$raw$-C$; original five and additional HumanML/KIT studies each adjust ten endpoints. DiP receives a source prefix. No attack or fitting is added.','tab:generalreleases','lrrrr',r'Release&Raw&$C$&$\Delta$&95\% interval',rows)
 rows=[]
 for dataset,method in EXTRA:
  key=dataset+'/'+method
  if key not in summaries:rows.append(f'{dataset}/{NAMES[method]}'+r'&\multicolumn{6}{c}{\textit{Running}}\\');continue
  f=summaries[key]['FID'];r=summaries[key]['R1'];lo,hi=f['family10_CI95'];rl,rh=r['family10_CI95'];rows.append(f"{dataset}/{NAMES[method]}&{f['raw']:.4f}&{f['canonical']:.4f}&$[{lo:+.5f},{hi:+.5f}]$&{r['raw']:.4f}&{r['canonical']:.4f}&$[{rl:+.4f},{rh:+.4f}]$\\\\")
 table('table_additional_motion_full',r'\textbf{Additional releases and KIT.} All 20 draws, own native evaluator normalization, same-output raw/C. Intervals for raw$-C$ adjust ten FID/R1 comparisons. HumanML MoMask uses 4,640 outputs; MotionDiffuse/DiP use 1,024 with full reference. KIT uses 704 outputs/reference. DiP is prefix-conditioned; published-score reproduction is not claimed.','tab:additionalfull','lrrrrrr',r'Data/release&Raw FID&$C$ FID&FID gap interval&Raw R1&$C$ R1&R1 gap interval',rows,True)
 nat=read('reports/cvpr_tmr_audit_20261013/analysis.json')['methods'];rows=[]
 for method,r in nat.items():
  q=r['R1'];lo,hi=q['family12_CI95'];c=r['matched_cosine'];cl,ch=c['family12_CI95'];rows.append(f"{NAMES[method]}&{q['raw']:.4f}&{q['canonical']:.4f}&$[{lo:+.4f},{hi:+.4f}]$&{c['raw']:.4f}&{c['canonical']:.4f}&$[{cl:+.5f},{ch:+.5f}]$\\\\")
 table('table_tmr_natural',r'\textbf{Different architecture, modest natural dependence.} Native TMR cosine/sample-mean inference on compatible existing cohorts; 20 draws, family12 intervals for raw$-C$. This is not the full official annotation/filter protocol.','tab:tmrnatural','lrrrrrr',r'Source&Raw R1&$C$ R1&R1 gap interval&Raw cosine&$C$ cosine&Cosine gap interval',rows,True)
 compact=[]
 for method,r in nat.items():
  if method=='real_controls':continue
  q=r['R1'];lo,hi=q['family12_CI95'];compact.append(f"{NAMES[method]}&{q['raw']:.4f}&$[{lo:+.4f},{hi:+.4f}]$\\\\")
 table('table_tmr_natural_compact',r'\textbf{Natural sensitivity in TMR is small.} Original five releases, 20 paired draws; intervals for raw$-C$ adjust twelve R1/cosine endpoints. Compatible cohorts, native cosine inference.','tab:tmrnaturalcompact','lrr',r'Release&Raw R1&95\% gap interval',compact)
 attack=read('reports/cvpr_tmr_attack_20261013/analysis.json');rows=[];full=[]
 for source,r in attack.items():
  rows.append(f"{source.upper()}&{r['identity']['R_precision'][0]:.3f}&{r['actual']['actual']['R_precision'][0]:.3f}&{r['actual']['canonical']['R_precision'][0]:.3f}\\\\")
  for condition in ['actual','wrong','canonical_gradient','generic_L2']:
   q=r[condition];wrong=q.get('wrong',{}).get('R_precision',[float('nan')])[0];wt='---' if not np.isfinite(wrong) else f'{wrong:.3f}';full.append(f"{source.upper()}&{condition.replace('_',' ')}&{q['actual']['R_precision'][0]:.3f}&{wt}&{q['canonical']['R_precision'][0]:.3f}&{q['actual']['matched_cosine']:.3f}\\\\")
 table('table_tmr_capability',r'\textbf{TMR can be redirected at fixed movement.} First 128 clips/source, actual-caption attack, 80 steps, bound 0.35. $C$ applies to the attacked output. These are target-aware capability scores, not production quality.','tab:tmrcapability','lrrr',r'Source&Original R1&Attack R1&$C$ attack',rows)
 table('table_tmr_capability_full',r'\textbf{All TMR attack controls.} Each source has 128 fixed clips. Wrong targets are a circular caption shift. The generic attack uses a standardized L2 radius matched per clip to the rotation-fiber attack and changes continuous features; its geometry need not be cross-block coherent.','tab:tmrattackfull','llrrrr',r'Source&Condition&Actual R1&Wrong R1&$C$ actual R1&Actual cosine',full,True)
 cal=read('reports/cvpr_physical_calibration_20261013/analysis.json')['conditions'];rows=[]
 labels={'original_vs_coherent_zero':'Original vs reconstruction','coherent_zero':'Coherent zero', 'C_zero':r'$C$ of coherent zero', 'global_plus010':r'Fixed recoding $+0.1$', 'global_minus010':r'Fixed recoding $-0.1$', 'global_plus035':r'Fixed recoding $+0.35$', 'global_minus035':r'Fixed recoding $-0.35$', 'random035_seed42':r'Random recoding $0.35$'}
 for name,r in cal.items():
  label=labels.get(name)
  if label is None:
   mode,dose=name.split('_');label=('Fixed axes' if mode=='static' else 'Framewise axes')+r', $'+dose.removesuffix('deg')+r'^\circ$'
  rows.append(f"{label}&{r['FID']['mean']:.4g}&{r['R1']['mean']:.4f}&{r['MM_distance']['mean']:.4f}\\\\")
 table('table_physical_calibration',r'\textbf{Coherent physical calibration.} Means across 20 full real cohorts; paired coherent zero reference, common four-frame crop. Original-to-zero cost is explicit. Doses are local nonroot rotation angles, with fixed-per-clip or independent-per-frame axes. All recodings preserve that reconstructed movement.','tab:physicalcal','lrrr',r'Condition&FID&R1&MM-dist.',rows)
 cross=read('reports/cvpr_blind_transfer_20261013/cross_reader_analysis.json');rows=[]
 for name,r in cross.items():
  for reader,key in [('Source','source_reader_metrics'),('Transfer','conditions')]:
   q=r[key];vals=[q[c]['raw']['FID'] for c in ['identity','blind_OT_bound010','blind_OT_bound035','canonical_reader_OT_bound035','random_bound035_seed42']];rows.append(name+'&'+reader+'&'+'&'.join(f'{v:.4f}' for v in vals)+r'\\')
 table('table_blind_transfer_full',r'\textbf{Sealed blind attacks on both generators and checkpoints.} FID for 960 outputs/cohort; both budgets, canonical-gradient and random controls. Destination checkpoint supplies no gradients. MDM groups overlap, so there is no pooled independent-clip interval.','tab:blindtransfer','llrrrrr',r'Cohort&Reader&Original&Bound0.1&Bound0.35&$C$ grad.&Random',rows,True)
 grid=read('reports/cvpr_reward_dev_search_20261013/selection.json');rows=[]
 for r in grid['all_DEV_rows']:
  q=r['scores'];rows.append(f"{r['lr']:g}&{r['weight']:g}&{q['raw']['R_precision'][0]:.4f}&{q['canonical']['R_precision'][0]:.4f}&{q['raw']['FID']:.4f}&{q['canonical']['FID']:.4f}\\\\")
 table('table_reward_dev_grid',r'\textbf{Finite DEV calibration fails its retrieval criterion.} Seed 42, 500 steps, 512 VAL outputs, full 1,504-reference. All eight fits remain included. No setting qualifies; no conditional TEST training/scoring follows.','tab:rewarddevgrid','rrrrrr',r'lr&Weight&Raw R1&$C$ R1&Raw FID&$C$ FID',rows)
 cost=read('reports/cvpr_motion_defense_cost_20261013/analysis.json')['real_data_R1'];rows=[]
 for name,r in cost.items():rows.append(name.replace('_',' ')+f"&{r['raw_R1']:.5f}&{r['canonical_R1']:.5f}&{r['canonical_minus_raw']:+.5f}\\\\")
 table('table_motion_defense_cost',r'\textbf{Measured real-motion accuracy cost.} Twenty paired native draws. HumanML primary/secondary share an architecture; TMR differs. KIT uses its 251-feature evaluator. All observed R1 gaps are zero, with zero-width paired intervals.','tab:defensecost','lrrr',r'Data/reader&Raw R1&$C$ R1&$C-$raw',rows)
 plt.rcParams.update({'pdf.fonttype':42,'font.family':'Liberation Sans','font.size':8.5});fig,axes=plt.subplots(1,2,figsize=(7.05,2.6),gridspec_kw={'width_ratios':[1.15,1]});angles=[.1,.3,1,3,10,30]
 for mode,label,color in [('static','Fixed axes through time','#2479aa'),('iid','Independent frame axes','#c77a24')]:
  values=[cal[f'{mode}_{degree:g}deg']['FID']['mean'] for degree in angles];axes[0].plot(angles,values,'o-',label=label,color=color,ms=3)
 for name,label,color in [('global_plus035','Same motion: fixed+0.35','#497741'),('global_minus035','Same motion: fixed−0.35','#8c5d99')]:axes[0].axhline(cal[name]['FID']['mean'],ls='--',color=color,label=label,lw=1)
 axes[0].set_xscale('log');axes[0].set_yscale('log');axes[0].set_xticks(angles,[str(v) for v in angles]);axes[0].set_ylim(1e-7,150);axes[0].set_xlabel('Local physical angular dose (degrees)');axes[0].set_ylabel('FID vs coherent zero');axes[0].legend(fontsize=6.8,loc='lower right',frameon=False);axes[0].set_title('Noise spectrum changes physical sensitivity',fontsize=8.5)
 locations=np.arange(4);width=.19
 for j,(key,label,color) in enumerate([('source_reader_metrics','Source reader','#2479aa'),('conditions','Untargeted checkpoint','#d17a39')]):
  ratios=[cross[n][key]['blind_OT_bound010']['raw']['FID']/cross[n][key]['identity']['raw']['FID'] for n in cross];axes[1].bar(locations+(j-.5)*width,ratios,width,color=color,label=label)
  harmful=[cross[n][key]['blind_OT_bound035']['raw']['FID']/cross[n][key]['identity']['raw']['FID'] for n in cross];axes[1].scatter(locations+(j-.5)*width,harmful,s=20,marker='x',color=color)
 axes[1].axhline(1,color='#75818a',lw=.8);axes[1].set_xticks(locations,['MDM42','MDM4242','MDM7777','MLD'],rotation=20,ha='right',fontsize=7);axes[1].set_ylim(0,4.8);axes[1].set_ylabel('FID / original FID');axes[1].set_title('Blind attack transfers; larger budget hurts',fontsize=8.5);axes[1].legend(fontsize=6.8,frameon=False,loc='upper left');axes[1].text(.98,.78,'Bars: bound 0.1\n×: bound 0.35',transform=axes[1].transAxes,ha='right',fontsize=7,color='#526174')
 for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 fig.tight_layout(w_pad=2.);fig.savefig(P/'figures/physical_and_blind_controls.pdf',bbox_inches='tight');fig.savefig(D/'physical_and_blind_controls.png',dpi=180,bbox_inches='tight');plt.close(fig)
 write(D/'presentation_status.json',{'final_endpoints_complete':not pending,'pending':pending,'draft_layout_only':bool(pending),'source_sha256':sha(__file__),'all_conditions_included':True})
 print('GENERALIZATION_PRESENTATION',{'pending':pending},flush=True)
if __name__=='__main__':main()
