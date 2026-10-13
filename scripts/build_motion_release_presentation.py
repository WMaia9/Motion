"""Tables/figures from complete retained endpoints; edits only paper/current."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('paper/current');R=Path('reports/cvpr_motion_prevalence_20261012');D=Path('reports/cvpr_motion_release_review_20261012')
def read(p):return json.loads(Path(p).read_text())
def table(name,caption,label,columns,header,rows,wide=False):
    env='table*' if wide else 'table'
    text='\\begin{'+env+'}[t]\n\\centering\\footnotesize\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{@{}'+columns+'@{}}\\toprule\n'+header+'\\\\\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\\end{'+env+'}\n'
    (P/(name+'.tex')).write_text(text)
def main():
    x=read(R/'analysis.json');names={'mdm50':'MDM50','mld':'MLD','motionlcm_v1':'MotionLCM-v1','t2mgpt':'T2M-GPT','t2m':'T2M'}
    rows=[]
    for m,r in x['methods'].items():
        f=r['FID'];a,b=f['paired_family_CI95'];rows.append(f"{names[m]}&{f['raw_mean']:.3f}&{f['canonical_mean']:.3f}&{f['paired_raw_minus_canonical']:+.4f}&$[{a:+.4f},{b:+.4f}]$\\\\")
    table('table_motion_releases',r'\textbf{Natural released outputs change scores.} FID of identical motion before/after 6D canonicalization, 20 native TEST repetitions per method. $\Delta=$raw$-C$; intervals adjust ten primary FID/R1 endpoints. MDM emits 1,024 samples; others/reference use 4,640. No attack or fitting is added.','tab:releases','lrrrr',r'Method&Raw&$C$&$\Delta$&95\% interval',rows)
    rows=[]
    for reader,entries in [('MDM',x['methods']),('GPT',x['secondary_official_GPT_checkpoint_same_architecture'])]:
        for m,r in entries.items():
            f=r['FID'];q=r['R1'];a,b=f['paired_family_CI95'];c,d=q['paired_family_CI95']
            rows.append(f"{reader}&{names[m]}&{f['raw_mean']:.4f}&{f['canonical_mean']:.4f}&$[{a:+.4f},{b:+.4f}]$&{q['raw_mean']:.4f}&{q['canonical_mean']:.4f}&$[{c:+.4f},{d:+.4f}]$\\\\")
    table('table_motion_release_readers',r'\textbf{Two official checkpoints, one evaluator architecture.} All five releases and 20 repetitions are retained. R1 uses native 32-candidate batches. Each reader uses a ten-endpoint FID/R1 interval family; these families are reported separately. GPT weights differ from the MDM-linked weights; the samples and normalization are shared. Intervals describe raw$-C$.','tab:releasereaders','llrrrrrr',r'Reader&Release&Raw FID&$C$ FID&FID gap interval&Raw R1&$C$ R1&R1 gap interval',rows,wide=True)
    blind=read('reports/cvpr_humanml_blind_repetitions_20261012/analysis.json');rows=[]
    for c in blind['cohorts']:
        f=c['FID'];rows.append(f"{c['generation_seed']}&{f['identity']:.3f}&{f['blind_OT_bound010']:.3f}&{f['blind_OT_bound035']:.3f}&{f['canonical_reader_OT_bound035']:.3f}&{f['random_bound035_seed42']:.3f}\\\\")
    table('table_blind_repetitions',r'\textbf{Caption-blind attack repetitions.} FID for 960 outputs per cohort, common TRAIN reference. Bounds 0.1/0.35 and canonical-gradient/random controls are retained. The new cohorts overlap by 493 source groups and overlap the first by 494/496; these are not independent clip samples.','tab:blindreps','rrrrrr',r'Gen. seed&Raw&0.1&0.35&$C$-grad.&Random',rows)
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','font.size':9})
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.2),gridspec_kw={'width_ratios':[1.1,1]})
    real=x['real_fixed_controls'];entries=real.get('conditions',real)
    # Summary schema is keyed directly by encoding (with metadata alongside).
    values={}
    for rep in range(20):
        row=read(R/'real_controls'/f'rep_{rep:02d}/metrics.json')
        for name,s in row['results'].items():values.setdefault(name,[]).append(s['FID'])
    order=['identity','global_plus010','global_minus010','random035_seed42','global_plus035','global_minus035']
    labels=['Original','Fixed +0.1','Fixed −0.1','Random 0.35','Fixed +0.35','Fixed −0.35']
    assert set(order)<=set(values),set(values)
    means=[np.mean(values[n]) for n in order];err=[1.96*np.std(values[n])/np.sqrt(20) for n in order]
    axes[0].barh(np.arange(6),means,xerr=err,color=['#818b94']*4+['#d17c2e']*2,height=.6,capsize=2)
    axes[0].set_yticks(np.arange(6),labels);axes[0].invert_yaxis();axes[0].set_xlim(0,4.2);axes[0].set_xlabel('FID (same real movement)');axes[0].set_title('Fixed encodings: 20 native repetitions',fontsize=9,pad=10)
    for i,val in enumerate(means):axes[0].text(val+.09,i,f'{val:.3f}',va='center',fontsize=8)
    for c,color in zip(blind['cohorts'],['#2479aa','#c77a24','#58833e']):
        f=c['FID'];axes[1].plot([0,.1,.35],[f['identity'],f['blind_OT_bound010'],f['blind_OT_bound035']],marker='o',color=color,label=str(c['generation_seed']))
    axes[1].set_xticks([0,.1,.35]);axes[1].set_ylim(0,1.65);axes[1].set_xlabel('Coefficient bound');axes[1].set_ylabel('FID (MDM)');axes[1].set_title('Caption-blind attack: 3 cohorts',fontsize=9,pad=10);axes[1].legend(title='Generation seed',fontsize=7,title_fontsize=7,frameon=False,loc='upper left')
    for a in axes:a.spines[['top','right']].set_visible(False);a.grid(axis='x' if a is axes[0] else 'y',alpha=.15);a.set_axisbelow(True)
    fig.tight_layout(w_pad=2);fig.savefig(P/'figures/motion_release_controls.pdf',bbox_inches='tight');fig.savefig(D/'motion_release_controls.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (D/'presentation_evidence.json').write_text(json.dumps({'primary_analysis':str(R/'analysis.json'),'fixed_control_inputs':[str(R/'real_controls'/f'rep_{r:02d}/metrics.json') for r in range(20)],'blind_analysis':'reports/cvpr_humanml_blind_repetitions_20261012/analysis.json','no_hidden_conditions_or_sample_selection':True,'figure_scope':'native real fixed encodings and caption-blind cohorts; not generated-quality improvements'},indent=2)+'\n')
if __name__=='__main__':main()
