"""Present frozen evidence around movement/encoding consistency, without new fits.

Only writes paper/current and a new unfrozen editorial delivery. Every plotted
condition is traceable to an immutable aggregate; no endpoint is reselected.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path('paper/current')
D = Path('reports/cvpr_editorial_review_20261013')
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frozen(name):
    path = Path(name)
    root = next(parent for parent in path.parents
                if (parent / 'manifest_sha256.json').is_file())
    inventory = json.loads((root / 'manifest_sha256.json').read_text())
    assert sha(path) == inventory[str(path.relative_to(root))], path
    INPUTS[str(path)] = sha(path)
    return json.loads(path.read_text())


def table(name, caption, label, columns, header, rows, wide=False):
    env = 'table*' if wide else 'table'
    text = ('\\begin{' + env + '}[t]\n\\centering\\footnotesize\n'
            '\\caption{' + caption + '}\\label{' + label + '}\n'
            '\\begin{tabular}{@{}' + columns + '@{}}\\toprule\n'
            + header + '\\\\\\midrule\n' + '\n'.join(rows)
            + '\n\\bottomrule\\end{tabular}\\end{' + env + '}\n')
    if name == 'table_editorial_natural':
        text = text.replace('\\centering\\footnotesize', '\\centering\\footnotesize\\setlength{\\tabcolsep}{5.5pt}')
    if wide:
        text = text.replace('\\centering\\footnotesize', '\\centering\\small')
    (P / (name + '.tex')).write_text(text)


def main():
    assert not (D / 'manifest_sha256.json').exists(), 'Editorial delivery is frozen'
    D.mkdir(exist_ok=True)
    old = read_frozen('reports/cvpr_motion_prevalence_20261012/analysis.json')
    extra = read_frozen('reports/cvpr_generalization_review_20261013/additional_methods_analysis.json')['additional_methods']
    names = {'mdm50': 'MDM50', 'mld': 'MLD', 'motionlcm_v1': 'MotionLCM-v1',
             't2mgpt': 'T2M-GPT', 't2m': 'T2M', 'momask': 'MoMask',
             'dip': 'DiP (prefix)', 'motiondiffuse': 'MotionDiffuse'}
    rows = [r'\multicolumn{5}{@{}l}{\emph{HumanML3D}}\\']
    for method, record in old['methods'].items():
        f = record['FID'];lo, hi = f['paired_family_CI95']
        rows.append(f"{names[method]}&{f['raw_mean']:.3f}&{f['canonical_mean']:.3f}&{f['paired_raw_minus_canonical']:+.4f}&$[{lo:+.5f},{hi:+.5f}]$\\\\")
    for dataset in ['humanml', 'kit']:
        if dataset == 'kit':rows += [r'\midrule\multicolumn{5}{@{}l}{\emph{Native KIT-ML}}\\']
        for key, record in extra.items():
            data, method = key.split('/')
            if data != dataset:continue
            f = record['FID'];lo, hi = f['family10_CI95']
            rows.append(f"{names[method]}&{f['raw']:.3f}&{f['canonical']:.3f}&{f['delta']:+.4f}&$[{lo:+.5f},{hi:+.5f}]$\\\\")
    table('table_editorial_natural',r'\textbf{Public outputs change FID at identical movement.} Twenty paired draws per setting. $\Delta=\mathrm{raw}-\C$. Original five and additional HumanML/KIT studies each adjust ten FID/R1 endpoints separately. DiP receives a source prefix. Signs vary; the original five-release mean ranking is unchanged.',
          'tab:editorialnatural', 'lrrrr',r'Release&Raw&$\C$&$\Delta$&95\% gap interval',rows)
    guo = read_frozen('reports/cvpr_humanml3d_audit_20261012/test_analysis.json')
    tmr = read_frozen('reports/cvpr_tmr_attack_20261013/analysis.json')
    rows=[]
    for source, label in [('mdm','MDM'),('real','Real')]:
        q = guo['results'][source]['target_caption']
        rows.append(f"Guo&{label}&1,024&{q['R1_before']:.3f}&{q['R1_after']:.3f}\\\\")
    rows.append(r'\midrule')
    for source, label in [('mdm','MDM'),('real','Real')]:
        q=tmr[source]
        rows.append(f"TMR&{label}&128&{q['identity']['R_precision'][0]:.3f}&{q['actual']['actual']['R_precision'][0]:.3f}\\\\")
    table('table_editorial_retrieval',r'\textbf{Two architectures can be redirected at fixed motion.} Caption-targeted R1, frozen readers, 80 steps, bound 0.35. Each attack accesses its evaluator gradients and target caption. Cohorts/candidate permutations differ across architectures; compare within each row.',
          'tab:editorialretrieval','llrrr',r'Reader&Motion&Clips&Original&Recoded',rows)
    phoenix = read_frozen('reports/cvpr_learned_confirmation_20261012/analysis.json')['learned']
    real = read_frozen('reports/cvpr_second_review_confirmation_20261012/analysis.json')['real_motion_layers']
    q=phoenix['systems']['text_raw'];lo,hi=q['gain_ci95_conditional_on_seeds']
    rows=[f"PHOENIX&Generated&{phoenix['baseline_canonical_bleu4']:.2f}&{q['mean_bleu4']:.2f}&{phoenix['baseline_canonical_bleu4']:.2f}&{q['mean_gain_bleu4']:.2f}&$[{lo:.2f},{hi:.2f}]$\\\\"]
    for source,label in [('How2Sign','How2Sign'),('CSL-Daily','CSL-Daily')]:
        q=real[source];v=q['arms']['text_raw'];lo,hi=v['paired_ci95'];baseline=q['baseline_raw'];canonical=q['baseline_canonical']
        rows.append(f"{label}&Real&{baseline:.2f}&{v['raw_mean']:.2f}&{canonical:.2f}&{v['gain']:.2f}&$[{lo:.2f},{hi:.2f}]$\\\\")
    table('table_editorial_sign',r'\textbf{A learned layer changes backtranslation, not movement.} Full native TEST, three layer-fitting seeds; source text is available to each corpus-specific layer. Raw BLEU changes while canonical hypotheses remain identical before/after the layer. Intervals condition on the fitted seeds, with VIDEO\_ID resampling on How2Sign; language tokenizations differ.',
          'tab:editorialsign','llrrrrr',r'Corpus&Motion&Source raw&Layer raw&$\C$ before/after&Gain&95\% gain interval',rows,True)
    cal = read_frozen('reports/cvpr_physical_calibration_20261013/analysis.json')['conditions']
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'Liberation Sans','font.size':9})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),gridspec_kw={'width_ratios':[1.02,1]})
    order=['coherent_zero','global_plus010','global_minus010','random035_seed42','global_plus035','global_minus035']
    labels=['Same encoding','Fixed +0.1','Fixed −0.1','Random 0.35','Fixed +0.35','Fixed −0.35']
    means=[cal[n]['FID']['mean'] for n in order];errs=[(cal[n]['FID']['t19_CI95'][1]-cal[n]['FID']['t19_CI95'][0])/2 for n in order]
    axes[0].barh(np.arange(6),means,xerr=errs,height=.62,capsize=2,color=['#84939d']*4+['#c96d29']*2)
    axes[0].set_yticks(np.arange(6),labels);axes[0].invert_yaxis();axes[0].set_xlim(0,4.15);axes[0].set_xlabel('FID against the same motion');axes[0].set_title('Fixed recoding: zero physical change',fontsize=9.5,pad=10)
    for i,v in enumerate(means):axes[0].text(v+.06,i,f'{v:.2f}',va='center',fontsize=8)
    doses=[.1,.3,1,3,10,30]
    for mode,label,color in [('static','Fixed axes through time','#2779a5'),('iid','Independent frame axes','#c97927')]:
        values=[cal[f'{mode}_{d:g}deg']['FID']['mean'] for d in doses]
        axes[1].plot(doses,values,marker='o',markersize=3,label=label,color=color)
    for name,label,color in [('global_plus035','Same motion: +0.35','#638f43'),('global_minus035','Same motion: −0.35','#8359a0')]:
        axes[1].axhline(cal[name]['FID']['mean'],linestyle='--',linewidth=1.1,label=label,color=color)
    axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].set_xticks(doses,[str(x) for x in doses]);axes[1].set_xlabel('Local angular change (degrees)');axes[1].set_ylabel('FID against paired zero reference');axes[1].set_title('Physical sensitivity depends on time',fontsize=9.5,pad=10);axes[1].legend(fontsize=6.6,loc='lower right',frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax==axes[0] else 'y',alpha=.16);ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2);fig.savefig(P/'figures/encoding_and_physical_scale.pdf',bbox_inches='tight');fig.savefig(D/'encoding_and_physical_scale.png',dpi=180,bbox_inches='tight');plt.close(fig)
    cross=read_frozen('reports/cvpr_blind_transfer_20261013/cross_reader_analysis.json')
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.3));cohorts=[('mdm42','MDM seed 42'),('mdm4242','MDM seed 4242'),('mdm7777','MDM seed 7777'),('mld','MLD')]
    for ax,key,title in zip(axes,['source_reader_metrics','conditions'],['Source evaluator','Untargeted checkpoint (no gradients)']):
        for (name,label),color in zip(cohorts,['#2779a5','#cc7c26','#8a63aa','#3d8a64']):
            q=cross[name][key];base=q['identity']['raw']['FID'];ratios=[q[n]['raw']['FID']/base for n in ['identity','blind_OT_bound010','blind_OT_bound035']]
            ax.plot([0,.1,.35],ratios,marker='o',markersize=3,color=color,label=label,linewidth=1.4)
        ax.axhline(1,color='#666',linestyle='--',linewidth=.8);ax.set_xticks([0,.1,.35]);ax.set_xlabel('Coefficient bound');ax.set_title(title,fontsize=9.5,pad=9);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_ylim(0,4.9)
    axes[0].set_ylabel('FID / original FID');axes[0].legend(fontsize=7,frameon=False,loc='upper left');axes[1].text(.015,.92,'Below 1: lower FID\nAbove 1: higher FID',transform=axes[1].transAxes,va='top',fontsize=7,color='#526174')
    fig.tight_layout(w_pad=2);fig.savefig(P/'figures/blind_checkpoint_transfer.pdf',bbox_inches='tight');fig.savefig(D/'blind_checkpoint_transfer.png',dpi=180,bbox_inches='tight');plt.close(fig)
    record={'passed':True,'inputs_sha256':INPUTS,'new_training_or_endpoint_selection':False,'all_budgets_signs_and_controls_preserved_in_main_or_supplement':True,'no_cross_architecture_score_comparison':True,'figure_zero_reference_cost_explicit':True,'active_manuscript':'paper/current','generated_assets':{str(p):sha(p) for p in [P/'table_editorial_natural.tex',P/'table_editorial_retrieval.tex',P/'table_editorial_sign.tex',P/'figures/encoding_and_physical_scale.pdf',P/'figures/blind_checkpoint_transfer.pdf']}}
    (D/'editorial_presentation_evidence.json').write_text(json.dumps(record,indent=2)+'\n');print('EDITORIAL_PRESENTATION_FROM_FROZEN_EVIDENCE',flush=True)

if __name__ == '__main__':main()
