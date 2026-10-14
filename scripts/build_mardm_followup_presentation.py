"""Build figures/tables from completed, verified and frozen follow-up evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import build_editorial_presentation as editorial

P = Path('paper/current')
D = Path('reports/cvpr_mardm_followup_20261013')
NAMES = {'mdm50':'MDM50', 'mld':'MLD', 'motionlcm_v1':'MotionLCM', 'motiondiffuse':'MotionDiffuse', 't2mgpt':'T2M-GPT', 'momask':'MoMask', 't2m':'T2M', 'dip':'DiP (prefix)'}


def main():
    assert not (D / 'manifest_sha256.json').exists()
    D.mkdir(exist_ok=True)
    read = editorial.read_frozen; table = editorial.table
    tmr = read('reports/cvpr_tmr1024_controls_20261013/analysis.json')
    guo = read('reports/cvpr_humanml3d_audit_20261012/test_analysis.json')
    mechanism = read('reports/cvpr_rotation_mechanism_20261013/analysis.json')
    essential = read('reports/cvpr_mardm_readers_20261013/analysis.json')
    blocks = {r:read(f'reports/cvpr_matched_blocks_20261013/{r}/analysis.json')['conditions'] for r in ['guo', 'tmr']}
    read('reports/cvpr_tmr1024_controls_20261013/verification.json')
    read('reports/cvpr_rotation_mechanism_20261013/verification.json')
    read('reports/cvpr_mardm_readers_20261013/verification.json')
    read('reports/cvpr_matched_blocks_20261013/verification.json')
    old = read('reports/cvpr_motion_prevalence_20261012/analysis.json')
    extra = read('reports/cvpr_generalization_review_20261013/additional_methods_analysis.json')['additional_methods']
    # Values and settings were source-verified in the previous frozen review.
    # Keep that table's fingerprint in this figure/table evidence map too.
    prior = Path('reports/cvpr_editorial_review_20261013/manuscript/table_published_scale.tex')
    inv = json.loads((prior.parents[1]/'manifest_sha256.json').read_text())
    assert editorial.sha(prior) == inv[str(prior.relative_to(prior.parents[1]))]
    editorial.INPUTS[str(prior)] = editorial.sha(prior)
    margins = {'mdm50':r'$0.489\pm.047$', 'mld':r'$0.473\pm.013$', 'motionlcm_v1':r'$0.467\pm.012$', 't2mgpt':r'$0.141\pm.005$', 't2m':r'$1.087\pm.021$', 'momask':r'$0.045\pm.002$', 'dip':r'$0.228\pm.027$', 'motiondiffuse':r'$0.630\pm.001$'}
    rows = [r'\multicolumn{6}{@{}l}{\emph{HumanML3D}}\\']
    for name, q in old['methods'].items():
        f=q['FID']; a,b=f['paired_family_CI95']
        rows.append(f"{NAMES[name]}&{f['raw_mean']:.3f}&{f['canonical_mean']:.3f}&{f['paired_raw_minus_canonical']:+.4f}&$[{a:+.5f},{b:+.5f}]$&{margins[name]}\\\\")
    for dataset in ['humanml','kit']:
        if dataset=='kit':rows.append(r'\midrule\multicolumn{6}{@{}l}{\emph{Native KIT-ML}}\\')
        for key,q in extra.items():
            data,name=key.split('/')
            if data!=dataset:continue
            f=q['FID']; a,b=f['family10_CI95']; pub=margins[name] if dataset=='humanml' else r'$0.204\pm.011$' if name=='momask' else r'$1.954\pm.062$'
            rows.append(f"{NAMES[name]}&{f['raw']:.3f}&{f['canonical']:.3f}&{f['delta']:+.4f}&$[{a:+.5f},{b:+.5f}]$&{pub}\\\\")
    table('table_editorial_natural',r'\textbf{Public outputs change FID at identical movement.} Twenty paired draws per release; $\Delta=\mathrm{raw}-\C$. Original five and additional HumanML/KIT studies each adjust ten FID/R1 endpoints separately. DiP uses a source prefix. Published FID and author margins give scale only: samples, readers and settings differ, precluding a cross-paper significance test. MDM50/DiP use corrected author results; T2M uses estimated length and MotionDiffuse its release preprint. Full sources/settings are supplemental.', 'tab:editorialnatural', 'lrrrrr', r'Release&Raw&$\C$&$\Delta$&95\% gap interval&Published FID', rows, True)
    rows = []
    for source, label in [('mdm','MDM'), ('real','Real')]:
        q = guo['results'][source]['target_caption']; rows.append(f"Guo&{label}&1,024&{q['R1_before']:.3f}&{q['R1_after']:.3f}\\\\")
    rows.append(r'\midrule')
    for source, label in [('mdm','MDM'), ('real','Real')]:
        q = tmr[source]; rows.append(f"TMR&{label}&1,024&{q['identity']['R_precision'][0]:.3f}&{q['actual']['actual']['R_precision'][0]:.3f}\\\\")
    table('table_editorial_retrieval',r'\textbf{Two architectures can be redirected at fixed motion.} Caption-targeted R1 on the same 1,024 clips per source; frozen readers, 80 steps, bound 0.35. Each attack accesses its evaluator gradients and target caption. Native similarities and candidate permutations differ across architectures; compare within each row.', 'tab:editorialretrieval', 'llrrr', r'Reader&Motion&Clips&Original&Recoded', rows)
    rows = []
    for source, label in [('real','Real'), ('mdm','MDM')]:
        q = tmr[source]
        rows.append(f"{label}&original&{q['identity']['R_precision'][0]:.3f}&---&{q['canonical']['R_precision'][0]:.3f}&{q['identity']['matched_cosine']:.3f}\\\\")
        for condition, caption in [('actual','actual target'), ('wrong','wrong target'), ('canonical_gradient','canonical gradient'), ('generic_L2','generic L2')]:
            a = q[condition]; wrong = f"{a['wrong']['R_precision'][0]:.3f}" if 'wrong' in a else '---'
            rows.append(f"{label}&{caption}&{a['actual']['R_precision'][0]:.3f}&{wrong}&{a['canonical']['R_precision'][0]:.3f}&{a['actual']['matched_cosine']:.3f}\\\\")
    table('table_tmr_capability_full',r'\textbf{All TMR attack controls on 1,024 clips/source.} Wrong targets are a circular caption shift. The generic attack matches the rotation-fiber attack\'s per-clip standardized L2 radius while changing continuous features. Canonical source and attacked R1 agree within each source; raw/canonical source scores may differ.', 'tab:tmrattackfull', 'llrrrr', r'Source&Condition&Actual R1&Wrong R1&$\C$ actual R1&Actual cosine', rows, True)
    ranking = essential['mean_FID_rankings']; order = ranking['guo_canonical']
    rows = [f"{NAMES[n]}&{ranking['guo_canonical'].index(n)+1}&{ranking['gru'].index(n)+1}&{ranking['clip'].index(n)+1}\\\\" for n in order]
    table('table_mardm_rankings',r'\textbf{Essential-only evaluation is a different remedy.} Within-reader ranks of mean embedding FID over twenty common-reference draws. Guo $\C$ retains rotations; MARDM GRU/CLIP use 67 essential features. CLIP FID is descriptive. Ranks compare these saved release settings, not published leaderboards or evaluator accuracy.', 'tab:mardmranks', 'lrrr', r'Release&Guo $\C$&E-GRU&E-CLIP', rows)
    rows = []
    for n in order:
        q = essential['methods'][n]
        rows.append(NAMES[n] + '&' + '&'.join(f"${q[r]['FID_mean']:.4f}\\, [{q[r]['t19_CI95'][0]:.4f},{q[r]['t19_CI95'][1]:.4f}]$" for r in ['guo_raw','guo_canonical','gru','clip']) + r'\\')
    table('table_mardm_reader_full',r'\textbf{Common-reference essential-reader comparison.} Mean embedding FID with nominal $t_{19}$ 95\% intervals over all twenty draws. The same saved real cohort is encoded separately by each reader; Guo is re-scored with this reference, which differs from its original independent-reference audit. FID scales differ across reader weights and architectures. E-CLIP FID is descriptive, not an official generation metric.', 'tab:mardmfull', 'lrrrr', r'Release&Guo raw&Guo $\C$&E-GRU&E-CLIP', rows, True)
    conditions = ['null_plus','null_minus','null_random','rotation','velocity','contact','position']
    labels = ['Null: fixed +','Null: fixed −','Null: random','Changed rotations','Velocities','Contacts','Positions']
    plt.rcParams.update({'pdf.fonttype':42, 'font.family':'Liberation Sans', 'font.size':9})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    for radius, marker, color in [(1., 'o', '#2d7a9c'), (3., 's', '#c87c31')]:
        for ax, reader, endpoint, factor in [(axes[0], 'guo', 'FID', 1.), (axes[1], 'tmr', 'R1', 100.)]:
            values=[]; lo=[]; hi=[]
            for c in conditions:
                q=blocks[reader][f'{c}_r{radius:g}'][endpoint]; v=q['mean_change'] * factor; a,b=np.array(q['family28_CI95']) * factor
                values.append(v);lo.append(max(v-a, 0));hi.append(max(b-v, 0))
            offset = -.11 if radius == 1 else .11
            ax.errorbar(values, np.arange(7)+offset, xerr=[lo,hi], fmt=marker, color=color, markersize=3.5, capsize=2, linewidth=1, label=fr'$\rho={radius:g}$')
    axes[0].set_xscale('symlog', linthresh=1e-5);axes[0].set_xticks([0,1e-5,1e-4,1e-3,1e-2,1e-1], ['0',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$']);axes[0].set_yticks(np.arange(7), labels);axes[1].set_yticks(np.arange(7), [])
    axes[0].set_xlabel('Guo paired-reference FID change');axes[1].set_xlabel('TMR R1 change (percentage points)')
    axes[0].set_title('Matched norm, distinct mechanisms', fontsize=9.5, pad=9);axes[1].set_title('A different architecture', fontsize=9.5, pad=9)
    axes[0].legend(fontsize=7, frameon=False, loc='upper left')
    for ax in axes:ax.invert_yaxis();ax.axvline(0, color='#777', linewidth=.8);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x', alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2);fig.savefig(P/'figures/matched_blocks.pdf',bbox_inches='tight');fig.savefig(D/'matched_blocks.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.35,2.45))
    colors={'diffusion':'#287a9d','VQ':'#cd7c2d','other_VAE':'#777','prefix_conditioned_diffusion':'#6a8e4f'}
    shifts={'mdm50':(5,7),'mld':(5,7),'motionlcm_v1':(-42,-12),'motiondiffuse':(-12,-13),'t2mgpt':(-18,7),'momask':(-33,7),'t2m':(5,-8),'dip':(5,-13)}
    for n,q in mechanism['methods'].items():
        y=q['FID']['delta'];a,b=q['FID']['CI95'];x=q['rotation_RMS_mean']
        ax.errorbar(x,y,yerr=[[y-a],[b-y]],fmt='o',markersize=4,color=colors[q['family']],capsize=2,linewidth=1)
        ax.annotate(NAMES[n],(x,y),xytext=shifts[n],textcoords='offset points',fontsize=7)
    ax.axhline(0,color='#777',linewidth=.8);ax.set_xlim(.01,.067);ax.set_ylim(-.042,.057);ax.set_xlabel('Raw-to-canonical rotation RMS');ax.set_ylabel('FID: raw − canonical');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
    fig.tight_layout();fig.savefig(P/'figures/rotation_mechanism.pdf',bbox_inches='tight');fig.savefig(D/'rotation_mechanism.png',dpi=180,bbox_inches='tight');plt.close(fig)
    rows = []
    for reader in ['guo','tmr']:
        for radius in [1.,3.]:
            for c, label in zip(conditions, labels):
                label = label.replace('−', r'$-$')
                q=blocks[reader][f'{c}_r{radius:g}']; endpoint='FID' if reader=='guo' else 'matched_cosine';a=q[endpoint];b=q['R1'];al,ah=a['family28_CI95'];bl,bh=b['family28_CI95']
                rows.append(f"{reader.upper()}&{radius:g}&{label}&{a['mean_change']:+.6f}&$[{al:+.6f},{ah:+.6f}]$&{b['mean_change']:+.6f}&$[{bl:+.6f},{bh:+.6f}]$\\\\")
    table('table_matched_blocks_full',r'\textbf{All equal-L2 endpoints.} Twenty paired draws of 4,640 entries, two preselected radii, and seven interventions. Fourth/fifth columns are Guo FID or TMR matched-cosine change, respectively; R1 is a fraction. Intervals adjust 28 endpoints per reader. The Guo contact control has zero consumed budget; non-null controls need not be physically consistent across blocks.', 'tab:matchedfull', 'rllrrrr', r'Reader&$\rho$&Intervention&$\Delta$ FID/cos&95\% interval&$\Delta$ R1&95\% interval', rows, True)
    record={'passed':True, 'inputs_sha256':editorial.INPUTS, 'all_new_studies_frozen_before_presentation':True, 'no_new_training_or_endpoint_selection':True, 'all_signs_radii_and_controls_in_main_or_supplement':True, 'generated_assets':{str(p):editorial.sha(p) for p in [P/'table_editorial_natural.tex',P/'table_editorial_retrieval.tex',P/'table_tmr_capability_full.tex',P/'table_mardm_rankings.tex',P/'table_mardm_reader_full.tex',P/'table_matched_blocks_full.tex',P/'figures/matched_blocks.pdf',P/'figures/rotation_mechanism.pdf']}}
    (D/'presentation_evidence.json').write_text(json.dumps(record,indent=2)+'\n')
    print('MARDM_PRESENTATION_FROM_FROZEN_EVIDENCE',flush=True)


if __name__=='__main__':main()
