"""Generate the active manuscript's stability panels from frozen audit results."""
from pathlib import Path
from freeze_motion_release_studies import read,write,sha

D=Path('reports/cvpr_sign_stability_review_20261014');P=Path('paper/current');R=Path('reports/cvpr_sign_stability_20261014')
LABELS={'raw':'Raw','canonical':'Canonical','gaussian_support_20261014':'Gaussian support 1','gaussian_support_20261015':'Gaussian support 2','gaussian_support_20261016':'Gaussian support 3','gaussian_support_mean':'Gaussian support mean','gaussian_all_20261014':'Gaussian all240','physical_1deg':r'Physical $1^\circ$','physical_matched':'Physical matched'}


def table(filename,caption,label,header,rows):
    text='\\begin{table*}[t]\n\\centering\\small\n'+f'\\caption{{{caption}}}\\label{{{label}}}\n'
    text+='\\begin{tabular}{@{}llrrrr@{}}\\toprule\n'+header+'\\\\\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\\end{table*}\n'
    (P/(filename+'.tex')).write_text(text)


def main():
    assert not (D/'manifest_sha256.json').exists();m=read(R/'manifest_sha256.json')
    for name in ['analysis.json','rotation_ratio.json','verification.json']:assert sha(R/name)==m[name]
    assert read(R/'verification.json')['passed'];a=read(R/'analysis.json')['results'];assets=[]
    rows=[]
    for corpus,label in [('How2Sign','How2Sign'),('CSL-Daily','CSL')]:
        for condition,q in a[corpus]['individual_conditions'].items():rows.append(f"{label}&{LABELS[condition]}&{100*q['token_changed_fraction']:.1f}&{q['mean_token_edit_distance']:.3f}&{q['mean_normalized_edit_distance']:.3f}&{q['BLEU4']:.4f}\\\\")
    table('table_sign_stability_all',r'\textbf{Every real-sign stability condition.} Same full native TEST clips and frozen decoder. Changed hypotheses use words on How2Sign and characters on CSL; normalized edit distance (NED) divides Levenshtein distance by the larger output length, with minimum one. Each Gaussian field and all BLEUs are retained. Same-support noise and physical controls protect the preprocessed left hand.', 'tab:signstabilityall',r'Corpus&Condition&Changed \%&Mean edits&NED&BLEU',rows);assets.append(P/'table_sign_stability_all.tex')
    rows=[]
    for corpus,label in [('How2Sign','How2Sign'),('CSL-Daily','CSL')]:
        for condition,q in a[corpus]['primary_summary'].items():
            lo,hi=[100*v for v in q['token_changed_fraction_family20_CI95']];nl,nh=q['mean_normalized_edit_distance_family20_CI95']
            rows.append(f"{label}&{LABELS[condition]}&{100*q['token_changed_fraction']:.1f}&$[{lo:.1f},{hi:.1f}]$&{q['mean_normalized_edit_distance']:.3f}&$[{nl:.3f},{nh:.3f}]$\\\\")
    table('table_sign_stability_intervals',r'\textbf{Primary paired stability intervals.} Twenty-endpoint-adjusted intervals, 20,000 cluster bootstraps; official VIDEO\_ID on How2Sign, clip on CSL. The Gaussian support mean averages the three fixed fields per clip. Intervals condition on these TEST inputs, reader weights and noise fields; they do not estimate model-training uncertainty.', 'tab:signstabilityintervals',r'Corpus&Condition&Changed \%&95\% interval&NED&95\% interval',rows);assets.append(P/'table_sign_stability_intervals.tex')
    rows=[]
    for corpus,label in [('How2Sign','How2Sign'),('CSL-Daily','CSL')]:
        for condition,q in a[corpus]['canonical_minus_control'].items():
            d=q['token_changed_fraction'];n=q['mean_normalized_edit_distance'];lo,hi=[100*v for v in d['family8_CI95']];nl,nh=n['family8_CI95']
            rows.append(f"{label}&{LABELS[condition]}&{100*d['canonical_minus_control']:+.1f}&$[{lo:+.1f},{hi:+.1f}]$&{n['canonical_minus_control']:+.3f}&$[{nl:+.3f},{nh:+.3f}]$\\\\")
    table('table_sign_stability_contrasts',r'\textbf{Canonicalization minus matched controls.} Separate eight-endpoint-adjusted paired contrasts. Positive changes indicate a larger decoding change from canonicalization; negative changes indicate a larger change from the control. Neither sign establishes semantic accuracy or that physical changes are linguistically irrelevant.', 'tab:signstabilitycontrasts',r'Corpus&Control&$\Delta$ changed (pp)&95\% interval&$\Delta$ NED&95\% interval',rows);assets.append(P/'table_sign_stability_contrasts.tex')
    # Retain the existing learned-layer table; append the complete main-text
    # comparison to avoid conflating conditional attacks and natural decodes.
    parent=Path('reports/cvpr_coherence_review_20261014/manuscript/table_editorial_sign.tex')
    parentroot=parent.parents[1];oldm=read(parentroot/'manifest_sha256.json');assert sha(parent)==oldm[str(parent.relative_to(parentroot))]
    s=parent.read_text()
    caption=r'\textbf{Natural inputs, stability controls and learned layers.} Top: full TEST layer attacks, three fitting seeds, source text available; canonical hypotheses stay identical. Language tokenizations differ. Bottom: untargeted real-input changed percentages / NED, with Gaussian mean over three same-support fields and physical changes matched to the per-clip canonicalization norm. Paired intervals and one-degree controls are supplemental.'
    # Replace the caption only, preserving its table label and all learned rows.
    begin=s.index('\\caption{');end=s.index('\\label{',begin);s=s[:begin]+'\\caption{'+caption+'}'+s[end:]
    panel=[r'\midrule',r'\multicolumn{7}{@{}l@{}}{Real-input controls: changed hypotheses (\%) / mean normalized edit distance}\\',r'Corpus&\multicolumn{2}{c}{Canonical}&\multicolumn{2}{c}{Gaussian matched}&\multicolumn{2}{c}{Physical matched}\\']
    for corpus,label in [('How2Sign','How2Sign'),('CSL-Daily','CSL')]:
        q=a[corpus]['primary_summary'];values=[f"{100*q[c]['token_changed_fraction']:.1f} / {q[c]['mean_normalized_edit_distance']:.3f}" for c in ['canonical','gaussian_support_mean','physical_matched']]
        panel.append(label+'&'+'&'.join(r'\multicolumn{2}{c}{'+v+'}' for v in values)+r'\\')
    s=s.replace(r'\bottomrule','\n'.join(panel)+'\n'+r'\bottomrule');(P/'table_editorial_sign.tex').write_text(s);assets.append(P/'table_editorial_sign.tex')
    write(D/'presentation_evidence.json',{'passed':True,'inputs_sha256':{str(R/n):sha(R/n) for n in ['analysis.json','rotation_ratio.json','verification.json']},'generated_assets':{str(p):sha(p) for p in assets},'original_layer_rows_parent_sha256':sha(parent),'all8_conditions_both_corpora_in_supplement':True,'changed_fraction_and_edit_distance_both_in_main_table':True,'builder_sha256':sha(__file__)})
    print('SIGN_STABILITY_PRESENTATION_BUILT',len(assets),flush=True)


if __name__=='__main__':main()
