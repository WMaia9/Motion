"""Render table values from frozen scientific records, never manual score edits."""
import json
from pathlib import Path
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_manuscript_review_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence_revision_20261011')
OLD=Path('reports/cvpr_evaluation_confirmation_20261011')
def read(p):return json.loads(Path(p).read_text())
def ci(v):return f"{v['delta_bleu4']:+.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    rankings=read(OLD/'analysis.json')['rankings']
    names={'geometry':'Geometry','velocity':'Velocity','physical_ce':'Reader CE','physical_generated_near':'Generated near'}
    rows=[names[k]+' & '+' & '.join(f"{rankings['scores'][r][k]:.6f}" for r in ['raw','canonical','gauge_scale090','gauge_scale110','gauge_shear010'])+r'\\' for k in names]
    pairs=[]
    for key,v in rankings['pairwise_comparisons'].items():
        a,b=key.split(' minus ');gap=v['raw_to_canonical_change_in_pairwise_gap']
        pairs.append(names[a]+' -- '+names[b]+' & '+ci(v['raw'])+' & '+ci(v['canonical'])+' & '+f"{gap['delta_gap']:+.4f} [{gap['ci95'][0]:.4f}, {gap['ci95'][1]:.4f}]"+r'\\')
    text=r'''\begin{table*}[t]\centering\small
\caption{All twenty PHOENIX TEST scores for the same four local motion sets. Reader CE denotes canonical-reader cross-entropy. Each row's conditions encode the same rotations.}\label{tab:allgenerators}
\begin{tabular}{lrrrrr}\toprule
Variant & Raw & Canonical & Scale 0.9 & Scale 1.1 & Shear 0.1\\\midrule
ROWS
\bottomrule\end{tabular}\end{table*}
\begin{table*}[t]\centering\small
\caption{All six paired generator comparisons, first minus second: gap and 95\% interval. Bootstrap uses 2,000 resamples and seed 42. No pair has opposite rankings supported by both intervals. Analyses are unadjusted exploratory comparisons, conditional on one training seed.}\label{tab:pairs}
\begin{tabular}{llll}\toprule
Pair & Raw & Canonical & Change in gap\\\midrule
PAIRS
\bottomrule\end{tabular}\end{table*}
'''.replace('ROWS','\n'.join(rows)).replace('PAIRS','\n'.join(pairs))
    (PAPER/'tables_generated.tex').write_text(text)
    ck=read(OLD/'sources/frozen_inputs.json')['frozen_models']
    assets=[('PHOENIX BT',Path('reports/cvpr_signspark_baseline_20261010/checkpoints/PHOENIX-2014T/best.ckpt')),('How2Sign BT',Path('reports/cvpr_signspark_baseline_20261010/checkpoints/How2Sign/best.ckpt')),('CSL BT',Path('reports/cvpr_representation_paper_20261011/checkpoints/CSL-Daily/best.ckpt')),('Local CTC',Path('reports/cvpr_motion_consistency_confirmation_20261010/ctc_reader/best.pt'))]
    assets += [(names[k],Path(path)) for k,path in zip(names,list(ck)[:4])]
    assets.append(('Shared initial DDPM',Path('reports/cvpr_physical_semantic_ddpm_20261010/shared_best.pt')))
    records=[{'label':label,'path':str(path),'sha256':sha(path)} for label,path in assets]
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{Exact checkpoint SHA256 identifiers. The full hashes, configurations, dataset revisions and code snapshots accompany the artifacts.}\label{tab:hashes}',r'\begin{tabular}{ll}\toprule',r'Checkpoint & SHA256\\\midrule']
    lines += [record['label']+' & '+r'\texttt{'+record['sha256']+r'}\\' for record in records]
    lines += [r'\bottomrule\end{tabular}\end{table*}']
    (PAPER/'tables_hashes.tex').write_text('\n'.join(lines)+'\n')
    (ROOT/'sources/manuscript_checkpoint_hashes.json').write_text(json.dumps(records,indent=2)+'\n')
    print('TABLES_WRITTEN',len(rows),len(pairs),len(records),flush=True)
if __name__=='__main__':main()
