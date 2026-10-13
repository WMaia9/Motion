"""Paired DEV comparisons for the actual DDPM pilot."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_signspark_geometry_dev import statistics,b4
from analyze_motion_confirmation import bootstrap_wer
from run_signspark_geometry_dev import matrices
ROOT=Path('reports/cvpr_physical_semantic_ddpm_20261010')

def main():
    names=['geometry','semantic_raw','semantic_physical','semantic_physical_shuffled'];rs={n:json.loads((ROOT/(n+'_scores.json')).read_text()) for n in names};refs=rs['geometry']['references'];rng=np.random.default_rng(42);weights=rng.multinomial(128,np.ones(128)/128,size=2000);boot={};table=[]
    for n,r in rs.items():
        for typ in ['raw','canonical']:
            s=statistics(refs,r[typ+'_hypotheses']);score=b4(s.sum(0));assert abs(score-r[typ+'_scores']['bleu']['bleu4'])<1e-10;boot[n,typ]=(score,np.asarray([b4(z) for z in weights@s]))
        table.append({'condition':n,'raw_bleu4':r['raw_scores']['bleu']['bleu4'],'canonical_bleu4':r['canonical_scores']['bleu']['bleu4'],'raw_gloss_wer':r['independent_raw_gloss']['wer'],'canonical_gloss_wer':r['independent_canonical_gloss']['wer'],'raw_canonical_changed_sentences':r['raw_canonical_changed_sentences']})
    out={'replicates':2000,'seed':42,'unit':'DEV clip, not signer','exploratory':True,'multiple_testing_adjustment':False,'comparisons':{}}
    for a,b in [('semantic_raw','geometry'),('semantic_physical','geometry'),('semantic_physical','semantic_raw'),('semantic_physical_shuffled','semantic_physical')]:
        q={}
        for typ in ['raw','canonical']:
            va,ba=boot[a,typ];vb,bb=boot[b,typ];q[typ+'_bleu4']={'delta':va-vb,'ci95':np.percentile(ba-bb,[2.5,97.5]).tolist()};ra=rs[a]['independent_'+typ+'_gloss'];rb=rs[b]['independent_'+typ+'_gloss'];q[typ+'_independent_wer']=bootstrap_wer(rb['errors'],ra['errors'],ra['reference_lengths'])
        out['comparisons'][a+' minus '+b]=q
    physical={}
    for name in names:
        values=[];speed=[]
        with np.load(ROOT/(name+'_poses.npz')) as d:
            for i in range(128):
                x=d[f'clip{i:04d}'];r=matrices(x);det=np.linalg.det(r);assert np.isfinite(x).all() and abs(det-1).max()<1e-6
                change=np.degrees(Rotation.from_matrix((r[:-1].transpose(0,1,3,2)@r[1:]).reshape(-1,3,3)).magnitude());speed.append(float(change.mean()) if len(x)>1 else 0.)
                raw=x.reshape(-1,40,6);values.append(float(abs(np.linalg.norm(raw[...,:3],axis=-1)-1).mean()))
        physical[name]={'mean_frame_rotation_change_deg':float(np.mean(speed)),'mean_raw_first_vector_norm_error':float(np.mean(values))}
    out['geometry_diagnostics']=physical;(ROOT/'analysis.json').write_text(json.dumps(out,indent=2))
    with (ROOT/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');x=np.arange(3);labels=['Geometry','Semantic raw','Semantic physical']
    axes[0].bar(x-.2,[z['raw_bleu4'] for z in table[:3]],.4,label='raw');axes[0].bar(x+.2,[z['canonical_bleu4'] for z in table[:3]],.4,label='canonical');axes[0].set_ylabel('BLEU-4 (training reader)');axes[0].legend()
    axes[1].bar(x,[z['canonical_gloss_wer'] for z in table[:3]],color=['#687d8e','#c7834e','#358f77']);axes[1].set_ylabel('Gloss WER % (independent reader)')
    for ax in axes:ax.set_xticks(x,labels,rotation=15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Actual DDPM pilot — PHOENIX DEV n=128, text + real duration');fig.savefig(ROOT/'ddpm_comparison.png',dpi=180);fig.savefig(ROOT/'ddpm_comparison.pdf');plt.close(fig);print('DDPM_ANALYSIS_COMPLETE',json.dumps(out),flush=True)

if __name__=='__main__':main()
