"""Matched prompt/CFG ablations and uncertainty; no TEST."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run_signspark_geometry_dev import matrices
from analyze_signspark_geometry_dev import statistics,b4
from analyze_motion_confirmation import bootstrap_wer

ROOT=Path('reports/cvpr_text_to_sign_diagnostics_20261010')
PREV=Path('reports/cvpr_motion_consistency_confirmation_20261010')
NAMES=['cfg0','cfg1','cfg3','cfg5','cfg3_shuffled','cfg3_horizontal']

def main():
    output={};table=[]
    for corpus in ['PHOENIX-2014T','How2Sign']:
        rs={n:json.loads((ROOT/corpus/(n+'_scores.json')).read_text()) for n in NAMES};names=rs['cfg3']['clip_ids'];refs=rs['cfg3']['references'];assert all(r['clip_ids']==names and r['references']==refs for r in rs.values())
        if corpus=='How2Sign':
            p=Path('/mnt/datasets/sign_datasets/raw/how2sign/How2Sign/sentence_level/val/text/en/raw_text/how2sign_val.csv')
            with p.open() as f:lookup={z['SENTENCE_NAME']:z['VIDEO_ID'] for z in csv.DictReader(f,delimiter='\t')}
            groups=[lookup[z] for z in names];unit='official VIDEO_ID'
        else:groups=names;unit='clip, not signer'
        gnames=sorted(set(groups));gi=np.asarray([gnames.index(g) for g in groups]);weights=np.random.default_rng(42).multinomial(len(gnames),np.ones(len(gnames))/len(gnames),size=2000);boots={}
        for n,r in rs.items():
            for typ,key in [('raw','hypotheses'),('canonical','canonical_hypotheses')]:
                st=statistics(refs,r[key]);gp=np.zeros((len(gnames),10),np.int64);np.add.at(gp,gi,st);value=b4(st.sum(0));assert abs(value-r[typ+'_scores']['bleu']['bleu4'])<1e-10;boots[n,typ]=(value,np.asarray([b4(z) for z in weights@gp]))
            table.append({'dataset':corpus,'condition':n,'raw_bleu4':r['raw_scores']['bleu']['bleu4'],'canonical_bleu4':r['canonical_scores']['bleu']['bleu4'],'raw_chrf':r['raw_scores']['chrf'],'canonical_chrf':r['canonical_scores']['chrf'],'unigram_matches':r['ngram_matches_1_to_4'][0],'quadgram_matches':r['ngram_matches_1_to_4'][3],'raw_canonical_changed_sentences':r['raw_canonical_changed_sentences'],'mean_reference_angle_deg':r['geometry_clip_mean']['mean_reference_rotation_error_deg'],'mean_frame_rotation_change_deg':r['geometry_clip_mean']['mean_frame_rotation_change_deg']})
        comparisons={}
        for a,b in [('cfg3','cfg1'),('cfg5','cfg1'),('cfg3_horizontal','cfg3'),('cfg3_shuffled','cfg3'),('cfg3','cfg0')]:
            comparisons[a+' minus '+b]={}
            for typ in ['raw','canonical']:
                va,ba=boots[a,typ];vb,bb=boots[b,typ];comparisons[a+' minus '+b][typ]={'delta_bleu4':va-vb,'ci95':np.percentile(ba-bb,[2.5,97.5]).tolist()}
        posechecks={}
        for a,b in [('cfg3','cfg0'),('cfg3_shuffled','cfg3'),('cfg3_horizontal','cfg3')]:
            angle=[]
            with np.load(ROOT/corpus/(a+'_poses.npz')) as za,np.load(ROOT/corpus/(b+'_poses.npz')) as zb:
                for i in range(128):
                    x=za[f'clip{i:04d}'];y=zb[f'clip{i:04d}'];assert x.shape==y.shape
                    delta=np.degrees(Rotation.from_matrix((matrices(x).transpose(0,1,3,2)@matrices(y)).reshape(-1,3,3)).magnitude());angle.append(float(delta.mean()))
            posechecks[a+' versus '+b]={'clip_mean_angle_deg':float(np.mean(angle)),'per_clip_mean_angle_deg':angle,'raw_changed_hypotheses':sum(x!=y for x,y in zip(rs[a]['hypotheses'],rs[b]['hypotheses'])),'canonical_changed_hypotheses':sum(x!=y for x,y in zip(rs[a]['canonical_hypotheses'],rs[b]['canonical_hypotheses']))}
        energies={}
        for stream in ['body','hand']:
            energies[stream]=json.loads((ROOT/'logs'/(corpus.replace('-','')+'_cfg3_horizontal_'+stream+'_stats.json')).read_text());assert energies[stream]['late_calls']>0
        output[corpus]={'clusters':len(gnames),'cluster_unit':unit,'replicates':2000,'seed':42,'exploratory_DEV':True,'multiple_testing_adjustment':False,'bleu_comparisons':comparisons,'physical_prompt_response':posechecks,'guidance_energy':energies}
        if corpus=='PHOENIX-2014T':
            ind=json.loads((ROOT/corpus/'independent_ctc.json').read_text());output[corpus]['independent_ctc']={}
            for a,b in [('cfg3','cfg1'),('cfg5','cfg1'),('cfg3_horizontal','cfg3'),('cfg3_shuffled','cfg3'),('cfg3','cfg0')]:
                output[corpus]['independent_ctc'][a+' minus '+b]={}
                for typ in ['raw','canonical']:
                    output[corpus]['independent_ctc'][a+' minus '+b][typ]=bootstrap_wer(ind[b][typ]['errors'],ind[a][typ]['errors'],ind[a][typ]['reference_lengths'])
            for r in table:
                if r['dataset']==corpus:r['independent_raw_wer']=ind[r['condition']]['raw']['wer'];r['independent_canonical_wer']=ind[r['condition']]['canonical']['wer']
    (ROOT/'analysis.json').write_text(json.dumps(output,indent=2))
    keys=list(table[0]);
    with (ROOT/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,c in zip(axes,['PHOENIX-2014T','How2Sign']):
        rows=[z for z in table if z['dataset']==c];x=np.arange(len(rows));ax.bar(x-.2,[z['raw_bleu4'] for z in rows],.4,label='raw 6D');ax.bar(x+.2,[z['canonical_bleu4'] for z in rows],.4,label='canonical rotations');ax.set_xticks(x,[z['condition'].replace('cfg3_','3:') for z in rows],rotation=28);ax.set_ylabel('BLEU-4');ax.set_title(c+' — text + reference duration, no keyframes');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'text_sign_cfg_comparison.png',dpi=180);fig.savefig(ROOT/'text_sign_cfg_comparison.pdf');plt.close(fig)
    print('ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
