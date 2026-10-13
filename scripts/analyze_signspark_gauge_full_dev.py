"""Paired clustered native BLEU bootstrap for the full DEV evaluation audit."""
import csv,json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
ROOT=Path('reports/cvpr_signspark_gauge_full_dev_20261010')

def main():
    output={};table=[]
    for corpus in ['PHOENIX-2014T','How2Sign']:
        folder=ROOT/corpus;selected=json.loads((folder/'selection.json').read_text());refs=[r['reference'] for r in selected]
        if corpus=='How2Sign':
            p=Path('/mnt/datasets/sign_datasets/raw/how2sign/How2Sign/sentence_level/val/text/en/raw_text/how2sign_val.csv')
            with p.open() as f:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(f,delimiter='\t')}
            groups=[lookup[r['clip_id']] for r in selected];unit='official VIDEO_ID'
        else:groups=[r['clip_id'] for r in selected];unit='clip, not signer'
        names=sorted(set(groups));indices=np.asarray([names.index(g) for g in groups]);rng=np.random.default_rng(42);weights=rng.multinomial(len(names),np.ones(len(names))/len(names),size=2000)
        comparisons={}
        for condition in ['correct_caption','rolled_wrong_caption']:
            r=json.loads((folder/f'{condition}.json').read_text());n=len(selected);assert len(r['raw_after_hypotheses'])==n
            def bootstrap(reference,hypotheses):
                stat=statistics(reference,hypotheses);grp=np.zeros((len(names),10),np.int64);np.add.at(grp,indices,stat)
                return b4(stat.sum(0)),np.asarray([b4(row) for row in weights@grp])
            values={}
            for name,reference,hypotheses,expected in [
                ('target_before',r['targets'],r['raw_original_hypotheses'],r['baseline_target_scores']),
                ('target_after',r['targets'],r['raw_after_hypotheses'],r['target_scores']),
                ('canonical_before',refs,r['canonical_original_hypotheses'],None),
                ('canonical_after',refs,r['canonical_after_hypotheses'],r['canonical_actual_scores'])]:
                score,boot=bootstrap(reference,hypotheses)
                if expected:assert abs(score-expected['bleu']['bleu4'])<1e-10
                values[name]=(score,boot)
            comparisons[condition]={}
            for a,b in [('target_after','target_before'),('canonical_after','canonical_before')]:
                comparisons[condition][f'{a} minus {b}']={'delta_bleu4':values[a][0]-values[b][0],'ci95':np.percentile(values[a][1]-values[b][1],[2.5,97.5]).tolist()}
            table.append({'dataset':corpus,'condition':condition,'clips':n,'raw_target_before':values['target_before'][0],'raw_target_after':values['target_after'][0],'canonical_actual_before':values['canonical_before'][0],'canonical_actual_after':values['canonical_after'][0],'rotation_error_max_deg':r['max_rotation_error_deg'],'raw_changed_sentences':r['raw_changed_sentences'],'canonical_changed_sentences':r['canonical_changed_sentences']})
        output[corpus]={'clips':len(selected),'clusters':len(names),'cluster_unit':unit,'replicates':2000,'seed':42,'multiple_testing_adjustment':False,'inference_is_exploratory_DEV':True,'comparisons':comparisons}
    (ROOT/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    with (ROOT/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,corpus in zip(axes,['PHOENIX-2014T','How2Sign']):
        rows=[r for r in table if r['dataset']==corpus];r=rows[0]
        ax.bar(['Raw baseline','Raw targeted','Canonical baseline','Canonical targeted'],[r['raw_target_before'],r['raw_target_after'],r['canonical_actual_before'],r['canonical_actual_after']],color=['#687d8e','#cf7155','#3c927a','#3c927a']);ax.set_ylim(0,100);ax.set_ylabel('BLEU-4');ax.set_title(f'{corpus}, full released DEV (n={r["clips"]})');ax.tick_params(axis='x',rotation=28);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Targeted caption audit: rotations preserved, reader weights fixed')
    fig.savefig(ROOT/'targeted_score_inflation.png',dpi=180);fig.savefig(ROOT/'targeted_score_inflation.pdf');plt.close(fig)
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()
