"""Paired DEV uncertainty; standard unit-cost WER and native BLEU-4."""
import csv, json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
ROOT=Path('reports/cvpr_motion_consistency_confirmation_20261010')

def bootstrap_wer(a,b,den,seed=42):
    rng=np.random.default_rng(seed);weights=rng.multinomial(len(den),np.ones(len(den))/len(den),size=2000)
    delta=100*((weights@np.asarray(b))-(weights@np.asarray(a)))/(weights@np.asarray(den))
    return {'delta_wer_points':100*(sum(b)-sum(a))/sum(den),'ci95':np.percentile(delta,[2.5,97.5]).tolist(),'replicates':2000,'seed':seed,'unit':'DEV clip, not signer','exploratory':True}

def main():
    reader=ROOT/'ctc_reader';baseline=json.loads((reader/'original.json').read_text());output={'ctc':{},'generator':{}};table=[]
    for name in ['canonical','reverse_time','identity_hands','scale09','scale11','shear01','transfer_transformer_correct_caption','transfer_transformer_rolled_wrong_caption']:
        r=json.loads((reader/(name+'.json')).read_text());output['ctc'][name]=bootstrap_wer(baseline['errors'],r['errors'],baseline['reference_lengths'])
    for name in ['correct_gloss','rolled_wrong_gloss']:
        r=json.loads((reader/(name+'.json')).read_text());output['ctc'][name]={}
        for a,b in [('baseline_target','target'),('baseline_target','actual_reference')]:
            if b=='actual_reference':before=baseline
            else:before=r[a]
            output['ctc'][name][b]=bootstrap_wer(before['errors'],r[b]['errors'],r[b]['reference_lengths'])
    for corpus in ['PHOENIX-2014T','How2Sign']:
        output['generator'][corpus]={}
        for mode in ['text_duration','oracle_keyframes']:
            r=json.loads((ROOT/'generated'/corpus/(mode+'_scores.json')).read_text());refs=r['references'];names=r['clip_ids']
            if corpus=='How2Sign':
                p=Path('/mnt/datasets/sign_datasets/raw/how2sign/How2Sign/sentence_level/val/text/en/raw_text/how2sign_val.csv')
                with p.open() as f:lookup={z['SENTENCE_NAME']:z['VIDEO_ID'] for z in csv.DictReader(f,delimiter='\t')}
                group=[lookup[z] for z in names];unit='official VIDEO_ID'
            else:group=names;unit='clip, not signer'
            groups=sorted(set(group));indices=np.asarray([groups.index(g) for g in group]);rng=np.random.default_rng(42);weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000)
            scores={};boots={}
            for key in ['raw','canonical']:
                s=statistics(refs,r[key+'_hypotheses']);g=np.zeros((len(groups),10),np.int64);np.add.at(g,indices,s);scores[key]=b4(s.sum(0));boots[key]=np.asarray([b4(z) for z in weights@g]);assert abs(scores[key]-r[key+'_scores']['bleu']['bleu4'])<1e-10
            delta=boots['canonical']-boots['raw'];result={'delta_bleu4':scores['canonical']-scores['raw'],'ci95':np.percentile(delta,[2.5,97.5]).tolist(),'clusters':len(groups),'unit':unit,'replicates':2000,'seed':42,'exploratory':True};output['generator'][corpus][mode]=result
            table.append({'dataset':corpus,'mode':mode,'clips':r['clips'],'raw_bleu4':scores['raw'],'canonical_bleu4':scores['canonical'],'delta_bleu4':result['delta_bleu4'],'ci95_low':result['ci95'][0],'ci95_high':result['ci95'][1],'changed_sentences':r['changed_sentences'],'max_angle_deg':r['max_angle_deg']})
    (ROOT/'analysis.json').write_text(json.dumps(output,indent=2))
    with (ROOT/'generator_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');vals=[baseline['wer'],json.loads((reader/'correct_gloss.json').read_text())['actual_reference']['wer'],json.loads((reader/'correct_gloss.json').read_text())['canonical_actual']['wer']]
    ax.bar(['Original motion','Equivalent 6D, targeted','Canonical equivalent'],vals,color=['#63748d','#c47742','#329170']);ax.set_ylim(0,100);ax.set_ylabel('Gloss WER (%) — lower is better');ax.set_title('Independent Conv/BiGRU/CTC, all 519 PHOENIX DEV clips')
    for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f}%',ha='center')
    ax.grid(axis='y',alpha=.2);fig.savefig(ROOT/'independent_reader.png',dpi=180);fig.savefig(ROOT/'independent_reader.pdf');plt.close(fig)
    print('ANALYSIS_COMPLETE',json.dumps(output),flush=True)

if __name__=='__main__':main()
