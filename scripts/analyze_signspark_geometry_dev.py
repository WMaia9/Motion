"""Native-metric checked, paired descriptive cluster bootstrap and figures."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('external/SignSparK_BT').resolve()))
from signspark_bt.external_metrics.sacrebleu import raw_corpus_bleu,compute_bleu
ROOT=Path('reports/cvpr_signspark_baseline_20261010')
PAIRS=[('sparse_stride10_slerp','sparse_stride10_projected6d'),('sparse_stride10_projected6d','sparse_stride10_linear6d'),('adaptive_all_slerp','sparse_stride10_slerp'),('adaptive_all_slerp','sparse_stride10_projected6d'),('adaptive_body_slerp','adaptive_hands_slerp'),('so3_roundtrip','original'),('reverse_time','original'),('identity_hands','original')]

def statistics(refs,hyps):
    values=[]
    for ref,hyp in zip(refs,hyps):
        b=raw_corpus_bleu([hyp],[[ref]])
        values.append([*b.counts,*b.totals,b.sys_len,b.ref_len])
    return np.asarray(values,dtype=np.int64)

def b4(row):
    return compute_bleu(row[:4].tolist(),row[4:8].tolist(),int(row[8]),int(row[9]),smooth_method='floor',use_effective_order=True).scores[3]

def main():
    result={};allrows=[]
    for corpus in ['PHOENIX-2014T','How2Sign']:
        folder=ROOT/corpus;summary=json.loads((folder/'summary.json').read_text())
        assert len(summary)==11
        references=json.loads((folder/'references.json').read_text());refs=[r['reference'] for r in references]
        if corpus=='How2Sign':
            p=Path('/mnt/datasets/sign_datasets/raw/how2sign/How2Sign/sentence_level/val/text/en/raw_text/how2sign_val.csv')
            with p.open() as f:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(f,delimiter='\t')}
            groups=[lookup[r['clip_id']] for r in references];cluster='official VIDEO_ID'
        else:groups=[r['clip_id'] for r in references];cluster='clip; signer not authenticated'
        names=sorted(set(groups));mapping={s:i for i,s in enumerate(names)};groupindices=np.asarray([mapping[s] for s in groups]);rng=np.random.default_rng(42)
        weights=rng.multinomial(len(names),np.ones(len(names))/len(names),size=2000)
        boot={};matches={};details={}
        for condition in summary:
            r=json.loads((folder/f'{condition}.json').read_text());hyps=r['hypotheses'];assert len(hyps)==len(refs)
            stat=statistics(refs,hyps);assert abs(b4(stat.sum(0))-r['scores']['bleu']['bleu4'])<1e-10
            grouped=np.zeros((len(names),10),dtype=np.int64);np.add.at(grouped,groupindices,stat)
            aggregate=weights@grouped;boot[condition]=np.asarray([b4(row) for row in aggregate])
            matches[condition]=sum(a==b for a,b in zip(hyps,json.loads((folder/'original.json').read_text())['hypotheses']))
            details[condition]=r
            if condition.startswith('adaptive_'):
                assert all(len(a)==len(np.unique(np.r_[np.arange(0,v['frames'],10),v['frames']-1])) for a,v in zip(r['anchors'],references))
            allrows.append({'dataset':corpus,'condition':condition,'clips':len(refs),'bleu4':r['scores']['bleu']['bleu4'],'chrf':r['scores']['chrf'],'rouge':r['scores']['rouge'],**r['geometry_clip_mean']})
        comparisons={}
        for a,b in PAIRS:
            delta=boot[a]-boot[b]
            comparisons[f'{a} minus {b}']={'delta_bleu4':summary[a]['scores']['bleu']['bleu4']-summary[b]['scores']['bleu']['bleu4'],'ci95':np.percentile(delta,[2.5,97.5]).tolist()}
        result[corpus]={'clips':len(refs),'cluster_unit':cluster,'clusters':len(names),'bootstrap_replicates':2000,'seed':42,'scores_verified_against_native':True,'paired_comparisons':comparisons,'same_hypotheses_as_original':matches,'adaptive_anchor_counts_exactly_matched':True,'anchor_fraction_mean':float(np.mean([len(np.unique(np.r_[np.arange(0,r['frames'],10),r['frames']-1]))/r['frames'] for r in references]))}
        (folder/'bootstrap_groups.json').write_text(json.dumps([{'clip_id':r['clip_id'],'group':g} for r,g in zip(references,groups)],indent=2))
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    chosen=['original','sparse_stride10_linear6d','sparse_stride10_projected6d','sparse_stride10_slerp','adaptive_all_slerp','adaptive_body_slerp','adaptive_hands_slerp']
    labels=['Original','Linear 6D','Projected 6D','Uniform SLERP','Adaptive all','Adaptive body','Adaptive hands']
    for ax,corpus in zip(axes,['PHOENIX-2014T','How2Sign']):
        rows={r['condition']:r for r in allrows if r['dataset']==corpus};ax.barh(labels,[rows[c]['bleu4'] for c in chosen],color=['#34495e']+['#779eb6']*3+['#3b927a']*3);ax.invert_yaxis();ax.set_xlabel('Native back-translation BLEU-4');ax.set_title(f'{corpus} DEV, n={result[corpus]["clips"]}');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Oracle motion reconstruction: matched sparse-anchor budgets, no text-only generation')
    fig.savefig(ROOT/'geometry_semantics.png',dpi=180);fig.savefig(ROOT/'geometry_semantics.pdf');plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
