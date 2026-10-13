"""Post-result sensitivity: invariance repair is not semantic accuracy repair."""
import csv,json
import numpy as np
from run_evaluation_heldout import ROOT,CORPORA
from analyze_signspark_geometry_dev import statistics,b4


def compare(refs,before,after,indices=None):
    a=statistics(refs,before);b=statistics(refs,after)
    if indices is not None:
        ga=np.zeros((max(indices)+1,10),np.int64);gb=ga.copy()
        np.add.at(ga,indices,a);np.add.at(gb,indices,b);a,b=ga,gb
    n=len(a);weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=2000)
    boot=np.array([b4(x)-b4(y) for x,y in zip(weights@b,weights@a)])
    return {'delta_bleu4':b4(b.sum(0))-b4(a.sum(0)),'ci95':np.percentile(boot,[2.5,97.5]).tolist(),'bootstrap_units':n}


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    result={'post_result_defense_accuracy_sensitivity':True,'replicates':2000,'seed':42,'results':{}}
    with (ROOT/'sources/how2sign_test.csv').open() as f:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(f,delimiter='\t')}
    for corpus in CORPORA:
        folder=ROOT/'heldout'/corpus;records={}
        for subset in ['full_real_scores','correct_caption_scores']:
            r=json.loads((folder/(subset+'.json')).read_text())
            before=r['raw']['hypotheses'] if subset=='full_real_scores' else r['baseline_actual']['hypotheses']
            after=r['canonical']['hypotheses'] if subset=='full_real_scores' else r['canonical_before_actual']['hypotheses']
            record={'clips':len(before),'clip_bootstrap':compare(r['references'],before,after)}
            if corpus=='How2Sign':
                groups=[lookup[key] for key in r['clip_ids']];unique=sorted(set(groups));indices=np.array([unique.index(g) for g in groups])
                record['official_VIDEO_ID_cluster_bootstrap']=compare(r['references'],before,after,indices)
            records[subset]=record
        result['results'][corpus]=records
    (ROOT/'canonical_accuracy_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
