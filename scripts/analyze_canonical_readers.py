"""Paired clip/video and seed uncertainty for canonical-reader adaptation."""
import json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric
from run_canonical_readers import ROOT,SEEDS,read,write

REPLICATES=20000


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    assert read(ROOT/'training_complete.json')['paired_minibatches_matched']
    results={}
    for corpus in ['How2Sign','CSL-Daily']:
        audit=read(ROOT/corpus/'data_audit.json');char=corpus=='CSL-Daily'
        first=read(ROOT/corpus/'public_raw.json');refs=first['references'];ids=first['clip_ids'];n=len(refs)
        groups=audit.get('DEV_video_groups',ids);names=sorted(set(groups));mapping={g:i for i,g in enumerate(names)}
        gi=np.asarray([mapping[g] for g in groups]);weights=np.random.default_rng(42).multinomial(len(names),np.ones(len(names))/len(names),size=REPLICATES)
        records={('public','raw'):first,('public','canonical'):read(ROOT/corpus/'public_canonical.json')}
        for seed in SEEDS:
            for arm in ['raw','canonical']:
                for condition in ['native','canonical']:
                    records[seed,arm,condition]=read(ROOT/corpus/f'seed{seed}'/arm/(condition+'_scores.json'))
        points={};boots={}
        for key,r in records.items():
            assert r['references']==refs and r['clip_ids']==ids
            rr=[' '.join(list(s.replace(' ',''))) for s in refs] if char else refs
            hh=[' '.join(list(s.replace(' ',''))) for s in r['hypotheses']] if char else r['hypotheses']
            stat=statistics(rr,hh);grouped=np.zeros((len(names),10),np.int64);np.add.at(grouped,gi,stat)
            point=b4(stat.sum(0));assert abs(point-r['scores']['bleu']['bleu4'])<1e-10 and abs(point-metric(refs,r['hypotheses'],char))<1e-10
            points[key]=point;boots[key]=np.asarray([b4(row) for row in weights@grouped])
        def compare(left,right):
            deltas=[];samples=[]
            for seed in SEEDS:
                a=(seed,*left);b=('public','raw') if right==('public','raw') else (seed,*right)
                deltas.append(points[a]-points[b]);samples.append(boots[a]-boots[b])
            matrix=np.stack(samples,1);draw=np.random.default_rng(2027).integers(0,3,size=(REPLICATES,3))
            return {'mean_delta_bleu4':float(np.mean(deltas)),'per_seed_deltas':deltas,
                'conditional_group_ci95':np.percentile(matrix.mean(1),[2.5,97.5]).tolist(),
                'exploratory_seed_and_group_ci95':np.percentile(np.take_along_axis(matrix,draw,axis=1).mean(1),[2.5,97.5]).tolist()}
        comparisons={'canonical_fine_vs_raw_fine':compare(('canonical','native'),('raw','native')),
                     'canonical_fine_vs_public_raw':compare(('canonical','native'),('public','raw')),
                     'posthoc_canonical_raw_fine_vs_raw_fine':compare(('raw','canonical'),('raw','native'))}
        margin=read(ROOT/'protocol.json')['exploratory_noninferiority_margins_BLEU4'][corpus]
        for r in comparisons.values():
            r['exploratory_margin_bleu4']=margin;r['lower_interval_exceeds_negative_margin']=r['exploratory_seed_and_group_ci95'][0]>-margin
        results[corpus]={'TRAIN_clips':audit['TRAIN_clips'],'DEV_clips':n,'bootstrap_unit':'VIDEO_ID' if 'DEV_video_groups' in audit else 'clip',
            'clusters':len(names),'seed_count':3,'public_raw':points['public','raw'],'public_canonical':points['public','canonical'],
            'raw_fine_scores':[points[s,'raw','native'] for s in SEEDS],
            'raw_fine_canonical_scores':[points[s,'raw','canonical'] for s in SEEDS],
            'canonical_fine_scores':[points[s,'canonical','native'] for s in SEEDS],
            'comparisons':comparisons,'noninferiority_scope':'exploratory DEV margins; no clinical/linguistic threshold claim'}
    write('analysis.json',{'results':results,'replicates':REPLICATES,'bootstrap_seed':42,'training_seeds':SEEDS,
        'heldout_split':'DEV; already inspected in this project','no_TEST_use':True,'no_full_pretraining_seed_variance':True})
    print('CANONICAL_READER_ANALYSIS',json.dumps(results),flush=True)


if __name__=='__main__':main()
