"""Fixed TEST analysis of previously frozen output layers and reader pairs."""
import csv
import shutil
from pathlib import Path
import numpy as np
from run_learned_confirmation import ROOT,SEEDS,read,write,READERS
from analyze_learned_equivalence import resample,summarize,REPLICATES
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    assert read(ROOT/'learned/complete.json')['all_canonical_hypotheses_unchanged']
    assert read(ROOT/'readers/complete.json')['all_canonical_fixed_gauge_hypotheses_unchanged']
    baseline=read(ROOT/'learned/baseline.json');refs=baseline['references'];n=len(refs)
    weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=REPLICATES)
    point,baseboot=resample(refs,baseline['hypotheses'],weights)
    learned={'TEST_clips':n,'baseline_canonical_bleu4':point,'systems':{},'caption_counterfactual':{},
        'TRAIN_clip_overlap':0,'DEV_clip_overlap':0,'eval_optimization_steps':0}
    for arm in ['text_raw','text_canonical','motion_raw']:
        points=[];boots=[];maximum=0.
        for seed in SEEDS:
            r=read(ROOT/f'learned/seed{seed}_{arm}_source_text.json');assert r['canonical']['hypotheses']==baseline['hypotheses']
            p,b=resample(refs,r['raw']['hypotheses'],weights);points.append(p);boots.append(b);maximum=max(maximum,r['max_rotation_error_deg'])
        learned['systems'][arm]={**summarize(points,boots,point,baseboot),'canonical_scores_per_seed':[point]*3,
            'canonical_hypothesis_changes':0,'max_rotation_error_deg':maximum}
    targets=refs[1:]+refs[:1];wrong_point,wrong_boot=resample(targets,baseline['hypotheses'],weights)
    wrong_points=[];wrong_boots=[];actual_points=[];actual_boots=[]
    for seed in SEEDS:
        r=read(ROOT/f'learned/seed{seed}_text_raw_rolled_adapter_text.json')
        p,b=resample(targets,r['raw']['hypotheses'],weights);wrong_points.append(p);wrong_boots.append(b)
        p,b=resample(refs,r['raw']['hypotheses'],weights);actual_points.append(p);actual_boots.append(b)
    learned['caption_counterfactual']={'different_captions':sum(a!=b for a,b in zip(refs,targets)),
        'baseline_wrong_caption_bleu4':wrong_point,'versus_wrong_caption':summarize(wrong_points,wrong_boots,wrong_point,wrong_boot),
        'versus_actual_caption':summarize(actual_points,actual_boots,point,baseboot)}
    flags=read(ROOT/'learned/data_audit.json')['TEST_exact_caption_seen_in_TRAIN'];subset=[i for i,seen in enumerate(flags) if not seen]
    subrefs=[refs[i] for i in subset];m=len(subset);w=np.random.default_rng(42).multinomial(m,np.ones(m)/m,size=REPLICATES)
    p,b=resample(subrefs,[baseline['hypotheses'][i] for i in subset],w);points=[];boots=[]
    for seed in SEEDS:
        r=read(ROOT/f'learned/seed{seed}_text_raw_source_text.json');x,y=resample(subrefs,[r['raw']['hypotheses'][i] for i in subset],w);points.append(x);boots.append(y)
    learned['unseen_exact_captions']={'clips':m,'excluded_seen_captions':len(flags)-m,'baseline_bleu4':p,**summarize(points,boots,p,b)}
    results={}
    for corpus in ['How2Sign','CSL-Daily']:
        folder=ROOT/'readers'/corpus;first=read(folder/'public_raw.json');refs=first['references'];ids=first['clip_ids'];char=corpus=='CSL-Daily'
        groups=ids
        if corpus=='How2Sign':
            path=Path('/mnt/datasets/sign_datasets/raw/how2sign/How2Sign/sentence_level/test/text/en/raw_text/how2sign_test.csv')
            with path.open() as stream:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(stream,delimiter='\t')}
            groups=[lookup[name] for name in ids];dest=ROOT/'sources';dest.mkdir(exist_ok=True);shutil.copy2(path,dest/'how2sign_test.csv')
        names=sorted(set(groups));gi=np.asarray([names.index(g) for g in groups]);w=np.random.default_rng(42).multinomial(len(names),np.ones(len(names))/len(names),size=REPLICATES)
        records={('public','raw'):first,('public','canonical'):read(folder/'public_canonical.json')}
        for seed in SEEDS:
            for arm in ['raw','canonical']:
                for kind in ['native','canonical'] if arm=='raw' else ['native','fixed_gauge']:
                    records[seed,arm,kind]=read(folder/f'seed{seed}_{arm}_{kind}.json')
        points={};boots={}
        for key,r in records.items():
            assert r['references']==refs and r['clip_ids']==ids
            rr=[' '.join(list(s.replace(' ',''))) for s in refs] if char else refs
            hh=[' '.join(list(s.replace(' ',''))) for s in r['hypotheses']] if char else r['hypotheses']
            stat=statistics(rr,hh);grouped=np.zeros((len(names),10),np.int64);np.add.at(grouped,gi,stat)
            p=b4(stat.sum(0));assert abs(p-r['scores']['bleu']['bleu4'])<1e-10 and abs(p-metric(refs,r['hypotheses'],char))<1e-10
            points[key]=p;boots[key]=np.asarray([b4(row) for row in w@grouped])
        def contrast(right):
            deltas=[];samples=[]
            for seed in SEEDS:
                a=(seed,'canonical','native');b=('public','raw') if right=='public' else (seed,'raw','native')
                deltas.append(points[a]-points[b]);samples.append(boots[a]-boots[b])
            matrix=np.stack(samples,1);draw=np.random.default_rng(2027).integers(0,3,size=(REPLICATES,3))
            return {'mean_delta_bleu4':float(np.mean(deltas)),'per_seed_deltas':deltas,
                'conditional_group_ci95':np.percentile(matrix.mean(1),[2.5,97.5]).tolist(),
                'exploratory_seed_and_group_ci95':np.percentile(np.take_along_axis(matrix,draw,axis=1).mean(1),[2.5,97.5]).tolist()}
        comparisons={'canonical_vs_paired_raw':contrast('paired'),'canonical_vs_public_raw':contrast('public')}
        margin=read(READERS/'protocol.json')['exploratory_noninferiority_margins_BLEU4'][corpus]
        for r in comparisons.values():r['exploratory_margin']=margin;r['lower_interval_exceeds_negative_margin']=r['exploratory_seed_and_group_ci95'][0]>-margin
        results[corpus]={'TEST_clips':len(ids),'bootstrap_unit':'VIDEO_ID' if corpus=='How2Sign' else 'clip','clusters':len(names),
            'public_raw':points['public','raw'],'public_canonical':points['public','canonical'],
            'raw_fine_scores':[points[s,'raw','native'] for s in SEEDS],
            'canonical_fine_scores':[points[s,'canonical','native'] for s in SEEDS],
            'comparisons':comparisons,'fixed_gauge_canonical_hypotheses_changed':0}
        for seed in SEEDS:assert records[seed,'canonical','native']['hypotheses']==records[seed,'canonical','fixed_gauge']['hypotheses']
    output={'learned':learned,'readers':results,'replicates':REPLICATES,'bootstrap_seed':42,
        'models_frozen_before_TEST_inference':True,'new_training_or_selection':False,
        'project_TEST_use_history_disclosed':True,'pretraining_variance_not_estimated':True}
    write('analysis.json',output)
    print('CONFIRMATION_ANALYSIS',str(ROOT/'analysis.json'),jsonsummary(learned,results),flush=True)


def jsonsummary(learned,readers):
    return {'baseline':learned['baseline_canonical_bleu4'],'text_raw':learned['systems']['text_raw'],
        'readers':{name:r['comparisons'] for name,r in readers.items()}}


if __name__=='__main__':main()
