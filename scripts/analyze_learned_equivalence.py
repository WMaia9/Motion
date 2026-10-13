"""Paired uncertainty for learned, rotation-preserving generator outputs."""
import json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from audit_reviewed_manuscript_evidence import metric
from run_learned_equivalence import ROOT, SEEDS, read, write

REPLICATES=20000


def resample(refs,hypotheses,weights):
    stat=statistics(refs,hypotheses)
    point=b4(stat.sum(0));assert abs(point-metric(refs,hypotheses))<1e-10
    return point,np.asarray([b4(row) for row in weights@stat])


def summarize(points,boots,baseline,baseboot):
    matrix=np.stack(boots,axis=1);draw=np.random.default_rng(2027).integers(0,len(SEEDS),size=(REPLICATES,len(SEEDS)))
    differences=matrix-baseboot[:,None]
    return {'scores_per_seed':points,'mean_bleu4':float(np.mean(points)),
        'mean_gain_bleu4':float(np.mean(points)-baseline),
        'gain_ci95_conditional_on_seeds':np.percentile(differences.mean(1),[2.5,97.5]).tolist(),
        'exploratory_seed_and_clip_gain_ci95':np.percentile(np.take_along_axis(differences,draw,axis=1).mean(1),[2.5,97.5]).tolist(),
        'uncertainty_scope':'three output-layer training seeds; one frozen generator/reader; no pretraining variability'}


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    assert read(ROOT/'evaluation/complete.json')['canonical_hypotheses_unchanged']
    audit=read(ROOT/'data_audit.json');baseline=read(ROOT/'evaluation/baseline.json')
    refs=baseline['references'];n=len(refs);weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=REPLICATES)
    point,baseboot=resample(refs,baseline['hypotheses'],weights)
    result={'DEV_clips':n,'TRAIN_clips':len(audit['train']),'baseline_canonical_bleu4':point,
        'replicates':REPLICATES,'bootstrap_seed':42,'training_seeds':SEEDS,
        'no_TEST_use':True,'no_eval_optimization':True,'physical_generator_frozen':True,
        'systems':{},'caption_counterfactual':{},'unseen_exact_captions':{},'public_transfer':{}}
    for arm in ['text_raw','text_canonical','motion_raw']:
        points=[];boots=[];max_angle=0.
        for seed in SEEDS:
            r=read(ROOT/f'evaluation/seed{seed}_{arm}_source_text.json')
            assert r['canonical']['hypotheses']==baseline['hypotheses']
            p,b=resample(refs,r['raw']['hypotheses'],weights);points.append(p);boots.append(b)
            max_angle=max(max_angle,r['max_rotation_error_deg'])
        result['systems'][arm]={**summarize(points,boots,point,baseboot),
            'canonical_scores_per_seed':[point]*3,'canonical_changes_per_seed':[0]*3,'max_rotation_error_deg':max_angle}
    targets=refs[1:]+refs[:1];wrong_base,wrong_boot=resample(targets,baseline['hypotheses'],weights)
    wrong_points=[];wrong_boots=[];actual_points=[];actual_boots=[]
    for seed in SEEDS:
        r=read(ROOT/f'evaluation/seed{seed}_text_raw_rolled_adapter_text.json')
        p,b=resample(targets,r['raw']['hypotheses'],weights);wrong_points.append(p);wrong_boots.append(b)
        p,b=resample(refs,r['raw']['hypotheses'],weights);actual_points.append(p);actual_boots.append(b)
    result['caption_counterfactual']={'different_captions':sum(a!=b for a,b in zip(refs,targets)),
        'baseline_wrong_caption_bleu4':wrong_base,
        'versus_wrong_caption':summarize(wrong_points,wrong_boots,wrong_base,wrong_boot),
        'versus_actual_caption':summarize(actual_points,actual_boots,point,baseboot),
        'interpretation':'only adapter source text changed; base rotations unchanged; not a new training arm'}
    mask=np.logical_not(audit['DEV_exact_caption_seen_in_TRAIN']);subset=[i for i,v in enumerate(mask) if v]
    subrefs=[refs[i] for i in subset];subbaseline=[baseline['hypotheses'][i] for i in subset];m=len(subset)
    w=np.random.default_rng(42).multinomial(m,np.ones(m)/m,size=REPLICATES)
    p,b=resample(subrefs,subbaseline,w);points=[];boots=[]
    for seed in SEEDS:
        r=read(ROOT/f'evaluation/seed{seed}_text_raw_source_text.json')
        x,y=resample(subrefs,[r['raw']['hypotheses'][i] for i in subset],w);points.append(x);boots.append(y)
    result['unseen_exact_captions']={'DEV_clips':m,'excluded_exact_TRAIN_caption_matches':n-m,
        'baseline_bleu4':p,**summarize(points,boots,p,b)}
    public=read(ROOT/'public_transfer/baseline.json');m=len(public['references'])
    w=np.random.default_rng(42).multinomial(m,np.ones(m)/m,size=REPLICATES)
    p,b=resample(public['references'],public['hypotheses'],w);points=[];boots=[]
    for seed in SEEDS:
        r=read(ROOT/f'public_transfer/seed{seed}.json');assert r['canonical']['hypotheses']==public['hypotheses']
        x,y=resample(public['references'],r['raw']['hypotheses'],w);points.append(x);boots.append(y)
    result['public_transfer']={'generator':'public SignSparK, native text-duration replay','DEV_clips':m,
        'no_new_training':True,'baseline_canonical_bleu4':p,'canonical_changes_per_seed':[0]*3,
        **summarize(points,boots,p,b),'caution':'the local replay is weak; no claim to reproduce published SOTA generator scores'}
    write('analysis.json',result)
    print('LEARNED_LAYER_ANALYSIS',json.dumps({'systems':result['systems'],'public_transfer':result['public_transfer'],'unseen_captions':result['unseen_exact_captions']}),flush=True)


if __name__=='__main__':main()
