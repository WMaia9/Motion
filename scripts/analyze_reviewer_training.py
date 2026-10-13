"""Matched full-DEV fine-tuning results; no inference or model selection."""
import json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics, b4

ROOT = Path('reports/cvpr_reviewer_followup_20261011')
SEEDS = [42,123,777]
ARMS = ['geometry','raw_ce','canonical_ce']


def read(p): return json.loads(p.read_text())


def interval(point, samples):
    return {'delta_bleu4': float(point), 'ci95': np.percentile(samples,[2.5,97.5]).tolist()}


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    completed = read(ROOT/'training/completed.json')
    assert completed['same_first_epoch_stochastic_inputs_in_each_seed']
    n = completed['dev_clips']; reps = 20000
    weights = np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=reps)
    scores = {}; boot = {}; comparisons = {}
    refs = ids = None
    for seed in SEEDS:
        scores[str(seed)] = {}
        for arm in ARMS:
            scores[str(seed)][arm] = {}
            for kind in ['raw','canonical']:
                r = read(ROOT/f'training/seed{seed}/{arm}_{kind}_scores.json')
                if refs is None: refs,ids = r['references'],r['clip_ids']
                assert r['references'] == refs and r['clip_ids'] == ids and len(refs) == n
                stat = statistics(refs,r['hypotheses']); point = b4(stat.sum(0))
                assert abs(point-r['scores']['bleu']['bleu4']) < 1e-10
                scores[str(seed)][arm][kind] = point
                boot[seed,arm,kind] = np.asarray([b4(row) for row in weights@stat])
        comparisons[str(seed)] = {}
        for reference in ['geometry','canonical_ce']:
            left = scores[str(seed)]['raw_ce']; right = scores[str(seed)][reference]
            raw_delta = boot[seed,'raw_ce','raw']-boot[seed,reference,'raw']
            can_delta = boot[seed,'raw_ce','canonical']-boot[seed,reference,'canonical']
            comparisons[str(seed)]['raw_ce minus '+reference] = {
                'raw': interval(left['raw']-right['raw'],raw_delta),
                'canonical': interval(left['canonical']-right['canonical'],can_delta),
                'excess_raw_gain': interval((left['raw']-right['raw'])-(left['canonical']-right['canonical']),raw_delta-can_delta)}
    averages = {arm:{kind:float(np.mean([scores[str(seed)][arm][kind] for seed in SEEDS]))
                     for kind in ['raw','canonical']} for arm in ARMS}
    aggregate = {}
    # Shared clip resampling preserves pairings and dependency of runs on the same corpus.
    seed_draw = np.random.default_rng(2027).integers(0,len(SEEDS),size=(reps,len(SEEDS)))
    for reference in ['geometry','canonical_ce']:
        aggregate[reference] = {}
        for effect in ['raw','canonical','excess_raw_gain']:
            per_seed = np.stack([boot[s,'raw_ce','raw']-boot[s,reference,'raw'] if effect == 'raw'
                else boot[s,'raw_ce','canonical']-boot[s,reference,'canonical'] if effect == 'canonical'
                else (boot[s,'raw_ce','raw']-boot[s,reference,'raw'])-(boot[s,'raw_ce','canonical']-boot[s,reference,'canonical'])
                for s in SEEDS],axis=1)
            points = [comparisons[str(s)]['raw_ce minus '+reference][effect]['delta_bleu4'] for s in SEEDS]
            joint = np.take_along_axis(per_seed,seed_draw,axis=1).mean(1)
            aggregate[reference][effect] = {'mean_delta_bleu4': float(np.mean(points)), 'per_seed_deltas': points,
                'paired_clip_interval_conditional_on_three_seeds': np.percentile(per_seed.mean(1),[2.5,97.5]).tolist(),
                'exploratory_seed_and_clip_bootstrap_ci95': np.percentile(joint,[2.5,97.5]).tolist(),
                'warning': 'three fine-tuning seeds from one shared initialization; seed bootstrap is coarse'}
    result = {'DEV_clips':n,'TRAIN_clips':completed['TRAIN_clips'],'seeds':SEEDS,
        'shared_initialization': True,'five_epochs_last_checkpoint':True,'semantic_weight':.02,
        'replicates':reps,'new_TEST_use':False,'inference_uses_reference_pose':False,
        'scores':scores,'mean_scores':averages,'per_seed_comparisons':comparisons,'mean_comparisons':aggregate,
        'scope': 'exploratory DEV intervention; not proof of changes to published benchmark scores'}
    (ROOT/'training_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TRAINING_ANALYSIS',json.dumps({'mean_scores':averages,'mean_comparisons':aggregate}),flush=True)


if __name__ == '__main__': main()
