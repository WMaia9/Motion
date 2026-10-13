"""Paired DEV effects, gradient audits and physical diagnostics."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_signspark_geometry_dev import statistics, b4
from analyze_motion_confirmation import bootstrap_wer
from run_signspark_geometry_dev import matrices, project
from run_fixed_prefix_diffusion import ROOT, OLD, ARMS
import run_physical_semantic_ddpm as previous


def main():
    names = ARMS + ['physical_generated_near_shuffled']
    results = {n: json.loads((ROOT/(n+'_scores.json')).read_text()) for n in names}
    for name in ['geometry', 'semantic_raw', 'semantic_physical']:
        results['previous_'+name] = json.loads((OLD/(name+'_scores.json')).read_text())
    references = results[names[0]]['references']
    ids = results[names[0]]['clip_ids']
    weights = np.random.default_rng(42).multinomial(128, np.ones(128)/128, size=2000)
    boot = {}; rows = []
    for name, result in results.items():
        assert result['clip_ids'] == ids and result['references'] == references
        for kind in ['raw', 'canonical']:
            stats = statistics(references, result[kind+'_hypotheses'])
            score = b4(stats.sum(0))
            assert abs(score-result[kind+'_scores']['bleu']['bleu4']) < 1e-10
            boot[name, kind] = np.asarray([b4(z) for z in weights@stats])
        rows.append({'condition':name, 'historical_reference':name.startswith('previous_'),
            'raw_bleu4':result['raw_scores']['bleu']['bleu4'],
            'canonical_bleu4':result['canonical_scores']['bleu']['bleu4'],
            'raw_gloss_wer':result['independent_raw_gloss']['wer'],
            'canonical_gloss_wer':result['independent_canonical_gloss']['wer'],
            'raw_canonical_changed_sentences':result['raw_canonical_changed_sentences']})
    candidate = 'physical_generated_near'
    pairs = [(candidate, n) for n in ['physical_ce', 'physical_real_near',
        'physical_generated_cyclic', 'raw_generated_near',
        'previous_geometry', 'previous_semantic_physical']]
    pairs.append(('physical_generated_near_shuffled', candidate))
    analysis = {'seed':42, 'replicates':2000, 'unit':'DEV clip, not signer',
        'exploratory':True, 'multiple_testing_adjustment':False,
        'train_seed_count':1, 'comparisons':{}, 'geometry':{}, 'gradient_audits':{},
        'stochastic_audits_equal':False}
    for left, right in pairs:
        result = {'historical_not_isolated_ablation':right.startswith('previous_')}
        for kind in ['raw', 'canonical']:
            delta = (results[left][kind+'_scores']['bleu']['bleu4'] -
                     results[right][kind+'_scores']['bleu']['bleu4'])
            result[kind+'_bleu4'] = {'delta':delta,
                'ci95':np.percentile(boot[left,kind]-boot[right,kind], [2.5,97.5]).tolist()}
            l = results[left]['independent_'+kind+'_gloss']
            r = results[right]['independent_'+kind+'_gloss']
            assert l['reference_lengths'] == r['reference_lengths']
            result[kind+'_independent_wer'] = bootstrap_wer(r['errors'], l['errors'], l['reference_lengths'])
        analysis['comparisons'][left+' minus '+right] = result
    stochastic = [json.loads((ROOT/(name+'_stochastic_audit.json')).read_text()) for name in ARMS]
    assert all(a == stochastic[0] for a in stochastic[1:])
    analysis['stochastic_audits_equal'] = True
    native = {e.sequence:e for e in previous.dataset('dev')}
    real = [project(native[name].sgn.numpy()) for name in ids]
    for name in names + ['ground_truth']:
        if name == 'ground_truth':
            motions = real
        else:
            with np.load(ROOT/(name+'_poses.npz')) as data:
                motions = [data[f'clip{i:04d}'].copy() for i in range(128)]
        speeds = []; errors = []; variabilities = []
        for motion, gt in zip(motions, real):
            assert motion.shape == gt.shape
            rotation = matrices(motion); actual = matrices(gt)
            delta = rotation.transpose(0,1,3,2) @ actual
            angle = np.degrees(Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude())
            change = rotation[:-1].transpose(0,1,3,2) @ rotation[1:]
            speed = np.degrees(Rotation.from_matrix(change.reshape(-1,3,3)).magnitude())
            errors.append(float(angle.mean()))
            speeds.append(float(speed.mean()) if len(motion)>1 else 0.)
            variabilities.append(float(np.std(project(motion),axis=0).mean()))
        analysis['geometry'][name] = {'mean_reference_frame_rotation_error_deg':float(np.mean(errors)),
            'mean_frame_rotation_change_deg':float(np.mean(speeds)),
            'mean_temporal_canonical_feature_std':float(np.mean(variabilities)),
            'interpretation':'same-frame errors, no DTW or human intelligibility claim'}
    for name in ARMS:
        analysis['gradient_audits'][name] = json.loads((ROOT/(name+'_gradient_audit.json')).read_text())
    (ROOT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    with (ROOT/'summary.csv').open('w') as output:
        writer = csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x = np.arange(5); labels = ['Physical CE','Real near','Generated cyclic','Generated near','Raw near']
    axes[0].bar(x-.2,[r['raw_bleu4'] for r in rows[:5]],.4,label='raw')
    axes[0].bar(x+.2,[r['canonical_bleu4'] for r in rows[:5]],.4,label='canonical')
    axes[0].set_ylabel('BLEU-4 — training reader');axes[0].legend()
    axes[1].bar(x,[r['canonical_gloss_wer'] for r in rows[:5]],color='#518975')
    axes[1].set_ylabel('Canonical gloss WER % — independent reader')
    for ax in axes:
        ax.set_xticks(x,labels,rotation=20);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed text prefix, physical negatives — PHOENIX DEV n=128')
    fig.savefig(ROOT/'fixed_prefix_comparison.png',dpi=180)
    fig.savefig(ROOT/'fixed_prefix_comparison.pdf');plt.close(fig)
    print('FIXED_PREFIX_ANALYSIS_COMPLETE',json.dumps(analysis['comparisons']),flush=True)


if __name__=='__main__':main()
