"""Verify release-level summaries and independently recompute rotation RMS."""
from pathlib import Path
import itertools
import numpy as np
from scipy.stats import spearmanr
from motion_release_audit_common import read, write, sha

R = Path('reports/cvpr_rotation_mechanism_20261013')
OLD = Path('reports/cvpr_motion_prevalence_20261012')
ADD = Path('reports/cvpr_additional_motion_audit_20261013')
MD = Path('reports/cvpr_motiondiffuse_compiled_20261013')


def rms(folder):
    X = np.load(folder / 'generated_raw.npy', mmap_mode='r')
    if (folder / 'generated_lengths.npy').exists():L = np.load(folder / 'generated_lengths.npy')
    else:L = np.load(folder / 'conditioning.npz')['lengths']
    total = 0.; count = 0
    for start in range(0, len(X), 32):
        d = np.array(X[start:start + 32, :, 67:193]).reshape(-1, 196, 21, 6)
        v = d.astype(float)
        first = v[..., :3] / np.sqrt(np.sum(v[..., :3] ** 2, axis=-1))[..., None]
        second = v[..., 3:] - np.sum(first * v[..., 3:], axis=-1)[..., None] * first
        second /= np.sqrt(np.sum(second ** 2, axis=-1))[..., None]
        C = np.concatenate([first, second], axis=-1).astype(d.dtype)
        valid = np.arange(196)[None, :] < L[start:start + 32, None]
        delta = v[valid] - C[valid]
        total += np.sum(delta * delta); count += delta.size
    return float(np.sqrt(total / count))


def main():
    assert not (R / 'manifest_sha256.json').exists()
    assert read(R / 'complete.json')['all8release20draw_RMS_sources']
    for path, h in read(R / 'protocol.json')['source_sha256'].items():assert sha(path) == h
    a = read(R / 'analysis.json'); names = list(a['methods']); draws = 0; max_error = 0.
    for name, row in a['methods'].items():
        new = name in ['momask', 'dip', 'motiondiffuse']
        root = MD / 'humanml/motiondiffuse' if name == 'motiondiffuse' else ADD / 'humanml' / name if new else OLD / name
        repetitions = range(20) if new else [0]
        for rep in repetitions:
            expected = row['rotation_RMS_per_draw'][rep]
            calculated = rms(root / f'rep_{rep:02d}')
            error = abs(calculated - expected); assert error < 1e-8
            max_error = max(max_error, error); draws += 1
        assert abs(np.mean(row['rotation_RMS_per_draw']) - row['rotation_RMS_mean']) < 1e-15
        assert abs(row['FID']['raw'] - row['FID']['canonical'] - row['FID']['delta']) < 1e-14
        print('MECHANISM_RMS_VERIFIED', name, flush=True)
    for family, row in a['family_descriptive_means'].items():
        expected = np.mean([a['methods'][n]['FID']['delta'] for n in row['methods']])
        assert abs(row['mean_signed_gap'] - expected) < 1e-15
    x = [a['methods'][n]['rotation_RMS_mean'] for n in names]
    permutations = list(itertools.permutations(range(8)))
    # Independent rank construction; enumerate exact permutations of rank vectors.
    rx = np.argsort(np.argsort(x)).astype(float) - 3.5
    for label, endpoint in a['correlations'].items():
        y = np.array([a['methods'][n]['FID']['delta'] for n in names])
        if label == 'absolute':y = abs(y)
        ry = np.argsort(np.argsort(y)).astype(float) - 3.5
        observed = float(spearmanr(x, y).statistic)
        null = np.array([rx @ ry[list(p)] / 42 for p in permutations])
        pvalue = float(np.mean(abs(null) >= abs(observed) - 1e-12))
        assert abs(observed - endpoint['Spearman_rho']) < 1e-12
        assert abs(pvalue - endpoint['exact_permutation_p']) < 1e-15
        assert endpoint['two_endpoint_adjusted_p'] == min(1., 2 * pvalue)
    write(R / 'verification.json', {'passed': True, 'independent_raw_RMS_recomputations': draws, 'all60_new_release_draws_and_original_five_first_draws': True, 'maximum_RMS_recompute_difference': max_error, 'all8_means_signed_deltas_families_and_exact_correlations_verified': True, 'historical_old_RMS_draws_previously_verified_and_frozen': True, 'no_population_or_causal_inference': True, 'verifier_sha256': sha(__file__)})
    print('ROTATION_MECHANISM_VERIFIED', draws, max_error, flush=True)


if __name__ == '__main__':main()
