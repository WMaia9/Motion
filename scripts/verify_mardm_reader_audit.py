"""Verify public essential-reader weights, saved endpoints and native inference."""
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import t, kendalltau
from threadpoolctl import threadpool_limits
from motion_release_audit_common import read, write, sha, seed, metric
from mardm_eval_common import EssentialEvaluators, SOURCE, MODELS

R = Path('reports/cvpr_mardm_readers_20261013')
OLD = Path('reports/cvpr_motion_prevalence_20261012')
ADD = Path('reports/cvpr_additional_motion_audit_20261013')
MD = Path('reports/cvpr_motiondiffuse_compiled_20261013')
METHODS = {n: OLD / n for n in ['mdm50', 'mld', 'motionlcm_v1', 't2mgpt', 't2m']}
METHODS.update(momask=ADD / 'humanml/momask', dip=ADD / 'humanml/dip', motiondiffuse=MD / 'humanml/motiondiffuse')


def wait(path):
    while not path.is_file():time.sleep(10)


def main():
    assert not (R / 'manifest_sha256.json').exists()
    protocol = read(R / 'protocol.json')
    for path, h in protocol['inputs_source_sha256'].items():assert sha(path) == h, path
    assert protocol['source_revision'] == '9b05ad867be916afb9fcbff663b42a03f933014a'
    seed(4333); model = EssentialEvaluators()
    sys.path.append(str(Path('external/MDM').resolve()))
    cells = 0; infermax = 0.; rows = []
    with threadpool_limits(limits=2):
        for rep in range(20):
            folder = R / f'rep_{rep:02d}'; wait(folder / 'metrics.json'); row = read(folder / 'metrics.json'); rows.append(row)
            guoref = np.load(OLD / 'real_controls' / f'rep_{rep:02d}' / 'identity_motion_embeddings.npy')
            for name, root in METHODS.items():
                src = root / f'rep_{rep:02d}'
                for reader in ['guo_raw', 'guo_canonical', 'gru', 'clip']:
                    if reader.startswith('guo'):
                        T = np.load(src / 'text_embeddings.npy'); E = np.load(src / (reader[4:] + '_motion_embeddings.npy')); reference = guoref
                    else:
                        T = np.load(folder / (name + '_' + reader + '_text.npy')); E = np.load(folder / (name + '_' + reader + '_embedding.npy')); reference = np.load(folder / (reader + '_real_reference.npy'))
                        if reader == 'clip':
                            np.testing.assert_allclose(np.linalg.norm(T, axis=1), 1., atol=3e-6, rtol=0)
                            np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1., atol=3e-6, rtol=0)
                    result = metric(T, E, reference); expected = row['methods'][name][reader]
                    assert abs(result['FID'] - expected['FID']) < 2e-5
                    assert abs(result['MM_distance'] - expected['MM_distance']) < 1e-6
                    np.testing.assert_array_equal(result['R_precision'], expected['R_precision']); cells += 1
                invariant = row['invariance_checks'][name]
                assert invariant['first67_bitwise_equal'] and invariant['native_gauge_embedding_max_change'] == {'gru': 0., 'clip': 0.}
                if rep in [0, 19]:
                    raw = np.array(np.load(src / 'generated_raw.npy', mmap_mode='r')[:32]); z = np.load(src / 'conditioning.npz'); c = {k:z[k][:32].copy() for k in z.files}
                    if (src / 'generated_lengths.npy').exists():c['lengths'] = np.load(src / 'generated_lengths.npy')[:32]
                    E = model.encode(raw, c)
                    recoded = raw.copy(); d = raw[..., 67:193].reshape(32, 196, 21, 6)
                    recoded[..., 67:193] = np.concatenate([np.exp(.35) * d[..., :3], np.exp(-.35) * d[..., 3:] + .35 * d[..., :3]], -1).reshape(32, 196, 126)
                    assert np.array_equal(raw[..., :67].view('u4'), recoded[..., :67].view('u4'))
                    new = model.encode(recoded, c)
                    for reader in ['gru', 'clip']:
                        assert np.array_equal(new[reader][1], E[reader][1])
                        for value, suffix in zip(E[reader], ['text', 'embedding']):
                            saved = np.load(folder / (name + '_' + reader + '_' + suffix + '.npy'))[:32]
                            er = float(abs(value - saved).max()); assert er < 3e-5; infermax = max(infermax, er)
            print('MARDM_READER_DRAW_VERIFIED', rep, flush=True)
    wait(R / 'complete.json'); analysis = read(R / 'analysis.json')
    for name in METHODS:
        for reader in ['guo_raw', 'guo_canonical', 'gru', 'clip']:
            values = np.array([row['methods'][name][reader]['FID'] for row in rows]); expected = analysis['methods'][name][reader]
            half = t.ppf(.975, 19) * values.std(ddof=1) / np.sqrt(20)
            np.testing.assert_allclose(expected['t19_CI95'], [values.mean() - half, values.mean() + half], atol=1e-12, rtol=0)
            assert abs(expected['FID_mean'] - values.mean()) < 1e-12
            assert abs(expected['R1_mean'] - np.mean([row['methods'][name][reader]['R_precision'][0] for row in rows])) < 1e-12
    rank = analysis['mean_FID_rankings']
    for reader, names in rank.items():assert names == sorted(METHODS, key=lambda n: analysis['methods'][n][reader]['FID_mean'])
    for reader in ['gru', 'clip']:
        agreement = kendalltau([rank['guo_canonical'].index(n) for n in METHODS], [rank[reader].index(n) for n in METHODS]).statistic
        assert abs(agreement - analysis['Kendall_vs_canonical_Guo'][reader]) < 1e-15
    write(R / 'verification.json', {'passed': True, 'registered_input_hashes_verified': True, 'native_model_parameters_strictly_loaded_and_CLIP_state_bitwise_verified': True, 'all640_native_saved_metric_cells_recomputed': cells == 640, 'first_and_last_draw_native_forward_and_invariance_fixture_clips': 512, 'native_forward_max_difference': infermax, 'all20_structural_invariance_records_checked': True, 'all_32_intervals_and_rankings_recomputed': True, 'CLIP_FID_is_descriptive_not_official_leaderboard_metric': True, 'verifier_sha256': sha(__file__)})
    print('MARDM_READERS_VERIFIED', cells, infermax, flush=True)


if __name__ == '__main__':main()
