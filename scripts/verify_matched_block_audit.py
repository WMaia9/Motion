"""Recompute matched-block scores and independently reconstruct saved fixtures.

The producer checks every output's norm. This verifier additionally checks all
saved norm arrays and reconstructs the first eight clips of every condition/draw.
It never claims a full native geometry replay for unsaved output arrays.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from motion_release_audit_common import read, write, sha, evaluator, score_embeddings, metric, seed, canonical
from score_humanml_blind_fid import matrices
from tmr_audit_common import NativeTMR
from verify_tmr1024_controls import score as cosine_score

R = Path('reports/cvpr_matched_blocks_20261013')
OLD = Path('reports/cvpr_motion_prevalence_20261012')
BLOCKS = {'velocity': (193, 259), 'contact': (259, 263), 'position': (4, 67)}
CONDITIONS = ['null_plus', 'null_minus', 'null_random', 'rotation', 'velocity', 'contact', 'position']


def reconstruct(x, L, std, condition, scalar, rep):
    # No call to the intervention, its search, or its basis helper.
    y = x.astype(float).copy()
    if condition in BLOCKS:
        begin, end = BLOCKS[condition]
        for i in range(len(x)):
            noise = np.random.default_rng(20261013 + 10000 * rep + i).normal(size=(196, end - begin))
            y[i, :, begin:end] += scalar[i] * noise * std[begin:end]
    else:
        d = x[..., 67:193].reshape(*x.shape[:2], 21, 6).astype(float)
        if condition.startswith('null'):
            for i in range(len(x)):
                v = np.random.default_rng(20261013 + 10000 * rep + i).normal(size=(196, 21, 3)) if condition == 'null_random' else np.array([1., -1., 1.]) * (1 if condition == 'null_plus' else -1)
                coeff = scalar[i] * v
                out = np.concatenate([np.exp(coeff[..., 0:1]) * d[i, ..., :3], np.exp(coeff[..., 1:2]) * d[i, ..., 3:] + coeff[..., 2:3] * d[i, ..., :3]], axis=-1)
                y[i, :, 67:193] = out.reshape(196, 126)
        else:
            rot = matrices(d).reshape(*x.shape[:2], 21, 3, 3)
            for i in range(len(x)):
                axes = np.random.default_rng(20261013 + 10000 * rep + i).normal(size=(196, 21, 3))
                axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
                Q = Rotation.from_rotvec((scalar[i] * axes).reshape(-1, 3)).as_matrix().reshape(196, 21, 3, 3)
                out = rot[i] @ Q
                y[i, :, 67:193] = np.concatenate([out[..., 0], out[..., 1]], axis=-1).reshape(196, 126)
        begin, end = 67, 193
    y = y.astype(x.dtype)
    y[..., :begin] = x[..., :begin]
    y[..., end:] = x[..., end:]
    mask = np.arange(196)[None, :] < L[:, None]
    y[~mask] = x[~mask]
    return y


def wait_for(path):
    while not path.is_file():
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader', choices=['guo', 'tmr'], required=True)
    args = parser.parse_args()
    assert not (R / 'manifest_sha256.json').exists()
    protocol = read(R / 'protocol.json')
    for path, h in protocol['producer_sha256'].items():
        assert sha(path) == h
    assert sha(OLD / 'manifest_sha256.json') == protocol['source_manifest_sha256']
    seed(444)
    if args.reader == 'guo':
        model, mean, std = evaluator()
    else:
        model = NativeTMR()
        std = model.std.cpu().numpy() + 1e-12
    root = R / args.reader
    np.testing.assert_array_equal(std, read(root / 'normalizer.json')['std'])
    sys.path.insert(0, str(Path('external/MDM').resolve()))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_from_rot
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
    checked = 0; fixture_pairs = 0; FK_pairs = 0; maxL2 = 0.; maxmatrix = 0.; maxFK = 0.; maxinfer = 0.
    all_rows = []
    with threadpool_limits(limits=2):
        for rep in range(20):
            folder = root / f'rep_{rep:02d}'
            wait_for(folder / 'metrics.json')
            row = read(folder / 'metrics.json'); all_rows.append(row)
            src = OLD / 'real_controls' / f'rep_{rep:02d}'
            X = np.load(src / 'source_raw.npy', mmap_mode='r'); z = np.load(src / 'conditioning.npz'); c = {k:z[k].copy() for k in z.files}; L = c['lengths']
            T = np.load(folder / 'text_embeddings.npy'); E0 = np.load(folder / 'reference_embeddings.npy'); EC0 = np.load(folder / 'canonical_source_embeddings.npy')
            if args.reader == 'tmr':model.texts(c['captions'].tolist())
            x = np.array(X[:8]); ll = L[:8]; mask = np.arange(196)[None, :] < ll[:, None]
            for label, reported in row['scores'].items():
                E = E0 if label == 'source' else EC0 if label == 'canonical_source' else np.load(folder / (label + '_embeddings.npy'))
                def check_metric(emb, report):
                    nonlocal checked
                    if args.reader == 'guo':
                        calculated = metric(T, emb, E0)
                        assert abs(calculated['FID'] - report['FID']) < 2e-5
                        assert abs(calculated['MM_distance'] - report['MM_distance']) < 1e-6
                        np.testing.assert_array_equal(calculated['R_precision'], report['R_precision'])
                    else:
                        ranks, cosine = cosine_score(T, emb)
                        np.testing.assert_array_equal(ranks, report['R_precision'])
                        assert abs(cosine - report['matched_cosine']) < 1e-7
                    checked += 1
                check_metric(E, reported)
                if label in ['source', 'canonical_source']:continue
                condition, radius = label.rsplit('_r', 1); radius = float(radius)
                p = np.load(folder / (label + '_parameters.npz'))
                np.testing.assert_array_equal(p['lengths'], L)
                mismatch = float(np.max(abs(p['realized_normalized_L2'] - radius * np.sqrt(L)) / (radius * np.sqrt(L))))
                assert mismatch < 5e-6
                Y = np.load(folder / (label + '_sample.npy'))
                expected = reconstruct(x, ll, std, condition, p['scalar'][:8], rep)
                assert abs(Y - expected).max() < 2e-6
                norm = np.linalg.norm((((Y.astype(float) - x) / std) * mask[..., None]).reshape(8, -1), axis=1)
                error = float(np.max(abs(norm - radius * np.sqrt(ll)) / (radius * np.sqrt(ll))))
                maxL2 = max(maxL2, error)
                assert error < 5e-6
                np.testing.assert_allclose(norm, p['realized_normalized_L2'][:8], atol=2e-7, rtol=2e-7)
                begin, end = BLOCKS.get(condition, (67, 193))
                assert np.array_equal(Y[..., :begin].view('u4'), x[..., :begin].view('u4'))
                assert np.array_equal(Y[..., end:].view('u4'), x[..., end:].view('u4'))
                assert np.array_equal(Y[~mask].view('u4'), x[~mask].view('u4'))
                if condition.startswith('null'):
                    d = x[..., 67:193].reshape(8, 196, 21, 6)[mask]; e = Y[..., 67:193].reshape(8, 196, 21, 6)[mask]
                    er = float(abs(matrices(d) - matrices(e)).max()); assert er < 2e-6; maxmatrix = max(maxmatrix, er)
                    C = np.load(folder / (label + '_canonical_embeddings.npy')); check_metric(C, row['checks'][label]['canonical_scores'])
                    assert abs(C - EC0).max() < 3e-5
                    px, py = torch.tensor(x[0, :ll[0]]), torch.tensor(Y[0, :ll[0]])
                    positions = recover_from_ric(px, 22); assert torch.equal(positions, recover_from_ric(py, 22))
                    sk = Skeleton(torch.tensor(t2m_raw_offsets), t2m_kinematic_chain, 'cpu'); sk.get_offsets_joints(positions[0])
                    er = float(abs(recover_from_rot(px, 22, sk) - recover_from_rot(py, 22, sk)).max()); assert er < 2e-4
                    maxFK = max(maxFK, er); FK_pairs += 1
                if condition == 'contact' and args.reader == 'guo':
                    assert np.array_equal(E, E0), 'Native Guo must ignore contact features'
                # One reader forward per condition on each draw, mapped back
                # through the native Guo caption/length ordering where necessary.
                if args.reader == 'tmr':
                    inferred = model.motions(Y, ll); saved = E[:8]
                else:
                    cc = {k:v[:8] for k,v in c.items()}; _, inferred = score_embeddings(model, Y, cc, mean, std)
                    ix32 = np.argsort(-c['sent_lengths'][:32], kind='stable'); order32 = ix32[np.argsort(L[:32][ix32])[::-1]]
                    ix8 = np.argsort(-c['sent_lengths'][:8], kind='stable'); order8 = ix8[np.argsort(ll[ix8])[::-1]]
                    saved = E[np.argsort(order32)[order8]]
                er = float(abs(inferred - saved).max()); assert er < 3e-5; maxinfer = max(maxinfer, er)
                fixture_pairs += 8
            print('MATCHED_BLOCK_DRAW_VERIFIED', args.reader, rep, flush=True)
    wait_for(root / 'complete.json')
    analysis = read(root / 'analysis.json')
    for label, record in analysis['conditions'].items():
        for endpoint, expected in record.items():
            def v(a):return a['R_precision'][0] if endpoint == 'R1' else a[endpoint]
            b = np.array([v(r['scores']['source']) for r in all_rows]); a = np.array([v(r['scores'][label]) for r in all_rows]); delta = a - b
            half = t.ppf(1 - .05 / (2 * 28), 19) * delta.std(ddof=1) / np.sqrt(20)
            np.testing.assert_allclose(expected['family28_CI95'], [delta.mean() - half, delta.mean() + half], atol=1e-12, rtol=0)
            assert abs(expected['mean_change'] - delta.mean()) < 1e-12
    write(root / 'verification.json', {'passed': True, 'native_saved_metric_cells': checked, 'norm_arrays_all_clips_all_draws_checked': True, 'independently_reconstructed_fixture_pairs': fixture_pairs, 'native_FK_sampled_pairs': FK_pairs, 'sample_L2_max_relative_error': maxL2, 'sample_rotation_matrix_max_change': maxmatrix, 'sample_native_FK_max_change_m': maxFK, 'sample_native_forward_max_difference': maxinfer, 'all_28_interval_endpoints_recomputed': True, 'full_unsaved_geometry_replay_claimed': False, 'verifier_sha256': sha(__file__)})
    print('MATCHED_BLOCK_VERIFIED', args.reader, checked, fixture_pairs, flush=True)


if __name__ == '__main__':main()
