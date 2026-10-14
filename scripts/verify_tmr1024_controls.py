"""Check the new 1,024-clip TMR study without writing to historical studies."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from motion_release_audit_common import read, write, sha, seed, canonical
from tmr_audit_common import NativeTMR
from score_humanml_blind_fid import matrices

R = Path('reports/cvpr_tmr1024_controls_20261013')
OLD = Path('reports/cvpr_humanml3d_audit_20261012')


def score(T, E):
    # Reimplement pooling and ranking rather than calling the producer metric.
    tn = T / np.sqrt(np.sum(T * T, axis=1))[:, None]
    en = E / np.sqrt(np.sum(E * E, axis=1))[:, None]
    hits = np.zeros(3, dtype=int)
    for start in range(0, len(T), 32):
        sim = tn[start:start + 32] @ en[start:start + 32].T
        order = np.argsort(-sim, axis=1)
        for k in range(1, 4):
            hits[k - 1] += np.any(order[:, :k] == np.arange(len(sim))[:, None], axis=1).sum()
    return hits / len(T), float(np.mean(np.sum(tn * en, axis=1)))


def main():
    assert not (R / 'manifest_sha256.json').exists()
    assert read(R / 'complete.json')['all_sources_and_controls_completed']
    protocol = read(R / 'protocol.json')
    assert protocol['clips_each'] == 1024
    for path, digest in protocol['source_sha256'].items():
        assert sha(path) == digest, path
    seed(424)
    model = NativeTMR()
    captions = [x['caption'] for x in read(OLD / 'test_selection.json')]
    targets = model.texts(captions)
    sys.path.insert(0, str(Path('external/MDM').resolve()))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_from_rot
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
    from data_loaders.humanml.common.quaternion import cont6d_to_matrix
    rows = {}
    metric_count = 0
    maxima = {'matrix': 0., 'native_matrix': 0., 'angle_deg': 0., 'FK_m': 0., 'native_embedding': 0., 'text': 0., 'canonical_embedding': 0.}
    for source in ['real', 'mdm']:
        folder = R / source
        z = np.load(folder / 'source.npz')
        X, L, T = z['raw'], z['lengths'], z['text']
        assert len(X) == len(L) == 1024
        maxima['text'] = max(maxima['text'], float(abs(T - targets).max()))
        assert maxima['text'] < 3e-5
        mask = np.arange(X.shape[1])[None, :] < L[:, None]
        reported = read(folder / 'metrics.json')
        scale = model.std.cpu().numpy() + 1e-12
        source_rows = {}

        def check_metric(E, expected, texts=T):
            nonlocal metric_count
            r, c = score(texts, E)
            np.testing.assert_array_equal(r, expected['R_precision'])
            assert abs(c - expected['matched_cosine']) < 1e-7
            metric_count += 1

        check_metric(z['embedding'], reported['identity'])
        check_metric(z['canonical_embedding'], reported['canonical'])
        for condition in ['actual', 'wrong', 'canonical_gradient', 'generic_L2']:
            a = np.load(folder / (condition + '.npz'))
            Y, E, CE = a['raw'], a['embedding'], a['canonical_embedding']
            check_metric(E, reported[condition]['actual'])
            check_metric(CE, reported[condition]['canonical'])
            if condition != 'generic_L2':
                check_metric(E, reported[condition]['wrong'], np.roll(T, 1, axis=0))
            # Native forward checks use the saved prefix, with captions cached
            # in exactly the original 64-caption BERT batches.
            for raw, saved in [(Y[:8], E[:8]), (canonical(Y[:8]), CE[:8])]:
                er = float(abs(model.motions(raw, L[:8]) - saved).max())
                assert er < 3e-5
                maxima['native_embedding'] = max(maxima['native_embedding'], er)
            assert np.array_equal(X[~mask].view('u4'), Y[~mask].view('u4'))
            if condition != 'generic_L2':
                assert np.array_equal(X[..., :67].view('u4'), Y[..., :67].view('u4'))
                assert np.array_equal(X[..., 193:].view('u4'), Y[..., 193:].view('u4'))
                coeff = a['coefficients']
                assert abs(coeff).max() <= .3500001
                assert np.count_nonzero(coeff[~mask]) == 0
                # Independent coefficient-to-raw reconstruction, float64.
                d = X[..., 67:193].reshape(*X.shape[:2], 21, 6)
                e = Y[..., 67:193].reshape(*Y.shape[:2], 21, 6)
                reconstructed = np.concatenate([np.exp(coeff[..., 0:1].astype(float)) * d[..., :3], np.exp(coeff[..., 1:2].astype(float)) * d[..., 3:] + coeff[..., 2:3].astype(float) * d[..., :3]], axis=-1)
                assert abs(reconstructed[mask] - e[mask]).max() < 2e-6
                rm = 0.; rn = 0.; angle = 0.
                for i in range(0, len(X), 32):
                    valid = mask[i:i + 32]
                    p, q = d[i:i + 32][valid], e[i:i + 32][valid]
                    r0, r1 = matrices(p), matrices(q)
                    rm = max(rm, float(abs(r0 - r1).max()))
                    angle = max(angle, float(np.degrees(Rotation.from_matrix((r0.swapaxes(-1, -2) @ r1).reshape(-1, 3, 3)).magnitude()).max()))
                    rn = max(rn, float(abs(cont6d_to_matrix(torch.tensor(p)) - cont6d_to_matrix(torch.tensor(q))).max()))
                assert rm < 2e-5 and rn < 5e-5 and angle < .01
                maxima['matrix'] = max(maxima['matrix'], rm)
                maxima['native_matrix'] = max(maxima['native_matrix'], rn)
                maxima['angle_deg'] = max(maxima['angle_deg'], angle)
                ce = float(abs(CE - z['canonical_embedding']).max())
                assert ce < 3e-5
                maxima['canonical_embedding'] = max(maxima['canonical_embedding'], ce)
                for i in range(8):
                    x, y = torch.tensor(X[i, :L[i]]), torch.tensor(Y[i, :L[i]])
                    pos = recover_from_ric(x, 22)
                    assert torch.equal(pos, recover_from_ric(y, 22))
                    sk = Skeleton(torch.tensor(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
                    sk.get_offsets_joints(pos[0])
                    error = float(abs(recover_from_rot(x, 22, sk) - recover_from_rot(y, 22, sk)).max())
                    assert error < 2e-4
                    maxima['FK_m'] = max(maxima['FK_m'], error)
                source_rows[condition] = {'all1024_rotation_pairs_verified': True, 'protected_features_and_padding_bitwise_equal': True, 'canonical_embedding_max_change': ce, 'native_FK_sampled_clips': 8}
            else:
                norm = np.sqrt(np.sum((((Y - X) / scale) * mask[..., None]) ** 2, axis=(1, 2)))
                excess = float(np.max(norm - a['radii']))
                assert excess < 1e-3
                assert np.array_equal(X[..., 259:].view('u4'), Y[..., 259:].view('u4'))
                source_rows[condition] = {'standardized_L2_max_excess': excess, 'contacts_and_padding_preserved': True, 'not_a_physically_consistent_motion_intervention': True}
            print('TMR1024_CONTROL_VERIFIED', source, condition, flush=True)
        rows[source] = source_rows
    write(R / 'verification.json', {'passed': True, 'protocol_hashes_verified': True, 'independently_recomputed_metric_cells': metric_count, 'geometry_clip_pairs': 6144, 'native_FK_pairs': 48, 'maxima': maxima, 'conditions': rows, 'verifier_sha256': sha(__file__)})
    print('TMR1024_VERIFIED', maxima, flush=True)


if __name__ == '__main__':
    main()
