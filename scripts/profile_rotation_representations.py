"""Feature diagnostics on already saved real/produced motions, no inference."""
import json
from pathlib import Path
import numpy as np
from run_signspark_geometry_dev import project
from verify_signspark_studies import sha

ROOT = Path('reports/cvpr_reviewer_followup_20261011')


def profile(arrays):
    rng = np.random.default_rng(2027); sample = []; per_clip = []; degeneracies = 0
    for x in arrays:
        d = x.reshape(-1, 40, 6).astype(np.float64)
        a, b = d[...,:3], d[...,3:]
        norm = np.linalg.norm(a, axis=-1)
        e = a/np.maximum(norm[...,None], 1e-12)
        along = (e*b).sum(-1)
        residual = b-along[...,None]*e
        beta = np.linalg.norm(residual, axis=-1)
        gamma = along/np.maximum(norm, 1e-12)
        degeneracies += int(((norm < 1e-8) | (beta < 1e-8)).sum())
        coefficient = np.stack([norm, beta, gamma], -1)
        take = rng.choice(len(d), min(20, len(d)), replace=False)
        sample.append(coefficient[take])
        canonical = project(x)
        per_clip.append({'frames': len(x), 'maximum_canonicalization_change': float(abs(x-canonical).max()),
            'rms_canonicalization_change': float(np.sqrt(np.mean((x-canonical)**2))),
            'mean_abs_alpha_minus_one': float(abs(norm-1).mean()),
            'mean_abs_beta_minus_one': float(abs(beta-1).mean()), 'mean_abs_gamma': float(abs(gamma).mean())})
    values = np.concatenate(sample, axis=0)
    streams = {'body': slice(0,10), 'left_hand': slice(10,25), 'right_hand': slice(25,40)}
    quantiles = {}
    for stream, subset in streams.items():
        quantiles[stream] = {name: np.percentile(values[:,subset,j], [0,1,5,50,95,99,100]).tolist()
                            for j,name in enumerate(['alpha_norm_a', 'beta_norm_b_perpendicular', 'gamma'])}
    return {'clips': len(per_clip), 'degenerate_frame_joints': degeneracies,
        'quantile_levels_percent': [0,1,5,50,95,99,100],
        'quantile_sampling': 'at most 20 frames uniformly per clip, seed2027; not all-frame quantiles',
        'sampled_frames': len(values), 'stream_quantiles': quantiles,
        'clip_mean': {key: float(np.mean([r[key] for r in per_clip])) for key in per_clip[0]},
        'per_clip': per_clip}


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    paths = {
        'real_PHX_TEST_subset': (Path('reports/cvpr_evaluation_confirmation_20261011/heldout/PHOENIX-2014T/correct_caption_poses.npz'), 'source'),
        'real_How2Sign_TEST_subset': (Path('reports/cvpr_evaluation_confirmation_20261011/heldout/How2Sign/correct_caption_poses.npz'), 'source'),
        'real_CSL_TEST_subset': (Path('reports/cvpr_representation_paper_20261011/CSL-Daily/correct_caption_poses.npz'), 'source'),
    }
    for arm in ['geometry', 'velocity', 'physical_ce', 'physical_generated_near']:
        paths['generated_'+arm] = (Path('reports/cvpr_evaluation_confirmation_20261011/rankings')/arm/'poses.npz', 'clip')
    out = {'scope': 'post-review diagnostics of motions already used in the frozen paper; no new training or inference',
        'source_hashes': {}, 'profiles': {},
        'warning': 'coordinate distribution differences do not prove semantic signal in redundant coordinates'}
    for name,(path,prefix) in paths.items():
        with np.load(path) as z:
            keys = sorted(k for k in z.files if k.startswith(prefix))
            assert keys
            out['profiles'][name] = profile([z[k] for k in keys])
        out['source_hashes'][str(path)] = sha(path)
        print('FEATURE_PROFILE', name, out['profiles'][name]['clips'], out['profiles'][name]['clip_mean'], flush=True)
    (ROOT/'feature_profiles.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__': main()
