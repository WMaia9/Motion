"""Paired analysis of the MLD test embeddings saved by reward_finetune_mld.py --test.

Same estimators as analyze_reward_finetune_test.py: top-1 R-Precision (32-candidate groups, five fixed
orders), FID against the real test motions, and 2,000 paired bootstrap resamples of clips for the
raw-minus-canonical gaps, averaged over seeds.  Keys 'reward'/'held' denote the MDM-linked and T2M-GPT
checkpoints; the MDM-linked checkpoint gave the reward.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_reward_finetune_test import fid, hits  # noqa: E402

R = Path('reports/cvpr_reward_finetune_20261014/test_mld')
RUN = 'mld_{arm}_kl0.0_w1.0_lr1e-05_t10_rmdm_s{seed}'


def main():
    ref = dict(np.load(R / 'reference.npz'))
    n = len(ref['reward_text']); rng = np.random.default_rng(42)
    idx = [rng.integers(0, n, n) for _ in range(2000)]
    out = {'per_model': {}, 'per_arm': {}}
    arms = {'pretrained': ['pretrained']} | {a: [RUN.format(arm=a, seed=s) for s in (42, 123, 777)] for a in ('mse', 'raw', 'canonical')}
    for arm, ms in arms.items():
        per = {}
        for m in ms:
            z = np.load(R / f'{m}.npz'); row = {'rms': float(z['rms'])}
            for k in ('reward', 'held'):
                for e in ('raw', 'canonical'):
                    h = hits(z[f'{k}_{e}'], ref[f'{k}_text']); per[(m, k, e)] = (h, z[f'{k}_{e}'])
                    row[f'{k}_{e}_R1'] = float(h.mean()); row[f'{k}_{e}_FID'] = fid(z[f'{k}_{e}'], ref[f'{k}_real'])
            out['per_model'][m] = row
        a = {'models': ms, 'rms': float(np.mean([out['per_model'][m]['rms'] for m in ms]))}
        for k in ('reward', 'held'):
            for met in ('R1', 'FID'):
                for e in ('raw', 'canonical'):
                    a[f'{k}_{e}_{met}'] = float(np.mean([out['per_model'][m][f'{k}_{e}_{met}'] for m in ms]))
            gr, gf = [], []
            for b in idx:
                gr.append(np.mean([per[(m, k, 'raw')][0][b].mean() - per[(m, k, 'canonical')][0][b].mean() for m in ms]))
                gf.append(np.mean([fid(per[(m, k, 'raw')][1][b], ref[f'{k}_real']) - fid(per[(m, k, 'canonical')][1][b], ref[f'{k}_real']) for m in ms]))
            a[f'{k}_R1_gap'] = a[f'{k}_raw_R1'] - a[f'{k}_canonical_R1']; a[f'{k}_R1_gap_CI95'] = [float(np.percentile(gr, q)) for q in (2.5, 97.5)]
            a[f'{k}_FID_gap'] = a[f'{k}_raw_FID'] - a[f'{k}_canonical_FID']; a[f'{k}_FID_gap_CI95'] = [float(np.percentile(gf, q)) for q in (2.5, 97.5)]
        out['per_arm'][arm] = a
        print(arm, 'rms %.4f' % a['rms'], {k: (round(a[k], 4), [round(x, 4) for x in a[k + '_CI95']]) for k in ('reward_FID_gap', 'reward_R1_gap', 'held_FID_gap', 'held_R1_gap')}, flush=True)
    (R / 'analysis.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
