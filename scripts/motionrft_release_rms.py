"""Off-canonical RMS of the released MotionRFT model and its base, from the saved outputs of
run_motion_reward_release.py (five native cohorts each; no new inference)."""
import json
from pathlib import Path

import numpy as np

R = Path('reports/cvpr_motion_reward_release_20261013')
P = Path('reports/cvpr_motion_prevalence_20261012/mdm50')
OUT = Path('reports/cvpr_reward_finetune_20261014/motionrft_release_rms.json')


def rms(x, L):
    d = x[..., 67:193].reshape(*x.shape[:2], 21, 6).astype(np.float64)
    mask = np.arange(x.shape[1])[None] < L[:, None]
    d = d[mask]
    a, b = d[..., :3], d[..., 3:]
    u = a / np.linalg.norm(a, axis=-1, keepdims=True)
    v = b - (u * b).sum(-1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return float(np.sqrt(((np.concatenate([u, v], -1) - d) ** 2).mean()))


def main():
    out = {}
    for m in ('base', 'reward_finetuned'):
        out[m] = [rms(np.load(R / m / f'rep_{k:02d}' / 'generated_raw.npy'),
                      np.load(P / f'rep_{k:02d}' / 'conditioning.npz')['lengths']) for k in range(5)]
    an = json.loads((R / 'analysis.json').read_text())['readers']
    out['raw_minus_canonical'] = {ck: {m: {k: {'mean': an[ck][m][k]['paired_raw_minus_canonical'],
                                              'CI95_bonferroni8': an[ck][m][k]['paired_family_CI95']}
                                          for k in ('FID', 'R1')} for m in ('base', 'reward_finetuned')} for ck in an}
    out['note'] = 'off-canonical RMS over valid frames of the 126 rotation channels; 5 cohorts of 1,024 outputs'
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({m: [round(x, 4) for x in out[m]] for m in ('base', 'reward_finetuned')}))


if __name__ == '__main__':
    main()
