"""Paired analysis of the test-set embeddings saved by eval_reward_finetune_test.py.

Per model and evaluator: top-1 R-Precision (32-candidate groups, 5 fixed orders) and FID
against the real test motions, for raw and canonicalized outputs.  Per arm: means over seeds,
and a bootstrap over clips (2,000 resamples for R1 and FID, same resampled clips for raw and canonical) of
the raw-minus-canonical gaps, averaged over seeds.  Keys 'reward' and 'held' always denote the
MDM-linked and the T2M-GPT checkpoints; for the g* arms the T2M-GPT checkpoint gave the reward.
"""
import json
from pathlib import Path

import numpy as np

R = Path('reports/cvpr_reward_finetune_20261014/test')
ORDERS = [11, 97, 909, 1337, 2027]


_REF = {}


def fid(a, b):
    """Frechet distance; tr sqrt(ca cb) = sum sqrt eig(sb ca sb) with sb = cb^1/2 cached per reference."""
    key = id(b)
    if key not in _REF:
        cb = np.cov(b, rowvar=False); w, V = np.linalg.eigh(cb)
        _REF[key] = (b, b.mean(0), cb, (V * np.sqrt(np.clip(w, 0, None))) @ V.T)
    _, mr, cb, sb = _REF[key]
    d = a.mean(0) - mr; ca = np.cov(a, rowvar=False)
    e = np.linalg.eigvalsh(sb @ ca @ sb)
    return float(d @ d + np.trace(ca) + np.trace(cb) - 2 * np.sqrt(np.clip(e, 0, None)).sum())


def hits(E, T):
    n = len(E) // 32 * 32; h = np.zeros(len(E))
    for s in ORDERS:
        o = np.random.default_rng(s).permutation(n)
        for g in o.reshape(-1, 32):
            d = ((T[g][:, None] - E[g][None]) ** 2).sum(-1)
            h[g] += (d.argmin(1) == np.arange(32))
    return h / len(ORDERS)


def main():
    ref = dict(np.load(R / 'reference.npz'))  # fixed arrays, so the FID cache hits
    models = sorted(p.stem for p in R.glob('*.npz') if p.stem != 'reference')
    table, per = {}, {}
    for m in models:
        z = np.load(R / f'{m}.npz'); row = {'rms': float(z['rms'])}
        for k in ('reward', 'held'):
            for enc in ('raw', 'canonical'):
                E = z[f'{k}_{enc}']
                h = hits(E, ref[f'{k}_text']); per[(m, k, enc)] = (h, E)
                row[f'{k}_{enc}_R1'] = float(h.mean()); row[f'{k}_{enc}_FID'] = fid(E, ref[f'{k}_real'])
        table[m] = row
    arms = {}
    rng = np.random.default_rng(42)
    n = len(ref['reward_text'])
    idx = [rng.integers(0, n, n) for _ in range(2000)]
    for arm in ('raw', 'canonical', 'mse', 'graw', 'gcanonical'):  # g*: reward from the T2M-GPT checkpoint
        ms = [m for m in models if m.startswith(arm + '_s')]
        if not ms:
            continue
        a = {'seeds': ms}
        for k in ('reward', 'held'):
            for metric in ('R1', 'FID', 'rms'):
                if metric == 'rms':
                    a['rms'] = float(np.mean([table[m]['rms'] for m in ms])); continue
                a[f'{k}_raw_{metric}'] = float(np.mean([table[m][f'{k}_raw_{metric}'] for m in ms]))
                a[f'{k}_canonical_{metric}'] = float(np.mean([table[m][f'{k}_canonical_{metric}'] for m in ms]))
            gr, gf = [], []
            for b in idx[:2000]:
                r_, f_ = [], []
                for m in ms:
                    hr, Er = per[(m, k, 'raw')]; hc, Ec = per[(m, k, 'canonical')]
                    r_.append(hr[b].mean() - hc[b].mean())
                    f_.append(fid(Er[b], ref[f'{k}_real']) - fid(Ec[b], ref[f'{k}_real']))
                gr.append(np.mean(r_))
                gf.append(np.mean(f_))
            a[f'{k}_R1_gap'] = float(np.mean([table[m][f'{k}_raw_R1'] - table[m][f'{k}_canonical_R1'] for m in ms]))
            a[f'{k}_R1_gap_CI95'] = [float(np.percentile(gr, 2.5)), float(np.percentile(gr, 97.5))]
            a[f'{k}_FID_gap'] = float(np.mean([table[m][f'{k}_raw_FID'] - table[m][f'{k}_canonical_FID'] for m in ms]))
            a[f'{k}_FID_gap_CI95'] = [float(np.percentile(gf, 2.5)), float(np.percentile(gf, 97.5))]
        arms[arm] = a
    out = {'per_model': table, 'per_arm': arms}
    (R / 'analysis.json').write_text(json.dumps(out, indent=1))
    for m, r in table.items():
        print(f"{m:14s} rms {r['rms']:.4f} | reward R1 {r['reward_raw_R1']:.3f}/{r['reward_canonical_R1']:.3f} "
              f"FID {r['reward_raw_FID']:.3f}/{r['reward_canonical_FID']:.3f} | held R1 {r['held_raw_R1']:.3f} FID {r['held_raw_FID']:.3f}")
    print(json.dumps(arms, indent=1))


if __name__ == '__main__':
    main()
