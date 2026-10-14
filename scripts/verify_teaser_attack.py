"""Independent check of the teaser result (targeted recoding, 1,024 HumanML3D test clips).

Recomputes from the saved arrays, with code written separately from the producer:
  1. geometry: non-rotation channels bitwise equal; rotation matrices equal;
  2. R-Precision top-1/2/3, MM-distance and FID for real and MDM motion,
     identity / attacked / canonicalized-attacked, with BOTH official checkpoints;
  3. calibration of each checkpoint on real motion against published values.
Inputs are read only; results go to a new report directory.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from humanml_audit_common import configure, load_evaluator, encode_raw, text_embeddings, canonical, read, write

R = Path('reports/cvpr_humanml3d_audit_20261012')
OUT = Path('reports/cvpr_teaser_audit_20261014')
SECOND = Path('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar')
PUBLISHED_REAL = {'R1': 0.511, 'R2': 0.703, 'R3': 0.797, 'MM': 2.974}  # Guo et al. 2022, Real row


def rot(d):
    a, b = d[..., :3].astype(np.float64), d[..., 3:].astype(np.float64)
    u = a / np.linalg.norm(a, axis=-1, keepdims=True)
    v = b - (u * b).sum(-1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return np.stack([u, v, np.cross(u, v)], -1)


def rprec(emb, txt, orders):
    tops = np.zeros(3)
    for order in orders:
        hit = np.zeros(3)
        for idx in order.reshape(-1, 32):
            d = ((txt[idx][:, None, :] - emb[idx][None, :, :]) ** 2).sum(-1)  # text i vs motion j
            rank = np.argsort(np.argsort(d, axis=1, kind='stable'), axis=1, kind='stable')
            r = np.diag(rank)
            for k in range(3):
                hit[k] += (r <= k).sum()
        tops += hit / len(emb)
    return tops / len(orders)


def fid(a, b):
    mu, mr = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    s = linalg.sqrtm(ca @ cb)
    s = s.real if np.iscomplexobj(s) else s
    return float((mu - mr) @ (mu - mr) + np.trace(ca + cb - 2 * s))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    sel = read(R / 'test_selection.json')
    orders = np.load(R / 'test_orders.npy')
    readers = {}
    for name, ck in [('mdm_linked_epoch11', None), ('t2mgpt_epoch28', SECOND)]:
        mv, mo, tx, mean, std = load_evaluator()
        if ck is not None:
            s = torch.load(ck, map_location='cpu', weights_only=False)
            for m, k in [(mv, 'movement_encoder'), (mo, 'motion_encoder'), (tx, 'text_encoder')]:
                m.load_state_dict(s[k], strict=True); m.eval()
        readers[name] = (mv, mo, tx, torch.tensor(mean, device='cuda'), torch.tensor(std, device='cuda'))

    @torch.no_grad()
    def embed(rd, X, L):
        mv, mo, _, mean, std = rd
        return np.concatenate([encode_raw(torch.tensor(X[i:i + 32], device='cuda'),
                                          torch.tensor(L[i:i + 32], device='cuda'), mv, mo, mean, std).cpu().numpy()
                               for i in range(0, len(X), 32)])

    with np.load(R / 'test_real.npz') as z:
        XR, LR = z['raw'].copy(), z['lengths'].copy()
    geometry, scores = {}, {}
    for source in ['real', 'mdm']:
        with np.load(R / f'test_{source}.npz') as z:
            X, L = z['raw'].copy(), z['lengths'].copy()
        with np.load(R / 'test' / source / 'target_caption.npz') as z:
            Y = z['raw'].copy()
        # geometry
        assert np.array_equal(X[..., :67].view(np.uint32), Y[..., :67].view(np.uint32))
        assert np.array_equal(X[..., 193:].view(np.uint32), Y[..., 193:].view(np.uint32))
        err = 0.0
        for i, n in enumerate(L):
            r0 = rot(X[i, :n, 67:193].reshape(n, 21, 6))
            r1 = rot(Y[i, :n, 67:193].reshape(n, 21, 6))
            err = max(err, float(np.abs(r0 - r1).max()))
        changed = float(np.abs(Y[..., 67:193] - X[..., 67:193]).max())
        geometry[source] = {'nonrotation_bitwise_equal': True, 'max_rotation_matrix_diff': err,
                            'max_raw_6D_change': changed}
        Cy = np.concatenate([canonical(torch.tensor(Y[i:i + 32])).numpy() for i in range(0, len(Y), 32)])
        for rn, rd in readers.items():
            T = text_embeddings(sel, rd[2])
            ref = embed(rd, XR, LR)
            row = {}
            for cond, arr in [('identity', X), ('attacked', Y), ('canonical_attacked', Cy)]:
                e = embed(rd, arr, L)
                r = rprec(e, T, orders)
                row[cond] = {'R1': float(r[0]), 'R2': float(r[1]), 'R3': float(r[2]),
                             'MM': float(np.linalg.norm(e - T, axis=1).mean()), 'FID': fid(e, ref)}
            scores.setdefault(rn, {})[source] = row
            print(rn, source, {c: round(v['R1'], 3) for c, v in row.items()}, flush=True)
    saved = {s: read(R / 'test' / s / 'target_caption.json')['actual_caption']['R_precision_top1'] for s in ['real', 'mdm']}
    saved_id = {s: read(R / 'test' / s / 'identity.json')['actual_caption']['R_precision_top1'] for s in ['real', 'mdm']}
    calib = {rn: {k: scores[rn]['real']['identity'][k] for k in ['R1', 'R2', 'R3', 'MM']} for rn in readers}
    out = {'geometry': geometry, 'scores': scores, 'saved_producer_R1': {'attacked': saved, 'identity': saved_id},
           'real_motion_calibration_1024_subset': calib, 'published_real_HumanML3D': PUBLISHED_REAL}
    write(OUT / 'analysis.json', out)
    print(json.dumps({'geometry': geometry, 'calibration': calib, 'saved': out['saved_producer_R1']}, indent=1))


if __name__ == '__main__':
    main()
