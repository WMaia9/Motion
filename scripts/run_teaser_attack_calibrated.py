"""Repeat the targeted recoding attack with the T2M-GPT-distributed evaluator checkpoint.

That checkpoint reproduces the published real-motion R-Precision (0.517 vs 0.511),
whereas the MDM-distributed one used originally gives 0.462.  Same 1,024 test clips,
same captions, candidate orders, bound 0.35, 80 Adam steps, lr 0.1 and wrong-caption
shift (N//2) as scripts/run_humanml_audit.py.  Inputs read only; new output directory.
"""
import json
from pathlib import Path

import numpy as np
import torch

from humanml_audit_common import (configure, load_evaluator, encode_raw, text_embeddings,
                                  canonical, transform, metrics, read, write, sha)

R = Path('reports/cvpr_humanml3d_audit_20261012')
OUT = Path('reports/cvpr_teaser_audit_20261014/calibrated_attack')
CK = Path('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar')
CONDITIONS = ['identity', 'canonical', 'target_caption', 'shifted_caption', 'canonical_reader_shifted_caption']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    movement, motion, text, mean, std = load_evaluator()
    s = torch.load(CK, map_location='cpu', weights_only=False)
    for m, k in [(movement, 'movement_encoder'), (motion, 'motion_encoder'), (text, 'text_encoder')]:
        m.load_state_dict(s[k], strict=True); m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    assert motion.gru.dropout == 0
    mean, std = torch.tensor(mean, device='cuda'), torch.tensor(std, device='cuda')
    sel = read(R / 'test_selection.json'); N = len(sel)
    T = text_embeddings(sel, text)
    orders = np.load(R / 'test_orders.npy')
    target = torch.tensor(T, device='cuda'); wrong = torch.roll(target, shifts=N // 2, dims=0)
    with np.load(R / 'test_real.npz') as z:
        XR, LR = z['raw'].copy(), z['lengths'].copy()
    with torch.no_grad():
        ref = np.concatenate([encode_raw(torch.tensor(XR[i:i + 32], device='cuda'), torch.tensor(LR[i:i + 32], device='cuda'),
                                         movement, motion, mean, std).cpu().numpy() for i in range(0, N, 32)])
    write(OUT / 'protocol.json', {'checkpoint': str(CK), 'checkpoint_sha256': sha(CK), 'clips': N, 'bound': 0.35,
                                  'steps': 80, 'lr': 0.1, 'wrong_shift': N // 2, 'code_sha256': sha(Path(__file__)),
                                  'same_selection_orders_as': str(R)})
    results = {}
    for source in ['real', 'mdm']:
        with np.load(R / f'test_{source}.npz') as z:
            X, L = z['raw'].copy(), z['lengths'].copy()
        results[source] = {}
        for cond in CONDITIONS:
            embs, cembs, ys = [], [], []
            for st in range(0, N, 32):
                x = torch.tensor(X[st:st + 32], device='cuda'); ll = torch.tensor(L[st:st + 32], device='cuda')
                mask = torch.arange(196, device='cuda')[None, :] < ll[:, None]; shape = (*x.shape[:2], 21, 3)
                if cond == 'identity':
                    y = x
                elif cond == 'canonical':
                    y = canonical(x)
                else:
                    q = torch.zeros(shape, device='cuda', requires_grad=True); opt = torch.optim.Adam([q], lr=.1)
                    goal = target[st:st + len(x)] if cond == 'target_caption' else wrong[st:st + len(x)]
                    motion.gru.train()
                    for _ in range(80):
                        c = .35 * torch.tanh(q) * mask[..., None, None]; y = transform(x, c)
                        z = canonical(y) if cond.startswith('canonical_reader') else y
                        loss = ((encode_raw(z, ll, movement, motion, mean, std) - goal) ** 2).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                    motion.gru.eval()
                    y = transform(x, .35 * torch.tanh(q.detach()) * mask[..., None, None])
                with torch.no_grad():
                    embs.append(encode_raw(y, ll, movement, motion, mean, std).cpu().numpy())
                    cembs.append(encode_raw(canonical(y), ll, movement, motion, mean, std).cpu().numpy())
                ys.append(y.detach().cpu().numpy())
            E, CE, Y = np.concatenate(embs), np.concatenate(cembs), np.concatenate(ys)
            assert np.array_equal(X[..., :67].view(np.uint32), Y[..., :67].view(np.uint32))
            assert np.array_equal(X[..., 193:].view(np.uint32), Y[..., 193:].view(np.uint32))
            np.savez_compressed(OUT / f'{source}_{cond}.npz', raw=Y, embeddings=E, canonical_embeddings=CE)
            act = metrics(E, T, ref, orders); sh = metrics(E, np.roll(T, N // 2, axis=0), ref, orders)
            can = metrics(CE, T, ref, orders)
            results[source][cond] = {'actual_R1': act['R_precision_top1'], 'shifted_R1': sh['R_precision_top1'],
                                     'FID': act['FID'], 'MM': act['MM_distance'],
                                     'canonical_actual_R1': can['R_precision_top1']}
            print(source, cond, {k: round(v, 3) for k, v in results[source][cond].items()}, flush=True)
            write(OUT / 'results.json', results)
    write(OUT / 'complete.json', {'all_conditions': True})


if __name__ == '__main__':
    main()
