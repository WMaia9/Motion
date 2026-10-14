"""Query-only (black-box) caption-blind recoding attack on HumanML3D FID.

The attacker may only ask for one scalar: the FID of the recoded outputs
against embeddings of 2,048 training motions, computed by the primary T2M
evaluator.  It receives no gradients, no per-clip embeddings, no captions and
no test reference.  A (1+1) evolution strategy searches either 3 global
recoding coefficients or 3 coefficients per joint (63), within a bound.  The
best coefficients found after 10/30/100/300 queries are then scored once
against the withheld test reference with the primary evaluator and with the
second official checkpoint, which the attacker never queried.  Saved cohorts
are read only; results go to a new report directory.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from humanml_audit_common import (configure, load_evaluator, encode_raw, text_embeddings,
                                  metrics, canonical, transform, read, write, sha)

OUT = Path('reports/cvpr_blackbox_attack_20261014')
COHORTS = {
    'mdm42': Path('reports/cvpr_humanml3d_blind_fid_20261012'),
    'mdm4242': Path('reports/cvpr_humanml_blind_repetitions_20261012/cohort_1'),
    'mdm7777': Path('reports/cvpr_humanml_blind_repetitions_20261012/cohort_2'),
    'mld': Path('reports/cvpr_blind_transfer_20261013/mld'),
}
SECOND = Path('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar')
BUDGETS = [10, 30, 100, 300]
BOUNDS = [0.1, 0.35]
SPACES = {'global': (1, 3), 'per_joint': (21, 3)}
ORDERS = [11, 97, 909, 1337, 2027]


def fid(a, b):
    from scipy import linalg
    mu, mr = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = linalg.sqrtm(ca @ cb)
    if np.iscomplexobj(root):
        root = root.real
    return max(float((mu - mr) @ (mu - mr) + np.trace(ca + cb - 2 * root)), 0.0)


class Reader:
    def __init__(self, checkpoint=None):
        self.movement, self.motion, self.text, mean, std = load_evaluator()
        if checkpoint is not None:
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            for model, key in [(self.movement, 'movement_encoder'), (self.motion, 'motion_encoder'),
                               (self.text, 'text_encoder')]:
                model.load_state_dict(ck[key], strict=True)
                model.eval()
        assert self.motion.gru.dropout == 0
        self.mean = torch.tensor(mean, device='cuda')
        self.std = torch.tensor(std, device='cuda')

    @torch.no_grad()
    def embed(self, X, L, batch=320):
        out = []
        for i in range(0, len(X), batch):
            x = X[i:i + batch] if torch.is_tensor(X) else torch.tensor(X[i:i + batch], device='cuda')
            l = L[i:i + batch] if torch.is_tensor(L) else torch.tensor(L[i:i + batch], device='cuda')
            out.append(encode_raw(x, l, self.movement, self.motion, self.mean, self.std).cpu().numpy())
        return np.concatenate(out)


def recode(X, L, theta, shape):
    c = torch.tensor(theta.reshape(shape), device='cuda', dtype=torch.float32)
    c = c.reshape(1, 1, shape[0], 3).expand(X.shape[0], X.shape[1], 21, 3)
    mask = torch.arange(X.shape[1], device='cuda')[None, :] < L[:, None]
    return transform(X, c * mask[..., None, None])


def search(oracle, dim, bound, budget, seed):
    rng = np.random.default_rng(seed)
    theta = np.zeros(dim)
    best = oracle(theta)
    sigma = 0.3 * bound
    trace, snapshots = [best], {}
    for q in range(2, budget + 1):
        cand = np.clip(theta + sigma * rng.standard_normal(dim), -bound, bound)
        val = oracle(cand)
        if val < best:
            theta, best, sigma = cand, val, min(sigma * 1.5, bound)
        else:
            sigma = max(sigma * 0.9, 0.01 * bound)
        trace.append(best)
        if q in BUDGETS:
            snapshots[q] = (theta.copy(), best)
    snapshots[1] = (np.zeros(dim), trace[0])
    return snapshots, trace


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    primary = Reader()
    second = Reader(SECOND)
    protocol = {
        'access': 'scalar FID against 2,048 training-motion embeddings from the primary reader; no gradients, '
                  'no per-clip embeddings, no captions, no test reference',
        'search': '(1+1)-ES, sigma0=0.3*bound, x1.5 on success, x0.9 on failure, start at identity, seed 0',
        'budgets_queries': BUDGETS, 'bounds': BOUNDS, 'spaces': {k: v[0] * v[1] for k, v in SPACES.items()},
        'evaluation': 'withheld test reference, primary and second official checkpoint; second never queried',
        'random_control': '10 uniform random coefficient vectors per space/bound, no search',
        'sources_sha256': {str(p): sha(p) for p in [Path(__file__), Path('scripts/humanml_audit_common.py'), SECOND]},
    }
    write(OUT / 'protocol.json', protocol)
    results = {}
    for name, src in COHORTS.items():
        t0 = time.time()
        train = np.load(src / 'TRAIN_motion_embeddings.npy')
        with np.load(src / 'test_mdm.npz') as z:
            X = torch.tensor(z['raw'], device='cuda')
            L = torch.tensor(z['lengths'], device='cuda')
        with np.load(src / 'test_real.npz') as z:
            ref1 = primary.embed(z['raw'], z['lengths'])
            ref2 = second.embed(z['raw'], z['lengths'])
        sel = read(src / 'test_selection.json')
        T1 = text_embeddings(sel, primary.text)
        T2 = text_embeddings(sel, second.text)
        orders = np.array([np.random.default_rng(s).permutation(len(sel)) for s in ORDERS])

        def score(Y):
            e1, e2 = primary.embed(Y, L), second.embed(Y, L)
            m1, m2 = metrics(e1, T1, ref1, orders), metrics(e2, T2, ref2, orders)
            return {'primary_FID': m1['FID'], 'primary_R1': m1['R_precision_top1'],
                    'second_FID': m2['FID'], 'second_R1': m2['R_precision_top1']}

        base = score(X)
        canon = score(canonical(X))
        cohort = {'identity': base, 'canonical': canon, 'runs': {}}
        for space, shape in SPACES.items():
            dim = shape[0] * shape[1]
            for bound in BOUNDS:
                calls = [0]

                def oracle(theta):
                    calls[0] += 1
                    return fid(primary.embed(recode(X, L, theta, shape), L), train)

                snaps, trace = search(oracle, dim, bound, max(BUDGETS), seed=0)
                assert calls[0] == max(BUDGETS)
                rng = np.random.default_rng(1)
                rand = [score(recode(X, L, rng.uniform(-bound, bound, dim), shape)) for _ in range(10)]
                run = {'oracle_trace_train_FID': trace, 'random_control_mean': {
                    k: float(np.mean([r[k] for r in rand])) for k in rand[0]}, 'budgets': {}}
                for q in BUDGETS:
                    theta, oracle_val = snaps[q]
                    Y = recode(X, L, theta, shape)
                    assert torch.equal(Y[..., :67], X[..., :67]) and torch.equal(Y[..., 193:], X[..., 193:])
                    ce = (primary.embed(canonical(Y), L) - primary.embed(canonical(X), L))
                    s = score(Y)
                    s.update({'queries': q, 'oracle_train_FID': oracle_val, 'theta': theta.tolist(),
                              'canonical_embedding_max_change': float(np.abs(ce).max()),
                              'primary_FID_reduction': 1 - s['primary_FID'] / base['primary_FID'],
                              'second_FID_reduction': 1 - s['second_FID'] / base['second_FID']})
                    run['budgets'][q] = s
                cohort['runs'][f'{space}_b{bound}'] = run
                best = run['budgets'][max(BUDGETS)]
                print(f"{name} {space} b={bound}: test FID {base['primary_FID']:.3f}->{best['primary_FID']:.3f} "
                      f"({best['primary_FID_reduction']:+.1%}); second {base['second_FID']:.3f}->"
                      f"{best['second_FID']:.3f} ({best['second_FID_reduction']:+.1%})", flush=True)
        cohort['seconds'] = time.time() - t0
        results[name] = cohort
        write(OUT / 'results.json', results)
    write(OUT / 'complete.json', {'cohorts': list(COHORTS), 'all_runs_completed': True})


if __name__ == '__main__':
    main()
