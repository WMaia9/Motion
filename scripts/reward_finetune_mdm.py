"""Reward fine-tuning of MDM against a learned evaluator (ReFL/DRaFT style).

Does a generator trained to please a raw-feature evaluator learn to use the
redundant 6D coordinates?  Three arms share everything but the reward input:

  mse        continued diffusion training only (control)
  raw        + evaluator reward on the generated motion as is
  canonical  + evaluator reward on the canonicalized generated motion (the
             redundant coordinates receive no reward gradient)

Each step samples motions for training captions with the model's own 50-step,
CFG-2.5 sampler, without gradients down to a random late step t_k, then predicts
x0 at t_k with gradients and applies a Euclidean InfoNCE reward from the frozen
MDM-linked T2M evaluator.  The usual diffusion x0 loss on real training motion
regularizes.  Periodic validation uses fixed val captions and reports, for the
reward evaluator and a held-out evaluator (T2M-GPT checkpoint), raw and
canonical R-Precision and FID, plus how far outputs are from canonical.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent))
from motion_release_audit_common import A, evaluator, seed as set_seed, write  # noqa: E402
from run_motion_release_audit import mdm_model  # noqa: E402

DATA = Path('models/humanml3d_audit/data/HumanML3D')
SECOND = Path('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar')
OUT = Path('reports/cvpr_reward_finetune_20261014')
ROT = slice(67, 193)


# ---------------------------------------------------------------- data
def load_split(split, rng, max_items=None):
    ids = [l.strip() for l in open(DATA / f'{split}.txt') if l.strip()]
    rng.shuffle(ids)
    items = []
    for i in ids:
        f = DATA / 'new_joint_vecs' / f'{i}.npy'
        if not f.exists():
            continue
        m = np.load(f)
        n = min(len(m), 196) // 4 * 4
        if n < 40:
            continue
        caps = []
        for line in open(DATA / 'texts' / f'{i}.txt'):
            parts = line.strip().split('#')
            if len(parts) >= 4 and float(parts[2]) == 0 and float(parts[3]) == 0:
                caps.append((parts[0], parts[1]))
        if not caps:
            continue
        items.append({'id': i, 'motion': m[:n].astype(np.float32), 'len': n, 'caps': caps})
        if max_items and len(items) >= max_items:
            break
    return items


class TextEncoder:
    def __init__(self, wrapper):
        sys.path.insert(0, str(Path('external/MDM').resolve()))
        from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
        self.vec = WordVectorizer(str(A / 'glove'), 'our_vab')
        self.enc = wrapper.text_encoder

    @torch.no_grad()
    def __call__(self, tokens_list):
        W, P, L = [], [], []
        for tok in tokens_list:
            t = ['sos/OTHER'] + tok.split()[:20] + ['eos/OTHER']
            L.append(len(t)); t += ['unk/OTHER'] * (22 - len(t))
            v = [self.vec[z] for z in t]
            W.append(np.stack([z[0] for z in v])); P.append(np.stack([z[1] for z in v]))
        l = torch.tensor(L); o = torch.argsort(l, descending=True, stable=True); inv = torch.argsort(o)
        w = torch.tensor(np.array(W), device='cuda').float(); p = torch.tensor(np.array(P), device='cuda').float()
        return self.enc(w[o], p[o], l[o])[inv]


def motion_embed(wrapper, x, L):
    o = torch.argsort(L, descending=True, stable=True); inv = torch.argsort(o)
    return wrapper.motion_encoder(wrapper.movement_encoder(x[o][..., :259]), (L[o] // 4).cpu())[inv]


def canonical_t(x):
    y = x.clone(); d = x[..., ROT].reshape(*x.shape[:-1], 21, 6)
    u = F.normalize(d[..., :3], dim=-1); b = d[..., 3:]
    v = F.normalize(b - (u * b).sum(-1, keepdim=True) * u, dim=-1)
    y[..., ROT] = torch.cat([u, v], -1).reshape(*x.shape[:-1], 126); return y


def fid(a, b):
    mu, mr = a.mean(0), b.mean(0); ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    s = linalg.sqrtm(ca @ cb); s = s.real if np.iscomplexobj(s) else s
    return float((mu - mr) @ (mu - mr) + np.trace(ca + cb - 2 * s))


def r1(E, T, orders):
    hits = []
    for o in orders:
        h = 0
        for idx in o.reshape(-1, 32):
            d = torch.cdist(torch.tensor(T[idx]), torch.tensor(E[idx]))
            h += int((d.argmin(1) == torch.arange(32)).sum())
        hits.append(h / len(E))
    return float(np.mean(hits))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['mse', 'raw', 'canonical'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight', type=float, default=0.1, help='reward weight')
    ap.add_argument('--tau', type=float, default=1.0, help='InfoNCE temperature on Euclidean distances')
    ap.add_argument('--tmax', type=int, default=10, help='reward at a random step in [0, tmax]')
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--dev-n', type=int, default=512)
    ap.add_argument('--reg', choices=['mse', 'canon'], default='mse',
                    help='mse: diffusion loss on raw features; canon: on canonicalized prediction (fiber left free)')
    ap.add_argument('--kl', type=float, default=0.0,
                    help='weight of a canonical-invariant penalty keeping x0 predictions close to the frozen pretrained model')
    ap.add_argument('--reward-ckpt', choices=['mdm', 'gpt'], default='mdm',
                    help='T2M checkpoint giving the reward; the other one is the held-out judge')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    name = a.name or f'{a.arm}_reg{a.reg}_kl{a.kl}_w{a.weight}_lr{a.lr}_t{a.tmax}_tau{a.tau}_s{a.seed}'
    if a.reward_ckpt != 'mdm' and not a.name:
        name += f'_r{a.reward_ckpt}'
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists(), f'{out} already complete'
    write(out / 'config.json', vars(a))

    set_seed(a.seed); rng = np.random.default_rng(a.seed)
    train = load_split('train', np.random.default_rng(0))
    val = load_split('val', np.random.default_rng(1), max_items=a.dev_n)
    print('train', len(train), 'val', len(val), flush=True)

    model, guided, diffusion = mdm_model()
    ref_guided = None
    if a.kl > 0:
        ref_model, ref_guided, _ = mdm_model()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)
        ref_model.eval()
    for p in model.clip_model.parameters():
        p.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('clip_model.')]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    mu = torch.tensor(np.load('external/HumanML3D/HumanML3D/Mean.npy'), device='cuda', dtype=torch.float32)
    sd = torch.tensor(np.load('external/HumanML3D/HumanML3D/Std.npy'), device='cuda', dtype=torch.float32)

    reward_ev, emu, esd = evaluator()
    held_ev, _, _ = evaluator()
    ck = torch.load(SECOND, map_location='cpu', weights_only=False)
    for m, k in [(held_ev.movement_encoder, 'movement_encoder'), (held_ev.motion_encoder, 'motion_encoder'),
                 (held_ev.text_encoder, 'text_encoder')]:
        m.load_state_dict(ck[k], strict=True); m.eval()
    if a.reward_ckpt == 'gpt':  # both checkpoints share the same input statistics
        reward_ev, held_ev = held_ev, reward_ev
    emu = torch.tensor(emu, device='cuda', dtype=torch.float32); esd = torch.tensor(esd, device='cuda', dtype=torch.float32)
    txt_r, txt_h = TextEncoder(reward_ev), TextEncoder(held_ev)

    # fixed validation set
    vrng = np.random.default_rng(7)
    vcap = [it['caps'][vrng.integers(len(it['caps']))] for it in val]
    vlen = torch.tensor([it['len'] for it in val], device='cuda')
    vreal = np.zeros((len(val), 196, 263), np.float32)
    for i, it in enumerate(val):
        vreal[i, :it['len']] = it['motion']
    orders = np.array([np.random.default_rng(s).permutation(len(val) // 32 * 32) for s in (11, 97, 909)])
    with torch.no_grad():
        VT = {k: torch.cat([t([c[1] for c in vcap[i:i + 64]]) for i in range(0, len(val), 64)]).cpu().numpy()
              for k, t in (('reward', txt_r), ('held', txt_h))}

    def embed_eval(ev, X, L):
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                x = X[i:i + 64]; l = L[i:i + 64]
                m = torch.arange(196, device='cuda')[None, :] < l[:, None]
                outs.append(motion_embed(ev, ((x - emu) / esd) * m[..., None], l).cpu().numpy())
        return np.concatenate(outs)

    with torch.no_grad():
        VR = torch.tensor(vreal, device='cuda')
        VREF = {k: embed_eval(ev, VR, vlen) for k, ev in (('reward', reward_ev), ('held', held_ev))}

    def sample(texts, L, grad_from=None):
        """Full sampler; if grad_from is set, stop at t=grad_from and return x0 prediction with grad."""
        B = len(texts); mask = torch.arange(196, device='cuda')[None, :] < L[:, None]
        y = {'text': texts, 'lengths': L, 'mask': mask[:, None, None, :],
             'scale': torch.full((B,), 2.5, device='cuda')}
        with torch.no_grad():
            y['text_embed'] = model.encode_text(texts)
        x = torch.randn(B, 263, 1, 196, device='cuda')
        stop = 0 if grad_from is None else grad_from
        with torch.no_grad():
            for t in range(diffusion.num_timesteps - 1, stop - 1 if grad_from is None else stop, -1):
                tt = torch.full((B,), t, device='cuda', dtype=torch.long)
                x = diffusion.p_sample(guided, x, tt, clip_denoised=False, model_kwargs={'y': y})['sample']
        if grad_from is None:
            out = x
        else:
            tt = torch.full((B,), grad_from, device='cuda', dtype=torch.long)
            out = diffusion.p_mean_variance(guided, x, tt, clip_denoised=False, model_kwargs={'y': y})['pred_xstart']
        phys = out[:, :, 0, :].permute(0, 2, 1) * sd + mu
        ref = None
        if grad_from is not None and ref_guided is not None:
            with torch.no_grad():
                r_ = diffusion.p_mean_variance(ref_guided, x, tt, clip_denoised=False, model_kwargs={'y': y})['pred_xstart']
                ref = (r_[:, :, 0, :].permute(0, 2, 1) * sd + mu) * mask[..., None]
        return phys * mask[..., None], mask, ref

    def evaluate(step):
        model.eval(); X = []
        g = torch.Generator(device='cuda').manual_seed(20261014)
        state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(20261014)
        for i in range(0, len(val), 32):
            x, _, _ = sample([c[0] for c in vcap[i:i + 32]], vlen[i:i + 32]); X.append(x)
        torch.cuda.set_rng_state(state)
        X = torch.cat(X); C = canonical_t(X)
        m = (torch.arange(196, device='cuda')[None, :] < vlen[:, None])
        dev = (X - C)[..., ROT][m]
        row = {'step': step, 'noncanonical_rms': float(dev.pow(2).mean().sqrt())}
        for k, ev in (('reward', reward_ev), ('held', held_ev)):
            for enc, Y in (('raw', X), ('canonical', C)):
                E = embed_eval(ev, Y, vlen)
                row[f'{k}_{enc}_R1'] = r1(E, VT[k], orders)
                row[f'{k}_{enc}_FID'] = fid(E, VREF[k])
        model.train(); return row

    log = {'train': [], 'eval': []}
    log['eval'].append(evaluate(0)); print('EVAL', json.dumps(log['eval'][-1]), flush=True)
    model.train()
    reward_ev.motion_encoder.gru.train(); assert reward_ev.motion_encoder.gru.dropout == 0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = rng.choice(len(train), a.batch, replace=False)
        caps = [train[i]['caps'][rng.integers(len(train[i]['caps']))] for i in idx]
        L = torch.tensor([train[i]['len'] for i in idx], device='cuda')
        mask = torch.arange(196, device='cuda')[None, :] < L[:, None]
        # diffusion regularizer on real motion
        X = torch.zeros(a.batch, 196, 263, device='cuda')
        for j, i in enumerate(idx):
            X[j, :train[i]['len']] = torch.tensor(train[i]['motion'], device='cuda')
        x0 = (((X - mu) / sd) * mask[..., None]).permute(0, 2, 1)[:, :, None, :]
        t = torch.randint(diffusion.num_timesteps, (a.batch,), device='cuda')
        xt = diffusion.q_sample(x0, t, noise=torch.randn_like(x0))
        y = {'text': [c[0] for c in caps], 'lengths': L, 'mask': mask[:, None, None, :]}
        pred = model(xt, diffusion._scale_timesteps(t), y=y)
        if a.reg == 'canon':  # canonicalize the prediction in physical units, then compare in normalized units
            pp = pred[:, :, 0, :].permute(0, 2, 1) * sd + mu
            pred = ((canonical_t(pp) - mu) / sd).permute(0, 2, 1)[:, :, None, :]
        mse = (((pred - x0) ** 2) * mask[:, None, None, :]).sum() / (mask.sum() * 263)
        loss = mse; rl = torch.zeros((), device='cuda'); kl = torch.zeros((), device='cuda')
        if a.arm != 'mse':
            model.eval()
            tk = int(rng.integers(0, a.tmax + 1))
            phys, m2, ref = sample([c[0] for c in caps], L, grad_from=tk)
            inp = canonical_t(phys) if a.arm == 'canonical' else phys
            E = motion_embed(reward_ev, ((inp - emu) / esd) * m2[..., None], L)
            T = txt_r([c[1] for c in caps])
            logits = -torch.cdist(E, T) / a.tau
            target = torch.arange(a.batch, device='cuda')
            rl = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            loss = loss + a.weight * rl
            if ref is not None:  # canonical-invariant: the 6D fiber stays free
                diff = (canonical_t(phys) - canonical_t(ref)) / sd
                kl = (diff ** 2 * m2[..., None]).sum() / (m2.sum() * 263)
                loss = loss + a.kl * kl
            model.train()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 25 == 0:
            r = {'step': step, 'mse': float(mse), 'reward_ce': float(rl), 'kl': float(kl), 'grad_norm': float(gn),
                 'sec_per_step': (time.time() - t0) / step}
            log['train'].append(r); print('TRAIN', json.dumps(r), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            log['eval'].append(evaluate(step)); print('EVAL', json.dumps(log['eval'][-1]), flush=True)
            write(out / 'log.json', log)
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith('clip_model.')}, out / 'final.pt')
    write(out / 'log.json', log); write(out / 'complete.json', {'done': True})


if __name__ == '__main__':
    main()
