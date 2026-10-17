"""Reward fine-tuning of MLD against a learned evaluator (second generator for the MDM study).

Same question and reward as reward_finetune_mdm.py, on a latent diffusion model: MLD denoises a
256-d latent and a frozen VAE decodes it to HumanML3D features.  Three arms share everything but
the reward input (mse: no reward; raw; canonical).  Unlike the MDM study, both regularizers are
MLD's native latent-space terms, which do not single out the 6D fiber: the usual epsilon loss on
VAE latents of real training motion, and a penalty keeping the guided epsilon prediction at the
reward step close to the frozen pretrained denoiser's.

Each reward step samples latents for training captions with DDIM-50 and CFG 7.5 (native settings)
without gradients down to a random late step, predicts x0 with gradients, decodes it, and applies
the Euclidean InfoNCE reward of a frozen T2M checkpoint.  --test scores a final (or the
pretrained) model once on the 2,048 test clips of eval_reward_finetune_test.py.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from motion_release_audit_common import evaluator, seed as set_seed, write  # noqa: E402
from reward_finetune_mdm import (OUT, ROT, SECOND, TextEncoder, canonical_t, fid, load_split,  # noqa: E402
                                 motion_embed, r1)
from run_motion_reward_release import BASE, NativeRelease  # noqa: E402

MDM_TEST = OUT / 'test'
TEST = OUT / 'test_mld'
STEPS, CFG = 50, 7.5


def load_readers(reward_ckpt):
    a, emu, esd = evaluator()
    b, _, _ = evaluator()
    ck = torch.load(SECOND, map_location='cpu', weights_only=False)
    for m, k in [(b.movement_encoder, 'movement_encoder'), (b.motion_encoder, 'motion_encoder'), (b.text_encoder, 'text_encoder')]:
        m.load_state_dict(ck[k], strict=True); m.eval()
    emu = torch.tensor(emu, device='cuda', dtype=torch.float32); esd = torch.tensor(esd, device='cuda', dtype=torch.float32)
    return (a, b, emu, esd) if reward_ckpt == 'mdm' else (b, a, emu, esd)  # (reward, held, ...)


class MLD:
    def __init__(self):
        g = NativeRelease(BASE)
        self.text, self.vae, self.den, self.sch = g.text, g.vae, g.denoiser, g.scheduler
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.ref = copy.deepcopy(self.den).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ac = torch.tensor(self.sch.alphas_cumprod, device='cuda', dtype=torch.float32) \
            if not torch.is_tensor(self.sch.alphas_cumprod) else self.sch.alphas_cumprod.to('cuda').float()
        self.mu = torch.tensor(np.load('external/HumanML3D/HumanML3D/Mean.npy'), device='cuda', dtype=torch.float32)
        self.sd = torch.tensor(np.load('external/HumanML3D/HumanML3D/Std.npy'), device='cuda', dtype=torch.float32)

    @torch.no_grad()
    def embed_text(self, caps):
        n = len(caps)
        return self.text.encode(list(caps) + [''] * n, show_progress_bar=False, convert_to_tensor=True,
                                batch_size=2 * n).clone().unsqueeze(1)

    def guided(self, den, z, t, E):
        eps = den(sample=torch.cat([z, z]), timestep=t, timestep_cond=None, encoder_hidden_states=E,
                  controlnet_residuals=None)[0]
        c, u = eps.chunk(2)
        return u + CFG * (c - u)

    def decode(self, z, L):
        mask = torch.arange(196, device='cuda')[None, :] < L[:, None]
        x = self.vae.decode(z, mask)
        x = torch.cat([x, x.new_zeros(x.shape[0], 196 - x.shape[1], x.shape[2])], 1) if x.shape[1] < 196 else x
        return (x * self.sd + self.mu) * mask[..., None], mask

    def sample(self, caps, L, k=None):
        """Full DDIM sample; with k, stop k steps before the end and return the decoded x0 prediction with
        gradients, plus the proximity penalty against the frozen denoiser at that step."""
        E = self.embed_text(caps)
        self.sch.set_timesteps(STEPS)
        ts = self.sch.timesteps.to('cuda')
        z = torch.randn(len(caps), 1, 256, device='cuda') * self.sch.init_noise_sigma
        stop = len(ts) if k is None else len(ts) - 1 - k
        with torch.no_grad():
            for t in ts[:stop]:
                z = self.sch.step(self.guided(self.den, z, t, E), t, z).prev_sample
        if k is None:
            return self.decode(z, L)[0], None
        t = ts[stop]
        eps = self.guided(self.den, z, t, E)
        with torch.no_grad():
            eps_ref = self.guided(self.ref, z, t, E)
        a = self.ac[t.long()]
        x0 = (z - (1 - a).sqrt() * eps) / a.sqrt()
        return self.decode(x0, L)[0], ((eps - eps_ref) ** 2).mean()

    def diffusion_loss(self, X, L, caps, rng):
        mask = torch.arange(196, device='cuda')[None, :] < L[:, None]
        with torch.no_grad():
            z, _ = self.vae.encode(((X - self.mu) / self.sd)[:, :int(L.max())], mask[:, :int(L.max())])
            caps = ['' if rng.random() < 0.1 else c for c in caps]
            E = self.text.encode(caps, show_progress_bar=False, convert_to_tensor=True).clone().unsqueeze(1)
        assert z.shape == (len(caps), 1, 256), z.shape
        noise = torch.randn_like(z)
        t = torch.randint(0, self.sch.config.num_train_timesteps, (len(caps),), device='cuda').long()
        eps = self.den(sample=self.sch.add_noise(z, noise, t), timestep=t, timestep_cond=None,
                       encoder_hidden_states=E, controlnet_residuals=None)[0]
        return F.mse_loss(eps, noise)


def embed_all(ev, X, L, emu, esd):
    o = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            x = X[i:i + 64]; l = L[i:i + 64]
            m = torch.arange(196, device='cuda')[None, :] < l[:, None]
            o.append(motion_embed(ev, ((x - emu) / esd) * m[..., None], l).cpu().numpy())
    return np.concatenate(o)


def generate(mld, caps, L, seed):
    mld.den.eval(); torch.cuda.manual_seed(seed); X = []
    with torch.no_grad():
        for i in range(0, len(caps), 64):
            X.append(mld.sample(caps[i:i + 64], L[i:i + 64])[0])
    return torch.cat(X)


def test(a):
    """Score one model once on the MDM study's 2,048 test clips, reusing its reference embeddings."""
    TEST.mkdir(parents=True, exist_ok=True)
    name = 'pretrained' if a.pretrained else a.name
    out = TEST / f'{name}.npz'
    assert not out.exists(), out
    items = load_split('test', np.random.default_rng(2027), max_items=2048)
    rng = np.random.default_rng(2027)
    caps = [it['caps'][rng.integers(len(it['caps']))] for it in items]
    proto = json.loads((MDM_TEST / 'protocol.json').read_text())
    assert proto['ids'] == [it['id'] for it in items] and proto['captions'] == [c[0] for c in caps]
    if not (TEST / 'reference.npz').exists():
        (TEST / 'reference.npz').write_bytes((MDM_TEST / 'reference.npz').read_bytes())
    L = torch.tensor([it['len'] for it in items], device='cuda')
    mld = MLD()
    if not a.pretrained:
        mld.den.load_state_dict(torch.load(OUT / a.name / 'final.pt', map_location='cpu'), strict=True)
    X = generate(mld, [c[0] for c in caps], L, 20261015); C = canonical_t(X)
    m = torch.arange(196, device='cuda')[None, :] < L[:, None]
    res = {'rms': np.array(float((X - C)[..., ROT][m].pow(2).mean().sqrt()))}
    mdm_ck, gpt_ck, emu, esd = load_readers('mdm')
    for k, ev in (('reward', mdm_ck), ('held', gpt_ck)):  # keys denote checkpoints, as in test/
        res[f'{k}_raw'] = embed_all(ev, X, L, emu, esd); res[f'{k}_canonical'] = embed_all(ev, C, L, emu, esd)
    np.savez_compressed(out, **res)
    print('SCORED', name, 'rms %.4f' % float(res['rms']), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['mse', 'raw', 'canonical'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--tmax', type=int, default=10)
    ap.add_argument('--kl', type=float, default=0.0, help='weight of the latent proximity penalty')
    ap.add_argument('--reward-ckpt', choices=['mdm', 'gpt'], default='mdm')
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--dev-n', type=int, default=512)
    ap.add_argument('--name', default=None)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--pretrained', action='store_true')
    a = ap.parse_args()
    if a.test:
        return test(a)
    name = a.name or f'mld_{a.arm}_kl{a.kl}_w{a.weight}_lr{a.lr}_t{a.tmax}_r{a.reward_ckpt}_s{a.seed}'
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists(), f'{out} already complete'
    write(out / 'config.json', vars(a) | {'generator': 'MLD'})
    set_seed(a.seed); rng = np.random.default_rng(a.seed)
    train = load_split('train', np.random.default_rng(0))
    val = load_split('val', np.random.default_rng(1), max_items=a.dev_n)
    mld = MLD()
    params = [p for p in mld.den.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    reward_ev, held_ev, emu, esd = load_readers(a.reward_ckpt)
    txt_r, txt_h = TextEncoder(reward_ev), TextEncoder(held_ev)

    vrng = np.random.default_rng(7)
    vcap = [it['caps'][vrng.integers(len(it['caps']))] for it in val]
    vlen = torch.tensor([it['len'] for it in val], device='cuda')
    vreal = torch.zeros(len(val), 196, 263, device='cuda')
    for i, it in enumerate(val):
        vreal[i, :it['len']] = torch.tensor(it['motion'], device='cuda')
    orders = np.array([np.random.default_rng(s).permutation(len(val) // 32 * 32) for s in (11, 97, 909)])
    with torch.no_grad():
        VT = {k: torch.cat([t([c[1] for c in vcap[i:i + 64]]) for i in range(0, len(val), 64)]).cpu().numpy()
              for k, t in (('reward', txt_r), ('held', txt_h))}
    VREF = {k: embed_all(ev, vreal, vlen, emu, esd) for k, ev in (('reward', reward_ev), ('held', held_ev))}

    def evaluate(step):
        state = torch.cuda.get_rng_state()
        X = generate(mld, [c[0] for c in vcap], vlen, 20261014)
        torch.cuda.set_rng_state(state); mld.den.train()
        C = canonical_t(X); m = torch.arange(196, device='cuda')[None, :] < vlen[:, None]
        row = {'step': step, 'noncanonical_rms': float((X - C)[..., ROT][m].pow(2).mean().sqrt())}
        for k, ev in (('reward', reward_ev), ('held', held_ev)):
            for enc, Y in (('raw', X), ('canonical', C)):
                E = embed_all(ev, Y, vlen, emu, esd)
                row[f'{k}_{enc}_R1'] = r1(E, VT[k], orders); row[f'{k}_{enc}_FID'] = fid(E, VREF[k])
        return row

    log = {'train': [], 'eval': [evaluate(0)]}
    print('EVAL', json.dumps(log['eval'][-1]), flush=True)
    reward_ev.motion_encoder.gru.train(); assert reward_ev.motion_encoder.gru.dropout == 0
    mld.den.train(); t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = rng.choice(len(train), a.batch, replace=False)
        caps = [train[i]['caps'][rng.integers(len(train[i]['caps']))] for i in idx]
        L = torch.tensor([train[i]['len'] for i in idx], device='cuda')
        X = torch.zeros(a.batch, 196, 263, device='cuda')
        for j, i in enumerate(idx):
            X[j, :train[i]['len']] = torch.tensor(train[i]['motion'], device='cuda')
        mse = mld.diffusion_loss(X, L, [c[0] for c in caps], rng)
        loss = mse; rl = torch.zeros((), device='cuda'); kl = torch.zeros((), device='cuda')
        if a.arm != 'mse':
            mld.den.eval()  # no dropout in the sampled branch, as in the MDM study
            phys, kl = mld.sample([c[0] for c in caps], L, k=int(rng.integers(0, a.tmax + 1)))
            mld.den.train()
            inp = canonical_t(phys) if a.arm == 'canonical' else phys
            m2 = torch.arange(196, device='cuda')[None, :] < L[:, None]
            E = motion_embed(reward_ev, ((inp - emu) / esd) * m2[..., None], L)
            T = txt_r([c[1] for c in caps])
            logits = -torch.cdist(E, T) / a.tau
            tgt = torch.arange(a.batch, device='cuda')
            rl = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            loss = loss + a.weight * rl + a.kl * kl
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 25 == 0:
            r = {'step': step, 'mse': float(mse), 'reward_ce': float(rl), 'kl': float(kl), 'grad_norm': float(gn),
                 'sec_per_step': (time.time() - t0) / step}
            log['train'].append(r); print('TRAIN', json.dumps(r), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            log['eval'].append(evaluate(step)); print('EVAL', json.dumps(log['eval'][-1]), flush=True)
            write(out / 'log.json', log)
    torch.save({k: v.cpu() for k, v in mld.den.state_dict().items()}, out / 'final.pt')
    write(out / 'log.json', log); write(out / 'complete.json', {'done': True})


if __name__ == '__main__':
    main()
