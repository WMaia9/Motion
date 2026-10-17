"""Where does an evaluator's reward gradient point?

For saved MDM outputs and real HumanML3D motion, compute the gradient of the
Euclidean InfoNCE retrieval loss (the reward used for fine-tuning) with respect
to the raw 263-dimensional features, and split its squared norm into:
  * fiber    : the 3 redundant directions per joint, tangent to (a,0), (0,b), (0,a)
  * rotation : the remaining 3 directions per joint inside the 6D block
  * other    : positions, velocities, contacts, root
An isotropic gradient would put half of the rotation-block norm on the fiber.
Read-only over saved arrays; no model is trained.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from motion_release_audit_common import evaluator  # noqa: E402
from reward_finetune_mdm import TextEncoder, motion_embed, load_split  # noqa: E402

OUT = Path('reports/cvpr_reward_finetune_20261014/gradient_fiber_share.json')
ROT = slice(67, 193)


def fiber_basis(d):
    """Orthonormal basis [...,3,6] of the fiber tangent at 6D vectors d [...,6]."""
    a, b = d[..., :3], d[..., 3:]
    z = torch.zeros_like(a)
    t = torch.stack([torch.cat([a, z], -1), torch.cat([z, b], -1), torch.cat([z, a], -1)], -2)
    q, _ = torch.linalg.qr(t.transpose(-1, -2))  # [...,6,3]
    return q.transpose(-1, -2)


def shares(wrapper, txt, X, L, tokens, emu, esd, batch=32, gen_sd=None):
    tot = {'fiber': 0.0, 'rotation': 0.0, 'other': 0.0}
    for i in range(0, len(X) // batch * batch, batch):
        x = torch.tensor(X[i:i + batch], device='cuda', requires_grad=True)
        l = torch.tensor(L[i:i + batch], device='cuda')
        m = torch.arange(x.shape[1], device='cuda')[None, :] < l[:, None]
        E = motion_embed(wrapper, ((x - emu) / esd) * m[..., None], l)
        T = txt(tokens[i:i + batch])
        logits = -torch.cdist(E, T)
        tgt = torch.arange(len(E), device='cuda')
        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt))
        g, = torch.autograd.grad(loss, x)
        g = g * m[..., None]
        d = x.detach()[..., ROT].reshape(*g.shape[:2], 21, 6)
        if gen_sd is not None:  # generator's normalized coordinates: z = (x - mu) / sd
            g = g * gen_sd
            s6 = gen_sd[ROT].reshape(21, 6)
            gr = g[..., ROT].reshape(*g.shape[:2], 21, 6)
            a_, b_ = d[..., :3], d[..., 3:]; zz = torch.zeros_like(a_)
            t = torch.stack([torch.cat([a_, zz], -1), torch.cat([zz, b_], -1), torch.cat([zz, a_], -1)], -2) / s6[:, None, :]
            q, _ = torch.linalg.qr(t.transpose(-1, -2)); basis = q.transpose(-1, -2)
        else:
            gr = g[..., ROT].reshape(*g.shape[:2], 21, 6)
            basis = fiber_basis(d)
        proj = torch.einsum('btjkc,btjc->btjk', basis, gr)
        f = float(proj.pow(2).sum())
        r = float(gr.pow(2).sum()) - f
        o = float(g.pow(2).sum()) - float(gr.pow(2).sum())
        tot['fiber'] += f; tot['rotation'] += r; tot['other'] += o
    s = sum(tot.values())
    return {k: v / s for k, v in tot.items()} | {'fiber_share_of_rotation_block': tot['fiber'] / (tot['fiber'] + tot['rotation'])}


def main():
    torch.backends.cudnn.enabled = False  # GRU input gradients in eval mode
    wrapper, emu, esd = evaluator()
    emu = torch.tensor(emu, device='cuda', dtype=torch.float32); esd = torch.tensor(esd, device='cuda', dtype=torch.float32)
    txt = TextEncoder(wrapper)
    val = load_split('val', np.random.default_rng(1), max_items=512)
    rng = np.random.default_rng(7)
    caps = [it['caps'][rng.integers(len(it['caps']))] for it in val]
    L = np.array([it['len'] for it in val])
    real = np.zeros((len(val), 196, 263), np.float32)
    for i, it in enumerate(val):
        real[i, :it['len']] = it['motion']
    tokens = [c[1] for c in caps]
    out = {'real_val': shares(wrapper, txt, real, L, tokens, emu, esd)}
    rep = Path('reports/cvpr_motion_prevalence_20261012/mdm50/rep_00')
    gen = np.load(rep / 'generated_raw.npy')[:512].astype(np.float32)
    gl = np.load(rep / 'generated_lengths.npy')[:512].astype(np.int64)
    cond = np.load(rep / 'conditioning.npz', allow_pickle=True)
    gtok = [' '.join(w for w in str(t).split('_') if not w.split('/')[0] in ('sos', 'eos', 'unk')) for t in cond['tokens'][:512]]
    if gtok is not None:
        out['mdm_released'] = shares(wrapper, txt, gen, gl, gtok, emu, esd)
    out['isotropic_reference_fiber_share_of_rotation_block'] = 0.5
    gsd = torch.tensor(np.load('external/HumanML3D/HumanML3D/Std.npy'), device='cuda', dtype=torch.float32)
    out['real_val_generator_space'] = shares(wrapper, txt, real, L, tokens, emu, esd, gen_sd=gsd)
    out['mdm_released_generator_space'] = shares(wrapper, txt, gen, gl, gtok, emu, esd, gen_sd=gsd)
    nrot = 126 / 263
    out['isotropic_reference_rotation_block_share'] = nrot
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
