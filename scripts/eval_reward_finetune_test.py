"""Score the final reward-fine-tuned MDM checkpoints once on the HumanML3D test split.

Fixed before any test inference: 2,048 test motions (seed 2027) with one caption each, the
real motions as the FID reference, 50-step CFG-2.5 sampling with generation seed 20261015,
and the pretrained MDM as baseline.  For each model and each of the two T2M checkpoints
(reward evaluator = MDM-linked; held-out = T2M-GPT) we save motion embeddings of the raw and
canonicalized outputs (keys 'reward' = MDM-linked checkpoint, 'held' = T2M-GPT checkpoint, whichever
gave the reward), so that paired bootstrap intervals over clips can be computed, plus
the non-canonical RMS of the outputs.  Nothing is selected on test scores.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from motion_release_audit_common import evaluator, write  # noqa: E402
from run_motion_release_audit import mdm_model  # noqa: E402
from reward_finetune_mdm import (SECOND, TextEncoder, canonical_t, load_split,  # noqa: E402
                                 motion_embed, ROT)

R = Path('reports/cvpr_reward_finetune_20261014')
OUT = R / 'test'
N = 2048


def runs():
    out = {'pretrained': None}
    for d in sorted(R.glob('*_regcanon_*')):
        cfg = json.loads((d / 'config.json').read_text())
        if not (d / 'complete.json').exists():
            continue
        if cfg['arm'] in ('raw', 'canonical') and cfg.get('kl', 0.0) == 100.0 and cfg['lr'] == 1e-5:
            prefix = 'g' if cfg.get('reward_ckpt', 'mdm') == 'gpt' else ''  # reward from the T2M-GPT checkpoint
            out[f"{prefix}{cfg['arm']}_s{cfg['seed']}"] = d
        if cfg['arm'] == 'mse' and cfg.get('reg') == 'canon':
            out[f"mse_s{cfg['seed']}"] = d
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = load_split('test', np.random.default_rng(2027), max_items=N)
    rng = np.random.default_rng(2027)
    caps = [it['caps'][rng.integers(len(it['caps']))] for it in test]
    L = torch.tensor([it['len'] for it in test], device='cuda')
    real = np.zeros((len(test), 196, 263), np.float32)
    for i, it in enumerate(test):
        real[i, :it['len']] = it['motion']
    write(OUT / 'protocol.json', {'clips': len(test), 'ids': [it['id'] for it in test], 'captions': [c[0] for c in caps],
                                  'generation_seed': 20261015, 'cfg': 2.5, 'steps': 50})

    readers = {}
    ev, emu, esd = evaluator(); readers['reward'] = ev
    ev2, _, _ = evaluator()
    ck = torch.load(SECOND, map_location='cpu', weights_only=False)
    for m, k in [(ev2.movement_encoder, 'movement_encoder'), (ev2.motion_encoder, 'motion_encoder'), (ev2.text_encoder, 'text_encoder')]:
        m.load_state_dict(ck[k], strict=True); m.eval()
    readers['held'] = ev2
    emu = torch.tensor(emu, device='cuda', dtype=torch.float32); esd = torch.tensor(esd, device='cuda', dtype=torch.float32)

    @torch.no_grad()
    def embed(ev, X):
        o = []
        for i in range(0, len(X), 64):
            x = X[i:i + 64]; l = L[i:i + 64]
            m = torch.arange(196, device='cuda')[None, :] < l[:, None]
            o.append(motion_embed(ev, ((x - emu) / esd) * m[..., None], l).cpu().numpy())
        return np.concatenate(o)

    with torch.no_grad():
        RX = torch.tensor(real, device='cuda')
        base = {k: {'text': torch.cat([TextEncoder(ev)([c[1] for c in caps[i:i + 64]]) for i in range(0, len(caps), 64)]).cpu().numpy(),
                    'real': embed(ev, RX)} for k, ev in readers.items()}
    np.savez_compressed(OUT / 'reference.npz', **{f'{k}_{n}': v for k, d in base.items() for n, v in d.items()})

    model, guided, diffusion = mdm_model()
    pre = {k: v.clone() for k, v in model.state_dict().items()}
    mu = torch.tensor(np.load('external/HumanML3D/HumanML3D/Mean.npy'), device='cuda', dtype=torch.float32)
    sd = torch.tensor(np.load('external/HumanML3D/HumanML3D/Std.npy'), device='cuda', dtype=torch.float32)
    for name, d in runs().items():
        out = OUT / f'{name}.npz'
        if out.exists():
            continue
        model.load_state_dict(pre)
        if d is not None:
            state = torch.load(d / 'final.pt', map_location='cpu')
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected and all(k.startswith('clip_model.') for k in missing)
        model.eval(); torch.cuda.manual_seed(20261015)
        X = []
        with torch.no_grad():
            for i in range(0, len(test), 32):
                l = L[i:i + 32]; mask = torch.arange(196, device='cuda')[None, :] < l[:, None]
                y = {'text': [c[0] for c in caps[i:i + 32]], 'lengths': l, 'mask': mask[:, None, None, :],
                     'scale': torch.full((len(l),), 2.5, device='cuda')}
                y['text_embed'] = model.encode_text(y['text'])
                x = diffusion.p_sample_loop(guided, (len(l), 263, 1, 196), clip_denoised=False, model_kwargs={'y': y},
                                            skip_timesteps=0, init_image=None, progress=False, dump_steps=None,
                                            noise=None, const_noise=False)
                X.append((x[:, :, 0, :].permute(0, 2, 1) * sd + mu) * mask[..., None])
        X = torch.cat(X); C = canonical_t(X)
        m = torch.arange(196, device='cuda')[None, :] < L[:, None]
        rms = float((X - C)[..., ROT][m].pow(2).mean().sqrt())
        res = {'rms': np.array(rms)}
        for k, ev in readers.items():
            res[f'{k}_raw'] = embed(ev, X); res[f'{k}_canonical'] = embed(ev, C)
        np.savez_compressed(out, **res)
        print('SCORED', name, 'rms %.4f' % rms, flush=True)
    write(OUT / 'complete.json', {'models': list(runs())})


if __name__ == '__main__':
    main()
