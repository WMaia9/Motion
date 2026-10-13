"""New DEV experiments requested by review; never writes to prior reports.

Three matched fine-tuning seeds from one frozen shared DDPM initialization.
TRAIN-only targets; full DEV sampling without reference poses. No new TEST use.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import run_physical_semantic_ddpm as previous
from run_evaluation_heldout import sample_without_reference_pose
from run_signspark_gauge_attack import SignCollator, Batch, PAD_TOKEN
from analyze_signspark_geometry_dev import statistics, b4
from verify_signspark_studies import sha

ROOT = Path('reports/cvpr_reviewer_followup_20261011')
SEEDS = [42, 123, 777]
ARMS = ['geometry', 'raw_ce', 'canonical_ce']
EPOCHS = 5
SEMANTIC_WEIGHT = .02


def read(p): return json.loads(Path(p).read_text())
def write(name, data):
    p = ROOT/name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def initialize():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'manifest_sha256.json').exists(), 'frozen report'
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    audit = read(previous.ROOT/'data_audit.json')
    with np.load(previous.ROOT/'normalization.npz') as z:
        mean, std = z['mean'].copy(), z['std'].copy()
    return audit, mean, std


def register():
    initialize()
    paths = [previous.ROOT/'shared_best.pt', previous.ROOT/'normalization.npz',
             previous.ROOT/'data_audit.json', previous.BASE/'checkpoints/PHOENIX-2014T/best.ckpt',
             previous.PREV/'data/lmdb/train/PHOENIX-2014T_reopt_train.lmdb/data.mdb']
    protocol = {'seeds': SEEDS, 'arms': ARMS, 'epochs': EPOCHS, 'semantic_weight': SEMANTIC_WEIGHT,
        'learning_rate': .0001, 'batch_size': 64, 'semantic_batch_limit': 16,
        'semantic_timesteps': 't <= 800, caption dropout excluded',
        'generator_selection': 'last epoch; no BLEU selection',
        'sampling': 'full PHOENIX DEV, 50 DDIM steps, CFG3, seed102, reference durations only',
        'reference_pose_read_by_sampler': False, 'new_TEST_use': False,
        'init': 'one shared frozen 20-epoch checkpoint; three fine-tuning seeds, not three pretraining seeds',
        'reader_fine_tuning': {'epochs': 5, 'lr': .00001, 'batch_size': 64, 'seed': 42,
            'arms': ['raw', 'canonical'], 'initialization': 'same public pretrained PHOENIX reader',
            'selection': 'last epoch', 'objective': 'token CE only; CTC disabled'},
        'inputs': {str(p): sha(p) for p in paths},
        'code_sha256': sha(Path(__file__)),
        'interpretation': 'exploratory response to review, existing DEV already inspected; cannot validate previously published performance'}
    path = ROOT/'training_protocol.json'
    if path.exists(): assert read(path) == protocol, 'protocol changed'
    else: write('training_protocol.json', protocol)
    print('REGISTERED', flush=True)


def batches(train):
    order = np.random.permutation(len(train)); ordered = []
    for pos in range(0, len(order), 256):
        ordered.extend(sorted(order[pos:pos+256], key=lambda i: len(train[i].sgn)))
    groups = [ordered[pos:pos+64] for pos in range(0, len(ordered), 64)]
    random.shuffle(groups)
    return groups


def fit_generator(model, train, dev, stoi, mean, std, alpha, teacher, arm, seed):
    previous.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    mean_gpu = torch.tensor(mean, device='cuda'); std_gpu = torch.tensor(std, device='cuda')
    collate = SignCollator(teacher.txt_vocab, None, 240); pad = teacher.txt_vocab.stoi[PAD_TOKEN]
    history = []; gradient_audit = None; stochastic = []
    for epoch in range(EPOCHS):
        started = time.time(); model.train(); mse_values = []; semantic_values = []; count = 0
        for ids in batches(train):
            x, text, lengths = previous.pack(train, ids, stoi, mean, std)
            dropped = torch.rand(len(ids), device='cuda') < .1
            text = text.clone(); text[dropped] = 0; text[dropped, 0] = 2
            t = torch.randint(0, 1000, (len(ids),), device='cuda'); noise = torch.randn_like(x)
            a = alpha[t][:, None, None]
            pred = model(a.sqrt()*x+(1-a).sqrt()*noise, t, text, lengths)
            mask = torch.arange(x.shape[1], device='cuda')[None] < lengths[:, None]
            mse = ((pred-x).square()*mask[:, :, None]).sum()/(mask.sum()*240)
            semantic = torch.zeros((), device='cuda')
            if epoch == 0:
                stochastic.append({'ids': list(map(int, ids)), 't': t.cpu().tolist(),
                    'drop': dropped.cpu().tolist(), 'noise_sum': float(noise.double().sum()),
                    'noise_squared_sum': float(noise.double().square().sum())})
            chosen = ((t <= 800) & ~dropped).nonzero().flatten()[:16]
            if arm != 'geometry' and len(chosen):
                examples = [copy.copy(train[ids[j]]) for j in chosen.tolist()]
                batch = Batch(collate(examples), txt_pad_index=pad, sgn_dim=240, use_cuda=True)
                maximum = batch.sgn.shape[1]
                poses = (pred[chosen, :maximum]*std_gpu+mean_gpu)*mask[chosen, :maximum, None]
                consumed = previous.physical(poses) if arm == 'canonical_ce' else poses
                output, _ = teacher.forward(consumed, batch.sgn_mask, batch.sgn_lengths, batch.txt_input, batch.txt_mask)
                logits = output[0]
                losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.txt.reshape(-1),
                                        ignore_index=pad, reduction='none').reshape(batch.txt.shape)
                semantic = (losses.sum(1)/(batch.txt != pad).sum(1)).mean()
                if gradient_audit is None:
                    g = torch.autograd.grad(semantic, poses, retain_graph=True)[0].reshape(-1, 40, 6)
                    d = poses.reshape(-1, 40, 6)
                    vertical = torch.stack([(g[...,:3]*d[...,:3]).sum(-1),
                        (g[...,3:]*d[...,3:]).sum(-1), (g[...,3:]*d[...,:3]).sum(-1)], -1)
                    gradient_audit = {'gradient_norm': float(g.norm()),
                        'maximum_vertical_dot': float(vertical.abs().max()), 'reader_parameters_frozen': True}
                count += len(chosen)
            loss = mse+SEMANTIC_WEIGHT*semantic
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            mse_values.append(float(mse.detach())); semantic_values.append(float(semantic.detach()))
        val = previous.validation(model, dev, stoi, mean, std, alpha)
        record = {'epoch': epoch+1, 'train_mse': float(np.mean(mse_values)),
            'semantic_ce': float(np.mean(semantic_values)), 'semantic_examples': count,
            'dev_fixed_noise_mse': val, 'seconds': time.time()-started}
        history.append(record); write(f'training/seed{seed}/{arm}_history.json', history)
        print('GENERATOR', seed, arm, json.dumps(record), flush=True)
    folder = ROOT/f'training/seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'seed': seed, 'epoch': EPOCHS,
        'selection': 'last epoch', 'arm': arm, 'semantic_weight': SEMANTIC_WEIGHT}, folder/(arm+'.pt'))
    write(f'training/seed{seed}/{arm}_gradient.json', gradient_audit)
    write(f'training/seed{seed}/{arm}_stochastic.json', stochastic)


def score(teacher, xs, references, ids):
    hyps = previous.back_translate(teacher, [torch.from_numpy(x) for x in xs], batch_size=32)
    result = {'references': references, 'clip_ids': ids, 'hypotheses': hyps,
              'scores': previous.score_translations(references, hyps, 'phoenix_2014_trans', 'word')}
    assert abs(b4(statistics(references, hyps).sum(0))-result['scores']['bleu']['bleu4']) < 1e-10
    return result


def generators():
    register(); audit, mean, std = initialize()
    train, dev = previous.dataset('train'), previous.dataset('dev')
    assert [e.sequence for e in train] == audit['train_ids']
    assert [e.sequence for e in dev] == audit['dev_ids']
    stoi = {w: i for i,w in enumerate(audit['vocab'])}
    shared = torch.load(previous.ROOT/'shared_best.pt', map_location='cpu', weights_only=False)['state_dict']
    teacher = previous.make_back_translation_model(previous.BASE/'checkpoints/PHOENIX-2014T')
    teacher.eval(); teacher.do_recognition = False
    for p in teacher.parameters(): p.requires_grad_(False)
    alpha = previous.alpha_schedule()
    entries = [{'clip_id': e.sequence, 'frames': len(e.sgn), 'text': e.txt} for e in dev]
    refs = [' '.join(e.txt) for e in dev]; ids = [e.sequence for e in dev]
    for seed in SEEDS:
        stochastic_checks = []
        for arm in ARMS:
            folder = ROOT/f'training/seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
            model = previous.Denoiser(len(stoi)).cuda(); model.load_state_dict(shared, strict=True)
            if not (folder/(arm+'.pt')).exists():
                fit_generator(model, train, dev, stoi, mean, std, alpha, teacher, arm, seed)
            stochastic_checks.append(read(folder/(arm+'_stochastic.json')))
            model.load_state_dict(torch.load(folder/(arm+'.pt'), map_location='cpu', weights_only=False)['state_dict'], strict=True)
            if not (folder/(arm+'_raw_scores.json')).exists():
                xs = sample_without_reference_pose(model, entries, stoi, mean, std, alpha)
                np.savez_compressed(folder/(arm+'_poses.npz'), **{f'clip{i:04d}': x for i,x in enumerate(xs)})
                for kind, poses in [('raw', xs), ('canonical', [previous.project(x) for x in xs])]:
                    result = score(teacher, poses, refs, ids)
                    write(f'training/seed{seed}/{arm}_{kind}_scores.json', result)
                    print('GENERATOR_SCORE', seed, arm, kind, result['scores']['bleu']['bleu4'], flush=True)
            del model; torch.cuda.empty_cache()
        assert stochastic_checks[0] == stochastic_checks[1] == stochastic_checks[2], 'unmatched training RNG'
    write('training/completed.json', {'seeds': SEEDS, 'arms': ARMS, 'dev_clips': len(dev),
        'same_first_epoch_stochastic_inputs_in_each_seed': True,
        'TRAIN_clips': len(train), 'TRAIN_DEV_overlap': 0, 'new_TEST_use': False})


def readers():
    register(); initialize()
    train, dev = previous.dataset('train'), previous.dataset('dev')
    xs = [e.sgn.numpy() for e in dev]; refs = [' '.join(e.txt) for e in dev]; ids = [e.sequence for e in dev]
    for arm in ['raw', 'canonical']:
        folder = ROOT/'readers'/arm; folder.mkdir(parents=True, exist_ok=True)
        model = previous.make_back_translation_model(previous.BASE/'checkpoints/PHOENIX-2014T')
        model.do_recognition = False
        collate = SignCollator(model.txt_vocab, None, 240); pad = model.txt_vocab.stoi[PAD_TOKEN]
        previous.seed(42)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00001, weight_decay=.0001)
        history = []
        for epoch in range(5):
            started = time.time(); model.train(); losses = []
            for batch_number, indices in enumerate(batches(train)):
                batch = Batch(collate([train[i] for i in indices]), txt_pad_index=pad, sgn_dim=240, use_cuda=True)
                poses = previous.physical(batch.sgn) if arm == 'canonical' else batch.sgn
                output, _ = model.forward(poses, batch.sgn_mask, batch.sgn_lengths, batch.txt_input, batch.txt_mask)
                logits = output[0]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.txt.reshape(-1), ignore_index=pad)
                assert torch.isfinite(loss)
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
                losses.append(float(loss.detach()))
            history.append({'epoch': epoch+1, 'token_ce': float(np.mean(losses)), 'seconds': time.time()-started})
            write(f'readers/{arm}/history.json', history)
            print('READER', arm, json.dumps(history[-1]), flush=True)
        model.eval()
        torch.save({'model_state': model.state_dict(), 'epoch': 5, 'selection': 'last epoch'}, folder/'fine_tuned.ckpt')
        inputs = [previous.project(x) for x in xs] if arm == 'canonical' else xs
        result = score(model, inputs, refs, ids); write(f'readers/{arm}/dev_scores.json', result)
        print('READER_SCORE', arm, result['scores']['bleu']['bleu4'], flush=True)
        # A fixed random redundant-coordinate perturbation; canonical reader consumes its projection.
        rng = np.random.default_rng(2027); attacked = []
        for x in xs:
            d = x.reshape(-1, 40, 6).copy(); c = rng.uniform(-.35, .35, (*d.shape[:-1], 3)).astype(np.float32)
            c[:,10:25] = 0; a = d[...,:3].copy()
            d[...,:3] = np.exp(c[...,0:1])*a
            d[...,3:] = np.exp(c[...,1:2])*d[...,3:]+c[...,2:3]*a
            y = d.reshape(x.shape); attacked.append(previous.project(y) if arm == 'canonical' else y)
        perturbation = score(model, attacked, refs, ids)
        perturbation['changed_hypotheses'] = sum(a != b for a,b in zip(result['hypotheses'], perturbation['hypotheses']))
        write(f'readers/{arm}/random_perturbation_scores.json', perturbation)
        del model; torch.cuda.empty_cache()
    write('readers/completed.json', {'fine_tuning': True, 'from_scratch_training': False,
        'TRAIN_clips': len(train), 'DEV_clips': len(dev), 'new_TEST_use': False,
        'cannot_resolve_How2Sign_CSL_distribution_shift': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['generators', 'readers'])
    args = parser.parse_args()
    generators() if args.phase == 'generators' else readers()
