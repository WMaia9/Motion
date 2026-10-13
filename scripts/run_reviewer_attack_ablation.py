"""Fixed DEV wrong-target budget/scope/step and untargeted random controls."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
from run_signspark_geometry_dev import ROOT as BASE, _build_split, project, matrices, make_back_translation_model, back_translate, score_translations
from run_signspark_gauge_attack import SignCollator, Batch, PAD_TOKEN
from verify_signspark_studies import sha

ROOT = Path('reports/cvpr_reviewer_followup_20261011')
BUDGETS = [.01, .05, .1, .35]
STEPS = [1, 5, 20, 80]
SCOPES = ['global_per_clip', 'per_frame', 'per_frame_joint']


def transformed(original, c):
    a = original[...,:3]
    return torch.cat([torch.exp(c[...,0:1])*a,
        torch.exp(c[...,1:2])*original[...,3:]+c[...,2:3]*a], -1)


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False; torch.manual_seed(2027)
    folder = ROOT/'attack_ablation'; folder.mkdir(exist_ok=True)
    dc = {'dataset': 'PHOENIX-2014T', 'base_path': str((BASE/'data/lmdb').resolve()),
          'index_cache': False, 'max_sgn_length': 300}
    dataset = _build_split(dc, 'dev', None)
    selection = json.loads(Path('reports/cvpr_signspark_gauge_attack_20261010/PHOENIX-2014T/selection.json').read_text())
    lookup = {dataset[i].sequence: dataset[i] for i in range(len(dataset))}
    entries = [lookup[r['clip_id']] for r in selection]; assert len(entries) == 64
    for e in entries:
        x = e.sgn.numpy().copy(); x[:,60:150] = project(x)[:,60:150]; e.sgn = torch.from_numpy(x)
    xs = [e.sgn.numpy() for e in entries]; refs = [' '.join(e.txt) for e in entries]
    targets = refs[1:]+refs[:1]
    protocol = {'corpus': 'PHOENIX-2014T', 'split': 'DEV', 'clips': 64,
        'clip_ids': [e.sequence for e in entries], 'budgets': BUDGETS, 'steps': STEPS, 'scopes': SCOPES,
        'lr': .1, 'target': 'one-position rolled reference; white-box capability experiment',
        'random_baseline': 'uniform coefficients in the same bounds; no target/gradients',
        'random_seed': 2027, 'left_hand_untouched': True,
        'selection': 'same frozen pilot DEV subset; already inspected, exploratory',
        'checkpoint_sha256': sha(BASE/'checkpoints/PHOENIX-2014T/best.ckpt'), 'code_sha256': sha(Path(__file__))}
    (folder/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    model = make_back_translation_model(BASE/'checkpoints/PHOENIX-2014T'); model.eval(); model.do_recognition = False
    for p in model.parameters(): p.requires_grad_(False)
    collator = SignCollator(model.txt_vocab, None, 240); pad = model.txt_vocab.stoi[PAD_TOKEN]
    baseline = back_translate(model, [torch.from_numpy(x) for x in xs], batch_size=32)
    canonical = back_translate(model, [torch.from_numpy(project(x)) for x in xs], batch_size=32)
    results = {}

    def record(name, ys, coefficients, losses, params):
        raw = back_translate(model, [torch.from_numpy(y) for y in ys], batch_size=32)
        defended = back_translate(model, [torch.from_numpy(project(y)) for y in ys], batch_size=32)
        error = max(float(np.degrees(Rotation.from_matrix((matrices(x).transpose(0,1,3,2)@matrices(y)).reshape(-1,3,3)).magnitude()).max()) for x,y in zip(xs,ys))
        assert error < .01 and all(np.array_equal(x[:,60:150],y[:,60:150]) for x,y in zip(xs,ys))
        changes = sum(a != b for a,b in zip(canonical,defended)); assert changes == 0
        result = {'parameters': params, 'target_bleu4': score_translations(targets,raw,'phoenix_2014_trans','word')['bleu']['bleu4'],
            'actual_bleu4': score_translations(refs,raw,'phoenix_2014_trans','word')['bleu']['bleu4'],
            'changed_from_raw': sum(a != b for a,b in zip(baseline,raw)), 'canonical_changes': changes,
            'max_rotation_error_deg': error, 'raw_hypotheses': raw, 'canonical_hypotheses': defended,
            'loss_logs': losses, 'maximum_absolute_coefficient': max(float(abs(c).max()) for c in coefficients)}
        np.savez_compressed(folder/(name+'_poses.npz'), **{f'{tag}{i:04d}': y for tag, arr in [('source',xs),('attacked',ys),('coefficient',coefficients)] for i,y in enumerate(arr)})
        (folder/(name+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        results[name] = {k:v for k,v in result.items() if 'hypotheses' not in k and k != 'loss_logs'}
        print('BUDGET_SCORE', name, result['target_bleu4'], result['actual_bleu4'], flush=True)

    for scope in SCOPES:
        for budget in BUDGETS:
            checkpoints = {step: [] for step in STEPS}; coeffs = {step: [] for step in STEPS}; losses = []
            random_x, random_c = [], []
            rng = np.random.default_rng(2027)
            for start in range(0,64,32):
                chunk = [copy.copy(e) for e in entries[start:start+32]]
                for e,target in zip(chunk,targets[start:start+32]): e.txt = target.split()
                batch = Batch(collator(chunk), txt_pad_index=pad, sgn_dim=240, use_cuda=True)
                original = batch.sgn.detach().reshape(len(chunk),-1,40,6)
                shape = (len(chunk), 1 if scope == 'global_per_clip' else original.shape[1],
                         40 if scope == 'per_frame_joint' else 1, 3)
                variable = torch.zeros(shape,device='cuda',requires_grad=True)
                mask = torch.ones(1,1,40,1,device='cuda'); mask[:,:,10:25] = 0
                opt = torch.optim.Adam([variable],lr=.1); history = []
                for step in range(1,81):
                    coefficient = budget*torch.tanh(variable)*mask
                    poses = transformed(original,coefficient).reshape(batch.sgn.shape)
                    out,_ = model.forward(poses,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                    logits = out[0]; loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad)
                    assert torch.isfinite(loss); opt.zero_grad(); loss.backward(); opt.step(); history.append(float(loss.detach()))
                    if step in STEPS:
                        c = (budget*torch.tanh(variable)*mask).detach()
                        ys = transformed(original,c).detach().cpu().numpy().reshape(batch.sgn.shape)
                        broadcast = c.expand(*original.shape[:-1],3).cpu().numpy()
                        for j,e in enumerate(chunk):
                            checkpoints[step].append(ys[j,:len(e.sgn)].copy()); coeffs[step].append(broadcast[j,:len(e.sgn)].copy())
                losses.append(history)
                c = torch.tensor(rng.uniform(-budget,budget,shape).astype(np.float32),device='cuda')*mask
                ys = transformed(original,c).cpu().numpy().reshape(batch.sgn.shape)
                broadcast = c.expand(*original.shape[:-1],3).cpu().numpy()
                for j,e in enumerate(chunk): random_x.append(ys[j,:len(e.sgn)].copy()); random_c.append(broadcast[j,:len(e.sgn)].copy())
            for step in STEPS:
                record(f'{scope}_bound{budget}_steps{step}',checkpoints[step],coeffs[step],
                       [log[:step] for log in losses], {'scope':scope,'budget':budget,'steps':step,'optimization':True})
            record(f'{scope}_bound{budget}_random',random_x,random_c,[],
                   {'scope':scope,'budget':budget,'steps':0,'optimization':False})
    (folder/'summary.json').write_text(json.dumps({'conditions':results,'references':refs,'targets':targets,
        'baseline_raw_hypotheses':baseline,'baseline_canonical_hypotheses':canonical,
        'baseline_target_bleu4':score_translations(targets,baseline,'phoenix_2014_trans','word')['bleu']['bleu4']},indent=2,ensure_ascii=False)+'\n')
    print('BUDGET_ABLATION_COMPLETE',flush=True)


if __name__ == '__main__': main()
