"""Matched DDPM ablations: same textual prefix, different physical movements."""
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
from run_signspark_gauge_attack import SignCollator, Batch, PAD_TOKEN

ROOT = Path('reports/cvpr_fixed_prefix_diffusion_20261010')
OLD = Path('reports/cvpr_physical_semantic_ddpm_20261010')
ARMS = ['physical_ce', 'physical_real_near', 'physical_generated_cyclic',
        'physical_generated_near', 'raw_generated_near']


def negative_indices(ids, tokens, dropped):
    """Caption-only deterministic mining, without consuming training RNG."""
    eligible = [j for j in range(len(ids)) if not bool(dropped[j])]
    result = {}
    for j in eligible:
        candidates = [k for k in eligible if tokens[ids[k]] != tokens[ids[j]]]
        if not candidates:
            continue
        anchor = set(tokens[ids[j]])
        def similarity(k):
            other = set(tokens[ids[k]])
            return len(anchor & other) / max(1, len(anchor | other))
        near = min(candidates, key=lambda k: (-similarity(k), ids[k]))
        cyclic = min(candidates, key=lambda k: (k-j) % len(ids))
        result[j] = {'near': near, 'cyclic': cyclic,
                     'near_jaccard': similarity(near),
                     'cyclic_jaccard': similarity(cyclic)}
    return result


def align_negative(predictions, lengths, chosen, negatives, maximum):
    """Preserve gradients to generated negatives, resample before projection."""
    sequences = []
    for i, j in zip(chosen, negatives):
        n = int(lengths[i]); m = int(lengths[j])
        sequence = predictions[j, :m].T[None]
        aligned = F.interpolate(sequence, size=n, mode='linear', align_corners=True)[0].T
        sequences.append(F.pad(aligned, (0, 0, 0, maximum-n)))
    return torch.stack(sequences)


def per_clip_ce(teacher, batch, poses, pad):
    output, _ = teacher.forward(poses, batch.sgn_mask, batch.sgn_lengths,
                                batch.txt_input, batch.txt_mask)
    logits = output[0]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           batch.txt.reshape(-1), ignore_index=pad,
                           reduction='none').reshape(batch.txt.shape)
    valid = batch.txt != pad
    return loss.sum(1) / valid.sum(1)


def fit(model, train, dev, stoi, mean, std, alpha, name, teacher):
    previous.seed(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    mean_gpu = torch.tensor(mean, device='cuda'); std_gpu = torch.tensor(std, device='cuda')
    collator = SignCollator(teacher.txt_vocab, None, 240)
    pad = teacher.txt_vocab.stoi[PAD_TOKEN]
    captions = [tuple(e.txt) for e in train]
    history = []; audits = []; gradient_audit = None; stochastic_audit = []
    for epoch in range(5):
        started = time.time(); model.train()
        order = np.random.permutation(len(train)); ordered = []
        for pos in range(0, len(order), 256):
            ordered.extend(sorted(order[pos:pos+256], key=lambda i: len(train[i].sgn)))
        groups = [ordered[pos:pos+64] for pos in range(0, len(ordered), 64)]
        random.shuffle(groups)
        mse_values = []; semantic_values = []; positive_ce = []; negative_ce = []
        examples_used = 0
        for batch_index, ids in enumerate(groups):
            x, text, lengths = previous.pack(train, ids, stoi, mean, std)
            dropped = torch.rand(len(ids), device='cuda') < .1
            text = text.clone(); text[dropped] = 0; text[dropped, 0] = 2
            timesteps = torch.randint(0, 1000, (len(ids),), device='cuda')
            noise = torch.randn_like(x); a = alpha[timesteps][:, None, None]
            prediction = model(a.sqrt()*x + (1-a).sqrt()*noise, timesteps, text, lengths)
            valid = torch.arange(x.shape[1], device='cuda')[None] < lengths[:, None]
            mse = ((prediction-x).square()*valid[:, :, None]).sum()/(valid.sum()*240)
            semantic = torch.zeros((), device='cuda')
            mappings = negative_indices(ids, captions, dropped)
            chosen = [j for j in range(len(ids)) if timesteps[j] <= 800 and j in mappings][:16]
            if epoch == 0:
                # Small sufficient audit of matching random choices and inputs.
                stochastic_audit.append({'ids': list(map(int, ids)),
                    'timesteps': timesteps.cpu().tolist(), 'drop': dropped.cpu().tolist(),
                    'noise_sum': float(noise.double().sum()),
                    'noise_squared_sum': float(noise.double().square().sum())})
            if chosen:
                entries = [copy.copy(train[ids[j]]) for j in chosen]
                batch = Batch(collator(entries), txt_pad_index=pad, sgn_dim=240, use_cuda=True)
                maximum = batch.sgn.shape[1]
                mask = valid[chosen, :maximum, None]
                positive = (prediction[chosen, :maximum]*std_gpu+mean_gpu)*mask
                physical = not name.startswith('raw_')
                positive_input = previous.physical(positive) if physical else positive
                ce_positive = per_clip_ce(teacher, batch, positive_input, pad)
                contrast = name != 'physical_ce'
                if contrast:
                    choice = 'cyclic' if name.endswith('cyclic') else 'near'
                    negatives = [mappings[j][choice] for j in chosen]
                    if name == 'physical_real_near':
                        negative = align_negative(x*std_gpu+mean_gpu, lengths,
                                                  chosen, negatives, maximum).detach()
                    else:
                        negative = align_negative(prediction*std_gpu+mean_gpu, lengths,
                                                  chosen, negatives, maximum)
                    negative_input = previous.physical(negative) if physical else negative
                    # Exactly the same Batch object: target, prefix, masks, durations.
                    ce_negative = per_clip_ce(teacher, batch, negative_input, pad)
                    semantic = (ce_positive + F.softplus(.5+ce_positive-ce_negative)).mean()
                    negative_ce.append(float(ce_negative.detach().mean()))
                else:
                    semantic = ce_positive.mean(); negatives = [mappings[j]['near'] for j in chosen]
                if gradient_audit is None:
                    inputs = [positive] + ([negative] if contrast and negative.requires_grad else [])
                    gradients = torch.autograd.grad(semantic, inputs, retain_graph=True)
                    def statistics(value, gradient):
                        d=value.reshape(-1,40,6); g=gradient.reshape(-1,40,6)
                        vertical = torch.stack([(g[...,:3]*d[...,:3]).sum(-1),
                            (g[...,3:]*d[...,3:]).sum(-1),
                            (g[...,3:]*d[...,:3]).sum(-1)], -1)
                        return {'gradient_norm': float(gradient.norm()),
                                'maximum_vertical_dot': float(vertical.abs().max())}
                    gradient_audit = {'physical_input':physical,'positive':statistics(positive,gradients[0]),
                        'negative_receives_gradient':len(inputs)>1,
                        'negative':statistics(negative,gradients[1]) if len(inputs)>1 else None,
                        'same_batch_prefix_target_masks':True}
                if epoch == 0:
                    audits.append({'batch':batch_index,'positive_ids':[train[ids[j]].sequence for j in chosen],
                        'negative_ids':[train[ids[j]].sequence for j in negatives],
                        'all_captions_distinct':all(captions[ids[j]]!=captions[ids[k]] for j,k in zip(chosen,negatives)),
                        'near_jaccard':[mappings[j]['near_jaccard'] for j in chosen],
                        'cyclic_jaccard':[mappings[j]['cyclic_jaccard'] for j in chosen],
                        'positive_lengths':[int(lengths[j]) for j in chosen],
                        'negative_original_lengths':[int(lengths[j]) for j in negatives]})
                positive_ce.append(float(ce_positive.detach().mean())); examples_used += len(chosen)
            loss = mse + .02*semantic
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step(); mse_values.append(float(mse.detach())); semantic_values.append(float(semantic.detach()))
        validation = previous.validation(model, dev, stoi, mean, std, alpha)
        record = {'epoch':epoch+1,'train_mse':float(np.mean(mse_values)),
            'semantic_term':float(np.mean(semantic_values)),
            'positive_ce':float(np.mean(positive_ce)),
            'negative_ce':float(np.mean(negative_ce)) if negative_ce else None,
            'semantic_examples':examples_used,'dev_fixed_noise_mse':validation,
            'seconds':time.time()-started}
        history.append(record)
        (ROOT/(name+'_history.json')).write_text(json.dumps(history,indent=2)+'\n')
        print('FIXED_PREFIX_TRAIN', name, json.dumps(record), flush=True)
    torch.save({'state_dict':model.state_dict(),'epoch':5,'selection':'last epoch',
                'shared_checkpoint':str(OLD/'shared_best.pt')}, ROOT/(name+'.pt'))
    (ROOT/(name+'_negative_audit.json')).write_text(json.dumps(audits,indent=2)+'\n')
    (ROOT/(name+'_stochastic_audit.json')).write_text(json.dumps(stochastic_audit,indent=2)+'\n')
    (ROOT/(name+'_gradient_audit.json')).write_text(json.dumps(gradient_audit,indent=2)+'\n')


def evaluate(models, dev, names, stoi, mean, std, alpha, teacher,
             shuffled_arm='physical_generated_near'):
    native = {e.sequence:e for e in dev}
    references = [' '.join(native[n].txt) for n in names]
    glosses = [' '.join(native[n].gls) for n in names]
    checkpoint = torch.load(previous.PREV/'ctc_reader/best.pt',map_location='cpu',weights_only=False)
    state=checkpoint['state_dict']
    reader=previous.Reader(len(checkpoint['vocab']),state['mean'].numpy(),state['std'].numpy()).cuda()
    reader.load_state_dict(state,strict=True)
    summary={}
    conditions=[(name,False) for name in ARMS]+[(shuffled_arm,True)]
    for arm, shuffled in conditions:
        name = arm + ('_shuffled' if shuffled else '')
        if (ROOT/(name+'_scores.json')).exists():
            summary[name]=json.loads((ROOT/(name+'_scores.json')).read_text());continue
        xs=previous.sample(models[arm],dev,names,stoi,mean,std,alpha,shuffled)
        ys=[previous.project(x) for x in xs]
        raw=previous.back_translate(teacher,[torch.from_numpy(x) for x in xs],batch_size=32)
        canonical=previous.back_translate(teacher,[torch.from_numpy(x) for x in ys],batch_size=32)
        hr=previous.decode(reader,xs,checkpoint['vocab']);hc=previous.decode(reader,ys,checkpoint['vocab'])
        result={'clips':128,'clip_ids':names,'references':references,'raw_hypotheses':raw,
            'canonical_hypotheses':canonical,
            'raw_scores':previous.score_translations(references,raw,'phoenix_2014_trans','word'),
            'canonical_scores':previous.score_translations(references,canonical,'phoenix_2014_trans','word'),
            'independent_raw_gloss':previous.metrics(glosses,hr),
            'independent_canonical_gloss':previous.metrics(glosses,hc),
            'independent_raw_hypotheses':hr,'independent_canonical_hypotheses':hc,
            'raw_canonical_changed_sentences':sum(a!=b for a,b in zip(raw,canonical))}
        np.savez_compressed(ROOT/(name+'_poses.npz'),**{f'clip{i:04d}':x for i,x in enumerate(xs)})
        (ROOT/(name+'_scores.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        summary[name]=result
        print('FIXED_PREFIX_SCORE',name,result['raw_scores']['bleu']['bleu4'],
              result['canonical_scores']['bleu']['bleu4'],result['independent_canonical_gloss']['wer'],flush=True)
    (ROOT/'summary.json').write_text(json.dumps({n:{k:v for k,v in r.items() if k.endswith('scores') or k.endswith('gloss')}
                                             for n,r in summary.items()},indent=2)+'\n')


def main():
    assert not (ROOT/'manifest_sha256.json').exists(),'report already frozen'
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    audit=json.loads((ROOT/'data_audit.json').read_text());train=previous.dataset('train');dev=previous.dataset('dev')
    assert [e.sequence for e in train]==audit['train_ids']
    assert [e.sequence for e in dev]==audit['dev_ids']
    names=[e['clip_id'] for e in audit['selection']]
    stoi={word:i for i,word in enumerate(audit['vocab'])}
    with np.load(ROOT/'normalization.npz') as data:mean=data['mean'].copy();std=data['std'].copy()
    shared=torch.load(OLD/'shared_best.pt',map_location='cpu',weights_only=False)['state_dict']
    teacher=previous.make_back_translation_model(previous.BASE/'checkpoints/PHOENIX-2014T')
    teacher.eval();teacher.do_recognition=False
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    alpha=previous.alpha_schedule();models={}
    for name in ARMS:
        model=previous.Denoiser(len(stoi)).cuda();model.load_state_dict(shared,strict=True)
        if not (ROOT/(name+'.pt')).exists():fit(model,train,dev,stoi,mean,std,alpha,name,teacher)
        model.load_state_dict(torch.load(ROOT/(name+'.pt'),map_location='cpu',weights_only=False)['state_dict'],strict=True)
        models[name]=model
    evaluate(models,dev,names,stoi,mean,std,alpha,teacher)
    print('FIXED_PREFIX_COMPLETE',flush=True)


if __name__=='__main__':main()
