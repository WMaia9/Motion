"""Fresh white-box DEV attack on both fine-tuned readers, same fixed budget."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from run_reviewer_attack_ablation import transformed
from run_reviewer_training import ROOT, previous, score, read, write
from run_signspark_gauge_attack import SignCollator, Batch, PAD_TOKEN
from run_evaluation_heldout import geometry


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    dev = {e.sequence:e for e in previous.dataset('dev')}
    selection = read(Path('reports/cvpr_signspark_gauge_attack_20261010/PHOENIX-2014T/selection.json'))
    entries = [dev[r['clip_id']] for r in selection]
    xs = [e.sgn.numpy() for e in entries]; ids = [e.sequence for e in entries]
    refs = [' '.join(e.txt) for e in entries]; targets = refs[1:]+refs[:1]
    write('readers/attack_protocol.json', {'clips':64,'split':'DEV','steps':80,'budget':.35,'lr':.1,
        'targets':targets,'references':refs,'clip_ids':ids,'selection':'same frozen pilot subset',
        'fresh_white_box_against_each_fine_tuned_reader':True,'canonicalization_inside_loss_for_canonical_arm':True})
    for arm in ['raw','canonical']:
        model = previous.make_back_translation_model(previous.BASE/'checkpoints/PHOENIX-2014T')
        checkpoint = torch.load(ROOT/f'readers/{arm}/fine_tuned.ckpt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state'],strict=True); model.eval(); model.do_recognition = False
        for p in model.parameters(): p.requires_grad_(False)
        collate = SignCollator(model.txt_vocab,None,240); pad = model.txt_vocab.stoi[PAD_TOKEN]
        baseline_x = [previous.project(x) for x in xs] if arm == 'canonical' else xs
        baseline = score(model,baseline_x,refs,ids)
        ys = []; coefficients = []; logs = []; gradient_max = 0.
        for start in range(0,len(entries),32):
            chunk = [copy.copy(e) for e in entries[start:start+32]]
            for e,target in zip(chunk,targets[start:start+32]): e.txt = target.split()
            batch = Batch(collate(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True)
            original = batch.sgn.detach().reshape(len(chunk),-1,40,6)
            u = torch.zeros((*original.shape[:-1],3),device='cuda',requires_grad=True)
            mask = torch.ones(1,1,40,1,device='cuda'); mask[:,:,10:25] = 0
            optimizer = torch.optim.Adam([u],lr=.1); losses = []
            for step in range(80):
                c = .35*torch.tanh(u)*mask
                poses = transformed(original,c).reshape(batch.sgn.shape)
                consumed = previous.physical(poses) if arm == 'canonical' else poses
                output,_ = model.forward(consumed,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                logits = output[0]; loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad)
                assert torch.isfinite(loss); optimizer.zero_grad(); loss.backward()
                gradient_max = max(gradient_max,float(u.grad.detach().abs().max())); optimizer.step(); losses.append(float(loss.detach()))
            c = (.35*torch.tanh(u)*mask).detach()
            pose = transformed(original,c).cpu().numpy().reshape(batch.sgn.shape); coeff = c.cpu().numpy()
            for j,e in enumerate(chunk): ys.append(pose[j,:len(e.sgn)].copy()); coefficients.append(coeff[j,:len(e.sgn)].copy())
            logs.append(losses)
        consumed = [previous.project(y) for y in ys] if arm == 'canonical' else ys
        attacked = score(model,consumed,refs,ids)
        result = {'clips':64,'clip_ids':ids,'references':refs,'targets':targets,'baseline':baseline,'attacked':attacked,
            'baseline_target_bleu4': previous.score_translations(targets,baseline['hypotheses'],'phoenix_2014_trans','word')['bleu']['bleu4'],
            'attacked_target_bleu4': previous.score_translations(targets,attacked['hypotheses'],'phoenix_2014_trans','word')['bleu']['bleu4'],
            'changed_hypotheses': sum(a != b for a,b in zip(baseline['hypotheses'],attacked['hypotheses'])),
            'max_optimization_gradient':gradient_max,'losslogs':logs,**geometry(xs,ys)}
        write(f'readers/{arm}/white_box_wrong_target.json',result)
        np.savez_compressed(ROOT/f'readers/{arm}/white_box_poses.npz',
            **{f'{tag}{i:04d}':x for tag,values in [('source',xs),('attacked',ys),('coefficient',coefficients)] for i,x in enumerate(values)})
        print('FINE_TUNED_READER_ATTACK',arm,result['attacked_target_bleu4'],result['changed_hypotheses'],gradient_max,flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__': main()
