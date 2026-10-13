"""Post-result DEV diagnostic: keep decoder prefixes fixed, vary physical poses."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from run_physical_semantic_ddpm import dataset,ROOT,BASE
from run_signspark_geometry_dev import project,make_back_translation_model
from run_signspark_gauge_attack import SignCollator,Batch,PAD_TOKEN

def main():
    torch.set_num_threads(8)
    audit=json.loads((ROOT/'data_audit.json').read_text());names=[x['clip_id'] for x in audit['selection']]
    dev={e.sequence:e for e in dataset('dev')};examples=[dev[n] for n in names]
    with np.load(ROOT/'semantic_physical_poses.npz') as z:generated=[project(z[f'clip{i:04d}']) for i in range(128)]
    real=[project(e.sgn.numpy()) for e in examples]
    teacher=make_back_translation_model(BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
    collator=SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[PAD_TOKEN]
    conditions=['real','generated_physical','reversed_real','rolled_real','constant_real_mean'];scores={n:[] for n in conditions}
    with torch.no_grad():
        for start in range(0,128,16):
            ids=list(range(start,start+16));chunk=[copy.copy(examples[i]) for i in ids]
            for condition in conditions:
                batch=Batch(collator(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True)
                x=torch.zeros_like(batch.sgn)
                for j,i in enumerate(ids):
                    n=len(real[i])
                    if condition=='real':pose=real[i]
                    elif condition=='generated_physical':pose=generated[i]
                    elif condition=='reversed_real':pose=real[i][::-1].copy()
                    elif condition=='constant_real_mean':pose=np.tile(real[i].mean(0,keepdims=True),(n,1));pose=project(pose)
                    else:
                        other=real[(i+1)%128];indices=np.linspace(0,len(other)-1,n).round().astype(int);pose=other[indices]
                    x[j,:n]=torch.as_tensor(pose,device=x.device)
                out,_=teacher.forward(x,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                logits=out[0];loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad,reduction='none').reshape(batch.txt.shape)
                valid=batch.txt!=pad;ce=(loss.sum(1)/valid.sum(1)).cpu().tolist();scores[condition].extend(ce)
    rng=np.random.default_rng(42);weights=rng.multinomial(128,np.ones(128)/128,size=2000)/128
    result={'post_result_exploratory_probe':True,'clips':128,'same_caption_and_decoder_prefix_in_all_conditions':True,'TEST_used':False,'not_a_new_training_arm':True,'mean_per_clip_cross_entropy':{n:float(np.mean(s)) for n,s in scores.items()},'per_clip_cross_entropy':scores,'versus_real':{}}
    for n in conditions[1:]:
        delta=np.asarray(scores[n])-scores['real'];result['versus_real'][n]={'delta_ce':float(delta.mean()),'ci95':np.percentile(weights@delta,[2.5,97.5]).tolist()}
    (ROOT/'teacher_prefix_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='per_clip_cross_entropy'}),flush=True)

if __name__=='__main__':main()
