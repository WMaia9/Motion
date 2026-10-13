"""Fixed third-language audit with native Chinese tokenization/scoring."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from run_signspark_geometry_dev import load_config,_build_split,project,make_back_translation_model,back_translate,score_translations
from run_signspark_gauge_attack import transform,SignCollator,Batch,PAD_TOKEN
from run_evaluation_heldout import geometry
from analyze_signspark_geometry_dev import statistics,b4
from verify_signspark_studies import sha

ROOT=Path('reports/cvpr_representation_paper_20261011')


def chinese(text):return [' '.join(list(t.replace(' ',''))) for t in text]


def scored(model,xs,refs,dc):
    h=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32)
    scores=score_translations(refs,h,dc['version'],dc['level'])
    assert abs(b4(statistics(chinese(refs),chinese(h)).sum(0))-scores['bleu']['bleu4'])<1e-10
    return {'hypotheses':h,'scores':scores}


def interval(refs,before,after):
    a=statistics(chinese(refs),chinese(before));b=statistics(chinese(refs),chinese(after));n=len(refs)
    weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=2000)
    values=np.array([b4(y)-b4(x) for x,y in zip(weights@a,weights@b)])
    return {'delta_bleu4':b4(b.sum(0))-b4(a.sum(0)),'ci95':np.percentile(values,[2.5,97.5]).tolist()}


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    registered=json.loads((ROOT/'sources/registered_inputs.json').read_text());assert sha(ROOT/'PROTOCOLO.md')==registered['protocol_sha256']
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=ROOT/'checkpoints/CSL-Daily';config=load_config(checkpoint/'config.yaml');dc=dict(config['data'])
    dc.update(base_path=str((ROOT/'data/lmdb').resolve()),index_cache=False,max_sgn_length=300)
    assert 'csldaily' in dc['version'].lower() and dc['tokenizer']=='jieba'
    data=_build_split(dc,'test',None);entries=[data[i] for i in range(len(data))];assert len(entries)>=128
    xs=[]
    for e in entries:
        x=e.sgn.numpy().copy();x[:,60:150]=project(x)[:,60:150];e.sgn=torch.from_numpy(x);xs.append(x)
    refs=[' '.join(e.txt) for e in entries];ids=[e.sequence for e in entries]
    selected_indices=np.sort(np.random.default_rng(2027).choice(len(entries),128,replace=False));selected=[entries[i] for i in selected_indices]
    folder=ROOT/'CSL-Daily';selection=[{'test_index':int(i),'clip_id':entries[i].sequence,'frames':len(entries[i].sgn),'reference':refs[i]} for i in selected_indices]
    (folder/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    (folder/'data_audit.json').write_text(json.dumps({'clips':len(entries),'clip_ids':ids,'selection_before_scores':True,'scoring':'native csldaily per-character; optimization native jieba words','no_local_generator_or_reader_training':True},ensure_ascii=False,indent=2)+'\n')
    model=make_back_translation_model(checkpoint);model.do_recognition=False
    for p in model.parameters():p.requires_grad_(False)
    raw=scored(model,xs,refs,dc);can=scored(model,[project(x) for x in xs],refs,dc)
    full={'clips':len(entries),'clip_ids':ids,'references':refs,'raw':raw,'canonical':can,
        'raw_canonical_changed_sentences':sum(a!=b for a,b in zip(raw['hypotheses'],can['hypotheses'])),
        'public_rounded_TEST_bleu4':6.98,'difference_from_public_rounded_value':raw['scores']['bleu']['bleu4']-6.98,
        'native_decode':{'beam_size':model.beam_size,'beam_alpha':model.beam_alpha,'max_output_len':model.max_output_len},
        'canonical_accuracy_change':interval(refs,raw['hypotheses'],can['hypotheses'])}
    (folder/'full_real_scores.json').write_text(json.dumps(full,ensure_ascii=False,indent=2)+'\n')
    print('CSL_FULL_TEST',len(entries),raw['scores']['bleu']['bleu4'],can['scores']['bleu']['bleu4'],flush=True)
    xs=[xs[i] for i in selected_indices];refs=[refs[i] for i in selected_indices]
    baseline=scored(model,xs,refs,dc);canonical=scored(model,[project(x) for x in xs],refs,dc)
    collator=SignCollator(model.txt_vocab,None,240);pad=model.txt_vocab.stoi[PAD_TOKEN]
    jointmask=torch.ones(1,1,40,1,device='cuda');jointmask[:,:,10:25]=0;summary={}
    for name in ['correct_caption','rolled_wrong_caption']:
        targets=refs if name=='correct_caption' else refs[1:]+refs[:1];outputs=[];coefficients=[];logs=[]
        for start in range(0,128,32):
            chunk=[copy.copy(e) for e in selected[start:start+32]]
            for e,target in zip(chunk,targets[start:start+32]):e.txt=target.split()
            batch=Batch(collator(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True)
            original=batch.sgn.detach().reshape(32,-1,40,6);u=torch.zeros((*original.shape[:-1],3),device='cuda',requires_grad=True)
            optimizer=torch.optim.Adam([u],lr=.1);lossvalues=[]
            for step in range(80):
                batch.sgn=transform(original,u*jointmask).reshape(32,-1,240)
                prediction,_=model.forward(batch.sgn,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                logits=prediction[0];loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad)
                assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();assert torch.isfinite(u.grad).all();optimizer.step();lossvalues.append(float(loss.detach()))
            y=transform(original,u*jointmask).detach().cpu().numpy().reshape(32,-1,240);g=(u*jointmask).detach().cpu().numpy()
            for j,e in enumerate(chunk):outputs.append(y[j,:len(e.sgn)].copy());coefficients.append(g[j,:len(e.sgn)].copy())
            logs.append({'start':start,'cross_entropy':lossvalues});print('CSL_ATTACK',name,start,lossvalues[0],lossvalues[-1],flush=True)
        after=scored(model,outputs,refs,dc);defended=scored(model,[project(x) for x in outputs],refs,dc)
        changed=sum(a!=b for a,b in zip(canonical['hypotheses'],defended['hypotheses']))
        target_before=score_translations(targets,baseline['hypotheses'],dc['version'],dc['level']);target_after=score_translations(targets,after['hypotheses'],dc['version'],dc['level'])
        r={'clips':128,'clip_ids':[e.sequence for e in selected],'references':refs,'targets':targets,'baseline_actual':baseline,
            'baseline_target_scores':target_before,'raw_after_actual':after,'raw_after_target_scores':target_after,
            'canonical_before_actual':canonical,'canonical_after_actual':defended,'canonical_changed_sentences':changed,
            'different_targets':sum(a!=b for a,b in zip(refs,targets)),'target_change':interval(targets,baseline['hypotheses'],after['hypotheses']),
            'actual_change':interval(refs,baseline['hypotheses'],after['hypotheses']),'losslogs':logs,**geometry(xs,outputs)}
        np.savez_compressed(folder/(name+'_poses.npz'),**{f'{prefix}{i:04d}':x for prefix,array in [('source',xs),('attacked',outputs),('gauge',coefficients)] for i,x in enumerate(array)})
        (folder/(name+'_scores.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
        summary[name]={k:r[k] for k in ['canonical_changed_sentences','different_targets','target_change','actual_change','max_rotation_error_deg']}
        summary[name].update(baseline_target_bleu4=target_before['bleu']['bleu4'],attacked_target_bleu4=target_after['bleu']['bleu4'],attacked_actual_bleu4=after['scores']['bleu']['bleu4'],canonical_actual_bleu4=defended['scores']['bleu']['bleu4'])
        print('CSL_ATTACK_RESULT',name,json.dumps(summary[name]),flush=True)
    (folder/'summary.json').write_text(json.dumps({'full_TEST_raw_bleu4':full['raw']['scores']['bleu']['bleu4'],'full_TEST_canonical_bleu4':full['canonical']['scores']['bleu']['bleu4'],'full_TEST_clips':full['clips'],'conditions':summary,'same_BT_architecture_family':True,'bootstrap_unit':'clip','replicates':2000,'seed':42},indent=2)+'\n')
    print('CSL_TEST_COMPLETE',flush=True)


if __name__=='__main__':main()
