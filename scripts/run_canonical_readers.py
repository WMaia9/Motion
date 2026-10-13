"""Paired full-TRAIN raw/canonical reader fine-tuning, How2Sign and CSL.

Same pretrained checkpoint and minibatches in each pair. Last-epoch selection;
DEV-only scores and fresh wrong-caption attacks. Never writes to old reports.
"""
import argparse
import copy
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run_physical_semantic_ddpm as old
from run_signspark_geometry_dev import _build_split, load_config, project, make_back_translation_model, back_translate, score_translations
from run_signspark_gauge_attack import SignCollator, Batch, PAD_TOKEN
from run_reviewer_attack_ablation import transformed
from run_evaluation_heldout import geometry
from audit_reviewed_manuscript_evidence import metric
from verify_signspark_studies import sha

ROOT = Path('reports/cvpr_canonical_readers_20261011')
CHECKPOINTS = {'How2Sign':Path('reports/cvpr_signspark_baseline_20261010/checkpoints/How2Sign'),
               'CSL-Daily':Path('reports/cvpr_representation_paper_20261011/checkpoints/CSL-Daily')}
SEEDS = [42,123,777]
EPOCHS = 5


def read(p): return json.loads(Path(p).read_text())
def write(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def setup():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
    assert (ROOT/'sources/downloads.json').exists()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    paths=[folder/'best.ckpt' for folder in CHECKPOINTS.values()]
    protocol={'corpora':list(CHECKPOINTS),'seeds':SEEDS,'arms':['raw','canonical'],
        'epochs':EPOCHS,'lr':.00001,'batch_size':64,'weight_decay':.0001,
        'selection':'last epoch; no DEV/BLEU model selection','objective':'token CE; CTC disabled',
        'training':'full available TRAIN, native max_sent_length filter, max_sgn_length300',
        'fine_tuning_not_training_from_scratch':True,'init':'same public pretrained checkpoint in each paired arm',
        'no_TEST_use':True,'attack':{'subset':64,'split':'DEV','selection_seed':2027,
            'target':'rolled next selected caption','steps':80,'lr':.1,'coefficient_bound':.35,'left_hand_unchanged':True},
        'accuracy_comparisons':'canonical fine-tune versus paired raw fine-tune and public raw baseline',
        'bootstrap_replicates':20000,'bootstrap_seed':42,
        'exploratory_noninferiority_margins_BLEU4':{'How2Sign':.2,'CSL-Daily':.3},
        'noninferiority_status':'exploratory DEV margins, not validated linguistic clinical thresholds',
        'checkpoints':{str(p):sha(p) for p in paths},'code_sha256':sha(Path(__file__))}
    path=ROOT/'protocol.json'
    if path.exists():assert read(path)==protocol,'protocol changed'
    else:write('protocol.json',protocol)


def datasets(corpus):
    config=load_config(CHECKPOINTS[corpus]/'config.yaml');dc=dict(config['data']);dc.pop('pseudo_gloss_dir',None)
    dc.update(index_cache=False,max_sgn_length=300)
    # Keep public DEV untouched, while downloaded TRAIN lives in this new report.
    dc['base_path']=str((ROOT/'data/lmdb').resolve())
    train=_build_split(dc,'train',dc.get('max_sent_length'))
    if corpus=='How2Sign':dc['base_path']=str(Path('reports/cvpr_signspark_baseline_20261010/data/lmdb').resolve())
    dev=_build_split(dc,'dev',None)
    train_entries=[train[i] for i in range(len(train))];dev_entries=[dev[i] for i in range(len(dev))]
    assert train_entries and dev_entries
    assert not {e.sequence for e in train_entries}&{e.sequence for e in dev_entries}
    audit={'TRAIN_clips':len(train_entries),'DEV_clips':len(dev_entries),
        'train_ids':[e.sequence for e in train_entries],'dev_ids':[e.sequence for e in dev_entries],
        'train_dev_clip_overlap':0,'vocab_and_tokenization':'native released checkpoint/config',
        'native_data_config':dc}
    if corpus=='How2Sign':
        path=Path('/mnt/datasets/sign_datasets/raw/how2sign/How2Sign/sentence_level/val/text/en/raw_text/how2sign_val.csv')
        if path.exists():
            with path.open() as stream:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(stream,delimiter='\t')}
            audit['DEV_video_groups']=[lookup[e.sequence] for e in dev_entries]
            dest=ROOT/'sources/how2sign_val.csv';shutil.copy2(path,dest)
            audit['video_metadata_sha256']=sha(dest)
        else:audit['DEV_video_groups_unavailable']=True
    write(corpus+'/data_audit.json',audit)
    print('READER_DATA',corpus,len(train_entries),len(dev_entries),flush=True)
    return train_entries,dev_entries,dc


def score(model,xs,entries,dc):
    refs=[' '.join(e.txt) for e in entries]
    hyps=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32)
    result={'clip_ids':[e.sequence for e in entries],'references':refs,'hypotheses':hyps,
        'scores':score_translations(refs,hyps,dc['version'],dc['level'])}
    assert abs(metric(refs,hyps,char=dc['version']=='csldaily')-result['scores']['bleu']['bleu4'])<1e-10
    return result


def train():
    setup()
    for corpus in CHECKPOINTS:
        data,dev,dc=datasets(corpus);xs=[e.sgn.numpy() for e in dev]
        public=make_back_translation_model(CHECKPOINTS[corpus]);public.do_recognition=False
        for kind,poses in [('raw',xs),('canonical',[project(x) for x in xs])]:
            path=ROOT/corpus/('public_'+kind+'.json')
            if not path.exists():
                result=score(public,poses,dev,dc);write(str(path.relative_to(ROOT)),result)
                print('PUBLIC_READER_SCORE',corpus,kind,result['scores']['bleu']['bleu4'],flush=True)
        del public;torch.cuda.empty_cache()
        for seed in SEEDS:
            matched=[]
            for arm in ['raw','canonical']:
                folder=ROOT/corpus/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
                model=make_back_translation_model(CHECKPOINTS[corpus]);model.do_recognition=False
                collator=SignCollator(model.txt_vocab,None,240);pad=model.txt_vocab.stoi[PAD_TOKEN]
                checkpoint=folder/'reader.ckpt'
                if not checkpoint.exists():
                    old.seed(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.00001,weight_decay=.0001)
                    history=[];orders=[]
                    for epoch in range(EPOCHS):
                        started=time.time();model.train();order=np.random.permutation(len(data));ordered=[]
                        for k in range(0,len(order),256):ordered.extend(sorted(order[k:k+256],key=lambda i:len(data[i].sgn)))
                        groups=[ordered[k:k+64] for k in range(0,len(ordered),64)];random.shuffle(groups);losses=[]
                        for batch_number,ids in enumerate(groups):
                            if epoch==0:orders.append(list(map(int,ids)))
                            batch=Batch(collator([data[i] for i in ids]),txt_pad_index=pad,sgn_dim=240,use_cuda=True)
                            poses=old.physical(batch.sgn) if arm=='canonical' else batch.sgn
                            output,_=model.forward(poses,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                            logits=output[0];loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad)
                            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                            losses.append(float(loss.detach()))
                        history.append({'epoch':epoch+1,'token_ce':float(np.mean(losses)),'seconds':time.time()-started})
                        write(str((folder/'history.json').relative_to(ROOT)),history)
                        print('CANONICAL_READER_TRAIN',corpus,seed,arm,json.dumps(history[-1]),flush=True)
                    torch.save({'model_state':model.state_dict(),'seed':seed,'arm':arm,'epoch':EPOCHS,'selection':'last epoch'},checkpoint)
                    write(str((folder/'batch_orders.json').relative_to(ROOT)),orders)
                matched.append(read(folder/'batch_orders.json'))
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'],strict=True);model.eval()
                for condition,poses in [('native',[project(x) for x in xs] if arm=='canonical' else xs),('canonical',[project(x) for x in xs])]:
                    result=score(model,poses,dev,dc);write(str((folder/(condition+'_scores.json')).relative_to(ROOT)),result)
                    print('CANONICAL_READER_SCORE',corpus,seed,arm,condition,result['scores']['bleu']['bleu4'],flush=True)
                del model;torch.cuda.empty_cache()
            assert matched[0]==matched[1],'batch order differs in paired training'
    write('training_complete.json',{'corpora':list(CHECKPOINTS),'seeds':SEEDS,'paired_minibatches_matched':True,'no_TEST_use':True})


def attack():
    setup()
    for corpus in CHECKPOINTS:
        # DEV-only: avoid loading TRAIN features a second time.
        audit=read(ROOT/corpus/'data_audit.json');dc=audit['native_data_config']
        dataset=_build_split(dc,'dev',None);entries=[dataset[i] for i in range(len(dataset))]
        selected=np.sort(np.random.default_rng(2027).choice(len(entries),64,replace=False))
        subset=[entries[i] for i in selected];xs=[e.sgn.numpy() for e in subset]
        refs=[' '.join(e.txt) for e in subset];targets=refs[1:]+refs[:1]
        write(corpus+'/attack_selection.json',{'indices':selected.tolist(),'ids':[e.sequence for e in subset],
            'references':refs,'targets':targets,'different_targets':sum(a!=b for a,b in zip(refs,targets))})
        for seed in SEEDS:
            for arm in ['raw','canonical']:
                folder=ROOT/corpus/f'seed{seed}'/arm
                model=make_back_translation_model(CHECKPOINTS[corpus]);model.load_state_dict(torch.load(folder/'reader.ckpt',map_location='cpu',weights_only=False)['model_state'],strict=True)
                model.eval();model.do_recognition=False
                for p in model.parameters():p.requires_grad_(False)
                collator=SignCollator(model.txt_vocab,None,240);pad=model.txt_vocab.stoi[PAD_TOKEN]
                before=score(model,[project(x) for x in xs] if arm=='canonical' else xs,subset,dc)
                outputs=[];coefficients=[];logs=[]
                for start in range(0,64,32):
                    chunk=[copy.copy(e) for e in subset[start:start+32]]
                    for j,e in enumerate(chunk):e.txt=list(subset[(start+j+1)%64].txt)
                    batch=Batch(collator(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True)
                    original=batch.sgn.detach().reshape(len(chunk),-1,40,6)
                    u=torch.zeros((*original.shape[:-1],3),device='cuda',requires_grad=True)
                    mask=torch.ones(1,1,40,1,device='cuda');mask[:,:,10:25]=0
                    opt=torch.optim.Adam([u],lr=.1);history=[]
                    for step in range(80):
                        c=.35*torch.tanh(u)*mask;y=transformed(original,c).reshape(batch.sgn.shape)
                        consumed=old.physical(y) if arm=='canonical' else y
                        output,_=model.forward(consumed,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                        logits=output[0];loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad)
                        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();history.append(float(loss.detach()))
                    c=(.35*torch.tanh(u)*mask).detach();y=transformed(original,c).cpu().numpy().reshape(batch.sgn.shape);c=c.cpu().numpy()
                    for j,e in enumerate(chunk):outputs.append(y[j,:len(e.sgn)].copy());coefficients.append(c[j,:len(e.sgn)].copy())
                    logs.append(history)
                after=score(model,[project(x) for x in outputs] if arm=='canonical' else outputs,subset,dc)
                result={'seed':seed,'arm':arm,'baseline':before,'after':after,'targets':targets,
                    'baseline_target_bleu4':score_translations(targets,before['hypotheses'],dc['version'],dc['level'])['bleu']['bleu4'],
                    'after_target_bleu4':score_translations(targets,after['hypotheses'],dc['version'],dc['level'])['bleu']['bleu4'],
                    'changed_hypotheses':sum(a!=b for a,b in zip(before['hypotheses'],after['hypotheses'])),
                    'losslogs':logs,**geometry(xs,outputs)}
                if arm=='canonical':assert before['hypotheses']==after['hypotheses']
                write(str((folder/'wrong_target_attack.json').relative_to(ROOT)),result)
                np.savez_compressed(folder/'attack_poses.npz',**{f'{tag}{i:04d}':x for tag,values in [('source',xs),('attacked',outputs),('coefficient',coefficients)] for i,x in enumerate(values)})
                print('CANONICAL_READER_ATTACK',corpus,seed,arm,result['after_target_bleu4'],result['changed_hypotheses'],flush=True)
                del model;torch.cuda.empty_cache()
    write('attacks_complete.json',{'clips_per_corpus':64,'seeds':SEEDS,'arms':['raw','canonical'],'all_canonical_hypotheses_unchanged':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['train','attack'])
    args=parser.parse_args();globals()[args.phase]()
