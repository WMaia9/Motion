"""New fixed TEST confirmation, after freezing TRAIN/DEV study models.

No fitting, optimizer or checkpoint selection. Prior TEST-use history remains
disclosed; these are split-disjoint from fitting, not project-wide untouched data.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_learned_equivalence import OutputLayer,token_tensor,GENERATOR,SEEDS
import run_physical_semantic_ddpm as base
from run_signspark_geometry_dev import project,_build_split,load_config,make_back_translation_model,back_translate,score_translations
from run_evaluation_heldout import geometry
from run_canonical_readers import CHECKPOINTS
from verify_signspark_studies import sha
from audit_reviewed_manuscript_evidence import metric

ROOT=Path('reports/cvpr_learned_confirmation_20261012')
LEARNED=Path('reports/cvpr_learned_equivalence_20261011')
READERS=Path('reports/cvpr_canonical_readers_20261011')


def read(p):return json.loads(Path(p).read_text())
def write(name,r):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')


def register():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    for root in [LEARNED,READERS]:assert (root/'manifest_sha256.json').exists(),'discovery not frozen'
    checkpoints=[LEARNED/f'training/seed{s}/{arm}.pt' for s in SEEDS for arm in ['text_raw','text_canonical','motion_raw']]
    checkpoints += [READERS/corpus/f'seed{s}'/arm/'reader.ckpt' for corpus in CHECKPOINTS for s in SEEDS for arm in ['raw','canonical']]
    protocol={'purpose':'fixed confirmation after freezing development results; no new fitting or selection',
        'seeds':SEEDS,'splits':['PHOENIX-2014T TEST','How2Sign TEST','CSL-Daily TEST'],
        'generator':'same frozen local DDPM; replay existing stored TEST motions',
        'learned_conditions':['text_raw/source_text','text_raw/rolled_adapter_text','text_canonical/source_text','motion_raw/source_text'],
        'reader_conditions':['raw fine-tune/native','raw fine-tune/canonical','canonical fine-tune/canonical','canonical fine-tune/fixed equivalent encoding'],
        'fixed_gauge':{'alpha':1.1,'beta':1.1,'gamma':.1,'left_hand_unchanged':True},
        'no_optimizer':True,'no_checkpoint_or_hyperparameter_selection':True,
        'discovery_manifest_hashes':{str(root):sha(root/'manifest_sha256.json') for root in [LEARNED,READERS]},
        'checkpoint_hashes':{str(p):sha(p) for p in checkpoints},
        'frozen_TEST_motion_sha256':sha(Path('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz')),
        'project_TEST_history':'public splits already used in earlier project experiments; not pristine project-wide holdouts',
        'code_sha256':sha(Path(__file__))}
    path=ROOT/'protocol.json'
    if path.exists():assert read(path)==protocol,'registered protocol changed'
    else:write('protocol.json',protocol)
    print('FIXED_TEST_CONFIRMATION_REGISTERED',flush=True)


def score(model,xs,refs,ids,dc):
    hyps=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32)
    result={'clip_ids':ids,'references':refs,'hypotheses':hyps,
            'scores':score_translations(refs,hyps,dc['version'],dc['level'])}
    assert abs(metric(refs,hyps,char=dc['version']=='csldaily')-result['scores']['bleu']['bleu4'])<1e-10
    return result


@torch.no_grad()
def learned():
    register();audit=read(base.ROOT/'data_audit.json');stoi={w:i for i,w in enumerate(audit['vocab'])}
    folder=Path('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry')
    source=read(folder/'raw_scores.json');ids=source['clip_ids'];refs=source['references']
    assert not set(ids)&set(audit['train_ids']) and not set(ids)&set(audit['dev_ids']) and len(ids)==642
    with np.load(folder/'poses.npz') as z:xs=[project(z[f'clip{i:04d}']) for i in range(len(ids))]
    records=[{'clip_id':name,'frames':len(x),'text':text.split()} for name,x,text in zip(ids,xs,refs)]
    generator=base.Denoiser(len(stoi)).cuda();generator.load_state_dict(torch.load(GENERATOR,map_location='cpu',weights_only=False)['state_dict'],strict=True);generator.eval()
    teacher=make_back_translation_model(base.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
    dc=load_config(base.BASE/'checkpoints/PHOENIX-2014T/config.yaml')['data']
    baseline=score(teacher,xs,refs,ids,dc);write('learned/baseline.json',baseline)
    write('learned/data_audit.json',{'clips':642,'clip_ids':ids,'TRAIN_clip_overlap':0,'DEV_clip_overlap':0,
        'TEST_exact_caption_seen_in_TRAIN':[r in {' '.join(e['text']) for e in read(LEARNED/'data_audit.json')['train']} for r in refs]})
    for seed in SEEDS:
        for arm in ['text_raw','text_canonical','motion_raw']:
            model=OutputLayer(generator).cuda();model.load_state_dict(torch.load(LEARNED/f'training/seed{seed}/{arm}.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True);model.eval()
            for condition in ['source_text','rolled_adapter_text'] if arm=='text_raw' else ['source_text']:
                outputs=[];coefficients=[]
                for start in range(0,len(xs),32):
                    group=records[start:start+32];length=torch.tensor([e['frames'] for e in group],device='cuda')
                    x=torch.zeros(len(group),int(length.max()),240,device='cuda')
                    for j,pose in enumerate(xs[start:start+32]):x[j,:len(pose)]=torch.from_numpy(pose).cuda()
                    conditioning=group if condition=='source_text' else [records[(start+j+1)%len(records)] for j in range(len(group))]
                    tokens=token_tensor(conditioning,stoi)
                    if arm=='motion_raw':tokens=torch.zeros_like(tokens);tokens[:,0]=2
                    y,c=model(x,tokens,length)
                    for j,n in enumerate(length.tolist()):outputs.append(y[j,:n].cpu().numpy().copy());coefficients.append(c[j,:n].cpu().numpy().copy())
                raw=score(teacher,outputs,refs,ids,dc);canonical=score(teacher,[project(x) for x in outputs],refs,ids,dc)
                changes=sum(a!=b for a,b in zip(baseline['hypotheses'],canonical['hypotheses']))
                targets=refs if condition=='source_text' else refs[1:]+refs[:1]
                result={'seed':seed,'arm':arm,'condition':condition,'clips':642,'raw':raw,'canonical':canonical,
                    'adapter_target_references':targets,'raw_adapter_target_bleu4':score_translations(targets,raw['hypotheses'],dc['version'],dc['level'])['bleu']['bleu4'],
                    'canonical_changed_hypotheses':changes,'evaluation_optimization_steps':0,**geometry(xs,outputs)}
                assert changes==0,'finite-precision hypothesis change'
                tag=f'seed{seed}_{arm}_{condition}';write('learned/'+tag+'.json',result)
                np.savez_compressed(ROOT/'learned'/(tag+'_poses.npz'),**{f'{p}{i:04d}':x for p,arr in [('attacked',outputs),('coefficient',coefficients)] for i,x in enumerate(arr)})
                print('CONFIRMED_LEARNED',tag,raw['scores']['bleu']['bleu4'],canonical['scores']['bleu']['bleu4'],result['raw_adapter_target_bleu4'],flush=True)
            del model;torch.cuda.empty_cache()
    write('learned/complete.json',{'clips':642,'seeds':SEEDS,'conditions':12,'all_canonical_hypotheses_unchanged':True})


@torch.no_grad()
def readers():
    register()
    for corpus,checkpoint in CHECKPOINTS.items():
        dc=dict(load_config(checkpoint/'config.yaml')['data']);dc.pop('pseudo_gloss_dir',None)
        oldroot=Path('reports/cvpr_representation_paper_20261011') if corpus=='CSL-Daily' else Path('reports/cvpr_evaluation_confirmation_20261011')
        dc.update(base_path=str((oldroot/'data/lmdb').resolve()),index_cache=False,max_sgn_length=300)
        dataset=_build_split(dc,'test',None);entries=[dataset[i] for i in range(len(dataset))]
        ids=[e.sequence for e in entries];refs=[' '.join(e.txt) for e in entries];xs=[e.sgn.numpy() for e in entries]
        audit=read(READERS/corpus/'data_audit.json');assert not set(ids)&set(audit['train_ids']) and not set(ids)&set(audit['dev_ids'])
        canonical=[project(x) for x in xs];gauged=[]
        for x in xs:
            d=x.reshape(-1,40,6).copy();mask=np.ones(40,bool);mask[10:25]=False;a=d[:,mask,:3].copy()
            d[:,mask,:3]*=1.1;d[:,mask,3:]=1.1*d[:,mask,3:]+.1*a;gauged.append(project(d.reshape(x.shape)))
        public=make_back_translation_model(checkpoint);public.eval();public.do_recognition=False
        for name,poses in [('raw',xs),('canonical',canonical)]:
            result=score(public,poses,refs,ids,dc);write(f'readers/{corpus}/public_{name}.json',result)
            print('CONFIRMED_PUBLIC_READER',corpus,name,result['scores']['bleu']['bleu4'],flush=True)
        del public;torch.cuda.empty_cache()
        write(f'readers/{corpus}/data_audit.json',{'clips':len(ids),'clip_ids':ids,'TRAIN_overlap':0,'DEV_overlap':0})
        for seed in SEEDS:
            for arm in ['raw','canonical']:
                model=make_back_translation_model(checkpoint);model.load_state_dict(torch.load(READERS/corpus/f'seed{seed}/{arm}/reader.ckpt',map_location='cpu',weights_only=False)['model_state'],strict=True);model.eval();model.do_recognition=False
                conditions=[('native',xs),('canonical',canonical)] if arm=='raw' else [('native',canonical),('fixed_gauge',gauged)]
                original=None
                for name,poses in conditions:
                    result=score(model,poses,refs,ids,dc);write(f'readers/{corpus}/seed{seed}_{arm}_{name}.json',result)
                    if original is None:original=result
                    elif arm=='canonical':assert original['hypotheses']==result['hypotheses'],'canonical fixed-gauge hypothesis changed'
                    print('CONFIRMED_CANONICAL_READER',corpus,seed,arm,name,result['scores']['bleu']['bleu4'],flush=True)
                del model;torch.cuda.empty_cache()
    write('readers/complete.json',{'seeds':SEEDS,'corpora':list(CHECKPOINTS),'all_canonical_fixed_gauge_hypotheses_unchanged':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['learned','readers'])
    args=parser.parse_args();globals()[args.phase]()
