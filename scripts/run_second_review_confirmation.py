"""Fixed TEST inference after all development models were frozen. No fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
import run_physical_semantic_ddpm as base
from run_signspark_geometry_dev import project,_build_split,load_config,make_back_translation_model
from run_evaluation_heldout import sample_without_reference_pose,geometry
from run_learned_confirmation import score
from run_real_motion_layers import RealLayer,tokens
from run_signspark_gauge_attack import SignCollator,Batch,PAD_TOKEN
from run_learned_budget import BoundedLayer,profiles,auc
import run_learned_equivalence as old
from verify_signspark_studies import sha

ROOT=Path('reports/cvpr_second_review_confirmation_20261012')
Q=Path('reports/cvpr_quotient_training_20261012');R=Path('reports/cvpr_real_motion_layers_20261012');B=Path('reports/cvpr_learned_budget_20261012')
SEEDS=[42,123,777];ARMS=['geodesic_none_w0','geodesic_raw_w1','geodesic_canonical_w1']
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def register():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
 for p in [Q,R,B]:assert (p/'manifest_sha256.json').exists(),'development must be frozen'
 checkpoints=[Q/f'training/seed{s}/{a}/epoch20.pt' for s in SEEDS for a in ARMS]
 checkpoints += [R/c/f'training/seed{s}/{a}/layer.pt' for c in ['How2Sign','CSL-Daily'] for a,ss in [('text_raw',SEEDS),('motion_raw',[42]),('text_canonical',[42])] for s in ss]
 checkpoints += [B/f'training/seed{s}/bound{b:g}/layer.pt' for s in SEEDS for b in [.01,.05,.1]]
 inputs=checkpoints+[p/'manifest_sha256.json' for p in [Q,R,B]]+[old.GENERATOR,base.ROOT/'normalization.npz',base.ROOT/'data_audit.json',Path('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz')]
 p={'purpose':'fixed TEST confirmation; no fitting, score-based selection or new optimization', 'generator_seeds':SEEDS,'ordinary_generator_arms':ARMS,'ordinary_generator_epoch':20,'ordinary_generator_PRIMARY':'raw-reader CE weight1 vs no CE and canonical-reader CE weight1; three declared gap contrasts',
 'sampling':'50 DDIM, CFG3, seed102; text and prescribed reference duration metadata only; no reference poses supplied to sampler',
 'real_motion_layer_primary_seeds':SEEDS,'real_motion_layer_exploratory_controls_seeds':[42],'bounds':[.01,.05,.1],'conditions':28,
 'source_of_layers':'source captions are inputs; constructed amortized attacks, distinct from ordinary generator training',
 'development_manifests':{str(p):sha(p/'manifest_sha256.json') for p in [Q,R,B]},'input_SHA256':{str(p):sha(p) for p in inputs},
 'TEST_history':'public TEST splits were evaluated during this project; all present model weights and settings fixed before this confirmation, TRAIN/TEST IDs disjoint; no claim of project-wide untouched holdouts',
 'source_code_sha256':sha(__file__)}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==p
 else:write('protocol.json',p)
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
def teacher_for(c):
 path=Path('reports/cvpr_representation_paper_20261011/checkpoints')/c if c=='CSL-Daily' else base.BASE/'checkpoints'/c
 teacher=make_back_translation_model(path);teacher.eval();teacher.do_recognition=False
 return teacher,dict(load_config(path/'config.yaml')['data'])

@torch.no_grad()
def ordinary():
 register();source=read('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/raw_scores.json');ids=source['clip_ids'];refs=source['references'];audit=read(base.ROOT/'data_audit.json');assert not set(ids)&set(audit['train_ids']) and not set(ids)&set(audit['dev_ids']);stoi={w:i for i,w in enumerate(audit['vocab'])}
 with np.load('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz') as z:lengths=[len(z[f'clip{i:04d}']) for i in range(len(ids))]
 entries=[{'clip_id':i,'frames':n,'text':r.split()} for i,n,r in zip(ids,lengths,refs)]
 with np.load(base.ROOT/'normalization.npz') as z:mean,std=z['mean'],z['std']
 teacher,dc=teacher_for('PHOENIX-2014T');alpha=base.alpha_schedule();write('ordinary/data_audit.json',{'clips':642,'clip_ids':ids,'TRAIN_overlap':0,'DEV_overlap':0,'reference_poses_consumed_by_sampler':False})
 for seed in SEEDS:
  for arm in ARMS:
   folder=ROOT/'ordinary'/f'seed{seed}_{arm}';folder.mkdir(parents=True,exist_ok=True)
   if (folder/'result.json').exists():continue
   model=base.Denoiser(len(stoi)).cuda();model.load_state_dict(torch.load(Q/f'training/seed{seed}/{arm}/epoch20.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True)
   xs=sample_without_reference_pose(model,entries,stoi,mean,std,alpha);cs=[project(x) for x in xs]
   raw=score(teacher,xs,refs,ids,dc);can=score(teacher,cs,refs,ids,dc)
   write(str(folder.relative_to(ROOT)/'result.json'),{'seed':seed,'arm':arm,'raw':raw,'canonical':can,'raw_canonical_comparison_of_same_output':True,'motions_may_differ_across_models':True,'noncanonicity':profiles(xs),**geometry(xs,cs)})
   np.savez_compressed(folder/'poses.npz',**{f'clip{i:04d}':x for i,x in enumerate(xs)})
   print('SECOND_TEST_ORDINARY',seed,arm,raw['scores']['bleu']['bleu4'],can['scores']['bleu']['bleu4'],flush=True);del model;torch.cuda.empty_cache()
 write('ordinary/complete.json',{'conditions':9,'clips':642,'no_fitting':True})

@torch.no_grad()
def real():
 register()
 for corpus in ['How2Sign','CSL-Daily']:
  teacher,dc=teacher_for(corpus);dc.pop('pseudo_gloss_dir',None);dataroot=Path('reports/cvpr_representation_paper_20261011') if corpus=='CSL-Daily' else Path('reports/cvpr_evaluation_confirmation_20261011');dc.update(base_path=str((dataroot/'data/lmdb').resolve()),index_cache=False,max_sgn_length=300)
  data=_build_split(dc,'test',None);entries=[data[i] for i in range(len(data))];ids=[e.sequence for e in entries];refs=[' '.join(e.txt) for e in entries];xs=[e.sgn.numpy().copy() for e in entries]
  audit=read(R/corpus/'data_audit.json');assert not set(ids)&set(audit['train_ids']) and not set(ids)&set(audit['dev_ids'])
  folder=ROOT/'real'/corpus;folder.mkdir(parents=True,exist_ok=True);baseline=score(teacher,[project(x) for x in xs],refs,ids,dc)
  write(f'real/{corpus}/baseline_raw.json',score(teacher,xs,refs,ids,dc));write(f'real/{corpus}/baseline_canonical.json',baseline);write(f'real/{corpus}/data_audit.json',{'clips':len(ids),'clip_ids':ids,'TRAIN_overlap':0,'DEV_overlap':0})
  collator=SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[PAD_TOKEN]
  for arm,seeds in [('text_raw',SEEDS),('motion_raw',[42]),('text_canonical',[42])]:
   for seed in seeds:
    tag=f'seed{seed}_{arm}'
    if (folder/(tag+'.json')).exists():continue
    model=RealLayer(teacher,pad).cuda();model.load_state_dict(torch.load(R/corpus/f'training/seed{seed}/{arm}/layer.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True);model.eval();ys=[];coeff=[]
    for start in range(0,len(entries),32):
     batch=Batch(collator(entries[start:start+32]),txt_pad_index=pad,sgn_dim=240,use_cuda=True);y,c=model(batch.sgn,tokens(batch,pad,arm),batch.sgn_lengths.to(batch.sgn.device))
     for j,n in enumerate(batch.sgn_lengths.tolist()):ys.append(y[j,:n].cpu().numpy().copy());coeff.append(c[j,:n].cpu().numpy().copy())
    raw=score(teacher,ys,refs,ids,dc);can=score(teacher,[project(x) for x in ys],refs,ids,dc);changes=sum(a!=b for a,b in zip(can['hypotheses'],baseline['hypotheses']))
    write(f'real/{corpus}/{tag}.json',{'seed':seed,'arm':arm,'raw':raw,'canonical':can,'canonical_hypothesis_changes':changes,'noncanonicity':profiles(ys),**geometry(xs,ys)})
    np.savez_compressed(folder/(tag+'_poses.npz'),**{f'{k}{i:04d}':v for k,arr in [('attacked',ys),('coefficient',coeff)] for i,v in enumerate(arr)})
    print('SECOND_TEST_REAL',corpus,seed,arm,raw['scores']['bleu']['bleu4'],can['scores']['bleu']['bleu4'],changes,flush=True);del model;torch.cuda.empty_cache()
  del teacher,entries,data;torch.cuda.empty_cache()
 write('real/complete.json',{'conditions':10,'no_fitting':True})

@torch.no_grad()
def budget():
 register();source=read('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/raw_scores.json');ids=source['clip_ids'];refs=source['references']
 with np.load('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz') as z:natural=[z[f'clip{i:04d}'].copy() for i in range(642)]
 xs=[project(x) for x in natural];records=[{'clip_id':id,'frames':len(x),'text':r.split()} for id,x,r in zip(ids,xs,refs)];audit=read(base.ROOT/'data_audit.json');stoi={w:i for i,w in enumerate(audit['vocab'])};g=base.Denoiser(len(stoi)).cuda();g.load_state_dict(torch.load(old.GENERATOR,map_location='cpu',weights_only=False)['state_dict'],strict=True);g.eval()
 teacher,dc=teacher_for('PHOENIX-2014T');baseline=score(teacher,xs,refs,ids,dc);write('budget/baseline.json',baseline);naturalp=profiles(natural);write('budget/natural_generator.json',naturalp)
 for seed in SEEDS:
  for bound in [.01,.05,.1]:
   tag=f'seed{seed}_bound{bound:g}';folder=ROOT/'budget'
   if (folder/(tag+'.json')).exists():continue
   model=BoundedLayer(g,bound).cuda();model.load_state_dict(torch.load(B/f'training/seed{seed}/bound{bound:g}/layer.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True);model.eval();ys=[];coeff=[]
   for start in range(0,len(records),32):
    group=records[start:start+32];n=torch.tensor([e['frames'] for e in group],device='cuda');x=torch.zeros(len(group),int(n.max()),240,device='cuda')
    for j,p in enumerate(xs[start:start+32]):x[j,:len(p)]=torch.from_numpy(p).cuda()
    y,c=model(x,old.token_tensor(group,stoi),n)
    for j,l in enumerate(n.tolist()):ys.append(y[j,:l].cpu().numpy().copy());coeff.append(c[j,:l].cpu().numpy().copy())
   raw=score(teacher,ys,refs,ids,dc);can=score(teacher,[project(x) for x in ys],refs,ids,dc);p=profiles(ys);a=auc(naturalp['clip_RMS_values'],p['clip_RMS_values'])
   write(f'budget/{tag}.json',{'seed':seed,'bound':bound,'raw':raw,'canonical':can,'canonical_hypothesis_changes':sum(a!=b for a,b in zip(can['hypotheses'],baseline['hypotheses'])),'noncanonicity':p,'RMS_detector_best_orientation_AUC':max(a,1-a),**geometry(xs,ys)})
   np.savez_compressed(folder/(tag+'_poses.npz'),**{f'{k}{i:04d}':v for k,arr in [('attacked',ys),('coefficient',coeff)] for i,v in enumerate(arr)})
   print('SECOND_TEST_BOUND',seed,bound,raw['scores']['bleu']['bleu4'],can['scores']['bleu']['bleu4'],p['mean_RMS'],max(a,1-a),flush=True);del model;torch.cuda.empty_cache()
 write('budget/complete.json',{'conditions':9,'clips':642,'no_fitting':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['ordinary','real','budget']);a=p.parse_args();globals()[a.phase]()
