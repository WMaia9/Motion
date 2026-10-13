"""No-fit CSL real-motion layer transfer to the native TRAIN-retrieval replay."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from run_real_motion_layers import RealLayer,tokens
from run_signspark_geometry_dev import _build_split,load_config,project,make_back_translation_model
from run_signspark_gauge_attack import SignCollator,Batch,PAD_TOKEN
from run_learned_confirmation import score
from run_evaluation_heldout import geometry
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_retrieval_transfer_20261012');REAL=Path('reports/cvpr_real_motion_layers_20261012')
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
@torch.no_grad()
def main():
 assert (ROOT/'baseline_complete.json').exists() and (REAL/'manifest_sha256.json').exists() and not (ROOT/'manifest_sha256.json').exists()
 weights=[REAL/f'CSL-Daily/training/seed{s}/text_raw/layer.pt' for s in [42,123,777]]
 inputs=weights+[REAL/'manifest_sha256.json',ROOT/'generated/poses.npz',ROOT/'evaluation/generated_raw.json',ROOT/'evaluation/generated_canonical.json']
 p={'purpose':'no-fit transfer of all three native CSL real-motion source-text layer seeds to public retrieval-generated DEV128 motion','no_training':True,'no_per_clip_optimization':True,'source_caption_is_layer_input':True,'weights_fixed_before_transfer':True,'input_SHA256':{str(p):sha(p) for p in inputs},'source_code_sha256':sha(__file__)}
 if (ROOT/'transfer_protocol.json').exists():assert read(ROOT/'transfer_protocol.json')==p
 else:write('transfer_protocol.json',p)
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 checkpoint=Path('reports/cvpr_representation_paper_20261011/checkpoints/CSL-Daily');teacher=make_back_translation_model(checkpoint);teacher.eval();teacher.do_recognition=False;dc=dict(load_config(checkpoint/'config.yaml')['data']);dc.pop('pseudo_gloss_dir',None);dc.update(base_path=str(Path('reports/cvpr_canonical_readers_20261011/data/lmdb').resolve()),index_cache=False,max_sgn_length=300)
 data=_build_split(dc,'dev',None);lookup={data[i].sequence:data[i] for i in range(len(data))};baseline=read(ROOT/'evaluation/generated_raw.json');canonical=read(ROOT/'evaluation/generated_canonical.json');ids=baseline['clip_ids'];refs=baseline['references'];entries=[]
 with np.load(ROOT/'generated/poses.npz') as z:xs=[z[id].copy() for id in ids]
 for id,x in zip(ids,xs):e=copy.copy(lookup[id]);e.sgn=torch.from_numpy(x);entries.append(e)
 collator=SignCollator(teacher.txt_vocab,None,240);pad=teacher.txt_vocab.stoi[PAD_TOKEN]
 for seed,path in zip([42,123,777],weights):
  tag=f'transfer/seed{seed}_text_raw'
  if (ROOT/(tag+'.json')).exists():continue
  model=RealLayer(teacher,pad).cuda();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True);model.eval();ys=[];coeff=[]
  for start in range(0,len(entries),32):
   batch=Batch(collator(entries[start:start+32]),txt_pad_index=pad,sgn_dim=240,use_cuda=True);y,c=model(batch.sgn,tokens(batch,pad,'text_raw'),batch.sgn_lengths.to(batch.sgn.device))
   for j,n in enumerate(batch.sgn_lengths.tolist()):ys.append(y[j,:n].cpu().numpy().copy());coeff.append(c[j,:n].cpu().numpy().copy())
  raw=score(teacher,ys,refs,ids,dc);can=score(teacher,[project(x) for x in ys],refs,ids,dc)
  write(tag+'.json',{'seed':seed,'raw':raw,'canonical':can,'baseline_raw':baseline['scores']['bleu']['bleu4'],'canonical_hypothesis_changes':sum(a!=b for a,b in zip(can['hypotheses'],canonical['hypotheses'])),**geometry(xs,ys)})
  np.savez_compressed(ROOT/(tag+'_poses.npz'),**{f'{k}{i:04d}':v for k,arr in [('attacked',ys),('coefficient',coeff)] for i,v in enumerate(arr)})
  print('RETRIEVAL_LAYER_TRANSFER',seed,raw['scores']['bleu']['bleu4'],can['scores']['bleu']['bleu4'],flush=True);del model;torch.cuda.empty_cache()
 write('transfer_complete.json',{'conditions':3,'clips':128,'no_fitting':True})
if __name__=='__main__':main()
