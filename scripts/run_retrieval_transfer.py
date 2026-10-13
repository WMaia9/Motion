"""Public TRAIN-dictionary CSL retrieval replay, separate from weak PHX no-KF replay."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch  # Initialize numerical bindings before vendor NumPy-2 pickle aliases.
from scipy.spatial.transform import Rotation
WORKSPACE=Path(__file__).resolve().parents[1];ROOT=WORKSPACE/'reports/cvpr_retrieval_transfer_20261012';REPO=WORKSPACE/'external/SignSparK_Retrieval_20261012'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(WORKSPACE/'scripts'))
from signspark_retrieval.datasets import get_dataset
from signspark_retrieval.dictionary import load_or_build
from signspark_retrieval.lmdb_io import LmdbReader,find_lmdb,write_lmdb
from signspark_retrieval.pipeline import Sentence,read_gloss_file,resolve_oov,stitch_sentences
from signspark_retrieval.signspark import keyframes_for,write_keyframe_file,flip_left_hand
from verify_signspark_studies import sha
OLD=WORKSPACE/'reports/cvpr_motion_consistency_confirmation_20261010'
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def register():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists();corpus='CSL-Daily';base=read(WORKSPACE/'reports/cvpr_canonical_readers_20261011/CSL-Daily/public_raw.json');ids=base['clip_ids'];indices=np.sort(np.random.default_rng(2027).choice(len(ids),128,replace=False));selected=[ids[i] for i in indices]
 inputs=[OLD/'checkpoints'/s/'ema_0.9999_200000.pt' for s in ['body','hand']]+[ROOT/'assets/annotations/csl_iso.train',ROOT/'assets/predictions/CSL-Daily/dev_pres.txt',OLD/'text_encoder/pytorch_model.bin']
 proto={'corpus':corpus,'split':'DEV','subset':128,'selection_seed':2027,'clip_ids':selected,'indices':indices.tolist(),
 'source_glosses':'released native T2G predictions; not ground-truth glosses','dictionary':'TRAIN-only, default medoid and same-signer where available; no DEV pose anchor','keyframes':'sparse first/middle/last from each retrieved TRAIN sign',
 'OOV':'official no-embed exact/fuzzy option; mappings and dropped glosses disclosed','reader':'native public CSL BT, native normalization and decoding',
 'generation':'official released CFM body/hand checkpoint; 50 ODE steps, seed123, no CFG (native default), seq_len304, batch16; keyframes from retrieval',
 'face':'not generated or evaluated; BT input uses only body+hands','scope':'native release pipeline replay, not proof of original-paper BT reproduction; paper GF/SR T2P table has no BT numbers',
 'no_TEST_use':True,'input_SHA256':{str(p):sha(p) for p in inputs},'retrieval_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'source_code_sha256':sha(__file__)}
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==proto,'protocol changed'
 else:write('protocol.json',proto)
 return selected

def prepare():
 selected=register();spec=get_dataset('CSL-Daily');datadir=ROOT/'source_data/lmdb';datadir.mkdir(parents=True,exist_ok=True)
 for split,path in [('train',WORKSPACE/'reports/cvpr_canonical_readers_20261011/data/lmdb/train/CSL-Daily_reopt_train.lmdb'),('dev',WORKSPACE/'reports/cvpr_canonical_readers_20261011/data/lmdb/dev/CSL-Daily_reopt_dev.lmdb')]:
  out=datadir/split;out.mkdir(exist_ok=True);link=out/path.name
  if not link.exists():link.symlink_to(path,target_is_directory=True)
 os.chdir(ROOT)
 dictionary=load_or_build(spec,'train',ROOT/'source_data',ROOT/'assets/annotations',split='dev');assert dictionary.split=='train'
 table=read_gloss_file(ROOT/'assets/predictions/CSL-Daily/dev_pres.txt',spec);sentences=[]
 with LmdbReader(find_lmdb(ROOT/'source_data','CSL-Daily','dev')) as reader:
  for id in selected:
   assert id in table;rec=reader.get(id);sentences.append(Sentence(id=id,text=rec['translation'],glosses=table[id],gt_len=len(rec['segment']),video=id))
 mapping=resolve_oov(sentences,dictionary,spec,embed=False);write('sources/oov_mapping.json',mapping)
 stitched=stitch_sentences(sentences,dictionary,spec,ROOT/'pipeline',split='dev',oov_map=mapping)
 write_keyframe_file(stitched,'dev','sparse',ROOT/'pipeline/keyframes.pt')
 sourcevids={inst.video for occ in dictionary.instances.values() for inst in occ} if hasattr(dictionary,'instances') else set()
 audit={'clips_requested':128,'dictionary_split':dictionary.split,'dictionary_vocab':len(dictionary),'clips_stitched':read(ROOT/'pipeline/summary.json')['stitched'],'TRAIN_dictionary_only':True,'gt_glosses_consumed':False,'DEV_pose_frames_consumed_as_anchors':False,'reference_duration_metadata_only':True,'real_DEV_poses_read_for_baseline_only':True,'OOV_no_embedding':True}
 write('sources/pipeline_audit.json',audit)
 for stream in ['body','hand']:
  dest=ROOT/'checkpoints'/stream;dest.mkdir(parents=True,exist_ok=True);path=dest/'ema_0.9999_200000.pt'
  if not path.exists():os.link(OLD/'checkpoints'/stream/path.name,path)
 print('RETRIEVAL_PIPELINE_PREPARED',json.dumps(audit),flush=True)


def generate():
 register();assert (ROOT/'sources/pipeline_audit.json').exists(),'preparation incomplete';modelrepo=WORKSPACE/'external/SignSparK';stitched=ROOT/'pipeline/lmdb/dev/CSL-Daily_stitch-train_dev.lmdb';assert stitched.exists()
 for stream in ['body','hand']:
  folder=ROOT/'generated';folder.mkdir(exist_ok=True);target=folder/(stream+'.npy')
  if target.exists():continue
  args=[str(WORKSPACE/'.venv_signspark_confirmation/bin/python'),'-u',str(WORKSPACE/'scripts/sample_signspark_confirmation.py'),f'data_v2={stream}','data_v2/common@common=inference',f'eval.model_path={ROOT}/checkpoints/{stream}/ema_0.9999_200000.pt',f'base_path={ROOT}/pipeline/lmdb','dev_data=[CSL-Daily_stitch-train]','eval.split=dev','eval.max_samples=-1','eval.candidate_num=1','eval.ode_stepnum=50','eval.batch_size=16','eval.seed=123','seq_len=304','eval.seq_len=304','flip_left_hand=True',f'custom_keyframe_file={ROOT}/pipeline/keyframes.pt',f'eval.note=CSL_DEV_retrieval_{stream}',f'text_enc_ver={OLD}/text_encoder',f'hydra.run.dir={ROOT}/hydra/{stream}']
  log=ROOT/'generated'/(stream+'.log')
  with log.open('w') as f:subprocess.run(args,cwd=modelrepo,stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'WANDB_MODE':'disabled','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4'})
  match=re.findall(r'### Written the decoded output to (\S+\.npy)',log.read_text());assert match
  output=Path(match[-1]);assert ROOT in output.resolve().parents,'sampling wrote outside new report';shutil.copy2(output,target)
  print('PUBLIC_RETRIEVAL_GENERATED',stream,sha(target),flush=True)


def score():
 register()
 import torch
 from run_signspark_geometry_dev import make_back_translation_model,load_config,project,back_translate,score_translations
 from audit_reviewed_manuscript_evidence import metric
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 checkpoint=WORKSPACE/'reports/cvpr_representation_paper_20261011/checkpoints/CSL-Daily';teacher=make_back_translation_model(checkpoint);teacher.eval();teacher.do_recognition=False;dc=load_config(checkpoint/'config.yaml')['data']
 body=np.load(ROOT/'generated/body.npy',allow_pickle=True).item();hand=np.load(ROOT/'generated/hand.npy',allow_pickle=True).item();bn=[str(v) for v in body['video_names'][0]];hn=[str(v) for v in hand['video_names'][0]];assert hn[::2]==bn and hn[1::2]==bn
 selected=read(ROOT/'protocol.json')['clip_ids'];poses={};refs={};real={};stitchedposes={}
 with LmdbReader(ROOT/'pipeline/lmdb/dev/CSL-Daily_stitch-train_dev.lmdb') as reader:
  for i,id in enumerate(bn):
   rec=reader.get(id);n=min(int(body['lengths'][0][i]),int(hand['lengths'][0][2*i]));b=body['pred_poses'][0][i,:,:n].T.astype(np.float32);left=flip_left_hand(hand['pred_poses'][0][2*i,:,:n].T);right=hand['pred_poses'][0][2*i+1,:,:n].T.astype(np.float32);x=np.concatenate((b,left,right),1);assert x.shape[1]==240
   kf=[k for k in keyframes_for(rec['segment'],'sparse') if k<n];orig=np.concatenate((rec['body_features'][:n,66:],rec['left_features'][:n,:90],rec['right_features'][:n,:90]),1);x[kf]=orig[kf];poses[id]=x[:300];stitchedposes[id]=orig[:300]
 with LmdbReader(find_lmdb(ROOT/'source_data','CSL-Daily','dev')) as reader:
  for id in selected:
   rec=reader.get(id);refs[id]=rec['translation'];real[id]=np.concatenate((rec['body_features'][:,66:],rec['left_features'][:,:90],rec['right_features'][:,:90]),1)[:300]
 valid=[id for id in selected if id in poses];references=[refs[id] for id in selected]
 results={}
 for name,values in [('generated_raw',poses),('generated_canonical',{id:project(x) for id,x in poses.items()}),('stitched_raw',stitchedposes),('real_raw',real)]:
  vid=[id for id in selected if id in values];h=back_translate(teacher,[torch.from_numpy(values[id]) for id in vid],batch_size=32);lookup=dict(zip(vid,h));hyp=[lookup.get(id,'') for id in selected];scores=score_translations(references,hyp,dc['version'],dc['level']);assert abs(metric(references,hyp,True)-scores['bleu']['bleu4'])<1e-10
  r={'clip_ids':selected,'references':references,'hypotheses':hyp,'scores':scores,'missing_generated_clips_counted_as_empty':len(selected)-len(vid)};write('evaluation/'+name+'.json',r);results[name]=scores['bleu']['bleu4'];print('RETRIEVAL_BASELINE_SCORED',name,scores['bleu']['bleu4'],flush=True)
 np.savez_compressed(ROOT/'generated/poses.npz',**{id:x for id,x in poses.items()});write('baseline_complete.json',{'requested':128,'generated':len(valid),'scores':results,'no_true_DEV_keyframes':True,'source_release_precludes_original_paper_BT_reproduction_claim':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','generate','score']);a=p.parse_args();globals()[a.phase]()
