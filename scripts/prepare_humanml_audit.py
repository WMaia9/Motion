"""Prepare a fixed DEV/TEST audit subset before any evaluator scores are read."""
import json,zipfile,shutil,subprocess
from pathlib import Path
import numpy as np
from humanml_audit_common import R,A,DATA,sha,read,write

def main():
 assert not (R/'manifest_sha256.json').exists()
 assert not (R/'protocol.json').exists(),'protocol already registered'
 DATA.mkdir(parents=True,exist_ok=True)
 source=Path('external/HumanML3D/HumanML3D')
 for name in ['Mean.npy','Std.npy','train.txt','val.txt','test.txt','all.txt']:shutil.copy2(source/name,DATA/name)
 with zipfile.ZipFile(source/'texts.zip') as z:z.extractall(DATA)
 if not (DATA/'new_joint_vecs/012314.npy').is_file():
  subprocess.run(['unrar','x','-idq','-o-','models/humanml3d_audit/new_joint_vecs.rar',str(DATA)+'/'],check=True)
 example=source/'new_joint_vecs/012314.npy';real=DATA/'new_joint_vecs/012314.npy'
 x0,x1=np.load(example),np.load(real)
 assert x0.shape==x1.shape and float(np.max(abs(x0-x1)))<2.5e-5,'mirror numerically incompatible with author example'
 write(R/'mirror_example_verification.json',{'byte_identical':sha(example)==sha(real),'source_dtype':str(x0.dtype),'mirror_dtype':str(x1.dtype),'max_absolute_difference':float(np.max(abs(x0-x1))),'RMS_difference':float(np.sqrt(np.mean((x0-x1)**2))),'author_example_sha256':sha(example),'mirror_example_sha256':sha(real),'not_a_full_archive_identity_proof':True})
 rng=np.random.default_rng(20261012);selections={};eligible_counts={}
 for split,count in [('val',128),('test',1024)]:
  candidates=[]
  for name in (source/(split+'.txt')).read_text().splitlines():
   path=DATA/'new_joint_vecs'/(name+'.npy')
   if not path.is_file():continue
   x=np.load(path,allow_pickle=False)
   if not (40<=len(x)<200 and x.shape[1]==263 and np.isfinite(x).all()):continue
   lines=[]
   for line in (DATA/'texts'/(name+'.txt')).read_text().splitlines():
    caption,tokens,start,end=line.split('#')
    if float(start)==0 and float(end)==0:lines.append((caption,tokens))
   if not lines:continue
   caption,tokens=lines[int(rng.integers(len(lines)))];length=(len(x)//4)*4;start=int(rng.integers(len(x)-length+1))
   candidates.append({'clip_id':name,'caption':caption,'tokens':tokens,'frames':length,'start_frame':start,'motion_sha256':sha(path),'text_sha256':sha(DATA/'texts'/(name+'.txt'))})
  eligible_counts[split]=len(candidates);assert len(candidates)>=count
  idx=rng.choice(len(candidates),count,replace=False);selections[split]=[candidates[int(i)] for i in idx];write(R/f'{split}_selection.json',selections[split])
  raw=np.zeros((count,196,263),np.float32)
  for i,item in enumerate(selections[split]):
   x=np.load(DATA/'new_joint_vecs'/(item['clip_id']+'.npy'));n=item['frames'];s=item['start_frame'];raw[i,:n]=x[s:s+n]
  np.savez_compressed(R/f'{split}_real.npz',raw=raw,lengths=np.array([x['frames'] for x in selections[split]]))
 inputs=[A/'t2m/text_mot_match/model/finest.tar',A/'t2m/Comp_v6_KLD01/meta/mean.npy',A/'t2m/Comp_v6_KLD01/meta/std.npy',A/'humanml_enc_512_50steps/model000750000.pt',A/'humanml_enc_512_50steps/args.json',source/'Mean.npy',source/'Std.npy',source/'test.txt',source/'val.txt']
 for split in selections:inputs +=[R/f'{split}_selection.json',R/f'{split}_real.npz']
 fixed=['identity','canonical','global_plus005','global_minus005','global_plus010','global_minus010','global_plus035','global_minus035','random035_seed42']
 write(R/'protocol.json',{'scope':'Frozen evaluator consistency audit, not a new generator or leaderboard reproduction. Previously unused public HumanML3D subset. No training.',
  'dataset':'HumanML3D263','rotation_slice':[67,193],'rotation_joint_count':21,'evaluator_channels':259,'column_rotation_convention':True,
  'real_example_numerically_matches_author_within_2_5e_minus5':True,'real_example_byte_identical':False,'processed_motion_source':'third-party public HF mirror with LFS SHA256, same byte count as MDM author-linked Drive; author example numerically checked with float32/float64 differences, not byte identical',
  'eligible_counts':eligible_counts,'clips':{'val':128,'test':1024},'selection_seed':20261012,'selection':'uniform without replacement from full-caption clips with 40<=frames<200; fixed first-caption draw and multiple-of-four crop; both mirrored and native samples possible',
  'generation':{'model':'official MDM50step encoder checkpoint','steps':50,'CFG':2.5,'seed':42,'source_text_and_duration':True,'reference_poses_supplied':False,'batch_size':32},
  'fixed_conditions':fixed,'global_coefficients':'(b,-b,b), signed b; alpha=exp(c0), beta=exp(c1), eta=c2 in UNNORMALIZED 6D coordinates',
  'adaptive_conditions':['target_caption','shifted_caption','canonical_reader_shifted_caption'],
  'adaptive':'per-clip Adam lr0.1, 80 fixed final steps, zero initialized; c=0.35*tanh(parameters), masked padded frames; squared text/motion embedding distance, no model fitting',
  'target_access':'source-caption target embeddings and frozen evaluator gradients; no claim of target-blind or spontaneous optimization',
  'TEST_rule':'same fixed complete conditions after DEV, no selection of best bound/checkpoint; no TEST optimization of global coefficients',
  'primary':'paired top1 R-precision change averaged five fixed 32-candidate permutations; MM-distance paired secondary; FID descriptive against fixed real reference embeddings',
  'R_precision_seeds':[11,97,909,1337,2027],'grouping':'native clip ID stripping leading M for mirror pairs; not verified actor IDs',
  'uncertainty':'10000 paired source-group bootstrap draws on R1 and MM-distance; family adjustment across 6 source/adaptive target endpoints; fixed transformations descriptive consistency controls',
  'invariance':'native continuous6D converter plus independent float64 reconstruction, root/positions/velocity/contact unchanged bitwise, native recover_from_ric unchanged, canonicalized embeddings unchanged',
  'inputs_sha256':{str(p):sha(p) for p in inputs},'scientific_sources_sha256':{str(p):sha(p) for p in [Path(__file__),Path('scripts/humanml_audit_common.py'),Path('scripts/run_humanml_audit.py'),Path('scripts/generate_humanml_mdm.py')]},
  'external_revisions':{str(p):subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip() for p in [Path('external/MDM'),Path('external/HumanML3D_Evaluator'),Path('external/HumanML3D'),Path('external/CLIP')]}})
 print('REGISTERED',eligible_counts,flush=True)
if __name__=='__main__':main()
