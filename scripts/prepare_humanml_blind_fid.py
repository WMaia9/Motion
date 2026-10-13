"""Register a caption-blind FID audit on uninspected HumanML3D test groups."""
import json,shutil
from pathlib import Path
import numpy as np
from humanml_audit_common import R as PREVIOUS,A,DATA,read,write,sha
R=Path('reports/cvpr_humanml3d_blind_fid_20261012')
def main():
 assert not (R/'protocol.json').exists();R.mkdir(exist_ok=True)
 # Train-distribution targeting is fixed before any new TEST metric is read.
 rng=np.random.default_rng(20261013);used={x['clip_id'].lstrip('M') for x in read(PREVIOUS/'test_selection.json')};selections={}
 for split,count in [('train',2048),('test',960)]:
  candidates=[]
  for name in (DATA/(split+'.txt')).read_text().splitlines():
   if split=='test' and name.lstrip('M') in used:continue
   path=DATA/'new_joint_vecs'/(name+'.npy')
   if not path.exists():continue
   x=np.load(path,allow_pickle=False)
   if not (40<=len(x)<200 and x.shape[1]==263 and np.isfinite(x).all()):continue
   lines=[]
   for line in (DATA/'texts'/(name+'.txt')).read_text().splitlines():
    cap,tok,start,end=line.split('#')
    if float(start)==float(end)==0.:lines.append((cap,tok))
   if not lines:continue
   cap,tok=lines[int(rng.integers(len(lines)))];n=(len(x)//4)*4;s=int(rng.integers(len(x)-n+1))
   candidates.append({'clip_id':name,'caption':cap,'tokens':tok,'frames':n,'start_frame':s,'motion_sha256':sha(path)})
  assert len(candidates)>=count,(split,len(candidates));idx=rng.choice(len(candidates),count,replace=False);selection=[candidates[int(i)] for i in idx];selections[split]=selection
  write(R/f'{split}_selection.json',selection);raw=np.zeros((count,196,263),np.float32)
  for i,item in enumerate(selection):
   x=np.load(DATA/'new_joint_vecs'/(item['clip_id']+'.npy'));s=item['start_frame'];n=item['frames'];raw[i,:n]=x[s:s+n]
  np.savez_compressed(R/f'{split}_real.npz',raw=raw,lengths=np.array([x['frames'] for x in selection]))
 train={x['clip_id'].lstrip('M') for x in selections['train']};test={x['clip_id'].lstrip('M') for x in selections['test']};assert not train&test and not test&used
 # Reuse the byte-identical native sampling producer on a new output root.
 # Its module constant is reassigned in this driver, not in the frozen source.
 import generate_humanml_mdm as generator
 generator.ROOT=R
 inputs=[A/'t2m/text_mot_match/model/finest.tar',A/'humanml_enc_512_50steps/model000750000.pt',A/'t2m/Comp_v6_KLD01/meta/mean.npy',A/'t2m/Comp_v6_KLD01/meta/std.npy',PREVIOUS/'protocol.json',PREVIOUS/'test_selection.json',R/'train_selection.json',R/'test_selection.json',R/'train_real.npz',R/'test_real.npz']
 write(R/'protocol.json',{'scope':'Prospective caption-blind distribution attack after earlier target-aware HumanML3D study; fresh TEST source groups. No model fitting.',
  'TRAIN_reference_clips':2048,'TEST_clips':960,'selection_seed':20261013,'TEST_group_overlap_with_previous_audit':0,'TRAIN_TEST_group_overlap':0,
  'generation':{'model':'official MDM50 checkpoint','steps':50,'CFG':2.5,'seed':42,'batch':32,'source_caption_supplied_only_to_generator':True,'reference_poses_not_supplied':True},
  'attack_access':'Generated motion arrays, frozen evaluator gradients and 2048 TRAIN motion embeddings. No captions, tokens, text embeddings, TEST real poses or TEST real embeddings accessed by attack.',
  'target_map':'Gaussian optimal-transport affine map between generated-output and TRAIN-reference embedding means/covariances; fixed ridge 1e-4 * trace(cov)/512; no TEST reference moments.',
  'conditions':['identity','canonical','blind_OT_bound010','blind_OT_bound035','canonical_reader_OT_bound035','random_bound035_seed42'],
  'optimizer':'Zero initialized per-clip coefficients, c=b*tanh(q), 80 Adam steps lr0.1, fixed final iterate; no FID-based iterate selection.',
  'primary':'FID against withheld TEST real embeddings after attack arrays and hashes are saved. Two bound endpoints, descriptive dataset-level FID (no calibrated fitting-seed CI).',
  'secondary':'R1/MM actual captions only after attack; canonicalized embedding invariance, position and rotation controls.',
  'generator_and_attack_frozen_without_TEST_score_selection':True,'not_natural_generator_training':True,'not_general_detector_evasion':True,
  'inputs_sha256':{str(x):sha(x) for x in inputs},'scientific_sources_sha256':{str(x):sha(x) for x in [Path(__file__),Path('scripts/run_humanml_blind_fid.py'),Path('scripts/generate_humanml_mdm.py'),Path('scripts/humanml_audit_common.py')]}})
 import sys
 sys.argv=['generate_humanml_mdm.py','test'];generator.main()
 print('BLIND_FID_REGISTERED_AND_GENERATED',flush=True)
if __name__=='__main__':main()
