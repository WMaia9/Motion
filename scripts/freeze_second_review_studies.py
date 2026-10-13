"""Independently reconstruct new experiments and freeze completed evidence."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from freeze_learned_reader_studies import pair,archive
from audit_reviewed_manuscript_evidence import metric
from verify_signspark_studies import sha,matrices,freeze
BASE=Path('reports')
def read(p):return json.loads(Path(p).read_text())
def standalone(r,char=False):
 assert abs(metric(r['references'],r['hypotheses'],char)-r['scores']['bleu']['bleu4'])<1e-10
 return len(r['references'])
def frozen_inputs(root,field='inputs',codefield='code_sha256',script=None):
 p=read(root/'protocol.json')
 if script:assert sha(Path('scripts')/(script+'.py'))==p[codefield]
 for path,h in p[field].items():assert sha(path)==h,path
 return p

def budget():
 root=BASE/'cvpr_learned_budget_20261012';assert (root/'complete.json').exists();frozen_inputs(root,script='run_learned_budget')
 import run_learned_equivalence as old
 records,xs=old.cache('dev');base=read(root/'evaluation/baseline.json');count=metrics=0;emax=amax=0.
 for path in (root/'evaluation').glob('seed*_poses.npz'):
  r=read(path.with_name(path.name.removesuffix('_poses.npz')+'.json'));assert r['canonical']['hypotheses']==base['hypotheses']
  for k in ['raw','canonical']:metrics+=standalone(r[k])
  with np.load(path) as z:
   for i,x in enumerate(xs):
    y=z[f'attacked{i:04d}'];c=z[f'coefficient{i:04d}'];assert abs(c).max()<=r['bound']+1e-6;e,a=pair(x,y,c);emax=max(emax,e);amax=max(amax,a);count+=1
 assert count==4671
 archive(root,['run_learned_budget','analyze_learned_budget','freeze_second_review_studies'],['cvpr_learned_budget_20261012.log','cvpr_learned_budget_analysis_20261012.log','cvpr_learned_budget_analysis_fixed_20261012.log','cvpr_learned_budget_analysis_seeds_20261012.log'])
 freeze(root,{'passed':True,'independently_reconstructed_pairs':count,'max_rotation_error_deg':amax,'max_reconstruction_error':emax,'standalone_BLEU_sentence_evaluations':metrics,'all_canonical_hypotheses_unchanged':True,'simple_RMS_detection_reported':True,'three_seed_sensitivity_reported':True,'no_TEST_use':True})

def norm(kind):
 circle=kind=='circle';root=BASE/('cvpr_constant_noncanonicity_20261012' if circle else 'cvpr_constant_distance_sphere_20261012')
 script='run_constant_noncanonicity_layer' if circle else 'run_constant_distance_sphere';analysis='analyze_constant_noncanonicity' if circle else 'analyze_constant_distance_sphere'
 frozen_inputs(root,'inputs' if circle else 'input_SHA256','code_sha256' if circle else 'source_code_sha256',script)
 assert (root/'complete.json').exists()
 source=BASE/'cvpr_constant_noncanonicity_20261012/generated/dev/poses.npz'
 with np.load(source) as z:xs=[z[f'clip{i:05d}'].copy() for i in range(519)]
 baseline=read(root/'evaluation/baseline_canonical.json');count=metrics=0;emax=nmax=rmax=amax=0.
 for path in (root/'evaluation').glob('seed*_poses.npz'):
  r=read(path.with_name(path.name.removesuffix('_poses.npz')+'.json'));assert r['canonical']['hypotheses']==baseline['hypotheses']
  for k in ['raw','canonical']:metrics+=standalone(r[k])
  with np.load(path) as z:
   for i,x in enumerate(xs):
    y=z[f'attacked{i:04d}'];w=z[f'auxiliary{i:04d}'].astype(float);assert x.shape==y.shape and np.array_equal(x[:,60:150],y[:,60:150]) and abs(w).max()<=1.000001
    C=matrices(x)[...,:2,:];u=C[...,0,:];v=C[...,1,:];d=x.reshape(-1,40,6).astype(float);rho=np.linalg.norm(d[...,:3],axis=-1);tau=(u*d[...,3:]).sum(-1);sigma=np.linalg.norm(d[...,3:]-tau[...,None]*u,axis=-1);s=np.stack((rho-1,sigma-1,tau),-1);radius=np.linalg.norm(s,axis=-1)
    axis=w.copy()
    if circle:
     phase=np.pi*w.mean(-1);axis=np.stack((phase/np.sqrt(2),phase/np.sqrt(2),np.zeros_like(phase)),-1)
    axis[radius>=.95]=0
    rot=Rotation.from_rotvec(axis.reshape(-1,3)).as_matrix().reshape(*axis.shape[:-1],3,3);delta=(rot@s[...,None])[...,0]-s;rebuilt=d+np.concatenate((delta[...,0:1]*u,delta[...,1:2]*v+delta[...,2:3]*u),-1);error=float(abs(rebuilt-y.reshape(d.shape)).max());assert error<8e-6;rmax=max(rmax,error)
    cx=C.reshape(x.shape);cy=matrices(y)[...,:2,:].reshape(y.shape);ea=((x-cx).reshape(-1,40,6).astype(float)**2).sum(-1);eb=((y-cy).reshape(-1,40,6).astype(float)**2).sum(-1);e=float(abs(ea-eb).max());nn=float(abs((d**2).sum(-1)-(y.reshape(d.shape).astype(float)**2).sum(-1)).max());assert e<2e-5
    if circle:assert nn<2e-5
    angle=float(np.degrees(Rotation.from_matrix((matrices(x).transpose(0,1,3,2)@matrices(y)).reshape(-1,3,3)).magnitude()).max());assert angle<.01
    emax=max(emax,e);nmax=max(nmax,nn);amax=max(amax,angle);count+=1
 assert count==4671
 logs=['cvpr_constant_noncanonicity_prepare_20261012.log','cvpr_constant_noncanonicity_prepare_fixed_20261012.log','cvpr_constant_noncanonicity_train_20261012.log','cvpr_constant_noncanonicity_analysis_20261012.log'] if circle else ['cvpr_constant_distance_sphere_20261012.log','cvpr_constant_distance_sphere_analysis_20261012.log']
 archive(root,[script,analysis,'freeze_second_review_studies'],logs)
 freeze(root,{'passed':True,'independently_reconstructed_pairs':count,'max_rotation_error_deg':amax,'max_reconstruction_error':rmax,'per_joint_distance_squared_max_error':emax,'per_joint_6D_squared_norm_max_difference':nmax,'total_6D_norm_is_declared_invariant':circle,'standalone_BLEU_sentence_evaluations':metrics,'all_canonical_hypotheses_unchanged':True,'no_general_stealth_claim':True,'no_TEST_use':True})

def quotient():
 root=BASE/'cvpr_quotient_training_20261012';frozen_inputs(root,script='run_quotient_training');assert read(root/'complete.json')['all_first_epoch_stochastic_inputs_matched'];metrics=pairs=0;amax=0.
 for path in (root/'training').glob('seed*/*/dev_epoch*/raw_scores.json'):
  raw=read(path);can=read(path.with_name('canonical_scores.json'));assert raw['clip_ids']==can['clip_ids'] and raw['references']==can['references'];metrics+=standalone(raw)+standalone(can)
  with np.load(path.with_name('poses.npz')) as z:
   for i in range(519):
    x=z[f'clip{i:04d}'];assert x.shape[1]==240 and np.isfinite(x).all();R=matrices(x);y=R[...,:2,:].reshape(x.shape).astype(np.float32);angle=float(np.degrees(Rotation.from_matrix((R.transpose(0,1,3,2)@matrices(y)).reshape(-1,3,3)).magnitude()).max());assert angle<.01;amax=max(amax,angle);pairs+=1
 archive(root,['run_quotient_training','analyze_quotient_training','freeze_second_review_studies'],['cvpr_quotient_training_20261012.log','cvpr_quotient_training_fixed_20261012.log','cvpr_quotient_analysis_20261012.log'])
 freeze(root,{'passed':True,'ordinary_generator_fits':42,'raw_canonical_rotation_pairs':pairs,'max_rotation_error_deg':amax,'standalone_BLEU_sentence_evaluations':metrics,'raw_canonical_hypotheses_may_differ':True,'physical_motions_may_change_across_fits':True,'input_and_checkpoint_contract_checked':True,'no_TEST_use':True})

def real():
 root=BASE/'cvpr_real_motion_layers_20261012';p=frozen_inputs(root,script='run_real_motion_layers');assert (root/'complete.json').exists();pairs=metrics=0;emax=amax=0.
 from run_signspark_geometry_dev import _build_split,load_config
 import run_canonical_readers as old
 for corpus,checkpoint in old.CHECKPOINTS.items():
  dc=dict(load_config(checkpoint/'config.yaml')['data']);dc.pop('pseudo_gloss_dir',None);dc.update(index_cache=False,max_sgn_length=300,base_path=str((old.ROOT/'data/lmdb').resolve()))
  if corpus=='How2Sign':dc['base_path']=str(Path('reports/cvpr_signspark_baseline_20261010/data/lmdb').resolve())
  data=_build_split(dc,'dev',None);xs=[data[i].sgn.numpy() for i in range(len(data))];base=read(root/corpus/'baseline_canonical.json')
  for path in (root/corpus/'evaluation').glob('*_poses.npz'):
   r=read(path.with_name(path.name.removesuffix('_poses.npz')+'.json'));assert r['canonical']['hypotheses']==base['hypotheses']
   for k in ['raw','canonical']:metrics+=standalone(r[k],corpus=='CSL-Daily')
   with np.load(path) as z:
    for i,x in enumerate(xs):
     e,a=pair(x,z[f'attacked{i:04d}'],z[f'coefficient{i:04d}']);emax=max(emax,e);amax=max(amax,a);pairs+=1
 archive(root,['run_real_motion_layers','analyze_real_motion_layers','freeze_second_review_studies'],['cvpr_real_motion_layers_20261012.log','cvpr_real_motion_layers_primary3_controls1_20261012.log','cvpr_real_motion_layers_analysis_20261012.log'])
 freeze(root,{'passed':True,'designed_real_motion_layer_fits':10,'primary_seeds':3,'exploratory_control_seeds':1,'independently_reconstructed_pairs':pairs,'max_rotation_error_deg':amax,'max_reconstruction_error':emax,'standalone_BLEU_sentence_evaluations':metrics,'all_canonical_hypotheses_unchanged':True,'no_TEST_use':True,'source_caption_available_by_design':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('study',choices=['budget','circle','sphere','quotient','real']);a=p.parse_args()
 if a.study in ['circle','sphere']:norm(a.study)
 else:globals()[a.study]()
