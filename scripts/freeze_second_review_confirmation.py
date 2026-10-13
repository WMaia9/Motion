"""Independent geometry, metric and frozen-input checks for final experiments."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from freeze_learned_reader_studies import pair,archive
from verify_signspark_studies import sha,matrices,freeze
from audit_reviewed_manuscript_evidence import metric
def read(p):return json.loads(Path(p).read_text())
def inputs(root,name='protocol.json',key='input_SHA256',codekey='source_code_sha256',script=None):
 p=read(root/name)
 for path,h in p[key].items():assert sha(path)==h,path
 if script:assert sha(Path('scripts')/(script+'.py'))==p[codekey]
def standalone(r,char=False):
 assert abs(metric(r['references'],r['hypotheses'],char)-r['scores']['bleu']['bleu4'])<1e-10
 return len(r['references'])
def angle(x,y):
 r=matrices(x).transpose(0,1,3,2)@matrices(y);a=float(np.degrees(Rotation.from_matrix(r.reshape(-1,3,3)).magnitude()).max());assert a<.01;return a
def confirmation():
 root=Path('reports/cvpr_second_review_confirmation_20261012');inputs(root,script='run_second_review_confirmation');assert (root/'analysis.json').exists();count=metrics=changes=0;amax=emax=0.
 for p in (root/'ordinary').glob('seed*/result.json'):
  r=read(p);metrics+=standalone(r['raw'])+standalone(r['canonical'])
  with np.load(p.with_name('poses.npz')) as z:
   for i in range(642):
    x=z[f'clip{i:04d}'];y=matrices(x)[...,:2,:].reshape(x.shape).astype(np.float32);amax=max(amax,angle(x,y));count+=1
 from run_signspark_geometry_dev import _build_split,load_config
 for c in ['How2Sign','CSL-Daily']:
  folder=root/'real'/c;baseline=read(folder/'baseline_canonical.json');checkpoint=Path('reports/cvpr_representation_paper_20261011/checkpoints')/c if c=='CSL-Daily' else Path('reports/cvpr_signspark_baseline_20261010/checkpoints')/c;dc=dict(load_config(checkpoint/'config.yaml')['data']);dc.pop('pseudo_gloss_dir',None);datafolder=Path('reports/cvpr_representation_paper_20261011') if c=='CSL-Daily' else Path('reports/cvpr_evaluation_confirmation_20261011');dc.update(base_path=str((datafolder/'data/lmdb').resolve()),index_cache=False,max_sgn_length=300);data=_build_split(dc,'test',None);xs=[data[i].sgn.numpy() for i in range(len(data))]
  for p in folder.glob('seed*.json'):
   r=read(p);changes+=r['canonical_hypothesis_changes'];metrics+=standalone(r['raw'],c=='CSL-Daily')+standalone(r['canonical'],c=='CSL-Daily')
   assert r['canonical']['hypotheses']==baseline['hypotheses']
   with np.load(p.with_name(p.stem+'_poses.npz')) as z:
    for i,x in enumerate(xs):
     e,a=pair(x,z[f'attacked{i:04d}'],z[f'coefficient{i:04d}']);emax=max(emax,e);amax=max(amax,a);count+=1
 with np.load('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz') as z:xs=[matrices(z[f'clip{i:04d}'])[...,:2,:].reshape(z[f'clip{i:04d}'].shape).astype(np.float32) for i in range(642)]
 baseline=read(root/'budget/baseline.json')
 for p in (root/'budget').glob('seed*.json'):
  r=read(p);metrics+=standalone(r['raw'])+standalone(r['canonical']);assert r['canonical']['hypotheses']==baseline['hypotheses'];changes+=r['canonical_hypothesis_changes']
  with np.load(p.with_name(p.stem+'_poses.npz')) as z:
   for i,x in enumerate(xs):
    c=z[f'coefficient{i:04d}'];assert abs(c).max()<=r['bound']+1e-6;e,a=pair(x,z[f'attacked{i:04d}'],c);emax=max(emax,e);amax=max(amax,a);count+=1
 assert count==28281,(count,'expected9*642+5*(2183+1162)+9*642')
 archive(root,['run_second_review_confirmation','analyze_second_review_results','freeze_second_review_confirmation'],['cvpr_second_TEST_ordinary_20261012.log','cvpr_second_TEST_real_20261012.log','cvpr_second_TEST_budget_20261012.log','cvpr_second_TEST_analysis_20261012.log'])
 freeze(root,{'passed':True,'conditions':28,'independently_verified_rotation_pairs':count,'max_rotation_error_deg':amax,'max_layer_reconstruction_error':emax,'canonical_layer_hypothesis_changes':changes,'standalone_BLEU_sentence_evaluations':metrics,'registered_weight_hashes_unchanged':True,'no_TEST_fitting_or_selection':True,'ordinary_generator_physical_outputs_may_differ_between_arms':True})
def retrieval():
 root=Path('reports/cvpr_retrieval_transfer_20261012');inputs(root,script='run_retrieval_transfer');inputs(root,'transfer_protocol.json',script='confirm_retrieval_layer_transfer');assert (root/'analysis.json').exists();base=read(root/'evaluation/generated_canonical.json');ids=base['clip_ids'];count=metrics=0;amax=emax=0.
 for p in (root/'evaluation').glob('*.json'):metrics+=standalone(read(p),True)
 with np.load(root/'generated/poses.npz') as z:xs=[z[id].copy() for id in ids]
 for p in (root/'transfer').glob('seed*.json'):
  r=read(p);metrics+=standalone(r['raw'],True)+standalone(r['canonical'],True);assert r['canonical']['hypotheses']==base['hypotheses']
  with np.load(p.with_name(p.stem+'_poses.npz')) as z:
   for i,x in enumerate(xs):e,a=pair(x,z[f'attacked{i:04d}'],z[f'coefficient{i:04d}']);emax=max(emax,e);amax=max(amax,a);count+=1
 provenance=[Path('reports/cvpr_canonical_readers_20261011/manifest_sha256.json')]+[Path('reports/cvpr_canonical_readers_20261011/data/lmdb')/s/f'CSL-Daily_reopt_{s}.lmdb/data.mdb' for s in ['train','dev']]
 (root/'sources/dataset_byte_provenance.json').write_text(json.dumps({'hashes':{str(p):sha(p) for p in provenance},'timing':'read-only post-replay provenance check against frozen native data; not a new prescoring registration'},indent=2)+'\n')
 archive(root,['run_retrieval_transfer','confirm_retrieval_layer_transfer','analyze_second_review_results','freeze_second_review_confirmation'],['cvpr_retrieval_prepare_20261012.log','cvpr_retrieval_prepare_fixed_20261012.log','cvpr_retrieval_generate_20261012.log','cvpr_retrieval_generate_fixed_20261012.log','cvpr_retrieval_score_20261012.log','cvpr_retrieval_layer_transfer_20261012.log','cvpr_retrieval_layer_analysis_20261012.log'])
 freeze(root,{'passed':True,'independently_reconstructed_layer_pairs':count,'max_rotation_error_deg':amax,'max_reconstruction_error':emax,'standalone_BLEU_sentence_evaluations':metrics,'canonical_hypothesis_changes':0,'native_retrieval_not_SOTA_reproduction':True,'no_TEST_use':True})
def transplant():
 root=Path('reports/cvpr_ordinary_transplant_20261012');inputs(root,script='run_ordinary_coordinate_transplant');assert (root/'analysis.json').exists();count=metrics=changes=0;amax=emax=0.
 for p in (root/'evaluation').glob('seed*.json'):
  r=read(p);metrics+=standalone(r['raw'])+standalone(r['canonical']);changes+=r['canonical_hypothesis_changes']
  with np.load(p.with_name(p.stem+'_poses.npz')) as z:
   for i in range(519):
    x=z[f'source{i:04d}'];alternate=z[f'alternate{i:04d}'];y=z[f'hybrid{i:04d}'];C=matrices(x)[...,:2,:];D=matrices(alternate)[...,:2,:];d=alternate.reshape(-1,40,6).astype(float);rho=np.linalg.norm(d[...,:3],axis=-1)[...,None];tau=(D[...,0,:]*d[...,3:]).sum(-1)[...,None];sigma=np.linalg.norm(d[...,3:]-tau*D[...,0,:],axis=-1)[...,None];rebuilt=np.concatenate((rho*C[...,0,:],sigma*C[...,1,:]+tau*C[...,0,:]),-1).reshape(x.shape);rebuilt[:,60:150]=x[:,60:150];e=float(abs(rebuilt-y).max());assert e<8e-6;emax=max(emax,e);amax=max(amax,angle(x,y));count+=1
 archive(root,['run_ordinary_coordinate_transplant','analyze_second_review_results','freeze_second_review_confirmation'],['cvpr_ordinary_transplant_20261012.log','cvpr_ordinary_transplant_analysis_20261012.log'])
 freeze(root,{'passed':True,'independently_reconstructed_rotation_pairs':count,'max_rotation_error_deg':amax,'max_reconstruction_error':emax,'canonical_hypothesis_changes':changes,'standalone_BLEU_sentence_evaluations':metrics,'exploratory_after_training':True,'no_TEST_use':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['confirmation','retrieval','transplant']);a=p.parse_args();globals()[a.phase]()
