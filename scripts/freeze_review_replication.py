"""Independent NumPy/SciPy verification of new, separately frozen studies."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from verify_signspark_studies import sha,matrices,freeze
from freeze_learned_reader_studies import archive
from audit_reviewed_manuscript_evidence import metric

def read(p):return json.loads(Path(p).read_text())
def scores(r):
 n=0
 for kind in ['raw','canonical']:
  x=r[kind];assert abs(metric(x['references'],x['hypotheses'])-x['scores']['bleu']['bleu4'])<1e-10;n+=len(x['references'])
 return n

def adaptive(confirmation=False):
 root=Path('reports')/('cvpr_adaptive_distance_confirmation_20261012' if confirmation else 'cvpr_adaptive_distance_attack_20261012');assert (root/'analysis.json').exists();p=read(root/'protocol.json')
 for path,h in p['inputs_sha256'].items():assert sha(path)==h,path
 if confirmation:
  with np.load('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz') as z:xs=[z[f'clip{i:04d}'].copy() for i in range(642)]
 else:
  with np.load('reports/cvpr_constant_noncanonicity_20261012/generated/dev/poses.npz') as z:xs=[z[f'clip{i:05d}'].copy() for i in p['selection_indices']]
 baseline=read(root/'baseline_canonical.json');count=metrics=0;anglemax=reconmax=emax=nmaxcircle=0.
 for file in sorted((root/'evaluation').glob('*_poses.npz')):
  name=file.name.removesuffix('_poses.npz');r=read(root/'evaluation'/f'{name}.json');metrics+=scores(r);assert r['canonical']['hypotheses']==baseline['hypotheses']
  circle=name.startswith('circle')
  with np.load(file) as z:
   assert len(z.files)==len(xs)*2
   for i,x in enumerate(xs):
    y=z[f'attacked{i:04d}'];w=z[f'auxiliary{i:04d}'].astype(float);d=x.reshape(-1,40,6).astype(float);C=matrices(x)[...,:2,:];u,v=C[...,0,:],C[...,1,:]
    assert x.shape==y.shape and np.isfinite(y).all();assert np.array_equal(x[:,60:150],y[:,60:150]);assert (w[:,10:25]==0).all()
    assert abs(w).max()<= (1.000001 if circle else np.pi/np.sqrt(3)+1e-6)
    rho=np.linalg.norm(d[...,:3],axis=-1);tau=(u*d[...,3:]).sum(-1);sigma=np.linalg.norm(d[...,3:]-tau[...,None]*u,axis=-1);s=np.stack((rho-1,sigma-1,tau),-1);radius=np.linalg.norm(s,axis=-1)
    axis=w.copy()
    if circle:
     phase=np.pi*w.mean(-1);axis=np.stack((phase/np.sqrt(2),phase/np.sqrt(2),np.zeros_like(phase)),-1)
    axis[radius>=.95]=0;Q=Rotation.from_rotvec(axis.reshape(-1,3)).as_matrix().reshape(*axis.shape[:-1],3,3);delta=(Q@s[...,None])[...,0]-s
    rebuilt=d+np.concatenate((delta[...,0:1]*u,delta[...,1:2]*v+delta[...,2:3]*u),-1);error=float(abs(rebuilt-y.reshape(d.shape)).max());assert error<8e-6;reconmax=max(reconmax,error)
    Cy=matrices(y)[...,:2,:];E=((d-C.reshape(d.shape))**2).sum(-1);Ey=((y.reshape(d.shape).astype(float)-Cy.reshape(d.shape))**2).sum(-1);e=float(abs(E-Ey).max());assert e<2e-5;emax=max(emax,e)
    nn=float(abs((d*d).sum(-1)-(y.reshape(d.shape).astype(float)**2).sum(-1)).max())
    if circle:assert nn<2e-5;nmaxcircle=max(nmaxcircle,nn)
    angle=float(np.degrees(Rotation.from_matrix((matrices(x).transpose(0,1,3,2)@matrices(y)).reshape(-1,3,3)).magnitude()).max());assert angle<.01;anglemax=max(anglemax,angle);count+=1
 expected=12 if confirmation else 16;assert count==len(xs)*expected
 script='run_adaptive_distance_confirmation' if confirmation else 'run_adaptive_distance_attack'
 archive(root,[script,'analyze_review_replication','freeze_review_replication'],[f'{root.name}.log',f'{root.name}_analysis.log','cvpr_adaptive_distance_analysis_20261012.log' if not confirmation else 'cvpr_adaptive_confirmation_analysis_20261012.log'])
 freeze(root,{'passed':True,'independently_reconstructed_pairs':count,'standalone_BLEU_sentence_evaluations':metrics,'max_reconstruction_error':reconmax,'max_rotation_error_deg':anglemax,'max_canonical_distance_squared_error':emax,'max_circle_6D_squared_norm_error':nmaxcircle,'canonical_hypotheses_unchanged':True,'not_amortized_or_ordinary_training':True,'not_general_detector_evasion':True,'TEST_confirmation':confirmation})

def seed_dose():
 root=Path('reports/cvpr_seed_dose_response_20261012');assert (root/'analysis.json').exists();p=read(root/'protocol.json')
 for key in ['inputs_sha256','scientific_sources_sha256']:
  for path,h in p[key].items():assert sha(path)==h,path
 weights=read(root/'weights_frozen_before_inference.json')
 for path,h in weights['checkpoints_sha256'].items():assert sha(path)==h
 count=metrics=0;amax=0.
 for path in root.glob('training/seed*/*/*_epoch*/result.json'):
  r=read(path);metrics+=scores(r);n=len(r['raw']['clip_ids']);assert n==(519 if r['split']=='dev' else 642)
  with np.load(path.with_name('poses.npz')) as z:
   for i in range(n):
    x=z[f'clip{i:04d}'];assert x.shape[1]==240 and np.isfinite(x).all();R=matrices(x);c=R[...,:2,:].reshape(x.shape).astype(np.float32);angle=float(np.degrees(Rotation.from_matrix((R.transpose(0,1,3,2)@matrices(c)).reshape(-1,3,3)).magnitude()).max());assert angle<.01;amax=max(amax,angle);count+=1
 assert count==32*(519+642)
 for seed in [42,123,777]:
  a=read(root/f'training/seed{seed}/geodesic_raw_w1/first_epoch_stochastic.json');b=read(f'reports/cvpr_quotient_training_20261012/training/seed{seed}/geodesic_raw_w1/first_epoch_stochastic.json');assert a==b
 archive(root,['run_seed_dose_response','analyze_review_replication','freeze_review_replication'],['cvpr_seed_dose_response_20261012.log','cvpr_seed_dose_dev_20261012.log','cvpr_seed_dose_test_20261012.log','cvpr_seed_dose_analysis_20261012.log'])
 freeze(root,{'passed':True,'fits':23,'new_seed_primary_fits':7,'new_seed_matched_control_fits':7,'dose_response_fits':9,'independent_raw_canonical_pairs':count,'standalone_BLEU_sentence_evaluations':metrics,'max_rotation_error_deg':amax,'all_checkpoints_hashed_before_TEST':True,'matched_stochastic_inputs_checked':True,'historical_first_epoch_replayed':True,'one_shared_pretrain':True,'not_pristine_public_TEST':True})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['adaptive','adaptive_confirmation','seed_dose']);a=ap.parse_args()
 if a.phase=='adaptive_confirmation':adaptive(True)
 else:globals()[a.phase]()
