"""Independent reconstruction and scoring before freezing the mechanism diagnostic."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from verify_signspark_studies import sha,matrices,freeze
from freeze_learned_reader_studies import archive
from audit_reviewed_manuscript_evidence import metric
R=Path('reports/cvpr_dose_transplant_20261012')
def read(p):return json.loads(Path(p).read_text())
def main():
 assert (R/'analysis.json').exists() and read(R/'complete.json')['conditions']==24
 for p,h in read(R/'protocol.json')['inputs_sha256'].items():assert sha(p)==h,p
 for p,h in read(R/'weights_before_diagnostic.json')['checkpoints_sha256'].items():assert sha(p)==h,p
 count=metrics=0;anglemax=errmax=0.
 for file in sorted((R/'evaluation').glob('*_poses.npz')):
  r=read(file.with_name(file.name.removesuffix('_poses.npz')+'.json'));assert r['canonical_hypothesis_changes']==0
  for kind in ['baseline_raw','raw','canonical']:
   x=r[kind];assert abs(metric(x['references'],x['hypotheses'])-x['scores']['bleu']['bleu4'])<1e-10;metrics+=519
  for kind,field in [('baseline_raw','baseline_wrong_target_scores'),('raw','after_wrong_target_scores')]:
   assert abs(metric(r['targets'],r[kind]['hypotheses'])-r[field]['bleu']['bleu4'])<1e-10;metrics+=519
  with np.load(file) as z:
   assert len(z.files)==519*3
   for i in range(519):
    x,y,h=[z[f'{k}{i:04d}'] for k in ['source','alternate','hybrid']];assert x.shape==y.shape==h.shape and np.isfinite(h).all();assert np.array_equal(x[:,60:150],h[:,60:150])
    C=matrices(x)[...,:2,:];Cy=matrices(y)[...,:2,:];d=y.reshape(-1,40,6).astype(float);u=Cy[...,0,:];rho=np.linalg.norm(d[...,:3],axis=-1)[...,None];tau=(u*d[...,3:]).sum(-1)[...,None];sigma=np.linalg.norm(d[...,3:]-tau*u,axis=-1)[...,None]
    rebuilt=np.concatenate((rho*C[...,0,:],sigma*C[...,1,:]+tau*C[...,0,:]),-1).reshape(x.shape).astype(np.float32);rebuilt[:,60:150]=x[:,60:150];error=float(abs(rebuilt-h).max());assert error<1e-6;errmax=max(errmax,error)
    angle=float(np.degrees(Rotation.from_matrix((matrices(x).transpose(0,1,3,2)@matrices(h)).reshape(-1,3,3)).magnitude()).max());assert angle<.01;anglemax=max(anglemax,angle);count+=1
 assert count==24*519
 archive(R,['run_dose_coordinate_transplant','analyze_dose_transplant','freeze_dose_transplant'],['cvpr_dose_transplant_register_20261012.log','cvpr_dose_transplant_20261012.log','cvpr_dose_transplant_analysis_20261012.log'])
 freeze(R,{'passed':True,'independently_reconstructed_pairs':count,'max_reconstruction_error':errmax,'max_rotation_error_deg':anglemax,'standalone_BLEU_sentence_evaluations':metrics,'canonical_hypotheses_unchanged':True,'no_extra_fitting_or_per_clip_optimization':True,'not_TEST_confirmation':True,'all_eight_conditions_three_seeds_reported':True})
if __name__=='__main__':main()
