"""Register all public-generator conditions before new test inference."""
from pathlib import Path
import subprocess
import numpy as np
from motion_release_audit_common import R,A,DATA,seed,native_dataset,sha,read,write

def main():
 assert not (R/'protocol.json').exists();seed(20261012);dataset=native_dataset()
 source=Path('reports/cvpr_humanml3d_audit_20261012/analysis.json');old=read(source)['TEST']['results'];base=old['mdm']['identity']['FID_after'];ratios={}
 for name in ['global_plus010','global_minus010','global_plus035','global_minus035','random035_seed42']:
  v=old['mdm'][name]['FID_after'];ratios[name]={'FID':v,'absolute_FID_over_MDM_baseline':v/base,'FID_change_over_MDM_baseline':(v-base)/base}
 write(R/'review_arithmetic.json',{'source_SHA256':sha(source),'baseline_MDM_FID':base,'MDM_conditions':ratios,'note':'6-9x is approximately the absolute FID ratio; actual MDM changes are4.76/8.30x. Small/random changes are smaller, not mathematically absent.'})
 releases={
 'mdm50':str(A/'humanml_enc_512_50steps/model000750000.pt'),
 'mld': 'models/motion_release_audit/mld.ckpt',
 'motionlcm_v1':'models/motion_release_audit/extracted/experiments_t2m/motionlcm_humanml/motionlcm_humanml_v1.ckpt',
 't2mgpt':'models/motion_release_audit/extracted/VQTransformer_corruption05/net_best_fid.pth',
 't2m':'models/humanml3d_audit/extracted/t2m/Comp_v6_KLD01/model/latest.tar'}
 inputs=[Path(x) for x in releases.values()]+[DATA/'test.txt',A/'t2m/text_mot_match/model/finest.tar',Path('external/MDM/dataset/t2m_mean.npy'),Path('external/MDM/dataset/t2m_std.npy')]
 write(R/'protocol.json',{'scope':'Paired raw/canonical audit of five accessible released generators, with20replications and full native ground-truth loader; no post-hoc model selection',
 'primary':'same-output raw-minus-canonical FID and R1, across20independently materialized native repetitions, not invented attacks or intentional prevalence',
 'release_candidates':releases,'unavailable_candidate':'MoMask author Drive quota; no weights substituted',
 'native_dataset_entries':len(dataset),'batch_size':32,'replications':20,'replication_seeds':[1234+i for i in range(20)],
 'native_preprocessing':'unmodified MDM Text2MotionDatasetV2 including subcaption segments, random caption draws, random multiple-of-four single/double crop, full native TEST, drop_last32',
 'mdm_official_cap':'native wo_mm nominal1000 actually1024outputs; full TEST reference and20replications',
 'other_generators':'full native TEST inputs per repetition; model-specific public inference configurations and native output-length rules; paired audit, not an assertion of exact published-score reproduction',
 'conditions':['raw','canonical'],'real_fixed_encoding_controls':['identity','canonical','global_plus010','global_minus010','global_plus035','global_minus035','random035_seed42'],
 'additional_model_audit_not_automatic_reward_exploitation_evidence':True,
 'summary':'native mean±1.96*std/sqrt20 plus paired raw-canonical t19 intervals; all generators retained; paired metrics adjusted across10generator/metric endpoints',
 'source_directions':'fixed encoding coefficients=(b,-b,b); alpha=exp(b),beta=exp(-b),eta=b in unnormalized features; random seeded independently of captions',
 'data_mirror_limitations':'same pinned numerical-compatible public mirror as prior frozen study; original full byte identity not proved',
 'inputs_sha256':{str(p):sha(p) for p in inputs},
 'native_revisions':{str(p):subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip() for p in map(Path,['external/MDM','external/MLD','external/MotionLCM','external/T2M_GPT','external/HumanML3D_Evaluator','external/MotionRFT'])},
 'producer_sha256':{str(p):sha(p) for p in map(Path,['scripts/prepare_motion_release_audit.py','scripts/motion_release_audit_common.py'])}})
 write(R/'loader_inventory.json',{'entries':len(dataset),'native_keys':list(dataset.name_list),'not_subset_full_caption_only':True,'native_loader_source_sha256':sha('external/MDM/data_loaders/humanml/data/dataset.py')})
 print('PREVALENCE_REGISTERED',len(dataset),flush=True)
if __name__=='__main__':main()
