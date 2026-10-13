"""Separate processes isolate the native generator and evaluator namespaces."""
import argparse,sys
from pathlib import Path
import numpy as np
from motion_release_audit_common import read,write,sha,seed
R=Path('reports/cvpr_humanml_blind_repetitions_20261012')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['generate','attack','score']);parser.add_argument('cohort',type=int,choices=[1,2]);args=parser.parse_args();root=R/f'cohort_{args.cohort}';p=read(root/'protocol.json')
    for key in ['inputs_sha256','scientific_sources_sha256']:
        for file,h in p[key].items():assert sha(file)==h,file
    if args.action=='generate':
        from run_motion_release_audit import mdm_model,generate_mdm
        assert not (root/'test_mdm.npz').exists();seed(p['generation']['seed']);state=mdm_model();items=read(root/'test_selection.json');L=np.array([x['frames'] for x in items]);cohort={'raw':np.zeros((len(items),196,263),np.float32),'lengths':L,'captions':[x['caption'] for x in items]};seed(p['generation']['seed']);raw=generate_mdm(state,cohort,args.cohort);np.savez_compressed(root/'test_mdm.npz',raw=raw,lengths=L);write(root/'generation.json',{'seed':p['generation']['seed'],'no_reference_poses':True,'clips':len(items),'output_sha256':sha(root/'test_mdm.npz')})
    elif args.action=='attack':
        import run_humanml_blind_fid as attack
        attack.R=root;attack.main()
    else:
        import score_humanml_blind_fid as scorer
        scorer.R=root;scorer.main();analysis=read(root/'analysis.json');analysis['one_MDM_generation_seed']=p['generation']['seed'];analysis['TEST_overlap_previous_blind_groups']=p['TEST_group_overlap_with_previous_blind'];analysis['TEST_previously_encoded_by_separate_prevalence_audit']=True;analysis['global_uninspected_TEST_claim']=False;write(root/'analysis.json',analysis)
    print('BLIND_REPETITION_STAGE_COMPLETE',args.action,args.cohort,flush=True)
if __name__=='__main__':main()
