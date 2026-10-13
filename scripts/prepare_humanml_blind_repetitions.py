"""Register two fixed caption-blind replication cohorts and generation seeds.

TEST groups are excluded from the original target-aware study. Cohorts may
overlap the previous blind study and each other, explicitly quantified.
The concurrent prevalence study has consumed full TEST: no claim of globally
uninspected TEST is made. Each attack still withholds its reference until seal.
"""
from pathlib import Path
import numpy as np
import os
from motion_release_audit_common import DATA,A,read,write,sha
R=Path('reports/cvpr_humanml_blind_repetitions_20261012')
PREV=Path('reports/cvpr_humanml3d_blind_fid_20261012')

def main():
    assert not (R/'protocol.json').exists();R.mkdir(parents=True,exist_ok=True)
    excluded={x['clip_id'].lstrip('M') for x in read('reports/cvpr_humanml3d_audit_20261012/test_selection.json')}
    oldblind={x['clip_id'].lstrip('M') for x in read(PREV/'test_selection.json')}
    groups=[];sources={}
    for name in (DATA/'test.txt').read_text().splitlines():
        if name.lstrip('M') in excluded:continue
        path=DATA/'new_joint_vecs'/(name+'.npy')
        if not path.exists():continue
        x=np.load(path,allow_pickle=False)
        if not (40<=len(x)<200 and x.shape[1]==263 and np.isfinite(x).all()):continue
        captions=[]
        for line in (DATA/'texts'/(name+'.txt')).read_text().splitlines():
            c,t,s,e=line.split('#')
            if float(s)==float(e)==0:captions.append((c,t))
        if captions:groups.append(name);sources[name]=(len(x),captions)
    assert len(groups)>=960;cohorts=[]
    for j,(selection_seed,generation_seed) in enumerate([(20261021,4242),(20261022,7777)]):
        root=R/f'cohort_{j+1}';root.mkdir();rng=np.random.default_rng(selection_seed);ids=rng.choice(groups,960,replace=False);items=[];raw=np.zeros((960,196,263),np.float32)
        for i,name in enumerate(ids):
            frames,caps=sources[name];n=(frames//4)*4;start=int(rng.integers(frames-n+1));caption,tokens=caps[int(rng.integers(len(caps)))];path=DATA/'new_joint_vecs'/(name+'.npy');raw[i,:n]=np.load(path)[start:start+n]
            items.append({'clip_id':str(name),'caption':caption,'tokens':tokens,'frames':n,'start_frame':start,'motion_sha256':sha(path)})
        write(root/'test_selection.json',items);np.savez_compressed(root/'test_real.npz',raw=raw,lengths=np.array([x['frames'] for x in items]));os.link(PREV/'train_real.npz',root/'train_real.npz');os.link(PREV/'train_selection.json',root/'train_selection.json')
        protocol=read(PREV/'protocol.json');protocol.update(scope='Caption-blind replication cohort, fixed new generation seed; source groups overlap previous blind study and may overlap each other; no global uninspected-TEST claim',selection_seed=selection_seed,generation={**protocol['generation'],'seed':generation_seed},TEST_group_overlap_with_previous_blind=len({x.lstrip('M') for x in ids}&oldblind),TEST_previously_encoded_by_separate_prevalence_audit=True,primary='Same fixed blind-OT conditions; FID scored after attack seals. Both bounds retained, no post-hoc seed/cohort selection. Across-cohort summaries descriptive, not independent clip CIs.')
        files=[root/'train_real.npz',root/'test_real.npz',root/'train_selection.json',root/'test_selection.json',A/'t2m/text_mot_match/model/finest.tar',A/'humanml_enc_512_50steps/model000750000.pt']
        producers=[Path('scripts/prepare_humanml_blind_repetitions.py'),Path('scripts/run_humanml_blind_repetitions.py'),Path('scripts/run_humanml_blind_fid.py'),Path('scripts/score_humanml_blind_fid.py'),Path('scripts/run_motion_release_audit.py'),Path('scripts/motion_release_audit_common.py'),Path('scripts/humanml_audit_common.py')]
        protocol['inputs_sha256']={str(p):sha(p) for p in files};protocol['scientific_sources_sha256']={str(p):sha(p) for p in producers};write(root/'protocol.json',protocol);cohorts.append({'root':str(root),'generation_seed':generation_seed,'source_groups':sorted({x.lstrip('M') for x in ids}),'previous_blind_overlap':protocol['TEST_group_overlap_with_previous_blind']})
    overlap=len(set(cohorts[0]['source_groups'])&set(cohorts[1]['source_groups']));write(R/'protocol.json',{'replications':cohorts,'cross_cohort_source_group_overlap':overlap,'original_conditions_reused_without_selection':True,'TRAIN_reference_shared':True,'not_globally_uninspected_TEST':True})
    print('BLIND_REPETITIONS_REGISTERED',overlap,flush=True)
if __name__=='__main__':main()
