"""Verify experiments, source hashes, physical invariants and freeze reports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path('reports/cvpr_signspark_baseline_20261010')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def matrices(x):
    d=x.reshape(-1,40,6).astype(np.float64);a=d[...,:3];a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=d[...,3:]-(a*d[...,3:]).sum(-1,keepdims=True)*a;b=b/np.linalg.norm(b,axis=-1,keepdims=True);return np.stack((a,b,np.cross(a,b)),axis=-2)

def freeze(root,checks):
    assert not (root/'manifest_sha256.json').exists(),'already frozen'
    for f in (root/'scripts').glob('*.py'):
        if (Path('scripts')/f.name).exists():assert sha(f)==sha(Path('scripts')/f.name),f
    (root/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    manifest={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest_sha256.json'}
    (root/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FROZEN',str(root),len(manifest),sha(root/'manifest_sha256.json'),flush=True)

def baseline():
    downloads=json.loads((BASE/'sources/downloads.json').read_text())
    for r in downloads:assert sha(r['path'])==r['sha256']
    checked=0
    for corpus,n in [('PHOENIX-2014T',519),('How2Sign',1650)]:
        folder=BASE/corpus;refs=json.loads((folder/'references.json').read_text());assert len(refs)==n
        conditions=[*json.loads((folder/'summary.json').read_text()),*json.loads((folder/'gauge_summary.json').read_text())];assert len(conditions)==16
        for condition in conditions:
            result=json.loads((folder/f'{condition}.json').read_text());assert len(result['hypotheses'])==n
            assert all(np.isfinite(v) for v in result['scores']['bleu'].values())
            if condition.startswith('gauge_scale') or condition=='gauge_shear010':assert result['maximum_rotation_error_deg']<.001
            if condition.startswith('adaptive_'):assert len(result['anchors'])==n
            checked+=n
        for path in folder.glob('*_examples.npz'):
            with np.load(path) as data:assert all(np.isfinite(data[k]).all() for k in data.files)
    freeze(BASE,{'source_download_hashes_checked':len(downloads),'conditions_per_corpus':16,'decoded_sentences':checked,'native_metric_sufficient_statistics_verified_in_analysis':True,'no_TEST_task_scores':True,'no_generator_training':True,'no_rendered_meshes':True})

def attack(root,full):
    checks={};expected={'PHOENIX-2014T':519 if full else 64,'How2Sign':1650 if full else 64}
    for corpus,n in expected.items():
        folder=root/corpus;selected=json.loads((folder/'selection.json').read_text());assert len(selected)==n
        assert len({r['clip_id'] for r in selected})==n
        records={}
        for condition in ['correct_caption','rolled_wrong_caption']:
            result=json.loads((folder/f'{condition}.json').read_text());assert len(result['raw_after_hypotheses'])==n;assert result['canonical_changed_sentences']==0
            assert result['canonical_original_hypotheses']==result['canonical_after_hypotheses']
            maxangle=0.;maximum_coefficient=0.
            with np.load(folder/f'{condition}_poses.npz') as data:
                for i in range(n):
                    tag=f'{i:04d}' if full else f'{i:02d}';x=data['source'+tag];y=data['attacked'+tag];u=data['gauge'+tag]
                    assert x.shape==y.shape==(selected[i]['frames'],240);assert u.shape==(len(x),40,3)
                    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(u).all()
                    if full:assert np.array_equal(x[:,60:150],y[:,60:150]);assert (u[:,10:25]==0).all()
                    d=x.reshape(-1,40,6);alpha=np.exp(.35*np.tanh(u[...,0:1]));beta=np.exp(.35*np.tanh(u[...,1:2]));gamma=.35*np.tanh(u[...,2:3]);reconstructed=np.concatenate((alpha*d[...,:3],beta*d[...,3:]+gamma*d[...,:3]),-1).reshape(y.shape);assert np.max(abs(reconstructed-y))<2e-6
                    rel=matrices(x).transpose(0,1,3,2)@matrices(y);angle=float(np.degrees(Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude()).max());maxangle=max(maxangle,angle);maximum_coefficient=max(maximum_coefficient,float(abs(.35*np.tanh(u)).max()))
            assert maxangle<.01;assert maximum_coefficient<=.350001
            for batch in result['losslogs']:assert len(batch['cross_entropy'])==80 and np.isfinite(batch['cross_entropy']).all()
            if full:
                contract=json.loads((folder/'contract_checks.json').read_text())[condition];assert contract['clips']==8 and contract['native_sample_loader_max_abs']<1e-5
            records[condition]={'verified_poses':n,'rotation_error_max_deg_independent':maxangle,'bounded_gauge_coefficient_max':maximum_coefficient,'canonical_changed_sentences':0,'steps_per_batch':80,'left_hand_bitwise_preserved':full}
        checks[corpus]=records
    freeze(root,{'checks':checks,'two_datasets_one_reader_architecture':True,'oracle_target_captions_explicit':True,'TEST_used':False,'new_generator':False,'mesh_rendered':False})

def access():
    root=Path('reports/cvpr_sign3d_access_20261010');audit=json.loads((root/'audit.json').read_text());assert audit['recordings']==2266 and audit['total_frames']==6165742;assert not audit['nonfinite'] and all(not x for x in audit['source_overlap'].values());assert sha(root/'sources/how2sign-synth3D-greenscreen-smplh.zip')==audit['archive_sha256']
    freeze(root,{'recordings':2266,'frames':6165742,'source_archive_hash_verified':True,'all_NPZ_read_and_CRC_checked_in_audit':True,'rotations_finite':True,'source_clock_missing':True,'SMPL_H_model_absent_user_confirmed':True,'task_scores_computed':False})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['access','baseline','pilot','full'],required=True);args=ap.parse_args()
    if args.phase=='access':access()
    elif args.phase=='baseline':baseline()
    else:attack(Path('reports/cvpr_signspark_gauge_full_dev_20261010' if args.phase=='full' else 'reports/cvpr_signspark_gauge_attack_20261010'),args.phase=='full')
