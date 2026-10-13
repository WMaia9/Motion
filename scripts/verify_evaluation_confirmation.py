"""Independent reconstruction and sampler replay; never alters frozen inputs."""
import ast
import inspect
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from run_evaluation_heldout import ROOT, CORPORA, CHECKPOINTS, REPRESENTATIONS, representation, sample_without_reference_pose, previous
from verify_signspark_studies import sha, matrices
from analyze_signspark_geometry_dev import statistics, b4


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    checks={'reference_pose_not_used_by_new_sampler':True,'new_training_runs':0,'rendered_meshes':False}
    assert sha(ROOT/'PROTOCOLO.md')==read(ROOT/'sources/protocol_sha256.json')['PROTOCOLO.md']
    for p,h in read(ROOT/'sources/frozen_inputs.json')['frozen_models'].items():
        assert sha(p)==h
    for item in read(ROOT/'sources/test_downloads.json'):
        path=ROOT/item['file'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    # Compile only the official converter, without importing licensed geometry.
    module=ast.parse(Path('external/SignSparK/tools/visualize.py').read_text())
    function=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='rot6d_to_matrix')
    scope={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),'<official_converter>','exec'),scope)
    official=scope['rot6d_to_matrix'];converter_difference=0.
    audit=read(ROOT/'test_audit.json');attacks={}
    for corpus in CORPORA:
        folder=ROOT/'heldout'/corpus;selection=read(folder/'selection.json');n=audit[corpus]['test_clips']
        chosen=np.sort(np.random.default_rng(2027).choice(n,128,replace=False)).tolist()
        assert chosen==audit[corpus]['attack_indices']==[r['test_index'] for r in selection]
        assert [audit[corpus]['test_ids'][i] for i in chosen]==[r['clip_id'] for r in selection]
        full=read(folder/'full_real_scores.json');assert full['clip_ids']==audit[corpus]['test_ids']
        for kind in ['raw','canonical']:
            assert abs(b4(statistics(full['references'],full[kind]['hypotheses']).sum(0))-full[kind]['scores']['bleu']['bleu4'])<1e-10
        records={}
        for condition in ['correct_caption','rolled_wrong_caption']:
            r=read(folder/(condition+'_scores.json'))
            assert r['clip_ids']==[x['clip_id'] for x in selection]
            assert r['canonical_before_actual']['hypotheses']==r['canonical_after_actual']['hypotheses']
            refs=r['references'];assert r['targets']==(refs if condition=='correct_caption' else refs[1:]+refs[:1])
            maxangle=0.;maximum_reconstruction=0.;coefficient=0.;mindegeneracy=float('inf')
            with np.load(folder/(condition+'_poses.npz')) as data:
                assert len(data.files)==384
                for i,e in enumerate(selection):
                    tag=f'{i:04d}';x=data['source'+tag];y=data['attacked'+tag];u=data['gauge'+tag]
                    assert x.shape==y.shape==(e['frames'],240) and u.shape==(len(x),40,3)
                    assert all(np.isfinite(a).all() for a in [x,y,u])
                    assert np.array_equal(x[:,60:150],y[:,60:150]) and (u[:,10:25]==0).all()
                    d=x.reshape(-1,40,6);a=d[...,:3];z=.35*np.tanh(u)
                    rebuilt=np.concatenate((np.exp(z[...,0:1])*a,np.exp(z[...,1:2])*d[...,3:]+z[...,2:3]*a),-1).reshape(y.shape)
                    maximum_reconstruction=max(maximum_reconstruction,float(abs(rebuilt-y).max()))
                    assert np.allclose(rebuilt,y,rtol=0,atol=2e-6)
                    coefficient=max(coefficient,float(abs(z).max()))
                    mx=matrices(x);my=matrices(y)
                    for value,matrix in [(x,mx),(y,my)]:
                        converted=official(torch.from_numpy(value.reshape(-1,40,6).astype(np.float64))).numpy()
                        converter_difference=max(converter_difference,float(abs(converted-matrix).max()))
                        dd=value.reshape(-1,40,6).astype(np.float64);aa=dd[...,:3];bb=dd[...,3:]
                        residual=bb-(aa*bb).sum(-1,keepdims=True)/(aa*aa).sum(-1,keepdims=True)*aa
                        mindegeneracy=min(mindegeneracy,float(np.linalg.norm(aa,axis=-1).min()),float(np.linalg.norm(residual,axis=-1).min()))
                    relative=mx.transpose(0,1,3,2)@my
                    angle=float(np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).max())
                    maxangle=max(maxangle,angle)
            assert maxangle<.01 and coefficient<=.350001 and mindegeneracy>1e-12
            assert abs(maxangle-r['max_rotation_error_deg'])<1e-8
            assert len(r['losslogs'])==4
            for batch in r['losslogs']:assert len(batch['cross_entropy'])==80 and np.isfinite(batch['cross_entropy']).all()
            records[condition]={'clips':128,'rotation_max_deg':maxangle,'gauge_reconstruction_max_abs':maximum_reconstruction,
                'gauge_bound_max':coefficient,'minimum_nonzero_vector_norm':mindegeneracy,'canonical_changes':0,'left_hand_bitwise_preserved':True}
        attacks[corpus]=records
        print('VERIFIED_ATTACKS',corpus,flush=True)
    checks['attacks']=attacks;checks['official_converter_float64_max_abs_difference']=converter_difference
    assert converter_difference<1e-12
    checked=0;defense=0;ranking_max=0.
    for name in CHECKPOINTS:
        folder=ROOT/'rankings'/name
        with np.load(folder/'poses.npz') as data:
            assert len(data.files)==642
            xs=[data[f'clip{i:04d}'].copy() for i in range(642)]
        baseline=read(folder/'canonical_scores.json')
        assert baseline['clip_ids']==audit['PHOENIX-2014T']['test_ids']
        assert all(x.shape==(len(x),240) and 0<len(x)<=300 and np.isfinite(x).all() for x in xs)
        for kind in REPRESENTATIONS:
            r=read(folder/(kind+'_scores.json'));assert r['clip_ids']==baseline['clip_ids'] and r['references']==baseline['references']
            assert abs(b4(statistics(r['references'],r['hypotheses']).sum(0))-r['scores']['bleu']['bleu4'])<1e-10
            ys=representation(xs,kind);maxangle=0.
            for x,y in zip(xs,ys):
                rel=matrices(x).transpose(0,1,3,2)@matrices(y)
                maxangle=max(maxangle,float(np.degrees(Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude()).max()))
            assert maxangle<.01 and abs(maxangle-r['max_rotation_error_deg'])<1e-8
            ranking_max=max(ranking_max,maxangle);checked+=642
            if kind.startswith('gauge_'):
                assert r['defended']['hypotheses']==baseline['hypotheses']
                assert abs(b4(statistics(r['references'],r['defended']['hypotheses']).sum(0))-r['defended']['scores']['bleu']['bleu4'])<1e-10
                defense+=642
        print('VERIFIED_RANKINGS',name,flush=True)
    checks['ranking_decodes_verified']=checked+defense;checks['canonical_generated_gauge_pairs']=defense;checks['ranking_rotation_error_max_deg']=ranking_max
    # The prior implementation allocates from a packed pose tensor. Verify that
    # removing that tensor leaves the exact same sampling computation on DEV.
    assert '.sgn' not in inspect.getsource(sample_without_reference_pose)
    dev=previous.dataset('dev');selected=dev[:16];ids=[e.sequence for e in selected]
    inputs=[{'clip_id':e.sequence,'frames':len(e.sgn),'text':list(e.txt)} for e in selected]
    oldaudit=read(previous.ROOT/'data_audit.json');stoi={w:i for i,w in enumerate(oldaudit['vocab'])}
    with np.load(previous.ROOT/'normalization.npz') as data:mean=data['mean'].copy();std=data['std'].copy()
    model=previous.Denoiser(len(stoi)).cuda();model.load_state_dict(torch.load(CHECKPOINTS['geometry'],map_location='cpu',weights_only=False)['state_dict'],strict=True)
    alpha=previous.alpha_schedule();old=previous.sample(model,dev,ids,stoi,mean,std,alpha)
    new=sample_without_reference_pose(model,inputs,stoi,mean,std,alpha)
    assert all(np.array_equal(a,b) for a,b in zip(old,new))
    checks['sampler_replay_DEV_clips']=16;checks['sampler_replay_bitwise_equal']=True
    checks['generation_condition']='text and reference duration capped300; no reference poses, glosses, or anchors'
    del model;torch.cuda.empty_cache();print('VERIFIED_SAMPLER_REPLAY',flush=True)
    ext=read(ROOT/'external_bt/contract_and_reproduction.json')
    assert ext['native_format_validation_passed'] and ext['absolute_difference']<1e-8 and not ext['6D_gauge_vulnerability_replication']
    for name in ['ground_truth','progressive_transformer']:
        r=read(ROOT/'external_bt'/(name+'_scores.json'));assert r['clips']==515
        assert abs(b4(statistics(r['references'],r['hypotheses']).sum(0))-r['bleu']['bleu4'])<1e-10
    tracked=subprocess.check_output(['git','-C',ext['repository'],'status','--porcelain','--untracked-files=no'],text=True)
    assert not tracked;checks['external_BT_native_contract_verified']=True;checks['external_BT_6D_vulnerability_replicated']=False
    checks['prior_frozen_reports']={}
    names=['cvpr_physical_semantic_ddpm_20261010','cvpr_fixed_prefix_diffusion_20261010','cvpr_motion_path_diffusion_20261010','cvpr_local_path_diffusion_20261011','cvpr_signspark_gauge_full_dev_20261010']
    for name in names:
        folder=Path('reports')/name;manifest=read(folder/'manifest_sha256.json')
        for filename,digest in manifest.items():assert sha(folder/filename)==digest,(name,filename)
        checks['prior_frozen_reports'][name]={'verified_files':len(manifest),'manifest_sha256':sha(folder/'manifest_sha256.json')}
        print('VERIFIED_PRIOR_REPORT',name,len(manifest),flush=True)
    (ROOT/'independent_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('EVALUATION_INDEPENDENT_VERIFICATION_COMPLETE',flush=True)


if __name__=='__main__':main()
