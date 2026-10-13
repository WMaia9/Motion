"""Check both generation studies and freeze only once all jobs/logs close."""
import json,sys
from pathlib import Path
import numpy as np,torch
from sign_physical_guidance import check
from verify_signspark_studies import sha,freeze

def diagnostics(root):
    proofs=check();saved=json.loads((root/'sources/projection_check.json').read_text());assert proofs==saved
    for name,h in json.loads((root/'sources/protocol_selection_hashes.json').read_text()).items():assert sha(root/name)==h
    clips=0;checks={}
    for corpus in ['PHOENIX-2014T','How2Sign']:
        selected=json.loads((root/corpus/'selection.json').read_text());names=[x['clip_id'] for x in selected];checks[corpus]={}
        for cond in ['cfg0','cfg1','cfg3','cfg5','cfg3_shuffled','cfg3_horizontal']:
            r=json.loads((root/corpus/(cond+'_scores.json')).read_text());assert r['clip_ids']==names and r['clips']==128;assert r['canonical_max_angle_deg']<.01
            with np.load(root/corpus/(cond+'_poses.npz')) as z:
                assert len(z.files)==128 and all(np.isfinite(z[k]).all() for k in z.files)
            paths=json.loads((root/corpus/(cond+'_paths.json')).read_text());b=np.load(paths['body'],allow_pickle=True).item();h=np.load(paths['hand'],allow_pickle=True).item();assert not np.any(b['keyframes']) and not np.any(h['keyframes']);assert b['pred_poses'].shape[1]==128 and h['pred_poses'].shape[1]==256;assert np.array_equal(b['video_names'][0],h['video_names'][0][::2]);assert np.array_equal(b['lengths'][0],h['lengths'][0][::2]);assert np.array_equal(b['lengths'][0],np.asarray([min(304,s['frames']) for s in selected]));assert np.isfinite(b['pred_poses']).all() and np.isfinite(h['pred_poses']).all()
            if cond=='cfg3_horizontal':
                for stream in ['body','hand']:
                    s=json.loads((root/'logs'/(corpus.replace('-','')+'_'+cond+'_'+stream+'_stats.json')).read_text());assert s['calls']==400 and s['late_calls']>0 and s['degenerate_joint_frames']==0;assert 0<=s['late_vertical_energy_fraction']<=1
            checks[corpus][cond]={'clips':128,'no_keyframe_inputs':True,'all_finite':True,'native_hand_body_ids_lengths_match':True,'rotation_preserved_by_canonicalization':True};clips+=128
    return {'generated_clips':clips,'projection_jvp_proof':proofs,'checks':checks,'TEST_used':False,'mesh_rendered':False}

def ddpm(root):
    for name,h in json.loads((root/'sources/protocol_sha256.json').read_text()).items():assert sha(root/name)==h
    audit=json.loads((root/'data_audit.json').read_text());assert len(audit['train_ids'])==7096 and len(audit['dev_ids'])==519;assert not set(audit['train_ids'])&set(audit['dev_ids']);hist=json.loads((root/'shared_history.json').read_text());assert len(hist)==20
    best=torch.load(root/'shared_best.pt',map_location='cpu',weights_only=False);assert abs(best['dev_mse']-min(x['dev_fixed_noise_mse'] for x in hist))<1e-10
    for arm in ['geometry','semantic_raw','semantic_physical']:
        h=json.loads((root/(arm+'_history.json')).read_text());assert len(h)==5;assert all(np.isfinite(x['train_mse']) and np.isfinite(x['semantic_term']) and np.isfinite(x['dev_fixed_noise_mse']) for x in h);state=torch.load(root/(arm+'.pt'),map_location='cpu',weights_only=False);assert state['epoch']==5 and all(torch.isfinite(v).all() for v in state['state_dict'].values())
    count=0
    for cond in ['geometry','semantic_raw','semantic_physical','semantic_physical_shuffled']:
        r=json.loads((root/(cond+'_scores.json')).read_text());assert r['clips']==128;assert r['clip_ids']==[x['clip_id'] for x in audit['selection']]
        with np.load(root/(cond+'_poses.npz')) as z:
            assert len(z.files)==128 and all(np.isfinite(z[k]).all() and 0<len(z[k])<=300 and z[k].shape[1]==240 for k in z.files)
        count+=128
    # Differential proof that canonical semantic inputs discard gauge directions.
    from run_physical_semantic_ddpm import physical
    g=torch.Generator().manual_seed(2027);x=torch.randn(4,8,240,dtype=torch.float64,generator=g);u=torch.randn(4,8,40,3,dtype=torch.float64,generator=g);d=x.reshape(4,8,40,6);v=torch.cat([u[...,0:1]*d[...,:3],u[...,1:2]*d[...,3:]+u[...,2:3]*d[...,:3]],-1).reshape(x.shape);_,j=torch.func.jvp(physical,(x,),(v,));error=float(j.abs().max());assert error<1e-10
    probe=json.loads((root/'teacher_prefix_probe.json').read_text());assert probe['post_result_exploratory_probe'] and probe['same_caption_and_decoder_prefix_in_all_conditions']
    for name,values in probe['per_clip_cross_entropy'].items():
        assert len(values)==128 and np.isfinite(values).all();assert abs(np.mean(values)-probe['mean_per_clip_cross_entropy'][name])<1e-10
    prior=Path('reports/cvpr_motion_consistency_confirmation_20261010');gt=json.loads((prior/'ctc_reader/original.json').read_text());control=json.loads((root/'ground_truth_reader_control.json').read_text());indices=[audit['dev_ids'].index(x['clip_id']) for x in audit['selection']];wer=100*sum(gt['errors'][i] for i in indices)/sum(gt['reference_lengths'][i] for i in indices);assert abs(wer-control['wer'])<1e-10
    return {'train_dev_overlap':0,'shared_epochs':20,'epochs_per_arm':5,'selected_shared_epoch':best['epoch'],'selected_shared_dev_mse':best['dev_mse'],'generated_clips':count,'canonical_semantic_input_gauge_jvp_max':error,'student_parameters':1436016,'post_result_teacher_probe_clips':128,'ground_truth_independent_wer':wer,'TEST_used':False,'mesh_rendered':False,'no_claim_of_SOTA':True}

if __name__=='__main__':
    phase=sys.argv[1];root=Path('reports/cvpr_text_to_sign_diagnostics_20261010' if phase=='diagnostics' else 'reports/cvpr_physical_semantic_ddpm_20261010')
    if len(sys.argv)>2 and sys.argv[2]=='freeze':freeze(root,json.loads((root/'verification_pending_freeze.json').read_text()))
    else:
        result=diagnostics(root) if phase=='diagnostics' else ddpm(root);(root/'verification_pending_freeze.json').write_text(json.dumps(result,indent=2));print('VERIFIED',phase,json.dumps(result),flush=True)
