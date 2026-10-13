"""Paired independent BT, alignment-reader diagnostic, physical geometry."""
import csv
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from analyze_signspark_geometry_dev import statistics,b4
from analyze_motion_confirmation import bootstrap_wer
from run_signspark_geometry_dev import matrices
from run_local_path_diffusion import ROOT,ARMS,CONTROLS,local_signatures,uniform
import run_physical_semantic_ddpm as previous


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8)
    names=CONTROLS+ARMS+['local_ctc_area_shuffled']
    records={n:json.loads((ROOT/(n+'_scores.json')).read_text()) for n in names}
    references=records[names[0]]['references'];ids=records[names[0]]['clip_ids']
    weights=np.random.default_rng(42).multinomial(128,np.ones(128)/128,size=2000)
    boot={};rows=[]
    for name,r in records.items():
        assert r['references']==references and r['clip_ids']==ids
        r['evaluation_provenance']={'BT_used_in_this_training':False,
            'CTC_used_to_construct_TRAIN_alignment':True,'CTC_fully_independent_evaluator':False,
            'legacy_independent_gloss_field_names_are_not_an_independence_claim':True,
            'reused_frozen_control':name in CONTROLS}
        (ROOT/(name+'_scores.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
        for kind in ['raw','canonical']:
            st=statistics(references,r[kind+'_hypotheses'])
            assert abs(b4(st.sum(0))-r[kind+'_scores']['bleu']['bleu4'])<1e-10
            boot[name,kind]=np.asarray([b4(s) for s in weights@st])
        rows.append({'condition':name,'raw_bleu4':r['raw_scores']['bleu']['bleu4'],
            'canonical_bleu4':r['canonical_scores']['bleu']['bleu4'],
            'alignment_reader_canonical_wer':r['independent_canonical_gloss']['wer'],
            'new_training_and_generation':name not in CONTROLS})
    pairs=[('local_ctc_area','local_uniform_area'),('local_ctc_endpoint','local_uniform_endpoint'),
        ('local_ctc_area','local_ctc_endpoint'),('local_uniform_area','local_uniform_endpoint')]
    pairs.extend((a,c) for a in ARMS for c in CONTROLS)
    pairs.append(('local_ctc_area_shuffled','local_ctc_area'))
    result={'seed':42,'bootstrap_replicates':2000,'unit':'DEV clip, not signer','train_seed_count':1,
        'exploratory':True,'multiple_testing_adjustment':False,'comparisons':{},'physical_diagnostics':{},
        'BT_is_independent_of_new_training':True,'CTC_is_alignment_source_not_independent':True,
        'new_training_arms':4,'new_generated_clips':640,'reused_control_clips':256}
    for left,right in pairs:
        comparison={}
        for kind in ['raw','canonical']:
            l=records[left];r=records[right]
            comparison[kind+'_bleu4']={'delta':l[kind+'_scores']['bleu']['bleu4']-r[kind+'_scores']['bleu']['bleu4'],
                'ci95':np.percentile(boot[left,kind]-boot[right,kind],[2.5,97.5]).tolist()}
            lg=l['independent_'+kind+'_gloss'];rg=r['independent_'+kind+'_gloss']
            comparison[kind+'_alignment_reader_wer']=bootstrap_wer(rg['errors'],lg['errors'],lg['reference_lengths'])
        result['comparisons'][left+' minus '+right]=comparison
    native={e.sequence:e for e in previous.dataset('dev')};gt=[previous.project(native[n].sgn.numpy()) for n in ids]
    with np.load(ROOT/'local_scales.npz') as data:scale={k:torch.tensor(data[k].copy()) for k in data.files}
    for name in names+['ground_truth']:
        if name=='ground_truth':xs=gt
        else:
            with np.load(ROOT/(name+'_poses.npz')) as data:xs=[data[f'clip{i:04d}'].copy() for i in range(128)]
        angle=[];speed=[];endpoint=[];area=[];variance=[]
        with torch.inference_mode():
            for x,y in zip(xs,gt):
                assert x.shape==y.shape
                rotation=matrices(x);reference=matrices(y)
                relative=rotation.transpose(0,1,3,2)@reference
                angle.append(float(np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).mean()))
                consecutive=rotation[:-1].transpose(0,1,3,2)@rotation[1:]
                speed.append(float(np.degrees(Rotation.from_matrix(consecutive.reshape(-1,3,3)).magnitude()).mean()))
                px=previous.physical(torch.tensor(x)[None]);py=torch.tensor(y)[None]
                bounds=[uniform(len(x),min(4,len(x)//2))]
                pf,pa,_=local_signatures(px,bounds);gf,ga,_=local_signatures(py,bounds)
                endpoint.append(float(((pf-gf)/scale['endpoint']).square().mean()))
                area.append(float(((pa-ga)/scale['area']).square().mean()))
                variance.append(float(px[0].std(0,unbiased=False).mean()))
        result['physical_diagnostics'][name]={'mean_reference_frame_angle_deg':float(np.mean(angle)),
            'mean_frame_rotation_change_deg':float(np.mean(speed)),'per_clip_reference_frame_angle_deg':angle,
            'per_clip_frame_rotation_change_deg':speed,'mean_temporal_canonical_feature_std':float(np.mean(variance)),
            'normalized_four_uniform_segment_endpoint_mse':float(np.mean(endpoint)),
            'normalized_four_uniform_segment_area_mse':float(np.mean(area)),
            'local_diagnostic_is_post_result_uniform_not_DEV_gloss_alignment':True,'same_frame_error_not_DTW':True}
    result['post_result_geometry_bootstrap']={}
    for left,right in pairs[:-1]:
        comparison={}
        for key in ['reference_frame_angle_deg','frame_rotation_change_deg']:
            difference=np.asarray(result['physical_diagnostics'][left]['per_clip_'+key])-np.asarray(result['physical_diagnostics'][right]['per_clip_'+key])
            comparison[key]={'delta_mean_deg':float(difference.mean()),'ci95':np.percentile(weights@difference/128,[2.5,97.5]).tolist()}
        result['post_result_geometry_bootstrap'][left+' minus '+right]=comparison
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(6)
    labels=['Geometry','Velocity','Uniform\nendpoint','Uniform\narea','CTC\nendpoint','CTC\narea']
    axes[0].bar(x,[r['canonical_bleu4'] for r in rows[:6]],color='#477a91')
    axes[0].set_ylabel('Canonical BLEU-4 — independent BT')
    axes[1].bar(x,[r['alignment_reader_canonical_wer'] for r in rows[:6]],color='#b28050')
    axes[1].set_ylabel('CTC WER (%) — alignment-source reader')
    for ax in axes:ax.set_xticks(x,labels,rotation=10);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Local physical path supervision — PHOENIX DEV n=128\nCTC is TRAIN alignment source; BT is not used in the new losses')
    fig.savefig(ROOT/'local_comparison.png',dpi=180);fig.savefig(ROOT/'local_comparison.pdf');plt.close(fig)
    print('LOCAL_ANALYSIS_COMPLETE',json.dumps(result['comparisons']),flush=True)


if __name__=='__main__':main()
