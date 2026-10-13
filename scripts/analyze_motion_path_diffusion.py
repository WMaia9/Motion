"""Independent-reader DEV comparisons for path-supervised DDPMs."""
import csv
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from analyze_signspark_geometry_dev import statistics,b4
from analyze_motion_confirmation import bootstrap_wer
from run_motion_path_diffusion import ROOT,ARMS,signatures
import run_physical_semantic_ddpm as previous
from run_signspark_geometry_dev import matrices


def main():
    torch.set_num_threads(8)
    names=ARMS+['physical_area_time_shuffled']
    records={n:json.loads((ROOT/(n+'_scores.json')).read_text()) for n in names}
    references=records[names[0]]['references'];ids=records[names[0]]['clip_ids']
    weights=np.random.default_rng(42).multinomial(128,np.ones(128)/128,size=2000)
    boot={};rows=[]
    for name,r in records.items():
        assert r['references']==references and r['clip_ids']==ids
        for kind in ['raw','canonical']:
            st=statistics(references,r[kind+'_hypotheses'])
            assert abs(b4(st.sum(0))-r[kind+'_scores']['bleu']['bleu4'])<1e-10
            boot[name,kind]=np.asarray([b4(s) for s in weights@st])
        rows.append({'condition':name,'raw_bleu4':r['raw_scores']['bleu']['bleu4'],
            'canonical_bleu4':r['canonical_scores']['bleu']['bleu4'],
            'raw_gloss_wer':r['independent_raw_gloss']['wer'],
            'canonical_gloss_wer':r['independent_canonical_gloss']['wer']})
    pairs=[(a,'geometry_replay') for a in ARMS[1:]]
    pairs.extend([('physical_area_time',n) for n in ARMS[1:-1]])
    pairs.append(('physical_area_time_shuffled','physical_area_time'))
    result={'seed':42,'replicates':2000,'unit':'DEV clip, not signer','exploratory':True,
        'multiple_testing_adjustment':False,'no_reader_used_in_training':True,
        'train_seed_count':1,'comparisons':{},'physical_diagnostics':{}}
    for left,right in pairs:
        comparison={}
        for kind in ['raw','canonical']:
            l=records[left];r=records[right]
            comparison[kind+'_bleu4']={'delta':l[kind+'_scores']['bleu']['bleu4']-r[kind+'_scores']['bleu']['bleu4'],
                'ci95':np.percentile(boot[left,kind]-boot[right,kind],[2.5,97.5]).tolist()}
            lg=l['independent_'+kind+'_gloss'];rg=r['independent_'+kind+'_gloss']
            comparison[kind+'_independent_wer']=bootstrap_wer(rg['errors'],lg['errors'],lg['reference_lengths'])
        result['comparisons'][left+' minus '+right]=comparison
    native={e.sequence:e for e in previous.dataset('dev')}
    gt=[previous.project(native[n].sgn.numpy()) for n in ids]
    with np.load(ROOT/'path_scales.npz') as data:
        scale={k:torch.tensor(data[k],dtype=torch.float32) for k in data.files}
    for name in names+['ground_truth']:
        if name=='ground_truth':xs=gt
        else:
            with np.load(ROOT/(name+'_poses.npz')) as data:xs=[data[f'clip{i:04d}'].copy() for i in range(128)]
        angle_errors=[];speeds=[];feature_variances=[];sf_errors=[];a_errors=[];at_errors=[]
        with torch.inference_mode():
            for x,y in zip(xs,gt):
                assert x.shape==y.shape
                rotation=matrices(x);reference=matrices(y)
                delta=rotation.transpose(0,1,3,2)@reference
                angle_errors.append(float(np.degrees(Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude()).mean()))
                consecutive=rotation[:-1].transpose(0,1,3,2)@rotation[1:]
                speeds.append(float(np.degrees(Rotation.from_matrix(consecutive.reshape(-1,3,3)).magnitude()).mean()))
                px=previous.physical(torch.tensor(x)[None]);py=torch.tensor(y)[None];length=torch.tensor([len(x)])
                first,area=signatures(px,length);gf,ga=signatures(py,length)
                _,ta=signatures(px,length,True);_,gta=signatures(py,length,True)
                sf_errors.append(float(((first-gf)/scale['endpoint']).square().mean()))
                a_errors.append(float(((area-ga)/scale['area']).square().mean()))
                at_errors.append(float(((ta-gta)/scale['area_time']).square().mean()))
                feature_variances.append(float(px[0].std(0,unbiased=False).mean()))
        result['physical_diagnostics'][name]={'mean_reference_frame_angle_deg':float(np.mean(angle_errors)),
            'mean_frame_rotation_change_deg':float(np.mean(speeds)),
            'mean_temporal_canonical_feature_std':float(np.mean(feature_variances)),
            'normalized_endpoint_mse':float(np.mean(sf_errors)),
            'normalized_area_mse':float(np.mean(a_errors)),
            'normalized_area_time_mse':float(np.mean(at_errors)),
            'per_clip_reference_frame_angle_deg':angle_errors,
            'per_clip_frame_rotation_change_deg':speeds,
            'same_frame_error_not_DTW':True}
    result['post_result_physical_bootstrap']={}
    baseline=result['physical_diagnostics']['geometry_replay']
    for arm in ARMS[1:]:
        comparison={}
        for key in ['reference_frame_angle_deg','frame_rotation_change_deg']:
            difference=np.asarray(result['physical_diagnostics'][arm]['per_clip_'+key])-baseline['per_clip_'+key]
            comparison[key]={'delta_mean_deg':float(difference.mean()),
                'ci95':np.percentile(weights@difference/128,[2.5,97.5]).tolist()}
        result['post_result_physical_bootstrap'][arm+' minus geometry_replay']=comparison
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(5);labels=['Geometry','Velocity','Endpoint','Area','Area + time']
    axes[0].bar(x-.2,[r['raw_bleu4'] for r in rows[:5]],.4,label='raw')
    axes[0].bar(x+.2,[r['canonical_bleu4'] for r in rows[:5]],.4,label='canonical');axes[0].legend()
    axes[0].set_ylabel('BLEU-4 — independent reader')
    axes[1].bar(x,[r['canonical_gloss_wer'] for r in rows[:5]],color='#53878b')
    axes[1].set_ylabel('Canonical gloss WER (%)\nIndependent reader')
    for ax in axes:ax.set_xticks(x,labels,rotation=15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Physical path signatures in DDPM — PHOENIX DEV n=128\nNeither reader was used in training')
    fig.savefig(ROOT/'path_comparison.png',dpi=180);fig.savefig(ROOT/'path_comparison.pdf');plt.close(fig)
    print('PATH_ANALYSIS_COMPLETE',json.dumps(result['comparisons']),flush=True)


if __name__=='__main__':main()
