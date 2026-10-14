"""Paired full-TEST string/edit controls; intervals conditional on native readers."""
import csv
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from motion_release_audit_common import read,write,sha
from restore_sign_natural_evidence import bleu

R=Path('reports/cvpr_sign_stability_20261014')


def tokens(text,corpus):return list(text.replace(' ','')) if corpus=='CSL-Daily' else text.split()


def edit(a,b):
    previous=list(range(len(b)+1))
    for i,u in enumerate(a,1):
        row=[i]
        for j,v in enumerate(b,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(u!=v)))
        previous=row
    return previous[-1]


def measurements(raw,variants,corpus):
    rows=[]
    for a,b in zip(raw,variants):
        x,y=tokens(a,corpus),tokens(b,corpus);d=edit(x,y)
        rows.append([float(a!=b),float(x!=y),float(d),d/max(len(x),len(y),1)])
    return np.array(rows)


def main():
    assert not (R/'manifest_sha256.json').exists();protocol=read(R/'protocol.json')
    out={};inputs={};NBOOT=20000
    with threadpool_limits(limits=2):
        for corpus in protocol['corpora']:
            folder=R/corpus;assert read(folder/'complete.json')['all8_conditions_full_nativeTEST']
            meta=read(folder/'inputs.json');refs=meta['references'];ids=meta['clip_ids'];raw=read(folder/'raw.json')['hypotheses'];N=len(raw)
            metrics={};individual={}
            for condition in protocol['conditions']:
                record=read(folder/(condition+'.json'));m=measurements(raw,record['hypotheses'],corpus)
                score,stats=bleu(refs,record['hypotheses'],corpus=='CSL-Daily');assert abs(score-record['scores']['bleu']['bleu4'])<1e-10
                metrics[condition]=m
                individual[condition]={'clips':N,'string_changed_count':int(m[:,0].sum()),'token_changed_count':int(m[:,1].sum()),'string_changed_fraction':float(m[:,0].mean()),'token_changed_fraction':float(m[:,1].mean()),'mean_token_edit_distance':float(m[:,2].mean()),'mean_normalized_edit_distance':float(m[:,3].mean()),'BLEU4':score,'BLEU_statistics':stats,'empty_hypotheses':sum(not h.strip() for h in record['hypotheses'])}
                inputs[str(folder/(condition+'.json'))]=sha(folder/(condition+'.json'))
            metrics['gaussian_support_mean']=np.mean([metrics[f'gaussian_support_{s}'] for s in protocol['Gaussian_seeds']],axis=0)
            primary=['canonical','gaussian_support_mean','gaussian_all_20261014','physical_1deg','physical_matched']
            feasible=np.array(read(folder/'physical_matched.json')['feasible']);assert feasible.all(),'Report unreachable matched physical clips explicitly before primary inference'
            if corpus=='How2Sign':
                csvpath=Path('reports/cvpr_evaluation_confirmation_20261011/sources/how2sign_test.csv')
                with csvpath.open() as f:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(f,delimiter='\t')}
                names=[lookup[key] for key in ids]
            else:names=ids
            groups=sorted(set(names));ix=np.array([groups.index(n) for n in names]);G=len(groups)
            counts=np.bincount(ix,minlength=G);weights=np.random.default_rng(42026+(corpus=='CSL-Daily')).multinomial(G,np.ones(G)/G,size=NBOOT)
            denom=weights@counts
            summaries={};boot={}
            for name in primary:
                values=metrics[name];sums=np.zeros((G,4));np.add.at(sums,ix,values)
                sampled=(weights@sums)/denom[:,None];boot[name]=sampled
                summaries[name]={'string_changed_fraction':float(values[:,0].mean()),'token_changed_fraction':float(values[:,1].mean()),'mean_token_edit_distance':float(values[:,2].mean()),'mean_normalized_edit_distance':float(values[:,3].mean())}
                for label,j in [('token_changed_fraction',1),('mean_normalized_edit_distance',3)]:summaries[name][label+'_family20_CI95']=np.percentile(sampled[:,j],[100*.05/(2*20),100*(1-.05/(2*20))]).tolist()
            contrasts={}
            for name in ['gaussian_support_mean','physical_matched']:
                contrast=metrics['canonical']-metrics[name];sample=boot['canonical']-boot[name];row={}
                for label,j in [('token_changed_fraction',1),('mean_normalized_edit_distance',3)]:row[label]={'canonical_minus_control':float(contrast[:,j].mean()),'family8_CI95':np.percentile(sample[:,j],[100*.05/(2*8),100*(1-.05/(2*8))]).tolist()}
                contrasts[name]=row
            np.savez_compressed(folder/'per_clip_measurements.npz',**metrics)
            budgets=np.array([p['budget'] for p in read(folder/'canonical.json')['parameters']]);lengths=np.array(meta['lengths']);rms=budgets/np.sqrt(lengths*240)
            physical={}
            for name in ['physical_1deg','physical_matched']:
                params=read(folder/(name+'.json'))['parameters'];angles=np.degrees([p['scalar'] for p in params]);physical[name]={'mean_active_joint_angle_deg':float(angles.mean()),'median_angle_deg':float(np.median(angles)),'min_angle_deg':float(angles.min()),'max_angle_deg':float(angles.max()),'percentiles_5_95_deg':np.percentile(angles,[5,95]).tolist(),'mean_realized_L2':float(np.mean([p['realized_L2'] for p in params]))}
            out[corpus]={'clips':N,'bootstrap_groups':G,'all_matched_physical_clips_feasible':True,'individual_conditions':individual,'primary_summary':summaries,'canonical_minus_control':contrasts,'canonical_displacement':{'clip_mean_RMS':float(rms.mean()),'frame_weighted_RMS':float(np.sqrt(np.sum(budgets**2)/(240*lengths.sum()))),'min_clip_RMS':float(rms.min()),'max_clip_RMS':float(rms.max()),'mean_L2':float(budgets.mean())},'physical_geometry':physical}
            print('SIGN_STABILITY_ANALYZED',corpus,{n:(v['token_changed_fraction'],v['mean_normalized_edit_distance']) for n,v in summaries.items()},flush=True)
    old=Path('reports/cvpr_matched_blocks_20261013');new=Path('reports/cvpr_coherent_blocks_20261014')
    a=read(old/'guo/analysis.json')['conditions'];b=read(new/'guo/analysis.json')['conditions'];assert read(old/'guo/normalizer.json')==read(new/'guo/normalizer.json')
    ratios={s:{'fiber_mean_FID':a[f'null_{s}_r3']['FID']['mean_change'],'physical_mean_FID':b[f'rotation_fixed_{s}_r3']['FID']['mean_change'],'ratio_of_mean_FID_changes':a[f'null_{s}_r3']['FID']['mean_change']/b[f'rotation_fixed_{s}_r3']['FID']['mean_change']} for s in ['plus','minus']}
    ratio_record={'kappa':3,'same_reader_normalizer_source_cohorts_and_rotation126_block':True,'directions':ratios,'direction_specific_not_universal_sensitivity':True,'not_a_new_prospective_hypothesis_test':True,'inputs_sha256':{str(p):sha(p) for p in [old/'guo/analysis.json',new/'guo/analysis.json',old/'guo/normalizer.json',new/'guo/normalizer.json']}}
    write(R/'rotation_ratio.json',ratio_record)
    write(R/'analysis.json',{'passed':True,'results':out,'raw_and_canonical_full_replay_exact':True,'primary_family':20,'contrast_family':8,'bootstrap_draws':NBOOT,'conditioned_on_test_inputs_frozen_reader_and_fixed_noise_fields':True,'no_human_semantic_accuracy_or_general_robustness_claim':True,'inputs_sha256':inputs,'analysis_source_sha256':sha(__file__)})
    print('SIGN_STABILITY_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
