"""Native BLEU checks, paired rank changes, fixed heldout oracle attacks."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from run_evaluation_heldout import ROOT,CORPORA,CHECKPOINTS,REPRESENTATIONS


def bootstrap(refs,hyps,weights):
    stats=statistics(refs,hyps)
    return b4(stats.sum(0)),np.asarray([b4(row) for row in weights@stats])


def difference(left,right,left_boot,right_boot):
    return {'delta_bleu4':float(left-right),'ci95':np.percentile(left_boot-right_boot,[2.5,97.5]).tolist()}


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    result={'seed':42,'replicates':2000,'multiple_testing_adjustment':False,'train_seed_count':1,
        'TEST_used_for_new_model_selection':False,'rankings':{},'attacks':{},'actual_external_BT_replication':{}}
    records={};boot={};points={};rows=[];n=642
    weights=np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=2000)
    for name in CHECKPOINTS:
        for kind in REPRESENTATIONS:
            r=json.loads((ROOT/'rankings'/name/(kind+'_scores.json')).read_text())
            assert r['clips']==n
            point,sampled=bootstrap(r['references'],r['hypotheses'],weights)
            assert abs(point-r['scores']['bleu']['bleu4'])<1e-10
            records[name,kind]=r;points[name,kind]=point;boot[name,kind]=sampled
            row={'system':name,'representation':kind,'bleu4':point,
                'changed_from_raw':sum(a!=b for a,b in zip(r['hypotheses'],records[name,'raw']['hypotheses'])),
                'max_rotation_error_deg':r['max_rotation_error_deg'],
                'canonical_defense_changes':r.get('defended_changed_sentences',0)}
            rows.append(row)
    ranks={kind:sorted(CHECKPOINTS,key=lambda name:(-points[name,kind],name)) for kind in REPRESENTATIONS}
    comparisons={}
    for left,right in itertools.combinations(CHECKPOINTS,2):
        comparison={kind:difference(points[left,kind],points[right,kind],boot[left,kind],boot[right,kind]) for kind in REPRESENTATIONS}
        comparison['raw_to_canonical_change_in_pairwise_gap']={
            'delta_gap':(points[left,'canonical']-points[right,'canonical'])-(points[left,'raw']-points[right,'raw']),
            'ci95':np.percentile((boot[left,'canonical']-boot[right,'canonical'])-(boot[left,'raw']-boot[right,'raw']),[2.5,97.5]).tolist()}
        comparison['point_ranking_flips_vs_raw']=[kind for kind in REPRESENTATIONS[1:]
            if (points[left,kind]-points[right,kind])*(points[left,'raw']-points[right,'raw'])<0]
        # Stronger inference: opposite rankings, each supported within representation.
        rawci=comparison['raw']['ci95']
        comparison['opposite_pairwise_ci_supported_rankings_vs_raw']=[kind for kind in REPRESENTATIONS[1:]
            if (rawci[0]>0 and comparison[kind]['ci95'][1]<0) or (rawci[1]<0 and comparison[kind]['ci95'][0]>0)]
        comparisons[left+' minus '+right]=comparison
    result['rankings']={'clips':n,'systems':list(CHECKPOINTS),'scores':{kind:{name:points[name,kind] for name in CHECKPOINTS} for kind in REPRESENTATIONS},
        'orders':ranks,'pairwise_comparisons':comparisons,
        'same_motion_in_all_representations':True,'new_generated_clips':4*n,
        'not_a_comparison_of_four_independent_published_generators':True}
    for corpus in CORPORA:
        folder=ROOT/'heldout'/corpus;conditions={}
        weights=np.random.default_rng(42).multinomial(128,np.ones(128)/128,size=2000)
        for name in ['correct_caption','rolled_wrong_caption']:
            r=json.loads((folder/(name+'_scores.json')).read_text());refs=r['references'];targets=r['targets']
            baseline,bbase=bootstrap(targets,r['baseline_actual']['hypotheses'],weights)
            attacked,bafter=bootstrap(targets,r['raw_after_actual']['hypotheses'],weights)
            actual,bactual=bootstrap(refs,r['raw_after_actual']['hypotheses'],weights)
            original,boriginal=bootstrap(refs,r['baseline_actual']['hypotheses'],weights)
            assert abs(baseline-r['baseline_target_scores']['bleu']['bleu4'])<1e-10
            assert abs(attacked-r['raw_after_target_scores']['bleu']['bleu4'])<1e-10
            assert abs(actual-r['raw_after_actual']['scores']['bleu']['bleu4'])<1e-10
            conditions[name]={'baseline_target_bleu4':baseline,'attacked_target_bleu4':attacked,
                'attacked_actual_reference_bleu4':actual,'target_gain':difference(attacked,baseline,bafter,bbase),
                'actual_reference_change':difference(actual,original,bactual,boriginal),
                'max_rotation_error_deg':r['max_rotation_error_deg'],'canonical_changes':r['canonical_changed_sentences'],
                'criterion_A_pass':attacked>baseline and r['max_rotation_error_deg']<.01 and r['canonical_changed_sentences']==0}
        full=json.loads((folder/'full_real_scores.json').read_text())
        result['attacks'][corpus]={'full_real_TEST_clips':full['clips'],'full_real_raw_bleu4':full['raw']['scores']['bleu']['bleu4'],
            'full_real_canonical_bleu4':full['canonical']['scores']['bleu']['bleu4'],
            'full_real_changed_sentences':full['raw_canonical_changed_sentences'],'subset_conditions':conditions,
            'bootstrap_unit':'clip; How2Sign VIDEO_ID cluster sensitivity separately reported'}
    result['actual_external_BT_replication']=json.loads((ROOT/'external_bt/contract_and_reproduction.json').read_text())
    external=json.loads((ROOT/'external_bt/progressive_transformer_scores.json').read_text())
    row=statistics(external['references'],external['hypotheses']).sum(0)
    result['external_public_PT_zero_bleu_diagnostic']={'counts_1_to_4':row[:4].tolist(),'totals_1_to_4':row[4:8].tolist(),
        'empty_hypotheses':sum(not h.strip() for h in external['hypotheses']),'BLEU4':b4(row),
        'ground_truth_native_score_reproduced_exactly':True,'not_claimed_as_independent_SOTA_generator_reproduction':True}
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'rankings/summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    x=np.arange(len(REPRESENTATIONS))
    for name in CHECKPOINTS:ax.plot(x,[points[name,kind] for kind in REPRESENTATIONS],marker='o',label=name)
    ax.set_xticks(x,['Raw','Canonical','Scale0.9','Scale1.1','Shear0.1']);ax.set_ylabel('Native backtranslation BLEU-4')
    ax.set_title('Same four generated motion sets, equivalent6D representations\nPHOENIX TEST n=642; no new model selection');ax.grid(alpha=.2);ax.legend()
    fig.savefig(ROOT/'rankings/representation_comparison.png',dpi=180);fig.savefig(ROOT/'rankings/representation_comparison.pdf');plt.close(fig)
    print('CONFIRMATION_ANALYSIS',json.dumps({'orders':ranks,'attack_summary':result['attacks']}),flush=True)


if __name__=='__main__':main()
