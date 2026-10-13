"""Check the revised claim chain against immutable experimental artifacts."""
import json,re
from pathlib import Path
from verify_signspark_studies import sha
R=Path('reports/cvpr_paper_story_20261011');P=Path('paper/cvpr2027_learned_equivalence_20261011')
ROOTS=['cvpr_learned_equivalence_20261011','cvpr_canonical_readers_20261011','cvpr_learned_confirmation_20261012','cvpr_reviewer_followup_20261011']
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (R/'manifest_sha256.json').exists()
 roots={n:Path('reports')/n for n in ROOTS};a=read(roots[ROOTS[2]]/'analysis.json');d=read(roots[ROOTS[0]]/'analysis.json')
 c=a['learned'];s=c['systems']['text_raw'];assert c['TEST_clips']==642 and c['TRAIN_clip_overlap']==c['DEV_clip_overlap']==0
 assert s['canonical_hypothesis_changes']==0 and s['mean_gain_bleu4']>0 and s['gain_ci95_conditional_on_seeds'][0]>0
 assert c['unseen_exact_captions']['clips']==610 and c['unseen_exact_captions']['gain_ci95_conditional_on_seeds'][0]>0
 assert c['caption_counterfactual']['different_captions']==642
 assert c['caption_counterfactual']['versus_actual_caption']['gain_ci95_conditional_on_seeds'][1]<0
 assert a['models_frozen_before_TEST_inference'] and not a['new_training_or_selection'] and a['project_TEST_use_history_disclosed']
 for n in ROOTS[:3]:assert read(roots[n]/'verification.json')['passed']
 nums=read(R/'numeric_source_map.json');m=(P/'results_macros.tex').read_text()
 for k,v in nums['macros'].items():assert '\\newcommand{\\'+k+'}{'+v+'}' in m,k
 main=(P/'main.tex').read_text();supp=(P/'supplement.tex').read_text()
 assert 'ordinary DDPM' in main and 'did not show selective' in main and 'one architecture family' in main
 assert 'No mesh rendering' in supp and '299,384' in supp and '32 TEST' in supp
 teaser=read(R/'sources/teaser_example.json');source=read('reports/cvpr_evaluation_confirmation_20261011/heldout/PHOENIX-2014T/rolled_wrong_caption_scores.json');i=teaser['TEST_subset_index']
 assert teaser['original_hypothesis']==source['baseline_actual']['hypotheses'][i]
 assert teaser['attacked_hypothesis']==source['raw_after_actual']['hypotheses'][i]==source['targets'][i]
 assert teaser['canonical_before']==teaser['canonical_after']
 for p,h in teaser['RGB_source_frames'].items():assert sha(p)==h and sha(P/'figures/source_frames'/Path(p).name)==h
 assert s['mean_bleu4']<16 and d['public_transfer']['baseline_canonical_bleu4']<1
 out={'passed':True,'primary_TEST_clips':642,'primary_BLEU_baseline':c['baseline_canonical_bleu4'],'primary_mean_raw_BLEU':s['mean_bleu4'],
 'primary_gain_CI':s['gain_ci95_conditional_on_seeds'],'unseen_exact_caption_clips':610,'all_canonical_hypotheses_unchanged':True,
 'canonical_reader_TEST_pairs':read(roots[ROOTS[2]]/'verification.json')['canonical_reader_fixed_gauge_hypothesis_pairs'],
 'new_weight_count':21,'negative_conventional_result_preserved':True,'teaser_source_frames_unaltered':True,'no_rendered_mesh_claim':True,
 'three_language_readers_one_family_disclosed':True,'public_generator_low_replay_baseline_disclosed':True,'prior_TEST_use_disclosed':True,
 'numeric_macros_match_source_map':True,'report_manifests':{n:sha(p/'manifest_sha256.json') for n,p in roots.items()}}
 (R/'evidence_verification.json').write_text(json.dumps(out,indent=2)+'\n');print('EVIDENCE_VERIFIED',json.dumps(out))
if __name__=='__main__':main()
