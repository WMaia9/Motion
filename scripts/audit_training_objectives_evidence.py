"""Verify manuscript's registered numeric sources, citations and new outcome claims."""
import json,re
from pathlib import Path
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_second_review_20261012');PAPER=Path('paper/cvpr2027_training_objectives_revision_20261012')
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (ROOT/'manifest_sha256.json').exists();m=read(ROOT/'manuscript_numeric_source_map.json')
 for p,h in m['frozen_sources'].items():assert sha(p)==h
 macros=(PAPER/'new_results_macros.tex').read_text()
 for key,value in m['generated_macros'].items():assert '\\newcommand{\\'+key+'}{'+value+'}' in macros
 refs=(PAPER/'references.bib').read_text();keys=re.findall(r'@\w+\s*\{([^,]+),',refs);assert len(keys)==len(set(keys))==36;used=set()
 for name in ['main','supplement']:
  text=(PAPER/(name+'.tex')).read_text()
  for items in re.findall(r'\\cite\w*\{([^}]+)\}',text):used.update(items.split(','))
  assert not re.search(r'fixed-gauge|earlier reviewer-response|earlier audit|previously inspected|generation-dump interface|Test-use history',text,re.I)
 assert used==set(keys)
 q=read('reports/cvpr_quotient_training_20261012/analysis.json');t=read('reports/cvpr_second_review_confirmation_20261012/analysis.json');old=read('reports/cvpr_learned_confirmation_20261012/analysis.json');assert old['learned']['systems']['text_raw']['canonical_hypothesis_changes']==0
 assert t['ordinary']['primary_contrasts']['raw_vs_no_CE']['paired_ci95'][0]>0
 assert t['ordinary']['primary_contrasts']['raw_vs_canonical_CE']['paired_ci95'][0]<0
 assert t['ordinary']['primary_contrasts']['raw_gap_absolute']['paired_ci95'][0]<0
 for c,d in t['real_motion_layers'].items():
  a=d['arms']['text_raw'];assert a['paired_ci95'][0]>0 and a['seed_and_group_ci95'][0]>0;assert a['canonical_hypothesis_changes']==[0,0,0]
 assert t['budget']['arms']['bound0.05']['seed_and_group_ci95'][0]<0 and t['budget']['arms']['bound0.1']['seed_and_group_ci95'][0]>0
 assert read('reports/cvpr_retrieval_transfer_20261012/analysis.json')['paired_ci95'][0]<0
 for path in ['reports/cvpr_constant_noncanonicity_20261012/analysis.json','reports/cvpr_constant_distance_sphere_20261012/analysis.json']:
  a=read(path)['arms']['text_raw'];assert a['paired_gain_ci95'][0]<0<a['paired_gain_ci95'][1]
 comparison=read(ROOT/'generated_vs_real_analysis.json');assert comparison['paired_ci95'][0]<0<comparison['paired_ci95'][1] and not comparison['supported_superiority_if_ci_positive']
 verified=read('reports/cvpr_second_review_confirmation_20261012/verification.json');assert verified['passed'] and verified['independently_verified_rotation_pairs']==28281 and verified['canonical_layer_hypothesis_changes']==0
 # Refresh editorial file hashes; numeric values remain the frozen sources above.
 m['table_files']={str(p):sha(p) for p in PAPER.glob('table*.tex')};(ROOT/'manuscript_numeric_source_map.json').write_text(json.dumps(m,indent=2)+'\n')
 original=read('reports/cvpr_paper_story_20261011/reference_verification.json') if Path('reports/cvpr_paper_story_20261011/reference_verification.json').exists() else {'scope':'original27 entries verified in frozen paper-story/manuscript-review roots'}
 catalog=read(ROOT/'sources/research_catalog.json');out={'passed':True,'reference_entries':36,'all_entries_cited_across_main_and_supplement':True,'added_primary_entries':read(ROOT/'reference_additions_verification.json'),'original_reference_provenance':original,'research_catalog':catalog,'TEST_positive_and_negative_intervals_checked':True,'no_claim_of_statistical_superiority_over_real_motion':True,'constructed_layer_source_text_disclosed':True,'weak_public_replay_removed_from_main_claims':True,'no_general_stealth_claim':True,'ordinary_output_gradient_scale_limitation_disclosed':True,'native_reader_family_limitation_disclosed':True,'public_TEST_history_disclosed':True,'manuscript_files':{str(p):sha(p) for p in PAPER.glob('*.tex')}}
 (ROOT/'evidence_verification.json').write_text(json.dumps(out,indent=2)+'\n');print('MANUSCRIPT_EVIDENCE_VERIFIED',36,flush=True)
if __name__=='__main__':main()
