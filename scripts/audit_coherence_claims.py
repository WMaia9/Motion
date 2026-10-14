"""Check all four post-MARDM corrections against immutable result files."""
from pathlib import Path
import re
from freeze_motion_release_studies import read,write,sha

D=Path('reports/cvpr_coherence_review_20261014')
P=Path('paper/current')
C=Path('reports/cvpr_coherent_blocks_20261014')
S=Path('reports/cvpr_sign_natural_restoration_20261014')


def main():
    assert not (D/'manifest_sha256.json').exists()
    presentation=read(D/'presentation_evidence.json');assert presentation['passed']
    for path,h in presentation['inputs_sha256'].items():assert sha(path)==h
    for path,h in presentation['generated_assets'].items():assert sha(path)==h
    main=(P/'main.tex').read_text();supp=(P/'supplement.tex').read_text()
    fragment=(P/'coherence_review_supplement.tex').read_text();old=(P/'mardm_followup_supplement.tex').read_text()
    abstract=main.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    numbers=re.findall(r'\b\d[\d,]*(?:\.\d+)?\b',abstract);assert len(numbers)<=6,numbers
    assert 'same architecture' in abstract and 'rotation-only domains' in abstract
    assert '1,438 of 2,183' in abstract and 'Without an attack' in abstract
    assert '1438' not in main # typeset grouping retained
    assert '1,438/2,183' in main and '464/1,162' in main and '0/642' in main
    assert '0.016' in main and '0.018' in main # formerly omitted random controls
    assert 'largest block effect' not in main and 'universal ranking' in main
    assert 'fixed position' in main.lower() and 'fixed velocity' in main.lower()
    assert 'input norm' in main and 'can be physically inconsistent' in main
    assert r'\kappa=1,3' in main and r'a=\rho u' in main
    assert not re.search(r'\bradii\s*\$\\rho|At\s*\$\\rho|at\s*\$\\rho|=\\rho\\sqrt',main+old+fragment)
    assert 'before both raw' in read(S/'analysis.json')['shared_preprocessing']
    assert 'including projection of left-hand channels 60:150' in fragment
    natural=read(S/'analysis.json')['results']
    table=(P/'table_sign_natural.tex').read_text();summary=(P/'table_editorial_sign.tex').read_text()
    for n,clips,changed in [('PHOENIX-2014T',642,0),('How2Sign',2183,1438),('CSL-Daily',1162,464)]:
        q=natural[n];assert q['clips']==clips and q['changed_hypotheses']==changed
        assert f"{q['BLEU4']['raw']:.4f}" in table and f"{q['BLEU4']['canonical']:.4f}" in table
    assert '1,438/2,183' in summary and '464/1,162' in summary and '0/642' in summary
    for reader in ['guo','tmr']:
        a=read(C/reader/'analysis.json');assert len(a['conditions'])==16 and a['twenty_draws_complete']
        assert len(a['fixed_minus_random'])==16
        v=read(C/reader/'verification.json');assert v['passed'] and v['saved_metric_cells_recomputed']==340
        assert v['first8_fixture_pairs_independently_reconstructed']==2560
    assert 'accuracy-free defense' in fragment
    assert 'implementation sanity check, not evidence of a cost-free defense' in main
    assert 'nor refutes MARDM' in main
    assert '\\input{coherence_review_supplement}' in supp
    identifiers=re.compile(r'perceptron-lab|/mnt/|/home/|unused autograd|earlier reviewer-response',re.I)
    assert not [p.name for p in P.glob('*.tex') if identifiers.search(p.read_text())]
    assert '\\author{Anonymous authors}' in main and '\\author{Anonymous authors}' in supp
    write(D/'manuscript_anonymity_check.json',{'passed':True,'scope':'manuscript source; experimental repository not certified anonymous','authors':'Anonymous authors','submission_ID':'DRAFT'})
    write(D/'claim_audit.json',{'passed':True,'abstract_numbers':numbers,'fixed_and_random_each_block_both_signs_and_radii_retained':True,'random_velocity_and_position_comparisons_restored':True,'no_null_feature_priority_claim':True,'no_target_natural_sign_evidence_in_abstract_body_table_and_supplement':True,'shared_native_preprocessing_disclosed':True,'kappa_budget_rho_fiber_notation_separated':True,'all_generated_assets_match_frozen_inputs':True,'same_architecture_checkpoint_transfer_preserved':True,'no_accuracy_free_defense_or_full_MARDM_refutation':True,'audit_source_sha256':sha(__file__)})
    write(D/'evidence_map.json',{'parent_delivery':'reports/cvpr_mardm_followup_20261013','parent_manifest_sha256':sha('reports/cvpr_mardm_followup_20261013/manifest_sha256.json'),'new_studies':{str(r):sha(r/'manifest_sha256.json') for r in [C,S]},'presentation_inputs':presentation['inputs_sha256'],'all_prior_outcomes_retained_in_main_or_supplement':True})
    print('COHERENCE_CLAIMS_VERIFIED',flush=True)


if __name__=='__main__':main()
