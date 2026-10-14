"""Validate directional ratios and the controlled natural-sign claims."""
import re
import sys
from pathlib import Path
from freeze_motion_release_studies import read,write,sha

P=Path('paper/current');D=Path('reports/cvpr_sign_stability_review_20261014');R=Path('reports/cvpr_sign_stability_20261014')


def main():
    assert not (D/'manifest_sha256.json').exists();m=read(R/'manifest_sha256.json')
    for name in ['analysis.json','rotation_ratio.json','verification.json']:assert sha(R/name)==m[name]
    a=read(R/'analysis.json');v=read(R/'verification.json');assert v['passed'] and v['allclip_condition_reconstructions']==26760 and v['native_fixture_decodes_exact']==648
    presentation=read(D/'presentation_evidence.json');assert presentation['passed']
    for field in ['inputs_sha256','generated_assets']:
        for p,h in presentation[field].items():assert sha(p)==h,p
    main=(P/'main.tex').read_text();supp=(P/'supplement.tex').read_text();fragment=(P/'sign_stability_supplement.tex').read_text()
    abstract=main.split('\\begin{abstract}')[1].split('\\end{abstract}')[0];numbers=re.findall(r'\b\d[\d,]*(?:\.\d+)?\b',abstract);assert len(numbers)<=6,numbers
    assert 'Physical controls also change decoded sentences' in abstract and 'equal-norm Gaussian controls' in abstract
    assert 'same architecture' in abstract and 'rotation-only domains' in abstract
    assert '126-dimensional rotation block' in main and 'about 7.7 times' in main and 'direction-specific' in main
    assert 'Physical controls of equal norm have larger mean changes' in main
    assert 'do not isolate all reader sensitivity to encoding' in main
    assert 'three fixed seeds' in main and 'cluster-contrast intervals exclude zero' in main
    assert 'CSL\'s changed-fraction contrast includes zero' in fragment
    assert 'do not estimate model-training uncertainty' in (P/'table_sign_stability_intervals.tex').read_text()
    assert '\\input{sign_stability_supplement}' in supp
    table=(P/'table_editorial_sign.tex').read_text()
    for corpus,row in a['results'].items():
        assert row['all_matched_physical_clips_feasible']
        for condition in ['canonical','gaussian_support_mean','physical_matched']:
            r=row['primary_summary'][condition];assert f"{100*r['token_changed_fraction']:.1f} / {r['mean_normalized_edit_distance']:.3f}" in table
        for endpoint,q in row['canonical_minus_control']['gaussian_support_mean'].items():assert q['family8_CI95'][0]>0
        # Preserve the stronger physical control's effects, including the
        # inconclusive changed-fraction contrast on CSL.
        assert row['primary_summary']['physical_matched']['token_changed_fraction']>row['primary_summary']['canonical']['token_changed_fraction']
    ratio=read(R/'rotation_ratio.json');assert all(abs(x['ratio_of_mean_FID_changes']-7.7)<.1 for x in ratio['directions'].values())
    assert 'not a new prospective hypothesis test' in fragment
    assert 'semantic accuracy remain separate requirements' in fragment
    identifiers=re.compile(r'perceptron-lab|/mnt/|/home/|unused autograd|earlier reviewer-response',re.I)
    assert not [p.name for p in P.glob('*.tex') if identifiers.search(p.read_text())]
    assert '\\author{Anonymous authors}' in main and '\\author{Anonymous authors}' in supp
    # Execute the actual printed portable interface against a small native
    # schema; this catches import/shape errors after presentation shortening.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    code=main.split('\\begin{verbatim}')[1].split('\\end{verbatim}')[0];scope={};exec(code,scope)
    import numpy as np
    x=np.zeros((2,4,263),dtype='float32');x[...,67:193]=np.tile([1.,0,0,0,1,0],21)
    out,mask=scope['reader_input'](x,np.array([4,2]),np.zeros(263),np.ones(263))
    assert out.shape==x.shape and mask.shape==(2,4) and np.all(out[1,2:]==0)
    write(D/'claim_audit.json',{'passed':True,'abstract_numbers':numbers,'same_sign_direction_specific_ratio_checked':True,'all_gaussian_fields_and_both_physical_controls_reported':True,'changed_fraction_and_edit_distance_both_in_main':True,'positive_Gaussian_contrasts_and_stronger_physical_effects_both_disclosed':True,'CSL_inconclusive_physical_changed_fraction_contrast_disclosed':True,'no_general_encoding_specificity_or_semantic_accuracy_claim':True,'printed_portable_interface_smoke_test_passed':True,'audit_sha256':sha(__file__)})
    write(D/'manuscript_anonymity_check.json',{'passed':True,'scope':'manuscript sources; experimental repository not certified anonymous','authors':'Anonymous authors','submission_ID':'DRAFT'})
    write(D/'evidence_map.json',{'parent_delivery':'reports/cvpr_coherence_review_20261014','parent_manifest_sha256':sha('reports/cvpr_coherence_review_20261014/manifest_sha256.json'),'new_study_manifest_sha256':sha(R/'manifest_sha256.json'),'presentation_inputs':presentation['inputs_sha256'],'all_prior_outcomes_retained':True})
    print('SIGN_STABILITY_CLAIMS_VERIFIED',flush=True)


if __name__=='__main__':main()
