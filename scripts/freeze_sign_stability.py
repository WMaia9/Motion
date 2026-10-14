"""Freeze completed real-sign stability controls and the directional FID ratio."""
import shutil
from pathlib import Path
from freeze_motion_release_studies import closure,read,write,sha

R=Path('reports/cvpr_sign_stability_20261014')


def main():
    assert not (R/'manifest_sha256.json').exists()
    v=read(R/'verification.json');assert v['passed'] and v['all28_adjusted_intervals_recomputed']
    p=read(R/'protocol.json')
    for field in ['inputs_sha256','scientific_sources_sha256']:
        for filename,h in p[field].items():assert sha(filename)==h
    for folder,h in p['source_manifest_sha256'].items():assert sha(Path(folder)/'manifest_sha256.json')==h
    for corpus in p['corpora']:assert read(R/corpus/'complete.json')['all8_conditions_full_nativeTEST']
    names=closure(R,['run_sign_stability_audit','sign_stability_controls','test_sign_stability_controls','analyze_sign_stability','verify_sign_stability','freeze_sign_stability'])
    (R/'docs').mkdir(exist_ok=True);shutil.copy2('docs/sign-stability-plan.md',R/'docs/sign-stability-plan.md')
    (R/'logs').mkdir(exist_ok=True)
    for filename in Path('/tmp').glob('cvpr_sign_stability*20261014.log'):shutil.copy2(filename,R/'logs'/filename.name)
    write(R/'complete.json',{'all_corpora_all8_conditions_complete_verified':True,'no_control_output_discarded':True})
    write(R/'summary.json',{'scope':'real sign-decoder response to canonicalization, norm-matched Gaussian and gauge-preserving physical controls','all26760_full_TEST_decodes':True,'same_raw_and_canonical_strings_as_prior_frozen_results':True,'all28_adjusted_intervals_verified':True,'no_general_encoding_specificity_or_human_semantic_accuracy_claim':True,'no_training_new_generators_or_reward_fits':True})
    write(R/'source_provenance.json',{'import_closure':names,'registered_sources_unchanged':True,'independent_verifier_sha256':v['source_sha256'],'freeze_source_sha256':sha(__file__)})
    m={str(f.relative_to(R)):sha(f) for f in sorted(R.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='manifest_sha256.json'}
    write(R/'manifest_sha256.pending',m);(R/'manifest_sha256.pending').replace(R/'manifest_sha256.json')
    print('SIGN_STABILITY_FROZEN',len(m),sha(R/'manifest_sha256.json'),flush=True)


if __name__=='__main__':main()
