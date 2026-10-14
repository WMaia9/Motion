"""Freeze the finite post-MARDM controls, preserving prior studies unchanged."""
import argparse
import shutil
from pathlib import Path
from freeze_motion_release_studies import closure,read,write,sha

STUDIES={
 'coherent':('cvpr_coherent_blocks_20261014',['run_coherent_block_audit','coherent_motion_blocks','verify_coherent_block_audit','test_coherent_motion_blocks']),
 'sign':('cvpr_sign_natural_restoration_20261014',['restore_sign_natural_evidence'])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',choices=STUDIES);args=p.parse_args()
    name,producers=STUDIES[args.study];R=Path('reports')/name;assert not (R/'manifest_sha256.json').exists()
    verified={}
    if args.study=='coherent':
        for reader in ['guo','tmr']:
            assert read(R/reader/'complete.json')['all20_draws_and_all16_conditions']
            assert read(R/reader/'verification.json')['passed']
        write(R/'verification.json',{'passed':True,'both_native_readers_all20_draws_verified':True,'per_reader':{r:read(R/r/'verification.json') for r in ['guo','tmr']}})
        write(R/'complete.json',{'both_readers_all20_draws_all16_conditions':True,'no_endpoint_selection':True})
        protocol=read(R/'protocol.json')
        for name,h in protocol['producer_sha256'].items():assert sha(name)==h;verified[name]=h
        for name,h in protocol['source_manifests'].items():assert sha(Path(name)/'manifest_sha256.json')==h
        old=read(R/'producer_revisions/protocol_v1.json')
        for name,h in old['producer_sha256'].items():
            saved=R/'producer_revisions'/(Path(name).stem+'_v1.py')
            source=saved if saved.exists() else Path(name)
            assert sha(source)==h,(source,'v1 fingerprint mismatch')
        write(R/'summary.json',{'scope':'matched temporal-direction sensitivity, not full physical block consistency','primary_family_per_reader':32,'secondary_fixed_minus_random_family_per_reader':32,'all_old_controls_retained':'cvpr_matched_blocks_20261013','intervention_revision_only':'contact float32 budget tolerance; prior incomplete attempts retained','no_new_generators_training_model_seeds_or_reward_fits':True})
    else:
        assert read(R/'verification.json')['passed']
        a=read(R/'analysis.json')
        for name,h in a['inputs_sha256'].items():assert sha(name)==h;verified[name]=h
        for name,h in a['source_manifest_sha256'].items():assert sha(name)==h
        write(R/'summary.json',{'scope':'real native-preprocessed sign inputs, no attack or learned layer','results':{n:{k:r[k] for k in ['clips','changed_hypotheses','BLEU4']} for n,r in a['results'].items()},'no_new_decode_training_or_layer':True})
    names=closure(R,producers+['freeze_coherence_studies'])
    (R/'docs').mkdir(exist_ok=True);shutil.copy2('docs/coherence-review-plan.md',R/'docs/coherence-review-plan.md')
    logs=R/'logs';logs.mkdir(exist_ok=True)
    if args.study=='coherent':
        for pattern in ['cvpr_coherent_*v2_20261014.log','cvpr_coherent_verify_*_20261014.log']:
            for file in Path('/tmp').glob(pattern):shutil.copy2(file,logs/file.name)
    write(R/'source_provenance.json',{'import_closure':names,'verified_registered_sources':verified,'freeze_source_sha256':sha(__file__),'prior_frozen_roots_untouched':True})
    m={str(file.relative_to(R)):sha(file) for file in sorted(R.rglob('*')) if file.is_file() and '__pycache__' not in file.parts and file.name!='manifest_sha256.json'}
    write(R/'manifest_sha256.pending',m);(R/'manifest_sha256.pending').replace(R/'manifest_sha256.json')
    print('COHERENCE_STUDY_FROZEN',args.study,len(m),sha(R/'manifest_sha256.json'),flush=True)


if __name__=='__main__':main()
