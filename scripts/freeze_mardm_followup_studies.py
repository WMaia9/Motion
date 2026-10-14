"""Freeze completed follow-up studies without touching historical inventories."""
import argparse
import shutil
from pathlib import Path
from freeze_motion_release_studies import closure, read, write, sha
from freeze_generalization_studies import fingerprint_maps

STUDIES = {
    'tmr1024': ('cvpr_tmr1024_controls_20261013', ['run_tmr_targeted_controls', 'verify_tmr1024_controls', 'measure_tmr1024_generic_geometry']),
    'mechanism': ('cvpr_rotation_mechanism_20261013', ['analyze_rotation_mechanism', 'verify_rotation_mechanism']),
    'blocks': ('cvpr_matched_blocks_20261013', ['run_matched_block_audit', 'verify_matched_block_audit', 'test_matched_motion_blocks']),
    'mardm': ('cvpr_mardm_readers_20261013', ['run_mardm_reader_audit', 'verify_mardm_reader_audit']),
}


def freeze(study, cache):
    name, producers = STUDIES[study]; root = Path('reports') / name
    assert not (root / 'manifest_sha256.json').exists(), 'Already frozen'
    if study == 'blocks':
        for reader in ['guo', 'tmr']:
            assert read(root / reader / 'complete.json')['all20_draws_and_all14_conditions']
            assert read(root / reader / 'verification.json')['passed']
        write(root / 'complete.json', {'both_readers_all20_draws_all14_conditions': True})
        write(root / 'verification.json', {'passed': True, 'both_native_reader_verifications_passed': True, 'per_reader': {r:read(root / r / 'verification.json') for r in ['guo', 'tmr']}, 'verification_sources_sha256': {str(Path('scripts') / (n + '.py')): sha(Path('scripts') / (n + '.py')) for n in ['verify_matched_block_audit', 'test_matched_motion_blocks']}})
    assert read(root / 'verification.json')['passed']
    if study == 'tmr1024':assert read(root / 'generic_geometry.json')['passed']
    current = {}; protocol = read(root / 'protocol.json')
    for mapping in fingerprint_maps(protocol):
        for filename, expected in mapping.items():
            if filename not in cache:cache[filename] = sha(filename)
            assert cache[filename] == expected, (filename, 'registered source changed')
            current[filename] = expected
    historical = {}
    if study == 'mardm':
        path = root / 'producer_revisions/protocol_v1.json'; p = read(path)
        for mapping in fingerprint_maps(p):
            for filename, expected in mapping.items():
                source = root / 'producer_revisions/v1.py' if Path(filename).name == 'run_mardm_reader_audit.py' else Path(filename)
                assert sha(source) == expected, (source, 'historical registration changed')
                historical[str(source)] = expected
        assert read(root / 'producer_revisions/amendment.json')['input_arrays_lengths_scores_weights_and_normalization_unchanged']
    names = closure(root, producers + ['freeze_mardm_followup_studies'])
    (root / 'tools').mkdir(exist_ok=True); shutil.copy2('tools/canonicalize_motion.py', root / 'tools/canonicalize_motion.py')
    for folder, filenames in [('configs', ['external_sources.json']), ('docs', ['evaluation-contract.md', 'mardm-followup-plan.md'])]:
        destination = root / folder; destination.mkdir(exist_ok=True)
        for filename in filenames:shutil.copy2(Path(folder) / filename, destination / filename)
    patterns = {'tmr1024':['cvpr_tmr1024*.log'], 'mechanism':['cvpr_rotation_mechanism_v2*.log', 'cvpr_rotation_mechanism_verify*.log'], 'blocks':['cvpr_matched*.log'], 'mardm':['cvpr_mardm_readers_v3*.log', 'cvpr_mardm_verify*.log']}[study]
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    for pattern in patterns:
        for source in Path('/tmp').glob(pattern):shutil.copy2(source, logs / source.name)
    write(root / 'source_provenance.json', {'producers':producers, 'import_closure':names, 'verified_current_registration':current, 'verified_historical_registration':historical, 'executed_verifier_sha256':read(root / 'verification.json').get('verifier_sha256'), 'freeze_source_sha256':sha(__file__), 'no_new_generation_or_training':True, 'partial_failed_attempts_excluded_from_final_endpoints':True})
    inventory = {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest_sha256.json'}
    write(root / 'manifest_sha256.pending', inventory); (root / 'manifest_sha256.pending').replace(root / 'manifest_sha256.json')
    print('MARDM_FOLLOWUP_STUDY_FROZEN', study, len(inventory), sha(root / 'manifest_sha256.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('studies', nargs='+', choices=STUDIES); args = parser.parse_args(); cache = {}
    for study in args.studies:freeze(study, cache)


if __name__ == '__main__':main()
