"""Freeze verified studies; never rewrite an existing evidence inventory."""
import argparse
import json
import shutil
from pathlib import Path

from freeze_motion_release_studies import closure, read, sha, write


STUDIES = {
    'tmr': ('cvpr_tmr_audit_20261013', ['run_tmr_release_audit', 'verify_tmr_consistency_audits']),
    'tmr_attack': ('cvpr_tmr_attack_20261013', ['run_tmr_targeted_controls', 'verify_tmr_consistency_audits']),
    'physical': ('cvpr_physical_calibration_20261013', ['run_physical_rotation_calibration', 'verify_physical_calibration']),
    'reward_dev': ('cvpr_reward_dev_search_20261013', ['run_reward_dev_search', 'verify_generalization_controls']),
    'reward_validation': ('cvpr_reward_dev_validation_20261013', ['run_reward_dev_validation', 'verify_generalization_controls']),
    'blind_transfer': ('cvpr_blind_transfer_20261013', ['run_humanml_blind_fid', 'score_humanml_blind_fid', 'score_blind_cross_reader', 'verify_generalization_controls']),
    'defense': ('cvpr_motion_defense_cost_20261013', ['measure_motion_defense_cost', 'verify_generalization_controls']),
    'additional': ('cvpr_additional_motion_audit_20261013', ['run_additional_motion_audit', 'additional_motion_adapters', 'coordinate_additional_motion_audit', 'verify_additional_motion_audits', 'download_kit_native']),
    'motiondiffuse': ('cvpr_motiondiffuse_compiled_20261013', ['run_motiondiffuse_compiled_audit', 'run_motiondiffuse_vectorized_audit', 'motiondiffuse_vectorized_adapter', 'verify_additional_motion_audits']),
}


def fingerprint_maps(value):
    """Find path->SHA256 maps in the differently named producer protocols."""
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if (key.endswith('sha256') and isinstance(item, dict) and item
                and all(isinstance(v, str) and len(v) == 64 for v in item.values())):
            yield item
        elif isinstance(item, dict):
            yield from fingerprint_maps(item)


def freeze(study, check_only, cache):
    name, producers = STUDIES[study]
    root = Path('reports') / name
    manifest = root / 'manifest_sha256.json'
    if manifest.exists():
        raise ValueError(f'Already frozen: {root}; use the read-only archive verifier')
    assert read(root / 'verification.json')['passed'], root
    checks = {}
    historical = {}
    protocols = list(root.rglob('*protocol.json')) + list(root.glob('*registration.json'))
    for path in protocols:
        for mapping in fingerprint_maps(read(path)):
            for filename, expected in mapping.items():
                source = Path(filename)
                if 'producer_revisions' in path.parts and (path.parent / source.name).is_file():
                    archived = path.parent / source.name
                    assert sha(archived) == expected, (path, archived)
                    historical[str(archived.relative_to(root))] = expected
                    continue
                assert source.is_file(), (path, source)
                if filename not in cache:
                    cache[filename] = sha(source)
                assert cache[filename] == expected, (path, source, 'fingerprint mismatch')
                assert checks.get(filename, expected) == expected
                checks[filename] = expected
    if study == 'motiondiffuse':
        for dataset in ['humanml', 'kit']:
            cell = root / dataset / 'motiondiffuse'
            assert read(cell / 'complete.json')['twenty_native_repetitions']
            assert len(list(cell.glob('rep_*/metrics.json'))) == 20
            assert read(cell / 'verification.json')['passed']
            assert read(cell / 'native_mask_parity.json')['compiled_forward_max_difference'] < .0005
    if study == 'reward_dev':
        assert len(read(root / 'selection.json')['all_DEV_rows']) == 8
    elif study == 'reward_validation':
        complete = read(root / 'complete.json')
        assert complete['workflow_complete']
        assert not complete['valid_reward_pilot_obtained']
        assert read(root / 'analysis.json')['no_TEST_training_or_inference']
    elif study == 'additional':
        assert read(root / 'complete.json')['all_scientific_endpoints_for_this_root_complete']
    print('REGISTERED_HASHES_VERIFIED', name, len(checks), flush=True)
    if check_only:
        return
    names = closure(root, producers + ['freeze_generalization_studies'])
    # Import-closure collection follows scripts; preserve the portable utility too.
    target = root / 'tools'
    target.mkdir(exist_ok=True)
    shutil.copy2('tools/canonicalize_motion.py', target / 'canonicalize_motion.py')
    for folder, filenames in [('configs', ['external_sources.json']),
                              ('docs', ['evaluation-contract.md', 'generalization-plan.md'])]:
        destination = root / folder
        destination.mkdir(exist_ok=True)
        for filename in filenames:
            shutil.copy2(Path(folder) / filename, destination / filename)
    # A central /tmp log can be active. Only copy logs belonging to this closed study.
    patterns = {
        'tmr': ['cvpr_tmr_audit*.log', 'cvpr_tmr_verification*.log'],
        'tmr_attack': ['cvpr_tmr_attack*.log', 'cvpr_tmr_verification*.log'],
        'physical': ['cvpr_physical*.log'],
        'reward_dev': ['cvpr_reward_dev_search*.log', 'cvpr_generalization_controls*.log'],
        'reward_validation': ['cvpr_reward_dev_validation*.log', 'cvpr_generalization_controls*.log'],
        'blind_transfer': ['cvpr_blind_transfer*.log', 'cvpr_blind_cross_reader*.log', 'cvpr_generalization_controls*.log'],
        'defense': ['cvpr_motion_defense_cost*.log', 'cvpr_generalization_controls*.log'],
        'additional': ['cvpr_additional*.log'],
        'motiondiffuse': ['cvpr_motiondiffuse_compiled*.log', 'cvpr_motiondiffuse_verification*.log'],
    }[study]
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    for pattern in patterns:
        for path in Path('/tmp').glob(pattern):
            shutil.copy2(path, logs / path.name)
    write(root / 'source_provenance.json', {
        'producers': producers, 'import_closure': names,
        'verified_registered_inputs_and_assets': checks,
        'verified_historical_producer_fingerprints': historical,
        'registered_sources_and_assets_match': True,
        'portable_utility_sha256': sha(target / 'canonicalize_motion.py'),
        'freeze_source_sha256': sha(__file__),
        'historical_executed_verifiers': 'verification_revisions/ where present; preserved without replacing scientific outputs',
        'excluded_attempts': 'Incomplete eager MotionDiffuse attempts are execution records, never final study endpoints',
        'human_authors_and_registration_deferred': True,
    })
    inventory = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts
                 and p.name != 'manifest_sha256.json'}
    temporary = root / 'manifest_sha256.pending'
    write(temporary, inventory)
    temporary.replace(manifest)
    print('FROZEN_GENERALIZATION', name, len(inventory), sha(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('studies', nargs='+', choices=STUDIES)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cache = {}
    for study in args.studies:
        freeze(study, args.check_only, cache)


if __name__ == '__main__':
    main()
