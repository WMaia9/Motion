"""Export compact frozen study evidence while preserving active manuscript edits.

Unlike the reviewed-delivery exporter, this command never copies paper sources
or reviewed PDFs. Full arrays, weights, logs and licensed assets remain local.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.export_repository_assets import METADATA, ROOT, copy_checked, read, sha, write_json

EXTRA = {
    'cvpr_tmr_audit_20261013': ['native_contract.json'],
    'cvpr_tmr_attack_20261013': [],
    'cvpr_physical_calibration_20261013': [],
    'cvpr_reward_dev_search_20261013': ['selection.json', 'prepared.json'],
    'cvpr_reward_dev_validation_20261013': [],
    'cvpr_blind_transfer_20261013': ['cross_reader_analysis.json', 'cross_reader_protocol.json',
                                  'cross_reader_complete.json', 'mld/protocol.json',
                                  'mld/analysis.json', 'mld/verification.json'],
    'cvpr_motion_defense_cost_20261013': [],
    'cvpr_additional_motion_audit_20261013': ['humanml/momask/protocol.json',
        'humanml/momask/verification.json', 'humanml/dip/protocol.json',
        'humanml/dip/verification.json', 'kit/momask/protocol.json',
        'kit/momask/verification.json', 'kit/real_controls/protocol.json',
        'kit/real_controls/verification.json'],
    'cvpr_motiondiffuse_compiled_20261013': ['humanml_compiled_registration.json',
        'kit_compiled_registration.json', 'humanml/motiondiffuse/protocol.json',
        'humanml/motiondiffuse/verification.json', 'kit/motiondiffuse/protocol.json',
        'kit/motiondiffuse/verification.json'],
}


def main():
    config = read(ROOT / 'configs/reviewed_delivery.json')
    index = read(ROOT / 'results/index.json')
    entries = {item['path']: item for item in index['files']}
    completed = []
    for name, extra in EXTRA.items():
        report = ROOT / 'reports' / name
        manifest_path = report / 'manifest_sha256.json'
        if not manifest_path.is_file():
            continue
        manifest = read(manifest_path)
        assert read(report / 'verification.json')['passed'], name
        for filename in dict.fromkeys((*METADATA, 'source_provenance.json', *extra)):
            source = report / filename
            if not source.is_file():
                continue
            assert source.stat().st_size <= 1024 * 1024, source
            relative = Path(name) / filename
            target = ROOT / 'results' / relative
            if filename == 'manifest_sha256.json':
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            else:
                copy_checked(source, target, report, manifest)
            entries[str(relative)] = {
                'path': str(relative), 'sha256': sha(source),
                'original_path': str(source.relative_to(ROOT)),
                'original_manifest_sha256': sha(manifest_path),
            }
        config.setdefault('additional_metadata', {})[name] = ['source_provenance.json', *extra]
        completed.append(name)
    index.update(files=[entries[k] for k in sorted(entries)],
                 report_count=len({Path(k).parts[0] for k in entries}),
                 last_completed_study_export_date='2026-10-13')
    write_json(ROOT / 'results/index.json', index)
    write_json(ROOT / 'configs/reviewed_delivery.json', config)
    print(json.dumps({'frozen_studies_exported': completed,
                      'active_sources_and_reviewed_PDFs_preserved': True}, indent=2))


if __name__ == '__main__':
    main()
