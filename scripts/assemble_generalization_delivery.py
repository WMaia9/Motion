"""Freeze a visually reviewed complete delivery, then export its reviewed paper.

Requires the exact final PDF hashes in visual_review.json. It does not submit
the paper, register authors or replace earlier scientific evidence archives.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from freeze_motion_release_studies import closure, read, sha, write
from freeze_generalization_studies import STUDIES

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'reports/cvpr_generalization_review_20261013'
P = ROOT / 'paper/current'
B = ROOT / '.build/paper'


def main():
    global D
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, default=Path('reports/cvpr_generalization_review_20261013'))
    parser.add_argument('--review-document', type=Path, default=Path('docs/generalization-review.md'))
    parser.add_argument('--editorial', action='store_true', help='Archive an editorial revision using unchanged frozen experiments')
    args = parser.parse_args()
    D = (ROOT / args.delivery).resolve()
    document = (ROOT / args.review_document).resolve()
    assert D.parent == ROOT / 'reports' and document.parent == ROOT / 'docs'
    if args.editorial:
        assert read(D / 'editorial_presentation_evidence.json')['passed']
        assert read(D / 'editorial_claim_audit.json')['passed']
    assert not (D / 'manifest_sha256.json').exists(), 'Delivery already frozen'
    assert read(D / 'presentation_status.json')['final_endpoints_complete']
    assert not read(D / 'presentation_status.json')['pending']
    visual = read(D / 'visual_review.json')
    assert visual['passed'] and visual['all_final_pages_inspected']
    layout = read(D / 'layout_verification.json')
    for name in ['main', 'supplement']:
        digest = sha(B / (name + '.pdf'))
        assert visual[name + '_pdf_sha256'] == digest
        assert layout[name]['pdf_sha256'] == digest
        assert layout[name]['latex_reference_and_box_checks_passed']
    assert layout['main']['content_pages'] <= 8
    assert read(D / 'anonymous_utility_verification.json')['utility_tests_passed']
    study_manifest_hashes = {}
    for name, _ in STUDIES.values():
        path = ROOT / 'reports' / name
        assert read(path / 'verification.json')['passed'], name
        study_manifest_hashes[name] = sha(path / 'manifest_sha256.json')
    md = read(ROOT / 'reports/cvpr_motiondiffuse_compiled_20261013/analysis.json')
    assert md['all20repetitions_each']
    reward = read(ROOT / 'reports/cvpr_reward_dev_search_20261013/selection.json')
    assert reward['chosen'] is None and reward['qualified'] == 0
    # Archive current sources; working directory remains paper/current.
    snapshot = D / 'manuscript'
    snapshot.mkdir(exist_ok=True)
    for suffix in ['*.tex', '*.bib', '*.sty', '*.bst']:
        for source in P.glob(suffix):
            shutil.copy2(source, snapshot / source.name)
    figures = snapshot / 'figures'
    figures.mkdir(exist_ok=True)
    for source in (P / 'figures').glob('*.pdf'):
        shutil.copy2(source, figures / source.name)
    build = snapshot / 'build'
    build.mkdir(exist_ok=True)
    for name in ['main', 'supplement']:
        shutil.copy2(B / (name + '.pdf'), build / (name + '.pdf'))
    with zipfile.ZipFile(D / 'manuscript_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(snapshot.rglob('*')):
            if source.is_file() and 'build' not in source.parts:
                archive.write(source, source.relative_to(snapshot))
    shutil.copy2(document, D / document.name)
    write(D / 'summary.json', {
        'scope': 'representation inconsistency, not better motion or systematic score inflation',
        'new_studies': study_manifest_hashes,
        'additional_methods': read(D / 'additional_methods_analysis.json'),
        'main_content_pages': layout['main']['content_pages'],
        'supplement_content_pages': layout['supplement']['content_pages'],
        'reference_entries': len(re.findall(r'@\w+\s*\{([^,]+),', (P / 'references.bib').read_text())),
        'review_kind': 'editorial revision, unchanged frozen experiments' if args.editorial else 'completed generalization delivery',
        'no_qualified_reward_pilot': True,
        'routine_training_exploitation_unconfirmed': True,
        'authors_registration_and_human_cold_review_deferred': True,
        'format_2027_provisional_official_2026_kit': True,
        'submission_or_publication': False,
    })
    write(D / 'verification.json', {
        'passed': True, 'all_nine_new_study_roots_frozen_and_verified': True,
        'all_native_twenty_draw_endpoints_completed': True,
        'all_outcomes_including_negative_controls_preserved': True,
        'exact_final_PDF_visual_and_layout_checks_passed': True,
        'portable_utility_tests_passed': True,
        'no_valid_reward_optimization_claim': True,
        'previous_frozen_integrity_record': 'frozen_integrity_20261013.json',
    })
    names = closure(D, ['assemble_generalization_delivery',
        'build_generalization_presentation', 'plot_generalization_teaser',
        *(['build_editorial_presentation'] if args.editorial else []),
        'audit_current_motion_layout', 'freeze_generalization_studies'])
    (D / 'tools').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'tools/canonicalize_motion.py', D / 'tools/canonicalize_motion.py')
    write(D / 'source_provenance.json', {
        'current_working_manuscript': 'paper/current',
        'snapshot_is_archive_not_another_working_location': True,
        'import_closure': names, 'assembly_source_sha256': sha(__file__),
        'frozen_study_manifests': study_manifest_hashes,
        'portable_utility_sha256': sha(D / 'tools/canonicalize_motion.py'),
    })
    inventory = {str(path.relative_to(D)): sha(path) for path in sorted(D.rglob('*'))
                 if path.is_file() and '__pycache__' not in path.parts
                 and path.name != 'manifest_sha256.json'}
    write(D / 'manifest_sha256.json', inventory)
    config_path = ROOT / 'configs/reviewed_delivery.json'
    config = read(config_path)
    config.update(review_date='2026-10-13', report=str(D.relative_to(ROOT)),
                  paper_source=str(snapshot.relative_to(ROOT)),
                  documents={document.name: document.name},
                  figure_scope='Procedural native RIC body rendering; represented rotations/FK verified separately; aggregate target-aware retrieval, no licensed mesh or human quality claim')
    config.setdefault('additional_metadata', {})[D.name] = [
        'source_provenance.json', 'primary_sources.json', 'layout_verification.json',
        'visual_review.json', 'evidence_map.json', 'teaser_evidence.json',
        'additional_methods_analysis.json', 'completed_studies.json',
        'anonymous_utility_verification.json',
        *(['writing_sources.json', 'editorial_change_map.json',
           'editorial_presentation_evidence.json', 'editorial_claim_audit.json',
           'manuscript_anonymity_check.json'] if args.editorial else []),
    ]
    write(config_path, config)
    subprocess.run([sys.executable, str(ROOT / 'tools/export_repository_assets.py')],
                   cwd=ROOT, check=True)
    print('GENERALIZATION_DELIVERY_FROZEN_AND_EXPORTED', len(inventory),
          sha(D / 'manifest_sha256.json'), flush=True)


if __name__ == '__main__':
    main()
