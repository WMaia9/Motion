"""Archive/export a fully verified follow-up; never register or submit a paper."""
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from freeze_motion_release_studies import closure, read, write, sha
from freeze_mardm_followup_studies import STUDIES

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'reports/cvpr_mardm_followup_20261013'
P = ROOT / 'paper/current'
B = ROOT / '.build/paper'
PREVIOUS = ROOT / 'reports/cvpr_editorial_review_20261013'


def main():
    assert not (D / 'manifest_sha256.json').exists()
    assert read(D / 'presentation_evidence.json')['passed']
    assert read(D / 'claim_audit.json')['passed']
    visual = read(D / 'visual_review.json'); layout = read(D / 'layout_verification.json')
    assert visual['passed'] and visual['all_final_pages_inspected']
    for name in ['main', 'supplement']:
        digest = sha(B / (name + '.pdf'))
        assert visual[name + '_pdf_sha256'] == layout[name]['pdf_sha256'] == digest
        assert layout[name]['latex_reference_and_box_checks_passed']
    assert layout['main']['content_pages'] <= 8
    assert read(D / 'anonymous_utility_verification.json')['utility_tests_passed']
    assert read(D / 'manuscript_anonymity_check.json')['passed']
    assert read(D / 'frozen_integrity.json')['previous_manifests_unchanged']
    integrity = read(D / 'new_study_integrity.json')
    assert integrity['passed']
    studies = {}
    for name, _ in STUDIES.values():
        path = ROOT / 'reports' / name
        assert read(path / 'verification.json')['passed']
        studies[name] = sha(path / 'manifest_sha256.json')
        assert integrity['studies'][name]['all_file_hashes_verified']
        assert integrity['studies'][name]['manifest_sha256'] == studies[name]
    snapshot = D / 'manuscript'; snapshot.mkdir(exist_ok=True)
    for pattern in ['*.tex', '*.bib', '*.sty', '*.bst']:
        for source in P.glob(pattern):shutil.copy2(source, snapshot / source.name)
    (snapshot / 'figures').mkdir(exist_ok=True)
    for source in (P / 'figures').glob('*.pdf'):shutil.copy2(source, snapshot / 'figures' / source.name)
    (snapshot / 'build').mkdir(exist_ok=True)
    for name in ['main', 'supplement']:shutil.copy2(B / (name + '.pdf'), snapshot / 'build' / (name + '.pdf'))
    with zipfile.ZipFile(D / 'manuscript_source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for source in sorted(snapshot.rglob('*')):
            if source.is_file() and 'build' not in source.parts:z.write(source, source.relative_to(snapshot))
    for filename in ['mardm-followup-plan.md', 'mardm-followup-review.md']:shutil.copy2(ROOT / 'docs' / filename, D / filename)
    names = closure(D, ['assemble_mardm_followup_delivery', 'build_mardm_followup_presentation', 'audit_current_motion_layout', 'audit_mardm_motion_layout', 'freeze_mardm_followup_studies', 'audit_mardm_followup_claims'])
    (D / 'tools').mkdir(exist_ok=True); shutil.copy2(ROOT / 'tools/canonicalize_motion.py', D / 'tools/canonicalize_motion.py')
    write(D / 'source_provenance.json', {'current_working_manuscript':'paper/current', 'snapshot_is_archive_not_another_working_location':True, 'import_closure':names, 'assembly_source_sha256':sha(__file__), 'new_frozen_study_manifests':studies, 'previous_editorial_manifest_sha256':sha(PREVIOUS / 'manifest_sha256.json'), 'portable_utility_sha256':sha(D / 'tools/canonicalize_motion.py')})
    write(D / 'summary.json', {'scope':'exact rotation-equivalence consistency, retaining rotations in rotation-only domains', 'new_studies':studies, 'main_content_pages':layout['main']['content_pages'], 'supplement_content_pages':layout['supplement']['content_pages'], 'reference_entries':len(re.findall(r'@\w+\s*\{([^,]+),', (P / 'references.bib').read_text())), 'no_new_training_reward_seeds_or_generators':True, 'does_not_refute_full_MARDM_mechanism':True, 'essential_reader_invariance_structural_not_accuracy_cost':True, 'routine_training_exploitation_unconfirmed':True, 'authors_registration_and_human_cold_review_deferred':True, 'format2027_provisional_available_official2026_kit':True, 'submission_or_publication':False})
    write(D / 'verification.json', {'passed':True, 'all_four_new_studies_complete_frozen_and_independently_verified':True, 'all_twenty_draw_endpoints_retained':True, 'all_controls_retained_in_main_or_supplement':True, 'exact_final_PDF_layout_and_visual_checks_passed':True, 'previous_frozen_manifests_unchanged':True, 'anonymous_source_and_portable_utility_verified':True})
    inventory = {str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest_sha256.json'}
    write(D / 'manifest_sha256.pending', inventory); (D / 'manifest_sha256.pending').replace(D / 'manifest_sha256.json')
    config_path = ROOT / 'configs/reviewed_delivery.json'; config = read(config_path)
    config.update(review_date='2026-10-13', report=str(D.relative_to(ROOT)), paper_source=str(snapshot.relative_to(ROOT)), documents={n:n for n in ['mardm-followup-review.md','mardm-followup-plan.md']})
    additional = config.setdefault('additional_metadata', {})
    for name, _ in STUDIES.values():additional[name] = ['source_provenance.json']
    additional['cvpr_tmr1024_controls_20261013'] += ['generic_geometry.json']
    additional['cvpr_matched_blocks_20261013'] += [f'{r}/{n}.json' for r in ['guo','tmr'] for n in ['analysis','verification','complete','normalizer']]
    additional[D.name] = ['source_provenance.json','primary_sources.json','layout_verification.json','visual_review.json','evidence_map.json','presentation_evidence.json','claim_audit.json','frozen_integrity.json','new_study_integrity.json','presentation_corrections.json','manuscript_anonymity_check.json','anonymous_utility_verification.json']
    write(config_path, config)
    subprocess.run([sys.executable, str(ROOT / 'tools/export_repository_assets.py')], cwd=ROOT, check=True)
    print('MARDM_FOLLOWUP_DELIVERY_FROZEN_AND_EXPORTED', len(inventory), sha(D / 'manifest_sha256.json'), flush=True)


if __name__ == '__main__':main()
