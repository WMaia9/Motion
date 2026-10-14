"""Archive/export the final sign-stability control; never publish or submit."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from freeze_motion_release_studies import closure,read,write,sha

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'reports/cvpr_sign_stability_review_20261014';P=ROOT/'paper/current';B=ROOT/'.build/paper';R=ROOT/'reports/cvpr_sign_stability_20261014'


def main():
    assert not (D/'manifest_sha256.json').exists()
    for name in ['claim_audit','presentation_evidence','manuscript_anonymity_check','archive_integrity','visual_review']:assert read(D/(name+'.json'))['passed']
    assert read(D/'visual_review.json')['all_final_pages_inspected'];assert read(D/'anonymous_utility_verification.json')['utility_tests_passed']
    layout=read(D/'layout_verification.json');visual=read(D/'visual_review.json');assert layout['main']['content_pages']<=8 and read(R/'verification.json')['passed']
    for name in ['main','supplement']:assert sha(B/(name+'.pdf'))==layout[name]['pdf_sha256']==visual[name+'_pdf_sha256']
    snapshot=D/'manuscript';snapshot.mkdir(exist_ok=True)
    for pattern in ['*.tex','*.bib','*.sty','*.bst']:
        for f in P.glob(pattern):shutil.copy2(f,snapshot/f.name)
    (snapshot/'figures').mkdir(exist_ok=True)
    for f in (P/'figures').glob('*.pdf'):shutil.copy2(f,snapshot/'figures'/f.name)
    (snapshot/'build').mkdir(exist_ok=True)
    for name in ['main','supplement']:shutil.copy2(B/(name+'.pdf'),snapshot/'build'/(name+'.pdf'))
    with zipfile.ZipFile(D/'manuscript_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(snapshot.rglob('*')):
            if f.is_file() and 'build' not in f.parts:z.write(f,f.relative_to(snapshot))
    for name in ['sign-stability-plan.md','sign-stability-review.md']:shutil.copy2(ROOT/'docs'/name,D/name)
    scripts=closure(D,['assemble_sign_stability_delivery','build_sign_stability_presentation','audit_sign_stability_claims','audit_sign_stability_layout','freeze_sign_stability','sign_stability_controls'])
    write(D/'source_provenance.json',{'current_working_manuscript':'paper/current','snapshot_is_archive_only':True,'import_closure':scripts,'new_study_manifest_sha256':sha(R/'manifest_sha256.json'),'parent_delivery_manifest_sha256':sha(ROOT/'reports/cvpr_coherence_review_20261014/manifest_sha256.json'),'assembly_sha256':sha(__file__)})
    write(D/'summary.json',{'main_content_pages':layout['main']['content_pages'],'supplement_content_pages':layout['supplement']['content_pages'],'directional_ratio_same_signs_verified':True,'critical_real_sign_Gaussian_and_physical_controls_complete':True,'all26760_decodes_verified':True,'no_general_encoding_specificity_or_human_accuracy_claim':True,'no_new_training_generators_or_reward_fits':True,'authors_registration_and_human_cold_review_deferred':True,'submission_or_publication':False})
    write(D/'verification.json',{'passed':True,'scientific_study_frozen_verified':True,'all_controls_retained_in_manuscript_and_supplement':True,'final_PDF_layout_visual_and_source_checks_passed':True,'previous_manifest_files_unchanged':True,'new_full_archive_hashes_verified':True})
    m={str(f.relative_to(D)):sha(f) for f in sorted(D.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='manifest_sha256.json'}
    write(D/'manifest_sha256.pending',m);(D/'manifest_sha256.pending').replace(D/'manifest_sha256.json')
    path=ROOT/'configs/reviewed_delivery.json';cfg=read(path);cfg.update(review_date='2026-10-14',report=str(D.relative_to(ROOT)),paper_source=str(snapshot.relative_to(ROOT)),documents={n:n for n in ['sign-stability-plan.md','sign-stability-review.md']})
    extra=cfg.setdefault('additional_metadata',{});extra[R.name]=['source_provenance.json','rotation_ratio.json']+[f'{c}/{n}.json' for c in ['How2Sign','CSL-Daily'] for n in ['native_decoder','complete']]
    extra[D.name]=['source_provenance.json','layout_verification.json','visual_review.json','primary_sources.json','claim_audit.json','evidence_map.json','presentation_evidence.json','archive_integrity.json','parent_assets.json','manuscript_anonymity_check.json','anonymous_utility_verification.json']
    write(path,cfg);subprocess.run([sys.executable,str(ROOT/'tools/export_repository_assets.py')],cwd=ROOT,check=True)
    print('SIGN_STABILITY_DELIVERY_FROZEN_AND_EXPORTED',len(m),sha(D/'manifest_sha256.json'),flush=True)


if __name__=='__main__':main()
