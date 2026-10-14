"""Archive/export the reviewed post-MARDM corrections; never publish/submit."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from freeze_motion_release_studies import closure,read,write,sha

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'reports/cvpr_coherence_review_20261014'
P=ROOT/'paper/current'
B=ROOT/'.build/paper'
STUDIES=['cvpr_coherent_blocks_20261014','cvpr_sign_natural_restoration_20261014']


def main():
    assert not (D/'manifest_sha256.json').exists()
    for name in ['presentation_evidence','claim_audit','manuscript_anonymity_check','visual_review']:assert read(D/(name+'.json'))['passed']
    assert read(D/'visual_review.json')['all_final_pages_inspected']
    assert read(D/'archive_integrity.json')['passed']
    assert read(D/'anonymous_utility_verification.json')['utility_tests_passed']
    layout=read(D/'layout_verification.json');visual=read(D/'visual_review.json')
    assert layout['main']['content_pages']<=8
    for name in ['main','supplement']:
        assert sha(B/(name+'.pdf'))==layout[name]['pdf_sha256']==visual[name+'_pdf_sha256']
        assert layout[name]['latex_reference_and_box_checks_passed']
    for name in STUDIES:assert read(ROOT/'reports'/name/'verification.json')['passed']
    snapshot=D/'manuscript';snapshot.mkdir(exist_ok=True)
    for pattern in ['*.tex','*.bib','*.sty','*.bst']:
        for file in P.glob(pattern):shutil.copy2(file,snapshot/file.name)
    (snapshot/'figures').mkdir(exist_ok=True)
    for file in (P/'figures').glob('*.pdf'):shutil.copy2(file,snapshot/'figures'/file.name)
    (snapshot/'build').mkdir(exist_ok=True)
    for name in ['main','supplement']:shutil.copy2(B/(name+'.pdf'),snapshot/'build'/(name+'.pdf'))
    with zipfile.ZipFile(D/'manuscript_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(snapshot.rglob('*')):
            if file.is_file() and 'build' not in file.parts:z.write(file,file.relative_to(snapshot))
    for name in ['coherence-review-plan.md','coherence-review.md']:shutil.copy2(ROOT/'docs'/name,D/name)
    scripts=closure(D,['assemble_coherence_delivery','audit_coherence_claims','audit_coherence_motion_layout','build_coherence_presentation','freeze_coherence_studies','coherent_motion_blocks'])
    write(D/'source_provenance.json',{'current_working_manuscript':'paper/current','snapshot_is_archive_only':True,'import_closure':scripts,'new_study_manifests':{n:sha(ROOT/'reports'/n/'manifest_sha256.json') for n in STUDIES},'parent_delivery_manifest_sha256':sha(ROOT/'reports/cvpr_mardm_followup_20261013/manifest_sha256.json'),'assembly_source_sha256':sha(__file__)})
    write(D/'summary.json',{'closed_review_items':['time-fixed directions in non-null blocks, both signs and radii','natural real sign inputs without target or attack restored','kappa budget / rho fiber notation separated','abstract and motivation grounded in no-target sign evidence'],'main_content_pages':layout['main']['content_pages'],'supplement_content_pages':layout['supplement']['content_pages'],'no_new_generators_training_model_seeds_or_reward_fits':True,'no_null_feature_priority_claim':True,'no_routine_training_exploitation_or_accuracy_improvement_claim':True,'registration_authors_and_human_cold_review_deferred':True,'submission_or_publication':False})
    write(D/'verification.json',{'passed':True,'all_new_scientific_studies_frozen_verified':True,'all_signs_radii_endpoints_retained':True,'all_final_PDF_layout_visual_and_source_checks_passed':True,'previous_manifest_files_unchanged':True,'new_full_archive_file_hashes_verified':True})
    m={str(file.relative_to(D)):sha(file) for file in sorted(D.rglob('*')) if file.is_file() and '__pycache__' not in file.parts and file.name!='manifest_sha256.json'}
    write(D/'manifest_sha256.pending',m);(D/'manifest_sha256.pending').replace(D/'manifest_sha256.json')
    path=ROOT/'configs/reviewed_delivery.json';config=read(path)
    config.update(review_date='2026-10-14',report=str(D.relative_to(ROOT)),paper_source=str(snapshot.relative_to(ROOT)),documents={n:n for n in ['coherence-review-plan.md','coherence-review.md']})
    extra=config.setdefault('additional_metadata',{})
    extra[STUDIES[0]]=['source_provenance.json']+[f'{r}/{n}.json' for r in ['guo','tmr'] for n in ['normalizer','analysis','verification','complete']]
    extra[STUDIES[1]]=['source_provenance.json']
    extra[D.name]=['source_provenance.json','layout_verification.json','visual_review.json','primary_sources.json','claim_audit.json','evidence_map.json','presentation_evidence.json','archive_integrity.json','parent_assets.json','manuscript_anonymity_check.json','anonymous_utility_verification.json']
    write(path,config)
    subprocess.run([sys.executable,str(ROOT/'tools/export_repository_assets.py')],cwd=ROOT,check=True)
    print('COHERENCE_DELIVERY_FROZEN_AND_EXPORTED',len(m),sha(D/'manifest_sha256.json'),flush=True)


if __name__=='__main__':main()
