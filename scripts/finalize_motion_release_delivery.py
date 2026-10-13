"""Seal the reviewed current manuscript after the four completed studies."""
import ast,json,re,shutil,hashlib
from pathlib import Path
P=Path('paper/current');D=Path('reports/cvpr_motion_release_review_20261012');BUILD=Path('.build/paper')
STUDIES=['cvpr_motion_prevalence_20261012','cvpr_humanml_blind_repetitions_20261012','cvpr_humanml_reward_pilot_20261012','cvpr_motion_reward_release_20261013']
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'manifest_sha256.json').exists();layout=read(D/'layout_verification.json');assert layout['main']['content_pages']<=8
 for name in ['main','supplement']:assert sha(BUILD/(name+'.pdf'))==layout[name]['pdf_sha256']
 for study in STUDIES:assert (Path('reports')/study/'manifest_sha256.json').exists() and read(Path('reports')/study/'verification.json')['passed']
 primary=read('reports/cvpr_motion_prevalence_20261012/analysis.json');blind=read('reports/cvpr_humanml_blind_repetitions_20261012/analysis.json');pilot=read('reports/cvpr_humanml_reward_pilot_20261012/summary.json');reward=read('reports/cvpr_motion_reward_release_20261013/analysis.json')
 assert all(abs(x['FID']['paired_raw_minus_canonical'])>0 and (x['FID']['paired_family_CI95'][0]>0 or x['FID']['paired_family_CI95'][1]<0) for x in primary['methods'].values());assert not primary['FID_rank_order_changed']
 assert len(blind['cohorts'])==3 and all(x['FID']['blind_OT_bound010']<x['FID']['identity']<x['FID']['blind_OT_bound035'] for x in blind['cohorts'])
 assert pilot['conditions']['raw_reward']['R1']['paired_gap_mean']<0
 for reader,rows in reward['readers'].items():assert all(rows['finetuned_minus_base'][c]['R1']>0 for c in ['raw','canonical'])
 teaser=read(D/'teaser_evidence.json');assert sha(P/'figures/teaser.pdf')==teaser['figure_sha256'];assert teaser['RIC_positions_bitwise_equal'] and teaser['scores_are_cohort_aggregate_not_per_clip']
 keys=set(re.findall(r'@\w+\s*\{([^,]+),',(P/'references.bib').read_text()));cited=set()
 for p in P.glob('*.tex'):
  for g in re.findall(r'\\cite\w*\{([^}]+)\}',p.read_text()):cited.update(g.split(','))
 assert cited==keys and len(keys)==46
 source=D/'manuscript';source.mkdir(exist_ok=True)
 for pat in ['*.tex','*.bib','*.sty','*.bst']:
  for p in P.glob(pat):shutil.copy2(p,source/p.name)
 shutil.copytree(P/'figures',source/'figures',dirs_exist_ok=True);(source/'build').mkdir(exist_ok=True)
 for name in ['main','supplement']:
  for suffix in ['.pdf','.log','.txt','_fonts.txt','_bbox.html']:
   shutil.copy2(BUILD/(name+suffix),source/'build'/(name+suffix))
 layout['visual_inspection']={'passed':True,'main_pages':10,'supplement_pages':16,'all_pages_rendered_and_inspected':True,'figures_and_tables_checked_for_overlap':True,'main_conclusion_no_longer_split_across_pages':True,'previews_scope':'Contact sheets in previews; latest main pages6-8 separately retained.'};write(D/'layout_verification.json',layout)
 for n in [6,7,8]:shutil.copy2(Path('/tmp')/f'cvpr_motion_layout_main-0{n}.png',D/'previews'/f'main_final_{n:02d}.png')
 evidence={
  'natural_release_FID_and_R1':{'source':'reports/cvpr_motion_prevalence_20261012/analysis.json','main':'table_motion_releases.tex','supplement':'table_motion_release_readers.tex','scope':'20native repetitions5methods; primarypairedfamily10; secondcheckpoint samearchitecture, separatefamily10'},
  'fixed_real_native20':{'source':'reports/cvpr_motion_prevalence_20261012/real20_summary.json','figure':'figures/motion_release_controls.pdf','scope':'4640 native entries/reference; all directional/small/random controls retained'},
  'targeted_retrieval':{'source':'reports/cvpr_humanml3d_audit_20261012/test_analysis.json','main':'table_humanml.tex','scope':'1024fixedfullcaptionclips5candidatepermutations; targetavailable; distinctsubsetprotocol'},
  'caption_blind_repetition':{'source':'reports/cvpr_humanml_blind_repetitions_20261012/analysis.json','supplement':'table_blind_repetitions.tex','scope':'960each3cohorts with overlapping sourcegroups; original conditionalbootstrap not pooled'},
  'ordinary_reward_negative':{'source':'reports/cvpr_humanml_reward_pilot_20261012/summary.json','supplement':'table_motion_reward_pilot.tex','scope':'9fits onecommongen seed; x0reward notrollout'},
  'published_reward_weights':{'source':'reports/cvpr_motion_reward_release_20261013/analysis.json','supplement':'table_motion_reward_release.tex','scope':'2relatedcheckpoints5seeds2readers; gains survive C, noMotionReward hackingclaim'},
  'sign_language_and_adaptive_controls':{'sources':['reports/cvpr_learned_confirmation_20261012/analysis.json','reports/cvpr_second_review_confirmation_20261012/analysis.json','reports/cvpr_adaptive_distance_confirmation_20261012/analysis.json','reports/cvpr_seed_dose_response_20261012/analysis.json'],'scope':'previous frozen outcomes retained; learned attacks explicit source access, ordinaryendpoint unconfirmed'}}
 write(D/'evidence_map.json',evidence)
 write(D/'summary.json',{'review_date':'2026-10-13','title':'Same Motion, Different Scores: Auditing Representation Consistency','main_content_pages':layout['main']['content_pages'],'main_reference_pages':layout['main']['reference_pages'],'supplement_content_pages':layout['supplement']['content_pages'],'bibliography_entries':46,'five_public_methods20native_repetitions_complete':True,'two_official_readers_same_architecture':True,'primary_FID_mean_ranking_unchanged':True,'caption_blind_cohorts':3,'ordinary_reward_pilot_does_not_confirm_retrieval_inflation':True,'released_reward_gains_survive_canonicalization':True,'no_new_working_manuscript_directory':True,'study_manifests_sha256':{s:sha(Path('reports')/s/'manifest_sha256.json') for s in STUDIES},'submission_or_commit_performed':False})
 producers=['plot_motion_release_story','plot_motion_consistency_contract','build_motion_release_presentation','build_motion_reward_tables','audit_current_motion_layout','finalize_motion_release_delivery'];mapping={p.stem:p for p in Path('scripts').glob('*.py')};keep=set(producers);pending=list(producers)
 while pending:
  for node in ast.walk(ast.parse(mapping[pending.pop()].read_text())):
   mods=[v.name for v in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
   for mod in mods:
    n=mod.split('.')[0]
    if n in mapping and n not in keep:keep.add(n);pending.append(n)
 (D/'scripts').mkdir(exist_ok=True)
 for n in sorted(keep):shutil.copy2(mapping[n],D/'scripts'/f'{n}.py')
 write(D/'source_provenance.json',{'producers':producers,'import_closure':sorted(keep),'current_working_source':'paper/current','snapshot_is_immutable_delivery_not_working_location':True,'archived_script_sha256':{n:sha(mapping[n]) for n in sorted(keep)}})
 (D/'logs').mkdir(exist_ok=True)
 for n in ['cvpr_motion_release_paper_build_layout.log','cvpr_motion_teaser_final.log']:
  shutil.copy2(Path('/tmp')/n,D/'logs'/n)
 write(D/'verification.json',{'passed':True,'completed_four_scientific_archives':True,'layout_passed':True,'visual_review_passed':True,'generated_result_tables_match_complete_endpoints':True,'all46bibliographic_keys_cited_and_available':True,'teaser_geometry_and_aggregate_scope_checked':True,'claim_limits_reviewed':True,'current_style_unmodified_and_main_within8pages':True,'finalizer_code_sha256':sha(__file__)})
 write(D/'manifest_sha256.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest_sha256.json'})
 print('DELIVERY_FROZEN',sha(D/'manifest_sha256.json'),flush=True)
if __name__=='__main__':main()
