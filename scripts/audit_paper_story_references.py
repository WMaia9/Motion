"""Verify unchanged reference identities and new primary publisher metadata."""
import json,re
from pathlib import Path
from audit_reviewed_references import entries,normalize,author_list
from verify_signspark_studies import sha
R=Path('reports/cvpr_paper_story_20261011');P=Path('paper/cvpr2027_learned_equivalence_20261011')
def main():
 assert not (R/'manifest_sha256.json').exists()
 refs=entries((P/'references.bib').read_text());old=entries(Path('paper/cvpr2027_rotation_equivalence_revision_20261011/references.bib').read_text())
 assert len(refs)==27
 checks={}
 for k,d in old.items():
  if k=='imai2025':continue
  assert refs[k]==d,k
  checks[k]={'unchanged_from_verified_reference':True,'prior_primary_verification':'reports/cvpr_manuscript_review_20261011/reference_verification.json'}
 for k,file in [('imai2025','silver.bib'),('levinson2020','svd.bib'),('stein2023','stein.bib'),('huang2023','mt_robustness.bib')]:
  d=next(iter(entries((R/'sources'/file).read_text()).values()))
  for field in ['title','author','year','pages','doi','volume']:
   if field not in d:continue
   if field=='author':assert author_list(refs[k][field])==author_list(d[field]),k
   elif field=='title':assert normalize(refs[k][field])==normalize(d[field]),k
   else:assert refs[k][field]==d[field],(k,field)
  checks[k]={'publisher_BibTeX_fields_verified':True,'source':file,'sha256':sha(R/'sources'/file)}
 d=refs['tan2026'];front=(R/'sources/easytune.txt').read_text()[:2500]
 assert normalize(d['title']) in normalize(front)
 for name in d['author'].split(' and '):assert normalize(name) in normalize(front),name
 assert 'ICLR 2026' in front
 checks['tan2026']={'title_all_authors_and_ICLR2026_match_primary_PDF':True,'source':'easytune.pdf','sha256':sha(R/'sources/easytune.pdf')}
 keys={k.strip() for doc in ['main.tex','supplement.tex'] for group in re.findall(r'\\cite\{([^}]+)\}',(P/doc).read_text()) for k in group.split(',')}
 assert keys==set(refs),(keys-set(refs),set(refs)-keys)
 assert set(checks)==set(refs)
 assert refs['levinson2020']['booktitle']=='Advances in Neural Information Processing Systems'
 out={'passed':True,'total_entries':len(refs),'all_entries_cited':True,'no_undefined_citation_keys':True,'references':checks,
 'scope':'primary metadata check plus detailed prior source audit; not a guarantee of exhaustive novelty search','SiLVER_updated_to_published_RANLP2025':True,'SVD_correctly_NeurIPS2020_not_CVPR':True}
 (R/'reference_verification.json').write_text(json.dumps(out,indent=2)+'\n');print('REFERENCES_VERIFIED',len(checks))
if __name__=='__main__':main()
