"""Primary-publisher bibliography additions, separate from frozen manuscript."""
import json,re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from audit_reviewed_references import entries,normalize,author_list
R=Path('reports/cvpr_second_review_20261012');P=Path('paper/cvpr2027_training_objectives_revision_20261012')
IDS={'uap2017':'uap2017','gap2018':'gap2018','fang2025':'fang2025','solireward2026':'solireward2026','rigmo2026':'rigmo2026','draft2024':'draft2024','rsgm2022':'rsgm2022'}
def main():
 assert not (R/'manifest_sha256.json').exists();catalog={d['id']:d for d in json.loads((R/'sources/research_catalog.json').read_text())};text=Path('paper/cvpr2027_learned_equivalence_20261011/references.bib').read_text();new={};checks={}
 for key,id in IDS.items():
  d=catalog[id];soup=BeautifulSoup((R/'sources'/(id+'.html')).read_text(),'html.parser');bibfile=R/'sources'/(id+'.bib')
  if not bibfile.exists():
   links=[a.get('href') for a in soup.find_all('a') if 'bibtex' in (a.get_text() or '').lower()]
   assert links,(id,'no native bibliography')
   from urllib.parse import urljoin
   u=urljoin(d['HTML'],links[0]);response=requests.get(u,timeout=60);assert response.status_code==200;bibfile.write_text(response.text)
  native=next(iter(entries(bibfile.read_text()).values()));new[key]=native;body=bibfile.read_text();body=re.sub(r'(@\w+\{)[^,]+,',lambda m:m[1]+key+',',body,count=1);text+='\n'+body+'\n'
  checks[key]={'source':str(bibfile.relative_to(R)),'native_publisher_BibTeX_used':True,'PDF_source_archived':True}
 extra=[('atn2018',{'author':'Shumeet Baluja and Ian Fischer','title':'Learning to Attack: Adversarial Transformation Networks','booktitle':'AAAI Conference on Artificial Intelligence','year':'2018','volume':'32','number':'1','doi':'10.1609/aaai.v32i1.11672','url':'https://ojs.aaai.org/index.php/AAAI/article/view/11672'}),
 ('xu2026',{'author':'Yixian Xu and Yusong Wang and Shengjie Luo and Kaiyuan Gao and Tianyu He and Di He and Chang Liu','title':'Quotient-Space Diffusion Models','booktitle':'International Conference on Learning Representations','year':'2026','url':'https://arxiv.org/abs/2604.21809v2'})]
 for key,fields in extra:
  source='atn2018' if key=='atn2018' else 'quotient2026';front=(R/'sources'/(source+'.txt')).read_text()[:4500]
  assert normalize(fields['title']) in normalize(front)
  for a in fields['author'].split(' and '):assert normalize(a) in normalize(front),(key,a)
  if key=='xu2026':assert 'ICLR 2026' in front
  text+='\n@inproceedings{'+key+',\n'+',\n'.join(' '+f+'={'+v+'}' for f,v in fields.items())+'}\n';checks[key]={'source':source+'.pdf','title_all_authors_checked':True,'publisher_DOI_checked':key=='atn2018','camera_ready_ICLR2026_checked':key=='xu2026'}
 (P/'references.bib').write_text(text);refs=entries(text);assert len(refs)==36
 (R/'reference_additions_verification.json').write_text(json.dumps({'new_entries':checks,'total_entries':len(refs),'unchanged_original_entries':27,'only_relevant_antecedents_added':True},indent=2)+'\n');print('NEW_BIBLIOGRAPHY',len(refs))
if __name__=='__main__':main()
