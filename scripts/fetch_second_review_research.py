"""Archive primary cross-disciplinary sources and direct attack antecedents."""
import json,hashlib,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
ROOT=Path('reports/cvpr_second_review_20261012');SOURCES=ROOT/'sources'
CATALOG=[
('uap2017','https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html','https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf'),
('gap2018','https://openaccess.thecvf.com/content_cvpr_2018/html/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.html','https://openaccess.thecvf.com/content_cvpr_2018/papers/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.pdf'),
('atn2018','https://ojs.aaai.org/index.php/AAAI/article/view/11672','https://ojs.aaai.org/index.php/AAAI/article/download/11672/11531'),
('rsgm2022','https://proceedings.neurips.cc/paper_files/paper/2022/hash/105112d52254f86d5854f3da734a52b4-Abstract-Conference.html','https://proceedings.neurips.cc/paper_files/paper/2022/file/105112d52254f86d5854f3da734a52b4-Paper-Conference.pdf'),
('rdm2022','https://proceedings.neurips.cc/paper_files/paper/2022/hash/123d3e814e257e0781e5d328232ead9b-Abstract-Conference.html','https://proceedings.neurips.cc/paper_files/paper/2022/file/123d3e814e257e0781e5d328232ead9b-Paper-Conference.pdf'),
('mixture2024','https://proceedings.mlr.press/v235/jo24a.html','https://raw.githubusercontent.com/mlresearch/v235/main/assets/jo24a/jo24a.pdf'),
('quotient2026','https://arxiv.org/abs/2604.21809v2','https://arxiv.org/pdf/2604.21809v2'),
('schwinger2026','https://arxiv.org/abs/2606.27481v2','https://arxiv.org/pdf/2606.27481v2'),
('landing2026','https://proceedings.iclr.cc/paper_files/paper/2026/hash/d1a6fe144bdff7b95138bca8097d4e69-Abstract-Conference.html','https://proceedings.iclr.cc/paper_files/paper/2026/file/d1a6fe144bdff7b95138bca8097d4e69-Paper-Conference.pdf'),
('rigmo2026','https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_RigMo_Unifying_Rig_and_Motion_Learning_for_Generative_Animation_CVPR_2026_paper.html','https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_RigMo_Unifying_Rig_and_Motion_Learning_for_Generative_Animation_CVPR_2026_paper.pdf'),
('easytune2026','https://proceedings.iclr.cc/paper_files/paper/2026/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html','https://proceedings.iclr.cc/paper_files/paper/2026/file/00b67df24009747e8bbed4c2c6f9c825-Paper-Conference.pdf')]
def fetch(item):
 id,html,pdf=item;record={'id':id,'HTML':html,'PDF':pdf,'retrieved_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 for ext,url in [('html',html),('pdf',pdf)]:
  path=SOURCES/(id+'.'+ext)
  try:
   r=requests.get(url,timeout=60,headers={'User-Agent':'Mozilla/5.0 academic research'});path.write_bytes(r.content);record[ext]={'status':r.status_code,'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'file':str(path.relative_to(ROOT))}
   if ext=='html' and r.status_code==200:
    soup=BeautifulSoup(r.content,'html.parser');record['citation_metadata']=[[m.get('name'),m.get('content')] for m in soup.find_all('meta') if (m.get('name') or '').startswith('citation_')]
    (SOURCES/(id+'_HTML.txt')).write_text(soup.get_text(' ',strip=True))
    bib=soup.find('pre',class_='bibtex') or soup.find('div',class_='bibref')
    if bib:(SOURCES/(id+'.bib')).write_text(bib.get_text())
   if ext=='pdf' and r.status_code==200 and r.content.startswith(b'%PDF'):
    subprocess.run(['pdftotext','-layout',str(path),str(SOURCES/(id+'.txt'))],check=True);record['pdf_text_extracted']=True
  except Exception as e:record[ext+'_error']=str(e)
 print('RESEARCH_SOURCE',id,record.get('pdf',{}).get('status'),flush=True);return record

def main():
 assert not (ROOT/'manifest_sha256.json').exists();SOURCES.mkdir(parents=True,exist_ok=True)
 with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,CATALOG))
 (SOURCES/'research_catalog.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
