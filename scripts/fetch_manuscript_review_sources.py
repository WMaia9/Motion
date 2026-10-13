"""Archive primary reference and format sources for a new immutable revision."""
import concurrent.futures,hashlib,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
import requests
ROOT=Path('reports/cvpr_manuscript_review_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence_revision_20261011')
URLS={
 'cvpr2027_guidelines.html':'https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines',
 'cvpr2027_cfp.html':'https://cvpr.thecvf.com/Conferences/2027/CallForPapers',
 'cvpr2026_guidelines.html':'https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines',
 'zhou2019.html':'https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html',
 'camgoz2018.html':'https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html',
 'camgoz2020.html':'https://openaccess.thecvf.com/content_CVPR_2020/html/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.html',
 'duarte2021.html':'https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html',
 'zhou2021.html':'https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.html',
 't2sgpt.bib':'https://aclanthology.org/2024.acl-long.183.bib',
 'bleu.bib':'https://aclanthology.org/P02-1040.bib',
 'sacrebleu.bib':'https://aclanthology.org/W18-6319.bib',
 'silver.html':'https://arxiv.org/abs/2509.03791',
 'silver_full.html':'https://arxiv.org/html/2509.03791v1',
 'bt2.html':'https://arxiv.org/abs/2606.28673',
 'bt2_full.html':'https://arxiv.org/html/2606.28673v1',
 'signspark.html':'https://arxiv.org/abs/2603.10446',
 'signspark_full.html':'https://arxiv.org/html/2603.10446v4',
 'm3t.html':'https://arxiv.org/abs/2603.23617',
 'm3t_full.html':'https://arxiv.org/html/2603.23617v2',
 'm3t_README.md':'https://raw.githubusercontent.com/ASH-FortyFive/M3T/main/README.md',
 'ddpm.html':'https://arxiv.org/abs/2006.11239',
 'ddim.html':'https://arxiv.org/abs/2010.02502',
 'cfg.html':'https://arxiv.org/abs/2207.12598',
 'progressive.html':'https://arxiv.org/abs/2004.14874',
 'graves2006.pdf':'https://www.cs.toronto.edu/~graves/icml_2006.pdf',
 'li2023.html':'https://researchportal.port.ac.uk/en/publications/adversarial-attacks-on-skeleton-based-sign-language-recognition/',
 'li2023.pdf':'https://pure.port.ac.uk/ws/portalfiles/portal/81362178/Li_et_al_2023_AAM.pdf',
 'liu2024.html':'https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Defense_Against_Adversarial_Attacks_on_No-Reference_Image_Quality_Models_with_CVPR_2024_paper.html',
 'liu2024.pdf':'https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Defense_Against_Adversarial_Attacks_on_No-Reference_Image_Quality_Models_with_CVPR_2024_paper.pdf',
 'author_kit_commit.json':'https://api.github.com/repos/cvpr-org/author-kit/commits/main',
 'BT_model_metadata.json':'https://huggingface.co/api/models/LionelLow/SignSparK_BT?blobs=true',
}
def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    records=[]
    def get(item):
        name,url=item
        try:
            r=requests.get(url,timeout=(15,40));p=ROOT/'sources'/name;p.write_bytes(r.content)
            record={'file':name,'url':url,'status':r.status_code,'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'retrieved_UTC':datetime.now(timezone.utc).isoformat()}
            if r.status_code==200 and name.endswith('.html'):
                record['citation_metadata']=re.findall(r'<meta\s+name="(citation_[^"]+)"\s+content="([^"]+)"',r.text)
            print('SOURCE',name,r.status_code,len(r.content),flush=True);return record
        except requests.RequestException as e:
            print('SOURCE_FAILED',name,str(e),flush=True);return {'file':name,'url':url,'error':str(e)}
    with concurrent.futures.ThreadPoolExecutor(6) as pool:records=list(pool.map(get,URLS.items()))
    commit=json.loads((ROOT/'sources/author_kit_commit.json').read_text())['sha']
    for name in ['cvpr.sty','ieeenat_fullname.bst','main.tex','preamble.tex','README.md','sec/2_formatting.tex']:
        url=f'https://raw.githubusercontent.com/cvpr-org/author-kit/{commit}/{name}'
        record=get(('kit_'+name.replace('/','_'),url));records.append(record)
        if name in ['cvpr.sty','ieeenat_fullname.bst']:
            assert record['status']==200;(PAPER/name).write_bytes((ROOT/'sources'/record['file']).read_bytes())
    model=json.loads((ROOT/'sources/BT_model_metadata.json').read_text())['sha']
    records.append(get(('BT_modelcard.md',f'https://huggingface.co/LionelLow/SignSparK_BT/raw/{model}/README.md')))
    for name in ['graves2006','li2023','liu2024']:
        pdf=ROOT/'sources'/(name+'.pdf')
        if pdf.is_file() and pdf.read_bytes().startswith(b'%PDF'):
            subprocess.run(['pdftotext','-layout',str(pdf),str(ROOT/'sources'/(name+'.txt'))],check=True)
    (ROOT/'sources/retrievals.json').write_text(json.dumps(records,indent=2)+'\n')
    (PAPER/'template_source.json').write_text(json.dumps({'repository':'https://github.com/cvpr-org/author-kit','commit':commit,'target_year':2027,'guidelines2027_status':next(x for x in records if x['file']=='cvpr2027_guidelines.html'),'upstream_main_source':'sources/kit_main.tex','adaptation':'Review style; year2027; anonymous; placeholder ID; final2027 compliance not certified'},indent=2)+'\n')
    print('FETCH_COMPLETE',len(records),commit,flush=True)
if __name__=='__main__':main()
