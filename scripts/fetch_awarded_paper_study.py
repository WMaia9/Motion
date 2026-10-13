"""Archive primary PDFs for a concrete study of CVPR exposition and figures."""
import concurrent.futures
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path('reports/cvpr_paper_story_20261011')
PAPERS = [
 ('d4rt',2026,'best paper','Zhang_Efficiently_Reconstructing_Dynamic_Scenes_One_D4RT_at_a_Time'),
 ('vggt',2025,'best paper','Wang_VGGT_Visual_Geometry_Grounded_Transformer'),
 ('richhf',2024,'best paper','Liang_Rich_Human_Feedback_for_Text-to-Image_Generation'),
 ('image_dynamics',2024,'best paper','Li_Generative_Image_Dynamics'),
 ('inverse_light',2025,'best student paper','Malik_Neural_Inverse_Rendering_from_Propagating_Light'),
 ('visprog',2023,'best paper','Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training'),
 ('hard_minimal',2022,'best paper','Hruby_Learning_To_Solve_Hard_Minimal_Problems'),
 ('megasam',2025,'best paper honorable mention','Li_MegaSaM_Accurate_Fast_and_Robust_Structure_and_Motion_from_Casual'),
]


def main():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
    folder = ROOT/'sources';folder.mkdir(exist_ok=True)
    def fetch(item):
        tag,year,award,stem = item
        url = f'https://openaccess.thecvf.com/content/CVPR{year}/papers/{stem}_CVPR_{year}_paper.pdf'
        path = folder/(tag+'.pdf')
        if not path.exists():
            response = requests.get(url,timeout=90);response.raise_for_status()
            assert response.content.startswith(b'%PDF'),(url,response.headers.get('content-type'))
            path.write_bytes(response.content)
        subprocess.run(['pdftotext','-layout',str(path),str(folder/(tag+'.txt'))],check=True)
        info = subprocess.check_output(['pdfinfo',str(path)],text=True)
        (folder/(tag+'_pdfinfo.txt')).write_text(info)
        subprocess.run(['pdftoppm','-f','1','-singlefile','-scale-to','1600','-png',str(path),str(folder/(tag+'_page1'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        record = {'id':tag,'year':year,'award':award,'pdf_url':url,'bytes':path.stat().st_size,
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'award_source':'https://www.thecvf.com/?page_id=413',
                  'retrieved_utc':datetime.now(timezone.utc).isoformat()}
        print('AWARDED_PAPER_ARCHIVED',tag,record['bytes'],flush=True)
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch,PAPERS))
    (ROOT/'sources/awarded_papers.json').write_text(json.dumps(records,indent=2)+'\n')
    urls = {'CVPR2025_awards':'https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos',
            'CVPR2026_awards':'https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers',
            'CVPR2027_guidelines':'https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines',
            'easytune':'https://arxiv.org/pdf/2602.07967',
            'levinson_svd':'https://proceedings.nips.cc/paper_files/paper/2020/file/fec3392b0dc073244d38eba1feb8e6b7-Paper.pdf'}
    additional=[]
    for tag,url in urls.items():
        response=requests.get(url,timeout=90)
        extension='.pdf' if response.content.startswith(b'%PDF') else '.html'
        path=folder/(tag+extension);path.write_bytes(response.content)
        if extension=='.pdf':subprocess.run(['pdftotext','-layout',str(path),str(folder/(tag+'.txt'))],check=True)
        additional.append({'id':tag,'url':url,'status':response.status_code,'sha256':hashlib.sha256(response.content).hexdigest(),
                           'bytes':len(response.content),'file':str(path.relative_to(ROOT))})
    (ROOT/'sources/additional_sources.json').write_text(json.dumps(additional,indent=2)+'\n')
    print('AWARDED_PAPER_STUDY_SOURCES_READY',flush=True)


if __name__=='__main__':main()
