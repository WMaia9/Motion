import concurrent.futures, hashlib, json
from pathlib import Path
import requests

root=Path('reports/cvpr_signspark_baseline_20261010')
jobs=[]
for repo,kind,revision in [('LionelLow/SignSparK_data','datasets','9fe7e04ef59204c787cd6094c747f889efff6ba0'),('LionelLow/SignSparK_BT','models','0ced459f6c24d453893809eeca87e9ec80b70198')]:
    meta=requests.get(f'https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true',timeout=60)
    meta.raise_for_status(); meta=meta.json()
    (root/'sources'/f'{kind}_metadata.json').write_text(json.dumps(meta,indent=2))
    for item in meta['siblings']:
        name=item['rfilename']
        if not any(x in name for x in ['How2Sign','PHOENIX-2014T']): continue
        if kind=='datasets' and not name.startswith('dev/'):continue
        dest=root/('data/lmdb' if kind=='datasets' else 'checkpoints')/name
        url=f'https://huggingface.co/{"datasets/" if kind=="datasets" else ""}{repo}/resolve/{revision}/{name}'
        jobs.append((url,dest,item))

def download(job):
    url,dest,item=job;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        response=requests.get(url,stream=True,timeout=(30,120));response.raise_for_status()
        part=dest.with_name(dest.name+'.partial')
        with part.open('wb') as out:
            for chunk in response.iter_content(4*1024*1024):out.write(chunk)
        part.rename(dest)
    digest=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
    assert dest.stat().st_size==item['size'],dest
    if item.get('lfs'):assert digest==item['lfs']['sha256'],dest
    print('verified',str(dest),dest.stat().st_size,flush=True)
    return {'path':str(dest),'bytes':dest.stat().st_size,'sha256':digest,'url':url}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results=list(executor.map(download,jobs))
(root/'sources'/'downloads.json').write_text(json.dumps(results,indent=2)+'\n')
