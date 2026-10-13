"""Pinned public assets; no gated body models and no TEST downloads."""
import concurrent.futures, hashlib, json
from pathlib import Path
import requests

ROOT=Path('reports/cvpr_motion_consistency_confirmation_20261010')
jobs=[]
for label,repo,prefix,destroot in [
    ('generator','LionelLow/SignSparK','','checkpoints'),
    ('clip','M-CLIP/XLM-Roberta-Large-Vit-B-16Plus','','text_encoder'),
    ('data','LionelLow/SignSparK_data','datasets/','data/lmdb')]:
    meta=json.loads((ROOT/'sources'/f'{label}_metadata.json').read_text())
    for item in meta['siblings']:
        name=item['rfilename']
        wanted=(label=='generator' and name.startswith(('body/','hand/'))) or (label=='clip' and name in ['config.json','pytorch_model.bin','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json']) or (label=='data' and name=='train/PHOENIX-2014T_reopt_train.lmdb/data.mdb')
        if wanted:jobs.append((f'https://huggingface.co/{prefix}{repo}/resolve/{meta["sha"]}/{name}',ROOT/destroot/name,item))

def download(job):
    url,dest,item=job;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        response=requests.get(url,stream=True,timeout=(30,120));response.raise_for_status()
        part=dest.with_name(dest.name+'.partial')
        with part.open('wb') as out:
            for chunk in response.iter_content(8*1024*1024):out.write(chunk)
        part.rename(dest)
    sha=hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
    digest=sha.hexdigest();assert dest.stat().st_size==item['size']
    if item.get('lfs'):assert digest==item['lfs']['sha256']
    print('verified',str(dest),dest.stat().st_size,flush=True)
    return {'path':str(dest),'bytes':dest.stat().st_size,'sha256':digest,'url':url}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(download,jobs))
(ROOT/'sources'/'downloads.json').write_text(json.dumps(results,indent=2))
