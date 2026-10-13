"""Pinned untouched SignSparK TEST assets in a new, registered report."""
import concurrent.futures
import json
from pathlib import Path
import requests
from verify_signspark_studies import sha

ROOT=Path('reports/cvpr_evaluation_confirmation_20261011')
REV='9fe7e04ef59204c787cd6094c747f889efff6ba0'


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    path=ROOT/'sources/protocol_sha256.json'
    digest=sha(ROOT/'PROTOCOLO.md')
    if path.exists():assert json.loads(path.read_text())['PROTOCOLO.md']==digest
    else:path.write_text(json.dumps({'PROTOCOLO.md':digest},indent=2)+'\n')
    url=f'https://huggingface.co/api/datasets/LionelLow/SignSparK_data/revision/{REV}?blobs=true'
    response=requests.get(url,timeout=45);response.raise_for_status();metadata=response.json()
    (ROOT/'sources/test_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    jobs=[]
    for item in metadata['siblings']:
        name=item['rfilename']
        if not name.startswith('test/') or not any(c in name for c in ['PHOENIX-2014T','How2Sign']):continue
        jobs.append(item)
    def download(item):
        name=item['rfilename'];dest=ROOT/'data/lmdb'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/datasets/LionelLow/SignSparK_data/resolve/{REV}/{name}'
        if not dest.exists():
            response=requests.get(url,stream=True,timeout=(30,90));response.raise_for_status()
            part=dest.with_name(dest.name+'.partial')
            with part.open('wb') as out:
                for block in response.iter_content(4*1024*1024):out.write(block)
            part.rename(dest)
        digest=sha(dest);assert dest.stat().st_size==item['size']
        if item.get('lfs'):assert digest==item['lfs']['sha256']
        result={'file':str(dest.relative_to(ROOT)),'url':url,'bytes':dest.stat().st_size,'sha256':digest}
        print('TEST_DOWNLOADED',json.dumps(result),flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(download,jobs))
    (ROOT/'sources/test_downloads.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
