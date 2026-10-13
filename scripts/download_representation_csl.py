"""Third language: pin public inputs before inference, verify LFS hashes."""
import concurrent.futures,json
from pathlib import Path
import requests
from verify_signspark_studies import sha

ROOT=Path('reports/cvpr_representation_paper_20261011');DATA_REV='9fe7e04ef59204c787cd6094c747f889efff6ba0'


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    registered=json.loads((ROOT/'sources/registered_inputs.json').read_text());assert sha(ROOT/'PROTOCOLO.md')==registered['protocol_sha256']
    model=requests.get('https://huggingface.co/api/models/LionelLow/SignSparK_BT?blobs=true',timeout=45);model.raise_for_status();model=model.json()
    revision=model['sha'];(ROOT/'sources/CSL_model_metadata.json').write_text(json.dumps(model,indent=2)+'\n')
    data=requests.get(f'https://huggingface.co/api/datasets/LionelLow/SignSparK_data/revision/{DATA_REV}?blobs=true',timeout=45);data.raise_for_status();data=data.json()
    (ROOT/'sources/CSL_data_metadata.json').write_text(json.dumps(data,indent=2)+'\n')
    jobs=[]
    for item in model['siblings']:
        if item['rfilename'].startswith('CSL-Daily/'):
            jobs.append((item,ROOT/'checkpoints'/item['rfilename'],f'https://huggingface.co/LionelLow/SignSparK_BT/resolve/{revision}/'+item['rfilename']))
    for item in data['siblings']:
        if item['rfilename'].startswith('test/CSL-Daily'):
            jobs.append((item,ROOT/'data/lmdb'/item['rfilename'],f'https://huggingface.co/datasets/LionelLow/SignSparK_data/resolve/{DATA_REV}/'+item['rfilename']))
    (ROOT/'sources/CSL_pinned_revisions.json').write_text(json.dumps({'model':revision,'data':DATA_REV,'protocol_sha256':registered['protocol_sha256']},indent=2)+'\n')
    def download(job):
        item,path,url=job;path.parent.mkdir(parents=True,exist_ok=True)
        response=requests.get(url,stream=True,timeout=(30,90));response.raise_for_status();part=path.with_name(path.name+'.partial')
        with part.open('wb') as f:
            for block in response.iter_content(4*1024*1024):f.write(block)
        part.rename(path);digest=sha(path);assert path.stat().st_size==item['size']
        if item.get('lfs'):assert digest==item['lfs']['sha256']
        result={'file':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest,'url':url}
        print('CSL_DOWNLOADED',json.dumps(result),flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(download,jobs))
    (ROOT/'sources/CSL_downloads.json').write_text(json.dumps(records,indent=2)+'\n')
    print('CSL_DOWNLOAD_COMPLETE',flush=True)


if __name__=='__main__':main()
