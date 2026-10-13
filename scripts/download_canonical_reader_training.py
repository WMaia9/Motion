"""Fetch fixed, public TRAIN/DEV assets needed for canonical-reader validation."""
import concurrent.futures
import json
from pathlib import Path
import requests
from verify_signspark_studies import sha

ROOT = Path('reports/cvpr_canonical_readers_20261011')
REVISION = '9fe7e04ef59204c787cd6094c747f889efff6ba0'
FILES = ['train/How2Sign_reopt_train.lmdb/data.mdb',
         'train/CSL-Daily_reopt_train.lmdb/data.mdb',
         'dev/CSL-Daily_reopt_dev.lmdb/data.mdb']


def main():
    ROOT.mkdir(exist_ok=True); (ROOT/'sources').mkdir(exist_ok=True)
    assert not (ROOT/'manifest_sha256.json').exists(), 'frozen report'
    metadata = json.loads(Path('reports/cvpr_representation_paper_20261011/sources/CSL_data_metadata.json').read_text())
    assert metadata['sha'] == REVISION
    items = {r['rfilename']:r for r in metadata['siblings']}
    jobs = [{'file':name,'bytes':items[name]['size'],'sha256':items[name]['lfs']['sha256'],
             'url':f'https://huggingface.co/datasets/LionelLow/SignSparK_data/resolve/{REVISION}/{name}'} for name in FILES]
    registration = {'revision':REVISION,'files':jobs,'no_TEST_downloads':True,
        'reason':'paired raw/canonical reader fine-tuning on corpora with noncanonical real features'}
    path = ROOT/'sources/download_registration.json'
    if path.exists(): assert json.loads(path.read_text()) == registration
    else: path.write_text(json.dumps(registration,indent=2)+'\n')
    def download(item):
        path = ROOT/'data/lmdb'/item['file'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            response = requests.get(item['url'],stream=True,timeout=(30,120));response.raise_for_status()
            temporary = path.with_name(path.name+'.partial');written = 0;last_print = 0
            with temporary.open('wb') as f:
                for block in response.iter_content(8*1024*1024):
                    f.write(block);written += len(block)
                    if written-last_print >= 1024**3:
                        print('DOWNLOAD_PROGRESS',item['file'],written,item['bytes'],flush=True);last_print=written
            assert written == item['bytes'];temporary.rename(path)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        print('DOWNLOAD_VERIFIED',item['file'],item['bytes'],flush=True)
        return {**item,'path':str(path)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download,jobs))
    (ROOT/'sources/downloads.json').write_text(json.dumps(records,indent=2)+'\n')
    print('READER_TRAINING_DATA_READY',flush=True)


if __name__ == '__main__':main()
