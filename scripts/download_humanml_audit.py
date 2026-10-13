"""Fetch author-linked public HumanML3D audit assets without changing environments."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import gdown

ROOT = Path('reports/cvpr_humanml3d_audit_20261012')
ASSETS = Path('models/humanml3d_audit')
FILES = {
    'new_joint_vecs.rar': ('14HIdUa_LPGEh-M1ICDZJp0tLJftewnAX', 'MDM author-linked processed HumanML3D folder'),
    't2m.zip': ('1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb', 'MDM prepare/download_t2m_evaluators.sh'),
    'glove.zip': ('1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n', 'MDM prepare/download_glove.sh'),
    'mdm50.zip': ('1cfadR1eZ116TIdXK7qDX1RugAerEiJXr', 'MDM README released 50-step checkpoint'),
}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def fetch(item):
    name,(identifier,provenance)=item;path=ASSETS/name
    result={'path':str(path),'url':'https://drive.google.com/uc?id='+identifier,'provenance':provenance}
    try:
        if not path.exists():
            value=gdown.download(id=identifier,output=str(path),quiet=True,resume=True)
            if not value:raise RuntimeError('download did not return a file')
        result.update(bytes=path.stat().st_size,sha256=digest(path),downloaded=True)
        print('DOWNLOADED',name,result['bytes'],flush=True)
    except Exception as e:
        result.update(downloaded=False,error=str(e));print('UNAVAILABLE',name,str(e)[:350],flush=True)
    return result

def main():
    ROOT.mkdir(exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'manifest_sha256.json').exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,FILES.items()))
    (ROOT/'downloads.json').write_text(json.dumps({'code_sha256':digest(Path(__file__)),'files':results},indent=2)+'\n')
if __name__=='__main__':main()
