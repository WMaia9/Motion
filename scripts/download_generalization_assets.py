"""Pinned author-linked acquisition for the finite generalization study."""
import argparse, hashlib, json, tarfile, zipfile
from pathlib import Path
import requests
import gdown
from huggingface_hub import HfApi, snapshot_download
from motion_release_audit_common import sha,write
R=Path('reports/cvpr_generalization_review_20261013');M=Path('models/generalization_audit')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('asset',choices=['distilbert','momask','momask_kit']);a=parser.parse_args();M.mkdir(parents=True,exist_ok=True)
 if a.asset=='distilbert':
  repo='distilbert/distilbert-base-uncased';revision=HfApi().model_info(repo).sha
  path=snapshot_download(repo_id=repo,revision=revision,local_dir=M/'distilbert-base-uncased',allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json'])
  record={'repo':repo,'revision':revision,'path':path,'files_sha256':{str(p):sha(p) for p in Path(path).rglob('*') if p.is_file() and '.cache' not in p.parts}}
 else:
  id='1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0' if a.asset=='momask' else '1FapdHNkxPouasVM8MWgg1f6sd_4Lua2q';p=M/(a.asset+'.zip')
  if not p.exists():gdown.download(id=id,output=str(p),quiet=False,resume=True)
  assert p.exists()
  target=M/a.asset;target.mkdir(exist_ok=True)
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None
   for f in z.infolist():assert (target/f.filename).resolve().is_relative_to(target.resolve())
   z.extractall(target)
  record={'author_script':'external/MoMask/prepare/download_models.sh','Drive_id':id,'path':str(p),'sha256':sha(p)}
 write(R/f'download_{a.asset}.json',record);print('ASSET_COMPLETE',a.asset,flush=True)
if __name__=='__main__':main()
