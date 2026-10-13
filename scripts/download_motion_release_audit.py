"""Download only author-linked release checkpoints; never execute cleanup scripts."""
from pathlib import Path
import concurrent.futures,json,hashlib
import gdown
R=Path('reports/cvpr_motion_prevalence_20261012');A=Path('models/motion_release_audit')
FILES={'momask.zip':'1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0','t2mgpt.zip':'1LaOvwypF-jM2Axnq5dc-Iuvv3w_G-WDE','mld.ckpt':'1hplrnQwUK_cZFHirZIOuVP0RSyZEC1YM','lcm_t2m.zip':'1U7homKobR2gaDLfL5flS3N0g7e0a_AQd','lcm_vae.zip':'15zFDitcOLhjbQ0CaOoM-QNKQUeyJw-Om'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fetch(item):
 name,id=item;p=A/name;record={'name':name,'path':str(p),'url':'https://drive.google.com/uc?id='+id,'source':'author repository prepare script'}
 try:
  if not p.exists():assert gdown.download(id=id,output=str(p),quiet=True,resume=True)
  record.update(downloaded=True,bytes=p.stat().st_size,sha256=sha(p));print('DOWNLOADED',name,p.stat().st_size,flush=True)
 except Exception as e:record.update(downloaded=False,error=str(e));print('UNAVAILABLE',name,str(e)[:180],flush=True)
 return record
def main():
 R.mkdir(exist_ok=True);A.mkdir(exist_ok=True);assert not (R/'manifest_sha256.json').exists()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:records=list(p.map(fetch,FILES.items()))
 (R/'downloads.json').write_text(json.dumps({'files':records,'source_code_sha256':sha(__file__)},indent=2)+'\n')
if __name__=='__main__':main()
