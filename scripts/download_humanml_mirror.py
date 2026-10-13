"""Public mirror fallback; records that author-linked Drive hit its quota."""
import hashlib,json,time
from pathlib import Path
import requests
ROOT=Path('reports/cvpr_humanml3d_audit_20261012')
REV='2bf72a85f0a174123aa85acf19f89661677719b3'
URL=f'https://huggingface.co/datasets/NamYeongCho/HumanML3D/resolve/{REV}/new_joint_vecs-003.rar'
EXPECTED='f38ac48c61adb8e65b3d13e1bed68118698d64acc55af5ae62b321821913f65b'
def main():
 assert not (ROOT/'manifest_sha256.json').exists()
 p=Path('models/humanml3d_audit/new_joint_vecs.rar');partial=p.with_suffix('.download')
 r=requests.get(URL,stream=True,timeout=(30,90));r.raise_for_status();h=hashlib.sha256();n=0;t=time.monotonic();last=t
 with partial.open('wb') as f:
  for chunk in r.iter_content(8*1024*1024):
   if chunk:f.write(chunk);h.update(chunk);n+=len(chunk)
   if time.monotonic()-last>20:print('DATA_BYTES',n,flush=True);last=time.monotonic()
 assert n==4297557992 and h.hexdigest()==EXPECTED,(n,h.hexdigest())
 partial.rename(p)
 (ROOT/'mirror_download.json').write_text(json.dumps({'url':URL,'revision':REV,'path':str(p),'bytes':n,'sha256':EXPECTED,'HF_LFS_hash_verified':True,'source':'public third-party mirror, not an author release','author_linked_Drive_expected_size_matches':True,'original_Drive_bytes_unavailable_for_full_hash_comparison':True,'must_verify_author_example_and_statistics_after_extraction':True,'elapsed_seconds':time.monotonic()-t,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
 print('DATA_DOWNLOADED',n,h.hexdigest(),flush=True)
if __name__=='__main__':main()
