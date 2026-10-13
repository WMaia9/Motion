"""Read-only SHA256 integrity check; optional output must be outside frozen roots."""
import argparse,json,time
from pathlib import Path
from verify_signspark_studies import sha

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
 roots=sorted(Path('reports').glob('*/manifest_sha256.json'));result={};started=time.monotonic()
 for manifest in roots:
  root=manifest.parent;record=json.loads(manifest.read_text());size=0
  for name,digest in record.items():
   p=root/name;assert p.is_file(),p;assert sha(p)==digest,('HASH_MISMATCH',p);size+=p.stat().st_size
  result[root.name]={'files':len(record),'bytes':size,'manifest_sha256':sha(manifest),'passed':True}
  print('VERIFIED_FROZEN_REPORT',root.name,len(record),flush=True)
 output={'passed':True,'reports':result,'total_files':sum(d['files'] for d in result.values()),'elapsed_seconds':time.monotonic()-started}
 if args.output:
  for m in roots:assert m.parent.resolve() not in args.output.resolve().parents,'output inside frozen root'
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2)+'\n')
 print('ALL_FROZEN_REPORTS_VERIFIED',output['total_files'],flush=True)
if __name__=='__main__':main()
