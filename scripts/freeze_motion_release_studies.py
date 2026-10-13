"""Freeze complete new studies without changing any previous evidence archive."""
import argparse,ast,hashlib,json,shutil
from pathlib import Path
ROOTS={
 'prevalence':('cvpr_motion_prevalence_20261012',['prepare_motion_release_audit','download_motion_release_audit','download_motion_release_additional','run_motion_release_audit','run_motion_release_audit_fast','run_motion_release_t2m_grouped','run_motion_release_mld_batched','run_motion_release_gpt_batched','verify_motion_release_adapters','coordinate_motion_gpu_jobs','score_motion_release_second_reader','analyze_motion_prevalence','verify_motion_prevalence','complete_motion_release_verification']),
 'blind':('cvpr_humanml_blind_repetitions_20261012',['prepare_humanml_blind_repetitions','run_humanml_blind_repetitions','run_humanml_blind_fid','score_humanml_blind_fid']),
 'pilot':('cvpr_humanml_reward_pilot_20261012',['run_humanml_reward_pilot','score_humanml_reward_pilot','verify_humanml_reward_pilot']),
 'reward':('cvpr_motion_reward_release_20261013',['run_motion_reward_release','complete_motion_release_verification'])}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def closure(root,names):
 mapping={p.stem:p for p in Path('scripts').glob('*.py')};pending=list(names);keep=set(names)
 while pending:
  for n in ast.walk(ast.parse(mapping[pending.pop()].read_text())):
   mods=[v.name for v in n.names] if isinstance(n,ast.Import) else [n.module or ''] if isinstance(n,ast.ImportFrom) else []
   for mod in mods:
    m=mod.split('.')[0]
    if m in mapping and m not in keep:keep.add(m);pending.append(m)
 target=root/'scripts';target.mkdir(exist_ok=True)
 for n in sorted(keep):shutil.copy2(mapping[n],target/mapping[n].name)
 return sorted(keep)
def check_protocol(p):
 x=read(p);verified={}
 for key in ['inputs_sha256','scientific_sources_sha256','inputs_and_sources_sha256','producer_sha256']:
  for path,value in x.get(key,{}).items():
   assert sha(path)==value,(p,path);verified[path]=value
 return verified

def main():
 parser=argparse.ArgumentParser();parser.add_argument('study',choices=ROOTS);args=parser.parse_args();name,producers=ROOTS[args.study];root=Path('reports')/name;assert not (root/'manifest_sha256.json').exists();checks=check_protocol(root/'protocol.json')
 if args.study=='blind':
  assert read(root/'complete.json')['all_stages_complete'];v={}
  for c in ['cohort_1','cohort_2']:
   checks.update(check_protocol(root/c/'protocol.json'));v[c]=read(root/c/'verification.json');assert v[c]['passed']
  write(root/'verification.json',{'passed':True,'new_cohorts':v,'original_frozen_cohort':'reports/cvpr_humanml3d_blind_fid_20261012','not_independent_clip_samples':True,'all_registered_inputs_and_sources_match':True})
 else:assert read(root/'verification.json')['passed']
 if args.study=='prevalence':
  assert read(root/'additional_verification.json')['passed'];assert read(root/'secondary_gpt_reader/complete.json')['replications_each']==20
  assert read(root/'gpu_queue.json')['all_workers_exited']
  assets={}
  for p in Path('models/motion_release_audit').rglob('*'):
   if p.is_file() and '.cache' not in p.parts and p.suffix in ['.bin','.ckpt','.pth','.tar','.json','.npy','.txt','.model'] and 'extracted' not in p.parts:assets[str(p)]=sha(p)
  write(root/'asset_inventory.json',{'current_public_assets_sha256':assets,'inventory_after_completed_runs':True,'prospective_inputs_separately_in_protocol':True})
 names=closure(root,producers+['freeze_motion_release_studies']);logs=root/'logs';logs.mkdir(exist_ok=True)
 patterns={'prevalence':['cvpr_motion_*20*.log','cvpr_motion_gpu_queue*.log','cvpr_motion_prevalence*.log','parity*.log'],'blind':['cvpr_blind_repetitions*.log'],'pilot':['cvpr_reward_pilot*.log'],'reward':['cvpr_motion_reward_release*.log']}[args.study]
 for pattern in patterns:
  for p in Path('/tmp').glob(pattern):shutil.copy2(p,logs/p.name)
 write(root/'source_provenance.json',{'producers':producers,'import_closure':names,'all_archived_scripts_match_live':all(sha(root/'scripts'/f'{n}.py')==sha(Path('scripts')/f'{n}.py') for n in names),'verified_registered_inputs_and_sources':checks,'historical_executed_sources_if_any':'producer_revisions/; preserved separately from current closure','freeze_code_sha256':sha(__file__),'completed_date':'2026-10-13'})
 inventory={}
 for p in sorted(root.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest_sha256.json':inventory[str(p.relative_to(root))]=sha(p)
 write(root/'manifest_sha256.json',inventory);print('FROZEN',name,len(inventory),sha(root/'manifest_sha256.json'),flush=True)
if __name__=='__main__':main()
