"""Execute independent native methods in separate Python namespaces."""
from pathlib import Path
import subprocess,time,json,argparse
R=Path('reports/cvpr_additional_motion_audit_20261013')
QUEUES={
 'humanml':[('momask','humanml'),('dip','humanml')],
 'kit_and_ddpm':[('real_controls','kit'),('momask','kit'),('motiondiffuse','humanml'),('motiondiffuse','kit')],
}
def main():
 p=argparse.ArgumentParser();p.add_argument('queue',choices=list(QUEUES));args=p.parse_args();rows=[]
 for method,dataset in QUEUES[args.queue]:
  log=Path('/tmp')/f'cvpr_additional_{method}_{dataset}_20261013.log';row={'method':method,'dataset':dataset,'start_unix':time.time(),'complete':False};rows.append(row)
  (R/(args.queue+'_queue.json')).write_text(json.dumps(rows,indent=2)+'\n')
  with log.open('w') as handle:
   code=subprocess.run(['venv_mska/bin/python','scripts/run_additional_motion_audit.py',method,dataset],stdout=handle,stderr=subprocess.STDOUT).returncode
  row.update(exit_code=code,end_unix=time.time(),complete=code==0);(R/(args.queue+'_queue.json')).write_text(json.dumps(rows,indent=2)+'\n')
  if code:raise RuntimeError((method,dataset,'stopped; inspect',str(log)))
  print('ADDITIONAL_QUEUE_FINISHED',method,dataset,flush=True)
if __name__=='__main__':main()
