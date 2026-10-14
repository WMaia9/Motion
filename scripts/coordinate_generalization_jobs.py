"""Serialize independent GPU studies; CPU downloads may proceed concurrently."""
import json,subprocess,time
from pathlib import Path
QUEUE=[('tmr_existing',None,'reports/cvpr_tmr_audit_20261013/complete.json'),('calibration','run_physical_rotation_calibration.py','reports/cvpr_physical_calibration_20261013/complete.json'),('reward_dev','run_reward_dev_search.py run','reports/cvpr_reward_dev_search_20261013/selection.json')]
R=Path('reports/cvpr_generalization_review_20261013')
def main():
 rows=[]
 for name,script,completion in QUEUE:
  start=time.time();row={'job':name,'start_unix':start,'done':False};rows.append(row);(R/'gpu_queue.json').write_text(json.dumps(rows,indent=2)+'\n')
  if script is None:
   while not Path(completion).exists():time.sleep(10)
  else:
   path=Path('/tmp')/f'cvpr_generalization_{name}_20261013.log'
   with path.open('w') as log:
    result=subprocess.run(['venv_mska/bin/python','scripts/'+script.split()[0],*script.split()[1:]],stdout=log,stderr=subprocess.STDOUT)
   row['exit_code']=result.returncode
   if result.returncode!=0:raise RuntimeError((name,'failed; queue stops; see log',str(path)))
   assert Path(completion).exists()
  row.update(done=True,end_unix=time.time());(R/'gpu_queue.json').write_text(json.dumps(rows,indent=2)+'\n');print('GENERALIZATION_JOB_COMPLETE',name,flush=True)
 print('GENERALIZATION_GPU_QUEUE_COMPLETE',flush=True)
if __name__=='__main__':main()
