"""Serialize existing GPU workers to avoid costly CUDA context time slicing.

No inputs, model settings, seeds or scores change. This only schedules this
study's processes; unrelated GPU users are untouched. Resume all on failure.
"""
import argparse,os,signal,time,json
from pathlib import Path

def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('jobs',help='JSON list of [method,pid] pairs');args=parser.parse_args();jobs=json.loads(args.jobs);root=Path('reports/cvpr_motion_prevalence_20261012');record={'scope':'GPU scheduling only; scientific conditions unchanged','jobs':jobs,'completed':[]};deadline=time.monotonic()+14400
    try:
        for _,pid in jobs:
            if alive(pid):os.kill(pid,signal.SIGSTOP)
        for name,pid in jobs:
            if alive(pid):os.kill(pid,signal.SIGCONT)
            record['running']=name;(root/'gpu_queue.json').write_text(json.dumps(record,indent=2)+'\n');print('GPU_JOB_RUNNING',name,pid,flush=True)
            while alive(pid):
                if time.monotonic()>deadline:raise RuntimeError('GPU queue timeout; resuming all workers')
                time.sleep(5)
            record['completed'].append(name);print('GPU_JOB_FINISHED',name,pid,flush=True)
        record['all_workers_exited']=True;(root/'gpu_queue.json').write_text(json.dumps(record,indent=2)+'\n')
    finally:
        for _,pid in jobs:
            if alive(pid):os.kill(pid,signal.SIGCONT)
if __name__=='__main__':main()
