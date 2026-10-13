"""Run an exclusive prespecified job partition with the unchanged fitting loop."""
import argparse,json
import torch
import run_seed_dose_response as study
from verify_signspark_studies import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('partition',type=int,choices=[0,1,2]);a=ap.parse_args()
 plan=study.read(study.ROOT/'execution_partition.json');protocol=study.read(study.ROOT/'protocol.json')
 for key in ['scientific_sources_sha256','inputs_sha256']:
  for p,h in protocol[key].items():assert sha(p)==h,p
 audit,mean,std=study.setup();train,dev=study.old.dataset('train'),study.old.dataset('dev');assert [e.sequence for e in train]==audit['train_ids'] and [e.sequence for e in dev]==audit['dev_ids']
 stoi={w:i for i,w in enumerate(audit['vocab'])};shared=torch.load(study.old.ROOT/'shared_best.pt',map_location='cpu',weights_only=False)['state_dict'];alpha=study.old.alpha_schedule()
 teacher=study.old.make_back_translation_model(study.old.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition=False
 for p in teacher.parameters():p.requires_grad_(False)
 for j in plan['partitions'][a.partition]:
  s,k,w,e=j['seed'],j['reader_input'],j['CE_weight'],j['epochs'];folder=study.folder(s,k,w)
  assert not folder.exists(),'worker partition collision; refuse overwrite'
  model=study.old.Denoiser(len(stoi)).cuda();model.load_state_dict(shared,strict=True)
  study.fit(model,train,dev,stoi,mean,std,alpha,teacher,'geodesic',k,w,s,e);del model;torch.cuda.empty_cache()
 study.write(f'worker{a.partition}_complete.json',{'jobs':plan['partitions'][a.partition],'worker_code_sha256':sha(__file__),'unmodified_fitting_source_sha256':sha('scripts/run_seed_dose_response.py')})
if __name__=='__main__':main()
