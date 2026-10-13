"""Seal all reward-pilot weights before paired TEST generation/evaluation."""
import numpy as np
import torch
from motion_release_audit_common import read,write,sha,seed,evaluator,score_embeddings,canonical,metric
from run_motion_release_audit import mdm_model,generate_mdm,geometry
from run_humanml_reward_pilot import R,register

def main():
    p=register();weights={}
    for s in p['training_seeds']:
        for c in p['conditions']:
            path=R/f'seed_{s}'/c/'final.pt';done=read(path.parent/'complete.json');assert sha(path)==done['weights_sha256'];weights[str(path)]=sha(path)
    seal=R/'weights_sealed_before_TEST.json'
    if not seal.exists():write(seal,{'weights_sha256':weights,'all9weights_complete':True,'pilot_TEST_generation_not_started':True})
    else:assert read(seal)['weights_sha256']==weights
    source='reports/cvpr_motion_prevalence_20261012/mdm50/rep_00/';z=np.load(source+'conditioning.npz');cohort={k:z[k].tolist() if k in ['captions','tokens','clip_ids'] else z[k].copy() for k in z.files};cohort['raw']=np.zeros((1024,196,263),np.float32);reference=np.load(source+'reference_embeddings.npy');wrapper,mean,std=evaluator();rows=[]
    for s in p['training_seeds']:
        for c in p['conditions']:
            folder=R/f'seed_{s}'/c
            if (folder/'metrics.json').exists():rows.append(read(folder/'metrics.json'));continue
            model,guided,diffusion=mdm_model();state=torch.load(folder/'final.pt',map_location='cpu',weights_only=True);missing,unexpected=model.load_state_dict(state,strict=False);assert not unexpected and all(k.startswith('clip_model.') for k in missing);model.eval();seed(20261031);raw=generate_mdm((model,guided,diffusion),cohort,s);np.save(folder/'test_generated_raw.npy',raw);results={};checks={}
            for name,x in [('raw',raw),('canonical',canonical(raw))]:
                T,E=score_embeddings(wrapper,x,cohort,mean,std);results[name]=metric(T,E,reference);checks[name]=geometry(raw,x,cohort['lengths']);np.save(folder/f'{name}_embeddings.npy',E)
            row={'training_seed':s,'condition':c,'results':results,'geometry':checks,'same_generation_seed':20261031,'TEST_clips':1024,'full_TEST_reference_clips':len(reference)};write(folder/'metrics.json',row);rows.append(row);print('PILOT_SCORED',row,flush=True);del model,guided,state,raw;torch.cuda.empty_cache()
    write(R/'analysis.json',{'conditions':rows,'training_seeds':3,'generation_seeds_per_fit':1,'pilot_not_standard20repetitions':True,'all9weights_sealed_before_TEST':True});print('REWARD_PILOT_COMPLETE',flush=True)
if __name__=='__main__':main()
