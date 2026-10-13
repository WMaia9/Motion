"""Secondary audit on the distinct author-linked T2M-GPT evaluator weights.

No generated samples, source models, or settings are selected by these scores.
The primary common-MDM-reader audit is retained in full. This is a second
checkpoint of the same evaluator architecture, not a second architecture.
"""
import time,os
from pathlib import Path
import numpy as np
import torch
from motion_release_audit_common import R,A,read,write,sha,seed,native_dataset,materialize,evaluator,score_embeddings,canonical,metric

def main():
    root=R/'secondary_gpt_reader';root.mkdir(exist_ok=True);checkpoint=Path('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar')
    files=[Path(__file__),Path('scripts/motion_release_audit_common.py'),checkpoint];record={'scope':'secondary distinct official GPT evaluator checkpoint, same architecture; all five methods20reps, no model/condition selection','sources_sha256':{str(p):sha(p) for p in files},'primary_reader_unchanged':True,'native_normalizations_numerically_identical_verified':True}
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==record
    else:write(root/'protocol.json',record)
    seed(20261012);dataset=native_dataset();wrapper,mean,std=evaluator();state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for key,model in [('movement_encoder',wrapper.movement_encoder),('motion_encoder',wrapper.motion_encoder),('text_encoder',wrapper.text_encoder)]:model.load_state_dict(state[key],strict=True);model.eval()
    completed=set();methods=['mdm50','mld','motionlcm_v1','t2mgpt','t2m'];deadline=time.monotonic()+21600
    while len(completed)<100:
        progressed=False
        for rep,s in enumerate(read(R/'protocol.json')['replication_seeds']):
            pending=[method for method in methods if (method,rep) not in completed and (R/method/f'rep_{rep:02d}/metrics.json').exists()]
            if not pending:continue
            refpath=root/f'reference_rep_{rep:02d}.npy'
            if refpath.exists():ref=np.load(refpath)
            else:
                reference=materialize(dataset,s+100000);_,ref=score_embeddings(wrapper,reference['raw'],reference,mean,std);np.save(refpath,ref);del reference
            for method in pending:
                source=R/method/f'rep_{rep:02d}';folder=root/method/f'rep_{rep:02d}';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'metrics.json').exists():completed.add((method,rep));continue
                z=np.load(source/'conditioning.npz');cohort={k:z[k].copy() for k in z.files};cohort['lengths']=np.load(source/'generated_lengths.npy');raw=np.load(source/'generated_raw.npy',mmap_mode='r');scores={}
                for condition,x in [('raw',raw),('canonical',canonical(raw))]:
                    T,E=score_embeddings(wrapper,x,cohort,mean,std);scores[condition]=metric(T,E,ref);np.save(folder/f'{condition}_embeddings.npy',E)
                write(folder/'metrics.json',{'method':method,'rep':rep,'second_evaluator_epoch':int(state['epoch']),'scores':scores,'generator_raw_array_sha256':sha(source/'generated_raw.npy')});completed.add((method,rep));progressed=True;print('SECOND_READER_SCORED',method,rep,scores,flush=True)
                del raw,cohort
        if not progressed:
            if time.monotonic()>deadline:raise RuntimeError('source audits incomplete; secondary audit retained partially')
            time.sleep(20)
    write(root/'complete.json',{'methods':methods,'replications_each':20,'distinct_checkpoint_same_architecture':True});print('SECOND_READER_COMPLETE',flush=True)
if __name__=='__main__':main()
