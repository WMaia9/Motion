"""Independent exhaustive CTC and double-sum local-signature checks."""
import itertools
import json
import numpy as np
import torch
from run_local_path_diffusion import ROOT,viterbi,local_signatures,bounded_partition,uniform
from check_motion_path_math import numpy_signature
from run_physical_semantic_ddpm import physical


def collapse(path):
    return [token for i,token in enumerate(path) if token!=0 and (i==0 or token!=path[i-1])]


def check():
    rng=np.random.default_rng(2027);error=0.;cases=0
    for steps in range(1,7):
        logits=rng.normal(size=(steps,3));logp=logits-np.log(np.exp(logits).sum(1))[:,None]
        paths=list(itertools.product(range(3),repeat=steps))
        for target in [[],[1],[2],[1,2],[1,1],[1,2,1]]:
            legal=[p for p in paths if collapse(p)==target]
            state,score=viterbi(logp,target)
            if not legal:assert state is None and score is None
            else:
                optimum=max(sum(logp[i,token] for i,token in enumerate(p)) for p in legal)
                labels=np.zeros(2*len(target)+1,int);labels[1::2]=target
                decoded=labels[state]
                assert collapse(decoded.tolist())==target
                error=max(error,abs(score-optimum));assert abs(score-optimum)<1e-10
            cases+=1
    generator=torch.Generator().manual_seed(2027)
    raw=torch.randn(3,13,240,generator=generator,dtype=torch.float64)
    bounds=[[0,4,9,13],[0,2,6,10],[0,5,11]]
    first,area,owners=local_signatures(physical(raw),bounds);canonical=physical(raw).numpy().reshape(3,13,40,6)
    maximum=0.;cursor=0
    for batch,partition in enumerate(bounds):
        for start,end in zip(partition,partition[1:]):
            nf,na=numpy_signature(canonical[batch,start:end])
            maximum=max(maximum,float(abs(nf-first[cursor].numpy()).max()),float(abs(na-area[cursor].numpy()).max()))
            assert int(owners[cursor])==batch;cursor+=1
    assert maximum<1e-10
    d=raw.reshape(3,13,40,6);p=torch.randn(3,13,40,3,generator=generator,dtype=torch.float64)
    changed=torch.cat([p[...,0:1].mul(.2).exp()*d[...,:3],p[...,1:2].mul(.2).exp()*d[...,3:]+p[...,2:3]*.3*d[...,:3]],-1).reshape(raw.shape)
    changed_first,changed_area,_=local_signatures(physical(changed),bounds)
    gauge=max(float((first-changed_first).abs().max()),float((area-changed_area).abs().max()));assert gauge<1e-10
    direction=torch.cat([p[...,0:1]*d[...,:3],p[...,1:2]*d[...,3:]+p[...,2:3]*d[...,:3]],-1).reshape(raw.shape)
    def features(value):
        f,a,_=local_signatures(physical(value),bounds)
        return torch.cat([f.flatten(),a.flatten()])
    _,jvp=torch.func.jvp(features,(raw,),(direction,));vertical=float(jvp.abs().max());assert vertical<1e-10
    partition_cases=0
    for length in range(2,301):
        for k in range(1,length//2+1):
            for candidates in [[-100]*(k-1),[1000]*(k-1),list(range(k-1))]:
                partition=bounded_partition(candidates,length,k)
                assert partition[0]==0 and partition[-1]==length and min(np.diff(partition))>=2
                partition_cases+=1
            assert min(np.diff(uniform(length,k)))>=2
    return {'exhaustive_CTC_cases':cases,'viterbi_max_score_error':error,
        'numpy_double_sum_local_signature_max_error':maximum,'gauge_invariance_max_error':gauge,
        'vertical_jvp_max_error':vertical,'partition_clamping_cases':partition_cases,
        'known_CTC_and_Chen_math_not_new_algorithms':True}


if __name__=='__main__':
    result=check();(ROOT/'sources/math_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('LOCAL_MATH_PASS',json.dumps(result),flush=True)
