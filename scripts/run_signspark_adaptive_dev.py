"""Exploratory classical greedy SO3 anchor allocation, exactly matched budgets."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from run_signspark_geometry_dev import ROOT,matrices,geometry,load_config,_build_split,make_back_translation_model,back_translate,score_translations

def bridge(r0,r1,ts):
    omega=Rotation.from_matrix(r0.transpose(0,2,1)@r1).as_rotvec()
    delta=Rotation.from_rotvec((ts[:,None,None]*omega[None]).reshape(-1,3)).as_matrix().reshape(len(ts),40,3,3)
    return r0[None]@delta

def adaptive(x,stream):
    n=len(x);r=matrices(x);k=len(np.unique(np.r_[np.arange(0,n,10),n-1]))
    selection=slice(None) if stream=='all' else slice(0,10) if stream=='body' else slice(10,40)
    def interval(lo,hi):
        if hi-lo<2:return None
        ts=np.arange(1,hi-lo)/(hi-lo)
        interpolated=bridge(r[lo],r[hi],ts)
        cosine=((interpolated*r[lo+1:hi]).sum((-1,-2))-1)/2
        errors=np.arccos(np.clip(cosine,-1,1))[:,selection].mean(-1)
        q=int(np.argmax(errors));return float(errors[q]),lo+1+q
    anchors={0,n-1};intervals={(0,n-1):interval(0,n-1)}
    while len(anchors)<k:
        valid=[(v[0],-v[1],pair) for pair,v in intervals.items() if v is not None]
        if not valid:break
        _,_,(lo,hi)=max(valid);_,mid=intervals.pop((lo,hi))
        anchors.add(mid);intervals[(lo,mid)]=interval(lo,mid);intervals[(mid,hi)]=interval(mid,hi)
    anchors=sorted(anchors);assert len(anchors)==k
    y=np.empty_like(r)
    for lo,hi in zip(anchors[:-1],anchors[1:]):y[lo:hi+1]=bridge(r[lo],r[hi],np.arange(hi-lo+1)/(hi-lo))
    if n==1:y=r.copy()
    return y[...,:2,:].reshape(n,240).astype(np.float32),anchors

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);args=parser.parse_args()
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    os.environ['DATA_ROOT']=str((ROOT/'data').resolve());dest=ROOT/args.dataset
    cfg=load_config(ROOT/'checkpoints'/args.dataset/'config.yaml');data_cfg=dict(cfg['data']);data_cfg.pop('pseudo_gloss_dir',None)
    dataset=_build_split(data_cfg,'dev',None);xs=[dataset[i].sgn.numpy() for i in range(len(dataset))]
    refs=[' '.join(e.txt) for e in dataset.examples];assert refs==[r['reference'] for r in json.loads((dest/'references.json').read_text())]
    model=make_back_translation_model(ROOT/'checkpoints'/args.dataset)
    order=np.argsort([len(x) for x in xs],kind='stable');summary=json.loads((dest/'summary.json').read_text())
    for stream in ['all','body','hands']:
        condition=f'adaptive_{stream}_slerp';started=time.time()
        computed=[adaptive(x,stream) for x in xs];ys=[v[0] for v in computed]
        assert all(np.isfinite(y).all() for y in ys)
        hypotheses=['']*len(xs)
        for start in range(0,len(order),32):
            ix=order[start:start+32];pred=back_translate(model,[torch.from_numpy(ys[i]) for i in ix],batch_size=32)
            for i,h in zip(ix,pred):hypotheses[i]=h
        stats=[geometry(x,y) for x,y in zip(xs,ys)]
        result={'scores':score_translations(refs,hypotheses,cfg['data'].get('version',''),cfg['data'].get('level','word')),
                'geometry_clip_mean':{k:float(np.mean([s[k] for s in stats])) for k in stats[0]},
                'geometry_per_clip':stats,'hypotheses':hypotheses,'anchors':[v[1] for v in computed],'seconds':time.time()-started}
        (dest/f'{condition}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        np.savez_compressed(dest/f'{condition}_examples.npz',**{f'clip{i:02d}':ys[i] for i in range(min(6,len(ys)))})
        summary[condition]={k:result[k] for k in ('scores','geometry_clip_mean','seconds')}
        (dest/'summary.json').write_text(json.dumps(summary,indent=2))
        print(args.dataset,condition,len(xs),json.dumps(result['scores']),round(time.time()-started,2),flush=True)

if __name__=='__main__':main()
