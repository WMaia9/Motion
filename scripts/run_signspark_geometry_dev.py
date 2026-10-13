"""Fixed DEV intervention study; no fitting and no TEST access."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0,str(Path('external/SignSparK_BT').resolve()))
from signspark_bt.helpers import load_config
from signspark_bt.data.loading import _build_split
from signspark_bt.inference import back_translate, make_back_translation_model
from signspark_bt.metrics import score_translations

ROOT=Path('reports/cvpr_signspark_baseline_20261010')

def matrices(x):
    d=x.reshape(-1,40,6).astype(np.float64)
    a=d[...,:3]; a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    b=d[...,3:]-(a*d[...,3:]).sum(-1,keepdims=True)*a
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-12)
    return np.stack((a,b,np.cross(a,b)),axis=-2)

def project(x):
    return matrices(x)[...,:2,:].reshape(x.shape).astype(np.float32)

def sparse(x,kind):
    n=len(x); knots=np.unique(np.r_[np.arange(0,n,10),n-1]);ts=np.arange(n)
    if n<2:return project(x) if kind!='linear6d' else x.copy()
    if kind in ('linear6d','projected6d'):
        y=np.stack([np.interp(ts,knots,x[knots,j]) for j in range(240)],axis=-1).astype(np.float32)
        return project(y) if kind=='projected6d' else y
    r=matrices(x[knots]);out=np.empty((n,40,3,3))
    for j in range(40):out[:,j]=Slerp(knots,Rotation.from_matrix(r[:,j]))(ts).as_matrix()
    return out[...,:2,:].reshape(n,240).astype(np.float32)

def transform(x,condition):
    if condition=='original':return x.copy()
    if condition=='so3_roundtrip':return project(x)
    if condition=='fp16_roundtrip':return x.astype(np.float16).astype(np.float32)
    if condition=='reverse_time':return x[::-1].copy()
    if condition=='identity_hands':
        y=x.copy();y[:,60:]=np.tile([1,0,0,0,1,0],30);return y
    return sparse(x,condition.removeprefix('sparse_stride10_'))

def geometry(x,y):
    d=y.reshape(-1,40,6).astype(np.float64)
    a,b=d[...,:3],d[...,3:]
    geod=Rotation.from_matrix((matrices(y).transpose(0,1,3,2)@matrices(x)).reshape(-1,3,3)).magnitude()
    return {'feature_rmse':float(np.sqrt(np.mean((x-y)**2))),
            'mean_joint_angle_deg':float(np.degrees(geod).mean()),
            'unit_norm_abs_error':float(np.mean(np.abs(np.linalg.norm(d.reshape(-1,3),axis=-1)-1))),
            'pair_abs_dot':float(np.mean(abs((a*b).sum(-1))))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    os.environ['DATA_ROOT']=str((ROOT/'data').resolve())
    dest=ROOT/args.dataset;dest.mkdir(exist_ok=True)
    cfg=load_config(ROOT/'checkpoints'/args.dataset/'config.yaml')
    data_cfg=dict(cfg['data'])
    # Pseudo-glosses are training targets for CTC, never translation inputs.
    # Native inference explicitly disables recognition; preserve released weights.
    data_cfg.pop('pseudo_gloss_dir',None)
    dataset=_build_split(data_cfg,'dev',None)
    xs=[dataset[i].sgn.numpy() for i in range(len(dataset))]
    assert all(x.shape[1]==240 and len(x)>0 and np.isfinite(x).all() for x in xs)
    refs=[' '.join(e.txt) for e in dataset.examples]
    (dest/'references.json').write_text(json.dumps([{'clip_id':e.sequence,'frames':len(x),'reference':r} for e,x,r in zip(dataset.examples,xs,refs)],indent=2,ensure_ascii=False))
    model=make_back_translation_model(ROOT/'checkpoints'/args.dataset)
    audit={'clips':len(xs),'frames_after_native_cap':sum(map(len,xs)),
           'max_frames':max(map(len,xs)),'native_cap':cfg['data'].get('max_sgn_length',300),
           'beam_size':model.beam_size,'beam_alpha':model.beam_alpha,'max_output_length':model.max_output_len,
           'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,
           'conditions':json.loads((ROOT/'plan.json').read_text())['conditions']}
    (dest/'audit.json').write_text(json.dumps(audit,indent=2))
    results={}
    # Native API padding restored in original order; length-sort reduces wasted GPU work.
    order=np.argsort([len(x) for x in xs],kind='stable')
    for condition in audit['conditions']:
        started=time.time();ys=[transform(x,condition) for x in xs]
        assert all(np.isfinite(y).all() for y in ys)
        hypotheses=['']*len(xs)
        for start in range(0,len(order),32):
            ix=order[start:start+32]
            decoded=back_translate(model,[torch.from_numpy(ys[i]) for i in ix],batch_size=32)
            for i,h in zip(ix,decoded):hypotheses[i]=h
        stats=[geometry(x,y) for x,y in zip(xs,ys)]
        results[condition]={'scores':score_translations(refs,hypotheses,cfg['data'].get('version',''),cfg['data'].get('level','word')),
                            'geometry_clip_mean':{k:float(np.mean([s[k] for s in stats])) for k in stats[0]},
                            'geometry_per_clip':stats,'hypotheses':hypotheses,'seconds':time.time()-started}
        (dest/f'{condition}.json').write_text(json.dumps(results[condition],indent=2,ensure_ascii=False))
        np.savez_compressed(dest/f'{condition}_examples.npz',**{f'clip{i:02d}':ys[i] for i in range(min(6,len(ys)))})
        summary={k:{'scores':v['scores'],'geometry_clip_mean':v['geometry_clip_mean'],'seconds':v['seconds']} for k,v in results.items()}
        (dest/'summary.json').write_text(json.dumps(summary,indent=2))
        print(args.dataset,condition,len(xs),json.dumps(results[condition]['scores']),round(time.time()-started,2),flush=True)

if __name__=='__main__':main()
